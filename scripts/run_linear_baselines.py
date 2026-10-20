"""Validate LS and select training-covariance shrinkage before a fresh SNR screen."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.data import tensor_hash
from dma_ce.experiment import file_hash, metric_rows, nmse, summarize, write_csv, write_json
from dma_ce.snr_study import sweep_cache
from dma_ce.structured import apply_lmmse, lmmse_statistics
from dma_ce.suite import Study


LABELS = {"ls_min_norm": "Minimum-norm LS", "lmmse_joint": "Original joint sample LMMSE",
    "lmmse_per_tone": "Per-tone sample LMMSE", "lmmse_shrunk": "Validation-selected covariance LMMSE"}


def plot(summary, export):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    hardware = sorted({(r["bandwidth_hz"], r["configurations"]) for r in summary})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (band, count) in zip(axes.flat, hardware):
        for method, label in LABELS.items():
            values = sorted((r for r in summary if r["bandwidth_hz"] == band
                and r["configurations"] == count and r["method"] == method), key=lambda r: r["snr_db"])
            ax.plot([r["snr_db"] for r in values], [r["mean_nmse_db"] for r in values], "o-", label=label)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_title(f"{band/1e9:g} GHz, {count} DMA configurations")
        ax.set_xlabel("Training-average received SNR (dB)")
        ax.set_ylabel("Mean linear NMSE, converted to dB")
        ax.grid(alpha=.25)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.suptitle("Linear baselines on fresh geometries; covariance choice frozen on validation")
    fig.tight_layout(rect=(0, .1, 1, .95))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(export / f"linear_baselines.{suffix}", dpi=180)
    plt.close(fig)


@torch.no_grad()
def run(config_path, parent_id, run_id, device):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", parent_id) is None:
        raise ValueError("safe parent run ID required")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("training_scenes", "validation_scenes", "screen_scenes", "timeout_seconds"):
        if type(settings[key]) is not int or settings[key] < 2:
            raise ValueError("positive budgets and at least two independent geometries required")
    grid = settings["frequency_shrinkage_grid"]
    if (not grid or len(set(grid)) != len(grid) or any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not np.isfinite(x) or not 0 <= x <= 1 for x in grid)
            or type(settings["screen_seed"]) is not int or settings["screen_seed"] < 0):
        raise ValueError("distinct finite shrinkage strengths in [0, 1] and a nonnegative seed required")
    if (ROOT / "runs" / run_id).exists():
        raise ValueError("use a new run ID for each baseline comparison")
    original = ROOT / "runs" / parent_id
    config = copy.deepcopy(json.loads((original / "config_resolved.json").read_text()))
    frozen_parent = json.loads((original / "stages/frozen.json").read_text())
    if json.loads((original / "status.json").read_text())["status"] != "complete":
        raise ValueError("the source SNR run must be complete")
    config["label"] = "linear_baseline_implementation_check" if settings.get("implementation_only") else "linear_baseline_exploration"
    config["data"].update(screen=settings["screen_scenes"], screen_seed=settings["screen_seed"])
    config["timeout_seconds"] = settings["timeout_seconds"]
    config["linear_baselines"] = {**settings, "source_snr_run": parent_id,
        "timing": "not measured; shared metric timing fields are placeholders",
        "selection": "one shrinkage strength per bandwidth, mean validation NMSE over all SNRs and configuration counts"}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        write_json(path, config)
        study = Study(ROOT, path, run_id, device)
    def timeout():
        write_json(study.path / "status.json", {"status": "timeout"})
        os._exit(124)
    timer = threading.Timer(settings["timeout_seconds"], timeout)
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        source = ROOT / "runs" / config["comparison"]["parent_run"] / "cache" / (frozen_parent["parent_provenance"]["cache_id"] + ".pt")
        if file_hash(source) != frozen_parent["parent_provenance"]["cache_sha256"]:
            raise ValueError("source training cache changed")
        parent = torch.load(source, map_location="cpu", weights_only=True)
        if any(settings[key] > len(parent["splits"][split][0]["truth"])
                for key, split in (("training_scenes", "train"), ("validation_scenes", "validation"))):
            raise ValueError("cannot add training or validation information beyond the source run")
        validation = sweep_cache(config, parent, "validation", settings["validation_scenes"], config["system"]["snrs_db"])
        train = {c["bandwidth_hz"]: e["truth"][:settings["training_scenes"]].to(study.device)
            for c, e in zip(parent["contexts"], parent["splits"]["train"])}
        validation_rows = []
        for context, entry in zip(validation["contexts"], validation["splits"]["validation"]):
            for strength in grid:
                stats = lmmse_statistics(train[context["bandwidth_hz"]], context["matrix"].to(study.device),
                    context["noise_complex_variance"], frequency_shrinkage=strength)
                prediction = apply_lmmse(stats, entry["observation"].to(study.device)).cpu()
                validation_rows.append({"bandwidth_hz": context["bandwidth_hz"], "configurations": context["configurations"],
                    "snr_db": context["snr_db"], "frequency_shrinkage": strength,
                    "mean_nmse_linear": float(nmse(entry["truth"], prediction).mean())})
            study.log("validation_context", context=entry["context_index"])
        choices = {str(band): min(grid, key=lambda strength: (np.mean([r["mean_nmse_linear"]
            for r in validation_rows if r["bandwidth_hz"] == band and r["frequency_shrinkage"] == strength]), -strength))
            for band in config["system"]["bandwidths_hz"]}
        write_csv(study.export / "validation_scores.csv", validation_rows)
        original_statistics_path = original / "cache/lmmse_statistics.pt"
        if file_hash(original_statistics_path) != frozen_parent["lmmse_sha256"]:
            raise ValueError("source LMMSE statistics changed")
        original_statistics = torch.load(original_statistics_path, map_location="cpu", weights_only=True)
        maximum_original_statistics_difference = 0.
        fitted = {}
        for context in validation["contexts"]:
            key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
            fitted[key] = {}
            for method, strength in (("lmmse_joint", 0.), ("lmmse_per_tone", 1.),
                    ("lmmse_shrunk", choices[str(key[0])])):
                stats = lmmse_statistics(train[key[0]], context["matrix"].to(study.device), context["noise_complex_variance"],
                    frequency_shrinkage=strength)
                fitted[key][method] = {name: value.cpu() for name, value in stats.items()}
            if settings["training_scenes"] == config["comparison"]["training_scenes"]:
                difference = max(float((value-original_statistics[key][name]).abs().max())
                    for name, value in fitted[key]["lmmse_joint"].items())
                maximum_original_statistics_difference = max(maximum_original_statistics_difference, difference)
                assert difference <= 1e-9
        stats_path = study.path / "cache/statistics.pt"
        torch.save(fitted, stats_path)
        frozen = {"selected_frequency_shrinkage": choices, "selection_split": "validation", "screening_used": False,
            "frozen_at_unix": time.time(), "statistics_sha256": file_hash(stats_path),
            "training_cache_sha256": file_hash(source), "source_snr_freeze_sha256": file_hash(original / "stages/frozen.json"),
            "validation_scores_sha256": file_hash(study.export / "validation_scores.csv")}
        write_json(study.path / "stages/frozen.json", frozen)
        write_json(study.export / "frozen.json", frozen)
        study.log("frozen", choices=choices)
        observations = sweep_cache(config, parent, "screen", settings["screen_scenes"], config["system"]["snrs_db"])
        previous = torch.load(original / "predictions/observations.pt", map_location="cpu", weights_only=True)
        screen_hashes = {tensor_hash(h) for h in observations["splits"]["screen"][0]["truth"]}
        seen = {tensor_hash(h) for split in parent["splits"].values() for h in split[0]["truth"]}
        seen.update(tensor_hash(h) for h in previous["splits"]["screen"][0]["truth"])
        if screen_hashes & seen or len(screen_hashes) != settings["screen_scenes"]:
            raise AssertionError("new screen overlaps previous training, validation or screening geometries")
        torch.save(observations, study.path / "predictions/observations.pt")
        rows, decomposition, maximum_decomposition_error = [], [], 0.
        for context, entry in zip(observations["contexts"], observations["splits"]["screen"]):
            key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
            a, y, h = (x.to(device=study.device, dtype=torch.complex128)
                for x in (context["matrix"], entry["observation"], entry["truth"]))
            pinv = torch.linalg.pinv(a)
            clean = torch.einsum("kjn,bkn->bkj", a, h)
            null = torch.einsum("knj,bkj->bkn", pinv, clean) - h
            noise_error = torch.einsum("knj,bkj->bkn", pinv, y-clean)
            energy = h.abs().square().sum((1, 2))
            floor = null.abs().square().sum((1, 2)) / energy
            amplified = noise_error.abs().square().sum((1, 2)) / energy
            ls = torch.einsum("knj,bkj->bkn", pinv, y)
            total = (ls-h).abs().square().sum((1, 2)) / energy
            difference = float((total-floor-amplified).abs().max())
            maximum_decomposition_error = max(maximum_decomposition_error, difference)
            assert torch.allclose(total, floor+amplified, atol=1e-10, rtol=1e-10)
            assert torch.allclose(a @ pinv, torch.eye(a.shape[1], device=a.device, dtype=a.dtype)[None], atol=1e-10, rtol=1e-10)
            expected = context["noise_complex_variance"] * pinv.abs().square().sum() / energy
            decomposition += [{"context": entry["context_index"], "scene": g, "null_floor_linear": float(floor[g]),
                "actual_noise_nmse_linear": float(amplified[g]), "expected_noise_nmse_linear": float(expected[g]),
                "ls_nmse_linear_complex128": float(total[g])} for g in range(len(y))]
            output = {"ls_min_norm": ls.to(entry["observation"].dtype).cpu()}
            output.update({method: apply_lmmse(stats, entry["observation"].to(study.device)).cpu()
                for method, stats in fitted[key].items()})
            score_entry = {**entry, "sbl_converged": torch.zeros(len(y), dtype=torch.bool),
                "sbl_iterations": torch.zeros(len(y), dtype=torch.int64), "sbl_relative_change": torch.zeros(len(y)),
                "coarse_ms_per_sample": 0.}
            for method, prediction in output.items():
                rows.extend(metric_rows(config, score_entry, context, prediction, method, 0., 0))
            torch.save(output, study.path / "predictions" / f"context_{entry['context_index']}.pt")
            study.log("screen_context", context=entry["context_index"])
        write_csv(study.export / "metrics.csv", rows)
        assert len(rows) == len(observations["contexts"]) * len(LABELS) * settings["screen_scenes"]
        hashes = {}
        for row in rows:
            key = row["context"], row["scene"]
            assert hashes.setdefault(key, row["observation_sha256"]) == row["observation_sha256"]
        write_csv(study.export / "ls_decomposition.csv", decomposition)
        summary, _ = summarize(rows, config)
        for row in summary:
            row["configurations"] = row["pilot_re"] // config["system"]["tones"]
            row["mean_nmse_linear"] = float(10 ** (row["mean_nmse_db"] / 10))
        write_csv(study.export / "summary.csv", summary)
        pooled = [{"snr_db": snr, "method": method, "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in rows
            if r["snr_db"] == snr and r["method"] == method]))} for snr in config["system"]["snrs_db"] for method in LABELS]
        for row in pooled:
            row["mean_nmse_db"] = float(10 * np.log10(row["mean_nmse_linear"]))
        write_csv(study.export / "pooled_snr_summary.csv", pooled)
        maximum_metric_difference = 0.
        for context, entry in zip(observations["contexts"], observations["splits"]["screen"]):
            outputs = torch.load(study.path / "predictions" / f"context_{entry['context_index']}.pt", weights_only=True)
            for method, prediction in outputs.items():
                actual = nmse(entry["truth"], prediction).tolist()
                saved = [r["nmse_linear"] for r in rows if r["context"] == entry["context_index"] and r["method"] == method]
                maximum_metric_difference = max(maximum_metric_difference, max(abs(a-b) for a, b in zip(actual, saved)))
        assert maximum_metric_difference == 0.
        assert file_hash(stats_path) == frozen["statistics_sha256"] and file_hash(source) == frozen["training_cache_sha256"]
        audit = {"metrics_rows": len(rows), "maximum_recomputed_nmse_difference": maximum_metric_difference,
            "maximum_original_statistics_difference": maximum_original_statistics_difference,
            "ls_decomposition_checks": len(decomposition), "maximum_ls_decomposition_difference": maximum_decomposition_error,
            "fresh_independent_geometries": len(screen_hashes), "validation_frozen_before_screen": True,
            "common_observation_checks": len(hashes),
            "statistics_and_source_cache_unchanged": True, "elapsed_seconds": time.time()-study.started}
        write_json(study.export / "audit.json", audit)
        plot(summary, study.export)
        write_json(study.path / "status.json", {"status": "complete", **audit})
        write_json(study.export / "status.json", {"status": "complete", **audit})
        study.log("complete", **audit)
        return study.export
    except Exception as error:
        write_json(study.path / "status.json", {"status": "failed", "error": str(error)})
        (study.path / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(run(args.config, args.parent, args.run_id, args.device))
