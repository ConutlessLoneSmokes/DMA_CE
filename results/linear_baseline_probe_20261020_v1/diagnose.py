"""Post-hoc LS error decomposition and covariance generalization controls."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.snr_study import sweep_cache
from dma_ce.structured import apply_lmmse, lmmse_statistics


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    original = ROOT / "runs/snr_comparison_20261020_v1"
    run = ROOT / "runs/linear_baseline_probe_20261020_v1"
    export = ROOT / "results" / run.name
    run.mkdir(exist_ok=False)
    export.mkdir(exist_ok=False)
    cfg = json.loads((original / "config_resolved.json").read_text())
    frozen = json.loads((original / "stages/frozen.json").read_text())
    source = ROOT / "runs" / cfg["comparison"]["parent_run"] / "cache" / (frozen["parent_provenance"]["cache_id"] + ".pt")
    assert file_hash(source) == frozen["parent_provenance"]["cache_sha256"]
    parent = torch.load(source, map_location="cpu", weights_only=True)
    statistics = torch.load(original / "cache/lmmse_statistics.pt", map_location="cpu", weights_only=True)
    previous = torch.load(original / "predictions/observations.pt", map_location="cpu", weights_only=True)
    print("Loaded frozen data", flush=True)
    device = torch.device("cuda")
    settings = {"purpose": "posthoc baseline diagnosis, no model selection", "source_run": original.name,
        "source_cache_sha256": file_hash(source), "source_statistics_sha256": file_hash(original / "cache/lmmse_statistics.pt"),
        "code_sha256": file_hash(Path(__file__)), "old_screen_used_only_for_diagnosis": True}
    for folder in (run, export):
        write_json(folder / "config.json", settings)
        shutil.copyfile(__file__, folder / "diagnose.py")
    rows, hardware, tones = [], [], {}
    started = time.time()
    caches = {split: sweep_cache(cfg, parent, split, cfg["comparison"][key], cfg["system"]["snrs_db"])
        for split, key in (("train", "training_scenes"), ("validation", "validation_scenes"))}
    caches["old_screen"] = previous
    checks = 0
    for split, cache in caches.items():
        split_key = "screen" if split == "old_screen" else split
        for context, entry in zip(cache["contexts"], cache["splits"][split_key]):
            a = context["matrix"].to(device=device, dtype=torch.complex128)
            h = entry["truth"].to(device=device, dtype=torch.complex128)
            y = entry["observation"].to(device=device, dtype=torch.complex128)
            clean = torch.einsum("kjn,bkn->bkj", a, h)
            noise = y - clean
            sigma = context["noise_complex_variance"]
            energy = h.abs().square().sum((1, 2))
            pinv = torch.linalg.pinv(a)
            ls = torch.einsum("knj,bkj->bkn", pinv, y)
            projected = torch.einsum("knj,bkj->bkn", pinv, clean)
            null = projected - h
            amplified = torch.einsum("knj,bkj->bkn", pinv, noise)
            null_error = null.abs().square().sum((1, 2)) / energy
            noise_error = amplified.abs().square().sum((1, 2)) / energy
            actual = (ls - h).abs().square().sum((1, 2)) / energy
            assert torch.allclose(actual, null_error + noise_error, atol=2e-10, rtol=2e-10)
            assert torch.allclose(a @ pinv, torch.eye(a.shape[1], device=device, dtype=a.dtype).expand(a.shape[0], -1, -1), atol=1e-10, rtol=1e-10)
            expected_noise = sigma * pinv.abs().square().sum() / energy
            key = context["bandwidth_hz"], context["configurations"], context["snr_db"]
            if key not in tones:
                train_h = next(e["truth"] for c, e in zip(parent["contexts"], parent["splits"]["train"])
                    if c["bandwidth_hz"] == key[0]).to(device)
                tones[key] = [lmmse_statistics(train_h[:, k:k+1], a[k:k+1], sigma) for k in range(a.shape[0])]
            tone = torch.cat([apply_lmmse(s, y[:, k:k+1]) for k, s in enumerate(tones[key])], 1)
            stats = {k: v.to(device) for k, v in statistics[key].items()}
            joint = apply_lmmse(stats, y)
            joint_clean = apply_lmmse(stats, clean)
            clean_error = (joint_clean - h).abs().square().sum((1, 2))
            expected_joint_error = clean_error + sigma * stats["gain"].abs().square().sum()
            for method, prediction in (("ls", ls), ("lmmse_joint", joint), ("lmmse_per_tone", tone)):
                errors = (prediction - h).abs().square().sum((1, 2))
                values = {"split": split, "bandwidth_hz": key[0], "configurations": key[1], "snr_db": key[2],
                    "method": method, "geometries": len(h), "mean_nmse_linear": float((errors / energy).mean()),
                    "ratio_of_summed_errors_to_energy": float(errors.sum() / energy.sum()),
                    "mean_mse": float(errors.mean() / h[0].numel()),
                    "mean_truth_energy": float(energy.mean()), "min_truth_energy": float(energy.min()),
                    "max_truth_energy": float(energy.max()), "mean_realized_snr_db": float(entry["realized_snr_db"].mean()),
                    "null_floor_linear": float(null_error.mean()), "ls_noise_nmse_linear": float(noise_error.mean()),
                    "expected_ls_noise_nmse_linear": float(expected_noise.mean()),
                    "expected_joint_nmse_linear": float((expected_joint_error / energy).mean()),
                    "joint_noiseless_bias_nmse_linear": float((clean_error / energy).mean()),
                    "expected_joint_noise_nmse_linear": float((sigma * stats["gain"].abs().square().sum() / energy).mean()),
                    "zero_estimator_nmse_linear": 1.}
                rows.append(values)
            if split == "train":
                s = torch.linalg.svdvals(a)
                hardware.append({"bandwidth_hz": key[0], "configurations": key[1], "snr_db": key[2],
                    "minimum_singular_value": float(s.min()), "maximum_condition_number": float((s[:, 0] / s[:, -1]).max()),
                    "noise_amplification_trace": float(pinv.abs().square().sum()),
                    "measured_noise_variance_ratio": float(noise.abs().square().mean() / sigma),
                    "channel_covariance_dimension": h[0].numel(), "maximum_sample_covariance_rank": len(h)-1,
                    "minimum_observed_covariance_eigenvalue": float(torch.linalg.eigvalsh(
                        (torch.einsum("kjn,bkn->bkj", a, h) - stats["observed_mean"]).flatten(1).T
                        @ (torch.einsum("kjn,bkn->bkj", a, h) - stats["observed_mean"]).flatten(1).conj() / (len(h)-1)).min())})
            checks += 1
        print(json.dumps({"completed_split": split, "contexts": checks, "seconds": time.time()-started}), flush=True)
    write_csv(export / "contexts.csv", rows)
    write_csv(export / "hardware.csv", hardware)
    write_json(export / "audit.json", {"ls_orthogonal_decomposition_checks": checks, "pseudoinverse_checks": checks,
        "noise_and_formula_dtype": "complex128", "elapsed_seconds": time.time()-started,
        "training_cache_unchanged": file_hash(source) == settings["source_cache_sha256"]})
    print(json.dumps({"status": "complete", "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
