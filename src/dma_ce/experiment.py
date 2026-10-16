"""Bounded, source-archived pilot experiment with validation frozen before screening."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile

import numpy as np
import torch

from .conditional import VARIANTS, predict, train_model, reliability_features
from ._upstream.ddpm import VPSchedule
from .data import encode, prepare, sync, tensor_hash
from .sbl import consistency


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    for name in ("train", "validation", "screen"):
        if type(config["data"][name]) is not int or config["data"][name] < 2:
            raise ValueError("at least two independent scenes per split required")
    system, channel, learning = config["system"], config["channel"], config["learning"]
    if (system["antennas"] < 4 or system["tones"] < 4 or not system["configurations"]
            or any(type(j) is not int or not 1 <= j <= system["antennas"] for j in system["configurations"])
            or any(not 0 < b < 2 * system["carrier_hz"] for b in system["bandwidths_hz"])
            or not system["snrs_db"] or not all(np.isfinite(system["snrs_db"]))
            or not 0 < channel["range_min_m"] < channel["range_max_m"]
            or not 0 < channel["angle_limit_deg"] < 90 or channel["paths"] < 1
            or channel["include_absorption"] is not False
            or not 2 <= learning["sampling_steps"] <= learning["diffusion_steps"]
            or min(learning["steps"], learning["batch_size"], learning["log_every"], config["timeout_seconds"]) < 1
            or min(config["evaluation"]["dc_multipliers"]) <= 0):
        raise ValueError("invalid physical, training, or evaluation configuration")
    for values in (system["configurations"], system["bandwidths_hz"], system["snrs_db"]):
        if len(set(values)) != len(values):
            raise ValueError("duplicate experimental conditions")


@torch.no_grad()
def predict_entry(config, cache, model, variant, entry, device, *, seed, shuffled=False):
    count, batch_size = len(entry["coarse"]), config["evaluation"]["batch_size"]
    reliable = reliability_features(entry["reliability"], cache)
    if shuffled:
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(config["seed"] + 317))
        reliable = reliable[permutation]
    schedule = VPSchedule(config["learning"]["diffusion_steps"]).to(device)
    estimates = []
    sync(device)
    started = time.perf_counter()
    for start in range(0, count, batch_size):
        coarse = encode(entry["coarse"][start:start + batch_size]).to(device)
        estimates.append(predict(model, schedule, coarse, reliable[start:start + batch_size].to(device),
            variant, cache["residual_real_scale"], config["learning"]["sampling_steps"], seed=seed + start).cpu())
    sync(device)
    return torch.cat(estimates), (time.perf_counter() - started) * 1000 / count


def apply_dc(estimate, entry, context, prior_variance, device):
    sync(device)
    started = time.perf_counter()
    output = consistency(estimate.to(device), context["matrix"].to(device), entry["observation"].to(device),
        entry["variance"].to(device), prior_variance).cpu()
    sync(device)
    return output, (time.perf_counter() - started) * 1000 / len(estimate)


def nmse(truth, estimate):
    return (estimate - truth).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2)).clamp_min(1e-20)


def freeze_correction(config, cache, models, device, log):
    grid = config["evaluation"]["dc_multipliers"]
    scores = {value: [] for value in grid}
    for index, entry in enumerate(cache["splits"]["validation"]):
        estimate, _ = predict_entry(config, cache, models["residual"], "residual", entry, device,
            seed=config["seed"] + 7000 + index * 1000)
        for value in grid:
            corrected, _ = apply_dc(estimate, entry, cache["contexts"][index],
                value * cache["dc_reference_complex_variance"], device)
            scores[value].extend(nmse(entry["truth"], corrected).tolist())
        log("validation", context=index, samples=len(estimate))
    means = {value: float(np.mean(values)) for value, values in scores.items()}
    choice = min(grid, key=lambda value: (means[value], value))
    return {"selected_multiplier": choice, "prior_complex_variance": choice * cache["dc_reference_complex_variance"],
        "selection_split": "validation", "selection_model": "residual", "shared_by_all_methods": True,
        "validation_mean_nmse_linear": means, "screening_used": False}


def metric_rows(config, entry, context, estimate, method, refine_ms, network_calls):
    if not torch.isfinite(estimate).all():
        raise ArithmeticError("nonfinite screen estimate")
    truth, observed, matrix = entry["truth"], entry["observation"], context["matrix"]
    error = estimate - truth
    energy = truth.abs().square().sum((1, 2)).clamp_min(1e-20)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    coordinate = torch.einsum("krn,bkn->bkr", vh, error)
    row = torch.einsum("krn,bkr->bkn", vh.conj(), coordinate)
    null = error - row
    projected = torch.einsum("kjn,bkn->bkj", matrix, estimate)
    errors = nmse(truth, estimate)
    row_errors = row.abs().square().sum((1, 2)) / energy
    null_errors = null.abs().square().sum((1, 2)) / energy
    if not torch.allclose(errors, row_errors + null_errors, atol=2e-4, rtol=2e-4):
        raise AssertionError("row/null error decomposition failed")
    for scene in range(len(truth)):
        yield {"scene": scene, "context": entry["context_index"], "method": method,
            "bandwidth_hz": context["bandwidth_hz"], "configurations": context["configurations"],
            "pilot_re": context["pilot_re"], "snr_db": context["snr_db"],
            "realized_snr_db": float(entry["realized_snr_db"][scene]),
            "observation_sha256": tensor_hash(observed[scene]),
            "nmse_linear": float(errors[scene]), "nmse_db": float(10 * errors[scene].clamp_min(1e-30).log10()),
            "row_nmse_linear": float(row_errors[scene]), "null_nmse_linear": float(null_errors[scene]),
            "observation_residual": float((projected[scene] - observed[scene]).abs().square().sum()
                / observed[scene].abs().square().sum().clamp_min(1e-20)),
            "sbl_converged": bool(entry["sbl_converged"][scene]),
            "sbl_iterations": int(entry["sbl_iterations"][scene]),
            "sbl_relative_change": float(entry["sbl_relative_change"][scene]),
            "coarse_batch_ms_per_sample": entry["coarse_ms_per_sample"],
            "refine_batch_ms_per_sample": refine_ms,
            "network_calls": network_calls}


def write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, config):
    groups = {}
    for row in rows:
        groups.setdefault((row["context"], row["method"]), []).append(row)
    summary = []
    for (context, method), values in sorted(groups.items()):
        summary.append({"context": context, "method": method, "samples": len(values),
            "bandwidth_hz": values[0]["bandwidth_hz"], "pilot_re": values[0]["pilot_re"],
            "snr_db": values[0]["snr_db"],
            "mean_nmse_db": float(10 * np.log10(np.mean([v["nmse_linear"] for v in values]))),
            "mean_observation_residual": float(np.mean([v["observation_residual"] for v in values])),
            "sbl_convergence_rate": float(np.mean([v["sbl_converged"] for v in values])),
            "coarse_batch_ms_per_sample": values[0]["coarse_batch_ms_per_sample"],
            "refine_batch_ms_per_sample": values[0]["refine_batch_ms_per_sample"]})
    # A geometry's conditions are paired; bootstrap geometry blocks, not individual condition rows.
    effects = []
    comparisons = [("reliable_residual", "residual"), ("reliable_residual_dc", "residual_dc"),
        ("reliable_residual_shuffled", "reliable_residual"),
        ("reliable_residual_shuffled_dc", "reliable_residual_dc"),
        ("deterministic_dc", "reliable_residual_dc")]
    rng = np.random.default_rng(config["seed"] + 391)
    for target, reference in comparisons:
        differences = {}
        lookup = {(r["context"], r["scene"], r["method"]): r["nmse_db"] for r in rows}
        for row in rows:
            if row["method"] == target:
                delta = row["nmse_db"] - lookup[(row["context"], row["scene"], reference)]
                differences.setdefault(row["scene"], []).append(delta)
        values = np.array([np.mean(v) for _, v in sorted(differences.items())])
        sampled = rng.choice(values, (config["evaluation"]["bootstrap_samples"], len(values)), replace=True).mean(1)
        low, high = np.quantile(sampled, [.025, .975])
        effects.append({"target": target, "reference": reference, "geometry_blocks": len(values),
            "mean_paired_db": float(values.mean()), "ci95_low_db": float(low), "ci95_high_db": float(high),
            "interpretation": "descriptive_single_training_seed"})
    return summary, effects


def calibration_rows(cache):
    edges = torch.tensor(cache["calibration_edges"][1:-1])
    rows = []
    for index, entry in enumerate(cache["splits"]["screen"]):
        predicted = entry["reliability"] * cache["reliability_scale"]
        actual = (entry["truth"] - entry["coarse"]).abs().square()
        bins = torch.bucketize(predicted.clamp_min(1e-10).log10().contiguous(), edges)
        for bucket in range(5):
            selected = bins == bucket
            if selected.any():
                rows.append({"context": index, "bin": bucket, "coordinates": int(selected.sum()),
                    "predicted_complex_mse": float(predicted[selected].mean()),
                    "actual_complex_mse": float(actual[selected].mean())})
    return rows


def plot_results(path, summary, calibration):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    methods = ("coarse", "residual", "reliable_residual", "residual_dc", "reliable_residual_dc", "deterministic_dc")
    bands, snrs = sorted({x["bandwidth_hz"] for x in summary}), sorted({x["snr_db"] for x in summary})
    figure, axes = plt.subplots(len(snrs), len(bands), figsize=(5 * len(bands), 3.5 * len(snrs)), squeeze=False)
    for row, snr in enumerate(snrs):
        for col, band in enumerate(bands):
            axis = axes[row, col]
            for method in methods:
                selected = sorted([x for x in summary if x["method"] == method and x["bandwidth_hz"] == band
                    and x["snr_db"] == snr], key=lambda x: x["pilot_re"])
                axis.plot([x["pilot_re"] for x in selected], [x["mean_nmse_db"] for x in selected], marker="o", label=method)
            axis.set(title=f"B={band / 1e9:g} GHz, SNR={snr:g} dB", xlabel="Pilot RE", ylabel="Full CSI NMSE (dB)")
            axis.grid(alpha=.25)
    axes[0, -1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path / "nmse.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4))
    for context in sorted({x["context"] for x in calibration}):
        selected = [x for x in calibration if x["context"] == context]
        axis.loglog([x["predicted_complex_mse"] for x in selected], [x["actual_complex_mse"] for x in selected],
            marker="o", label=str(context))
    points = [x["predicted_complex_mse"] for x in calibration] + [x["actual_complex_mse"] for x in calibration]
    low, high = min(points), max(points)
    axis.loglog([low, high], [low, high], "k--", linewidth=1)
    axis.set(xlabel="Predicted coarse-error variance", ylabel="Actual coarse squared error", title="Training-calibrated reliability: screening")
    axis.legend(title="Context", fontsize=7, ncol=2)
    axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(path / "calibration.png", dpi=160)
    plt.close(figure)


def run(config_path, run_id, root, device_override=None):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("run ID must contain only letters, digits, underscores, and hyphens")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if device_override:
        config["device"] = device_override
    validate_config(config)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select CPU for a smoke run")
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "models").mkdir()
    write_json(path / "config_resolved.json", config)
    started = time.time()
    def log(event, **fields):
        record = {"event": event, "elapsed_seconds": round(time.time() - started, 3), **fields}
        with (path / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    def timeout():
        log("hard_timeout", limit_seconds=config["timeout_seconds"])
        write_json(path / "status.json", {"status": "timeout", "elapsed_seconds": time.time() - started})
        os._exit(124)
    timer = threading.Timer(config["timeout_seconds"], timeout)
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "command": sys.argv, "training_seed": config["seed"]}
        write_json(path / "environment.json", environment)
        sources = [root / ".gitignore", root / "AGENTS.md", root / "pyproject.toml"]
        for folder in ("src", "scripts", "configs", "tests"):
            sources.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        with zipfile.ZipFile(path / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in sources:
                archive.write(source, source.relative_to(root).as_posix())
        write_json(path / "source_manifest.json", {p.relative_to(root).as_posix(): file_hash(p) for p in sources})
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        write_json(path / "version.json", {"git_commit": git_head.stdout.strip() if git_head.returncode == 0 else None,
            "source_zip_sha256": file_hash(path / "source.zip")})
        write_json(path / "status.json", {"status": "preparing", "label": config["label"]})
        log("started", device=str(device), label=config["label"])
        cache = prepare(config, device, log)
        torch.save(cache, path / "data.pt")
        cache_hash = file_hash(path / "data.pt")
        log("data_complete", data_sha256=cache_hash)
        write_json(path / "status.json", {"status": "training", "label": config["label"]})
        models = {variant: train_model(config, cache, variant, device, path / "models" / f"{variant}.pt", log) for variant in VARIANTS}
        frozen = freeze_correction(config, cache, models, device, log)
        frozen.update({"data_sha256": cache_hash, "models": {v: file_hash(path / "models" / f"{v}.pt") for v in VARIANTS},
            "frozen_at_unix": time.time(), "geometry_hashes": cache["geometry_hashes"]})
        write_json(path / "frozen_validation.json", frozen)
        freeze_hash = file_hash(path / "frozen_validation.json")
        log("frozen", multiplier=frozen["selected_multiplier"], sha256=freeze_hash)
        write_json(path / "status.json", {"status": "screening", "label": config["label"]})
        rows, tone_rows = [], []
        predictions = []
        for index, entry in enumerate(cache["splits"]["screen"]):
            context = cache["contexts"][index]
            estimates = {"coarse": (entry["coarse"], 0., 0)}
            for variant in VARIANTS:
                estimate, ms = predict_entry(config, cache, models[variant], variant, entry, device,
                    seed=config["seed"] + 17000 + index * 1000)
                calls = 1 if variant == "deterministic" else config["learning"]["sampling_steps"]
                estimates[variant] = (estimate, ms, calls)
                corrected, dc_ms = apply_dc(estimate, entry, context, frozen["prior_complex_variance"], device)
                estimates[variant + "_dc"] = (corrected, ms + dc_ms, calls)
            estimate, ms = predict_entry(config, cache, models["reliable_residual"], "reliable_residual", entry, device,
                seed=config["seed"] + 17000 + index * 1000, shuffled=True)
            estimates["reliable_residual_shuffled"] = (estimate, ms, config["learning"]["sampling_steps"])
            corrected, dc_ms = apply_dc(estimate, entry, context, frozen["prior_complex_variance"], device)
            estimates["reliable_residual_shuffled_dc"] = (corrected, ms + dc_ms, config["learning"]["sampling_steps"])
            for method, (estimate, ms, calls) in estimates.items():
                rows.extend(metric_rows(config, entry, context, estimate, method, ms, calls))
                errors = (estimate - entry["truth"]).abs().square().sum(-1)
                energy = entry["truth"].abs().square().sum(-1).clamp_min(1e-20)
                for tone in range(energy.shape[1]):
                    tone_rows.append({"context": index, "method": method, "tone": tone,
                        "mean_nmse_linear": float((errors / energy)[:, tone].mean())})
            predictions.append({method: values[0] for method, values in estimates.items()})
            log("screen", context=index, methods=len(estimates), samples=len(entry["truth"]))
        expected = len(cache["contexts"]) * config["data"]["screen"] * 11
        assert len(rows) == expected, (len(rows), expected)
        assert file_hash(path / "frozen_validation.json") == freeze_hash
        torch.save(predictions, path / "predictions.pt")
        summary, effects = summarize(rows, config)
        calibration = calibration_rows(cache)
        export = root / "results" / run_id
        export.mkdir(parents=True, exist_ok=False)
        write_csv(export / "metrics.csv", rows)
        write_csv(export / "summary.csv", summary)
        write_csv(export / "paired_effects.csv", effects)
        write_csv(export / "calibration.csv", calibration)
        write_csv(export / "tone_metrics.csv", tone_rows)
        write_json(export / "summary.json", {"label": config["label"], "groups": summary, "paired_effects": effects,
            "metrics_rows": len(rows), "independent_screen_geometries": config["data"]["screen"],
            "conditions": len(cache["contexts"]), "single_training_seed": True,
            "timing": "batched amortized time, not batch-one latency", "frozen_validation_sha256": freeze_hash})
        for name in ("config_resolved.json", "environment.json", "version.json", "source_manifest.json", "frozen_validation.json", "source.zip"):
            (export / name).write_bytes((path / name).read_bytes())
        plot_results(export, summary, calibration)
        lines = ["# 首轮条件生成实验", "", "## Material Passport", "", "- Origin Skill: ARS-Codex experiment-agent",
            "- Origin Mode: run", "- Origin Date: 2026-10-16", "- Verification Status: ANALYZED",
            "- Version Label: pilot_v1", "", f"状态：完成。标签：{config['label']}。",
            f"独立筛选几何{config['data']['screen']}个，条件{len(cache['contexts'])}组，逐场景指标{len(rows)}行。",
            "同一几何的条件按块配对；区间仅描述该单训练种子下的几何差异。",
            "", "| 对照 | 配对平均差/dB | 描述性95%区间/dB |", "|---|---:|---|",]
        lines.extend(f"| {x['target']} − {x['reference']} | {x['mean_paired_db']:.4f} | [{x['ci95_low_db']:.4f}, {x['ci95_high_db']:.4f}] |" for x in effects)
        lines.extend(["", "负值表示左侧方法更好。此表均值是逐场景dB配对差，不等于线性均值再转dB之差。",
            "", "![完整CSI](nmse.png)", "", "![可靠度校准](calibration.png)", "",
            "SBL非收敛比例见summary.csv，逐场景迭代/残差见metrics.csv。后验对角是模型内代理，图中实测误差只用于评分。",
            "计算时间为批量均摊，不是单样本在线时延；该缩小系统不支持完整尺寸或通信净速率结论。"])
        (export / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        status = {"status": "complete", "label": config["label"], "elapsed_seconds": time.time() - started,
            "metrics_rows": len(rows), "data_sha256": cache_hash, "frozen_validation_sha256": freeze_hash}
        write_json(path / "status.json", status)
        write_json(export / "status.json", status)
        log("complete", rows=len(rows), output=str(export))
        return export
    except Exception as error:
        write_json(path / "status.json", {"status": "failed", "error": str(error), "elapsed_seconds": time.time() - started})
        (path / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        timer.cancel()
