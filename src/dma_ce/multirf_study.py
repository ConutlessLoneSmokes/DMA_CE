"""Planar multi-RF DMA: paired LS, selected LMMSE, SBL and residual DDPM."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from ._upstream.ddpm import VPSchedule
from .conditional import ConditionalCNN, predict
from .data import encode, estimate_entry, tensor_hash
from .experiment import apply_dc, file_hash, nmse, write_csv, write_json
from .factor_lmmse import apply as apply_factor, fit as fit_factor
from .physical_study import archive, case_hardware, make_entries, prepare_case, score_context, validate
from .research import save_tensor
from .snr_study import train_network


METHODS = ("ls_min_norm", "lmmse_selected", "sbl_dense", "ddpm")


def validate_multirf(config):
    validate(config)
    s, d = config["system"], config["ddpm"]
    if (len(config["cases"]) != 1 or len(s["bandwidths_hz"]) != 1 or len(s["configurations"]) != 1
            or "snrs_db" not in s or type(s["rf_chains"]) is not int or s["rf_chains"] < 2
            or type(s["elements_per_strip"]) is not int or s["elements_per_strip"] < 2
            or s["antennas"] != s["rf_chains"] * s["elements_per_strip"]
            or not np.isfinite([s["spacing_wavelengths"], s["strip_spacing_wavelengths"], config["channel"]["elevation_deg"]]).all()
            or min(s["spacing_wavelengths"], s["strip_spacing_wavelengths"]) <= 0
            or not -90 < config["channel"]["elevation_deg"] < 90):
        raise ValueError("one case/band/J and a consistent planar DMA with fixed known elevation required")
    for key in ("seed", "prior_steps", "batch_size", "log_every", "averages", "coarse_iterations", "width"):
        if type(d[key]) is not int or d[key] < (0 if key == "seed" else 1):
            raise ValueError("positive DDPM training, inference and coarse budgets required")
    checkpoints = d["prior_checkpoints"]
    if (not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != d["prior_steps"]
            or any(type(x) is not int or x < 1 for x in checkpoints) or d["width"] < 4
            or type(d["diffusion_steps"]) is not int or type(d["sampling_steps"]) is not int
            or not 2 <= d["sampling_steps"] <= d["diffusion_steps"]
            or not d["dc_multipliers"] or any(not math.isfinite(x) or x < 0 for x in d["dc_multipliers"])):
        raise ValueError("valid checkpoints, VP schedule and nonnegative correction grid required")
    for key in ("iterations", "batch_size", "angle_points", "range_points"):
        if type(config["sbl"][key]) is not int or config["sbl"][key] < 1:
            raise ValueError("positive SBL budgets and dictionary sizes required")
    if not math.isfinite(config["sbl"]["tolerance"]) or config["sbl"]["tolerance"] <= 0:
        raise ValueError("positive finite SBL tolerance required")


def choose_lmmse(config, cache, device, folder, log):
    rows = []
    h = cache["training"][0].to(device)
    for index, (context, entry) in enumerate(zip(cache["contexts"], cache["validation"])):
        for strength in config["frequency_shrinkage_grid"]:
            stats = fit_factor(h, context["matrix"].to(device), context["noise_complex_variance"], strength)
            prediction = apply_factor(stats, entry["observation"].to(device)).cpu()
            rows.append({"context": index, "snr_db": context["snr_db"], "frequency_shrinkage": strength,
                "mean_nmse_linear": float(nmse(entry["truth"], prediction).mean())})
            del stats
        log("lmmse_validation", context=index)
    choice = min(config["frequency_shrinkage_grid"], key=lambda v: (
        np.mean([r["mean_nmse_linear"] for r in rows if r["frequency_shrinkage"] == v]), -v))
    # Store factors per SNR to keep the full-size model off the GPU between uses.
    for index, context in enumerate(cache["contexts"]):
        stats = fit_factor(h, context["matrix"].to(device), context["noise_complex_variance"], choice)
        save_tensor(folder / f"lmmse_{index}.pt", {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in stats.items()})
    return choice, rows


@torch.no_grad()
def ddpm_prediction(config, model, entry, residual_scale, device, seed):
    d = config["ddpm"]
    schedule = VPSchedule(d["diffusion_steps"]).to(device)
    total = torch.zeros_like(entry["coarse"])
    for draw in range(d["averages"]):
        for start in range(0, len(total), d["batch_size"]):
            span = slice(start, start + d["batch_size"])
            coarse = encode(entry["coarse"][span]).to(device)
            zeros = coarse.new_zeros((len(coarse), 1, *coarse.shape[-2:]))
            total[span] += predict(model, schedule, coarse, zeros, "residual", residual_scale,
                d["sampling_steps"], seed=seed + draw * 1000003 + start).cpu()
    return total / d["averages"]


def fit_ddpm(config, cache, training, validation, folder, device, log):
    d = config["ddpm"]
    error = torch.stack([(e["truth"] - e["coarse"]).abs().square().mean() for e in training]).mean()
    scale = float((error / 2).clamp_min(1e-12).sqrt())
    torch.manual_seed(d["seed"])
    model = ConditionalCNN(d["width"]).to(device)
    schedule = VPSchedule(d["diffusion_steps"]).to(device)
    def loss(batch_rng, noise_rng):
        entry = training[int(torch.randint(len(training), (), generator=batch_rng))]
        indices = torch.randint(len(entry["truth"]), (d["batch_size"],), generator=batch_rng)
        coarse = encode(entry["coarse"][indices]).to(device)
        target = encode((entry["truth"][indices] - entry["coarse"][indices]) / scale).to(device)
        times = torch.randint(d["diffusion_steps"], (len(target),), generator=noise_rng, device=device)
        noise = torch.randn(target.shape, generator=noise_rng, device=device)
        zeros = coarse.new_zeros((len(coarse), 1, *coarse.shape[-2:]))
        return F.mse_loss(model(schedule.perturb(target, times, noise), coarse, zeros, times + 1), noise)
    def score():
        scores = []
        for index, entry in enumerate(validation):
            prediction = ddpm_prediction(config, model, entry, scale, device, d["seed"] + 7000 + 1000 * index)
            scores.append(float(nmse(entry["truth"], prediction).mean()))
        return np.mean(scores)
    checkpoint, records, chosen = train_network(model, folder, d, d["seed"], "prior", loss, score, device, log)
    rows = []
    for index, (entry, context) in enumerate(zip(validation, cache["contexts"])):
        prediction = ddpm_prediction(config, model, entry, scale, device, d["seed"] + 7000 + 1000 * index)
        for multiplier in d["dc_multipliers"]:
            output, _ = apply_dc(prediction, entry, context, multiplier * float(error), device)
            rows.append({"context": index, "multiplier": multiplier,
                "mean_nmse_linear": float(nmse(entry["truth"], output).mean())})
    correction = min(d["dc_multipliers"], key=lambda v: (
        np.mean([r["mean_nmse_linear"] for r in rows if r["multiplier"] == v]), v))
    return {"checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint),
        "residual_real_scale": scale, "selected_checkpoint": chosen, "checkpoint_scores": records,
        "correction_multiplier": correction, "correction_complex_variance": correction * float(error)}, rows


def coarse_cache(config, cache, selected, hardware, path, device, log):
    result = {}
    settings = {**config["sbl"], "iterations": config["ddpm"]["coarse_iterations"], "early_stopping": False}
    dictionary = hardware[0]["dictionary"].to(device)
    for split in ("train", "validation"):
        entries = make_entries(config, selected, hardware, cache, "train")[0] if split == "train" else cache["validation"]
        result[split] = []
        for index, (entry, context) in enumerate(zip(entries, cache["contexts"])):
            destination = path / f"coarse_{split}_{index}.pt"
            if destination.exists():
                estimated = torch.load(destination, weights_only=True)
                if tensor_hash(estimated["observation"]) != tensor_hash(entry["observation"]):
                    raise ValueError("coarse cache does not match observations")
            else:
                estimated = estimate_entry(entry, context, dictionary, settings, device, log)
                save_tensor(destination, estimated)
            result[split].append(estimated)
            log("ddpm_coarse", split=split, context=index)
    return result


def export_curves(config, folder, rows):
    rng = np.random.default_rng(config["seed"] + 819)
    indices = rng.integers(config["data"]["screen"], size=(config["bootstrap_samples"], config["data"]["screen"]))
    summary = []
    for snr in config["system"]["snrs_db"]:
        for method in METHODS:
            local = sorted((r for r in rows if r["snr_db"] == snr and r["method"] == method), key=lambda r: r["scene"])
            values = np.array([r["nmse_linear"] for r in local])
            low, high = np.quantile(10 * np.log10(values[indices].mean(1)), [.025, .975])
            summary.append({"snr_db": snr, "method": method, "scenes": len(local),
                "mean_nmse_linear": float(values.mean()), "mean_scene_nmse_db": float(10 * np.log10(values.mean())),
                "nmse_db_ci95_low": float(low), "nmse_db_ci95_high": float(high),
                "mean_physical_mse": float(np.mean([r["mse_physical"] for r in local])),
                "sbl_nonconverged": sum(not r["sbl_converged"] for r in local) if method == "sbl_dense" else ""})
    write_csv(folder / "metrics.csv", rows)
    write_csv(folder / "summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = dict(zip(METHODS, ("LS", "Validation-selected LMMSE", "SBL", "DDPM (SBL residual)")))
    for method in METHODS:
        local = [r for r in summary if r["method"] == method]
        x = [r["snr_db"] for r in local]
        line, = ax.plot(x, [r["mean_scene_nmse_db"] for r in local], "o-", label=labels[method])
        ax.fill_between(x, [r["nmse_db_ci95_low"] for r in local], [r["nmse_db_ci95_high"] for r in local],
            color=line.get_color(), alpha=.1)
        if method == "sbl_dense":
            bad = [r for r in local if r["sbl_nonconverged"]]
            ax.scatter([r["snr_db"] for r in bad], [r["mean_scene_nmse_db"] for r in bad], marker="x", color="black")
    s = config["system"]
    prefix = "IMPLEMENTATION CHECK | " if config.get("implementation_only", False) else ""
    ax.set(xlabel="Training-average received SNR per measurement (dB)", ylabel="Mean scene NMSE (dB)",
        title=prefix + f"DMA {s['rf_chains']} x {s['elements_per_strip']}, {s['rf_chains']} RF chains; B={s['bandwidths_hz'][0]/1e9:g} GHz, J={s['configurations'][0]}")
    ax.set_xticks(s["snrs_db"])
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"snr_nmse.{suffix}", dpi=180)
    plt.close(fig)
    return summary


def run(root, config_path, run_id, device_override=None):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("safe independent run ID required")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if device_override:
        config["device"] = device_override
    validate_multirf(config)
    root = Path(root)
    path, export = root / "runs" / run_id, root / "results" / run_id
    if path.exists():
        if json.loads((path / "config_resolved.json").read_text()) != config:
            raise ValueError("configuration changed; use a new run ID")
        manifest = json.loads((path / "source_manifest.json").read_text())
        current = {p.relative_to(root).as_posix(): file_hash(p) for folder in ("src", "scripts")
            for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts}
        if current != {k: v for k, v in manifest.items() if k.startswith(("src/", "scripts/")) and k.endswith(".py")}:
            raise ValueError("execution source changed; use a new run ID")
    else:
        path.mkdir(parents=True)
        export.mkdir(parents=True, exist_ok=False)
        for folder in ("cache", "inputs", "predictions", "models"):
            (path / folder).mkdir()
        archive(root, path, export, config)
    def log(event, **fields):
        record = {"event": event, "at_unix": time.time(), **fields}
        with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    def status(value):
        write_json(path / "status.json", value)
        write_json(export / "status.json", value)
    def timeout():
        status({"status": "timeout", "limit_seconds": config["timeout_seconds"]})
        os._exit(124)
    timer = threading.Timer(config["timeout_seconds"], timeout)
    timer.daemon = True
    timer.start()
    started = time.time()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(config["device"])
        case = config["cases"][0]
        selected, _, hardware = case_hardware(config, case)
        cache_path = path / "cache/physical.pt"
        frozen_path = path / "frozen.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            for name, digest in frozen["files_sha256"].items():
                if file_hash(path / name) != digest:
                    raise ValueError("frozen training, model or source file changed")
        if cache_path.exists():
            cache = torch.load(cache_path, weights_only=True)
        else:
            cache = prepare_case(config, case)
            save_tensor(cache_path, cache)
        if not frozen_path.exists():
            choice, scores = choose_lmmse(config, cache, device, path / "cache", log)
            write_csv(export / "lmmse_validation.csv", scores)
            coarse = coarse_cache(config, cache, selected, hardware, path / "cache", device, log)
            model_info, correction_scores = fit_ddpm(config, cache, coarse["train"], coarse["validation"],
                path / "models", device, log)
            write_csv(export / "ddpm_correction_validation.csv", correction_scores)
            frozen = {"selection_split": "validation", "screening_used": False, "frozen_at_unix": time.time(),
                "frequency_shrinkage": choice, "ddpm": model_info,
                "files_sha256": {p.relative_to(path).as_posix(): file_hash(p) for p in
                    [*sorted((path / "cache").glob("*.pt")), path / "models" / model_info["checkpoint"], path / "source.zip"]}}
            write_json(frozen_path, frozen)
            del coarse
        shutil.copyfile(frozen_path, export / "frozen.json")
        log("frozen", sha256=file_hash(frozen_path))
        model = ConditionalCNN(config["ddpm"]["width"]).to(device)
        model.load_state_dict(torch.load(path / "models" / frozen["ddpm"]["checkpoint"], weights_only=True)["state_dict"])
        model.eval()
        observed_path = path / "predictions/observations.pt"
        if observed_path.exists():
            observations = torch.load(observed_path, weights_only=True)
        else:
            entries, ids = make_entries(config, selected, hardware, cache, "screen")
            observations = {"entries": entries, "geometry_ids": ids, "frozen_sha256": file_hash(frozen_path), "created_at_unix": time.time()}
            save_tensor(observed_path, observations)
        ids = observations["geometry_ids"]
        if (observations["frozen_sha256"] != file_hash(frozen_path)
                or observations["created_at_unix"] < frozen["frozen_at_unix"]
                or len(set(ids)) != config["data"]["screen"]
                or set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"])):
            raise AssertionError("independent screen observations must follow the frozen choices")
        rows, maximum_difference = [], 0.
        coarse_settings = {**config["sbl"], "iterations": config["ddpm"]["coarse_iterations"], "early_stopping": False}
        dictionary = hardware[0]["dictionary"].to(device)
        for index, (context, entry) in enumerate(zip(cache["contexts"], observations["entries"])):
            destination = path / "predictions" / f"context_{index}.pt"
            observation_hash = tensor_hash(entry["observation"])
            if destination.exists():
                saved = torch.load(destination, weights_only=True)
                if saved["observation_sha256"] != observation_hash or saved["frozen_sha256"] != file_hash(frozen_path):
                    raise ValueError("prediction cache does not match the frozen observations")
            else:
                a, y = context["matrix"].to(device), entry["observation"].to(device)
                outputs = {"ls_min_norm": torch.einsum("knj,bkj->bkn", torch.linalg.pinv(a.to(torch.complex128)),
                    y.to(torch.complex128)).to(y.dtype).cpu()}
                stats = torch.load(path / "cache" / f"lmmse_{index}.pt", weights_only=True)
                outputs["lmmse_selected"] = apply_factor(stats, y).cpu()
                del stats
                estimated = estimate_entry(entry, context, dictionary, config["sbl"], device, log)
                outputs["sbl_dense"] = estimated["coarse"]
                coarse = estimate_entry(entry, context, dictionary, coarse_settings, device, log)
                prediction = ddpm_prediction(config, model, coarse, frozen["ddpm"]["residual_real_scale"], device,
                    config["ddpm"]["seed"] + 17000 + 1000 * index)
                outputs["ddpm"] = apply_dc(prediction, entry, context, frozen["ddpm"]["correction_complex_variance"], device)[0]
                saved = {"outputs": outputs, "solver": {k: v for k, v in estimated.items() if k.startswith("sbl_") or k == "coarse_ms_per_sample"},
                    "ddpm_coarse": coarse["coarse"], "observation_sha256": observation_hash, "frozen_sha256": file_hash(frozen_path)}
                save_tensor(destination, saved)
            scored = score_context(config, cache, context, entry, saved)
            for method in METHODS:
                for row in (r for r in scored if r["method"] == method):
                    i = row["scene"]
                    truth = entry["truth"][i].to(torch.complex128)
                    error = saved["outputs"][method][i].to(torch.complex128) - truth
                    exact = float(error.abs().square().sum() / truth.abs().square().sum())
                    maximum_difference = max(maximum_difference, abs(exact - row["nmse_linear"]) / max(exact, 1e-30))
                    row.update(rf_chains=context["rf_chains"], measurements_per_tone=context["measurements_per_tone"],
                        geometry_id=ids[i])
            rows.extend(scored)
            log("screen_context", context=index, snr_db=context["snr_db"])
        if len(rows) != len(cache["contexts"]) * len(ids) * len(METHODS) or maximum_difference > 1e-5:
            raise AssertionError("incomplete results or mismatched independently recomputed NMSE")
        summary = export_curves(config, export, rows)
        budgets = []
        for c in cache["contexts"]:
            duration = c["configurations"] * config["system"]["tones"] / c["bandwidth_hz"]
            budgets.append({"snr_db": c["snr_db"], "rf_chains": c["rf_chains"], "configurations": c["configurations"],
                "pilot_re": c["pilot_re"], "measurements_per_tone": c["measurements_per_tone"],
                "transmit_power_dbm": c["transmit_power_dbm"], "useful_pilot_duration_s": duration,
                "useful_pilot_energy_j": c["transmit_power_w"] * duration,
                "physical_white_noise_per_tone_w": c["physical_white_noise_per_tone_w"],
                "raw_noise_min_w": float(c["raw_noise_per_tone_w"].min()), "raw_noise_max_w": float(c["raw_noise_per_tone_w"].max()),
                "channel_scale": cache["channel_scale"]})
        write_csv(export / "link_budgets.csv", budgets)
        snr_error = max(abs(10 * math.log10(c["training_white_power"] / c["noise_complex_variance"]) - c["snr_db"])
            for c in cache["contexts"])
        if len({e["physical_noise_sha256"] for e in observations["entries"]}) != 1 or snr_error > 1e-6:
            raise AssertionError("physical noise or training-only SNR calibration changed across SNRs")
        audit = {"status": "passed", "metrics_recomputed": len(rows), "maximum_relative_nmse_difference": maximum_difference,
            "independent_screen_geometries": len(ids), "frozen_before_screen": True,
            "physical_noise_fixed_across_snr": True, "maximum_snr_calibration_error_db": snr_error,
            "implementation_only": config.get("implementation_only", False),
            "sbl_nonconverged": sum(r["sbl_nonconverged"] for r in summary if r["method"] == "sbl_dense"),
            "elapsed_seconds": time.time() - started}
        write_json(export / "audit.json", audit)
        report = ["# 多RF DMA的四方法SNR比较", "",
            "实现检查，不用于性能结论。" if config.get("implementation_only", False) else "单组训练几何、单组DMA状态、单次网络初始化的探索性实验。",
            f"阵列：{config['system']['rf_chains']}条微带，每带{config['system']['elements_per_strip']}单元；J={config['system']['configurations'][0]}。",
            f"独立几何：训练{config['data']['train']}、验证{config['data']['validation']}、筛选{len(ids)}。",
            "横轴是训练平均的每复观测接收SNR；纵轴先平均逐场景线性NMSE，再转dB。四方法共用观测和物理噪声。",
            f"验证冻结：LMMSE收缩系数={frozen['frequency_shrinkage']}；DDPM断点={frozen['ddpm']['selected_checkpoint']['step']}；一致性倍率={frozen['ddpm']['correction_multiplier']}。",
            f"SBL未收敛输出{audit['sbl_nonconverged']}个，仍包含于曲线；完整记录见metrics.csv。",
            "DDPM为当前SBL条件残差模型，在本阵列上重新训练；多次DDIM输出平均后执行一次观测一致性更新。",
            "区间只覆盖固定训练/硬件下的筛选几何波动。完整配置、源码、模型、观测和预测已归档。", "",
            "| SNR/dB | LS | Validation-selected LMMSE | SBL | DDPM |", "|---:|---:|---:|---:|---:|"]
        for snr in config["system"]["snrs_db"]:
            values = {r["method"]: r["mean_scene_nmse_db"] for r in summary if r["snr_db"] == snr}
            report.append(f"| {snr:g} | " + " | ".join(f"{values[m]:.4f}" for m in METHODS) + " |")
        (export / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        status({**audit, "status": "complete"})
        return export
    except BaseException as error:
        status({"status": "failed", "error": repr(error), "elapsed_seconds": time.time() - started})
        raise
    finally:
        timer.cancel()
