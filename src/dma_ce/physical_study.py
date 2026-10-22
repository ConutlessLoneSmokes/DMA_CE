"""Physical baselines with fixed thermal noise and power or observation sweeps."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import re
import shutil
import sys
import threading
import time
import zipfile

import numpy as np
import torch

from .data import build_hardware, estimate_entry, physical_channels, sample_scenes, tensor_hash
from .experiment import file_hash, metric_rows, nmse, write_csv, write_json
from .research import save_tensor
from .sbl import SOLVER_VERSION
from .structured import apply_lmmse, lmmse_statistics


METHODS = ("ls_min_norm", "lmmse_joint", "lmmse_per_tone", "lmmse_shrunk", "sbl_dense")


def validate(config):
    system, channel = config["system"], config["channel"]
    if ("snrs_db" in system) == ("transmit_power_dbm" in system):
        raise ValueError("choose received-SNR targets or transmit-power levels, not both")
    sweep = "snrs_db" if "snrs_db" in system else "transmit_power_dbm"
    if (type(config["seed"]) is not int or config["seed"] < 0
            or type(config["data"]["screen_seed"]) is not int or config["data"]["screen_seed"] < 0
            or any(type(config["data"][k]) is not int or config["data"][k] < 2 for k in ("train", "validation", "screen"))
            or type(system["antennas"]) is not int or system["antennas"] < 4
            or type(system["tones"]) is not int or system["tones"] < 2
            or not 0 < channel["range_min_m"] < channel["range_max_m"]
            or not 0 < channel["angle_limit_deg"] < 90 or type(channel["paths"]) is not int or channel["paths"] < 1
            or not math.isfinite(channel["antenna_gain_dbi"])
            or not math.isfinite(channel["reflection_refractive_index"]) or channel["reflection_refractive_index"] < 1
            or not math.isfinite(channel["surface_roughness_m"]) or channel["surface_roughness_m"] < 0
            or config["timeout_seconds"] <= 0 or config["bootstrap_samples"] < 2):
        raise ValueError("valid physical system, independent scene counts and finite propagation settings required")
    for name in ("bandwidths_hz", "configurations", sweep):
        values = system[name]
        if not values or len(set(values)) != len(values) or not all(np.isfinite(values)):
            raise ValueError("nonempty unique finite physical conditions required")
    with np.errstate(over="ignore", under="ignore"):
        powers = np.power(10., (np.asarray(system[sweep]) - (0 if sweep == "snrs_db" else 30)) / 10)
    if (any(type(j) is not int or j < 1 for j in system["configurations"])
            or any(not 0 < b < 2 * system["carrier_hz"] for b in system["bandwidths_hz"])
            or not np.isfinite(powers).all() or (powers <= 0).any()):
        raise ValueError("valid bandwidths, configuration counts and positive finite pilot powers required")
    grid = config["frequency_shrinkage_grid"]
    if not grid or len(set(grid)) != len(grid) or any(not np.isfinite(x) or not 0 <= x <= 1 for x in grid):
        raise ValueError("finite covariance shrinkage strengths in [0, 1] required")
    names = []
    for case in config["cases"]:
        if (re.fullmatch(r"[A-Za-z0-9_-]+", case["name"]) is None
                or case["model"] not in ("gaussian", "li2026") or type(case["include_absorption"]) is not bool):
            raise ValueError("safe case names and known propagation models required")
        names.append(case["name"])
    if not names or len(set(names)) != len(names):
        raise ValueError("unique propagation cases required")


def geometry_ids(scenes):
    radius, angle, _ = scenes
    return [hashlib.sha256(np.concatenate((r, a)).tobytes()).hexdigest() for r, a in zip(radius, angle)]


def case_hardware(config, case):
    selected = copy.deepcopy(config)
    selected["channel"].update({k: case[k] for k in ("model", "include_absorption")})
    system = selected["system"]
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 41]))
    bank = rng.uniform(system["carrier_hz"] - system["tuning_bandwidth_hz"] / 2,
        system["carrier_hz"] + system["tuning_bandwidth_hz"] / 2,
        (max(system["configurations"]), system["antennas"]))
    hardware = [build_hardware(selected, band, bank) for band in system["bandwidths_hz"]]
    return selected, bank, hardware


def prepare_case(config, case):
    selected, bank, hardware = case_hardware(config, case)
    scenes = sample_scenes(selected, 0, config["data"]["train"])
    training = [physical_channels(hw, scenes) for hw in hardware]
    scale = float(torch.stack([h.abs().square().mean() for h in training]).mean().sqrt() / np.sqrt(2))
    if not math.isfinite(scale) or scale <= 0:
        raise ArithmeticError("invalid training-only channel scale")
    cache = {"case": case, "state_bank": torch.tensor(bank), "channel_scale": scale,
        "training": [h / scale for h in training], "geometry_ids": {"train": geometry_ids(scenes)}, "contexts": []}
    sweep = "snrs_db" if "snrs_db" in config["system"] else "transmit_power_dbm"
    for band, hw, h in zip(config["system"]["bandwidths_hz"], hardware, cache["training"]):
        for count, level in itertools.product(config["system"]["configurations"], config["system"][sweep]):
            measurements = count * hw["rf_chains"]
            matrix = hw["matrix"][:, :measurements].contiguous()
            noise = hw["thermal_budget"]["antenna_noise_per_tone_w"]
            received = float(torch.einsum("kjn,bkn->bkj", matrix, h).abs().square().mean())
            if sweep == "snrs_db":
                variance = received / 10 ** (level / 10)
                tone_power = noise / (variance * scale ** 2)
                power = tone_power * config["system"]["tones"]
                power_dbm = float(10 * np.log10(power / 1e-3))
                snr = float(level)
            else:
                power_dbm = level
                power = 10 ** ((power_dbm - 30) / 10)
                tone_power = power / config["system"]["tones"]
                variance = noise / (tone_power * scale ** 2)
                snr = float(10 * np.log10(received / variance))
            cache["contexts"].append({"bandwidth_hz": band, "configurations": count, "pilot_re": count * h.shape[1],
                "rf_chains": hw["rf_chains"], "measurements_per_tone": measurements,
                "transmit_power_dbm": power_dbm, "transmit_power_w": power, "pilot_power_per_tone_w": tone_power,
                "snr_db": snr, "noise_complex_variance": variance,
                "physical_white_noise_per_tone_w": noise, "thermal_budget": hw["thermal_budget"],
                "raw_noise_per_tone_w": hw["noise_gain"][:, :measurements] * noise,
                "training_white_power": received, "matrix": matrix, "matrix_sha256": tensor_hash(matrix)})
    entries, ids = make_entries(config, selected, hardware, cache, "validation")
    cache["validation"], cache["geometry_ids"]["validation"] = entries, ids
    if set(ids) & set(cache["geometry_ids"]["train"]):
        raise AssertionError("training and validation geometries overlap")
    return cache


def make_entries(config, selected, hardware, cache, split):
    system = config["system"]
    index = ("train", "validation", "screen").index(split)
    count = config["data"][split]
    scenes = sample_scenes(selected, index, count)
    entries = []
    for band_index, hw in enumerate(hardware):
        truth = physical_channels(hw, scenes) / cache["channel_scale"]
        seed = config["data"]["screen_seed"] if split == "screen" else config["seed"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, band_index, 811]))
        shape = (count, system["tones"], len(cache["state_bank"]) * hw["rf_chains"])
        standard = torch.tensor((rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2), dtype=truth.dtype)
        for context_index, context in enumerate(cache["contexts"]):
            if context["bandwidth_hz"] != system["bandwidths_hz"][band_index]:
                continue
            clean = torch.einsum("kjn,bkn->bkj", context["matrix"], truth)
            base_noise = standard[:, :, :context["matrix"].shape[1]]
            noise = base_noise * np.sqrt(context["noise_complex_variance"])
            observed = clean + noise
            # Dividing a known pilot changes effective noise units, not physical receiver noise.
            physical_noise = base_noise * np.sqrt(context["physical_white_noise_per_tone_w"])
            physical_observed = (context["pilot_power_per_tone_w"] ** .5 * cache["channel_scale"]) * clean + physical_noise
            reconstructed = (context["pilot_power_per_tone_w"] ** .5 * cache["channel_scale"]) * observed
            relative = float((physical_observed - reconstructed).abs().max() / physical_observed.abs().max().clamp_min(1e-30))
            if relative > 1e-5:
                raise AssertionError("physical pilot-power and normalized observations differ")
            entries.append({"context_index": context_index, "truth": truth, "observation": observed,
                "variance": torch.full((count,), context["noise_complex_variance"]),
                "physical_noise_sha256": tensor_hash(physical_noise), "pilot_equivalence_error": relative,
                "realized_snr_db": 10 * torch.log10(clean.abs().square().sum((1, 2)) / noise.abs().square().sum((1, 2)))})
    return entries, geometry_ids(scenes)


def select_statistics(config, cache, device):
    rows, fitted = [], {}
    for context, entry in zip(cache["contexts"], cache["validation"]):
        band = config["system"]["bandwidths_hz"].index(context["bandwidth_hz"])
        for strength in config["frequency_shrinkage_grid"]:
            stats = lmmse_statistics(cache["training"][band].to(device), context["matrix"].to(device),
                context["noise_complex_variance"], frequency_shrinkage=strength)
            prediction = apply_lmmse(stats, entry["observation"].to(device)).cpu()
            rows.append({"case": cache["case"]["name"], "bandwidth_hz": context["bandwidth_hz"],
                "configurations": context["configurations"], "transmit_power_dbm": context["transmit_power_dbm"],
                "frequency_shrinkage": strength, "mean_nmse_linear": float(nmse(entry["truth"], prediction).mean())})
    choices = {str(band): min(config["frequency_shrinkage_grid"], key=lambda strength: (
        np.mean([r["mean_nmse_linear"] for r in rows if r["bandwidth_hz"] == band and r["frequency_shrinkage"] == strength]),
        -strength)) for band in config["system"]["bandwidths_hz"]}
    for index, context in enumerate(cache["contexts"]):
        band = config["system"]["bandwidths_hz"].index(context["bandwidth_hz"])
        fitted[index] = {}
        for method, strength in (("lmmse_joint", 0.), ("lmmse_per_tone", 1.),
                ("lmmse_shrunk", choices[str(context["bandwidth_hz"])])):
            stats = lmmse_statistics(cache["training"][band].to(device), context["matrix"].to(device),
                context["noise_complex_variance"], frequency_shrinkage=strength)
            fitted[index][method] = {k: v.cpu() for k, v in stats.items()}
    return rows, choices, fitted


def archive(root, run, export, config):
    paths = [root / name for name in ("AGENTS.md", "README.md", "pyproject.toml", ".gitignore")]
    paths += [p for folder in ("src", "scripts", "tests", "configs", "docs") for p in (root / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md")]
    manifest = {p.relative_to(root).as_posix(): file_hash(p) for p in paths}
    with zipfile.ZipFile(run / "source.zip", "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in paths:
            zipped.write(path, path.relative_to(root).as_posix())
    write_json(run / "source_manifest.json", manifest)
    write_json(run / "config_resolved.json", config)
    write_json(run / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
        "device": config["device"], "gpu": torch.cuda.get_device_name() if config["device"].startswith("cuda") else None,
        "source_zip_sha256": file_hash(run / "source.zip"), "command": sys.argv})
    source = Path(config["channel"]["absorption_table_path"])
    if not source.is_absolute():
        source = root / source
    if file_hash(source) != config["channel"]["absorption_table_sha256"]:
        raise ValueError("absorption table does not match its declared provenance")
    shutil.copyfile(source, run / "inputs" / "absorption_table.txt")
    write_json(run / "input_provenance.json", {"absorption_table_sha256": file_hash(source),
        "source_path": str(source.resolve()), "archived_path": "inputs/absorption_table.txt",
        "atmospheric_conditions": "unknown in supplied table; not verified as 296 K / 1 atm"})
    for name in ("source.zip", "source_manifest.json", "config_resolved.json", "environment.json", "input_provenance.json"):
        shutil.copyfile(run / name, export / name)


def score_context(config, cache, context, entry, saved):
    output = []
    for method, prediction in saved["outputs"].items():
        n = len(prediction)
        solver = saved["solver"] if method == "sbl_dense" else {
            "sbl_converged": torch.zeros(n, dtype=torch.bool), "sbl_iterations": torch.zeros(n, dtype=torch.int64),
            "sbl_relative_change": torch.zeros(n), "sbl_optimality_residual": torch.zeros(n),
            "sbl_objective_per_tone": torch.zeros(n), "sbl_solver": "not_applicable", "sbl_early_stopping": False,
            "coarse_ms_per_sample": 0.}
        scored = {**entry, **solver}
        mse = (prediction.to(torch.complex128) - entry["truth"].to(torch.complex128)).abs().square().mean((1, 2))
        for row in metric_rows(config, scored, context, prediction, method, 0., 0):
            output.append({"case": cache["case"]["name"], **row,
                "transmit_power_dbm": context["transmit_power_dbm"],
                "physical_white_noise_per_tone_w": context["physical_white_noise_per_tone_w"],
                "mse_normalized": float(mse[row["scene"]]),
                "mse_physical": float(mse[row["scene"]]) * cache["channel_scale"] ** 2,
                "solver_applicable": method == "sbl_dense"})
    return output


def exports(config, export, rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["case"], row["context"], row["method"]), []).append(row)
    summary = []
    for (case, context, method), values in sorted(groups.items()):
        first = values[0]
        summary.append({"case": case, "context": context, "method": method, "scenes": len(values),
            **{k: first[k] for k in ("bandwidth_hz", "configurations", "transmit_power_dbm", "snr_db")},
            "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in values])),
            "mean_nmse_db": float(10 * np.log10(np.mean([r["nmse_linear"] for r in values]))),
            "mean_mse_normalized": float(np.mean([r["mse_normalized"] for r in values])),
            "mean_mse_physical": float(np.mean([r["mse_physical"] for r in values])),
            "mean_row_nmse_linear": float(np.mean([r["row_nmse_linear"] for r in values])),
            "mean_null_nmse_linear": float(np.mean([r["null_nmse_linear"] for r in values])),
            "sbl_convergence_rate": float(np.mean([r["sbl_converged"] for r in values])) if method == "sbl_dense" else "",
            "maximum_sbl_iterations": max(r["sbl_iterations"] for r in values) if method == "sbl_dense" else ""})
    write_csv(export / "metrics.csv", rows)
    if "snrs_db" in config["system"]:
        observation_exports(config, export, groups, summary)
        return summary, []
    write_csv(export / "summary.csv", summary)
    pooled = []
    for case, power, method in itertools.product([c["name"] for c in config["cases"]],
            config["system"]["transmit_power_dbm"], METHODS):
        values = [r for r in rows if r["case"] == case and r["transmit_power_dbm"] == power and r["method"] == method]
        pooled.append({"case": case, "transmit_power_dbm": power, "method": method,
            "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in values])),
            "mean_nmse_db": float(10 * np.log10(np.mean([r["nmse_linear"] for r in values])))})
    write_csv(export / "pooled_power_summary.csv", pooled)
    effects = []
    rng = np.random.default_rng(config["seed"] + 819)
    cases = {c["name"] for c in config["cases"]}
    for target, reference in (("physical_free_space", "gaussian_free_space"), ("physical_absorption", "physical_free_space")):
        if not {target, reference} <= cases:
            continue
        for power, method in itertools.product(config["system"]["transmit_power_dbm"], METHODS):
            means = []
            for case in (target, reference):
                local = [r for r in rows if r["case"] == case and r["transmit_power_dbm"] == power and r["method"] == method]
                means.append(np.array([np.mean([r["nmse_linear"] for r in local if r["scene"] == g])
                    for g in range(config["data"]["screen"])]))
            indices = rng.integers(len(means[0]), size=(config["bootstrap_samples"], len(means[0])))
            draws = 10 * np.log10(means[0][indices].mean(1) / means[1][indices].mean(1))
            lo, hi = np.quantile(draws, [.025, .975])
            effects.append({"target": target, "reference": reference, "method": method, "transmit_power_dbm": power,
                "delta_db": float(10 * np.log10(means[0].mean() / means[1].mean())),
                "ci95_low_db": float(lo), "ci95_high_db": float(hi), "independent_geometry_blocks": len(means[0]),
                "scope": "paired_geometry_bootstrap; fixed_training_sample"})
    if effects:
        write_csv(export / "paired_effects.csv", effects)
    from matplotlib import pyplot as plt
    labels = {"ls_min_norm": "LS", "lmmse_joint": "Joint LMMSE", "lmmse_per_tone": "Per-tone LMMSE",
        "lmmse_shrunk": "Shrunk LMMSE", "sbl_dense": "Dense SBL"}
    for case in config["cases"]:
        folder = export / case["name"]
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(len(config["system"]["bandwidths_hz"]), len(config["system"]["configurations"]),
            figsize=(11, 8), squeeze=False, sharex=True, sharey=True)
        for i, band in enumerate(config["system"]["bandwidths_hz"]):
            for j, count in enumerate(config["system"]["configurations"]):
                ax = axes[i, j]
                for method in METHODS:
                    local = sorted([r for r in summary if r["case"] == case["name"] and r["method"] == method
                        and r["bandwidth_hz"] == band and r["configurations"] == count], key=lambda r: r["transmit_power_dbm"])
                    ax.plot([r["transmit_power_dbm"] for r in local], [r["mean_nmse_db"] for r in local], "o-", label=labels[method])
                ax.axhline(0, color="gray", ls=":")
                ax.set_title(f"{band/1e9:g} GHz, J={count}")
                ax.set_xlabel("Total transmit pilot power (dBm)")
                ax.set_ylabel("Mean linear NMSE, converted to dB")
                ax.grid(alpha=.25)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3)
        fig.suptitle(case["name"] + "; fixed thermal environment")
        fig.tight_layout(rect=(0, .09, 1, .95))
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"power_nmse.{suffix}", dpi=180)
        plt.close(fig)
    return summary, pooled


def observation_exports(config, export, groups, summary):
    rng = np.random.default_rng(config["seed"] + 819)
    indices = rng.integers(config["data"]["screen"], size=(config["bootstrap_samples"], config["data"]["screen"]))
    for row in summary:
        values = sorted(groups[row["case"], row["context"], row["method"]], key=lambda r: r["scene"])
        for metric in ("nmse_linear", "mse_physical"):
            draws = np.array([r[metric] for r in values])[indices].mean(1)
            if metric == "nmse_linear":
                draws = 10 * np.log10(draws)
            low, high = np.quantile(draws, [.025, .975])
            row[f"{metric}_ci95_low"] = float(low)
            row[f"{metric}_ci95_high"] = float(high)
    write_csv(export / "summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    labels = {"ls_min_norm": "LS", "lmmse_joint": "Joint sample LMMSE",
        "lmmse_per_tone": "Per-tone LMMSE", "lmmse_shrunk": "Validation-selected LMMSE", "sbl_dense": "Dense SBL"}
    levels = config["system"]["snrs_db"]
    for case, band in itertools.product(config["cases"], config["system"]["bandwidths_hz"]):
        folder = export / case["name"]
        folder.mkdir(exist_ok=True)
        for metric, mean in (("nmse_linear", "mean_nmse_db"), ("mse_physical", "mean_mse_physical")):
            fig, axes = plt.subplots((len(levels)+1)//2, 2, figsize=(12, 8), squeeze=False, sharex=True, sharey=True)
            for ax, snr in zip(axes.flat, levels):
                for method in METHODS:
                    local = sorted((r for r in summary if r["case"] == case["name"] and r["bandwidth_hz"] == band
                        and r["snr_db"] == snr and r["method"] == method), key=lambda r: r["configurations"])
                    x, y = [r["configurations"] for r in local], [r[mean] for r in local]
                    line, = ax.plot(x, y, "o-", label=labels[method])
                    ax.fill_between(x, [r[f"{metric}_ci95_low"] for r in local],
                        [r[f"{metric}_ci95_high"] for r in local], color=line.get_color(), alpha=.1)
                    if method == "sbl_dense":
                        bad = [r for r in local if r["sbl_convergence_rate"] < 1]
                        ax.scatter([r["configurations"] for r in bad], [r[mean] for r in bad], marker="x", color="black", zorder=4)
                if metric == "mse_physical":
                    ax.set_yscale("log")
                ax.set_title(f"Training-average received SNR = {snr:g} dB")
                ax.set_xlabel("DMA observations per subcarrier (J)")
                ax.set_ylabel("Mean scene NMSE (dB)" if metric == "nmse_linear" else "Mean physical channel MSE")
                ax.set_xticks(config["system"]["configurations"])
                ax.grid(alpha=.25)
            for ax in list(axes.flat)[len(levels):]:
                ax.set_visible(False)
            fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3)
            fig.suptitle(f"{case['name']}, {band/1e9:g} GHz; shaded: geometry 95% CI; x: SBL budget limited")
            fig.tight_layout(rect=(0, .1, 1, .94))
            for suffix in ("png", "svg"):
                fig.savefig(folder / f"observations_{metric}_{band/1e9:g}ghz.{suffix}", dpi=180)
            plt.close(fig)


def observation_report(config, summary, audit):
    report = ["# DMA观测数与传统信道估计误差", "",
        f"独立几何：训练{config['data']['train']}，验证{config['data']['validation']}，筛选{config['data']['screen']}。",
        "横轴J为每子载波DMA配置数，全部频点共KJ个复观测；不是用于估计协方差的训练几何数。",
        "热噪声与接收机噪声固定；仅用训练集平均接收功率，为每个J和目标SNR校准导频功率。",
        "相同几何、嵌套DMA状态与配对噪声用于全部方法。更多配置增加导频时间及能量，不是固定总导频能量的比较。",
        "训练归一化不随J/SNR变化。收缩系数只在验证集选择，每带宽一个系数、全部J/SNR共用；冻结后生成新筛选几何。",
        "NMSE逐几何除以真值能量，在线性尺度平均后转dB；另保存每系数MSE，并还原到物理信道尺度。MMSE是估计准则，不作为误差指标名称。",
        "阴影为固定训练集与DMA状态下、按独立筛选几何重采样的描述性95%区间，不包含训练集或硬件状态重新抽样的波动。",
        f"SBL未达到一阶阈值：{audit['sbl_nonconverged']}/{audit['sbl_outputs']}；相关点保留，并在图中以x标记。",
        f"NMSE重算最大差{audit['maximum_nmse_difference']:g}；目标接收SNR最大校准误差{audit['maximum_snr_calibration_error_db']:.3g} dB。",
        "导频功率、每子载波噪声、有效导频时长与能量见link_budgets.csv；时长按K/B计算，不包含循环前缀和切换开销。", "",
        "| 模型 | 带宽/GHz | 接收SNR/dB | J | LS | 联合LMMSE | 逐频LMMSE | 验证收缩LMMSE | 密SBL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case, band, snr, count in itertools.product(config["cases"], config["system"]["bandwidths_hz"],
            config["system"]["snrs_db"], config["system"]["configurations"]):
        values = {r["method"]: r["mean_nmse_db"] for r in summary if r["case"] == case["name"]
            and r["bandwidth_hz"] == band and r["snr_db"] == snr and r["configurations"] == count}
        report.append(f"| {case['name']} | {band/1e9:g} | {snr:g} | {count} | "
            + " | ".join(f"{values[m]:.3f}" for m in METHODS) + " |")
    report += ["", "这是单组训练几何、单组DMA调谐状态的探索性比较。联合版本使用样本统计量，不是已知总体协方差的理想LMMSE。",
        "物理传播及噪声假设沿用docs/PHYSICAL_MODEL.md，6 dB噪声系数和材料折射率均为仿真假设；吸收表大气条件未知。"]
    return report


def run(root, config_path, run_id, device_override=None, bandwidth_hz=None):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("safe independent run ID required")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if bandwidth_hz is not None:
        if bandwidth_hz not in config["system"]["bandwidths_hz"]:
            raise ValueError("requested bandwidth is absent from the configuration")
        config["system"]["bandwidths_hz"] = [bandwidth_hz]
    if device_override:
        config["device"] = device_override
    validate(config)
    root = Path(root)
    path, export = root / "runs" / run_id, root / "results" / run_id
    if path.exists():
        if json.loads((path / "config_resolved.json").read_text()) != config:
            raise ValueError("configuration changed; use a new run ID")
        archived = json.loads((path / "source_manifest.json").read_text())
        current = {p.relative_to(root).as_posix(): file_hash(p) for folder in ("src", "scripts")
            for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts}
        if current != {k: v for k, v in archived.items() if k.startswith(("src/", "scripts/")) and k.endswith(".py")}:
            raise ValueError("execution source changed; use a new run ID")
    else:
        path.mkdir(parents=True)
        export.mkdir(parents=True, exist_ok=False)
        for folder in ("cache", "inputs", "predictions"):
            (path / folder).mkdir()
        archive(root, path, export, config)
    def log(event, **fields):
        record = {"event": event, "at_unix": time.time(), **fields}
        with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    def timeout():
        write_json(path / "status.json", {"status": "timeout", "limit_seconds": config["timeout_seconds"]})
        import os
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
        caches, fitted, choices, validation_rows = {}, {}, {}, []
        frozen_path = path / "frozen.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            for name, digest in frozen["cache_sha256"].items():
                if file_hash(path / "cache" / name) != digest:
                    raise ValueError("frozen training/statistics cache changed")
            fitted = torch.load(path / "cache/statistics.pt", weights_only=True)
            choices = frozen["selected_frequency_shrinkage"]
        for case in config["cases"]:
            name = case["name"]
            cache_path = path / "cache" / f"{name}.pt"
            if cache_path.exists():
                cache = torch.load(cache_path, weights_only=True)
            else:
                cache = prepare_case(config, case)
                save_tensor(cache_path, cache)
            caches[name] = cache
            if not frozen_path.exists():
                scores, choice, statistics = select_statistics(config, cache, device)
                validation_rows += scores
                choices[name], fitted[name] = choice, statistics
            log("validation_ready", case=name, selected_shrinkage=choices[name])
        identities = [c["geometry_ids"] for c in caches.values()]
        if any(ids != identities[0] for ids in identities[1:]):
            raise AssertionError("propagation cases must share the same geometries")
        if not frozen_path.exists():
            save_tensor(path / "cache/statistics.pt", fitted)
            write_csv(export / "validation_scores.csv", validation_rows)
            frozen = {"selected_frequency_shrinkage": choices, "frozen_at_unix": time.time(),
                "selection_split": "validation", "screening_used": False,
                "cache_sha256": {p.name: file_hash(p) for p in (path / "cache").glob("*.pt")},
                "source_archive_sha256": file_hash(path / "source.zip")}
            write_json(frozen_path, frozen)
        shutil.copyfile(frozen_path, export / "frozen.json")
        if file_hash(path / "source.zip") != frozen["source_archive_sha256"]:
            raise ValueError("frozen source archive changed")
        log("frozen", sha256=file_hash(frozen_path))
        rows, checks, maximum_difference = [], 0, 0.
        screen_ids = None
        maximum_equivalence = 0.
        physical_noise_hashes = {}
        for case in config["cases"]:
            name, cache = case["name"], caches[case["name"]]
            selected, _, hardware = case_hardware(config, case)
            folder = path / "predictions" / name
            folder.mkdir(exist_ok=True)
            observed_path = folder / "observations.pt"
            if observed_path.exists():
                observations = torch.load(observed_path, weights_only=True)
                if observations["frozen_sha256"] != file_hash(frozen_path):
                    raise ValueError("screen observations belong to a different freeze")
            else:
                entries, ids = make_entries(config, selected, hardware, cache, "screen")
                observations = {"entries": entries, "geometry_ids": ids, "created_at_unix": time.time(),
                    "frozen_sha256": file_hash(frozen_path)}
                save_tensor(observed_path, observations)
            ids = observations["geometry_ids"]
            if (set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"])
                    or len(set(ids)) != config["data"]["screen"] or observations["created_at_unix"] < frozen["frozen_at_unix"]):
                raise AssertionError("fresh independent screen geometries must be generated after freeze")
            if screen_ids is not None and ids != screen_ids:
                raise AssertionError("screen geometries must remain paired across propagation cases")
            screen_ids = ids
            noise_hashes = {}
            if len(observations["entries"]) != len(cache["contexts"]):
                raise AssertionError("one observation entry per physical condition required")
            for context, entry in zip(cache["contexts"], observations["entries"]):
                index = entry["context_index"]
                key = context["bandwidth_hz"], context["configurations"]
                if noise_hashes.setdefault(key, entry["physical_noise_sha256"]) != entry["physical_noise_sha256"]:
                    raise AssertionError("physical receiver noise changed across transmit powers")
                if physical_noise_hashes.setdefault(key, entry["physical_noise_sha256"]) != entry["physical_noise_sha256"]:
                    raise AssertionError("propagation cases must share the same physical receiver noise")
                maximum_equivalence = max(maximum_equivalence, entry["pilot_equivalence_error"])
                prediction_path = folder / f"context_{index}.pt"
                observation_hash = tensor_hash(entry["observation"])
                if prediction_path.exists():
                    saved = torch.load(prediction_path, weights_only=True)
                    if saved["observation_sha256"] != observation_hash or saved["frozen_sha256"] != file_hash(frozen_path):
                        raise ValueError("cached estimates do not match observations or frozen statistics")
                else:
                    a = context["matrix"].to(device)
                    y = entry["observation"].to(device)
                    ls = torch.einsum("knj,bkj->bkn", torch.linalg.pinv(a.to(torch.complex128)), y.to(torch.complex128)).to(y.dtype).cpu()
                    outputs = {"ls_min_norm": ls}
                    outputs.update({method: apply_lmmse(stats, y).cpu() for method, stats in fitted[name][index].items()})
                    band = config["system"]["bandwidths_hz"].index(context["bandwidth_hz"])
                    estimated = estimate_entry(entry, context, hardware[band]["dictionary"].to(device), config["sbl"], device, log)
                    outputs["sbl_dense"] = estimated["coarse"]
                    saved = {"outputs": outputs, "solver": {k: v for k, v in estimated.items() if k.startswith("sbl_") or k == "coarse_ms_per_sample"},
                        "observation_sha256": observation_hash, "frozen_sha256": file_hash(frozen_path)}
                    save_tensor(prediction_path, saved)
                scored = score_context(config, cache, context, entry, saved)
                for method in METHODS:
                    recomputed = nmse(entry["truth"], saved["outputs"][method]).tolist()
                    recorded = [r["nmse_linear"] for r in scored if r["method"] == method]
                    maximum_difference = max(maximum_difference, max(abs(a-b) for a, b in zip(recomputed, recorded)))
                rows += scored
                checks += len(entry["truth"])
                log("screen_context", case=name, context=index, transmit_power_dbm=context["transmit_power_dbm"],
                    training_received_snr_db=context["snr_db"], converged=int(saved["solver"]["sbl_converged"].sum()))
        expected = (len(config["cases"]) * len(config["system"]["bandwidths_hz"])
            * len(config["system"]["configurations"]) * len(config["system"].get("snrs_db", config["system"].get("transmit_power_dbm")))
            * config["data"]["screen"] * len(METHODS))
        if len(rows) != expected:
            raise AssertionError("incomplete per-scene physical comparison")
        summary, pooled = exports(config, export, rows)
        sbl_rows = [r for r in rows if r["solver_applicable"]]
        audit = {"status": "passed", "metrics_recomputed": len(rows), "maximum_nmse_difference": maximum_difference,
            "common_observation_checks": checks, "independent_screen_geometries": len(screen_ids),
            "physical_noise_fixed_across_power": True, "paired_geometries_across_cases": True,
            "maximum_pilot_normalization_equivalence_error": maximum_equivalence,
            "sbl_outputs": len(sbl_rows), "sbl_nonconverged": sum(not r["sbl_converged"] for r in sbl_rows),
            "sbl_solver": SOLVER_VERSION, "elapsed_seconds": time.time()-started}
        assert maximum_difference == 0.
        for name, digest in frozen["cache_sha256"].items():
            assert file_hash(path / "cache" / name) == digest
        write_json(export / "audit.json", audit)
        budgets = [{"case": name, "bandwidth_hz": c["bandwidth_hz"], "configurations": c["configurations"],
            "transmit_power_dbm": c["transmit_power_dbm"], "training_received_snr_db": c["snr_db"],
            "pilot_power_per_tone_w": c["pilot_power_per_tone_w"],
            "physical_white_noise_per_tone_w": c["physical_white_noise_per_tone_w"],
            "effective_normalized_noise_variance": c["noise_complex_variance"],
            "useful_pilot_duration_s": c["configurations"] * config["system"]["tones"] / c["bandwidth_hz"],
            "useful_pilot_energy_j": c["transmit_power_w"] * c["configurations"] * config["system"]["tones"] / c["bandwidth_hz"],
            "channel_scale": cache["channel_scale"], **c["thermal_budget"]} for name, cache in caches.items() for c in cache["contexts"]]
        write_csv(export / "link_budgets.csv", budgets)
        if "snrs_db" in config["system"]:
            calibration_error = max(abs(10 * np.log10(c["training_white_power"] / c["noise_complex_variance"]) - c["snr_db"])
                for cache in caches.values() for c in cache["contexts"])
            if calibration_error > 1e-10:
                raise AssertionError("training-average received SNR differs from its target")
            audit["maximum_snr_calibration_error_db"] = calibration_error
            write_json(export / "audit.json", audit)
            report = observation_report(config, summary, audit)
        else:
            report = power_report(config, pooled, rows, sbl_rows, audit, maximum_difference)
        (export / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        write_json(path / "status.json", {**audit, "status": "complete"})
        write_json(export / "status.json", {**audit, "status": "complete"})
        log("complete", **audit)
        return export
    except Exception as error:
        write_json(path / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        timer.cancel()


def power_report(config, pooled, rows, sbl_rows, audit, maximum_difference):
    report = ["# 固定热噪声与物理路径增益：配对探索实验", "",
        f"训练/验证/筛选独立几何：{config['data']['train']}/{config['data']['validation']}/{config['data']['screen']}。",
        "全部带宽、配置、功率和方法按相同几何配对；筛选在统计量冻结后生成。",
        "各子载波噪声为 k_B T_ant B/K；接收机附加噪声经DMA输出协方差白化处理。",
        "固定温度及噪声系数，只改变总发射导频功率；算法输入除以已知导频和训练全局尺度后，有效噪声方差相应改变。",
        "每带宽单独选择一个收缩系数，目标为验证集所有配置和功率的线性NMSE均值。", "",
        "| 场景模型 | 发射功率/dBm | LS | 联合LMMSE | 逐频LMMSE | 收缩LMMSE | 密SBL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for case, power in itertools.product([c["name"] for c in config["cases"]], config["system"]["transmit_power_dbm"]):
        values = {r["method"]: r["mean_nmse_db"] for r in pooled if r["case"] == case and r["transmit_power_dbm"] == power}
        report.append(f"| {case} | {power:g} | " + " | ".join(f"{values[m]:.3f}" for m in METHODS) + " |")
    report += ["", "各数值先在四种硬件条件与几何上平均线性NMSE，再转dB。固定发射功率下，各带宽和模型的接收SNR不同；见link_budgets.csv。",
        f"核验：{len(rows)}条指标重算，最大差{maximum_difference:g}；{len(sbl_rows)}个SBL输出中{audit['sbl_nonconverged']}个未达到一阶阈值，失败项保留。",
        "paired_effects.csv按几何块配对重采样，区间仅覆盖固定训练数据下的筛选几何变化。",
        "", "## 范围与假设", "", "本轮是32×32 DMA机制对照，不复现论文144阵元AoSA及18米场景。",
        "Gaussian对照沿用原随机路径增益，三组均使用相同15 dBi接收天线增益。物理组使用一个LoS及两个单次反射路径；不额外乘Rayleigh随机增益。",
        "反射率使用论文式(18)--(21)，常数折射率1.5为显式材料假设，粗糙度8.8e-5 m来自论文仿真设置。",
        "吸收表温度、压力、湿度及HITRAN版本元数据未知；本轮仅按原MATLAB功率衰减系数口径恢复，不能声称复原论文296 K/1 atm吸收谱。",
        "天线噪声温度已包含热环境，不额外叠加一次大气发射噪声。6 dB接收机噪声系数为仿真假设。",
        "SBL字典采用接收端角距球面波；未知发射端至反射点距离不提供给估计器，其公共相位和损耗由频点系数吸收，阵元间幅度仍是字典近似。",
        "原运行保持不变。配置、来源哈希、源码、完整缓存及预测在独立runs目录归档。"]
    return report
