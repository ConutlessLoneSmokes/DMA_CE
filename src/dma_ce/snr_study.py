"""Frozen-model SNR comparison, with a clean-channel DDPM prior and learned Gaussian updates."""
import copy
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import traceback

import numpy as np
import torch

from ._upstream.ddpm import DenoisingCNN, VPSchedule, noise_loss
from .data import build_hardware, encode, estimate_entry, physical_channels, sample_scenes, sync, tensor_hash
from .experiment import file_hash, metric_rows, nmse, write_csv, write_json
from .research import infer, learning_rate, load_model, save_tensor
from .sbl import SOLVER_VERSION
from .structured import VarianceUpdate, apply_lmmse, lmmse_statistics, structured_recover
from .suite import Study


LABELS = {
    "ls_min_norm": "Minimum-norm LS", "lmmse_joint": "Joint-frequency LMMSE",
    "sbl_dense": "SBL (dense dictionary)", "sbl_coarse_100": "SBL (100 updates)",
    "ddpm_latest_avg1": "Latest conditional DDPM (1 draw)",
    "ddpm_latest_avg4": "Latest conditional DDPM (4 draws)",
    "ddpm_latest_avg16": "Latest conditional DDPM (16 draws)",
    "deterministic_latest": "Latest deterministic refinement",
    "prior_fixed": "DDPM prior + fixed Gaussian updates",
    "prior_learned": "DDPM prior + learned variance updates"}


def validate_comparison(spec):
    integers = ("training_scenes", "validation_scenes", "screen_scenes", "batch_size", "solver_batch_size",
        "prior_steps", "update_steps", "update_width", "structured_iterations", "sbl_max_iterations",
        "timeout_seconds", "bootstrap_samples", "log_every")
    if (any(type(spec[k]) is not int or spec[k] < 1 for k in integers)
            or min(spec[k] for k in ("training_scenes", "validation_scenes", "screen_scenes", "bootstrap_samples")) < 2
            or type(spec["screen_seed"]) is not int or spec["screen_seed"] < 0
            or type(spec.get("implementation_only", False)) is not bool
            or not math.isfinite(spec["sbl_tolerance"]) or spec["sbl_tolerance"] <= 0):
        raise ValueError("positive finite SNR-study budgets and independent split sizes required")
    for key, maximum in (("prior_checkpoints", spec["prior_steps"]), ("update_checkpoints", spec["update_steps"])):
        values = spec[key]
        if (not values or values != sorted(set(values)) or values[-1] != maximum
                or any(type(v) is not int or v < 1 for v in values)):
            raise ValueError("checkpoints must be sorted positive integers ending at the training budget")
    if (not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"])
            or any(type(v) is not int or v < 0 for v in spec["seeds"])
            or not spec["snrs_db"] or spec["snrs_db"] != sorted(set(spec["snrs_db"]))
            or not all(math.isfinite(v) for v in spec["snrs_db"])
            or not spec["strength_grid"] or not all(math.isfinite(v) and v > 0 for v in spec["strength_grid"])):
        raise ValueError("unique seeds, ordered finite SNRs and positive validation strengths required")


def sweep_cache(config, parent, split, count, snrs):
    """Reuse training hardware/scales; one standard-noise draw per geometry is shared across SNRs."""
    system = config["system"]
    by_band = {c["bandwidth_hz"]: i for i, c in enumerate(parent["contexts"])}
    by_hardware = {(c["bandwidth_hz"], c["configurations"]): c for c in parent["contexts"]}
    result = {k: parent[k] for k in ("channel_scale", "state_bank", "residual_real_scale",
        "reliability_scale", "reliability_log_mean", "reliability_log_std", "dc_reference_complex_variance", "calibration_edges")}
    result.update(contexts=[], splits={split: []}, geometry_hashes={})
    split_index = ("train", "validation", "screen").index(split)
    if split == "screen":
        scenes = sample_scenes(config, split_index, count)
        packed = np.concatenate((scenes[0], scenes[1], scenes[2].real, scenes[2].imag), 1)
        import hashlib
        result["geometry_hashes"][split] = hashlib.sha256(packed.tobytes()).hexdigest()
    else:
        if count > len(parent["splits"][split][0]["truth"]):
            raise ValueError("the comparison cannot invent additional parent training/validation channels")
        result["geometry_hashes"][split] = parent["geometry_hashes"][split]
    for band_index, bandwidth in enumerate(system["bandwidths_hz"]):
        if split == "screen":
            hw = build_hardware(config, bandwidth, parent["state_bank"].numpy())
            truth = physical_channels(hw, scenes) / parent["channel_scale"]
            noise_count = count
        else:
            original = parent["splits"][split][by_band[bandwidth]]["truth"]
            truth, noise_count = original[:count], len(original)
        seed = config["data"]["screen_seed"] if split == "screen" else config["seed"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, band_index, 811]))
        shape = (noise_count, system["tones"], len(parent["state_bank"]) * system.get("rf_chains", 1))
        standard = torch.tensor((rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2), dtype=truth.dtype)[:count]
        for configurations, snr in itertools.product(system["configurations"], snrs):
            known = by_hardware[bandwidth, configurations]
            variance = known["training_white_power"] / 10 ** (snr / 10)
            context = {**known, "snr_db": snr, "noise_complex_variance": variance,
                "physical_element_noise_variance": variance * parent["channel_scale"] ** 2}
            clean = torch.einsum("kjn,bkn->bkj", known["matrix"], truth)
            noise = standard[:, :, :known["matrix"].shape[1]] * np.sqrt(variance)
            entry = {"truth": truth, "observation": clean + noise,
                "variance": torch.full((count,), variance), "context_index": len(result["contexts"]),
                "realized_snr_db": 10 * torch.log10(clean.abs().square().sum((1, 2)) / noise.abs().square().sum((1, 2)))}
            result["contexts"].append(context)
            result["splits"][split].append(entry)
    return result


def train_network(model, folder, settings, seed, kind, loss_function, validation_function, device, log):
    """One optimizer/RNG trajectory per model, with validation-only checkpoint choice."""
    folder.mkdir(parents=True, exist_ok=True)
    steps, points = settings[f"{kind}_steps"], settings[f"{kind}_checkpoints"]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    batch_rng = torch.Generator().manual_seed(seed + 1013)
    noise_rng = torch.Generator(device=device).manual_seed(seed + 2017)
    first = 0
    if (folder / "last.pt").exists():
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        first = saved["step"]
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        batch_rng.set_state(saved["batch_rng"])
        noise_rng.set_state(saved["noise_rng"])
    records = json.loads((folder / "validation.json").read_text()) if (folder / "validation.json").exists() else []
    losses = []
    for step in range(first, steps):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(.001, step, steps)
        loss = loss_function(batch_rng, noise_rng)
        if not torch.isfinite(loss):
            raise ArithmeticError(f"nonfinite {kind} training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        current = step + 1
        if current % settings["log_every"] == 0 or current in points:
            log("training", kind=kind, seed=seed, step=current, loss=float(np.mean(losses)))
            losses = []
        if current in points:
            model.eval()
            with torch.no_grad():
                score = float(validation_function())
            if not math.isfinite(score):
                raise ArithmeticError("nonfinite validation score")
            state = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": current, "seed": seed, "kind": kind,
                "parameters": sum(p.numel() for p in model.parameters())}
            save_tensor(folder / f"step_{current}.pt", state)
            records = [r for r in records if r["step"] < current] + [{"step": current, "score": score}]
            temporary = folder / "validation.tmp"
            write_json(temporary, records)
            temporary.replace(folder / "validation.json")
            # The optimizer snapshot commits its already-persisted checkpoint and validation score.
            save_tensor(folder / "last.pt", {**state, "optimizer": optimizer.state_dict(),
                "batch_rng": batch_rng.get_state(), "noise_rng": noise_rng.get_state()})
            log("validation", kind=kind, seed=seed, step=current, score=score)
    chosen = min(records, key=lambda r: (r["score"], r["step"]))
    path = folder / f"step_{chosen['step']}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    return path, records, chosen


def fit_models(study, parent, training, validation, statistics):
    settings, device = study.config["comparison"], study.device
    schedule = VPSchedule(study.config["learning"]["diffusion_steps"]).to(device)
    bands = study.config["system"]["bandwidths_hz"]
    unique = {c["bandwidth_hz"]: e["truth"] for c, e in zip(training["contexts"], training["splits"]["train"])}
    clean = encode(torch.cat([unique[b] for b in bands])).to(device)
    unique_val = {c["bandwidth_hz"]: e["truth"] for c, e in zip(validation["contexts"], validation["splits"]["validation"])}
    clean_val = encode(torch.cat([unique_val[b] for b in bands])).to(device)
    training_cases = []
    for context, entry in zip(training["contexts"], training["splits"]["train"]):
        key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
        stats = statistics[key]
        y = entry["observation"].to(device)
        training_cases.append((context["matrix"].to(device), y, entry["variance"].to(device),
            apply_lmmse(stats, y), stats["diagonal"].to(device=device, dtype=y.real.dtype).clamp_min(1e-8)[None].expand(len(y), -1, -1),
            entry["truth"].to(device)))
    validation_cases = []
    for context, entry in zip(validation["contexts"], validation["splits"]["validation"]):
        key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
        y = entry["observation"].to(device)
        stats = statistics[key]
        validation_cases.append((context["matrix"].to(device), y, entry["variance"].to(device),
            apply_lmmse(stats, y), stats["diagonal"].to(device=device, dtype=y.real.dtype).clamp_min(1e-8)[None].expand(len(y), -1, -1),
            entry["truth"].to(device)))
    models = {}
    for seed in settings["seeds"]:
        torch.manual_seed(seed)
        prior = DenoisingCNN(width=study.config["learning"]["width"]).to(device)
        def prior_loss(batch_rng, noise_rng):
            indices = torch.randint(len(clean), (settings["batch_size"],), generator=batch_rng)
            return noise_loss(prior, schedule, clean[indices.to(device)], generator=noise_rng)
        def prior_validation():
            rng = torch.Generator(device=device).manual_seed(study.config["seed"] + 451)
            return noise_loss(prior, schedule, clean_val, generator=rng)
        prior_path, prior_records, prior_choice = train_network(prior, study.path / "models" / f"prior_{seed}",
            settings, seed, "prior", prior_loss, prior_validation, device, study.log)
        prior.eval().requires_grad_(False)
        def score(updater, strength):
            values = []
            for a, y, variance, initial, diagonal, truth in validation_cases:
                for start in range(0, len(y), settings["batch_size"]):
                    sl = slice(start, start + settings["batch_size"])
                    output = structured_recover(prior, schedule, a, y[sl], variance[sl], initial[sl], diagonal[sl],
                        updater=updater, iterations=settings["structured_iterations"], strength=strength)[0]
                    values.extend(nmse(truth[sl], output).tolist())
            return float(np.mean(values))
        with torch.no_grad():
            fixed_scores = {str(s): score(None, s) for s in settings["strength_grid"]}
        strength = min(settings["strength_grid"], key=lambda s: (fixed_scores[str(s)], s))
        torch.manual_seed(seed + 37)
        updater = VarianceUpdate(settings["update_width"]).to(device)
        def update_loss(batch_rng, noise_rng):
            number = int(torch.randint(len(training_cases), (1,), generator=batch_rng))
            a, y, variance, initial, diagonal, truth = training_cases[number]
            indices = torch.randint(len(y), (settings["batch_size"],), generator=batch_rng).to(device)
            output = structured_recover(prior, schedule, a, y[indices], variance[indices], initial[indices], diagonal[indices],
                updater=updater, iterations=settings["structured_iterations"], strength=strength)[0]
            return nmse(truth[indices], output).mean()
        update_path, update_records, update_choice = train_network(updater, study.path / "models" / f"update_{seed}",
            settings, seed, "update", update_loss, lambda: score(updater, strength), device, study.log)
        models[str(seed)] = {"prior_path": prior_path.relative_to(study.path).as_posix(), "prior_sha256": file_hash(prior_path),
            "update_path": update_path.relative_to(study.path).as_posix(), "update_sha256": file_hash(update_path),
            "prior_choice": prior_choice, "prior_validation": prior_records,
            "update_choice": update_choice, "update_validation": update_records,
            "strength": strength, "fixed_validation": fixed_scores,
            "prior_parameters": sum(p.numel() for p in prior.parameters()),
            "update_parameters": sum(p.numel() for p in updater.parameters())}
        study.log("models_ready", seed=seed, prior_step=prior_choice["step"], update_step=update_choice["step"], strength=strength)
        del prior, updater
    return models


def result_rows(config, entry, context, output, method, seed, milliseconds, calls, *, solver=None, variance_update_calls=0):
    count = len(output)
    metadata = {"sbl_converged": torch.zeros(count, dtype=torch.bool), "sbl_iterations": torch.zeros(count, dtype=torch.int64),
        "sbl_relative_change": torch.zeros(count), "coarse_ms_per_sample": 0., **(solver or {})}
    rows = list(metric_rows(config, {**entry, **metadata}, context, output, method, milliseconds, calls))
    for row in rows:
        for key in ("sbl_optimality_residual", "sbl_objective_per_tone", "sbl_solver", "sbl_early_stopping"):
            row.setdefault(key, None)
        row.update(seed=seed, solver_applicable=solver is not None,
            total_batch_ms_per_sample=metadata["coarse_ms_per_sample"] + milliseconds,
            denoiser_calls=calls - variance_update_calls, variance_update_calls=variance_update_calls)
    return rows


def timed(function, device, count):
    sync(device)
    started = time.perf_counter()
    output = function()
    sync(device)
    return output, 1000 * (time.perf_counter() - started) / count


def verify_frozen_models(path, frozen):
    records = [(item["path"], item["sha256"]) for item in frozen["latest_models"].values()]
    records += [(item[kind + "_path"], item[kind + "_sha256"])
        for item in frozen["models"].values() for kind in ("prior", "update")]
    for name, expected in records:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "models":
            raise ValueError("invalid frozen model path")
        if file_hash(path / relative) != expected:
            raise ValueError("frozen model hash changed")


@torch.no_grad()
def evaluate(study, parent, frozen, statistics):
    config, settings, device = study.config, study.config["comparison"], study.device
    verify_frozen_models(study.path, frozen)
    observed_path = study.path / "predictions" / "observations.pt"
    if observed_path.exists():
        observations = torch.load(observed_path, map_location="cpu", weights_only=True)
    else:
        observations = sweep_cache(config, parent, "screen", settings["screen_scenes"], settings["snrs_db"])
        save_tensor(observed_path, observations)
    schedule = VPSchedule(config["learning"]["diffusion_steps"]).to(device)
    rows, files = [], {}
    for context, entry in zip(observations["contexts"], observations["splits"]["screen"]):
        c = entry["context_index"]
        destination = study.path / "predictions" / f"context_{c}.pt"
        table = study.path / "predictions" / f"context_{c}.csv"
        if destination.exists() and table.exists():
            with table.open(encoding="utf-8", newline="") as stream:
                cached_rows = list(csv.DictReader(stream))
            # Tables are reloaded at the end with explicit numeric types.
            files[destination.name] = file_hash(destination)
            study.log("context_reused", context=c, rows=len(cached_rows))
            continue
        outputs = {}
        a, y, variance = context["matrix"].to(device), entry["observation"].to(device), entry["variance"].to(device)
        count = len(y)
        pinv, setup_ls = timed(lambda: torch.linalg.pinv(a.to(torch.complex128)), device, 1)
        torch.einsum("knj,bkj->bkn", pinv, y[:2].to(torch.complex128))
        ls, ls_ms = timed(lambda: torch.einsum("knj,bkj->bkn", pinv, y.to(torch.complex128)).to(y.dtype), device, count)
        ls = ls.cpu()
        values = result_rows(config, entry, context, ls, "ls_min_norm", -1, ls_ms, 0)
        outputs["ls_min_norm/-1"] = ls
        stats = statistics[context["bandwidth_hz"], context["configurations"], context["snr_db"]]
        apply_lmmse(stats, y[:2])
        initial, lmmse_ms = timed(lambda: apply_lmmse(stats, y), device, count)
        diagonal = stats["diagonal"].to(device=device, dtype=y.real.dtype).clamp_min(1e-8)[None].expand(count, -1, -1)
        outputs["lmmse_joint/-1"] = initial.cpu()
        values += result_rows(config, entry, context, initial.cpu(), "lmmse_joint", -1, lmmse_ms, 0)
        for label, grid, budget, stopping in (("sbl_coarse_100", frozen["coarse_grid"], frozen["coarse_iterations"], False),
                ("sbl_dense", frozen["dense_grid"], settings["sbl_max_iterations"], True)):
            candidate = copy.deepcopy(config)
            candidate["sbl"].update(grid)
            dictionary = build_hardware(candidate, context["bandwidth_hz"], parent["state_bank"].numpy())["dictionary"].to(device)
            solver = {**candidate["sbl"], "iterations": budget, "early_stopping": stopping,
                "tolerance": settings["sbl_tolerance"], "batch_size": settings["solver_batch_size"]}
            estimated = estimate_entry(entry, context, dictionary, solver, device, study.log)
            outputs[label + "/-1"] = estimated["coarse"]
            values += result_rows(config, entry, context, estimated["coarse"], label, -1, 0., 0, solver=estimated)
            if label == "sbl_coarse_100":
                coarse_entry = estimated
            study.log("solver_complete", context=c, method=label, convergence=float(estimated["sbl_converged"].float().mean()),
                max_iterations=int(estimated["sbl_iterations"].max()))
        for seed in settings["seeds"]:
            for kind in ("diffusion", "deterministic"):
                item = frozen["latest_models"][f"{kind}_none_{seed}"]
                path = study.path / item["path"]
                if file_hash(path) != item["sha256"]:
                    raise ValueError("inherited model hash changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                model = load_model(saved, device)
                averages = (1, 4, 16) if kind == "diffusion" else (1,)
                predictions = infer(config, parent, model, saved["recipe"], coarse_entry, device,
                    seed=config["seed"] + 17000 + c * 1000, averages=averages)
                for average, (output, milliseconds, calls) in predictions.items():
                    method = f"ddpm_latest_avg{average}" if kind == "diffusion" else "deterministic_latest"
                    outputs[f"{method}/{seed}"] = output
                    values += result_rows(config, coarse_entry, context, output, method, seed, milliseconds, calls, solver=coarse_entry)
                del model
            item = frozen["models"][str(seed)]
            for kind in ("prior", "update"):
                if file_hash(study.path / item[kind + "_path"]) != item[kind + "_sha256"]:
                    raise ValueError("new frozen model hash changed")
            prior = DenoisingCNN(width=config["learning"]["width"]).to(device)
            prior.load_state_dict(torch.load(study.path / item["prior_path"], map_location="cpu", weights_only=True)["state_dict"])
            prior.eval().requires_grad_(False)
            updater = VarianceUpdate(settings["update_width"]).to(device)
            updater.load_state_dict(torch.load(study.path / item["update_path"], map_location="cpu", weights_only=True)["state_dict"])
            updater.eval()
            for method, network in (("prior_fixed", None), ("prior_learned", updater)):
                estimates = []
                # The same warm-up and batching apply to the fixed and learned updates.
                size = min(count, settings["batch_size"])
                structured_recover(prior, schedule, a, y[:size], variance[:size], initial[:size], diagonal[:size],
                    updater=network, iterations=settings["structured_iterations"], strength=item["strength"])
                def recover_all():
                    for start in range(0, count, settings["batch_size"]):
                        sl = slice(start, start + settings["batch_size"])
                        estimates.append(structured_recover(prior, schedule, a, y[sl], variance[sl], initial[sl], diagonal[sl],
                            updater=network, iterations=settings["structured_iterations"], strength=item["strength"])[0].cpu())
                    return torch.cat(estimates)
                output, milliseconds = timed(recover_all, device, count)
                outputs[f"{method}/{seed}"] = output
                update_calls = settings["structured_iterations"] if network is not None else 0
                values += result_rows(config, entry, context, output, method, seed, milliseconds + lmmse_ms,
                    settings["structured_iterations"] + update_calls, variance_update_calls=update_calls)
            del prior, updater
        save_tensor(destination, outputs)
        write_csv(table, values)
        files[destination.name] = file_hash(destination)
        study.log("screen_context_complete", context=c, rows=len(values), snr_db=context["snr_db"],
            bandwidth_hz=context["bandwidth_hz"], configurations=context["configurations"])
    for table in sorted((study.path / "predictions").glob("context_*.csv")):
        with table.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                for key in ("scene", "context", "seed", "configurations", "pilot_re", "network_calls", "sbl_iterations",
                        "denoiser_calls", "variance_update_calls"):
                    row[key] = int(row[key])
                for key in ("bandwidth_hz", "snr_db", "realized_snr_db", "nmse_linear", "nmse_db", "row_nmse_linear",
                        "null_nmse_linear", "observation_residual", "total_batch_ms_per_sample"):
                    row[key] = float(row[key])
                rows.append(row)
    write_csv(study.export / "metrics.csv", rows)
    write_json(study.export / "evaluation_provenance.json", {"observation_sha256": file_hash(observed_path),
        "geometry_hashes": observations["geometry_hashes"], "prediction_sha256": files,
        "frozen_sha256": file_hash(study.path / "stages" / "frozen.json"), "scored_at_unix": time.time(),
        "ls_pseudoinverse_setup": "known-matrix preprocessing excluded from amortized application time",
        "lmmse_setup": "training-only joint statistics and known-matrix preprocessing excluded from amortized application time"})
    return rows


def summarize_results(rows, settings, export):
    groups = {}
    hashes = {}
    for row in rows:
        key = row["context"], row["scene"]
        if hashes.setdefault(key, row["observation_sha256"]) != row["observation_sha256"]:
            raise AssertionError("methods did not receive the same observation")
        groups.setdefault((row["context"], row["method"]), []).append(row)
    rng = np.random.default_rng(settings["screen_seed"] + 51)
    resamples = rng.integers(settings["screen_scenes"], size=(settings["bootstrap_samples"], settings["screen_scenes"]))
    summary, geometry_values = [], {}
    for (context, method), values in sorted(groups.items()):
        per_geometry = {}
        for row in values:
            per_geometry.setdefault(row["scene"], []).append(row["nmse_linear"])
        if sorted(per_geometry) != list(range(settings["screen_scenes"])):
            raise AssertionError("incomplete independent geometry blocks")
        expected = 1 if values[0]["seed"] == -1 else len(settings["seeds"])
        if any(len(v) != expected for v in per_geometry.values()):
            raise AssertionError("missing/duplicated training initialization")
        array = np.array([np.mean(per_geometry[g]) for g in sorted(per_geometry)])
        geometry_values[context, method] = array
        intervals = np.percentile(10 * np.log10(array[resamples].mean(1)), [2.5, 97.5])
        first = values[0]
        summary.append({"context": context, "method": method, "bandwidth_hz": first["bandwidth_hz"],
            "configurations": first["configurations"], "snr_db": first["snr_db"],
            "independent_geometries": len(array), "training_seeds": expected if expected > 1 else int(first["seed"] != -1),
            "mean_nmse_linear": float(array.mean()), "mean_nmse_db": float(10 * np.log10(array.mean())),
            "geometry_ci_low_db": float(intervals[0]), "geometry_ci_high_db": float(intervals[1]),
            "mean_total_ms": float(np.mean([v["total_batch_ms_per_sample"] for v in values])),
            "network_calls": first["network_calls"],
            "sbl_convergence_rate": float(np.mean([v["sbl_converged"] in (True, "True") for v in values]))
                if first["solver_applicable"] in (True, "True") else None})
    write_csv(export / "summary.csv", summary)
    effects = []
    for context in sorted({c for c, m in geometry_values}):
        proposed = geometry_values[context, "prior_learned"]
        for reference in ("prior_fixed", "lmmse_joint", "sbl_dense", "ddpm_latest_avg16", "deterministic_latest"):
            baseline = geometry_values[context, reference]
            delta = 10 * np.log10(proposed[resamples].mean(1) / baseline[resamples].mean(1))
            lo, hi = np.percentile(delta, [2.5, 97.5])
            effects.append({"context": context, "method": "prior_learned", "reference": reference,
                "delta_nmse_db": float(10 * np.log10(proposed.mean() / baseline.mean())),
                "geometry_ci_low_db": float(lo), "geometry_ci_high_db": float(hi)})
    write_csv(export / "paired_effects.csv", effects)
    return summary


def plot_results(summary, export):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    bands = sorted({r["bandwidth_hz"] for r in summary})
    budgets = sorted({r["configurations"] for r in summary})
    selections = {
        "snr_nmse_main": ("ls_min_norm", "lmmse_joint", "sbl_dense", "ddpm_latest_avg16", "prior_learned"),
        "snr_nmse_ablation": ("lmmse_joint", "deterministic_latest", "ddpm_latest_avg1", "ddpm_latest_avg16", "prior_fixed", "prior_learned")}
    colors = ("#000000", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00")
    markers = ("o", "s", "^", "D", "v", "P")
    for name, methods in selections.items():
        fig, axes = plt.subplots(len(bands), len(budgets), figsize=(11, 7.5), squeeze=False,
            sharex=True, sharey=True, layout="constrained")
        for i, band in enumerate(bands):
            for j, budget in enumerate(budgets):
                ax = axes[i, j]
                for number, method in enumerate(methods):
                    values = sorted((r for r in summary if r["method"] == method and r["bandwidth_hz"] == band
                        and r["configurations"] == budget), key=lambda r: r["snr_db"])
                    snrs = [r["snr_db"] for r in values]
                    ax.plot(snrs, [r["mean_nmse_db"] for r in values], color=colors[number],
                        marker=markers[number], markersize=4, linewidth=1.5, linestyle="--" if number % 2 else "-",
                        label=LABELS[method])
                    ax.fill_between(snrs, [r["geometry_ci_low_db"] for r in values],
                        [r["geometry_ci_high_db"] for r in values], color=colors[number], alpha=.08)
                ax.set_title(f"B = {band / 1e9:g} GHz, J = {budget}")
                ax.set_xlabel("Training-reference receive SNR (dB)")
                ax.set_ylabel("Mean NMSE (dB)")
                ax.set_xticks(sorted({r["snr_db"] for r in values}))
                ax.grid(alpha=.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        for extension in ("png", "svg"):
            fig.savefig(export / f"{name}.{extension}", dpi=180)
        plt.close(fig)


def audit_results(study, rows, frozen, statistics):
    from .continuation import verify_archive
    verify_archive(study.path)
    verify_frozen_models(study.path, frozen)
    if any(r["network_calls"] != r["denoiser_calls"] + r["variance_update_calls"] for r in rows):
        raise AssertionError("network call accounting differs")
    lookup = {(r["context"], r["method"], r["seed"], r["scene"]): r for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError("duplicate metric rows")
    observed = torch.load(study.path / "predictions" / "observations.pt", map_location="cpu", weights_only=True)
    maximum, count = 0., 0
    for context, entry in zip(observed["contexts"], observed["splits"]["screen"]):
        c = entry["context_index"]
        predictions = torch.load(study.path / "predictions" / f"context_{c}.pt", map_location="cpu", weights_only=True)
        for label, output in predictions.items():
            method, seed = label.split("/")
            errors = nmse(entry["truth"], output)
            for scene, value in enumerate(errors.tolist()):
                row = lookup[c, method, int(seed), scene]
                if row["observation_sha256"] != tensor_hash(entry["observation"][scene]):
                    raise AssertionError("observation hash mismatch")
                difference = abs(value - row["nmse_linear"])
                maximum = max(maximum, difference)
                if difference > 1e-7 + abs(value) * 2e-6:
                    raise AssertionError("stored prediction does not reproduce NMSE")
                count += 1
    if count != len(rows):
        raise AssertionError("not every reported metric was audited")
    device, settings = study.device, study.config["comparison"]
    schedule = VPSchedule(study.config["learning"]["diffusion_steps"]).to(device)
    seed = settings["seeds"][0]
    item = frozen["models"][str(seed)]
    prior = DenoisingCNN(width=study.config["learning"]["width"]).to(device)
    prior.load_state_dict(torch.load(study.path / item["prior_path"], map_location="cpu", weights_only=True)["state_dict"])
    prior.eval().requires_grad_(False)
    updater = VarianceUpdate(settings["update_width"]).to(device)
    updater.load_state_dict(torch.load(study.path / item["update_path"], map_location="cpu", weights_only=True)["state_dict"])
    updater.eval()
    replay_difference, replays = 0., 0
    with torch.no_grad():
        for c in sorted({0, len(observed["contexts"]) - 1}):
            context, entry = observed["contexts"][c], observed["splits"]["screen"][c]
            stats = statistics[context["bandwidth_hz"], context["configurations"], context["snr_db"]]
            y, a = entry["observation"].to(device), context["matrix"].to(device)
            initial = apply_lmmse(stats, y)
            diagonal = stats["diagonal"].to(device=device, dtype=y.real.dtype).clamp_min(1e-8)[None].expand(len(y), -1, -1)
            saved = torch.load(study.path / "predictions" / f"context_{c}.pt", map_location="cpu", weights_only=True)
            sl = slice(0, settings["batch_size"])
            for method, network in (("prior_fixed", None), ("prior_learned", updater)):
                actual = structured_recover(prior, schedule, a, y[sl], entry["variance"][sl].to(device), initial[sl], diagonal[sl],
                    updater=network, iterations=settings["structured_iterations"], strength=item["strength"])[0].cpu()
                difference = float((actual - saved[f"{method}/{seed}"][sl]).abs().max())
                replay_difference = max(replay_difference, difference)
                if difference > 2e-6:
                    raise AssertionError("frozen structured recovery replay differs")
                replays += 1
    report = {"status": "passed", "source_archive_verified": True, "metrics_recomputed": count,
        "maximum_metric_difference": maximum, "structured_replays": replays,
        "maximum_replay_difference": replay_difference, "same_observations": True,
        "selection_split": "validation", "normalization": "parent training-only global scale",
        "frozen_at_unix": frozen["frozen_at_unix"], "screen_generated_after_freeze":
            (study.path / "predictions" / "observations.pt").stat().st_mtime >= frozen["frozen_at_unix"]}
    if not report["screen_generated_after_freeze"]:
        raise AssertionError("screening observations predate the validation freeze")
    write_json(study.export / "audit.json", report)
    return report


def write_report(study, summary, audit):
    cfg, settings = study.config, study.config["comparison"]
    overall = {}
    for method in LABELS:
        values = [r["mean_nmse_linear"] for r in summary if r["method"] == method]
        if values:
            overall[method] = 10 * np.log10(np.mean(values))
    lines = ["# DMA信道估计：SNR扫描与DDPM先验结构化更新", "", f"运行：`{study.run_id}`。父运行：`{settings['parent_run']}`。",
        "", "这是在既有实验结果基础上设计的新探索性比较。所有方法共享每个场景的真实DMA观测和噪声。",
        f"系统：{cfg['system']['antennas']}阵元、{cfg['system']['tones']}频点、单RF链；角度±{cfg['channel']['angle_limit_deg']}°、距离{cfg['channel']['range_min_m']}–{cfg['channel']['range_max_m']} m。",
        f"SNR点：{settings['snrs_db']} dB；独立筛选几何{settings['screen_scenes']}个；学习方法初始化{settings['seeds']}。",
        "SNR按训练集在白化观测域中的平均接收功率定义；逐场景实际SNR也保存在原始CSV。",
        "", "## 方法与信息条件", "", "- LS使用逐频Moore–Penrose最小范数解。",
        "- LMMSE使用训练几何估计完整阵元—频率联合均值和协方差，保留跨频统计。",
        f"- 主SBL采用密字典，最多{settings['sbl_max_iterations']}次更新，按标准化一阶条件{settings['sbl_tolerance']}停止；逐场景实际次数和非收敛完整保存。",
        "- 最新DDPM为父运行已冻结的条件残差模型；主图使用16次平均，辅助图保留单次输出。其SBL初始化与训练时预算一致。",
        "- 新DDPM直接学习完整干净信道分布；冻结后以LMMSE初始化，交替DDPM去噪与真实逐频DMA的高斯观测更新。",
        "- 学习更新版本只预测正的阵元—频率先验方差，复均值仍由解析条件式计算。固定更新消融使用同一先验、初始化、迭代数及验证选择的全局强度。",
        "- DDPM参数、更新网络及超参数只从训练/验证数据获得；新筛选几何在冻结记录写出后生成。",
        "- 旧条件模型和新更新网络的带噪训练仍使用父运行的SNR配置；其余SNR检验冻结权重的迁移。DDPM先验训练的人工扩散噪声不等同于导频SNR。",
        "- 该交替算法是plug-and-play近似；内部对角方差未标定为真实后验误差。Li等TCOM2026提供学习先验方差更新的启发，其原文不使用DDPM。",
        "", "## 曲线", "", "![五方法SNR–NMSE](snr_nmse_main.png)", "", "![固定/学习更新与确定性对照](snr_nmse_ablation.png)",
        "", "每个标记对应实际运行点，相邻线段仅用于连接。NMSE先逐场景归一化、在线性尺度平均，再转dB。",
        "阴影为95%几何场景bootstrap区间；先平均各训练初始化，再按几何块重采样，包含所有配对SNR/硬件条件。阴影不包括训练初始化不确定性。",
        "", "## 全部条件平均", "", "| 方法 | NMSE/dB |", "|---|---:|"]
    lines += [f"| {LABELS[k]} | {v:.4f} |" for k, v in overall.items()]
    bad = [r for r in summary if r["method"] == "sbl_dense" and r["sbl_convergence_rate"] < .9]
    lines += ["", f"密字典SBL有{len(bad)}个条件的一阶条件通过率低于90%；各点通过率见summary.csv。",
        f"审计重算{audit['metrics_recomputed']}条指标，最大差{audit['maximum_metric_difference']}；{audit['structured_replays']}次结构化恢复重放最大差{audit['maximum_replay_difference']}。",
        "", "[逐场景数据](metrics.csv) · [汇总与区间](summary.csv) · [配对差异](paired_effects.csv) · [冻结参数](frozen.json) · [审计](audit.json)",
        "", "在线时间包含初始化与恢复的批量均摊应用时间，已知矩阵预计算、训练和预热另行排除；它不等于单样本时延。",
        "当与固定更新、LMMSE或密字典SBL相比没有改善时，保留原始结果，不用观测残差下降替代完整CSI改善。"]
    if settings.get("implementation_only"):
        lines += ["", "本运行仅为缩小预算的实现验证，其性能数值不作为方法结论。"]
    (study.export / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root, config_path, parent_id, run_id, device_override=None):
    root = Path(root)
    if any(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None for value in (parent_id, run_id)) or parent_id == run_id:
        raise ValueError("distinct safe parent and comparison run IDs required")
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_comparison(settings)
    parent_path = root / "runs" / parent_id
    original = json.loads((parent_path / "config_resolved.json").read_text())
    parent_frozen = json.loads((parent_path / "stages" / "frozen_base.json").read_text())
    if json.loads((parent_path / "stages" / "final.json").read_text())["status"] != "complete":
        raise ValueError("the parent final comparison must be complete")
    for seed in settings["seeds"]:
        if any(f"{kind}_none_{seed}" not in parent_frozen["models"] for kind in ("diffusion", "deterministic")):
            raise ValueError("requested seed has no frozen parent conditional/deterministic model")
    if any(settings[new] > original["data"][old] for new, old in (("training_scenes", "train"), ("validation_scenes", "validation"))):
        raise ValueError("parent training/validation geometry counts cannot be exceeded")
    config = copy.deepcopy(original)
    config["label"] = "snr_ddpm_prior_structured_implementation_only" if settings.get("implementation_only") else "snr_ddpm_prior_structured_exploration"
    config["system"]["snrs_db"] = settings["snrs_db"]
    config["data"]["screen_seed"] = settings["screen_seed"]
    config["data"]["screen"] = settings["screen_scenes"]
    config["timeout_seconds"] = settings["timeout_seconds"]
    config["comparison"] = {**settings, "parent_run": parent_id, "training_snrs_db": original["system"]["snrs_db"]}
    with tempfile.TemporaryDirectory() as temporary:
        resolved = Path(temporary) / "config.json"
        write_json(resolved, config)
        study = Study(root, resolved, run_id, device_override)
    config = study.config
    def timeout():
        study.log("hard_timeout", limit_seconds=settings["timeout_seconds"])
        write_json(study.path / "status.json", {"status": "timeout"})
        os._exit(124)
    timer = threading.Timer(settings["timeout_seconds"], timeout)
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        cache_id = parent_frozen["training_cache_id"]
        source = parent_path / "cache" / f"{cache_id}.pt"
        source_hash = file_hash(source)
        if source_hash != parent_frozen["training_cache_sha256"]:
            raise ValueError("parent training cache differs from its frozen provenance")
        parent = torch.load(source, map_location="cpu", weights_only=True)
        if set(parent["splits"]) != {"train", "validation"} or any(e["sbl_solver"] != SOLVER_VERSION
                for split in parent["splits"].values() for e in split):
            raise ValueError("only corrected, training/validation-only parent caches may be inherited")
        provenance = {"parent_run": parent_id, "cache_id": cache_id, "cache_sha256": source_hash,
            "parent_frozen_sha256": file_hash(parent_path / "stages" / "frozen_base.json"),
            "parent_source_sha256": file_hash(parent_path / "source.zip"), "splits": ["train", "validation"]}
        write_json(study.export / "parent_provenance.json", provenance)
        training = sweep_cache(config, parent, "train", settings["training_scenes"], original["system"]["snrs_db"])
        validation = sweep_cache(config, parent, "validation", settings["validation_scenes"], settings["snrs_db"])
        # Geometry hashes are checked on the actual inherited channel rows, including subset smoke runs.
        bands = config["system"]["bandwidths_hz"]
        train_h = {c["bandwidth_hz"]: e["truth"] for c, e in zip(training["contexts"], training["splits"]["train"])}
        val_h = {c["bandwidth_hz"]: e["truth"] for c, e in zip(validation["contexts"], validation["splits"]["validation"])}
        train_hashes = {tensor_hash(h) for h in train_h[bands[0]]}
        if train_hashes.intersection(tensor_hash(h) for h in val_h[bands[0]]):
            raise AssertionError("training and validation geometry overlap")
        stats_path = study.path / "cache" / "lmmse_statistics.pt"
        if stats_path.exists():
            statistics = torch.load(stats_path, map_location="cpu", weights_only=True)
        else:
            statistics = {}
            for context in [*training["contexts"], *validation["contexts"]]:
                key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
                if key not in statistics:
                    fitted = lmmse_statistics(train_h[key[0]].to(study.device), context["matrix"].to(study.device), context["noise_complex_variance"])
                    statistics[key] = {k: v.cpu() for k, v in fitted.items()}
                    study.log("lmmse_statistics", bandwidth_hz=key[0], configurations=key[1], snr_db=key[2])
            save_tensor(stats_path, statistics)
        frozen_path = study.path / "stages" / "frozen.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            if frozen["parent_provenance"] != provenance or frozen["lmmse_sha256"] != file_hash(stats_path):
                raise ValueError("frozen parent/statistics changed")
        else:
            write_json(study.path / "status.json", {"status": "training"})
            models = fit_models(study, parent, training, validation, statistics)
            latest = {}
            for seed in settings["seeds"]:
                for kind in ("diffusion", "deterministic"):
                    key = f"{kind}_none_{seed}"
                    item = parent_frozen["models"][key]
                    old = parent_path / item["path"]
                    if file_hash(old) != item["sha256"]:
                        raise ValueError("parent frozen model changed")
                    destination = study.path / "models" / f"latest_{kind}_{seed}.pt"
                    destination.write_bytes(old.read_bytes())
                    latest[key] = {**item, "path": destination.relative_to(study.path).as_posix()}
            chosen = parent_frozen["selected_recipe"]
            frozen = {"models": models, "latest_models": latest, "lmmse_sha256": file_hash(stats_path),
                "parent_provenance": provenance, "training_geometry_sha256": sorted(train_hashes),
                "validation_geometry_sha256": sorted(tensor_hash(h) for h in val_h[bands[0]]),
                "training_snrs_db": original["system"]["snrs_db"], "test_snrs_db": settings["snrs_db"],
                "coarse_iterations": chosen["iterations"],
                "coarse_grid": {k: original["sbl"][k] for k in ("angle_points", "range_points")},
                "dense_grid": {k: parent_frozen["baselines"]["dense"]["grid"][k] for k in ("angle_points", "range_points")},
                "selection_split": "validation", "screening_used": False, "frozen_at_unix": time.time()}
            temporary = frozen_path.with_suffix(".tmp")
            write_json(temporary, frozen)
            temporary.replace(frozen_path)
            study.log("frozen", sha256=file_hash(frozen_path))
        exported_freeze = study.export / "frozen.json"
        if exported_freeze.exists() and exported_freeze.read_bytes() != frozen_path.read_bytes():
            raise ValueError("exported validation freeze differs from the run record")
        exported_freeze.write_bytes(frozen_path.read_bytes())
        write_json(study.path / "status.json", {"status": "screening"})
        rows = evaluate(study, parent, frozen, statistics)
        summary = summarize_results(rows, settings, study.export)
        plot_results(summary, study.export)
        audit = audit_results(study, rows, frozen, statistics)
        write_report(study, summary, audit)
        write_json(study.path / "status.json", {"status": "complete", "metrics_rows": len(rows), "audit": audit})
        study.log("complete", metrics_rows=len(rows), result=str(study.export))
        return study.export
    except Exception as error:
        write_json(study.path / "status.json", {"status": "failed", "error": str(error)})
        (study.path / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        study.log("failed", error=str(error))
        raise
    finally:
        timer.cancel()
