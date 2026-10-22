"""Replay completed predictions and independently check physical units and LS errors."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.data import physical_channels, sample_scenes, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.physical_study import METHODS, case_hardware, geometry_ids


RUN = ROOT / "runs/physical_baselines_20261021_v2"
OUT = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
config = json.loads((RUN / "config_resolved.json").read_text())
assert json.loads((RUN / "status.json").read_text())["status"] == "complete"
frozen = json.loads((RUN / "frozen.json").read_text())
freeze_hash = file_hash(RUN / "frozen.json")
assert not frozen["screening_used"] and frozen["selection_split"] == "validation"
assert file_hash(RUN / "source.zip") == frozen["source_archive_sha256"]
manifest = json.loads((RUN / "source_manifest.json").read_text())
with zipfile.ZipFile(RUN / "source.zip") as archive:
    for name, digest in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        if name.startswith(("src/", "scripts/")) and name.endswith(".py"):
            assert file_hash(ROOT / name) == digest
for name, digest in frozen["cache_sha256"].items():
    assert file_hash(RUN / "cache" / name) == digest
assert file_hash(RUN / "inputs/absorption_table.txt") == config["channel"]["absorption_table_sha256"]
workers = [json.loads(p.read_text()) for p in sorted(RUN.glob("parallel*worker.json"))]
assert workers
for worker in workers:
    assert file_hash(ROOT / worker["source_path"]) == worker["source_sha256"]

with (OUT / "metrics.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
records = {(r["case"], int(r["context"]), r["method"], int(r["scene"])): r for r in rows}
assert len(records) == len(rows)
statistics = torch.load(RUN / "cache/statistics.pt", weights_only=True)
max_metric_relative = max_replay_relative = max_observation_relative = 0.
max_thermal_relative = max_ls_identity_relative = 0.
checked = nonconverged = 0
common_ids, noise_hashes = {}, {}
ls_rows, solver_rows = [], []
profiles, physical_reference = [], None

for case in config["cases"]:
    name = case["name"]
    cache = torch.load(RUN / "cache" / f"{name}.pt", weights_only=True)
    selected, _, hardware = case_hardware(config, case)
    observations = torch.load(RUN / "predictions" / name / "observations.pt", weights_only=True)
    assert observations["frozen_sha256"] == freeze_hash
    assert observations["created_at_unix"] >= frozen["frozen_at_unix"]
    ids = {**cache["geometry_ids"], "screen": observations["geometry_ids"]}
    for split_index, split in enumerate(("train", "validation", "screen")):
        scenes = sample_scenes(selected, split_index, config["data"][split])
        assert geometry_ids(scenes) == ids[split]
        assert common_ids.setdefault(split, ids[split]) == ids[split]
        assert len(set(ids[split])) == config["data"][split]
    assert not (set(ids["train"]) & set(ids["validation"]))
    assert not (set(ids["train"] + ids["validation"]) & set(ids["screen"]))
    train_scenes = sample_scenes(selected, 0, config["data"]["train"])
    training = [physical_channels(hw, train_scenes) for hw in hardware]
    profile = {"case": name, "split": "train", "geometries": len(train_scenes[0]),
        "complex_channel_rms": [float(h.abs().square().mean().sqrt()) for h in training],
        "received_snr_at_0_dbm": [{"bandwidth_hz": c["bandwidth_hz"], "configurations": c["configurations"],
            "snr_db": c["snr_db"]} for c in cache["contexts"] if c["transmit_power_dbm"] == 0]}
    if case["model"] == "li2026":
        energies = []
        for path in range(config["channel"]["paths"]):
            mask = np.zeros_like(train_scenes[2])
            mask[:, path] = 1
            part = physical_channels(hardware[0], (*train_scenes[:2], mask))
            energies.append(part.abs().square().sum((1, 2)))
        energy = torch.stack(energies, 1)
        ratio = energy[:, 1:].sum(1)/energy[:, 0]
        profile.update(reflected_to_los_incoherent_energy_median_db=float(10*ratio.median().log10()),
            reflected_to_los_incoherent_energy_q95_db=float(10*torch.quantile(ratio, .95).log10()))
        if not case["include_absorption"]:
            physical_reference = training
        elif physical_reference is not None:
            profile["absorption_only_channel_relative_mse"] = [float(((h-ref).abs().square().sum((1, 2)) /
                ref.abs().square().sum((1, 2))).mean()) for h, ref in zip(training, physical_reference)]
    profiles.append(profile)
    scale = float(torch.stack([h.abs().square().mean() for h in training]).mean().sqrt() / np.sqrt(2))
    assert scale == cache["channel_scale"]
    assert all(torch.equal(h / scale, saved) for h, saved in zip(training, cache["training"]))
    screen_scenes = sample_scenes(selected, 2, config["data"]["screen"])
    truths = [physical_channels(hw, screen_scenes) / scale for hw in hardware]
    standard = []
    for band_index, band in enumerate(config["system"]["bandwidths_hz"]):
        rng = np.random.default_rng(np.random.SeedSequence([config["data"]["screen_seed"], 2, band_index, 811]))
        shape = (config["data"]["screen"], config["system"]["tones"], len(cache["state_bank"]))
        standard.append(torch.tensor((rng.normal(size=shape) + 1j*rng.normal(size=shape))/np.sqrt(2), dtype=truths[band_index].dtype))
    for context, entry in zip(cache["contexts"], observations["entries"]):
        index, count = entry["context_index"], context["configurations"]
        band_index = config["system"]["bandwidths_hz"].index(context["bandwidth_hz"])
        assert torch.equal(entry["truth"], truths[band_index])
        thermal = config["system"]["thermal_noise"]
        kbt = 1.380649e-23 * thermal["antenna_temperature_k"] * context["bandwidth_hz"] / config["system"]["tones"]
        expected_variance = kbt / (context["pilot_power_per_tone_w"] * scale**2)
        max_thermal_relative = max(max_thermal_relative, abs(context["noise_complex_variance"]/expected_variance-1))
        assert np.isclose(expected_variance, context["noise_complex_variance"], rtol=1e-14)
        base = standard[band_index][:, :, :count]
        physical_noise = base * np.sqrt(kbt)
        assert tensor_hash(physical_noise) == entry["physical_noise_sha256"]
        key = (context["bandwidth_hz"], count)
        assert noise_hashes.setdefault(key, entry["physical_noise_sha256"]) == entry["physical_noise_sha256"]
        a = context["matrix"]
        clean = torch.einsum("kjn,bkn->bkj", a, entry["truth"])
        regenerated = clean + base * np.sqrt(expected_variance)
        assert torch.equal(regenerated, entry["observation"])
        physical = np.sqrt(context["pilot_power_per_tone_w"])*scale*clean + physical_noise
        normalized = np.sqrt(context["pilot_power_per_tone_w"])*scale*entry["observation"]
        relative = float((physical-normalized).abs().max()/physical.abs().max())
        max_observation_relative = max(max_observation_relative, relative)
        assert relative < 1e-5
        noise_gain = hardware[band_index]["noise_gain"][:, :count]
        assert torch.equal(noise_gain*kbt, context["raw_noise_per_tone_w"])
        truth, y = entry["truth"].to(torch.complex128), entry["observation"].to(torch.complex128)
        inverse = torch.linalg.pinv(a.to(torch.complex128))
        ls = torch.einsum("knj,bkj->bkn", inverse, y).to(entry["truth"].dtype)
        saved = torch.load(RUN / "predictions" / name / f"context_{index}.pt", weights_only=True)
        assert saved["frozen_sha256"] == freeze_hash and saved["observation_sha256"] == tensor_hash(entry["observation"])
        assert set(saved["outputs"]) == set(METHODS)
        replays = {"ls_min_norm": ls}
        for method, stats in statistics[name][index].items():
            residual = (y - stats["observed_mean"]).flatten(1)
            replays[method] = (stats["mean"].flatten()[None] + residual @ stats["gain"].T).reshape_as(truth).to(entry["truth"].dtype)
        for method, replay in replays.items():
            prediction = saved["outputs"][method]
            relative = float(torch.linalg.vector_norm(replay-prediction)/torch.linalg.vector_norm(prediction).clamp_min(1e-30))
            max_replay_relative = max(max_replay_relative, relative)
            assert relative < 2e-5
        energy = truth.abs().square().sum((1, 2))
        for method, prediction in saved["outputs"].items():
            error = prediction.to(torch.complex128) - truth
            metric = error.abs().square().sum((1, 2))/energy
            original = torch.tensor([float(records[name, index, method, scene]["nmse_linear"]) for scene in range(len(truth))], dtype=torch.float64)
            maximum = float(((metric-original).abs()/original.clamp_min(1e-30)).max())
            max_metric_relative = max(max_metric_relative, maximum)
            assert maximum < 2e-5
            assert all(records[name, index, method, scene]["observation_sha256"] == tensor_hash(entry["observation"][scene]) for scene in range(len(truth)))
            checked += len(truth)
        projected = torch.einsum("knj,bkj->bkn", inverse, clean.to(torch.complex128))
        bias = (projected-truth).abs().square().sum((1, 2))/energy
        noise_fit = (saved["outputs"]["ls_min_norm"].to(torch.complex128)-projected).abs().square().sum((1, 2))/energy
        empirical = (saved["outputs"]["ls_min_norm"].to(torch.complex128)-truth).abs().square().sum((1, 2))/energy
        relative = float(((empirical-bias-noise_fit).abs()/empirical.clamp_min(1e-30)).max())
        max_ls_identity_relative = max(max_ls_identity_relative, relative)
        assert relative < 1e-5
        expected_noise = expected_variance * inverse.abs().square().sum() / energy
        ls_rows.append({"case": name, "context": index, "bandwidth_hz": context["bandwidth_hz"], "configurations": count,
            "transmit_power_dbm": context["transmit_power_dbm"], "training_received_snr_db": context["snr_db"],
            "null_energy_fraction": float(bias.mean()), "expected_noise_nmse": float(expected_noise.mean()),
            "realized_noise_nmse": float(noise_fit.mean()), "expected_ls_nmse": float((bias+expected_noise).mean()),
            "realized_ls_nmse": float(empirical.mean())})
        solver = saved["solver"]
        residual, converged = solver["sbl_optimality_residual"], solver["sbl_converged"]
        assert torch.isfinite(residual).all()
        assert torch.equal(converged, residual <= config["sbl"]["tolerance"])
        assert int(solver["sbl_iterations"].max()) <= config["sbl"]["iterations"]
        nonconverged += int((~converged).sum())
        solver_rows.append({"case": name, "context": index, "nonconverged": int((~converged).sum()),
            "max_optimality_residual": float(residual.max()), "maximum_iterations": int(solver["sbl_iterations"].max())})
    print(json.dumps({"replayed_case": name, "metrics_checked": checked}), flush=True)

expected = len(config["cases"])*len(config["system"]["bandwidths_hz"])*len(config["system"]["configurations"])*len(config["system"]["transmit_power_dbm"])*config["data"]["screen"]*len(METHODS)
assert checked == len(records) == expected
assert nonconverged == json.loads((OUT / "audit.json").read_text())["sbl_nonconverged"]
write_csv(OUT / "ls_error_budget.csv", ls_rows)
write_csv(OUT / "solver_audit.csv", solver_rows)
write_json(OUT / "training_model_profiles.json", {"scope": "truth-based training diagnostics; not estimator inputs or causal isolation", "profiles": profiles})
write_json(OUT / "independent_audit.json", {"status": "passed", "script_sha256": file_hash(Path(__file__)),
    "metrics_recomputed_in_float64": checked, "maximum_nmse_relative_difference": max_metric_relative,
    "linear_predictions_replayed": checked*4//5, "maximum_linear_prediction_relative_difference": max_replay_relative,
    "maximum_ls_bias_noise_identity_relative_error": max_ls_identity_relative,
    "maximum_thermal_variance_relative_error": max_thermal_relative,
    "maximum_physical_observation_equivalence_relative_error": max_observation_relative,
    "independent_screen_geometry_count": len(common_ids["screen"]), "screen_created_after_freeze": True,
    "training_only_scale_reproduced": True, "paired_physical_noise_regenerated_from_seed": True,
    "sbl_nonconverged": nonconverged, "sbl_scope": "saved outputs and first-order stopping records checked; solver not rerun",
    "parallel_workers": workers, "source_archive_sha256": file_hash(RUN / "source.zip")})
print(json.dumps({"status": "passed", "metrics": checked, "sbl_nonconverged": nonconverged}), flush=True)
