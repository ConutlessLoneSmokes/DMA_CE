"""Bounded coherent-prior experiment; validation precedes both evaluation cohorts."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from dma_ce.coherent import coherent_posterior, delay_model, smooth_envelope
from dma_ce.data import sample_scenes, sync, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.physical_study import archive, case_hardware, make_entries
from dma_ce.physics import element_channel, free_space_amplitude, scene_points
from dma_ce.prior_probe import covariance_sbl, propagation_basis
from dma_ce.research import save_tensor
from dma_ce.sbl import posterior


def risk(truth, prediction):
    return ((truth.to(torch.complex128) - prediction.to(torch.complex128)).abs().square().sum((1, 2))
        / truth.to(torch.complex128).abs().square().sum((1, 2)))


def representation_check(config, hardware, cache, count):
    """Training-only oracle diagnostic; none of these geometry inputs reaches inference."""
    radii, angles, gains = (values[:count] for values in sample_scenes(config, 0, config["data"]["train"]))
    geometry = torch.tensor(np.stack((angles, radii), -1), dtype=torch.float64)
    f = torch.tensor(hardware["frequencies"], dtype=torch.float64)
    positions = torch.tensor(hardware["positions"], dtype=torch.float64)
    table = hardware["absorption_table"]
    kappa = torch.tensor(table.coefficient(f.numpy()), dtype=torch.float64)
    parts = propagation_basis(geometry, f, positions, kappa, config["channel"], cache["channel_scale"])
    columns = []
    for radius, angle in zip(radii, angles):
        atoms = []
        for r, theta in zip(radius, angle):
            atom = element_channel(f.numpy(), positions.numpy(), r, theta, absorption_table=table,
                include_absorption=True, elevation_deg=config["channel"]["elevation_deg"])
            atom /= (free_space_amplitude(f.numpy(), r) * np.exp(-2j * np.pi * f.numpy() * r / 299792458.))[:, None]
            atoms.append(atom / np.linalg.norm(atom, axis=-1, keepdims=True))
        columns.append(np.stack(atoms, -1))
    spatial = torch.tensor(np.stack(columns), dtype=torch.complex128)
    points = scene_points(radii, angles, config["channel"]["elevation_deg"])
    length = np.linalg.norm(points - points[:, :1], axis=-1) + radii
    parts *= torch.exp(-2j * math.pi * f.mean() * torch.tensor(length) / 299792458.)[:, None, None]
    phase = torch.exp(-2j * math.pi * (f - f.mean())[None, :, None] * torch.tensor(length)[:, None] / 299792458.)
    coefficients = (spatial.conj() * parts).sum(2)
    envelope = smooth_envelope(f, config["probe"]["envelope_weights"]).to(torch.complex128)
    fitted = torch.einsum("kp,pj,bjl->bkl", envelope, torch.linalg.pinv(envelope), coefficients / phase) * phase
    truth = parts.sum(-1)
    torch.testing.assert_close(truth, cache["training"][0][:count].to(torch.complex128), rtol=1e-6, atol=1e-6)
    free = (spatial * coefficients[:, :, None]).sum(-1)
    constrained = (spatial * fitted[:, :, None]).sum(-1)
    extra = (fitted - coefficients).abs().square().sum((1, 2)) / coefficients.abs().square().sum((1, 2))
    return {"label": "training_geometry_oracle_representation_only", "scenes": count,
        "free_spatial_nmse": risk(truth, free).tolist(), "smooth_coherent_nmse": risk(truth, constrained).tolist(),
        "additional_coefficient_envelope_error": extra.tolist(),
        "mean_free_spatial_nmse_db": float(10 * risk(truth, free).mean().log10()),
        "mean_smooth_coherent_nmse_db": float(10 * risk(truth, constrained).mean().log10()),
        "mean_additional_envelope_error_db": float(10 * extra.mean().log10()),
        "geometry_not_provided_to_actual_inference": True}


def run(smoke):
    settings = json.loads((ROOT / "configs/coherent_delay_probe.json").read_text())
    if smoke:
        settings.update(run_id="coherent_delay_checks_20261026_v3", validation_geometries=4, screen_geometries=4,
            historical_screen_geometries=0, snrs_db=[-10, 30], sbl_iterations=3, coherent_atoms=4,
            delay_grid=32, strengths=[0, .5], timeout_seconds=180, screen_seed=2026102699)
    if (any(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", settings[key]) is None
            for key in ("run_id", "parent_run", "configuration_reference"))
            or not 0 < settings["timeout_seconds"] <= 1800
            or not 2 <= settings["validation_geometries"] <= 32
            or not 2 <= settings["screen_geometries"] <= 32
            or not 0 <= settings["historical_screen_geometries"] <= 16
            or not 1 <= settings["sbl_iterations"] <= 100
            or settings["strengths"][0] != 0 or any(not 0 <= x <= 1 for x in settings["strengths"])
            or settings["fixed_strength"] not in settings["strengths"]):
        raise ValueError("This probe requires its bounded configuration")
    parent = ROOT / "runs" / settings["parent_run"]
    if json.loads((parent / "status.json").read_text())["status"] != "complete":
        raise ValueError("Complete parent run required")
    original = json.loads((parent / "config_resolved.json").read_text())
    frozen_parent = json.loads((parent / "frozen.json").read_text())
    configuration_file = ROOT / "results" / settings["configuration_reference"] / "frozen.json"
    configuration = json.loads(configuration_file.read_text())
    indices = [original["system"]["snrs_db"].index(snr) for snr in settings["snrs_db"]]
    used = ["cache/physical.pt", "source.zip", *[f"cache/coarse_validation_{i}.pt" for i in indices]]
    for name in used:
        if file_hash(parent / name) != frozen_parent["files_sha256"][name]:
            raise ValueError(f"Frozen parent input changed: {name}")
    config = copy.deepcopy(original)
    config.update(label="coherent_delay_exploratory", device=settings["device"], timeout_seconds=settings["timeout_seconds"],
        implementation_only=smoke, probe=settings)
    config["system"]["snrs_db"] = settings["snrs_db"]
    config["data"].update(screen=settings["screen_geometries"], screen_seed=settings["screen_seed"])
    path = ROOT / "runs" / settings["run_id"]
    export = path / "export"
    if path.exists() or shutil.disk_usage(path.parent).free < 3_000_000_000:
        raise ValueError("Unused run ID and 3 GB free storage required")
    path.mkdir()
    for name in ("inputs", "cache", "predictions", "export"):
        (path / name).mkdir()
    archive(ROOT, path, export, config)
    shutil.copyfile(parent / "cache/physical.pt", path / "inputs/physical_parent.pt")
    provenance = {"parent_run": settings["parent_run"], "original_context_indices": indices,
        "used_parent_files_sha256": {name: file_hash(parent / name) for name in used},
        "configuration_freeze_sha256": file_hash(configuration_file), "selected_bank_sha256": configuration["selected_bank_sha256"]}
    write_json(path / "parent_provenance.json", provenance)

    def status(value):
        write_json(path / "status.json", value)
        write_json(export / "status.json", value)

    def log(event, **fields):
        record = {"event": event, "at_unix": time.time(), **fields}
        with (path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def timeout():
        status({"status": "timeout", "limit_seconds": settings["timeout_seconds"]})
        os._exit(124)

    started = time.time()
    timer = threading.Timer(settings["timeout_seconds"], timeout)
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(settings["device"])
        cache = torch.load(path / "inputs/physical_parent.pt", weights_only=True, map_location="cpu")
        selected, bank, hardware = case_hardware(config, config["cases"][0])
        hw = hardware[0]
        assert tensor_hash(torch.tensor(bank)) == configuration["selected_bank_sha256"] == tensor_hash(cache["state_bank"])
        matrix = cache["contexts"][0]["matrix"].to(device)
        assert tensor_hash(hw["matrix"]) == tensor_hash(matrix)
        dictionary = hw["dictionary"].to(device)
        matrix_double, dictionary_double = matrix.to(torch.complex128), dictionary.to(torch.complex128)
        sensing = matrix @ dictionary
        frequencies = torch.tensor(hw["frequencies"], device=device, dtype=torch.float64)
        diagnostic = representation_check(selected, hw, cache, 4 if smoke else 16)
        write_json(export / "representation_diagnostic.json", diagnostic)
        log("representation_diagnostic", **{key: value for key, value in diagnostic.items() if key.startswith("mean_")})
        permutation = np.random.default_rng(settings["shuffle_seed"]).permutation(min(settings["coherent_atoms"], dictionary.shape[-1]))
        if np.array_equal(permutation, np.arange(len(permutation))):
            permutation = np.roll(permutation, 1)

        def infer(observation, variance, strengths, shuffled):
            collected, models, gammas, residuals, objectives, solutions = {}, [], [], [], {}, {}
            timing = {"base_seconds": 0., "delay_model_seconds": 0., "posterior_seconds": {}}
            for first in range(0, len(observation), config["sbl"]["batch_size"]):
                y = observation[first:first + config["sbl"]["batch_size"]].to(device)
                noise = variance[first:first + len(y)].to(device)
                sync(device)
                start = time.perf_counter()
                base = covariance_sbl(y, matrix, dictionary, noise, dictionary[..., :1], 0., settings["sbl_iterations"])
                # shortcut: spatial powers stay fixed after SBL100, revisit if this isolated update helps.
                coefficients, uncertainty, _, _ = posterior(y, sensing, dictionary, noise, base["gamma"], channel_moments=False)
                sync(device)
                timing["base_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                model = delay_model(coefficients, uncertainty, base["gamma"], frequencies, settings)
                sync(device)
                timing["delay_model_seconds"] += time.perf_counter() - start
                outputs = {"sbl100": base["mean"]}
                for strength, wrong in [(s, False) for s in strengths] + [(s, True) for s in shuffled]:
                    name = ("shuffle_" if wrong else "rho_") + str(strength)
                    used_model = {**model, "delays": model["delays"][:, permutation]} if wrong else model
                    start = time.perf_counter()
                    result = coherent_posterior(y, matrix_double, dictionary_double, noise, base["gamma"], used_model, strength)
                    sync(device)
                    timing["posterior_seconds"][name] = timing["posterior_seconds"].get(name, 0.) + time.perf_counter() - start
                    outputs[name] = result["mean"]
                    objectives.setdefault(name, []).append(result["objective_per_tone"].cpu())
                    solutions.setdefault(name, []).append(result["precision_observation"].cpu())
                for name, value in outputs.items():
                    collected.setdefault(name, []).append(value.cpu().to(torch.complex64))
                models.append({key: value.cpu() for key, value in model.items()})
                gammas.append(base["gamma"].cpu())
                residuals.append(base["optimality_residual"].cpu())
                log("inference_batch", completed=min(first + len(y), len(observation)), count=len(observation))
            return {"outputs": {key: torch.cat(values) for key, values in collected.items()},
                "gamma": torch.cat(gammas), "base_optimality_residual": torch.cat(residuals),
                "model": {key: models[0][key] if key == "envelope" else torch.cat([item[key] for item in models]) for key in models[0]},
                "objectives": {key: torch.cat(value) for key, value in objectives.items()},
                "precision_solutions": {key: torch.cat(value) for key, value in solutions.items()}, "timing": timing}

        status({"status": "validation", "run_id": settings["run_id"]})
        validation, choices = [], {}
        for local, index in enumerate(indices):
            entry = cache["validation"][index]
            n = settings["validation_geometries"]
            predicted = infer(entry["observation"][:n], entry["variance"][:n], settings["strengths"], [])
            scores = {}
            for strength in settings["strengths"]:
                score = float(risk(entry["truth"][:n], predicted["outputs"]["rho_" + str(strength)]).mean())
                scores[str(strength)] = score
                validation.append({"context": local, "snr_db": settings["snrs_db"][local], "strength": strength,
                    "mean_nmse_linear": score, "scenes": n})
            choices[str(local)] = min(settings["strengths"], key=lambda x: (scores[str(x)], x))
            save_tensor(path / f"cache/validation_{local}.pt", {**predicted, "truth": entry["truth"][:n],
                "observation": entry["observation"][:n], "variance": entry["variance"][:n],
                "geometry_ids": cache["geometry_ids"]["validation"][:n]})
            log("validation_context", snr_db=settings["snrs_db"][local], selected_strength=choices[str(local)], scores=scores)
        write_csv(export / "validation.csv", validation)
        frozen = {"selection_split": "validation", "screening_used": False, "frozen_at_unix": time.time(),
            "strength_by_context": choices, "settings": settings, "selected_bank_sha256": configuration["selected_bank_sha256"],
            "validation_geometry_ids": cache["geometry_ids"]["validation"][:settings["validation_geometries"]],
            "files_sha256": {p.relative_to(path).as_posix(): file_hash(p) for p in
                [path / "source.zip", path / "inputs/physical_parent.pt", *sorted((path / "cache").glob("validation_*.pt"))]}}
        write_json(path / "frozen.json", frozen)
        write_json(export / "frozen.json", frozen)
        reduced_cache = {**cache, "contexts": [cache["contexts"][i] for i in indices]}
        new_entries, ids = make_entries(config, selected, hardware, reduced_cache, "screen")
        known = set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"])
        for filename in (ROOT / "results").glob("*/metrics.csv"):
            with filename.open() as stream:
                known.update(row["geometry_id"] for row in csv.DictReader(stream) if row.get("geometry_id"))
        assert len(set(ids)) == len(ids) and not set(ids) & known
        new_observations = {"entries": new_entries, "geometry_ids": ids, "frozen_sha256": file_hash(path / "frozen.json"),
            "created_at_unix": time.time()}
        save_tensor(path / "predictions/new_observations.pt", new_observations)
        metrics, diagnostics = [], []

        def evaluate_cohort(name, observations, historical_sources=None):
            status({"status": name, "run_id": settings["run_id"]})
            cohort_ids = observations["geometry_ids"]
            for local, (index, entry) in enumerate(zip(indices, observations["entries"])):
                choice = choices[str(local)]
                predicted = infer(entry["observation"], entry["variance"], sorted({0, settings["fixed_strength"], choice}), [settings["fixed_strength"]])
                raw = predicted.pop("outputs")
                outputs = {"sbl100": raw["sbl100"], "free": raw["rho_0"],
                    "coherent_selected": raw["rho_" + str(choice)],
                    "coherent_half": raw["rho_" + str(settings["fixed_strength"])],
                    "shuffled_half": raw["shuffle_" + str(settings["fixed_strength"])]}
                if historical_sources is not None:
                    original_prediction = torch.load(parent / f"predictions/context_{index}.pt", weights_only=True, map_location="cpu")
                    assert original_prediction["observation_sha256"] == historical_sources["observation_hashes"][local]
                    replay = float((outputs["sbl100"] - original_prediction["ddpm_coarse"][:len(cohort_ids)]).abs().square().sum()
                        / original_prediction["ddpm_coarse"][:len(cohort_ids)].abs().square().sum())
                    assert replay < 1e-9
                    predicted["legacy_replay_relative_squared_error"] = replay
                    outputs["sbl_full"] = original_prediction["outputs"]["sbl_dense"][:len(cohort_ids)]
                    predicted["full_sbl_info"] = {key: value[:len(cohort_ids)] if isinstance(value, torch.Tensor) else value
                        for key, value in original_prediction["solver"].items()}
                    provenance.setdefault("historical_prediction_sha256", {})[str(index)] = file_hash(parent / f"predictions/context_{index}.pt")
                save_tensor(path / f"predictions/{name}_{local}.pt", {**predicted, "outputs": outputs, "geometry_ids": cohort_ids,
                    "frozen_sha256": file_hash(path / "frozen.json")})
                truth, observed = entry["truth"].to(torch.complex128), entry["observation"].to(torch.complex128)
                a = matrix_double.cpu()
                for method, estimate in outputs.items():
                    error = estimate.to(torch.complex128) - truth
                    values = risk(truth, estimate)
                    residual = (torch.einsum("kjn,bkn->bkj", a, estimate.to(torch.complex128)) - observed).abs().square().sum((1, 2)) / observed.abs().square().sum((1, 2))
                    physical_mse = error.abs().square().mean((1, 2)) * cache["channel_scale"] ** 2
                    for scene, geometry_id in enumerate(cohort_ids):
                        metrics.append({"cohort": name, "context": local, "snr_db": settings["snrs_db"][local],
                            "method": method, "scene": scene, "geometry_id": geometry_id,
                            "nmse_linear": float(values[scene]), "mse_physical": float(physical_mse[scene]),
                            "observation_residual": float(residual[scene]), "selected_strength": choice})
                gamma = predicted["gamma"]
                mass = gamma.gather(1, predicted["model"]["indices"]).sum(1) / gamma.sum(1)
                for scene, geometry_id in enumerate(cohort_ids):
                    diagnostics.append({"cohort": name, "snr_db": settings["snrs_db"][local], "geometry_id": geometry_id,
                        "base_kkt": float(predicted["base_optimality_residual"][scene]),
                        "coherent_spatial_variance_mass": float(mass[scene]),
                        "mean_retained_delay_mass": float(predicted["model"]["weights"][scene].sum(-1).mean())})
                log("evaluation_context", cohort=name, snr_db=settings["snrs_db"][local], scenes=len(cohort_ids))

        evaluate_cohort("new", new_observations)
        if settings["historical_screen_geometries"]:
            historical = torch.load(parent / "predictions/observations.pt", weights_only=True, map_location="cpu")
            assert historical["frozen_sha256"] == file_hash(parent / "frozen.json")
            n = settings["historical_screen_geometries"]
            old_entries = [{key: value[:n] if isinstance(value, torch.Tensor) and value.ndim else value for key, value in historical["entries"][i].items()} for i in indices]
            old = {"entries": old_entries, "geometry_ids": historical["geometry_ids"][:n],
                "frozen_sha256": file_hash(path / "frozen.json"), "previously_used_for_research": True,
                "observation_hashes": [tensor_hash(historical["entries"][i]["observation"]) for i in indices]}
            assert not set(old["geometry_ids"]) & set(ids)
            save_tensor(path / "predictions/historical_observations.pt", old)
            provenance["historical_observations_sha256"] = file_hash(parent / "predictions/observations.pt")
            evaluate_cohort("historical", old, old)
        write_csv(export / "metrics.csv", metrics)
        write_csv(export / "diagnostics.csv", diagnostics)
        write_json(path / "parent_provenance.json", provenance)
        shutil.copyfile(path / "parent_provenance.json", export / "parent_provenance.json")
        status({"status": "complete", "run_id": settings["run_id"], "seconds": time.time() - started,
            "new_geometries": len(ids), "historical_geometries": settings["historical_screen_geometries"],
            "implementation_only": smoke, "frozen_before_evaluation": True})
        log("complete", seconds=time.time() - started, metrics=len(metrics))
    except Exception as error:
        status({"status": "failed", "seconds": time.time() - started, "error": repr(error)})
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
