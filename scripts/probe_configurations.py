"""One bounded, validation-only choice between two fixed feasible DMA banks."""
import copy
import json
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

from dma_ce._upstream.dma import frequency_tiled_resonances
from dma_ce.data import build_hardware, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.factor_lmmse import apply as apply_lmmse, fit as fit_lmmse
from dma_ce.physical_study import archive
from dma_ce.research import save_tensor
from dma_ce.sbl import joint_sbl


def balanced_bank(system, seed):
    strips, elements = system["rf_chains"], system["antennas"] // system["rf_chains"]
    return np.stack([np.concatenate([frequency_tiled_resonances(elements,
        system["carrier_hz"], system["tuning_bandwidth_hz"], seed=seed + j * strips + r)
        for r in range(strips)]) for j in range(max(system["configurations"]))])


def risk(truth, prediction):
    return ((prediction.to(torch.complex128) - truth.to(torch.complex128)).abs().square().sum((1, 2))
        / truth.to(torch.complex128).abs().square().sum((1, 2)))


def choose(lower_bounds):
    return "balanced" if all(value > 0 for value in lower_bounds) else "random"


def self_check():
    system = {"rf_chains": 2, "antennas": 16, "configurations": [3],
        "carrier_hz": 300e9, "tuning_bandwidth_hz": 30e9}
    bank = balanced_bank(system, 123)
    expected = np.linspace(285e9, 315e9, 8)
    assert bank.shape == (3, 16) and np.array_equal(bank, balanced_bank(system, 123))
    assert all(np.array_equal(np.sort(strip), expected) for row in bank for strip in row.reshape(2, 8))
    assert not np.array_equal(bank[0], bank[1])
    truth = torch.ones((2, 3, 4), dtype=torch.complex64)
    assert torch.equal(risk(truth, 2 * truth), torch.ones(2, dtype=torch.float64))
    assert choose([.1, .2, .3]) == "balanced" and choose([.1, 0., -.1]) == "random"
    print("Candidate, score and conservative selection self-checks passed", flush=True)


def run():
    settings = json.loads((ROOT / "configs/configuration_probe.json").read_text())
    if (any(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", settings[k]) is None for k in ("run_id", "parent_run"))
            or settings["reference_snrs_db"] != [-10, 10, 30]
            or settings["validation_geometries"] != 32 or settings["sbl_iterations"] != 100
            or settings["bootstrap_samples"] != 2000 or settings["timeout_seconds"] != 600):
        raise ValueError("This one-off probe requires the declared two-candidate budget")
    parent = ROOT / "runs" / settings["parent_run"]
    if json.loads((parent / "status.json").read_text())["status"] != "complete":
        raise ValueError("Complete parent run required")
    original = json.loads((parent / "config_resolved.json").read_text())
    frozen_parent = json.loads((parent / "frozen.json").read_text())
    indices = [original["system"]["snrs_db"].index(snr) for snr in settings["reference_snrs_db"]]
    used = ["cache/physical.pt", *[f"cache/coarse_validation_{i}.pt" for i in indices],
        *[f"cache/lmmse_{i}.pt" for i in indices]]
    for name in used:
        if file_hash(parent / name) != frozen_parent["files_sha256"][name]:
            raise ValueError(f"Frozen parent input changed: {name}")
    config = copy.deepcopy(original)
    config.update(label="bounded_configuration_validation_only", device=settings["device"], probe=settings)
    path = ROOT / "runs" / settings["run_id"]
    if path.exists() or shutil.disk_usage(path.parent).free < 2_000_000_000:
        raise ValueError("Unused run ID and 2 GB free storage required")
    path.mkdir()
    export = path / "export"
    for name in ("inputs", "cache", "export"):
        (path / name).mkdir()
    archive(ROOT, path, export, config)
    shutil.copyfile(parent / "cache/physical.pt", path / "inputs/physical_parent.pt")
    write_json(path / "parent_provenance.json", {"parent_run": settings["parent_run"],
        "files_sha256": {name: file_hash(parent / name) for name in used},
        "parent_frozen_sha256": file_hash(parent / "frozen.json")})

    def status(value):
        write_json(path / "status.json", value)
        write_json(export / "status.json", value)

    def log(event, **fields):
        record = {"event": event, "at_unix": time.time(), **fields}
        with (path / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def timeout():
        status({"status": "timeout", "seconds": settings["timeout_seconds"],
            "selected_configuration": "random", "reason": "Budget exhausted; no completed comparison"})
        os._exit(124)

    timer = threading.Timer(settings["timeout_seconds"], timeout)
    timer.daemon = True
    started = time.time()
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(settings["device"])
        cache = torch.load(path / "inputs/physical_parent.pt", weights_only=True, map_location="cpu")
        ids = cache["geometry_ids"]["validation"][:settings["validation_geometries"]]
        assert len(set(ids)) == len(ids) and not set(ids) & set(cache["geometry_ids"]["train"])
        selected = copy.deepcopy(original)
        selected["channel"].update({key: original["cases"][0][key] for key in ("model", "include_absorption")})
        bank = balanced_bank(selected["system"], settings["candidate_seed"])
        status({"status": "validation", "run_id": settings["run_id"]})
        hardware = build_hardware(selected, selected["system"]["bandwidths_hz"][0], bank)
        matrix = hardware["matrix"].to(device)
        dictionary = hardware["dictionary"].to(device)
        original_matrix = cache["contexts"][0]["matrix"].to(device)
        assert matrix.shape == original_matrix.shape and not torch.equal(matrix, original_matrix)
        assert hardware["thermal_budget"] == cache["contexts"][0]["thermal_budget"]
        training = cache["training"][0].to(device)
        inverse = {"random": torch.linalg.pinv(original_matrix.to(torch.complex128)),
            "balanced": torch.linalg.pinv(matrix.to(torch.complex128))}
        save_tensor(path / "inputs/configurations.pt", {"random": cache["state_bank"],
            "balanced": torch.tensor(bank), "balanced_matrix": matrix.cpu(),
            "balanced_noise_gain": hardware["noise_gain"]})
        rows, summary, effects, checks = [], [], [], []
        rng = np.random.default_rng(settings["candidate_seed"])
        draws = rng.integers(0, len(ids), (settings["bootstrap_samples"], len(ids)))
        for index in indices:
            context, entry = cache["contexts"][index], cache["validation"][index]
            assert torch.equal(context["matrix"], cache["contexts"][0]["matrix"])
            truth = entry["truth"][:len(ids)].to(device)
            observed = entry["observation"][:len(ids)].to(device)
            variance = context["noise_complex_variance"]
            noise = observed - torch.einsum("kjn,bkn->bkj", original_matrix, truth)
            new_observed = torch.einsum("kjn,bkn->bkj", matrix, truth) + noise
            # Equal standardized whitened noise pairs configurations under the declared Gaussian model.
            training_power = float(torch.einsum("kjn,bkn->bkj", matrix, training).abs().square().mean())
            assert np.isclose(variance, context["physical_white_noise_per_tone_w"]
                / (context["pilot_power_per_tone_w"] * cache["channel_scale"] ** 2), rtol=1e-6)
            old = torch.load(parent / f"cache/coarse_validation_{index}.pt", weights_only=True, map_location="cpu")
            assert torch.equal(old["observation"][:len(ids)], observed.cpu())
            assert (old["sbl_iterations"][:len(ids)] == settings["sbl_iterations"]).all()
            chunks, solver = [], []
            sensing = matrix @ dictionary
            for first in range(0, len(ids), original["sbl"]["batch_size"]):
                y = new_observed[first:first + original["sbl"]["batch_size"]]
                prediction, _, info = joint_sbl(y, sensing, dictionary,
                    torch.full((len(y),), variance, device=device), iterations=settings["sbl_iterations"],
                    tolerance=original["sbl"]["tolerance"], early_stopping=False)
                chunks.append(prediction.cpu())
                solver.append({key: value.cpu() for key, value in info.items()})
                log("sbl_batch", reference_snr_db=context["snr_db"], completed=min(first + len(y), len(ids)))
            outputs = {"random": {"sbl100": old["coarse"][:len(ids)]},
                "balanced": {"sbl100": torch.cat(chunks)}}
            for name, a, y in (("random", original_matrix, observed), ("balanced", matrix, new_observed)):
                outputs[name]["ls"] = torch.einsum("knj,bkj->bkn", inverse[name], y.to(torch.complex128)).cpu().to(torch.complex64)
                statistics = (torch.load(parent / f"cache/lmmse_{index}.pt", weights_only=True, map_location="cpu")
                    if name == "random" else fit_lmmse(training, a, variance, frozen_parent["frequency_shrinkage"]))
                outputs[name]["lmmse"] = apply_lmmse(statistics, y).cpu()
                del statistics
                for method, prediction in outputs[name].items():
                    values = risk(truth.cpu(), prediction).numpy()
                    for geometry_id, value in zip(ids, values):
                        rows.append({"configuration": name, "reference_snr_db": context["snr_db"],
                            "method": method, "geometry_id": geometry_id, "nmse_linear": float(value)})
                    summary.append({"configuration": name, "reference_snr_db": context["snr_db"],
                        "actual_training_snr_db": context["snr_db"] if name == "random" else float(10 * np.log10(training_power / variance)),
                        "method": method, "mean_nmse_db": float(10 * np.log10(values.mean())),
                        "transmit_power_dbm": context["transmit_power_dbm"], "scenes": len(ids)})
            for method in outputs["random"]:
                a = risk(truth.cpu(), outputs["random"][method]).numpy()
                b = risk(truth.cpu(), outputs["balanced"][method]).numpy()
                differences = 10 * np.log10(a[draws].mean(1) / b[draws].mean(1))
                low, high = np.quantile(differences, [.025, .975])
                effects.append({"reference_snr_db": context["snr_db"], "method": method,
                    "improvement_db": float(10 * np.log10(a.mean() / b.mean())),
                    "ci95_low_db": float(low), "ci95_high_db": float(high)})
            packed = {key: torch.cat([chunk[key] for chunk in solver]) for key in solver[0]}
            checks.append({"reference_snr_db": context["snr_db"],
                "random_nonconverged": int((~old["sbl_converged"][:len(ids)]).sum()),
                "balanced_nonconverged": int((~packed["converged"]).sum())})
            save_tensor(path / f"cache/validation_{index}.pt", {"truth": truth.cpu(),
                "observations": {"random": observed.cpu(), "balanced": new_observed.cpu()},
                "outputs": outputs, "balanced_sbl_info": packed, "geometry_ids": ids})
            log("completed_context", reference_snr_db=context["snr_db"], effects=effects[-3:])
        choice = choose([row["ci95_low_db"] for row in effects if row["method"] == "sbl100"])
        result = {"selected_configuration": choice, "selection_split": "validation", "screening_used": False,
            "selection_rule": settings["selection_rule"], "validation_geometry_ids": ids,
            "selected_bank_sha256": tensor_hash(cache["state_bank"] if choice == "random" else torch.tensor(bank)),
            "candidate_seed": settings["candidate_seed"], "frozen_at_unix": time.time(),
            "configuration_cache_sha256": file_hash(path / "inputs/configurations.pt")}
        write_json(path / "frozen.json", result)
        write_json(export / "frozen.json", result)
        write_csv(export / "metrics.csv", rows)
        write_csv(export / "summary.csv", summary)
        write_csv(export / "paired_effects.csv", effects)
        write_json(export / "solver_checks.json", checks)
        shutil.copyfile(path / "parent_provenance.json", export / "parent_provenance.json")
        status({"status": "complete", "run_id": settings["run_id"], "selected_configuration": choice,
            "seconds": time.time() - started, "selection_only": True, "new_screen_geometries": 0})
        log("frozen_configuration", **result)
    except Exception as error:
        status({"status": "failed", "error": repr(error), "seconds": time.time() - started})
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    self_check()
    if sys.argv[1:] != ["--self-check"]:
        if sys.argv[1:]:
            raise ValueError("Run without arguments, or use --self-check")
        run()
