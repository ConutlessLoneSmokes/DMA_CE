"""Paired error decomposition and fixed phase controls on archived observations."""
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
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from dma_ce.coherent import coherent_posterior
from dma_ce.data import sync, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.physical_study import archive, case_hardware


def decomposition(truth, reference, candidate, basis):
    """All orthogonal components use the full truth energy as denominator."""
    if (truth.shape != reference.shape or truth.shape != candidate.shape or truth.ndim != 3
            or basis.ndim != 3 or basis.shape[0] != truth.shape[1]
            or basis.shape[-1] != truth.shape[-1]
            or not all(torch.isfinite(v).all() for v in (truth, reference, candidate, basis))):
        raise ValueError("finite aligned complex channels and a per-tone row basis required")
    energy = truth.abs().square().sum((1, 2))
    if (energy <= 0).any():
        raise ValueError("positive complete truth energy required")
    residual, change = truth - reference, candidate - reference
    row_r = torch.einsum("kmn,bkn->bkm", basis, residual)
    row_d = torch.einsum("kmn,bkn->bkm", basis, change)
    null_r = residual - torch.einsum("kmn,bkm->bkn", basis.conj(), row_r)
    null_d = change - torch.einsum("kmn,bkm->bkn", basis.conj(), row_d)
    result = {}
    for name, r, d in (("full", residual, change), ("row", row_r, row_d), ("null", null_r, null_d)):
        before = r.abs().square().sum((1, 2)) / energy
        correction = d.abs().square().sum((1, 2)) / energy
        alignment = 2 * (r.conj() * d).real.sum((1, 2)) / energy
        after = (r - d).abs().square().sum((1, 2)) / energy
        torch.testing.assert_close(after, before + correction - alignment, rtol=1e-9, atol=1e-12)
        result[name] = {"before": before, "after": after, "correction": correction,
            "alignment": alignment, "gain": before - after,
            "cosine": alignment / (2 * (before * correction).sqrt()).clamp_min(1e-30)}
    for key in ("before", "after", "correction", "alignment", "gain"):
        torch.testing.assert_close(result["full"][key], result["row"][key] + result["null"][key],
            rtol=1e-8, atol=1e-12)
    return result


def prepare_controls(y, matrix, dictionary, noise, gamma, model, strength):
    """Prepare unchanged marginal factors once for all delay controls."""
    dtype, device = torch.complex128, y.device
    powers = gamma.to(device, torch.float64)
    indices = model["indices"]
    weights = model["weights"] / model["weights"].sum(-1, keepdim=True).clamp_min(1)
    independent = powers.clone()
    independent.scatter_(1, indices, powers.gather(1, indices) * (1-strength*weights.sum(-1)))
    covariance, cross, selected_dictionary, selected_sensing = [], [], [], []
    # Chunk the full dictionary to avoid full-size double-precision GPU temporaries.
    for first in range(0, len(matrix), 4):
        d = dictionary[first:first+4].to(device, dtype)
        phi = matrix[first:first+4] @ d
        weighted = phi[None] * independent[:, None, None]
        covariance.append(weighted @ phi.mH + noise.to(torch.float64)[:, None, None, None]
            * torch.eye(matrix.shape[-2], dtype=dtype, device=device))
        cross.append(d[None] @ weighted.mH)
        selected_dictionary.append(d[None].expand(len(y), -1, -1, -1).gather(3,
            indices[:, None, None].expand(-1, len(d), d.shape[-2], -1)))
        selected_sensing.append(phi[None].expand(len(y), -1, -1, -1).gather(3,
            indices[:, None, None].expand(-1, len(d), phi.shape[-2], -1)))
    factor = torch.linalg.cholesky(torch.cat(covariance, dim=1))
    return {"factor": factor, "solved": torch.cholesky_solve(y.to(dtype)[..., None], factor).squeeze(-1),
        "cross": torch.cat(cross, dim=1), "dictionary": torch.cat(selected_dictionary, dim=1),
        "sensing": torch.cat(selected_sensing, dim=1),
        "weight_scale": (strength*powers.gather(1, indices)[..., None]*weights).sqrt(),
        "envelope": model["envelope"]}


def control_posterior(prepared, delays):
    batch, tones = prepared["solved"].shape[:2]
    phase = torch.exp(-2j*math.pi*torch.arange(tones, dtype=torch.float64, device=delays.device)[None, :, None, None]*delays[:, None])
    mixing = phase[..., None]*prepared["envelope"][None, :, None, None]*prepared["weight_scale"][:, None, :, :, None]
    low_rank = (prepared["sensing"][..., None, None]*mixing[:, :, None]).flatten(-3)
    inverse = torch.cholesky_solve(low_rank, prepared["factor"])
    small = torch.eye(low_rank.shape[-1], dtype=low_rank.dtype, device=delays.device) + (low_rank.mH @ inverse).sum(1)
    factor = torch.linalg.cholesky((small+small.mH)/2)
    coordinates = torch.cholesky_solve((low_rank.conj()*prepared["solved"][..., None]).sum((1, 2))[..., None], factor).squeeze(-1)
    solved = prepared["solved"] - (inverse @ coordinates[:, None, :, None]).squeeze(-1)
    contribution = (mixing*coordinates.reshape(batch, *mixing.shape[2:])[:, None]).sum((-1,-2))
    return (prepared["cross"] @ solved[..., None]).squeeze(-1) + (prepared["dictionary"] @ contribution[..., None]).squeeze(-1)


def self_check():
    dtype = torch.complex128
    truth = torch.tensor([[[1+2j, 3+4j]]], dtype=dtype)
    zero = torch.zeros_like(truth)
    row = torch.tensor([[[1, 0]]], dtype=dtype)
    good = torch.tensor([[[1+2j, 0]]], dtype=dtype)
    result = decomposition(truth, zero, good, row)
    torch.testing.assert_close(result["row"]["gain"], torch.tensor([1/6], dtype=torch.float64))
    torch.testing.assert_close(result["null"]["gain"], torch.zeros(1, dtype=torch.float64))
    bad = torch.tensor([[[1+2j, -3-4j]]], dtype=dtype)
    assert decomposition(truth, zero, bad, row)["full"]["gain"].item() < 0
    rotated = torch.tensor([[[1, 1j]]], dtype=dtype) / math.sqrt(2)
    projected = rotated.mH @ torch.linalg.solve(rotated @ rotated.mH, rotated @ truth[0, :, :, None])
    result = decomposition(truth, zero, good, rotated)
    torch.testing.assert_close(result["row"]["before"],
        projected.abs().square().sum().reshape(1) / truth.abs().square().sum())
    try:
        decomposition(zero, zero, zero, row)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-energy truth was accepted")
    torch.manual_seed(20261028)
    dictionary = torch.randn(4, 5, 7, dtype=dtype)
    matrix = torch.randn(4, 2, 5, dtype=dtype)
    y = torch.randn(2, 4, 2, dtype=dtype)
    gamma = torch.rand(2, 7, dtype=torch.float64)+.2
    noise = torch.tensor([.1,.3], dtype=torch.float64)
    envelope = torch.randn(4,3,dtype=torch.float64)
    envelope /= envelope.norm(dim=-1,keepdim=True)
    model = {"indices":torch.tensor([[0,2],[1,4]]),
        "weights":torch.tensor([[[.3,.2],[.1,.5]],[[.5,.1],[.2,.4]]],dtype=torch.float64),
        "delays":torch.rand(2,2,2,dtype=torch.float64),"envelope":envelope}
    prepared = prepare_controls(y,matrix,dictionary,noise,gamma,model,.5)
    for delay in (model["delays"], torch.zeros_like(model["delays"]), (model["delays"]+.25).remainder(1)):
        actual = control_posterior(prepared,delay)
        expected = coherent_posterior(y,matrix,dictionary,noise,gamma,{**model,"delays":delay},.5)["mean"]
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def run():
    self_check()
    settings = json.loads((ROOT / "configs/coherent_mechanism.json").read_text())
    for key in ("run_id", "parent_run"):
        if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", settings[key]) is None:
            raise ValueError("safe independent run IDs required")
    if (settings["control_strength"] != .5 or settings["delay_offset_bins"] != [.25, .5, 1.]
            or settings["include_zero_delay"] is not True or not 0 < settings["timeout_seconds"] <= 1800):
        raise ValueError("this diagnostic uses its predeclared fixed controls")
    parent = ROOT / "runs" / settings["parent_run"]
    path, out = ROOT / "runs" / settings["run_id"], ROOT / "results" / settings["run_id"]
    if path.exists() or out.exists():
        raise FileExistsError("use an unused diagnostic run ID")
    if shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise OSError("2 GB free storage required")
    expected = json.loads((parent / "delivery_manifest.json").read_text())
    names = ["config_resolved.json", "frozen.json", "source.zip", "inputs/physical_parent.pt",
        "inputs/absorption_table.txt", *[f"predictions/{c}_observations.pt" for c in ("new", "historical")],
        *[f"predictions/{c}_{i}.pt" for c in ("new", "historical") for i in range(5)]]
    digests = {name: file_hash(parent / name) for name in names}
    assert all(digests[n] == expected[n] for n in names), "parent archive changed"
    for folder in (path, out, path / "export", path / "predictions", path / "inputs"):
        folder.mkdir()
    config = json.loads((parent / "config_resolved.json").read_text())
    config["device"] = settings["device"]
    config["channel"]["absorption_table_path"] = str(parent / "inputs/absorption_table.txt")
    config["mechanism"] = settings
    archive(ROOT, path, path / "export", config)
    write_json(path / "input_provenance.json", {"parent_run": settings["parent_run"],
        "files_sha256": digests, "scope": "post-hoc paired diagnosis; no parameter selection or new geometry"})
    started = time.monotonic()

    def log(event, **fields):
        record = {"event": event, "elapsed_seconds": time.monotonic() - started, **fields}
        with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if time.monotonic() - started > settings["timeout_seconds"]:
            raise TimeoutError("declared computation budget reached")

    def status(value):
        for folder in (path, out):
            write_json(folder / "status.json", value)

    status({"status": "running", "stage": "saved_prediction_decomposition"})
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(settings["device"])
        cache = torch.load(parent / "inputs/physical_parent.pt", weights_only=True, map_location="cpu")
        matrices = [c["matrix"] for c in cache["contexts"]]
        assert all(torch.equal(a, matrices[0]) for a in matrices)
        matrix = matrices[0].to(device, torch.complex128)
        _, singular, basis = torch.linalg.svd(matrix, full_matrices=False)
        threshold = torch.finfo(torch.float64).eps * matrix.shape[-1] * singular[:, :1]
        assert (singular > threshold).all(), "full row rank required for this row basis"
        torch.testing.assert_close(basis @ basis.mH,
            torch.eye(matrix.shape[-2], dtype=matrix.dtype, device=device).expand(matrix.shape[0], -1, -1),
            rtol=1e-9, atol=1e-10)
        observations = {c: torch.load(parent / f"predictions/{c}_observations.pt",
            weights_only=True, map_location="cpu") for c in ("new", "historical")}
        scores = {(r["cohort"], int(r["context"]), r["method"], int(r["scene"])): float(r["nmse_linear"])
            for r in csv.DictReader((parent / "export/metrics.csv").open())}
        rows, summaries = [], []
        maximum_score_error = 0.
        draws = {c: np.random.default_rng(settings["bootstrap_seed"] + i).integers(0, n,
            (settings["bootstrap_samples"], n)) for i, (c, n) in enumerate((("new", 32), ("historical", 16)))}

        def compare(cohort, context, ids, truth, outputs, candidate, reference):
            result = decomposition(truth, outputs[reference], outputs[candidate], basis)
            for region, values in result.items():
                arrays = {key: value.cpu().numpy() for key, value in values.items()}
                for scene, geometry_id in enumerate(ids):
                    rows.append({"cohort": cohort, "context": context,
                        "snr_db": config["probe"]["snrs_db"][context], "candidate": candidate,
                        "reference": reference, "region": region, "scene": scene, "geometry_id": geometry_id,
                        **{key: float(value[scene]) for key, value in arrays.items()}})
                bootstrap = arrays["gain"][draws[cohort]].mean(1)
                low, high = np.quantile(bootstrap, [.025, .975])
                summaries.append({"cohort": cohort, "snr_db": config["probe"]["snrs_db"][context],
                    "candidate": candidate, "reference": reference, "region": region, "scenes": len(ids),
                    **{key: float(value.mean()) for key, value in arrays.items()},
                    "gain_ci95_low": float(low), "gain_ci95_high": float(high),
                    "scene_win_rate": float(np.mean(arrays["gain"] > 0))})

        for cohort, observed in observations.items():
            for context, entry in enumerate(observed["entries"]):
                saved = torch.load(parent / f"predictions/{cohort}_{context}.pt", weights_only=True, map_location="cpu")
                assert saved["geometry_ids"] == observed["geometry_ids"]
                truth = entry["truth"].to(device, torch.complex128)
                outputs = {key: value.to(device, torch.complex128) for key, value in saved["outputs"].items()}
                for method, value in outputs.items():
                    exact = ((truth - value).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2))).cpu()
                    recorded = torch.tensor([scores[cohort, context, method, s] for s in range(len(truth))], dtype=torch.float64)
                    maximum_score_error = max(maximum_score_error, float(((exact - recorded).abs() / exact).max()))
                for candidate, reference in (("coherent_selected", "free"), ("coherent_half", "free"),
                        ("shuffled_half", "free"), ("coherent_half", "shuffled_half")):
                    compare(cohort, context, observed["geometry_ids"], truth, outputs, candidate, reference)
                log("saved_decomposition", cohort=cohort, context=context)
        assert maximum_score_error < 1e-10
        write_csv(out / "decomposition.csv", rows)
        write_csv(out / "summary.csv", summaries)

        status({"status": "running", "stage": "fixed_phase_controls"})
        _, bank, hardware = case_hardware(config, config["cases"][0])
        assert tensor_hash(torch.tensor(bank)) == tensor_hash(cache["state_bank"])
        assert tensor_hash(hardware[0]["matrix"]) == tensor_hash(matrices[0])
        dictionary = hardware[0]["dictionary"]
        replay_error = 0.
        for context, entry in enumerate(observations["new"]["entries"]):
            saved = torch.load(parent / f"predictions/new_{context}.pt", weights_only=True, map_location="cpu")
            truth = entry["truth"].to(device, torch.complex128)
            outputs = {k: v.to(device, torch.complex128) for k, v in saved["outputs"].items()}
            controls = {"zero_delay": 0., **{f"offset_{v:g}_bin": v for v in settings["delay_offset_bins"]}}
            predicted = {key: [] for key in controls}
            for first in range(0, len(truth), settings["batch_size"]):
                last = min(first + settings["batch_size"], len(truth))
                model = {key: (value if key == "envelope" else value[first:last]).to(device)
                    for key, value in saved["model"].items()}
                y, noise = entry["observation"][first:last].to(device), entry["variance"][first:last].to(device)
                gamma = saved["gamma"][first:last].to(device)
                prepared = prepare_controls(y,matrix,dictionary,noise,gamma,model,settings["control_strength"])
                if first == 0:
                    replay = control_posterior(prepared,model["delays"])
                    old = outputs["coherent_half"][first:last]
                    discrepancy = ((replay - old).abs().square().sum((1, 2)) / old.abs().square().sum((1, 2))).max()
                    replay_error = max(replay_error, float(discrepancy))
                    assert discrepancy < 1e-10
                for name, offset in controls.items():
                    delay = (torch.zeros_like(model["delays"]) if name == "zero_delay"
                        else (model["delays"] + offset / truth.shape[1]).remainder(1))
                    result = control_posterior(prepared,delay)
                    predicted[name].append(result.cpu().to(torch.complex64))
                sync(device)
                log("phase_controls", context=context, completed=last, count=len(truth))
            predicted = {key: torch.cat(value) for key, value in predicted.items()}
            torch.save({"outputs": predicted, "geometry_ids": observations["new"]["geometry_ids"],
                "input_prediction_sha256": digests[f"predictions/new_{context}.pt"]}, path / f"predictions/controls_{context}.pt")
            outputs.update({key: value.to(device, torch.complex128) for key, value in predicted.items()})
            for name in controls:
                for reference in ("free", "coherent_half"):
                    compare("new", context, observations["new"]["geometry_ids"], truth, outputs, name, reference)
            write_csv(out / "decomposition.csv", rows)
            write_csv(out / "summary.csv", summaries)
        assert all(file_hash(parent / name) == value for name, value in digests.items())
        write_json(out / "verification.json", {"status": "passed", "maximum_relative_score_error": maximum_score_error,
            "maximum_replay_relative_squared_difference": replay_error, "row_rank_each_tone": matrix.shape[-2],
            "orthogonal_decomposition_checked": True, "hand_calculated_self_check": True,
            "parent_inputs_unchanged": True, "posthoc_existing_geometries": True,
            "bootstrap_seed": settings["bootstrap_seed"], "bootstrap_samples": settings["bootstrap_samples"]})
        for name in ("config_resolved.json", "environment.json", "source_manifest.json", "input_provenance.json"):
            shutil.copyfile(path / name, out / name)
        for name in ("decomposition.csv", "summary.csv", "verification.json"):
            shutil.copyfile(out / name, path / "export" / name)
        status({"status": "complete", "elapsed_seconds": time.monotonic() - started,
            "scene_region_rows": len(rows), "new_controls": list(controls)})
        log("complete")
    except BaseException as error:
        status({"status": "failed", "error": repr(error), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
        print("hand-calculated complex row/null decomposition checks passed")
    else:
        run()
