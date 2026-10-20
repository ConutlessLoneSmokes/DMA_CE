"""Paired dictionary/stopping diagnostics on two fixed, saved validation geometries."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "runs" / OUT.name
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.data import build_hardware, tensor_hash
from dma_ce.sbl import joint_sbl, posterior

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda")
started = time.time()


def log(event, **fields):
    print(json.dumps({"event": event, "seconds": round(time.time() - started, 3), **fields}), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nmse(truth, value):
    return (truth - value).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2))


def initial_gamma(y, sensing, noise):
    power = sensing.abs().square().sum(-1).mean()
    initial = (y.abs().square().mean((1, 2)) - noise).clamp_min(1e-6) / power
    return initial[:, None].expand(-1, sensing.shape[-1]).clone().clamp_min(1e-8)


def em_step(y, sensing, dictionary, noise, gamma):
    mean, variance, _, _ = posterior(y, sensing, dictionary, noise, gamma, channel_moments=False)
    return (.5 * gamma + .5 * (mean.abs().square() + variance).mean(1)).clamp_min(1e-8)


@torch.no_grad()
def trace(y, sensing, dictionary, noise, truth, label, context, steps, *, reference=None):
    gamma = initial_gamma(y, sensing, noise)
    first = {tol: torch.zeros(len(y), dtype=torch.long, device=DEVICE) for tol in (1e-3, 1e-5)}
    records = []
    for step in range(1, max(steps) + 1):
        new = em_step(y, sensing, dictionary, noise, gamma)
        change = (new - gamma).norm(dim=-1) / gamma.norm(dim=-1).clamp_min(1e-12)
        gamma = new
        for tol in first:
            hit = (first[tol] == 0) & (change < tol) & (step >= 5)
            first[tol][hit] = step
        if step in steps:
            _, _, estimate, diagonal = posterior(y, sensing, dictionary, noise, gamma)
            if reference is not None and step == 20:
                assert torch.equal(estimate, reference)
            error = nmse(truth, estimate)
            residual = y - torch.einsum("kjn,bkn->bkj", matrices[context], estimate)
            cov = torch.einsum("kjg,bg,klg->bkjl", sensing, gamma, sensing.conj())
            cov += noise[:, None, None, None] * torch.eye(y.shape[-1], device=DEVICE, dtype=y.dtype)
            chol = torch.linalg.cholesky(cov)
            solved = torch.cholesky_solve(y[..., None], chol).squeeze(-1)
            nll = (2 * chol.diagonal(dim1=-2, dim2=-1).real.log().sum(-1) + (y.conj() * solved).sum(-1).real).mean(1)
            for b in range(len(y)):
                records.append({"method": label, "context": context, "scene": int(data["indices"][b]), "step": step,
                    "nmse_linear": float(error[b]), "observation_mse": float(residual[b].abs().square().mean()),
                    "marginal_nll_per_tone": float(nll[b]), "relative_gamma_change": float(change[b]),
                    "first_below_1e3": int(first[1e-3][b]), "first_below_1e5": int(first[1e-5][b]),
                    "effective_gamma_count": float(gamma[b].sum().square() / gamma[b].square().sum()),
                    "mean_posterior_variance": float(diagonal[b].mean())})
            log("trace", method=label, context=context, step=step, nmse_db=float(10 * error.mean().log10()),
                first_1e3=first[1e-3].tolist(), first_1e5=first[1e-5].tolist())
    return records


def timeout():
    write_json(OUT / "status.json", {"status": "timeout", "seconds": 600})
    os._exit(124)


timer = threading.Timer(600, timeout)
timer.daemon = True
timer.start()
try:
    assert hashlib.sha256((RUN / "input.pt").read_bytes()).hexdigest() == "be98e4df6e50e1a47a53e86036acfdedc54fe9e58cea4fdba3bba1f72bc6e594"
    data = torch.load(RUN / "input.pt", map_location="cpu", weights_only=True)
    config = data["config"]
    write_json(OUT / "config_resolved.json", config)
    write_json(OUT / "input_provenance.json", {**data["provenance"], "scene_indices": data["indices"],
        "input_sha256": hashlib.sha256((RUN / "input.pt").read_bytes()).hexdigest(),
        "trace_contexts": [0, 3, 4, 7], "trace_maximum_steps": 8000, "duplicate_contexts": [3, 7],
        "design": "same observations, truth and noise; fixed first two validation geometries; no screening or retraining"})
    write_json(OUT / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": str(DEVICE), "gpu": torch.cuda.get_device_name(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    sources = [p for folder in ("src", "scripts", "configs") for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    with zipfile.ZipFile(OUT / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(Path(__file__), "diagnose.py")
    write_json(OUT / "source_manifest.json", {**{p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "diagnose.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    hardware = {}
    for band in config["system"]["bandwidths_hz"]:
        for grid in config["research"]["grids"]:
            local = copy.deepcopy(config)
            local["sbl"].update({k: grid[k] for k in ("angle_points", "range_points")})
            hardware[band, grid["name"]] = build_hardware(local, band, data["state_bank"].numpy())
    nested = torch.tensor([2 * a * 13 + 2 * r for a in range(49) for r in range(7)])
    matrices, checks, replay_rows, correlation_rows, traces = {}, [], [], [], []
    for context, known in enumerate(data["contexts"]):
        band, j = known["bandwidth_hz"], known["configurations"]
        matrices[context] = known["matrix"].to(DEVICE)
        coarse = hardware[band, "coarse"]["dictionary"]
        dense = hardware[band, "dense"]["dictionary"]
        assert torch.equal(coarse, dense[:, :, nested])
        assert torch.equal(known["matrix"], hardware[band, "coarse"]["matrix"][:, :j])
        assert torch.equal(known["matrix"], hardware[band, "dense"]["matrix"][:, :j])
        checks.append({"context": context, "nested_max_abs_difference": 0., "matrix_max_abs_difference": 0.})
        entry = data["splits"]["validation"][context]
        y, noise, truth = (entry[k].to(DEVICE) for k in ("observation", "variance", "truth"))
        for label, dictionary, maximum in (("coarse", coarse, 1000), ("dense", dense, 2000)):
            dictionary = dictionary.to(DEVICE)
            sensing = matrices[context] @ dictionary
            estimate, _, info = joint_sbl(y, sensing, dictionary, noise, iterations=maximum, tolerance=1e-3)
            saved = data["references"][label][context]["coarse"].to(DEVICE)
            difference = float((estimate - saved).abs().max())
            assert torch.allclose(estimate, saved, atol=2e-4, rtol=2e-4), (label, context, difference)
            for b, error in enumerate(nmse(truth, estimate).tolist()):
                replay_rows.append({"context": context, "method": label, "scene": data["indices"][b],
                    "nmse_linear": error, "iterations": int(info["iterations"][b]), "converged": bool(info["converged"][b]),
                    "maximum_replay_abs_difference": difference, "observation_sha256": tensor_hash(entry["observation"][b])})
            nr = 7 if label == "coarse" else 13
            for space, atoms in (("channel", dictionary), ("observation", sensing)):
                indices = torch.tensor([g for g in range(atoms.shape[-1] - 1) if g % nr < nr - 1], device=DEVICE)
                a, b = atoms[..., indices], atoms[..., indices + 1]
                corr = (a.conj() * b).sum(-2).abs() / (a.abs().square().sum(-2) * b.abs().square().sum(-2)).sqrt()
                correlation_rows.append({"context": context, "grid": label, "space": space, "adjacent_parameter": "inverse_range",
                    "p10_correlation": float(corr.quantile(.1)), "median_correlation": float(corr.median()), "p90_correlation": float(corr.quantile(.9))})
            if context in (0, 3, 4, 7):
                reference = joint_sbl(y, sensing, dictionary, noise, iterations=20, tolerance=1e-12)[0]
                traces += trace(y, sensing, dictionary, noise, truth, label, context,
                    (20, 100, 500, 1000, 2000, 4000, 8000), reference=reference)
            if label == "coarse" and context in (3, 7):
                repeated_d, repeated_a = dictionary.repeat(1, 1, 4), sensing.repeat(1, 1, 4)
                repeated, _, repeated_info = joint_sbl(y, repeated_a, repeated_d, noise, iterations=2000, tolerance=1e-3)
                for b, error in enumerate(nmse(truth, repeated).tolist()):
                    replay_rows.append({"context": context, "method": "coarse_four_identical_copies", "scene": data["indices"][b],
                        "nmse_linear": error, "iterations": int(repeated_info["iterations"][b]), "converged": bool(repeated_info["converged"][b]),
                        "maximum_replay_abs_difference": 0., "observation_sha256": tensor_hash(entry["observation"][b])})
                traces += trace(y, repeated_a, repeated_d, noise, truth, "coarse_four_identical_copies", context,
                    (20, 100, 500, 1000, 2000, 4000, 8000))
        write_csv("default_replays.csv", replay_rows)
        write_csv("forced_iteration_traces.csv", traces)
        write_csv("adjacent_atom_correlations.csv", correlation_rows)
        log("context_complete", context=context)
    # Exact duplicate columns preserve the initial prior covariance but slow this EM update by four.
    context = 3
    entry = data["splits"]["validation"][context]
    y, noise = entry["observation"].to(DEVICE, dtype=torch.complex128), entry["variance"].to(DEVICE, dtype=torch.float64)
    dictionary = hardware[data["contexts"][context]["bandwidth_hz"], "coarse"]["dictionary"].to(DEVICE, dtype=torch.complex128)
    sensing = matrices[context].to(torch.complex128) @ dictionary
    gamma = initial_gamma(y, sensing, noise)
    repeated_d, repeated_a = dictionary.repeat(1, 1, 4), sensing.repeat(1, 1, 4)
    split = gamma.repeat(1, 4) / 4
    ordinary = em_step(y, sensing, dictionary, noise, gamma)
    expanded = em_step(y, repeated_a, repeated_d, noise, split).reshape(len(y), 4, -1).sum(1)
    expected = gamma + (ordinary - gamma) / 4
    assert torch.allclose(expanded, expected, atol=1e-12, rtol=1e-10)
    before = posterior(y, sensing, dictionary, noise, gamma)[2]
    duplicate_before = posterior(y, repeated_a, repeated_d, noise, split)[2]
    assert torch.allclose(before, duplicate_before, atol=1e-10, rtol=1e-10)
    write_json(OUT / "checks.json", {"nested_grids": checks, "trace_matches_core_20_updates_exactly": True,
        "duplicate_initial_channel_max_difference": float((before - duplicate_before).abs().max()),
        "duplicate_aggregate_update_max_difference": float((expanded - expected).abs().max()),
        "duplicate_change_ratio": ((expanded - gamma).norm(dim=-1) / (ordinary - gamma).norm(dim=-1)).tolist(),
        "original_gamma_tolerance": 1e-3, "forced_iterations": 8000, "diagnostic_only": True})
    write_json(OUT / "status.json", {"status": "complete", "seconds": time.time() - started, "geometries": 2,
        "default_replay_contexts": 8, "forced_iteration_contexts": 4, "trace_rows": len(traces)})
    log("complete", trace_rows=len(traces))
except Exception as error:
    write_json(OUT / "status.json", {"status": "failed", "error": str(error), "seconds": time.time() - started})
    raise
finally:
    timer.cancel()
