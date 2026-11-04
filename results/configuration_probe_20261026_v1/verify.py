"""Recompute saved scores and replay hardware/LS without rerunning SBL."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce._upstream.atmosphere import C0_M_PER_S
from dma_ce._upstream.dma import dma_weight_matrix


run = ROOT / "runs/configuration_probe_20261026_v1"
out = Path(__file__).resolve().parent
config = json.loads((run / "config_resolved.json").read_text())
frozen = json.loads((run / "frozen.json").read_text())
assert json.loads((run / "status.json").read_text())["status"] == "complete"
cache = torch.load(run / "inputs/physical_parent.pt", weights_only=True, map_location="cpu")
banks = torch.load(run / "inputs/configurations.pt", weights_only=True, map_location="cpu")
settings, system = config["probe"], config["system"]
assert torch.equal(banks["random"], cache["state_bank"])
assert frozen["selection_split"] == "validation" and not frozen["screening_used"]
assert not set(frozen["validation_geometry_ids"]) & set(cache["geometry_ids"]["train"])
assert hashlib.sha256((run / "inputs/configurations.pt").read_bytes()).hexdigest() == frozen["configuration_cache_sha256"]
elements, strips = system["antennas"] // system["rf_chains"], system["rf_chains"]
grid = np.linspace(system["carrier_hz"] - system["tuning_bandwidth_hz"] / 2,
    system["carrier_hz"] + system["tuning_bandwidth_hz"] / 2, elements)
assert all(np.array_equal(np.sort(strip), grid)
    for state in banks["balanced"].numpy().reshape(-1, strips, elements) for strip in state)
frequencies = system["carrier_hz"] + (np.arange(system["tones"]) - (system["tones"] - 1) / 2) * system["bandwidths_hz"][0] / system["tones"]
feed = np.tile(np.arange(elements) * system["spacing_wavelengths"] * C0_M_PER_S / system["carrier_hz"], strips)
response = np.stack([dma_weight_matrix(frequencies, row, feed,
    quality_factor=system["quality_factor"], reference_frequency_hz=system["carrier_hz"],
    waveguide_effective_index=system["waveguide_effective_index"],
    waveguide_end_power_fraction=system["waveguide_end_power_fraction"])
    for row in banks["balanced"].numpy()], axis=1)
raw = np.zeros((system["tones"], len(banks["balanced"]), strips, system["antennas"]), dtype=complex)
for strip in range(strips):
    span = slice(strip * elements, (strip + 1) * elements)
    raw[:, :, strip, span] = response[:, :, span]
raw = raw.reshape(system["tones"], -1, system["antennas"])
gain = (abs(raw) ** 2).sum(-1)
gain += cache["contexts"][0]["thermal_budget"]["receiver_noise_fraction"]
assert np.allclose(gain, banks["balanced_noise_gain"].numpy(), rtol=1e-12, atol=1e-12)
matrix_error = float(abs(raw.conj() / np.sqrt(gain)[..., None] - banks["balanced_matrix"].numpy()).max())
assert matrix_error < 1e-7
metrics = list(csv.DictReader((out / "metrics.csv").open()))
summary = list(csv.DictReader((out / "summary.csv").open()))
paired = list(csv.DictReader((out / "paired_effects.csv").open()))
draws = np.random.default_rng(settings["candidate_seed"]).integers(0,
    settings["validation_geometries"], (settings["bootstrap_samples"], settings["validation_geometries"]))
maximum_score_error, maximum_noise_error, maximum_ls_error = 0., 0., 0.
for snr in settings["reference_snrs_db"]:
    index = system["snrs_db"].index(snr)
    saved = torch.load(run / f"cache/validation_{index}.pt", weights_only=True, map_location="cpu")
    context = cache["contexts"][index]
    truth = saved["truth"].numpy().astype(complex)
    denominator = (abs(truth) ** 2).sum((1, 2))
    residuals, scores = {}, {}
    for name in ("random", "balanced"):
        a = (context["matrix"] if name == "random" else banks["balanced_matrix"]).numpy().astype(complex)
        observation = saved["observations"][name].numpy().astype(complex)
        residuals[name] = observation - np.einsum("kjn,bkn->bkj", a, truth)
        solution = np.linalg.solve(a @ a.conj().transpose(0, 2, 1), observation.transpose(1, 2, 0))
        replay = (a.conj().transpose(0, 2, 1) @ solution).transpose(2, 0, 1)
        recorded = saved["outputs"][name]["ls"].numpy()
        ls_error = float((abs(replay - recorded) ** 2).sum() / (abs(replay) ** 2).sum())
        maximum_ls_error = max(maximum_ls_error, ls_error)
        assert ls_error < 1e-12
        for method, prediction in saved["outputs"][name].items():
            values = (abs(prediction.numpy().astype(complex) - truth) ** 2).sum((1, 2)) / denominator
            scores[name, method] = values
            rows = [r for r in metrics if r["configuration"] == name and r["method"] == method
                and float(r["reference_snr_db"]) == snr]
            assert [r["geometry_id"] for r in rows] == saved["geometry_ids"]
            error = float(np.max(abs(values - np.array([float(r["nmse_linear"]) for r in rows])) / values))
            maximum_score_error = max(maximum_score_error, error)
            assert error < 1e-12
            row = next(r for r in summary if r["configuration"] == name and r["method"] == method
                and float(r["reference_snr_db"]) == snr)
            assert abs(float(row["mean_nmse_db"]) - 10 * np.log10(values.mean())) < 1e-10
            assert float(row["transmit_power_dbm"]) == context["transmit_power_dbm"]
    noise_error = float(np.linalg.norm(residuals["random"] - residuals["balanced"]) / np.linalg.norm(residuals["random"]))
    maximum_noise_error = max(maximum_noise_error, noise_error)
    assert noise_error < 1e-4
    for method in ("sbl100", "ls", "lmmse"):
        a, b = scores["random", method], scores["balanced", method]
        low, high = np.quantile(10 * np.log10(a[draws].mean(1) / b[draws].mean(1)), [.025, .975])
        row = next(r for r in paired if r["method"] == method and float(r["reference_snr_db"]) == snr)
        assert max(abs(float(row["improvement_db"]) - 10 * np.log10(a.mean() / b.mean())),
            abs(float(row["ci95_low_db"]) - low), abs(float(row["ci95_high_db"]) - high)) < 1e-10
expected = "balanced" if all(float(r["ci95_low_db"]) > 0 for r in paired if r["method"] == "sbl100") else "random"
assert frozen["selected_configuration"] == expected
result = {"status": "passed", "metric_rows": len(metrics), "maximum_relative_score_difference": maximum_score_error,
    "maximum_ls_replay_relative_squared_error": maximum_ls_error, "maximum_paired_white_noise_relative_error": maximum_noise_error,
    "maximum_matrix_absolute_error": matrix_error, "selected_configuration": expected,
    "scope": "Hardware/noise replay, LS normal-equation replay, saved scores, paired intervals and frozen validation selection; no independent screening or SBL rerun"}
(out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
