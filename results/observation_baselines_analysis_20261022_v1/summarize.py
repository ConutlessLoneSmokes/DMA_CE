"""Paired descriptive comparisons of the completed, frozen observation sweeps."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
rng = np.random.default_rng(20261016 + 819)
indices = rng.integers(128, size=(5000, 128))
effects, endpoints, convergence, choices = [], [], [], []


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


for band in (3, 30):
    folder = root / f"results/observation_baselines_20261022_{band}ghz_v1"
    metrics, summary = read(folder / "metrics.csv"), read(folder / "summary.csv")
    lookup = {(int(r["configurations"]), float(r["snr_db"]), r["method"], int(r["scene"])): r for r in metrics}
    for count in (4, 8, 12, 16, 24, 32, 48, 64):
        for snr in (-10, 0, 15, 30):
            if snr >= 0:
                rows = [r for r in summary if int(r["configurations"]) == count and float(r["snr_db"]) == snr]
                assert min(rows, key=lambda r: float(r["mean_nmse_linear"]))["method"] == "sbl_dense"
            for reference in ("lmmse_shrunk", "lmmse_joint", "lmmse_per_tone", "ls_min_norm"):
                target = np.array([float(lookup[count, snr, "sbl_dense", g]["nmse_linear"]) for g in range(128)])
                baseline = np.array([float(lookup[count, snr, reference, g]["nmse_linear"]) for g in range(128)])
                draws = 10 * np.log10(target[indices].mean(1) / baseline[indices].mean(1))
                low, high = np.quantile(draws, [.025, .975])
                effects.append({"bandwidth_ghz": band, "configurations": count, "snr_db": snr,
                    "target": "sbl_dense", "reference": reference,
                    "delta_db": float(10 * np.log10(target.mean() / baseline.mean())),
                    "ci95_low_db": float(low), "ci95_high_db": float(high),
                    "geometry_blocks": 128, "bootstrap_draws": 5000,
                    "scope": "descriptive_paired_geometry;fixed_training_and_DMA;no_multiplicity_adjustment"})
    for snr in (-10, 0, 15, 30):
        row = {"bandwidth_ghz": band, "snr_db": snr, "configurations": 64}
        for method in ("ls_min_norm", "lmmse_joint", "lmmse_per_tone", "lmmse_shrunk", "sbl_dense"):
            r = next(r for r in summary if int(r["configurations"]) == 64 and float(r["snr_db"]) == snr and r["method"] == method)
            row[method + "_nmse_db"] = float(r["mean_nmse_db"])
            row[method + "_mse_physical"] = float(r["mean_mse_physical"])
        endpoints.append(row)
    for method in {r["method"] for r in summary}:
        for snr in (-10, 0, 15, 30):
            selected = sorted([r for r in summary if r["method"] == method and float(r["snr_db"]) == snr], key=lambda r: int(r["configurations"]))
            if method == "sbl_dense":
                assert all(float(a["mean_nmse_linear"]) > float(b["mean_nmse_linear"]) for a, b in zip(selected, selected[1:]))
                assert all(float(a["mean_mse_physical"]) > float(b["mean_mse_physical"]) for a, b in zip(selected, selected[1:]))
    sbl = [r for r in metrics if r["method"] == "sbl_dense"]
    bad = [r for r in sbl if r["sbl_converged"] != "True"]
    convergence.append({"bandwidth_ghz": band, "outputs": len(sbl), "nonconverged": len(bad),
        "median_updates": float(np.median([int(r["sbl_iterations"]) for r in sbl])),
        "maximum_updates": max(int(r["sbl_iterations"]) for r in sbl),
        "maximum_recorded_kkt": max(float(r["sbl_optimality_residual"]) for r in sbl),
        "nonconverged_rows": bad})
    frozen = json.loads((folder / "frozen.json").read_text())
    selected = next(iter(frozen["selected_frequency_shrinkage"]["physical_absorption"].values()))
    val = read(folder / "validation_scores.csv")
    losses = {strength: float(np.mean([float(r["mean_nmse_linear"]) for r in val if float(r["frequency_shrinkage"]) == strength]))
        for strength in sorted({float(r["frequency_shrinkage"]) for r in val})}
    assert selected == min(losses, key=lambda strength: (losses[strength], -strength))
    choices.append({"bandwidth_ghz": band, "frozen_coefficient": selected, "validation_mean_linear_nmse": losses})

for name, rows in (("paired_effects.csv", effects), ("j64_results.csv", endpoints)):
    with (out / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(out / "summary_checks.json").write_text(json.dumps({"status": "passed", "sbl_mean_nmse_and_mse_decrease_with_J_in_all_8_curves": True,
    "sbl_lowest_mean_nmse_in_all_48_conditions_at_snr_0_15_30": True,
    "convergence": convergence, "validation_selection": choices,
    "summary_ci_schema_note": "Formal summary fields nmse_linear_ci95_low/high contain dB limits; plots correctly use them with mean_nmse_db. Raw exports preserved.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
print(json.dumps({"convergence": convergence, "validation_selection": choices}, indent=2))
for row in effects:
    if row["reference"] == "lmmse_shrunk" and (row["configurations"] == 64 or row["snr_db"] == -10 and row["configurations"] in (16, 24, 32)):
        print(json.dumps(row))
