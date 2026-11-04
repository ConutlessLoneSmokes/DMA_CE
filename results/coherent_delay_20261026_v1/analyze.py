"""Independent saved-score checks and forward Gaussian certificates; no SBL rerun."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dma_ce.physical_study import case_hardware

OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / OUT.name
config = json.loads((RUN / "config_resolved.json").read_text())
settings = config["probe"]
frozen = json.loads((RUN / "frozen.json").read_text())
status = json.loads((RUN / "status.json").read_text())
assert status["status"] == "complete" and not status["implementation_only"]
assert frozen["selection_split"] == "validation" and not frozen["screening_used"]
for name, digest in frozen["files_sha256"].items():
    assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest, name
cache = torch.load(RUN / "inputs/physical_parent.pt", weights_only=True, map_location="cpu")
provenance = json.loads((RUN / "parent_provenance.json").read_text())
indices = provenance["original_context_indices"]
rows = list(csv.DictReader((OUT / "metrics.csv").open()))
assert len(rows) == 5 * (32 * 5 + 16 * 6)


def risks(truth, estimate):
    return (abs(estimate - truth) ** 2).sum((1, 2)) / (abs(truth) ** 2).sum((1, 2))


maximum_score_error, maximum_mse_error = 0., 0.
for local, index in enumerate(indices):
    validation = torch.load(RUN / f"cache/validation_{local}.pt", weights_only=True, map_location="cpu")
    truth = validation["truth"].numpy().astype(complex)
    scores = {strength: float(risks(truth, validation["outputs"]["rho_" + str(strength)].numpy().astype(complex)).mean())
        for strength in settings["strengths"]}
    choice = min(settings["strengths"], key=lambda value: (scores[value], value))
    assert choice == frozen["strength_by_context"][str(local)]
    assert validation["geometry_ids"] == frozen["validation_geometry_ids"]

observations = {}
for cohort in ("new", "historical"):
    observed = torch.load(RUN / f"predictions/{cohort}_observations.pt", weights_only=True, map_location="cpu")
    observations[cohort] = observed
    assert len(set(observed["geometry_ids"])) == len(observed["geometry_ids"])
    assert observed["frozen_sha256"] == hashlib.sha256((RUN / "frozen.json").read_bytes()).hexdigest()
    assert not set(observed["geometry_ids"]) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"])
    if cohort == "new":
        assert observed["created_at_unix"] >= frozen["frozen_at_unix"]
        assert len({entry["physical_noise_sha256"] for entry in observed["entries"]}) == 1
    for local, (index, entry) in enumerate(zip(indices, observed["entries"])):
        saved = torch.load(RUN / f"predictions/{cohort}_{local}.pt", weights_only=True, map_location="cpu")
        assert saved["geometry_ids"] == observed["geometry_ids"]
        context = cache["contexts"][index]
        expected_variance = context["physical_white_noise_per_tone_w"] / (context["pilot_power_per_tone_w"] * cache["channel_scale"] ** 2)
        assert np.allclose(entry["variance"].numpy(), expected_variance, rtol=1e-6)
        truth = entry["truth"].numpy().astype(complex)
        for method, prediction in saved["outputs"].items():
            estimate = prediction.numpy().astype(complex)
            values = risks(truth, estimate)
            mse = (abs(estimate - truth) ** 2).mean((1, 2)) * cache["channel_scale"] ** 2
            local_rows = [r for r in rows if r["cohort"] == cohort and int(r["context"]) == local and r["method"] == method]
            assert [r["geometry_id"] for r in local_rows] == observed["geometry_ids"]
            score_error = np.max(abs(values - [float(r["nmse_linear"]) for r in local_rows]) / values)
            mse_error = np.max(abs(mse - [float(r["mse_physical"]) for r in local_rows]) / mse)
            maximum_score_error = max(maximum_score_error, float(score_error))
            maximum_mse_error = max(maximum_mse_error, float(mse_error))
            assert score_error < 1e-11 and mse_error < 1e-11
assert not set(observations["new"]["geometry_ids"]) & set(observations["historical"]["geometry_ids"])

# This forward covariance action checks the saved inverse solution without another inverse.
_, _, hardware = case_hardware(config, config["cases"][0])
a = cache["contexts"][0]["matrix"].numpy().astype(complex)
d = hardware[0]["dictionary"].numpy().astype(complex)
phi = a @ d
permutation = np.random.default_rng(settings["shuffle_seed"]).permutation(settings["coherent_atoms"])
if np.array_equal(permutation, np.arange(len(permutation))):
    permutation = np.roll(permutation, 1)
certificate_rows = []
for local in (0, 2, 4):
    saved = torch.load(RUN / f"predictions/new_{local}.pt", weights_only=True, map_location="cpu")
    entry = observations["new"]["entries"][local]
    gamma = saved["gamma"][0].numpy().astype(float)
    model = {key: value.numpy() for key, value in saved["model"].items()}
    active, weights = model["indices"][0], model["weights"][0]
    weights = weights / np.maximum(1, weights.sum(-1, keepdims=True))
    envelope, delays = model["envelope"], model["delays"][0]
    choice = frozen["strength_by_context"][str(local)]
    checks = {"rho_0": (0., False, "free"),
        "rho_0.5": (.5, False, "coherent_half"),
        "shuffle_0.5": (.5, True, "shuffled_half"),
        "rho_" + str(choice): (choice, False, "coherent_selected")}
    for name, (strength, shuffled, method) in checks.items():
        z = saved["precision_solutions"][name][0].numpy()
        projection = np.einsum("kmg,km->kg", phi.conj(), z)
        independent = gamma.copy()
        independent[active] *= 1 - strength * weights.sum(-1)
        coefficients = independent[None] * projection
        used_delays = delays[permutation] if shuffled else delays
        for position, atom in enumerate(active):
            phase = np.exp(-2j * np.pi * np.arange(len(a))[:, None] * used_delays[position][None])
            basis = (phase[:, :, None] * envelope[:, None]
                * np.sqrt(strength * gamma[atom] * weights[position])[None, :, None]).reshape(len(a), -1)
            coefficients[:, atom] += basis @ (basis.conj().T @ projection[:, atom])
        predicted_observation = np.einsum("kmg,kg->km", phi, coefficients) + float(entry["variance"][0]) * z
        y = entry["observation"][0].numpy()
        linear_error = float(np.linalg.norm(predicted_observation - y) / np.linalg.norm(y))
        predicted_channel = np.einsum("kng,kg->kn", d, coefficients)
        recorded = saved["outputs"][method][0].numpy()
        posterior_error = float((abs(recorded - predicted_channel) ** 2).sum() / (abs(predicted_channel) ** 2).sum())
        assert linear_error < 1e-6 and posterior_error < 1e-12, (local, name, linear_error, posterior_error)
        certificate_rows.append({"context": local, "method": method, "forward_covariance_residual": linear_error,
            "posterior_relative_squared_error": posterior_error})
del d, phi, hardware


def write_csv(name, values):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


summary, effects = [], []
for cohort, n in (("new", 32), ("historical", 16)):
    rng = np.random.default_rng(settings["screen_seed"] + (1 if cohort == "historical" else 0))
    draws = rng.integers(0, n, (settings["bootstrap_samples"], n))
    comparisons = [("coherent_selected", "sbl100"), ("coherent_selected", "free"),
        ("coherent_half", "free"), ("coherent_half", "shuffled_half"), ("free", "sbl100")]
    if cohort == "historical":
        comparisons += [("coherent_selected", "sbl_full"), ("sbl100", "sbl_full")]
    for snr in settings["snrs_db"]:
        local = [r for r in rows if r["cohort"] == cohort and float(r["snr_db"]) == snr]
        scores = {}
        for method in dict.fromkeys(r["method"] for r in local):
            method_rows = [r for r in local if r["method"] == method]
            values = np.array([float(r["nmse_linear"]) for r in method_rows])
            assert len(values) == n
            scores[method] = values
            low, high = np.quantile(10 * np.log10(values[draws].mean(1)), [.025, .975])
            summary.append({"cohort": cohort, "snr_db": snr, "method": method, "scenes": n,
                "mean_nmse_linear": float(values.mean()), "mean_nmse_db": float(10 * np.log10(values.mean())),
                "nmse_db_ci95_low": float(low), "nmse_db_ci95_high": float(high),
                "mean_observation_residual": float(np.mean([float(r["observation_residual"]) for r in method_rows]))})
        for candidate, reference in comparisons:
            x, y = scores[candidate], scores[reference]
            low, high = np.quantile(10 * np.log10(y[draws].mean(1) / x[draws].mean(1)), [.025, .975])
            effects.append({"cohort": cohort, "snr_db": snr, "candidate": candidate, "reference": reference,
                "improvement_db": float(10 * np.log10(y.mean() / x.mean())),
                "ci95_low_db": float(low), "ci95_high_db": float(high), "scene_win_rate": float(np.mean(x < y))})
write_csv("summary.csv", summary)
write_csv("paired_effects.csv", effects)
write_csv("posterior_certificates.csv", certificate_rows)
verification = {"status": "passed", "metric_rows": len(rows), "maximum_relative_score_error": maximum_score_error,
    "maximum_relative_physical_mse_error": maximum_mse_error, "forward_certificates": certificate_rows,
    "validation_selection_recomputed": True, "new_geometry_independent": True, "cohorts_not_pooled": True,
    "frozen_before_new_observations": True, "scope": "Saved scoring, split/freeze and independent forward Gaussian equations; no SBL rerun or calibrated-delay claim"}
(OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
labels = {"sbl100": "SBL100", "free": "Matched free posterior", "coherent_selected": "Coherent (validation selected)",
    "coherent_half": "Coherent (rho=0.5)", "shuffled_half": "Shuffled delays (rho=0.5)", "sbl_full": "Original converged/budget SBL"}
for ax, cohort, n in zip(axes, ("new", "historical"), (32, 16)):
    methods = ["free", "coherent_selected", "coherent_half", "shuffled_half"] + (["sbl_full"] if cohort == "historical" else [])
    for method in methods:
        local = [r for r in summary if r["cohort"] == cohort and r["method"] == method]
        line, = ax.plot([r["snr_db"] for r in local], [r["mean_nmse_db"] for r in local], "o-", label=labels[method])
        if method in ("free", "coherent_selected"):
            ax.fill_between([r["snr_db"] for r in local], [r["nmse_db_ci95_low"] for r in local],
                [r["nmse_db_ci95_high"] for r in local], color=line.get_color(), alpha=.12)
    ax.set_title(f"{'New independent' if cohort == 'new' else 'Historical paired review'} geometries: {n}")
    ax.set_xlabel("Training-average received SNR (dB)")
    ax.set_xticks(settings["snrs_db"])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Mean scene full-CSI NMSE (dB)")
fig.suptitle("Fixed DMA; scene bootstrap intervals; one fixed spatial-prior update")
fig.tight_layout()
for suffix in ("png", "svg"):
    fig.savefig(OUT / ("coherent_delay." + suffix), dpi=180)
plt.close(fig)
print(json.dumps({"verification": {key: value for key, value in verification.items() if key != "forward_certificates"},
    "new_selected_effects": [r for r in effects if r["cohort"] == "new" and r["candidate"] == "coherent_selected" and r["reference"] == "free"],
    "historical_full_sbl_effects": [r for r in effects if r["cohort"] == "historical" and r["candidate"] == "coherent_selected" and r["reference"] == "sbl_full"]}))
