"""Post-hoc frozen-model scoring and geometry-block linear-mean comparisons."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.conditional import ConditionalCNN
from dma_ce.data import sample_scenes
from dma_ce.experiment import file_hash, nmse, predict_entry, write_csv, write_json

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
result = Path(__file__).resolve().parent
run = ROOT / "runs" / result.name
config = json.loads((run / "config_resolved.json").read_text())
old_config = json.loads((ROOT / "runs/pilot_20261016_v1/config_resolved.json").read_text())
cache = torch.load(run / "data.pt", map_location="cpu", weights_only=True)
old = torch.load(ROOT / "runs/pilot_20261016_v1/data.pt", map_location="cpu", weights_only=True)
for split in ("train", "validation"):
    assert cache["geometry_hashes"][split] == old["geometry_hashes"][split]
    for current, previous in zip(cache["splits"][split], old["splits"][split]):
        assert torch.equal(current["truth"], previous["truth"])
        assert torch.equal(current["observation"], previous["observation"])
scene_sets = []
for settings in (old_config, config):
    radius, angle, gain = sample_scenes(settings, 2, settings["data"]["screen"])
    packed = np.concatenate((radius, angle, gain.real, gain.imag), axis=1)
    scene_sets.append({row.tobytes() for row in packed})
assert not scene_sets[0].intersection(scene_sets[1])
assert cache["geometry_hashes"]["screen"] != old["geometry_hashes"]["screen"]
with (result / "metrics.csv").open(newline="") as stream:
    metrics = list(csv.DictReader(stream))
methods = ("coarse", "deterministic_no_reliability", "deterministic", "deterministic_shuffled")
values = {}
for method in methods:
    selected = sorted((r for r in metrics if r["method"] == method), key=lambda r: (int(r["scene"]), int(r["context"])))
    values[method] = np.array([float(r["nmse_linear"]) for r in selected]).reshape(128, 8)
rng = np.random.default_rng(config["seed"] + 872)
indices = rng.integers(0, 128, (config["evaluation"]["bootstrap_samples"], 128))
effects = []
for scope, columns in (("all", range(8)), ("3GHz", range(4)), ("30GHz", range(4, 8))):
    for target, reference in (("deterministic", "deterministic_no_reliability"),
            ("deterministic_no_reliability", "coarse"), ("deterministic", "coarse"),
            ("deterministic_shuffled", "deterministic")):
        a, b = values[target][:, list(columns)].mean(1), values[reference][:, list(columns)].mean(1)
        bootstrap = 10 * np.log10(a[indices].mean(1) / b[indices].mean(1))
        lo, hi = np.quantile(bootstrap, [.025, .975])
        effects.append({"scope": scope, "target": target, "reference": reference,
            "change_db_of_linear_means": float(10 * np.log10(a.mean() / b.mean())),
            "ci95_low_db": float(lo), "ci95_high_db": float(hi),
            "interpretation": "descriptive_single_training_seed_fixed_shuffle"})
write_csv(result / "linear_mean_effects.csv", effects)
device = torch.device(config["device"])
models = {}
for variant in ("deterministic_no_reliability", "deterministic"):
    model = ConditionalCNN(config["learning"]["width"]).to(device)
    checkpoint = torch.load(run / "models" / f"{variant}.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    models[variant] = model.eval()
split_rows = []
for split in ("train", "validation"):
    for context, entry in enumerate(cache["splits"][split]):
        estimates = {"coarse": entry["coarse"]}
        for variant, model in models.items():
            estimates[variant], _ = predict_entry(config, cache, model, variant, entry, device, seed=config["seed"])
        for method, estimate in estimates.items():
            error = nmse(entry["truth"], estimate)
            assert torch.isfinite(error).all()
            split_rows.append({"split": split, "context": context, "method": method,
                "samples": len(error), "mean_nmse_db": float(10 * error.mean().log10()),
                "mean_complex_mse": float((entry["truth"] - estimate).abs().square().mean())})
write_csv(result / "split_metrics.csv", split_rows)
selection_path = ROOT / "results" / config["sbl_budget_selection"]["run_id"] / "selection.json"
assert file_hash(selection_path) == config["sbl_budget_selection"]["selection_sha256"]
with (selection_path.parent / "summary.csv").open(newline="") as stream:
    profile = [row for row in csv.DictReader(stream) if int(row["budget"]) == config["sbl"]["iterations"]]
for row in profile:
    actual = next(x for x in split_rows if x["split"] == "validation" and x["method"] == "coarse" and x["context"] == int(row["context"]))
    assert abs(actual["mean_nmse_db"] - float(row["mean_nmse_db"])) < 1e-4
write_json(result / "partition_audit.json", {"preserved_train_and_validation_geometry": True,
    "preserved_train_and_validation_observations": True, "old_new_screen_geometry_overlap": 0,
    "new_screen_geometry_sha256": cache["geometry_hashes"]["screen"],
    "old_screen_geometry_sha256": old["geometry_hashes"]["screen"],
    "budget_selection_bound_and_matched": True, "analysis_status": "posthoc_fixed_models_no_selection",
    "analysis_source_sha256": file_hash(Path(__file__))})
print(json.dumps({"overall_linear_mean_effects": [row for row in effects if row["scope"] == "all"],
    "split_diagnostic_rows": len(split_rows), "partition_audit": "passed"}, indent=2))
