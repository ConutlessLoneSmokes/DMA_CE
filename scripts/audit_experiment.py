"""Independently check saved pilot metrics, provenance, splits, and one sampler replay."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.conditional import ConditionalCNN, VARIANTS
from dma_ce.data import sample_scenes, tensor_hash
from dma_ce.experiment import file_hash, predict_entry, write_csv, write_json


def audit(run_id):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("invalid run ID")
    run, result = ROOT / "runs" / run_id, ROOT / "results" / run_id
    status = json.loads((result / "status.json").read_text())
    config = json.loads((result / "config_resolved.json").read_text())
    frozen = json.loads((result / "frozen_validation.json").read_text())
    assert status["status"] == "complete"
    assert file_hash(run / "data.pt") == status["data_sha256"] == frozen["data_sha256"]
    assert file_hash(result / "frozen_validation.json") == status["frozen_validation_sha256"]
    assert (run / "frozen_validation.json").read_bytes() == (result / "frozen_validation.json").read_bytes()
    manifest = json.loads((result / "source_manifest.json").read_text())
    version = json.loads((result / "version.json").read_text())
    assert file_hash(result / "source.zip") == version["source_zip_sha256"]
    with zipfile.ZipFile(result / "source.zip") as archive:
        assert set(archive.namelist()) == set(manifest)
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    progress = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    assert max(x["elapsed_seconds"] for x in progress if x["event"] == "frozen") < min(
        x["elapsed_seconds"] for x in progress if x["event"] == "screen")
    assert frozen["selection_split"] == "validation" and not frozen["screening_used"]
    cache = torch.load(run / "data.pt", map_location="cpu", weights_only=True)
    predictions = torch.load(run / "predictions.pt", map_location="cpu", weights_only=True)
    scene_sets = []
    for index, split in enumerate(("train", "validation", "screen")):
        radius, angle, gain = sample_scenes(config, index, config["data"][split])
        packed = np.concatenate((radius, angle, gain.real, gain.imag), axis=1)
        assert hashlib.sha256(packed.tobytes()).hexdigest() == cache["geometry_hashes"][split]
        scene_sets.append({row.tobytes() for row in packed})
    assert all(not a.intersection(b) for i, a in enumerate(scene_sets) for b in scene_sets[i + 1:])
    train = cache["splits"]["train"]
    mse = torch.stack([(x["truth"] - x["coarse"]).abs().square().mean() for x in train]).mean()
    predicted_variance = torch.stack([x["reliability"].mean() for x in train]).mean()
    assert np.isclose(cache["reliability_scale"], float(mse / predicted_variance))
    assert np.isclose(cache["residual_real_scale"], float((mse / 2).sqrt()))
    assert np.isclose(cache["dc_reference_complex_variance"], float(mse))
    assert np.isclose(float(torch.stack([x["truth"].abs().square().mean() for x in train]).mean()), 2.)
    with (result / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(cache["contexts"]) * config["data"]["screen"] * 11
    assert len(rows) == expected == status["metrics_rows"]
    lookup = {(int(r["context"]), r["method"], int(r["scene"])): r for r in rows}
    assert len(lookup) == expected
    maximum_metric_difference = 0.
    decomposition = []
    for context, saved in enumerate(predictions):
        entry = cache["splits"]["screen"][context]
        assert len(saved) == 11
        residual = entry["truth"] - entry["coarse"]
        energy = entry["truth"].abs().square().sum((1, 2))
        baseline = residual.abs().square().sum((1, 2)) / energy
        for method, estimate in saved.items():
            assert estimate.shape == entry["truth"].shape and torch.isfinite(estimate).all()
            errors = (estimate - entry["truth"]).abs().square().sum((1, 2)) / entry["truth"].abs().square().sum((1, 2))
            if method != "coarse":
                correction = estimate - entry["coarse"]
                added = correction.abs().square().sum((1, 2)) / energy
                aligned = 2 * (correction.conj() * residual).real.sum((1, 2)) / energy
                assert torch.allclose(added - aligned, errors - baseline, rtol=1e-4, atol=1e-5)
                decomposition.append({"context": context, "method": method,
                    "coarse_mean_nmse_linear": float(baseline.mean()), "correction_energy_normalized": float(added.mean()),
                    "twice_alignment_normalized": float(aligned.mean()), "change_mean_nmse_linear": float((errors - baseline).mean())})
            for scene, error in enumerate(errors.tolist()):
                row = lookup[context, method, scene]
                assert row["observation_sha256"] == tensor_hash(entry["observation"][scene])
                maximum_metric_difference = max(maximum_metric_difference, abs(error - float(row["nmse_linear"])))
                assert np.isclose(error, float(row["nmse_linear"]), rtol=2e-6, atol=1e-7)
                assert np.isclose(float(row["row_nmse_linear"]) + float(row["null_nmse_linear"]), error, rtol=2e-4, atol=2e-4)
    parameters, training_seconds = {}, {}
    for variant in VARIANTS:
        model_path = run / "models" / f"{variant}.pt"
        assert file_hash(model_path) == frozen["models"][variant]
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        assert checkpoint["training_steps"] == config["learning"]["steps"]
        parameters[variant], training_seconds[variant] = checkpoint["parameters"], checkpoint["seconds"]
    assert len(set(parameters.values())) == 1
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    model = ConditionalCNN(config["learning"]["width"]).to(device)
    checkpoint = torch.load(run / "models/reliable_residual.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    batch = config["evaluation"]["batch_size"]
    entry = cache["splits"]["screen"][0]
    replay_entry = {key: value[:batch] if torch.is_tensor(value) and value.ndim else value for key, value in entry.items()}
    replay, _ = predict_entry(config, cache, model, "reliable_residual", replay_entry, device, seed=config["seed"] + 17000)
    replay_difference = float((replay - predictions[0]["reliable_residual"][:batch]).abs().max())
    assert torch.allclose(replay, predictions[0]["reliable_residual"][:batch], atol=2e-6, rtol=2e-6)
    write_csv(result / "residual_decomposition.csv", decomposition)
    report = {"status": "passed", "metrics_recomputed": expected, "observation_hashes_checked": expected,
        "disjoint_geometry_splits": True, "training_only_scales_recomputed": True,
        "freeze_before_screen": True, "source_zip_and_model_hashes_verified": True,
        "maximum_metric_difference": maximum_metric_difference, "sampler_replay_maximum_difference": replay_difference,
        "posthoc_residual_decomposition_checked": True,
        "model_parameters": parameters, "training_seconds": training_seconds,
        "sbl_screen_converged": sum(int(x["sbl_converged"].sum()) for x in cache["splits"]["screen"]),
        "sbl_screen_total": len(cache["contexts"]) * config["data"]["screen"]}
    write_json(result / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit(sys.argv[1])
