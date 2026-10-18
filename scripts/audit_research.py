"""Recompute frozen study metrics, observations, source hashes and sampler replays."""
import argparse
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
from dma_ce.data import sample_scenes, tensor_hash
from dma_ce.experiment import apply_dc, file_hash, nmse, write_json
from dma_ce.research import communication_rates, infer, load_model, paired_effects
from dma_ce.suite import STAGES


def geometry(config, split):
    radius, angle, gain = sample_scenes(config, ("train", "validation", "screen").index(split), config["data"][split])
    return np.concatenate((radius, angle, gain.real, gain.imag), axis=1)


def audit(run_id, root=ROOT):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("invalid study ID")
    run, result = root / "runs" / run_id, root / "results" / run_id
    config = json.loads((run / "config_resolved.json").read_text())
    for stage in STAGES:
        assert json.loads((run / "stages" / f"{stage}.json").read_text())["status"] == "complete", stage
    environment = json.loads((run / "environment.json").read_text())
    manifest = json.loads((run / "source_manifest.json").read_text())
    assert file_hash(run / "source.zip") == environment["source_zip_sha256"]
    with zipfile.ZipFile(run / "source.zip") as archive:
        assert set(archive.namelist()) == set(manifest)
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    cache_count = 0
    for meta in (run / "cache").glob("*.json"):
        assert file_hash(meta.with_suffix(".pt")) == json.loads(meta.read_text())["sha256"], meta.name
        cache_count += 1
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows_checked = model_count = replays = scenarios = 0
    maximum_metric_difference = maximum_replay_difference = 0.
    for profile in sorted((result / "final").iterdir()):
        if not profile.is_dir():
            continue
        frozen = json.loads((profile / "frozen_validation.json").read_text())
        assert frozen["selection_split"] == "validation" and not frozen["screening_used"]
        assert file_hash(profile / "frozen_validation.json") == file_hash(run / "stages" / f"frozen_{profile.name}.json")
        data = run / "cache" / f"{frozen['training_cache_id']}.pt"
        assert file_hash(data) == frozen["training_cache_sha256"]
        cache = torch.load(data, map_location="cpu", weights_only=True)
        train = cache["splits"]["train"]
        errors = torch.stack([(e["truth"] - e["coarse"]).abs().square().mean() for e in train]).mean()
        variances = torch.stack([e["reliability"].mean() for e in train]).mean()
        assert np.isclose(float(errors / variances), cache["reliability_scale"])
        assert np.isclose(float((errors / 2).clamp_min(1e-12).sqrt()), cache["residual_real_scale"])
        assert np.isclose(float(torch.stack([e["truth"].abs().square().mean() for e in train]).mean()), 2.)
        train_sets = []
        for split in ("train", "validation"):
            values = geometry(frozen["profile_config"], split)
            assert hashlib.sha256(values.tobytes()).hexdigest() == cache["geometry_hashes"][split]
            train_sets.append({row.tobytes() for row in values})
        assert not train_sets[0].intersection(train_sets[1])
        for item in frozen["models"].values():
            checkpoint = run / item["path"]
            assert file_hash(checkpoint) == item["sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["cache_sha256"] == frozen["training_cache_sha256"]
            assert saved["recipe"] == item["recipe"] and saved["seed"] == item["seed"] and saved["step"] == item["steps"]
            model_count += 1
        for folder in sorted(profile.iterdir()):
            if not folder.is_dir():
                continue
            complete = json.loads((folder / "complete.json").read_text())
            assert complete["status"] == "complete" and complete["scored_at_unix"] > frozen["frozen_at_unix"]
            assert complete["frozen_validation_sha256"] == file_hash(profile / "frozen_validation.json")
            observation_file = run / "predictions" / f"{profile.name}_{folder.name}_observations.pt"
            assert file_hash(observation_file) == complete["observation_file_sha256"]
            observations = torch.load(observation_file, map_location="cpu", weights_only=True)
            packed = geometry(complete["test_config"], "screen")
            assert hashlib.sha256(packed.tobytes()).hexdigest() == observations["geometry_hashes"]["screen"]
            screen_set = {row.tobytes() for row in packed}
            assert all(not screen_set.intersection(s) for s in train_sets)
            assert observations["channel_scale"] == cache["channel_scale"]
            for known, tested in zip(cache["contexts"], observations["contexts"]):
                assert torch.equal(known["matrix"], tested["matrix"])
                assert known["noise_complex_variance"] == tested["noise_complex_variance"]
            for name, expected in complete["prediction_sha256"].items():
                assert file_hash(run / "predictions" / name) == expected, name
            with (folder / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in ("scene", "context", "seed"):
                    row[key] = int(row[key])
                for key in ("nmse_linear", "row_nmse_linear", "null_nmse_linear", "correction_energy_normalized",
                        "twice_alignment_normalized", "net_rate", "gross_rate"):
                    row[key] = float(row[key])
            lookup = {(r["method"], r["seed"], r["context"], r["scene"]): r for r in rows}
            assert len(rows) == len(lookup) == complete["metrics_rows"]
            baselines = {label: torch.load(run / "predictions" / f"{profile.name}_{folder.name}_{label}.pt",
                map_location="cpu", weights_only=True) for label in ("sbl_budget", "sbl_stable", "sbl_dense")}
            current = baselines["sbl_budget"]
            expected_rows = 0
            dc_variance = frozen["dc_multiplier"] * cache["dc_reference_complex_variance"]

            def check(predictions, label, seed, entries):
                nonlocal rows_checked, maximum_metric_difference, expected_rows
                for c, (prediction, entry) in enumerate(zip(predictions, entries)):
                    context = current["contexts"][c]
                    for suffix, strength in (("", 0.), ("_dc", dc_variance)):
                        corrected, _ = apply_dc(prediction, entry, context, strength, torch.device(config["device"]))
                        values = nmse(entry["truth"], corrected)
                        residual, correction = entry["truth"] - entry["coarse"], corrected - entry["coarse"]
                        energy = entry["truth"].abs().square().sum((1, 2)).clamp_min(1e-20)
                        added = correction.abs().square().sum((1, 2)) / energy
                        aligned = 2 * (correction.conj() * residual).real.sum((1, 2)) / energy
                        gross, net, _ = communication_rates(corrected, entry["truth"], context, entry["variance"], config["research"]["communication"])
                        for g, value in enumerate(values.tolist()):
                            row = lookup[label + suffix, seed, c, g]
                            assert row["observation_sha256"] == tensor_hash(entry["observation"][g])
                            delta = abs(row["nmse_linear"] - value)
                            maximum_metric_difference = max(maximum_metric_difference, delta)
                            assert np.isclose(row["nmse_linear"], value, rtol=2e-6, atol=1e-7)
                            assert np.isclose(row["row_nmse_linear"] + row["null_nmse_linear"], value, rtol=2e-4, atol=2e-4)
                            assert np.isclose(row["correction_energy_normalized"], float(added[g]), rtol=2e-5, atol=1e-7)
                            assert np.isclose(row["twice_alignment_normalized"], float(aligned[g]), rtol=2e-5, atol=1e-7)
                            assert np.isclose(row["net_rate"], float(net[g]), rtol=2e-6, atol=1e-7)
                            assert np.isclose(row["gross_rate"], float(gross[g]), rtol=2e-6, atol=1e-7)
                            rows_checked += 1
                            expected_rows += 1

            for label, baseline in baselines.items():
                for a, b in zip(baseline["splits"]["screen"], observations["splits"]["screen"]):
                    assert torch.equal(a["truth"], b["truth"]) and torch.equal(a["observation"], b["observation"])
                for seed in config["research"]["final_seeds"]:
                    entries = baseline["splits"]["screen"]
                    check([e["coarse"] for e in entries], label, seed, entries)
            for item in frozen["models"].values():
                predictions = torch.load(run / "predictions" / f"{profile.name}_{folder.name}_{item['label']}_{item['seed']}.pt",
                    map_location="cpu", weights_only=True)
                for count in item["averages"]:
                    label = item["label"] + (f"_avg{count}" if item["recipe"]["kind"] == "diffusion" else "")
                    check([value[count] for value in predictions], label, item["seed"], current["splits"]["screen"])
                if folder.name == "matched" and item["seed"] == config["research"]["final_seeds"][0]:
                    checkpoint = torch.load(run / item["path"], map_location="cpu", weights_only=True)
                    model = load_model(checkpoint, torch.device(config["device"]))
                    entry = current["splits"]["screen"][0]
                    batch = config["evaluation"]["batch_size"]
                    entry = {k: v[:batch] if torch.is_tensor(v) and v.ndim else v for k, v in entry.items()}
                    replay = infer(config, cache, model, item["recipe"], entry, torch.device(config["device"]),
                        seed=config["seed"] + 17000, averages=item["averages"])
                    for count in item["averages"]:
                        delta = float((replay[count][0] - predictions[0][count][:batch]).abs().max())
                        maximum_replay_difference = max(maximum_replay_difference, delta)
                        assert torch.allclose(replay[count][0], predictions[0][count][:batch], atol=2e-6, rtol=2e-6)
                        replays += 1
                    del model
            assert expected_rows == len(rows)
            expected_effects = paired_effects(rows, config["evaluation"]["bootstrap_samples"], config["seed"] + 71)
            with (folder / "paired_effects.csv").open(newline="") as stream:
                effects = list(csv.DictReader(stream))
            assert len(effects) == len(expected_effects)
            for a, b in zip(effects, expected_effects):
                assert a["method"] == b["method"] and a["reference"] == b["reference"]
                for key in ("delta_linear_mean_db", "ci95_low_db", "ci95_high_db"):
                    assert np.isclose(float(a[key]), b[key], rtol=1e-10, atol=1e-10)
            scenarios += 1
    report = {"status": "passed", "source_archive_verified": True, "data_caches_verified": cache_count,
        "frozen_models_verified": model_count, "scenarios_verified": scenarios, "metrics_recomputed": rows_checked,
        "maximum_metric_difference": maximum_metric_difference, "sampler_replays": replays,
        "maximum_replay_difference": maximum_replay_difference, "training_only_calibration": True,
        "fresh_geometry_and_common_observations": True, "communication_rates_recomputed": True,
        "crossed_bootstrap_recomputed": True}
    write_json(result / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    audit(parser.parse_args().run_id)
