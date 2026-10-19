"""Recompute frozen study metrics, observations, source hashes and sampler replays."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.data import sample_scenes, tensor_hash
from dma_ce.continuation import FROZEN_STAGES, verify_archive, verify_lineage
from dma_ce.experiment import apply_dc, file_hash, nmse, write_json
from dma_ce.research import checkpoint_budgets, communication_rates, digest, infer, load_model, paired_effects, recipe
from dma_ce.suite import STAGES


def geometry(config, split):
    radius, angle, gain = sample_scenes(config, ("train", "validation", "screen").index(split), config["data"][split])
    return np.concatenate((radius, angle, gain.real, gain.imag), axis=1)


def audit_validation(run, result, config, states, cache_count, lineage):
    chosen = states["budget"]["selection"]
    data = run / "cache" / f"{chosen['cache_id']}.pt"
    cache_hash = file_hash(data)
    cache = torch.load(data, map_location="cpu", weights_only=True)
    assert set(cache["splits"]) == {"train", "validation"}
    train = cache["splits"]["train"]
    errors = torch.stack([(e["truth"] - e["coarse"]).abs().square().mean() for e in train]).mean()
    variances = torch.stack([e["reliability"].mean() for e in train]).mean()
    assert np.isclose(float(errors / variances), cache["reliability_scale"])
    assert np.isclose(float((errors / 2).clamp_min(1e-12).sqrt()), cache["residual_real_scale"])
    packed = [geometry(config, split) for split in ("train", "validation")]
    for split, values in zip(("train", "validation"), packed):
        assert hashlib.sha256(values.tobytes()).hexdigest() == cache["geometry_hashes"][split]
    assert not {r.tobytes() for r in packed[0]}.intersection(r.tobytes() for r in packed[1])
    with (result / "diffusion" / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["method"], int(r["seed"]), int(r["context"]), int(r["scene"])): r for r in rows}
    assert len(lookup) == len(rows)
    points, averages = checkpoint_budgets(config, "diffusion"), config["research"]["averages"]
    maximum, checked, models, means = 0., 0, 0, {}
    for mode in sorted({"none", chosen["reliability"]}):
        spec = recipe(config, chosen["architecture"], chosen["loss"], mode, "diffusion")
        for seed in config["research"]["training_seeds"]:
            job = run / "models" / chosen["cache_id"] / digest(spec)[:12] / f"seed_{seed}"
            identity = json.loads((job / "identity.json").read_text())
            assert identity == {"recipe": spec, "seed": seed, "cache_sha256": cache_hash,
                "horizon": max(config["research"]["checkpoints"]), "learning": config["learning"],
                "eval_every": config["research"]["eval_every"]}
            last = torch.load(job / "last.pt", map_location="cpu", weights_only=True)
            assert last["step"] == max(points) and last["optimizer"]["state"]
            assert all(torch.is_tensor(last[k]) and last[k].dtype == torch.uint8 for k in ("batch_rng", "noise_rng"))
            for step in points:
                saved = torch.load(job / f"step_{step}.pt", map_location="cpu", weights_only=True)
                assert saved["recipe"] == spec and saved["seed"] == seed and saved["step"] == step
                assert saved["cache_sha256"] == cache_hash
                if step == max(points):
                    assert all(torch.equal(v, last["state_dict"][k]) for k, v in saved["state_dict"].items())
                model = load_model(saved, torch.device(config["device"]))
                for c, entry in enumerate(cache["splits"]["validation"]):
                    replay = infer(config, cache, model, spec, entry, torch.device(config["device"]),
                        seed=config["seed"] + 7000 + c * 1000, averages=averages)
                    for count in averages:
                        label = f"diffusion_{mode}_step{step}" + (f"_avg{count}" if len(averages) > 1 else "")
                        values = nmse(entry["truth"], replay[count][0])
                        means.setdefault((mode, step, count), []).extend(values.tolist())
                        for g, value in enumerate(values.tolist()):
                            row = lookup[label, seed, c, g]
                            maximum = max(maximum, abs(float(row["nmse_linear"]) - value))
                            assert np.isclose(float(row["nmse_linear"]), value, rtol=2e-6, atol=1e-7)
                            assert row["observation_sha256"] == tensor_hash(entry["observation"][g])
                            checked += 1
                del model
                models += 1
    assert checked == len(rows)
    means = {k: float(np.mean(v)) for k, v in means.items()}
    best = min(means, key=means.get)
    state = states["diffusion"]
    if state["status"] == "complete":
        selected = state["selection"]["selected"]
        best = (selected["mode"], selected["steps"], selected["average"])
        assert np.isclose(means[best], min(means.values()), rtol=1e-10, atol=1e-12)
    mode, step, count = best
    gain = float(10 * np.log10(means[mode, points[-2], count] / means[mode, points[-1], count]))
    limited = step == max(points) and gain > config["research"]["training_stability_db"]
    assert limited == (state["status"] != "complete")
    if limited:
        boundary = json.loads((result / "diffusion" / "budget_boundary.json").read_text())
        assert boundary["validation_only"] and np.isclose(gain, boundary["last_interval_improvement_db"])
    report = {"status": "passed", "scope": "validation_only", "ready_for_final": not limited,
        "diffusion_status": state["status"], "source_archive_verified": True,
        "continuation_verified": lineage is not None, "data_caches_verified": cache_count,
        "diffusion_checkpoints_replayed": models, "validation_metrics_recomputed": checked,
        "maximum_metric_difference": maximum, "last_interval_improvement_db": gain,
        "training_only_calibration": True, "training_validation_geometry_disjoint": True,
        "screening_data_read": False}
    write_json(result / "validation_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def audit(run_id, root=ROOT, validation_only=False):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None:
        raise ValueError("invalid study ID")
    run, result = root / "runs" / run_id, root / "results" / run_id
    config = json.loads((run / "config_resolved.json").read_text())
    states = {stage: json.loads((run / "stages" / f"{stage}.json").read_text())
        if (run / "stages" / f"{stage}.json").exists() else {"status": "pending"} for stage in STAGES}
    incomplete = [stage for stage in STAGES if states[stage]["status"] != "complete"]
    legacy_boundary = (states["diffusion"]["status"] == "failed"
        and "diffusion validation still improves" in states["diffusion"].get("error", ""))
    if (not validation_only and incomplete) or (validation_only and (any(states[s]["status"] != "complete"
            for s in FROZEN_STAGES) or states["diffusion"]["status"] not in ("complete", "budget_limited") and not legacy_boundary)):
        report = {"status": "not_ready", "incomplete_stages": incomplete,
            "message": "Use --validation-only after diffusion completes or reaches its declared budget boundary."}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    verify_archive(run)
    lineage = verify_lineage(run, result)
    cache_count = 0
    for meta in (run / "cache").glob("*.json"):
        assert file_hash(meta.with_suffix(".pt")) == json.loads(meta.read_text())["sha256"], meta.name
        cache_count += 1
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if validation_only:
        return audit_validation(run, result, config, states, cache_count, lineage)
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
        "crossed_bootstrap_recomputed": True, "continuation_verified": lineage is not None}
    write_json(result / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--validation-only", action="store_true", help="Audit validation artifacts without requiring final/screening evaluation")
    args = parser.parse_args()
    report = audit(args.run_id, validation_only=args.validation_only)
    sys.exit(0 if report["status"] == "passed" else 2)
