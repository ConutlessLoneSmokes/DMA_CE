"""Read-only checks and analytic diagnostics of the two completed observation sweeps."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch


root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
torch.set_num_threads(4)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (out / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


checks, hardware, decompositions, subspaces, linear_risk, manifests = [], [], [], [], [], []
paired_ids, paired_bank = None, None
for band in (3, 30):
    name = f"observation_baselines_20261022_{band}ghz_v1"
    run, export = root / "runs" / name, root / "results" / name
    cache = torch.load(run / "cache/physical_absorption.pt", map_location="cpu", weights_only=True)
    observed = torch.load(run / "predictions/physical_absorption/observations.pt", map_location="cpu", weights_only=True)
    frozen = json.loads((run / "frozen.json").read_text())
    config = json.loads((run / "config_resolved.json").read_text())
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    assert frozen["selection_split"] == "validation" and not frozen["screening_used"]
    assert observed["frozen_sha256"] == sha(run / "frozen.json")
    assert observed["created_at_unix"] >= frozen["frozen_at_unix"]
    for file, digest in frozen["cache_sha256"].items():
        assert sha(run / "cache" / file) == digest
    assert sha(run / "source.zip") == frozen["source_archive_sha256"]
    source = json.loads((run / "source_manifest.json").read_text())
    with zipfile.ZipFile(run / "source.zip") as archive:
        assert all(hashlib.sha256(archive.read(file)).hexdigest() == digest for file, digest in source.items())
    assert all(sha(root / file) == digest for file, digest in source.items()
        if file.startswith(("src/", "scripts/")) and file.endswith(".py"))
    for split, count in (("train", 512), ("validation", 64)):
        assert len(set(cache["geometry_ids"][split])) == count
    ids = observed["geometry_ids"]
    assert len(set(ids)) == 128
    assert not set(ids) & set(cache["geometry_ids"]["train"] + cache["geometry_ids"]["validation"])
    assert not set(cache["geometry_ids"]["train"]) & set(cache["geometry_ids"]["validation"])
    if paired_ids is not None:
        assert paired_ids == {**cache["geometry_ids"], "screen": ids}
        assert torch.equal(paired_bank, cache["state_bank"])
    paired_ids, paired_bank = {**cache["geometry_ids"], "screen": ids}, cache["state_bank"]
    metrics = read(export / "metrics.csv")
    lookup = {(int(r["context"]), r["method"], int(r["scene"])): r for r in metrics}
    assert len(metrics) == len(lookup) == 20480
    maximum = dict(nmse_relative=0., mse_relative=0., ls_relative=0., snr_db=0., paired_noise=0.)
    train = cache["training"][0].to(torch.complex128)
    mean = train.mean(0)
    x = (train - mean).flatten(1)
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    rank = int((singular > singular[0] * max(x.shape) * torch.finfo(torch.float64).eps).sum())
    assert rank == 511
    basis = vh[:rank]
    eig = singular.square()
    truth = observed["entries"][0]["truth"].to(torch.complex128)
    energy = truth.abs().square().sum((1, 2))
    centered_screen = (truth - mean).flatten(1)
    outside = centered_screen - centered_screen @ basis.mH @ basis
    for size in (128, 256, 511):
        residual = centered_screen - centered_screen @ basis[:size].mH @ basis[:size]
        subspaces.append({"bandwidth_ghz": band, "training_vectors": 512, "channel_dimension": 1024,
            "directions": size, "training_energy_fraction": float(eig[:size].sum() / eig.sum()),
            "screen_affine_subspace_nmse_floor": float((residual.abs().square().sum(1) / energy).mean()),
            "sample_covariance_effective_rank": float(eig.sum().square() / eig.square().sum())})
    base_noise = None
    assert len(cache["contexts"]) == len(observed["entries"]) == 32
    for index, (context, entry) in enumerate(zip(cache["contexts"], observed["entries"])):
        assert entry["context_index"] == index and torch.equal(entry["truth"].to(torch.complex128), truth)
        a = context["matrix"].to(torch.complex128)
        y = entry["observation"].to(torch.complex128)
        clean = torch.einsum("kjn,bkn->bkj", a, truth)
        noise = y - clean
        variance = context["noise_complex_variance"]
        count, snr = context["configurations"], context["snr_db"]
        z = noise / np.sqrt(variance)
        if count == 64 and snr == -10:
            base_noise = z
        training_y = torch.einsum("kjn,bkn->bkj", a, train)
        q = float(training_y.abs().square().mean())
        maximum["snr_db"] = max(maximum["snr_db"], abs(10 * np.log10(q / variance) - snr))
        saved_path = run / f"predictions/physical_absorption/context_{index}.pt"
        saved = torch.load(saved_path, map_location="cpu", weights_only=True)
        assert saved["frozen_sha256"] == sha(run / "frozen.json")
        original = entry["observation"].contiguous().numpy().tobytes()
        assert saved["observation_sha256"] == hashlib.sha256(original).hexdigest()
        for method, prediction in saved["outputs"].items():
            assert torch.isfinite(prediction).all()
            error = (prediction.to(torch.complex128) - truth).abs().square().sum((1, 2))
            nmse, mse = error / energy, error / truth[0].numel() * cache["channel_scale"] ** 2
            recorded = torch.tensor([float(lookup[index, method, i]["nmse_linear"]) for i in range(128)], dtype=torch.float64)
            recorded_mse = torch.tensor([float(lookup[index, method, i]["mse_physical"]) for i in range(128)], dtype=torch.float64)
            maximum["nmse_relative"] = max(maximum["nmse_relative"], float(((nmse - recorded).abs() / nmse).max()))
            maximum["mse_relative"] = max(maximum["mse_relative"], float(((mse - recorded_mse).abs() / mse).max()))
        inverse = torch.linalg.pinv(a)
        ls = torch.einsum("knj,bkj->bkn", inverse, y)
        replay_error = float((ls - saved["outputs"]["ls_min_norm"]).norm() / ls.norm())
        maximum["ls_relative"] = max(maximum["ls_relative"], replay_error)
        null = truth - torch.einsum("knj,bkj->bkn", inverse, clean)
        floor = (null.abs().square().sum((1, 2)) / energy).mean()
        actual = (torch.einsum("knj,bkj->bkn", inverse, noise).abs().square().sum((1, 2)) / energy).mean()
        expected = variance * inverse.abs().square().sum() * (1 / energy).mean()
        decompositions.append({"bandwidth_ghz": band, "configurations": count, "snr_db": snr,
            "ls_null_floor_linear": float(floor), "ls_actual_noise_linear": float(actual),
            "ls_expected_noise_linear": float(expected), "ls_reconstructed_nmse_db": float(10 * torch.log10(floor + actual))})
        if snr == -10:
            sv = torch.linalg.svdvals(a)
            ranks = torch.linalg.matrix_rank(a)
            assert (ranks == min(count, 32)).all()
            hardware.append({"bandwidth_ghz": band, "configurations": count,
                "minimum_rank": int(ranks.min()), "minimum_nonzero_singular_value": float(sv.min()),
                "maximum_condition_number": float((sv[:, 0] / sv[:, -1]).max()),
                "noise_gain_trace_sum_over_tones": float(inverse.abs().square().sum())})
        if snr == 30 and count in (16, 64):
            # Dual training-space solve independently checks the archived observation-space formula.
            ztrain = (training_y - training_y.mean(0)).flatten(1)
            coefficient = torch.linalg.solve(ztrain @ ztrain.mH + (len(train) - 1) * variance
                * torch.eye(len(train), dtype=torch.complex128), x)
            gain_t = ztrain.mH @ coefficient
            predicted = mean.flatten() + (y - training_y.mean(0)).flatten(1) @ gain_t
            archived = saved["outputs"]["lmmse_joint"].to(torch.complex128).flatten(1)
            relative = float((predicted - archived).norm() / predicted.norm())
            assert relative < 2e-6
            noise_energy = variance * gain_t.abs().square().sum()
            risks = []
            for values, observations in ((train, training_y), (truth, clean)):
                bias = mean.flatten() + (observations - training_y.mean(0)).flatten(1) @ gain_t - values.flatten(1)
                risks.append(float(((bias.abs().square().sum(1) + noise_energy)
                    / values.abs().square().sum((1, 2))).mean()))
            outside_prediction = archived - mean.flatten()
            outside_prediction -= outside_prediction @ basis.mH @ basis
            linear_risk.append({"bandwidth_ghz": band, "configurations": count, "snr_db": snr,
                "dual_replay_prediction_relative_error": relative,
                "expected_train_nmse_linear": risks[0], "expected_screen_nmse_linear": risks[1],
                "screen_training_subspace_floor_linear": float((outside.abs().square().sum(1) / energy).mean()),
                "predicted_outside_training_subspace_nmse": float((outside_prediction.abs().square().sum(1) / energy).mean())})
    for context, entry in zip(cache["contexts"], observed["entries"]):
        clean = torch.einsum("kjn,bkn->bkj", context["matrix"].to(torch.complex128), truth)
        z = (entry["observation"].to(torch.complex128) - clean) / np.sqrt(context["noise_complex_variance"])
        maximum["paired_noise"] = max(maximum["paired_noise"], float((z - base_noise[:, :, :context["configurations"]]).abs().max()))
    assert maximum["nmse_relative"] < 2e-6 and maximum["mse_relative"] < 1e-12
    assert maximum["ls_relative"] < 2e-6 and maximum["snr_db"] < 2e-5
    assert maximum["paired_noise"] < 1e-4
    checks.append({"run": name, "metrics_independently_recomputed_complex128": len(metrics),
        "geometry_splits_disjoint": True, "source_and_freeze_hashes_verified": True,
        "full_column_or_row_rank_all_tones": True, "maximum_errors": maximum})
    for path in sorted(export.rglob("*")):
        if path.is_file():
            manifests.append({"run": name, "path": path.relative_to(export).as_posix(), "sha256": sha(path)})
    print(json.dumps(checks[-1]), flush=True)

write("hardware.csv", hardware)
write("ls_decomposition.csv", decompositions)
write("training_subspace.csv", subspaces)
write("joint_linear_risk.csv", linear_risk)
write("export_hashes.csv", manifests)
(out / "verification.json").write_text(json.dumps({"status": "passed", "runs": checks,
    "paired_geometry_and_dma_bank_across_bandwidths": True,
    "script_sha256": sha(Path(__file__)),
    "scope": "Read-only saved-output checks; analytic covariance/LS diagnostics; no SBL refit or hyperparameter selection."}, indent=2) + "\n")
