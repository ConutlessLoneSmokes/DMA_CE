"""Replay all frozen linear predictions and regenerate the independently seeded screen."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import torch
from dma_ce.experiment import file_hash, metric_rows, write_json
from dma_ce.snr_study import sweep_cache
from dma_ce.structured import apply_lmmse, lmmse_statistics


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    run = ROOT / "runs/linear_baselines_20261021_v2"
    export = ROOT / "results" / run.name
    config = json.loads((run / "config_resolved.json").read_text())
    frozen = json.loads((run / "stages/frozen.json").read_text())
    old = ROOT / "runs" / config["linear_baselines"]["source_snr_run"]
    old_frozen = json.loads((old / "stages/frozen.json").read_text())
    source = ROOT / "runs" / config["comparison"]["parent_run"] / "cache" / (old_frozen["parent_provenance"]["cache_id"] + ".pt")
    assert file_hash(source) == frozen["training_cache_sha256"]
    assert file_hash(run / "cache/statistics.pt") == frozen["statistics_sha256"]
    parent = torch.load(source, map_location="cpu", weights_only=True)
    original = torch.load(run / "predictions/observations.pt", map_location="cpu", weights_only=True)
    regenerated = sweep_cache(config, parent, "screen", config["linear_baselines"]["screen_scenes"], config["system"]["snrs_db"])
    statistics = torch.load(run / "cache/statistics.pt", map_location="cpu", weights_only=True)
    with (export / "metrics.csv").open(newline="") as stream:
        saved = {(int(r["context"]), r["method"], int(r["scene"])): r for r in csv.DictReader(stream)}
    prediction_max, metric_max, statistics_max = 0., 0., 0.
    prediction_checks, metric_checks, statistics_checks = 0, 0, 0
    for index, (context, entry) in enumerate(zip(regenerated["contexts"], regenerated["splits"]["screen"])):
        for name in ("truth", "observation", "variance", "realized_snr_db"):
            assert torch.equal(entry[name], original["splits"]["screen"][index][name])
        assert torch.equal(context["matrix"], original["contexts"][index]["matrix"])
        key = (context["bandwidth_hz"], context["configurations"], context["snr_db"])
        y = entry["observation"].cuda()
        a = context["matrix"].cuda()
        outputs = {"ls_min_norm": torch.einsum("knj,bkj->bkn", torch.linalg.pinv(a.to(torch.complex128)),
            y.to(torch.complex128)).to(y.dtype).cpu()}
        outputs.update({method: apply_lmmse(stats, y).cpu() for method, stats in statistics[key].items()})
        stored = torch.load(run / "predictions" / f"context_{index}.pt", weights_only=True)
        for method, prediction in outputs.items():
            prediction_max = max(prediction_max, float((prediction-stored[method]).abs().max()))
            prediction_checks += 1
            score_entry = {**entry, "sbl_converged": torch.zeros(len(y), dtype=torch.bool),
                "sbl_iterations": torch.zeros(len(y), dtype=torch.int64), "sbl_relative_change": torch.zeros(len(y)),
                "coarse_ms_per_sample": 0.}
            for row in metric_rows(config, score_entry, context, prediction, method, 0., 0):
                reference = saved[index, method, row["scene"]]
                assert row["observation_sha256"] == reference["observation_sha256"]
                for name in ("nmse_linear", "nmse_db", "row_nmse_linear", "null_nmse_linear", "observation_residual"):
                    metric_max = max(metric_max, abs(row[name]-float(reference[name])))
                metric_checks += 1
        if index in (0, 8, 18, 35):
            training = next(e["truth"] for c, e in zip(parent["contexts"], parent["splits"]["train"])
                if c["bandwidth_hz"] == key[0])[:config["linear_baselines"]["training_scenes"]].cuda()
            fit = lmmse_statistics(training, a, context["noise_complex_variance"],
                frequency_shrinkage=frozen["selected_frequency_shrinkage"][str(key[0])])
            statistics_max = max(statistics_max, max(float((value.cpu()-statistics[key]["lmmse_shrunk"][name]).abs().max())
                for name, value in fit.items()))
            statistics_checks += 1
    assert prediction_max == metric_max == statistics_max == 0.
    assert metric_checks == len(saved)
    result = {"status": "passed", "regenerated_observation_contexts": len(regenerated["contexts"]),
        "frozen_prediction_replays": prediction_checks, "recomputed_metric_rows": metric_checks,
        "selected_covariance_refits": statistics_checks, "maximum_prediction_difference": prediction_max,
        "maximum_five_metric_difference": metric_max, "maximum_refit_statistics_difference": statistics_max,
        "source_statistics_sha256": frozen["statistics_sha256"], "audit_source_sha256": file_hash(Path(__file__))}
    write_json(export / "replay_audit.json", result)
    shutil.copyfile(__file__, export / "audit_replay.py")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
