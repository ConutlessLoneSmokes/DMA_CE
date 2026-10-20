"""Validation-only nested training-count control; no change to the frozen screen."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.experiment import file_hash, nmse, write_csv, write_json
from dma_ce.snr_study import sweep_cache
from dma_ce.structured import apply_lmmse, lmmse_statistics
from dma_ce.suite import Study


@torch.no_grad()
def main():
    original = ROOT / "runs/snr_comparison_20261020_v1"
    config = json.loads((original / "config_resolved.json").read_text())
    config["label"] = "posthoc_covariance_sample_count_validation_only"
    config["diagnostic"] = {"training_counts": [64, 128, 256, 512], "snrs_db": [10, 20, 30],
        "frequency_shrinkage": [0., 1.], "nested_subsets": True, "screening_read": False,
        "frozen_baseline_run": "linear_baselines_20261020_v1", "source_script_sha256": file_hash(Path(__file__))}
    frozen = json.loads((original / "stages/frozen.json").read_text())
    source = ROOT / "runs" / config["comparison"]["parent_run"] / "cache" / (frozen["parent_provenance"]["cache_id"] + ".pt")
    assert file_hash(source) == frozen["parent_provenance"]["cache_sha256"]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        write_json(path, config)
        study = Study(ROOT, path, "baseline_covariance_probe_20261020_v1", "cuda")
    for folder in (study.path, study.export):
        shutil.copyfile(__file__, folder / "diagnose.py")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    parent = torch.load(source, map_location="cpu", weights_only=True)
    validation = sweep_cache(config, parent, "validation", 64, config["diagnostic"]["snrs_db"])
    rows = []
    for context, entry in zip(validation["contexts"], validation["splits"]["validation"]):
        training = next(e["truth"] for c, e in zip(parent["contexts"], parent["splits"]["train"])
            if c["bandwidth_hz"] == context["bandwidth_hz"]).to(study.device)
        for count in config["diagnostic"]["training_counts"]:
            for strength in config["diagnostic"]["frequency_shrinkage"]:
                stats = lmmse_statistics(training[:count], context["matrix"].to(study.device),
                    context["noise_complex_variance"], frequency_shrinkage=strength)
                prediction = apply_lmmse(stats, entry["observation"].to(study.device)).cpu()
                errors = (prediction-entry["truth"]).abs().square()
                rows.append({"bandwidth_hz": context["bandwidth_hz"], "configurations": context["configurations"],
                    "snr_db": context["snr_db"], "training_geometries": count, "frequency_shrinkage": strength,
                    "mean_nmse_linear": float(nmse(entry["truth"], prediction).mean()), "mean_mse": float(errors.mean()),
                    "statistical_posterior_mse": float(stats["diagonal"].mean())})
        study.log("validation_context", context=entry["context_index"])
    write_csv(study.export / "sample_count.csv", rows)
    write_json(study.export / "audit.json", {"screening_read": False, "rows": len(rows),
        "original_training_cache_unchanged": file_hash(source) == frozen["parent_provenance"]["cache_sha256"]})
    write_json(study.path / "status.json", {"status": "complete", "rows": len(rows)})
    # Complete the earlier probe's provenance from the unchanged source run used at its execution.
    probe = ROOT / "runs/linear_baseline_probe_20261020_v1"
    export = ROOT / "results" / probe.name
    for folder in (probe, export):
        shutil.copyfile(original / "config_resolved.json", folder / "source_config_resolved.json")
        shutil.copyfile(original / "source.zip", folder / "inherited_source.zip")
        env = json.loads((original / "environment.json").read_text())
        env["provenance_note"] = "Same server/interpreter/libraries as the source run; command and script below identify this diagnostic."
        env["command"] = [str(sys.executable), "tmp/diagnose_linear_baselines.py"]
        env["diagnostic_script_sha256"] = file_hash(probe / "diagnose.py")
        write_json(folder / "environment.json", env)
    study.log("complete", rows=len(rows))


if __name__ == "__main__":
    main()
