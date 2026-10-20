"""Two fixed validation geometries, paired observations, fixed-budget corrected SBL."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
OUT = Path(__file__).resolve().parent
ROOT, RUN = OUT.parents[1], OUT.parents[1] / "runs" / OUT.name
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.data import build_hardware, tensor_hash
from dma_ce.experiment import file_hash, nmse, write_csv, write_json
from dma_ce.sbl import SOLVER_VERSION, joint_sbl

RUN.mkdir(parents=True, exist_ok=False)
source = ROOT / "runs/sbl_dictionary_diagnostic_20261019_v1/input.pt"
assert file_hash(source) == "be98e4df6e50e1a47a53e86036acfdedc54fe9e58cea4fdba3bba1f72bc6e594"
shutil.copy2(source, RUN / "input.pt")
data = torch.load(RUN / "input.pt", map_location="cpu", weights_only=True)
config = json.loads((ROOT / "configs/research.json").read_text())
assert all(config[k] == data["config"][k] for k in ("seed", "system", "channel"))
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
write_json(OUT / "config_resolved.json", config)
write_json(OUT / "provenance.json", {**data["provenance"], "input_sha256": file_hash(source),
    "geometries": data["indices"], "contexts": [0, 3, 4, 7], "budgets": [2000, 4000, 8000],
    "solver": SOLVER_VERSION, "early_stopping": False, "diagnostic_only": True})
write_json(OUT / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
    "numpy": np.__version__, "device": "cuda", "gpu": torch.cuda.get_device_name(),
    "script_sha256": file_hash(Path(__file__))})
sources = [p for folder in ("src", "scripts", "configs", "tests", "docs") for p in (ROOT / folder).rglob("*")
    if p.is_file() and "__pycache__" not in p.parts] + [Path(__file__)]
with zipfile.ZipFile(OUT / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sources:
        archive.write(path, path.relative_to(ROOT).as_posix())
write_json(OUT / "source_manifest.json", {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources})
shutil.copy2(ROOT / "results/sbl_dictionary_diagnostic_20261019_v1/paired_summary.csv", OUT / "previous_em_summary.csv")
started = time.perf_counter()
timer = threading.Timer(600, lambda: os._exit(124))
timer.daemon = True
timer.start()
rows, saved, checks = [], {}, []
try:
    for context in (0, 3, 4, 7):
        entry, known = data["splits"]["validation"][context], data["contexts"][context]
        y, noise, truth = [entry[k].cuda() for k in ("observation", "variance", "truth")]
        dictionaries = {}
        for grid in config["research"]["grids"]:
            local = copy.deepcopy(config)
            local["sbl"].update({k: grid[k] for k in ("angle_points", "range_points")})
            hw = build_hardware(local, known["bandwidth_hz"], data["state_bank"].numpy())
            assert torch.equal(hw["matrix"][:, :known["configurations"]], known["matrix"])
            dictionary = hw["dictionary"].cuda()
            dictionaries[grid["name"]] = dictionary
            sensing = known["matrix"].cuda() @ dictionary
            joint_sbl(y, sensing, dictionary, noise, iterations=5, early_stopping=False)
            for budget in (2000, 4000, 8000):
                torch.cuda.synchronize()
                before = time.perf_counter()
                mean, variance, info = joint_sbl(y, sensing, dictionary, noise,
                    iterations=budget, tolerance=config["sbl"]["tolerance"], early_stopping=False)
                torch.cuda.synchronize()
                milliseconds = (time.perf_counter() - before) * 1000 / len(y)
                assert (info["iterations"] == budget).all()
                assert all(torch.isfinite(x).all() for x in (mean, variance, *info.values()))
                values = nmse(truth, mean)
                for i, error in enumerate(values.tolist()):
                    rows.append({"context": context, "grid": grid["name"], "scene": data["indices"][i],
                        "bandwidth_ghz": known["bandwidth_hz"] / 1e9, "dma_configurations": known["configurations"],
                        "nominal_snr_db": known["snr_db"], "budget": budget, "actual_iterations": int(info["iterations"][i]),
                        "nmse_linear": error, "converged": bool(info["converged"][i]),
                        "optimality_residual": float(info["optimality_residual"][i]),
                        "objective_per_tone": float(info["objective_per_tone"][i]),
                        "mean_variance": float(variance[i].mean()), "batch_ms_per_sample": milliseconds,
                        "observation_sha256": tensor_hash(entry["observation"][i])})
                saved[f"{context}_{grid['name']}_{budget}"] = {"mean": mean.cpu(), "variance": variance.cpu(),
                    "info": {k: v.cpu() for k, v in info.items()}}
                write_csv(OUT / "metrics.csv", rows)
                print(json.dumps({"context": context, "grid": grid["name"], "budget": budget,
                    "nmse_db": float(10 * values.mean().log10()), "kkt": info["optimality_residual"].tolist(),
                    "seconds": time.perf_counter() - started}), flush=True)
        nested = torch.tensor([2 * a * 13 + 2 * r for a in range(49) for r in range(7)], device="cuda")
        assert torch.equal(dictionaries["coarse"], dictionaries["dense"][:, :, nested])
        checks.append({"context": context, "nested_dictionary_difference": 0., "physical_matrix_difference": 0.})
    torch.save(saved, RUN / "predictions.pt")
    summary = []
    for context in (0, 3, 4, 7):
        item = {"context": context}
        for grid in ("coarse", "dense"):
            for budget in (2000, 4000, 8000):
                selected = [r for r in rows if r["context"] == context and r["grid"] == grid and r["budget"] == budget]
                item[f"{grid}_{budget}_nmse_db"] = float(10 * np.log10(np.mean([r["nmse_linear"] for r in selected])))
        item["dense_gain_8000_db"] = item["coarse_8000_nmse_db"] - item["dense_8000_nmse_db"]
        summary.append(item)
    write_csv(OUT / "paired_summary.csv", summary)
    write_json(OUT / "checks.json", {"physical_checks": checks, "all_actual_counts_equal_budgets": True,
        "metrics_rows": len(rows), "source_input_unchanged": file_hash(source) == file_hash(RUN / "input.pt")})
    write_json(OUT / "status.json", {"status": "complete", "seconds": time.perf_counter() - started, "metrics_rows": len(rows)})
except Exception as error:
    write_json(OUT / "status.json", {"status": "failed", "error": str(error), "seconds": time.perf_counter() - started})
    raise
finally:
    timer.cancel()
    for path in OUT.iterdir():
        if path.is_file():
            shutil.copy2(path, RUN / path.name)
