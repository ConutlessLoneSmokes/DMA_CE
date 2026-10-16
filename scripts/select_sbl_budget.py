"""Select an SBL budget using fixed validation convergence and stability criteria."""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.data import build_hardware, sync
from dma_ce.experiment import file_hash, nmse, write_csv, write_json
from dma_ce.sbl import joint_sbl


def select(parent_id, run_id):
    if any(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None for value in (parent_id, run_id)):
        raise ValueError("invalid run ID")
    parent, run, export = ROOT / "runs" / parent_id, ROOT / "runs" / run_id, ROOT / "results" / run_id
    config = json.loads((parent / "config_resolved.json").read_text())
    settings = {"parent": parent_id, "split": "validation", "budgets": [500, 1000, 2000, 4000],
        "minimum_overall_convergence": .95, "minimum_context_convergence": .90,
        "maximum_nmse_change_db": .1, "maximum_variance_relative_change": .05,
        "parent_config": config, "parent_data_sha256": file_hash(parent / "data.pt")}
    run.mkdir(parents=True, exist_ok=False)
    export.mkdir(parents=True, exist_ok=False)
    write_json(run / "config_resolved.json", settings)
    environment = json.loads((parent / "environment.json").read_text())
    environment["command"] = sys.argv
    write_json(run / "environment.json", environment)
    sources = [ROOT / "AGENTS.md", ROOT / "docs/SECOND_STAGE_PLAN.md", ROOT / "pyproject.toml"]
    for folder in ("src", "scripts", "configs", "tests"):
        sources.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(run / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(ROOT).as_posix())
    write_json(run / "source_manifest.json", {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sources})
    started = time.perf_counter()
    def timeout():
        write_json(run / "status.json", {"status": "timeout", "elapsed_seconds": time.perf_counter() - started})
        os._exit(124)
    timer = threading.Timer(1800, timeout)
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        cache = torch.load(parent / "data.pt", map_location="cpu", weights_only=True)
        hardware, rows, summary, previous, selection = {}, [], [], None, None
        write_json(run / "status.json", {"status": "running"})
        for budget in settings["budgets"]:
            current = []
            for index, entry in enumerate(cache["splits"]["validation"]):
                context = cache["contexts"][index]
                band = context["bandwidth_hz"]
                if band not in hardware:
                    hardware[band] = build_hardware(config, band, cache["state_bank"].numpy())
                dictionary = hardware[band]["dictionary"].to(device)
                sensing = context["matrix"].to(device) @ dictionary
                estimates, variances, converged, iterations = [], [], [], []
                for start in range(0, len(entry["truth"]), config["sbl"]["batch_size"]):
                    stop = start + config["sbl"]["batch_size"]
                    mean, variance, info = joint_sbl(entry["observation"][start:stop].to(device), sensing, dictionary,
                        entry["variance"][start:stop].to(device), iterations=budget, tolerance=config["sbl"]["tolerance"])
                    estimates.append(mean.cpu()); variances.append(variance.cpu())
                    converged.append(info["converged"].cpu()); iterations.append(info["iterations"].cpu())
                values = nmse(entry["truth"], torch.cat(estimates))
                flags, counts, diagonal = torch.cat(converged), torch.cat(iterations), torch.cat(variances)
                assert torch.isfinite(values).all() and torch.isfinite(diagonal).all()
                row = {"budget": budget, "context": index, "samples": len(values),
                    "mean_nmse_db": float(10 * values.mean().log10()),
                    "convergence_rate": float(flags.float().mean()), "mean_variance": float(diagonal.mean()),
                    "mean_iterations": float(counts.float().mean()), "maximum_iterations": int(counts.max())}
                current.append(row); summary.append(row)
                rows.extend({"budget": budget, "context": index, "scene": scene,
                    "nmse_linear": float(value), "converged": bool(flags[scene]), "iterations": int(counts[scene])}
                    for scene, value in enumerate(values))
                event = {"elapsed_seconds": time.perf_counter() - started, **row}
                with (run / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
            if previous is not None:
                max_nmse_change = max(abs(a["mean_nmse_db"] - b["mean_nmse_db"]) for a, b in zip(current, previous))
                max_variance_change = max(abs(a["mean_variance"] / b["mean_variance"] - 1) for a, b in zip(current, previous))
                if (np.mean([x["convergence_rate"] for x in previous]) >= .95
                        and min(x["convergence_rate"] for x in previous) >= .90
                        and max_nmse_change <= .1 and max_variance_change <= .05):
                    selection = {"selected_iterations": previous[0]["budget"], "checked_against": budget,
                        "maximum_nmse_change_db": max_nmse_change, "maximum_variance_relative_change": max_variance_change,
                        "overall_convergence": float(np.mean([x["convergence_rate"] for x in previous])),
                        "minimum_context_convergence": min(x["convergence_rate"] for x in previous)}
                    break
            previous = current
        write_csv(export / "metrics.csv", rows)
        write_csv(export / "summary.csv", summary)
        write_json(run / "selection.json", {"selection": selection, "selection_split": "validation"})
        write_json(run / "status.json", {"status": "complete" if selection else "no_stable_budget",
            "elapsed_seconds": time.perf_counter() - started, "metrics_rows": len(rows)})
        for name in ("config_resolved.json", "environment.json", "source.zip", "source_manifest.json", "selection.json", "status.json"):
            (export / name).write_bytes((run / name).read_bytes())
        print(json.dumps({"selection": selection}), flush=True)
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    select(sys.argv[1], sys.argv[2])
