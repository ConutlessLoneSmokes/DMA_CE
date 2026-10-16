"""Fixed validation probes: SBL 100 versus 500 iterations and 1/4/16 sample means."""
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce.conditional import ConditionalCNN
from dma_ce.data import build_hardware
from dma_ce.experiment import file_hash, nmse, predict_entry, write_csv, write_json
from dma_ce.sbl import joint_sbl


def diagnose(parent_id, run_id):
    if any(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None for value in (parent_id, run_id)):
        raise ValueError("invalid run ID")
    parent = ROOT / "runs" / parent_id
    config = json.loads((parent / "config_resolved.json").read_text())
    settings = {"parent_run": parent_id, "split": "validation", "first_geometries": 32,
        "sbl_iterations": 500, "sample_means": [1, 4, 16], "timeout_seconds": 1800,
        "parent_data_sha256": file_hash(parent / "data.pt"), "parent_config": config,
        "sampling_seed": "parent_seed + 7000 + 1000*context + 100000*draw"}
    assert config["sbl"]["iterations"] == 100 and config["sbl"]["batch_size"] == 32
    assert config["data"]["validation"] >= 32 and config["evaluation"]["batch_size"] == 32
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    result = ROOT / "results" / run_id
    result.mkdir(parents=True, exist_ok=False)
    write_json(run / "config_resolved.json", settings)
    (run / "parent_source.zip").write_bytes((parent / "source.zip").read_bytes())
    (run / "diagnose_validation.py").write_bytes(Path(__file__).read_bytes())
    environment = json.loads((parent / "environment.json").read_text())
    environment["command"] = sys.argv
    write_json(run / "environment.json", environment)
    frozen = json.loads((parent / "frozen_validation.json").read_text())
    write_json(run / "provenance.json", {"models": frozen["models"],
        "parent_source_zip_sha256": file_hash(run / "parent_source.zip"),
        "diagnostic_script_sha256": file_hash(Path(__file__)), "parent_version": json.loads((parent / "version.json").read_text())})
    started = time.perf_counter()
    timer = threading.Timer(1800, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        device = torch.device(config["device"])
        cache = torch.load(parent / "data.pt", map_location="cpu", weights_only=True)
        model = ConditionalCNN(config["learning"]["width"]).to(device)
        checkpoint = torch.load(parent / "models/reliable_residual.pt", map_location=device, weights_only=True)
        assert file_hash(parent / "models/reliable_residual.pt") == frozen["models"]["reliable_residual"]
        model.load_state_dict(checkpoint["state_dict"])
        rows, saved, hardware = [], [], {}
        for index, entry in enumerate(cache["splits"]["validation"]):
            subset = {key: value[:32] if torch.is_tensor(value) and value.ndim else value for key, value in entry.items()}
            context = cache["contexts"][index]
            band = context["bandwidth_hz"]
            if band not in hardware:
                hardware[band] = build_hardware(config, band, cache["state_bank"].numpy())
            dictionary = hardware[band]["dictionary"].to(device)
            assert torch.equal(hardware[band]["matrix"][:, :context["configurations"]], context["matrix"])
            sensing = context["matrix"].to(device) @ dictionary
            mean, variance, info = joint_sbl(subset["observation"].to(device), sensing, dictionary,
                subset["variance"].to(device), iterations=500, tolerance=config["sbl"]["tolerance"])
            mean = mean.cpu()
            methods = {"sbl_100": subset["coarse"], "sbl_500": mean}
            total = torch.zeros_like(subset["coarse"])
            for draw in range(16):
                estimate, _ = predict_entry(config, cache, model, "reliable_residual", subset, device,
                    seed=config["seed"] + 7000 + 1000 * index + 100000 * draw)
                total += estimate
                if draw + 1 in (1, 4, 16):
                    methods[f"generative_mean_{draw + 1}"] = total.clone() / (draw + 1)
            for method, estimate in methods.items():
                values = nmse(subset["truth"], estimate)
                assert torch.isfinite(values).all()
                for scene, value in enumerate(values.tolist()):
                    rows.append({"context": index, "scene": scene, "method": method,
                        "bandwidth_hz": band, "configurations": context["configurations"],
                        "snr_db": context["snr_db"], "nmse_linear": value,
                        "sbl_converged": bool(info["converged"][scene]) if method == "sbl_500" else bool(subset["sbl_converged"][scene])})
            saved.append({"estimates": methods, "sbl_500_variance": variance.cpu(),
                "sbl_500_iterations": info["iterations"].cpu(), "sbl_500_converged": info["converged"].cpu()})
            event = {"context": index, "elapsed_seconds": time.perf_counter() - started,
                "sbl_500_converged": int(info["converged"].sum())}
            with (run / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        torch.save(saved, run / "predictions.pt")
        write_csv(result / "metrics.csv", rows)
        summary = []
        for index in range(len(cache["contexts"])):
            for method in saved[index]["estimates"]:
                selected = [row for row in rows if row["context"] == index and row["method"] == method]
                summary.append({"context": index, "method": method,
                    "mean_nmse_db": float(10 * np.log10(np.mean([row["nmse_linear"] for row in selected]))),
                    "sbl_convergence_rate": float(np.mean([row["sbl_converged"] for row in selected]))})
        write_csv(result / "summary.csv", summary)
        assert len(rows) == len(cache["contexts"]) * 32 * 5
        write_json(run / "status.json", {"status": "complete", "elapsed_seconds": time.perf_counter() - started,
            "metrics_rows": len(rows), "independent_validation_geometries": 32, "screening_used": False})
        for name in ("config_resolved.json", "environment.json", "provenance.json", "diagnose_validation.py", "status.json"):
            (result / name).write_bytes((run / name).read_bytes())
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    diagnose(sys.argv[1], sys.argv[2])
