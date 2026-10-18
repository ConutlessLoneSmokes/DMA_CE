"""Two-scene 128 x 128 implementation check, not a performance experiment."""
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from dma_ce.experiment import file_hash, write_json
from dma_ce.research import diagnostic_rows, export_table, infer, load_model, recipe
from dma_ce.suite import Study


config = json.loads((ROOT / "configs/research.json").read_text())
config["label"] = "shape128_two_scene_implementation_only"
config["system"].update(antennas=128, tones=128, bandwidths_hz=[30e9], configurations=[16], snrs_db=[10.])
ratio = (127 / 31) ** 2
for key in ("range_min_m", "range_max_m"):
    config["channel"][key] *= ratio
config["data"].update(train=2, validation=2, screen=2)
config["sbl"].update(iterations=2, batch_size=1)
config["learning"].update(steps=2, batch_size=2, log_every=1)
config["evaluation"].update(batch_size=2, bootstrap_samples=10)
config["research"].update(checkpoints=[1, 2], baseline_iterations=[2, 5], eval_every=2,
    training_seeds=[101], final_seeds=[201], shuffle_seeds=[301], train_curve_samples=2,
    sbl_budgets=[2], expansions=[], mismatches=[])
temporary = ROOT / "tmp" / "shape128_config.json"
write_json(temporary, config)
study = Study(ROOT, temporary, "research_shape128_20261018_v2")
(study.export / "probe.py").write_bytes(Path(__file__).read_bytes())
started = time.perf_counter()
def timeout():
    write_json(study.export / "status.json", {"status": "timeout", "limit_seconds": 180})
    os._exit(124)
timer = threading.Timer(180, timeout)
timer.daemon = True
timer.start()
try:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    cache = study.cache("shape128_coarse", config)
    dense_config = copy.deepcopy(config)
    dense_config["sbl"].update(angle_points=97, range_points=13)
    dense = study.cache("shape128_dense", dense_config, cache, ["validation"])
    for a, b in zip(cache["splits"]["validation"], dense["splits"]["validation"]):
        assert torch.equal(a["observation"], b["observation"])
        assert torch.isfinite(b["coarse"]).all() and tuple(b["coarse"].shape) == (2, 128, 128)
    rows, models = [], []
    for architecture in ("cross_frequency", "per_tone"):
        for kind in ("deterministic", "diffusion"):
            spec = recipe(config, architecture, "condition", "map", kind)
            path = study.train(cache, spec, 101, 2)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = load_model(saved, study.device)
            entry = cache["splits"]["validation"][0]
            averages = (1,) if kind == "deterministic" else (1, 4, 16)
            outputs = infer(config, cache, model, spec, entry, study.device, seed=17000, averages=averages)
            for count, (value, milliseconds, calls) in outputs.items():
                assert value.shape == entry["truth"].shape and torch.isfinite(value).all()
                rows += diagnostic_rows(config, cache, entry, cache["contexts"][0], value,
                    f"{architecture}_{kind}_avg{count}", 101, milliseconds, calls)
            models.append({"recipe": spec, "parameters": saved["parameters"], "checkpoint_sha256": file_hash(path)})
            del model
    export_table(study.export, "metrics.csv", rows)
    result = {"status": "passed", "label": config["label"], "scenes_per_split": 2, "antennas": 128,
        "tones": 128, "dictionary_atoms": [49 * 7, 97 * 13], "models": models,
        "metrics_rows": len(rows), "seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "probe_sha256": file_hash(study.export / "probe.py")}
    write_json(study.export / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)
except Exception as error:
    write_json(study.export / "status.json", {"status": "failed", "error": str(error)})
    raise
finally:
    timer.cancel()
