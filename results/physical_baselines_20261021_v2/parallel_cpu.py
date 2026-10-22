"""Compute frozen physical cases on the second GPU; publish identical cached outputs atomically."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from dma_ce.data import estimate_entry, tensor_hash
from dma_ce.experiment import file_hash, write_json
from dma_ce.physical_study import case_hardware, make_entries
from dma_ce.structured import apply_lmmse


RUN = ROOT / "runs/physical_baselines_20261021_v2"
torch.set_num_threads(16)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
config = json.loads((RUN / "config_resolved.json").read_text())
frozen = json.loads((RUN / "frozen.json").read_text())
for name, digest in frozen["cache_sha256"].items():
    assert file_hash(RUN / "cache" / name) == digest
source = json.loads((RUN / "source_manifest.json").read_text())
for name, digest in source.items():
    if name.startswith(("src/", "scripts/")) and name.endswith(".py"):
        assert file_hash(ROOT / name) == digest
write_json(RUN / "parallel_cpu_worker.json", {"source_sha256": file_hash(Path(__file__)),
    "source_path": str(Path(__file__).relative_to(ROOT)), "compute_device": "cpu", "cpu_threads": torch.get_num_threads(),
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_at_unix": time.time(),
    "scope": "same frozen solver/settings; online latency is not compared"})
statistics = torch.load(RUN / "cache/statistics.pt", weights_only=True)


def publish(path, value):
    temporary = path.with_name(path.name + f".worker-{os.getpid()}.tmp")
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()
    return torch.load(path, weights_only=True)


def log(event, **fields):
    print(json.dumps({"event": event, "at_unix": time.time(), **fields}), flush=True)


for case_name in ("physical_free_space",):
    case = next(c for c in config["cases"] if c["name"] == case_name)
    cache = torch.load(RUN / "cache" / f"{case_name}.pt", weights_only=True)
    selected, _, hardware = case_hardware(config, case)
    folder = RUN / "predictions" / case_name
    folder.mkdir(exist_ok=True)
    observed_path = folder / "observations.pt"
    if not observed_path.exists():
        entries, ids = make_entries(config, selected, hardware, cache, "screen")
        publish(observed_path, {"entries": entries, "geometry_ids": ids, "created_at_unix": time.time(),
            "frozen_sha256": file_hash(RUN / "frozen.json")})
    observations = torch.load(observed_path, weights_only=True)
    assert observations["created_at_unix"] >= frozen["frozen_at_unix"]
    for context, entry in zip(cache["contexts"], observations["entries"]):
        # Yield this case once the primary process arrives; at most one in-flight context overlaps.
        if case_name == "physical_free_space" and '"case": "physical_free_space", "context"' in (RUN / "events.jsonl").read_text():
            log("yield_to_primary", case=case_name)
            break
        index = entry["context_index"]
        output_path = folder / f"context_{index}.pt"
        if output_path.exists():
            continue
        a, y = context["matrix"].to("cpu"), entry["observation"].to("cpu")
        with torch.no_grad():
            outputs = {"ls_min_norm": torch.einsum("knj,bkj->bkn",
                torch.linalg.pinv(a.to(torch.complex128)), y.to(torch.complex128)).to(y.dtype).cpu()}
            outputs.update({method: apply_lmmse(stats, y).cpu() for method, stats in statistics[case_name][index].items()})
            band = config["system"]["bandwidths_hz"].index(context["bandwidth_hz"])
            solved = estimate_entry(entry, context, hardware[band]["dictionary"].to("cpu"), config["sbl"], "cpu", log)
            outputs["sbl_dense"] = solved["coarse"]
        assert len(outputs) == 5 and all(torch.isfinite(x).all() for x in outputs.values())
        saved = publish(output_path, {"outputs": outputs,
            "solver": {k: v for k, v in solved.items() if k.startswith("sbl_") or k == "coarse_ms_per_sample"},
            "observation_sha256": tensor_hash(entry["observation"]), "frozen_sha256": file_hash(RUN / "frozen.json")})
        assert saved["observation_sha256"] == tensor_hash(entry["observation"])
        log("cpu_worker_context", case=case_name, context=index, converged=int(saved["solver"]["sbl_converged"].sum()))
log("cpu_worker_complete")
