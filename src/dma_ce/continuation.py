"""Copy a validation-only study into an independent, auditable diffusion continuation."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch

from .experiment import file_hash, write_json
from .research import checkpoint_budgets, digest, recipe
from .sbl import SOLVER_VERSION


FROZEN_STAGES = ("baseline", "curves", "factorial", "reliability", "budget")


def verify_archive(run):
    environment = json.loads((run / "environment.json").read_text())
    manifest = json.loads((run / "source_manifest.json").read_text())
    if file_hash(run / "source.zip") != environment["source_zip_sha256"]:
        raise ValueError("source archive hash mismatch")
    with zipfile.ZipFile(run / "source.zip") as archive:
        if set(archive.namelist()) != set(manifest) or any(
                hashlib.sha256(archive.read(name)).hexdigest() != value for name, value in manifest.items()):
            raise ValueError("source archive manifest mismatch")


def verify_lineage(run, result):
    path = run / "continuation.json"
    if not path.exists():
        return None
    lineage = json.loads(path.read_text())
    if (result / path.name).read_bytes() != path.read_bytes():
        raise ValueError("continuation provenance export differs")
    for name, expected in lineage["immutable_files"].items():
        parts = Path(name).parts
        if parts[0] not in ("run", "result") or ".." in parts or Path(name).is_absolute():
            raise ValueError("invalid continuation manifest path")
        target = (run if parts[0] == "run" else result).joinpath(*parts[1:])
        if file_hash(target) != expected:
            raise ValueError(f"inherited file changed: {name}")
    for name, expected in lineage["log_prefixes"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).parts[0] != "models":
            raise ValueError("invalid continuation log path")
        raw = (run / name).read_bytes()[:expected["bytes"]]
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"inherited training log prefix changed: {name}")
    return lineage


def plan_continuation(root, parent_id, added):
    parent, result = root / "runs" / parent_id, root / "results" / parent_id
    config = json.loads((parent / "config_resolved.json").read_text())
    identity = json.loads((parent / "study_identity.json").read_text())
    if identity["config"] != config:
        raise ValueError("parent resolved configuration differs from its identity")
    previous = checkpoint_budgets(config, "diffusion")
    if not added or added != sorted(set(added)) or any(type(v) is not int or v <= max(previous) for v in added):
        raise ValueError("new diffusion checkpoints must be sorted unique integers above the parent maximum")
    verify_archive(parent)
    verify_lineage(parent, result)
    if ((parent / "stages" / "final.json").exists() or list((parent / "stages").glob("frozen_*.json"))
            or any((parent / "predictions").iterdir()) or (result / "final").exists()):
        raise ValueError("continuation requires a parent that has not started screening/final evaluation")
    for stage in FROZEN_STAGES:
        path = parent / "stages" / f"{stage}.json"
        if json.loads(path.read_text())["status"] != "complete":
            raise ValueError(f"parent stage {stage} must be complete")
        if path.read_bytes() != (result / f"{stage}_selection.json").read_bytes():
            raise ValueError(f"parent stage {stage} export differs")
    for meta in (parent / "cache").glob("*.json"):
        if file_hash(meta.with_suffix(".pt")) != json.loads(meta.read_text())["sha256"]:
            raise ValueError(f"parent cache hash mismatch: {meta.name}")
    chosen = json.loads((parent / "stages" / "budget.json").read_text())["selection"]
    cache_hash = json.loads((parent / "cache" / f"{chosen['cache_id']}.json").read_text())["sha256"]
    cache = torch.load(parent / "cache" / f"{chosen['cache_id']}.pt", map_location="cpu", weights_only=True)
    if any(entry.get("sbl_solver") != SOLVER_VERSION for split in ("train", "validation") for entry in cache["splits"][split]):
        raise ValueError("parent uses an older SBL solver; rebuild baselines and refinements in a new study")
    origins = []
    for mode in sorted({"none", chosen["reliability"]}):
        spec = recipe(config, chosen["architecture"], chosen["loss"], mode, "diffusion")
        for seed in config["research"]["training_seeds"]:
            job = Path("models") / chosen["cache_id"] / digest(spec)[:12] / f"seed_{seed}"
            expected = {"recipe": spec, "seed": seed, "cache_sha256": cache_hash,
                "horizon": max(config["research"]["checkpoints"]), "learning": config["learning"],
                "eval_every": config["research"]["eval_every"]}
            if json.loads((parent / job / "identity.json").read_text()) != expected:
                raise ValueError("parent diffusion training identity mismatch")
            saved = torch.load(parent / job / "last.pt", map_location="cpu", weights_only=True)
            point = torch.load(parent / job / f"step_{max(previous)}.pt", map_location="cpu", weights_only=True)
            if (saved["step"] != max(previous) or point["step"] != max(previous)
                    or any(saved[k] != expected[k] or point[k] != expected[k] for k in ("recipe", "seed", "cache_sha256"))
                    or not saved.get("optimizer", {}).get("state") or not saved["optimizer"].get("param_groups")
                    or any(not torch.is_tensor(saved.get(k)) or saved[k].dtype != torch.uint8 for k in ("batch_rng", "noise_rng"))
                    or set(saved["state_dict"]) != set(point["state_dict"])
                    or any(not torch.equal(v, point["state_dict"][k]) for k, v in saved["state_dict"].items())):
                raise ValueError("parent diffusion checkpoint lacks a matching optimizer/RNG continuation state")
            origins.append({"path": (job / "last.pt").as_posix(), "step": saved["step"], "seed": seed,
                "recipe": spec, "sha256": file_hash(parent / job / "last.pt")})
    config = copy.deepcopy(config)
    config["research"]["diffusion_checkpoints"] = [*previous, *added]
    return {"config": config, "parent_id": identity["parent_id"], "parent_run_id": parent_id,
        "added_checkpoints": added, "origin_states": origins}


def inherit_continuation(root, run, result, plan):
    parent, exported = root / "runs" / plan["parent_run_id"], root / "results" / plan["parent_run_id"]
    immutable, prefixes = {}, {}

    def transfer(source, target, frozen=True):
        expected = file_hash(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_hash(target) != expected:
            raise ValueError("continuation copy hash mismatch")
        if frozen:
            key = "run/" + target.relative_to(run).as_posix() if target.is_relative_to(run) else "result/" + target.relative_to(result).as_posix()
            immutable[key] = expected

    for source in (parent / "cache").glob("*"):
        if source.suffix in (".pt", ".json"):
            transfer(source, run / "cache" / source.name)
    for source in (parent / "models").rglob("*"):
        if source.is_file() and source.suffix in (".pt", ".json", ".jsonl"):
            target = run / source.relative_to(parent)
            transfer(source, target, source.name == "identity.json" or source.name.startswith("step_"))
            if source.suffix == ".jsonl":
                prefixes[target.relative_to(run).as_posix()] = {"bytes": target.stat().st_size, "sha256": file_hash(target)}
    for stage in FROZEN_STAGES:
        transfer(parent / "stages" / f"{stage}.json", run / "stages" / f"{stage}.json")
        transfer(exported / f"{stage}_selection.json", result / f"{stage}_selection.json")
    for name in ("base_baseline", *FROZEN_STAGES):
        for source in (exported / name).rglob("*"):
            if source.is_file():
                transfer(source, result / source.relative_to(exported))
    snapshot = run / "parent_snapshot"
    for name in ("source.zip", "source_manifest.json", "config_resolved.json", "study_identity.json",
            "environment.json", "continuation.json", "parent_provenance.json"):
        if (parent / name).exists():
            transfer(parent / name, snapshot / name)
    for name in ("parent_snapshot", "stages"):
        for source in (parent / name).rglob("*"):
            if source.is_file():
                transfer(source, snapshot / source.relative_to(parent))
    for source in (exported / "diffusion").rglob("*"):
        if source.is_file():
            transfer(source, snapshot / "results" / source.relative_to(exported))
    for state in plan["origin_states"]:
        target = snapshot / state["path"]
        transfer(parent / state["path"], target)
        if file_hash(target) != state["sha256"]:
            raise ValueError("parent optimizer state changed during copy")
        state["snapshot_path"] = target.relative_to(run).as_posix()
    lineage = {k: v for k, v in plan.items() if k != "config"}
    lineage.update(immutable_files=immutable, log_prefixes=prefixes, screening_used=False,
        original_learning_rate_horizon=max(plan["config"]["research"]["checkpoints"]),
        extension_learning_rate=plan["config"]["learning"]["learning_rate"] * .1,
        rule="preserve the original cosine prefix, then hold its terminal learning rate")
    write_json(run / "continuation.json", lineage)
    write_json(result / "continuation.json", lineage)
    verify_lineage(run, result)
