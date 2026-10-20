"""Validation-only stages followed by frozen, fresh-geometry system/mismatch evaluation."""
import copy
import itertools
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile

import numpy as np
import torch

from .continuation import inherit_continuation, plan_continuation, verify_lineage
from .data import prepare, reestimate, tensor_hash
from .experiment import apply_dc, file_hash, nmse, validate_config, write_json
from .research import (aggregate, checkpoint_budgets, diagnostic_rows, export_table, infer, load_model,
    paired_effects, plot_tradeoff, recipe, save_tensor, train_job)
from .sbl import SOLVER_VERSION


STAGES = ("baseline", "curves", "factorial", "reliability", "budget", "diffusion", "final")


class TrainingBudgetBoundary(RuntimeError):
    """Validation has not met its declared solver/training budget; preserve the expected stop."""


def validate_study(config):
    validate_config(config)
    study = config["research"]
    learning, sbl = config["learning"], config["sbl"]
    if (type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 63 - 1000000
            or "screen_seed" not in config["data"] or type(config["data"]["screen_seed"]) is not int
            or type(learning["width"]) is not int or learning["width"] < 4
            or not np.isfinite(learning["learning_rate"]) or learning["learning_rate"] <= 0
            or type(sbl["batch_size"]) is not int or sbl["batch_size"] < 1
            or not np.isfinite(sbl["tolerance"]) or sbl["tolerance"] <= 0
            or type(sbl.get("early_stopping", True)) is not bool
            or config["evaluation"]["bootstrap_samples"] < 2):
        raise ValueError("invalid study seed, learning rate, width, SBL or bootstrap setting")
    for key in ("checkpoints", "baseline_iterations", "sbl_budgets", "averages"):
        values = study[key]
        if not values or any(type(v) is not int or v < 1 for v in values) or values != sorted(set(values)):
            raise ValueError(f"{key} must be sorted unique positive integers")
    points = checkpoint_budgets(config, "diffusion")
    if (any(type(v) is not int or v < 1 for v in points) or points != sorted(set(points))
            or points[:len(study["checkpoints"])] != study["checkpoints"]):
        raise ValueError("diffusion checkpoints must retain the original training checkpoints as a prefix")
    if len(study["baseline_iterations"]) < 2 or len(study["checkpoints"]) < 2 or study["averages"][0] != 1:
        raise ValueError("stability needs two iteration budgets; averaging must include one output")
    for key in ("training_seeds", "final_seeds", "shuffle_seeds"):
        values = study[key]
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"invalid {key}")
    if (set(study["training_seeds"]) & set(study["final_seeds"])):
        raise ValueError("final training initializations must be independent of selection initializations")
    if [g["name"] for g in study["grids"]] != ["coarse", "dense"] or any(
            type(g[k]) is not int or g[k] < 2 for g in study["grids"] for k in ("angle_points", "range_points")):
        raise ValueError("declare coarse and dense physical grids")
    if any(study["grids"][0][k] != sbl[k] for k in ("angle_points", "range_points")):
        raise ValueError("the coarse study grid must equal the training SBL grid")
    limits = study["stability"]
    if (not 0 <= limits["overall_convergence"] <= 1 or not 0 <= limits["context_convergence"] <= 1
            or min(limits["nmse_db"], limits["variance_relative"], study["accuracy_tolerance_db"]) < 0
            or not np.isfinite(study["training_stability_db"]) or study["training_stability_db"] < 0
            or min(study["eval_every"], study["train_curve_samples"]) < 1):
        raise ValueError("invalid stability/accuracy or curve settings")
    communication = study["communication"]
    if communication["coherence_symbols"] <= 0 or communication["switch_symbols"] < 0:
        raise ValueError("invalid communication block or switching overhead")
    for size in study["expansions"]:
        if (re.fullmatch(r"[A-Za-z0-9_-]+", size["name"]) is None
                or min(size["antennas"], size["tones"]) < 4):
            raise ValueError("invalid system expansion")
    names = [p["name"] for p in study["expansions"]]
    if len(set(["base", *names])) != len(names) + 1:
        raise ValueError("duplicate system profile names")
    names = []
    for mismatch in study["mismatches"]:
        if (re.fullmatch(r"[A-Za-z0-9_-]+", mismatch["name"]) is None
                or not set(mismatch["channel"]).issubset(config["channel"])):
            raise ValueError("mismatch may override known channel-generation fields only")
        candidate = copy.deepcopy(config)
        candidate["channel"].update(mismatch["channel"])
        validate_config(candidate)
        names.append(mismatch["name"])
    if len(set(["matched", *names])) != len(names) + 1:
        raise ValueError("duplicate mismatch names")


def stable_budget(previous, current, limits):
    changes = {"maximum_nmse_change_db": max(abs(a["mean_nmse_db"] - b["mean_nmse_db"]) for a, b in zip(previous, current)),
        "maximum_variance_relative_change": max(abs(a["mean_variance"] / b["mean_variance"] - 1) for a, b in zip(previous, current)),
        "overall_convergence": min(float(np.mean([r["convergence_rate"] for r in rows])) for rows in (previous, current)),
        "minimum_context_convergence": min(r["convergence_rate"] for r in [*previous, *current]),
        "full_budgets_executed": all(r["full_budget_executed"] for r in [*previous, *current])}
    passed = (changes["maximum_nmse_change_db"] <= limits["nmse_db"]
        and changes["maximum_variance_relative_change"] <= limits["variance_relative"]
        and changes["overall_convergence"] >= limits["overall_convergence"]
        and changes["minimum_context_convergence"] >= limits["context_convergence"]
        and changes["full_budgets_executed"])
    return passed, changes


class Study:
    def __init__(self, root, config_path, run_id, device_override=None, parent_id=None,
            continue_from=None, diffusion_checkpoints=None):
        if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) is None or (parent_id is not None and
                re.fullmatch(r"[A-Za-z0-9_-]{1,80}", parent_id) is None) or (continue_from is not None and
                re.fullmatch(r"[A-Za-z0-9_-]{1,80}", continue_from) is None):
            raise ValueError("invalid run ID")
        self.root, self.run_id = Path(root), run_id
        self.path, self.export = self.root / "runs" / run_id, self.root / "results" / run_id
        plan = None
        if continue_from:
            if config_path is not None or parent_id is not None or continue_from == run_id:
                raise ValueError("continuation needs a new run ID and cannot combine --config or --parent")
            if self.path.exists():
                if not (self.path / "continuation.json").exists():
                    raise ValueError("incomplete continuation initialization; use a new run ID")
                lineage = verify_lineage(self.path, self.export)
                if lineage["parent_run_id"] != continue_from or lineage["added_checkpoints"] != diffusion_checkpoints:
                    raise ValueError("continuation parent/checkpoints changed; use a new run ID")
                self.config = json.loads((self.path / "config_resolved.json").read_text())
                parent_id = lineage["parent_id"]
            else:
                plan = plan_continuation(self.root, continue_from, diffusion_checkpoints)
                self.config, parent_id = plan["config"], plan["parent_id"]
            if device_override and device_override != self.config["device"]:
                raise ValueError("continuation must preserve the parent's RNG device")
        else:
            if config_path is None or diffusion_checkpoints is not None:
                raise ValueError("select --config, or --continue-from with new --diffusion-checkpoints")
            self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if device_override:
            self.config["device"] = device_override
        validate_study(self.config)
        self.device = torch.device(self.config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select CPU explicitly for smoke testing")
        self.started = time.time()
        self.parent_id = parent_id
        identity = {"config": self.config, "parent_id": parent_id}
        if continue_from:
            identity["continue_from"] = continue_from
        if self.path.exists():
            if json.loads((self.path / "study_identity.json").read_text()) != identity:
                raise ValueError("study configuration/parent changed; use a new run ID")
            manifest = json.loads((self.path / "source_manifest.json").read_text())
            current = {p.relative_to(self.root).as_posix(): file_hash(p) for folder in ("src", "scripts")
                for p in (self.root / folder).rglob("*.py") if "__pycache__" not in p.parts}
            archived = {k: v for k, v in manifest.items() if k.startswith(("src/", "scripts/")) and k.endswith(".py")}
            if current != archived:
                raise ValueError("study source changed; use a new run ID")
            self.export.mkdir(parents=True, exist_ok=True)
        else:
            self.path.mkdir(parents=True)
            self.export.mkdir(parents=True, exist_ok=False)
            for name in ("cache", "models", "stages", "predictions"):
                (self.path / name).mkdir()
            write_json(self.path / "study_identity.json", identity)
            write_json(self.path / "config_resolved.json", self.config)
            self.archive()
            if plan is not None:
                inherit_continuation(self.root, self.path, self.export, plan)
        self.hashes = {}

    def archive(self):
        sources = [self.root / name for name in ("AGENTS.md", "README.md", "pyproject.toml", ".gitignore", ".gitattributes")]
        sources += [p for folder in ("src", "scripts", "tests", "configs", "docs") for p in (self.root / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts]
        with zipfile.ZipFile(self.path / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sources:
                archive.write(p, p.relative_to(self.root).as_posix())
        write_json(self.path / "source_manifest.json", {p.relative_to(self.root).as_posix(): file_hash(p) for p in sources})
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True, text=True)
        write_json(self.path / "environment.json", {"python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "device": str(self.device),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "command": sys.argv, "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "source_zip_sha256": file_hash(self.path / "source.zip")})
        for name in ("source.zip", "source_manifest.json", "environment.json", "config_resolved.json", "study_identity.json"):
            (self.export / name).write_bytes((self.path / name).read_bytes())

    def log(self, event, **fields):
        record = {"event": event, "elapsed_seconds_this_process": round(time.time() - self.started, 3), **fields}
        with (self.path / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def cache(self, name, config, source=None, splits=None):
        path = self.path / "cache" / f"{name}.pt"
        identity = {"config": config, "source": source["_cache_id"] if source else None, "splits": splits,
            "parent_id": self.parent_id if name == "base_observations" else None}
        meta = path.with_suffix(".json")
        if path.exists():
            expected = json.loads(meta.read_text())
            if expected["identity"] != identity or file_hash(path) != expected["sha256"]:
                raise ValueError("cached data identity/hash mismatch")
            value = torch.load(path, map_location="cpu", weights_only=True)
        else:
            if source is not None:
                value = reestimate(config, source, self.device, self.log, splits=splits)
            elif self.parent_id and name == "base_observations":
                parent = self.root / "runs" / self.parent_id
                parent_config = json.loads((parent / "config_resolved.json").read_text())
                for key in ("seed", "system", "channel"):
                    if parent_config[key] != config[key]:
                        raise ValueError(f"parent physical configuration differs: {key}")
                if any(parent_config["data"][s] != config["data"][s] for s in ("train", "validation")):
                    raise ValueError("parent scene counts differ")
                value = torch.load(parent / "data.pt", map_location="cpu", weights_only=True)
                value["splits"] = {s: value["splits"][s] for s in ("train", "validation")}
                value["geometry_hashes"] = {s: value["geometry_hashes"][s] for s in value["splits"]}
                value = reestimate(config, value, self.device, self.log)
                write_json(self.path / "parent_provenance.json", {"parent_id": self.parent_id,
                    "data_sha256": file_hash(parent / "data.pt"), "splits_used": ["train", "validation"]})
            else:
                value = prepare(config, self.device, self.log, split_names=("train", "validation"))
            value["_cache_id"] = name
            save_tensor(path, value)
            write_json(meta, {"identity": identity, "sha256": file_hash(path)})
        self.hashes[name] = json.loads(meta.read_text())["sha256"]
        return value

    def get_cache(self, name):
        path = self.path / "cache" / f"{name}.pt"
        expected = json.loads(path.with_suffix(".json").read_text())["sha256"]
        if file_hash(path) != expected:
            raise ValueError("cached data hash mismatch")
        self.hashes[name] = expected
        return torch.load(path, map_location="cpu", weights_only=True)

    def train(self, cache, spec, seed, steps):
        return train_job(self.config, cache, cache["_cache_id"], self.hashes[cache["_cache_id"]], spec, seed, steps,
            self.path / "models", self.device, self.log)

    def score(self, cache, checkpoint, method, *, averages=(1,), shuffle_seed=None):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model, rows = load_model(saved, self.device), []
        for index, entry in enumerate(cache["splits"]["validation"]):
            outputs = infer(self.config, cache, model, saved["recipe"], entry, self.device,
                seed=self.config["seed"] + 7000 + index * 1000, averages=averages, shuffle_seed=shuffle_seed)
            for count, (output, milliseconds, calls) in outputs.items():
                label = method if len(averages) == 1 else f"{method}_avg{count}"
                rows += diagnostic_rows(self.config, cache, entry, cache["contexts"][index], output,
                    label, saved["seed"], milliseconds, calls)
        del model
        return rows

    def result(self, stage):
        value = json.loads((self.path / "stages" / f"{stage}.json").read_text())
        if value["status"] != "complete":
            raise RuntimeError(f"finish stage {stage} before continuing")
        return value["selection"]

    def table(self, stage, rows):
        folder = self.export / stage
        folder.mkdir(exist_ok=True)
        export_table(folder, "metrics.csv", rows)
        export_table(folder, "summary.csv", aggregate(rows))
        return folder

    def profile_baselines(self, prefix, config, observations):
        selections, all_rows, diagnostics = {}, [], []
        for grid in config["research"]["grids"]:
            previous, selected = None, None
            for iterations in config["research"]["baseline_iterations"]:
                candidate = copy.deepcopy(config)
                candidate["sbl"].update({k: grid[k] for k in ("angle_points", "range_points")},
                    iterations=iterations, early_stopping=False)
                cache = self.cache(f"{prefix}_{grid['name']}_v{iterations}", candidate, observations, ["validation"])
                current = []
                for context, entry in enumerate(cache["splits"]["validation"]):
                    values = nmse(entry["truth"], entry["coarse"])
                    row = {"profile": prefix, "grid": grid["name"], "iterations": iterations, "context": context,
                        "mean_nmse_db": float(10 * values.mean().log10()),
                        "mean_variance": float(entry["reliability"].mean()),
                        "convergence_rate": float(entry["sbl_converged"].float().mean()),
                        "full_budget_executed": bool((entry["sbl_iterations"] == iterations).all()),
                        "minimum_actual_iterations": int(entry["sbl_iterations"].min()),
                        "maximum_optimality_residual": float(entry["sbl_optimality_residual"].max()),
                        "mean_objective_per_tone": float(entry["sbl_objective_per_tone"].mean()),
                        "ms_per_sample": entry["coarse_ms_per_sample"]}
                    current.append(row)
                    all_rows += [{**row, "scene": i, "nmse_linear": float(v),
                        "actual_iterations": int(entry["sbl_iterations"][i]),
                        "optimality_residual": float(entry["sbl_optimality_residual"][i]),
                        "objective_per_tone": float(entry["sbl_objective_per_tone"][i]),
                        "observation_sha256": tensor_hash(entry["observation"][i])} for i, v in enumerate(values)]
                diagnostics += current
                if previous is not None:
                    passed, changes = stable_budget(previous, current, config["research"]["stability"])
                    if passed:
                        selected = {"iterations": previous[0]["iterations"], "checked_against": iterations,
                            "grid": grid, **changes}
                        break
                previous = current
            selections[grid["name"]] = selected
        folder = self.export / f"{prefix}_baseline"
        folder.mkdir(exist_ok=True)
        export_table(folder, "metrics.csv", all_rows)
        export_table(folder, "summary.csv", diagnostics)
        write_json(folder / "selection.json", selections)
        if any(v is None for v in selections.values()):
            raise TrainingBudgetBoundary("no stable SBL budget for at least one grid; inspect baseline tables and extend budgets in a new study")
        return selections

    def baseline(self):
        config = copy.deepcopy(self.config)
        config["sbl"]["iterations"] = config["research"]["baseline_iterations"][0]
        original = self.cache("base_observations", config)
        selections = self.profile_baselines("base", self.config, original)
        chosen = copy.deepcopy(self.config)
        chosen["sbl"].update(iterations=selections["coarse"]["iterations"], early_stopping=False)
        chosen["sbl"].update({k: selections["coarse"]["grid"][k] for k in ("angle_points", "range_points")})
        cache = self.cache("base_stable", chosen, original)
        return {"baselines": selections, "cache_id": cache["_cache_id"], "screening_used": False}

    def curves(self):
        cache = self.get_cache(self.result("baseline")["cache_id"])
        study, scores = self.config["research"], []
        curve_rows, loss_rows = [], []
        for seed in study["training_seeds"]:
            mode = "none"
            spec = recipe(self.config, loss="condition", reliability=mode)
            last = self.train(cache, spec, seed, max(study["checkpoints"]))
            curve_rows += [{**json.loads(line), "reliability": mode} for line in (last.parent / "curve.jsonl").read_text().splitlines()]
            loss_rows += [{**json.loads(line), "reliability": mode, "seed": seed}
                for line in (last.parent / "loss.jsonl").read_text().splitlines()]
            for step in study["checkpoints"]:
                scores += self.score(cache, last.parent / f"step_{step}.pt", f"steps_{step}_{mode}")
        folder = self.table("curves", scores)
        export_table(folder, "training_validation_curves.csv", curve_rows)
        export_table(folder, "training_loss.csv", loss_rows)
        means = {step: float(np.mean([r["nmse_linear"] for r in scores if r["method"].startswith(f"steps_{step}_")]))
            for step in study["checkpoints"]}
        chosen = min(means, key=lambda v: (means[v], v))
        self.plot_curves(folder, curve_rows)
        last, before = study["checkpoints"][-1], study["checkpoints"][-2]
        improvement = float(10 * np.log10(means[before] / means[last]))
        if chosen == last and improvement > study["training_stability_db"]:
            write_json(folder / "budget_boundary.json", {"last_interval_improvement_db": improvement,
                "required_maximum_db": study["training_stability_db"], "validation_only": True})
            raise TrainingBudgetBoundary("deterministic validation still improves materially at the largest training budget; extend checkpoints in a new study")
        return {"steps": chosen, "validation_mean_nmse_linear": means,
            "best_at_largest_budget": chosen == max(means), "last_interval_improvement_db": improvement,
            "learning_rate_horizon": max(study["checkpoints"])}

    def factorial(self):
        steps = self.result("curves")["steps"]
        return {**recipe(self.config, loss="condition", reliability="none"), "steps": steps,
            "selection": "fixed cross-frequency/condition-loss recipe; retain only the no-reliability refinement"}

    def reliability(self):
        return self.result("factorial")

    def budget(self):
        original = self.get_cache("base_observations")
        baseline, chosen = self.result("baseline"), self.result("reliability")
        budgets = sorted(set([*self.config["research"]["sbl_budgets"], baseline["baselines"]["coarse"]["iterations"]]))
        rows, candidates = [], []
        for iterations in budgets:
            config = copy.deepcopy(self.config)
            config["sbl"]["iterations"] = iterations
            cache = self.cache(f"budget_{iterations}", config, original)
            local = []
            for seed in self.config["research"]["training_seeds"]:
                spec = recipe(self.config, chosen["architecture"], chosen["loss"], chosen["reliability"])
                checkpoint = self.train(cache, spec, seed, chosen["steps"])
                local += self.score(cache, checkpoint, f"refine_{iterations}")
                for index, entry in enumerate(cache["splits"]["validation"]):
                    rows += diagnostic_rows(self.config, cache, entry, cache["contexts"][index], entry["coarse"],
                        f"sbl_{iterations}", seed, 0., 0)
            rows += local
            candidates.append({"iterations": iterations, "cache_id": cache["_cache_id"],
                "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in local])),
                "mean_total_ms": float(np.mean([r["total_batch_ms_per_sample"] for r in local]))})
        best = min(c["mean_nmse_linear"] for c in candidates)
        eligible = [c for c in candidates if 10 * np.log10(c["mean_nmse_linear"] / best) <= self.config["research"]["accuracy_tolerance_db"]]
        selected = min(eligible, key=lambda c: (c["mean_total_ms"], c["iterations"]))
        folder = self.table("budget", rows)
        plot_tradeoff(folder, rows)
        return {**chosen, **selected, "candidates": candidates,
            "rule": "least measured total amortized time within declared dB tolerance of the best validation refinement"}

    def diffusion(self):
        chosen, rows, candidates = self.result("budget"), [], []
        cache = self.get_cache(chosen["cache_id"])
        modes = sorted({"none", chosen["reliability"]})
        checkpoints = checkpoint_budgets(self.config, "diffusion")
        for mode in modes:
            spec = recipe(self.config, chosen["architecture"], chosen["loss"], mode, "diffusion")
            for seed in self.config["research"]["training_seeds"]:
                last = self.train(cache, spec, seed, max(checkpoints))
                for step in checkpoints:
                    rows += self.score(cache, last.parent / f"step_{step}.pt", f"diffusion_{mode}_step{step}",
                        averages=self.config["research"]["averages"])
            for step, count in itertools.product(checkpoints, self.config["research"]["averages"]):
                label = f"diffusion_{mode}_step{step}_avg{count}"
                if len(self.config["research"]["averages"]) == 1:
                    label = f"diffusion_{mode}_step{step}"
                local = [r for r in rows if r["method"] == label]
                candidates.append({"mode": mode, "steps": step, "average": count,
                    "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in local])),
                    "mean_total_ms": float(np.mean([r["total_batch_ms_per_sample"] for r in local]))})
        folder = self.table("diffusion", rows)
        plot_tradeoff(folder, rows)
        selected = min(candidates, key=lambda c: (c["mean_nmse_linear"], c["mean_total_ms"], c["steps"]))
        before, last = checkpoints[-2:]
        previous = next(c for c in candidates if c["mode"] == selected["mode"] and c["average"] == selected["average"] and c["steps"] == before)
        latest = next(c for c in candidates if c["mode"] == selected["mode"] and c["average"] == selected["average"] and c["steps"] == last)
        improvement = float(10 * np.log10(previous["mean_nmse_linear"] / latest["mean_nmse_linear"]))
        if selected["steps"] == last and improvement > self.config["research"]["training_stability_db"]:
            write_json(folder / "budget_boundary.json", {"last_interval_improvement_db": improvement,
                "required_maximum_db": self.config["research"]["training_stability_db"],
                "interval_steps": [before, last], "selected": selected, "validation_only": True})
            raise TrainingBudgetBoundary("diffusion validation still improves materially at the largest training budget; continue from this run into a new study")
        return {"selected": selected, "candidates": candidates,
            "best_at_largest_budget": selected["steps"] == last,
            "last_interval_improvement_db": improvement,
            "interval_steps": [before, last], "learning_rate_horizon": max(self.config["research"]["checkpoints"]),
            "deterministic_validation_mean_nmse_linear": chosen["mean_nmse_linear"]}

    def final(self):
        chosen, diffusion = self.result("budget"), self.result("diffusion")
        study = self.config["research"]
        profiles = [{"name": "base", "antennas": self.config["system"]["antennas"], "tones": self.config["system"]["tones"]}, *study["expansions"]]
        completed = []
        for number, size in enumerate(profiles):
            folder = self.export / "final" / size["name"]
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "complete.json").exists():
                completed.append(json.loads((folder / "complete.json").read_text()))
                continue
            config = copy.deepcopy(self.config)
            config["system"].update(antennas=size["antennas"], tones=size["tones"])
            ratio = ((size["antennas"] - 1) / (self.config["system"]["antennas"] - 1)) ** 2
            for key in ("range_min_m", "range_max_m"):
                config["channel"][key] *= ratio
            config["sbl"]["iterations"] = chosen["iterations"]
            if size["name"] == "base":
                cache = self.get_cache(chosen["cache_id"])
                baselines = self.result("baseline")["baselines"]
            else:
                cache = self.cache(f"{size['name']}_observations", config)
                baselines = self.profile_baselines(size["name"], config, cache)
            policies = []
            modes = ("none",)
            for mode in modes:
                policies.append((f"deterministic_{mode}", recipe(self.config, chosen["architecture"], chosen["loss"], mode), chosen["steps"], (1,)))
            for mode in sorted({"none", diffusion["selected"]["mode"]}):
                policies.append((f"diffusion_{mode}", recipe(self.config, chosen["architecture"], chosen["loss"], mode, "diffusion"),
                    diffusion["selected"]["steps"], tuple(study["averages"])))
            models = {}
            for label, spec, steps, averages in policies:
                for seed in study["final_seeds"]:
                    checkpoint = self.train(cache, spec, seed, steps)
                    models[f"{label}_{seed}"] = {"path": checkpoint.relative_to(self.path).as_posix(),
                        "sha256": file_hash(checkpoint), "recipe": spec, "steps": steps, "averages": list(averages),
                        "seed": seed, "label": label}
            scores = {str(v): [] for v in config["evaluation"]["dc_multipliers"]}
            for seed in study["final_seeds"]:
                item = models[f"deterministic_none_{seed}"]
                saved = torch.load(self.path / item["path"], map_location="cpu", weights_only=True)
                model = load_model(saved, self.device)
                for index, entry in enumerate(cache["splits"]["validation"]):
                    output = infer(config, cache, model, saved["recipe"], entry, self.device,
                        seed=config["seed"] + 7000 + index * 1000)[1][0]
                    for value in config["evaluation"]["dc_multipliers"]:
                        corrected, _ = apply_dc(output, entry, cache["contexts"][index], value * cache["dc_reference_complex_variance"], self.device)
                        scores[str(value)] += nmse(entry["truth"], corrected).tolist()
                del model
            scores = {k: float(np.mean(v)) for k, v in scores.items()}
            dc = min(scores, key=lambda v: (scores[v], float(v)))
            frozen = {"profile": size, "profile_config": config, "models": models, "selected_recipe": chosen,
                "baselines": baselines, "training_cache_sha256": self.hashes[cache["_cache_id"]],
                "training_cache_id": cache["_cache_id"],
                "calibration": {k: cache[k] for k in ("channel_scale", "residual_real_scale", "reliability_scale", "reliability_log_mean", "reliability_log_std")},
                "dc_multiplier": float(dc), "dc_validation_scores": scores, "selection_split": "validation", "screening_used": False,
                "frozen_at_unix": time.time(), "screen_seeds": [config["data"]["screen_seed"] + 100 * number + i for i in range(1 + len(study["mismatches"]))]}
            freeze_path = self.path / "stages" / f"frozen_{size['name']}.json"
            if freeze_path.exists():
                old = json.loads(freeze_path.read_text())
                for key in ("models", "training_cache_sha256", "selected_recipe", "dc_validation_scores"):
                    if old[key] != frozen[key]:
                        raise ValueError("frozen validation decision changed on resume")
                frozen = old
            else:
                write_json(freeze_path, frozen)
            (folder / "frozen_validation.json").write_bytes(freeze_path.read_bytes())
            self.log("frozen", profile=size["name"], sha256=file_hash(freeze_path))
            for scenario_index, scenario in enumerate([{"name": "matched", "channel": {}}, *study["mismatches"]]):
                self.evaluate_profile(config, cache, models, frozen, size["name"], scenario, scenario_index, folder)
            complete = {"profile": size["name"], "status": "complete", "training_seeds": len(study["final_seeds"]),
                "freeze_sha256": file_hash(freeze_path), "rayleigh_distance_m": cache["contexts"][0]["rayleigh_distance_m"],
                "distance_range_m": [config["channel"]["range_min_m"], config["channel"]["range_max_m"]]}
            write_json(folder / "complete.json", complete)
            completed.append(complete)
        return {"profiles": completed, "test_selection_used": False}

    def evaluate_profile(self, config, cache, models, frozen, name, scenario, index, folder):
        output = folder / scenario["name"]
        output.mkdir(exist_ok=True)
        if (output / "complete.json").exists():
            return
        test_config = copy.deepcopy(config)
        test_config["data"]["screen_seed"] = frozen["screen_seeds"][index]
        test_config["channel"].update(scenario["channel"])
        # Range mismatch is relative to the profile's Rayleigh-scaled training range.
        for key in ("range_min_m", "range_max_m"):
            if key in scenario["channel"]:
                test_config["channel"][key] = scenario["channel"][key] * config["channel"][key] / self.config["channel"][key]
        observed_path = self.path / "predictions" / f"{name}_{scenario['name']}_observations.pt"
        if observed_path.exists():
            observations = torch.load(observed_path, map_location="cpu", weights_only=True)
        else:
            observations = prepare(test_config, self.device, self.log, split_names=("screen",), reference=cache, estimate=False)
            save_tensor(observed_path, observations)
        estimates = {"sbl_budget": reestimate(config, observations, self.device, self.log)}
        for grid, label in (("coarse", "sbl_stable"), ("dense", "sbl_dense")):
            candidate = copy.deepcopy(config)
            settings = frozen["baselines"][grid]
            candidate["sbl"].update(iterations=settings["iterations"])
            candidate["sbl"].update({k: settings["grid"][k] for k in ("angle_points", "range_points")})
            estimates[label] = reestimate(candidate, observations, self.device, self.log)
        working = estimates["sbl_budget"]
        rows, prediction_hashes = [], {}
        dc_variance = frozen["dc_multiplier"] * cache["dc_reference_complex_variance"]
        for label, reference in estimates.items():
            for seed in self.config["research"]["final_seeds"]:
                for c, entry in enumerate(reference["splits"]["screen"]):
                    for suffix, strength in (("", 0.), ("_dc", dc_variance)):
                        rows += diagnostic_rows(config, reference, entry, reference["contexts"][c], entry["coarse"],
                            label + suffix, seed, 0., 0, dc_variance=strength)
        for item in models.values():
            checkpoint = self.path / item["path"]
            if file_hash(checkpoint) != item["sha256"]:
                raise ValueError("frozen model hash mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = load_model(saved, self.device)
            predictions = []
            for c, entry in enumerate(working["splits"]["screen"]):
                outputs = infer(config, cache, model, item["recipe"], entry, self.device,
                    seed=config["seed"] + 17000 + c * 1000, averages=item["averages"])
                predictions.append({count: value[0] for count, value in outputs.items()})
                for count, (prediction, milliseconds, calls) in outputs.items():
                    label = item["label"] + (f"_avg{count}" if item["recipe"]["kind"] == "diffusion" else "")
                    for suffix, strength in (("", 0.), ("_dc", dc_variance)):
                        rows += diagnostic_rows(config, working, entry, working["contexts"][c], prediction,
                            label + suffix, item["seed"], milliseconds, calls, dc_variance=strength)
            saved_path = self.path / "predictions" / f"{name}_{scenario['name']}_{item['label']}_{item['seed']}.pt"
            save_tensor(saved_path, predictions)
            prediction_hashes[saved_path.name] = file_hash(saved_path)
            del model
        for label, reference in estimates.items():
            path = self.path / "predictions" / f"{name}_{scenario['name']}_{label}.pt"
            save_tensor(path, reference)
            prediction_hashes[path.name] = file_hash(path)
        observation_hashes = {}
        for row in rows:
            key = row["context"], row["scene"], row["seed"]
            if observation_hashes.setdefault(key, row["observation_sha256"]) != row["observation_sha256"]:
                raise AssertionError("methods did not share the same physical observation")
        export_table(output, "metrics.csv", rows)
        export_table(output, "summary.csv", aggregate(rows))
        export_table(output, "paired_effects.csv", paired_effects(rows, config["evaluation"]["bootstrap_samples"], config["seed"] + 71))
        plot_tradeoff(output, rows)
        write_json(output / "complete.json", {"status": "complete", "metrics_rows": len(rows), "scenario": scenario,
            "test_config": test_config, "observation_file_sha256": file_hash(observed_path),
            "geometry_hashes": observations["geometry_hashes"], "prediction_sha256": prediction_hashes,
            "frozen_validation_sha256": file_hash(folder / "frozen_validation.json"), "scored_at_unix": time.time(),
            "communication": "single-user, one RF chain, one shared feasible state chosen from the J pilot states; training plus switching overhead"})
        self.log("screen", profile=name, scenario=scenario["name"], rows=len(rows))

    def plot_curves(self, path, rows):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 4))
        for mode, split in itertools.product(sorted({r["reliability"] for r in rows}), ("train", "validation")):
            steps = sorted({r["step"] for r in rows})
            values = [10 * np.log10(np.mean([r["mean_nmse_linear"] for r in rows
                if r["reliability"] == mode and r["split"] == split and r["step"] == step])) for step in steps]
            axis.plot(steps, values, marker="o", label=f"{mode}: {split}")
        axis.set(xlabel="Training updates (shared schedule)", ylabel="Full CSI linear-mean NMSE (dB)")
        axis.grid(alpha=.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path / "training_curves.png", dpi=160)
        plt.close(figure)

    def run(self, stage):
        stages = STAGES if stage == "all" else (stage,)
        if any(s not in STAGES for s in stages):
            raise ValueError("unknown research stage")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        for name in stages:
            path = self.path / "stages" / f"{name}.json"
            if path.exists() and json.loads(path.read_text())["status"] == "complete":
                self.log("stage_reused", stage=name)
                continue
            for dependency in STAGES[:STAGES.index(name)]:
                self.result(dependency)
            def timeout():
                write_json(path, {"status": "timeout", "limit_seconds": self.config["timeout_seconds"]})
                os._exit(124)
            timer = threading.Timer(self.config["timeout_seconds"], timeout)
            timer.daemon = True
            timer.start()
            try:
                write_json(path, {"status": "running", "started_at_unix": time.time()})
                self.log("stage_started", stage=name)
                selected = getattr(self, name)()
                write_json(path, {"status": "complete", "selection": selected, "completed_at_unix": time.time()})
                (self.export / f"{name}_selection.json").write_bytes(path.read_bytes())
                self.log("stage_complete", stage=name, selection=selected)
            except TrainingBudgetBoundary as error:
                write_json(path, {"status": "budget_limited", "error": str(error), "at_unix": time.time()})
                (self.export / f"{name}_selection.json").write_bytes(path.read_bytes())
                self.log("budget_limited", stage=name, reason=str(error))
                self.write_report()
                raise
            except Exception as error:
                write_json(path, {"status": "failed", "error": str(error), "at_unix": time.time()})
                (self.path / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            finally:
                timer.cancel()
        self.write_report()
        return self.export

    def write_report(self):
        states = {p.stem: json.loads(p.read_text()) for p in (self.path / "stages").glob("*.json") if p.stem in STAGES}
        lines = ["# 有界场景方法比较执行记录", "", f"运行：{self.run_id}。标签：{self.config['label']}。", "",
            f"SBL求解器：{SOLVER_VERSION}。基线检查强制执行完整预算，收敛标记来自边缘似然的一阶条件。",
            "确定性精修只保留无可靠度版本，采用跨频卷积和按条件归一化的训练损失。", "",
            "各选择只使用训练/验证数据；最终各系统在保存模型和参数哈希后生成新的筛选几何。",
            "独立重复为训练初始化种子和几何场景；同一几何的带宽、配置与SNR按块配对。", "",
            "| 阶段 | 状态 |", "|---|---|"]
        lines += [f"| {s} | {states.get(s, {}).get('status', 'pending')} |" for s in STAGES]
        lines += ["", "训练、验证曲线和精度—时间图保存在相应阶段目录。主NMSE先在线性尺度汇总再转dB。",
            "修正能量与真实残差对齐只用于评分诊断。可靠度标定和条件损失权重仅来自训练集。",
            "最终通信指标是单用户、单接收RF链、跨频共享DMA状态的接收速率，扣除导频与配置切换开销。",
            "所报时间包含SBL和精修的批量均摊时间，排除编译预热；不等同于单样本时延。", ""]
        if "smoke" in self.config["label"]:
            lines += ["这是小规模流程验证，迭代和训练预算已缩短；数值不能作为方法性能或论文结论。", ""]
        for p in sorted((self.export / "final").glob("*/*/summary.csv")):
            lines.append(f"- [{p.parent.parent.name}/{p.parent.name}]({p.relative_to(self.export).as_posix()})")
        (self.export / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
