"""Shared training, paired diagnostics and communication scoring for the staged study."""
import hashlib
from functools import lru_cache
import json
import math
import time

import numpy as np
import torch

from ._upstream.ddpm import VPSchedule
from .conditional import ConditionalCNN, predict, reliability_features
from .data import encode, sync
from .experiment import apply_dc, metric_rows, nmse, write_csv, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_tensor(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@lru_cache(maxsize=None)
def matched_width(width, architecture):
    if architecture == "cross_frequency":
        return width
    target = sum(p.numel() for p in ConditionalCNN(width).parameters())
    sizes = {w: sum(p.numel() for p in ConditionalCNN(w, architecture).parameters()) for w in range(4, 4 * width + 1)}
    return min(sizes, key=lambda w: (abs(sizes[w] - target), w))


def recipe(config, architecture="cross_frequency", loss="uniform", reliability="map", kind="deterministic"):
    return {"architecture": architecture, "loss": loss, "reliability": reliability, "kind": kind,
        "width": matched_width(config["learning"]["width"], architecture)}


def loss_weights(entries, loss):
    if loss not in ("uniform", "condition"):
        raise ValueError("unknown residual loss")
    errors = torch.stack([(e["truth"] - e["coarse"]).abs().square().mean() for e in entries]).clamp_min(1e-12)
    return torch.ones_like(errors) if loss == "uniform" else errors.mean() / errors


def learning_rate(base, step, horizon):
    """One fixed cosine schedule; shortening a run retains the exact longer-run prefix."""
    return base * (.1 + .9 * (1 + math.cos(math.pi * min(step, horizon) / horizon)) / 2)


def load_model(checkpoint, device):
    spec = checkpoint["recipe"]
    model = ConditionalCNN(spec["width"], spec["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.no_grad()
def infer(config, cache, model, spec, entry, device, *, seed, averages=(1,), shuffle_seed=None):
    if not averages or any(type(a) is not int or a < 1 for a in averages):
        raise ValueError("positive integer output averages required")
    deterministic = spec["kind"] == "deterministic"
    if deterministic and tuple(averages) != (1,):
        raise ValueError("deterministic models have one output")
    reliable = reliability_features(entry["reliability"], cache, spec["reliability"])
    if shuffle_seed is not None:
        order = torch.randperm(len(reliable), generator=torch.Generator().manual_seed(shuffle_seed))
        reliable = reliable[order]
    schedule = VPSchedule(config["learning"]["diffusion_steps"]).to(device)
    batch = config["evaluation"]["batch_size"]
    variant = "deterministic" if deterministic else "reliable_residual"
    # Warm up the same network and shape before recording amortized inference time.
    predict(model, schedule, encode(entry["coarse"][:batch]).to(device), reliable[:batch].to(device),
        variant, cache["residual_real_scale"], config["learning"]["sampling_steps"], seed=seed)
    total, outputs, elapsed = None, {}, 0.
    for draw in range(1, max(averages) + 1):
        estimates = []
        sync(device)
        started = time.perf_counter()
        for start in range(0, len(reliable), batch):
            estimates.append(predict(model, schedule, encode(entry["coarse"][start:start + batch]).to(device),
                reliable[start:start + batch].to(device), variant, cache["residual_real_scale"],
                config["learning"]["sampling_steps"], seed=seed + (draw - 1) * 1000003 + start).cpu())
        value = torch.cat(estimates)
        total = value if total is None else total + value
        sync(device)
        elapsed += time.perf_counter() - started
        if draw in averages:
            outputs[draw] = (total / draw, 1000 * elapsed / len(reliable),
                draw * (1 if deterministic else config["learning"]["sampling_steps"]))
    return outputs


def train_job(config, cache, cache_id, cache_hash, spec, seed, steps, models_path, device, log):
    """Resume at an archived optimizer/RNG state; keep data on CPU and move only each minibatch."""
    settings, study = config["learning"], config["research"]
    horizon = max(study["checkpoints"])
    if not 1 <= steps <= horizon:
        raise ValueError("training budget must lie within the declared schedule")
    path = models_path / cache_id / digest(spec)[:12] / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    identity = {"recipe": spec, "seed": seed, "cache_sha256": cache_hash, "horizon": horizon,
        "learning": settings, "eval_every": study["eval_every"]}
    if (path / "identity.json").exists():
        if json.loads((path / "identity.json").read_text()) != identity:
            raise ValueError("training identity changed; use a new study ID")
    else:
        write_json(path / "identity.json", identity)
    target_path = path / f"step_{steps}.pt"
    if target_path.exists():
        return target_path
    torch.manual_seed(seed)
    model = ConditionalCNN(spec["width"], spec["architecture"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    batches = torch.Generator().manual_seed(seed + 1013)
    noises = torch.Generator(device=device).manual_seed(seed + 2017)
    first = 0
    if (path / "last.pt").exists():
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        first = last["step"]
        if first >= steps:
            raise RuntimeError("requested checkpoint is missing behind a later resume state")
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        batches.set_state(last["batch_rng"])
        noises.set_state(last["noise_rng"])
    for name in ("loss.jsonl", "curve.jsonl"):
        if (path / name).exists():
            records = [json.loads(line) for line in (path / name).read_text().splitlines()]
            (path / name).write_text("".join(json.dumps(r) + "\n" for r in records if r["step"] <= first))
    entries = cache["splits"]["train"]
    weights = loss_weights(entries, spec["loss"])
    sizes = [len(e["truth"]) for e in entries]
    bounds = np.cumsum([0, *sizes])
    # Stacking only the current batch also bounds host RAM for 128 x 128 profiles.
    schedule = VPSchedule(settings["diffusion_steps"]).to(device)
    losses = []
    sync(device)
    started = time.perf_counter()
    for step in range(first, steps):
        indices = torch.randint(int(bounds[-1]), (settings["batch_size"],), generator=batches).tolist()
        selected = [(int(np.searchsorted(bounds, i, side="right") - 1), i) for i in indices]
        pairs = [(c, i - int(bounds[c])) for c, i in selected]
        coarse = encode(torch.stack([entries[c]["coarse"][i] for c, i in pairs])).to(device)
        truth = encode(torch.stack([entries[c]["truth"][i] for c, i in pairs])).to(device)
        diagonals = torch.stack([entries[c]["reliability"][i] for c, i in pairs])
        reliable = reliability_features(diagonals, cache, spec["reliability"]).to(device)
        clean = (truth - coarse) / cache["residual_real_scale"]
        if spec["kind"] == "deterministic":
            state, times, target = torch.zeros_like(clean), torch.zeros(len(clean), device=device), clean
        else:
            times = torch.randint(len(schedule.beta), (len(clean),), device=device, generator=noises)
            target = torch.randn(clean.shape, device=device, generator=noises)
            state = schedule.perturb(clean, times, target)
            times = times + 1
        prediction = model(state, coarse, reliable, times)
        per_sample = (prediction - target).square().mean((1, 2, 3))
        loss = (per_sample * weights[[c for c, _ in pairs]].to(device)).mean()
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite research training loss")
        lr = learning_rate(settings["learning_rate"], step, horizon)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        completed = step + 1
        if completed % settings["log_every"] == 0 or completed == steps:
            record = {"step": completed, "loss": float(np.mean(losses)), "learning_rate": lr}
            with (path / "loss.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            log("training", cache=cache_id, recipe=spec, seed=seed, **record)
            losses = []
        evaluate = completed in study["checkpoints"] or completed % study["eval_every"] == 0 or completed == steps
        if evaluate:
            for split in ("train", "validation"):
                for context, entry in enumerate(cache["splits"][split]):
                    if split == "train":
                        count = min(len(entry["truth"]), study["train_curve_samples"])
                        entry = {k: v[:count] if torch.is_tensor(v) and v.ndim else v for k, v in entry.items()}
                    output = infer(config, cache, model, spec, entry, device,
                        seed=config["seed"] + 7000 + 1000 * context)[1][0]
                    values = nmse(entry["truth"], output)
                    record = {"step": completed, "split": split, "context": context, "samples": len(values),
                        "mean_nmse_linear": float(values.mean()), "seed": seed,
                        "single_diffusion_draw": spec["kind"] != "deterministic"}
                    with (path / "curve.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
            model.train()
            state = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "recipe": spec, "seed": seed, "step": completed, "cache_sha256": cache_hash,
                "parameters": sum(p.numel() for p in model.parameters())}
            if completed in study["checkpoints"] or completed == steps:
                save_tensor(path / f"step_{completed}.pt", state)
            save_tensor(path / "last.pt", {**state, "optimizer": optimizer.state_dict(),
                "batch_rng": batches.get_state(), "noise_rng": noises.get_state()})
    sync(device)
    log("training_complete", cache=cache_id, recipe=spec, seed=seed, steps=steps,
        seconds=time.perf_counter() - started)
    return target_path


def communication_rates(estimate, truth, context, variance, communication):
    """One receive RF chain selects ONE feasible shared-frequency DMA state for a data block."""
    matrix = context["matrix"]
    predicted = torch.einsum("kjn,bkn->bkj", matrix, estimate).abs().square() / variance[:, None, None]
    true_power = torch.einsum("kjn,bkn->bkj", matrix, truth).abs().square() / variance[:, None, None]
    chosen = torch.log2(1 + predicted).mean(1).argmax(-1)
    rates_by_state = torch.log2(1 + true_power).mean(1)
    rate = rates_by_state.gather(1, chosen[:, None]).squeeze(1)
    oracle = rates_by_state.max(-1).values
    overhead = context["training_symbols"] + communication["switch_symbols"] * context["switches"]
    fraction = max(0., 1 - overhead / communication["coherence_symbols"])
    return rate, fraction * rate, fraction * oracle


def diagnostic_rows(config, cache, entry, context, output, method, seed, refine_ms, calls, *, dc_variance=0.):
    if dc_variance:
        output, dc_ms = apply_dc(output, entry, context, dc_variance, torch.device(config["device"]))
        refine_ms += dc_ms
    residual, correction = entry["truth"] - entry["coarse"], output - entry["coarse"]
    energy = entry["truth"].abs().square().sum((1, 2)).clamp_min(1e-20)
    added = correction.abs().square().sum((1, 2)) / energy
    aligned = 2 * (correction.conj() * residual).real.sum((1, 2)) / energy
    baseline = nmse(entry["truth"], entry["coarse"])
    if not torch.allclose(nmse(entry["truth"], output) - baseline, added - aligned, atol=1e-5, rtol=2e-4):
        raise AssertionError("residual decomposition does not match full CSI error")
    gross, net, oracle = communication_rates(output, entry["truth"], context, entry["variance"], config["research"]["communication"])
    rows = list(metric_rows(config, entry, context, output, method, refine_ms, calls))
    for scene, row in enumerate(rows):
        row.update(seed=seed, correction_energy_normalized=float(added[scene]), twice_alignment_normalized=float(aligned[scene]),
            gross_rate=float(gross[scene]), net_rate=float(net[scene]), perfect_csi_net_rate=float(oracle[scene]),
            total_batch_ms_per_sample=row["coarse_batch_ms_per_sample"] + refine_ms)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["seed"], row["context"])
        groups.setdefault(key, []).append(row)
    return [{"method": m, "seed": s, "context": c, "samples": len(v),
        "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in v])),
        "mean_nmse_db": float(10 * np.log10(np.mean([r["nmse_linear"] for r in v]))),
        "mean_total_ms": float(np.mean([r["total_batch_ms_per_sample"] for r in v])),
        "convergence_rate": float(np.mean([r["sbl_converged"] for r in v])),
        "mean_net_rate": float(np.mean([r["net_rate"] for r in v])),
        "mean_correction_energy": float(np.mean([r["correction_energy_normalized"] for r in v])),
        "mean_twice_alignment": float(np.mean([r["twice_alignment_normalized"] for r in v]))}
        for (m, s, c), v in sorted(groups.items())]


def paired_effects(rows, repetitions, seed):
    """Resample independent training seeds and geometry blocks; all contexts remain paired."""
    methods = sorted({r["method"] for r in rows})
    lookup = {(r["method"], r["seed"], r["scene"], r["context"]): r["nmse_linear"] for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError("duplicate paired metric rows")
    if "sbl_stable" not in methods:
        return []
    effects = []
    comparisons = [(m, "sbl_stable_dc" if m.endswith("_dc") else "sbl_stable") for m in methods
        if m not in ("sbl_stable", "sbl_stable_dc")]
    for m in methods:
        suffix = "_dc" if m.endswith("_dc") else ""
        if m.startswith("deterministic_") and m != "deterministic_none" + suffix:
            comparisons.append((m, "deterministic_none" + suffix))
        if m.startswith("diffusion_"):
            mode = m.split("_")[1]
            if "deterministic_" + mode + suffix in methods:
                comparisons.append((m, "deterministic_" + mode + suffix))
    for method, reference in comparisons:
        seeds = sorted({r["seed"] for r in rows if r["method"] == method})
        scenes = sorted({r["scene"] for r in rows if r["method"] == method})
        contexts = sorted({r["context"] for r in rows if r["method"] == method})
        a = np.array([[[lookup[method, s, g, c] for c in contexts] for g in scenes] for s in seeds])
        b = np.array([[[lookup[reference, s, g, c] for c in contexts] for g in scenes] for s in seeds])
        rng = np.random.default_rng(seed)
        sampled = []
        for _ in range(repetitions):
            ss = rng.integers(len(seeds), size=len(seeds))
            gg = rng.integers(len(scenes), size=len(scenes))
            sampled.append(10 * np.log10(a[ss][:, gg].mean() / b[ss][:, gg].mean()))
        low, high = np.quantile(sampled, [.025, .975])
        effects.append({"method": method, "reference": reference, "training_seeds": len(seeds),
            "geometry_blocks": len(scenes), "delta_linear_mean_db": float(10 * np.log10(a.mean() / b.mean())),
            "ci95_low_db": float(low), "ci95_high_db": float(high),
            "bootstrap": "crossed_seed_and_geometry_blocks"})
    return effects


def export_table(path, name, rows):
    if rows:
        write_csv(path / name, rows)


def plot_tradeoff(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for method in sorted({r["method"] for r in rows}):
        values = [r for r in rows if r["method"] == method]
        axis.scatter(np.mean([r["total_batch_ms_per_sample"] for r in values]),
            10 * np.log10(np.mean([r["nmse_linear"] for r in values])), label=method, s=24)
    axis.set(xlabel="SBL + refinement, amortized ms/sample", ylabel="Full CSI linear-mean NMSE (dB)")
    axis.grid(alpha=.25)
    axis.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    figure.tight_layout()
    figure.savefig(path / "accuracy_time.png", dpi=160)
    plt.close(figure)
