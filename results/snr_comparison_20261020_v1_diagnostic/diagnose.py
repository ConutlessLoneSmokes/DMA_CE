"""Post-hoc diagnosis: frozen denoising stages and equal-MSE independent Gaussian errors."""
import json
import os
from pathlib import Path
import platform
import shutil
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from dma_ce._upstream.ddpm import DenoisingCNN, VPSchedule
from dma_ce.data import encode, decode
from dma_ce.experiment import file_hash, metric_rows, write_csv, write_json
from dma_ce.snr_study import sweep_cache, verify_frozen_models
from dma_ce.structured import VarianceUpdate, apply_lmmse, gaussian_update, structured_recover


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    run_id = "snr_comparison_20261020_v1"
    original = ROOT / "runs" / run_id
    frozen_path = original / "stages/frozen.json"
    frozen = json.loads(frozen_path.read_text())
    verify_frozen_models(original, frozen)
    cfg = json.loads((original / "config_resolved.json").read_text())
    run = ROOT / "runs" / (run_id + "_diagnostic")
    export = ROOT / "results" / run.name
    run.mkdir(exist_ok=False)
    export.mkdir(exist_ok=False)
    for folder in (run, export):
        shutil.copyfile(__file__, folder / "diagnose.py")
    settings = {"run": run_id, "seeds": cfg["comparison"]["seeds"], "snrs_db": [-10, 10, 20, 30],
        "splits": ["train", "validation"], "scenes_per_split": 64, "noise_seed": 20261020,
        "posthoc": True, "selection_or_retraining": False,
        "truth_usage": "stage scoring and synthetic equal-MSE independent Gaussian control only",
        "frozen_sha256": file_hash(frozen_path), "source_sha256": file_hash(Path(__file__)),
        "structured_source_sha256": file_hash(ROOT / "src/dma_ce/structured.py")}
    write_json(run / "config.json", settings)
    write_json(run / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    parent_path = ROOT / "runs" / cfg["comparison"]["parent_run"] / "cache" / (frozen["parent_provenance"]["cache_id"] + ".pt")
    assert file_hash(parent_path) == frozen["parent_provenance"]["cache_sha256"]
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    statistics = torch.load(original / "cache/lmmse_statistics.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    schedule = VPSchedule(cfg["learning"]["diffusion_steps"]).to(device)
    bars = schedule.alpha_bar
    ratios = (1 - bars) / bars
    rows, replay_maximum, checks = [], 0., 0
    for seed in settings["seeds"]:
        item = frozen["models"][str(seed)]
        prior = DenoisingCNN(width=cfg["learning"]["width"]).to(device)
        prior.load_state_dict(torch.load(original / item["prior_path"], map_location="cpu", weights_only=True)["state_dict"])
        prior.eval().requires_grad_(False)
        updater = VarianceUpdate(cfg["comparison"]["update_width"]).to(device)
        updater.load_state_dict(torch.load(original / item["update_path"], map_location="cpu", weights_only=True)["state_dict"])
        updater.eval()
        for split_number, split in enumerate(settings["splits"]):
            cache = sweep_cache(cfg, parent, split, settings["scenes_per_split"], settings["snrs_db"])
            for context, entry in zip(cache["contexts"], cache["splits"][split]):
                y, a, variance = entry["observation"].to(device), context["matrix"].to(device), entry["variance"].to(device)
                stats = statistics[context["bandwidth_hz"], context["configurations"], context["snr_db"]]
                initial = apply_lmmse(stats, y)
                diagonal = stats["diagonal"].to(device=device, dtype=y.real.dtype).clamp_min(1e-8)[None].expand_as(initial.real)
                truth = entry["truth"].to(device)
                count = len(y)
                scoring_entry = {**entry, "sbl_converged": torch.zeros(count, dtype=torch.bool),
                    "sbl_iterations": torch.zeros(count, dtype=torch.int64), "sbl_relative_change": torch.zeros(count), "coarse_ms_per_sample": 0.}
                def record(output, variant, stage):
                    values = list(metric_rows(cfg, scoring_entry, context, output.cpu(), stage, 0., 0))
                    rows.append({"seed": seed, "split": split, "context": entry["context_index"],
                        "bandwidth_hz": context["bandwidth_hz"], "configurations": context["configurations"],
                        "snr_db": context["snr_db"], "variant": variant, "stage": stage, "geometries": count,
                        "mean_nmse_linear": float(np.mean([r["nmse_linear"] for r in values])),
                        "mean_mse": float((output - truth).abs().square().mean()),
                        "mean_row_nmse": float(np.mean([r["row_nmse_linear"] for r in values])),
                        "mean_null_nmse": float(np.mean([r["null_nmse_linear"] for r in values])),
                        "mean_observation_residual": float(np.mean([r["observation_residual"] for r in values])),
                        "initial_statistical_variance": float(diagonal.mean())})
                def denoise(current, target):
                    indices = (ratios[None] - target.mean((1, 2))[:, None] / 2).abs().argmin(-1)
                    alpha = bars[indices].to(current.real)[:, None, None, None]
                    image = alpha.sqrt() * encode(current)
                    return decode((image - (1 - alpha).sqrt() * prior(image, indices + 1)) / alpha.sqrt())
                record(initial, "initial", "lmmse")
                generator = torch.Generator().manual_seed(settings["noise_seed"] + seed * 1000 + split_number * 100 + entry["context_index"])
                noise = torch.randn(truth.shape, generator=generator, dtype=truth.dtype).to(device)
                noise *= ((initial - truth).abs().square().mean() / noise.abs().square().mean()).sqrt()
                synthetic = truth + noise
                assert torch.allclose((synthetic-truth).abs().square().mean(), (initial-truth).abs().square().mean(), rtol=2e-6, atol=1e-8)
                record(synthetic, "equal_mse_independent_gaussian", "input")
                record(denoise(synthetic, diagonal * item["strength"]), "equal_mse_independent_gaussian", "denoised")
                for variant, network in (("fixed", None), ("learned", updater)):
                    current = initial
                    for iteration in range(cfg["comparison"]["structured_iterations"]):
                        target = diagonal * (item["strength"] * .5 ** iteration)
                        current = denoise(current, target)
                        record(current, variant, f"denoised_{iteration+1}")
                        if network is not None:
                            residual = y - torch.einsum("kjn,bkn->bkj", a, current)
                            target = network(current, target, torch.einsum("kjn,bkj->bkn", a.conj(), residual), variance)
                        current = gaussian_update(current, a, y, variance, target)[0]
                        record(current, variant, f"gaussian_{iteration+1}")
                    replay = structured_recover(prior, schedule, a, y, variance, initial, diagonal,
                        updater=network, iterations=cfg["comparison"]["structured_iterations"], strength=item["strength"])[0]
                    difference = float((current-replay).abs().max())
                    assert difference <= 2e-6
                    replay_maximum = max(replay_maximum, difference)
                    checks += 1
        print(json.dumps({"seed_complete": seed, "trace_rows": len(rows)}), flush=True)
    write_csv(export / "stages.csv", rows)
    write_json(export / "audit.json", {"stage_replays": checks, "maximum_difference": replay_maximum,
        "equal_mse_control_checked": True, "row_null_decomposition_checked": True, "models_hash_verified": True})
    for name in ("config.json", "environment.json"):
        shutil.copyfile(run / name, export / name)
    for name in ("stages.csv", "audit.json"):
        shutil.copyfile(export / name, run / name)


if __name__ == "__main__":
    main()
