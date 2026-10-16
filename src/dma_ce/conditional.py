"""Matched conditional CNNs, VP training, and deterministic DDIM inference."""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from ._upstream.ddpm import DenoisingCNN, VPSchedule
from .data import encode, decode, sync


VARIANTS = ("conditional", "residual", "reliable_residual", "deterministic")
SUPPORTED_VARIANTS = VARIANTS + ("deterministic_no_reliability",)


class ConditionalCNN(DenoisingCNN):
    """Reuse the upstream time-modulated CNN; all controls have the same five input channels."""
    def __init__(self, width=32):
        super().__init__(channels=2, width=width, time_dim=16)
        self.pre[0] = nn.Conv2d(5, width // 2, 3, padding=1)

    def forward(self, state, coarse, reliability, step):
        return super().forward(torch.cat((state, coarse, reliability), dim=1), step)


def reliability_features(diagonal, cache):
    value = (diagonal * cache["reliability_scale"]).clamp_min(1e-10).log10()
    value = (value - cache["reliability_log_mean"]) / cache["reliability_log_std"]
    return value.transpose(-1, -2)[:, None].contiguous()


def make_bundle(entries, cache, device):
    return {"truth": torch.cat([encode(x["truth"]) for x in entries]).to(device),
        "coarse": torch.cat([encode(x["coarse"]) for x in entries]).to(device),
        "reliability": torch.cat([reliability_features(x["reliability"], cache) for x in entries]).to(device)}


def training_target(bundle, variant, residual_scale):
    if variant not in SUPPORTED_VARIANTS or residual_scale <= 0:
        raise ValueError("known model and positive training residual scale required")
    return bundle["truth"] if variant == "conditional" else (bundle["truth"] - bundle["coarse"]) / residual_scale


def selected_reliability(bundle, variant):
    return bundle["reliability"] if variant in ("reliable_residual", "deterministic") else torch.zeros_like(bundle["reliability"])


def train_model(config, cache, variant, device, model_path, log):
    settings = config["learning"]
    torch.manual_seed(config["seed"] + 911)
    model = ConditionalCNN(settings["width"]).to(device)
    schedule = VPSchedule(settings["diffusion_steps"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    anneal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["steps"], eta_min=settings["learning_rate"] * .1)
    bundle = make_bundle(cache["splits"]["train"], cache, device)
    target = training_target(bundle, variant, cache["residual_real_scale"])
    reliability = selected_reliability(bundle, variant)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1013)
    sync(device)
    started = time.perf_counter()
    losses = []
    model.train()
    for step in range(settings["steps"]):
        indices = torch.randint(len(target), (settings["batch_size"],), device=device, generator=generator)
        clean = target[indices]
        if variant.startswith("deterministic"):
            state, times = torch.zeros_like(clean), torch.zeros(len(clean), device=device)
            predicted = model(state, bundle["coarse"][indices], reliability[indices], times)
            loss = F.mse_loss(predicted, clean)
        else:
            times = torch.randint(len(schedule.beta), (len(clean),), device=device, generator=generator)
            noise = torch.randn(clean.shape, device=device, generator=generator)
            state = schedule.perturb(clean, times, noise)
            predicted = model(state, bundle["coarse"][indices], reliability[indices], times + 1)
            loss = F.mse_loss(predicted, noise)
        if not torch.isfinite(loss):
            raise ArithmeticError(f"nonfinite training loss for {variant}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        anneal.step()
        losses.append(float(loss.detach()))
        if (step + 1) % settings["log_every"] == 0 or step == 0 or step + 1 == settings["steps"]:
            sync(device)
            log("training", variant=variant, step=step + 1,
                loss=sum(losses[-settings["log_every"]:]) / min(len(losses), settings["log_every"]),
                seconds=round(time.perf_counter() - started, 3))
    sync(device)
    seconds = time.perf_counter() - started
    torch.save({"state_dict": model.cpu().state_dict(), "variant": variant,
        "training_steps": settings["steps"], "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": seconds, "loss_first": losses[0], "loss_last": sum(losses[-100:]) / min(100, len(losses))}, model_path)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, schedule, coarse, reliability, variant, residual_scale, sampling_steps, *, seed):
    """Inputs contain estimates and known reliability only; no target CSI is accepted."""
    if (variant not in SUPPORTED_VARIANTS or coarse.ndim != 4 or coarse.shape[1] != 2
            or reliability.shape != (len(coarse), 1, *coarse.shape[-2:])
            or not torch.isfinite(coarse).all() or not torch.isfinite(reliability).all()
            or not 2 <= sampling_steps <= len(schedule.beta)):
        raise ValueError("compatible finite conditions and valid sampling budget required")
    model.eval()
    condition = reliability if variant in ("reliable_residual", "deterministic") else torch.zeros_like(reliability)
    if variant.startswith("deterministic"):
        target = model(torch.zeros_like(coarse), coarse, condition, torch.zeros(len(coarse), device=coarse.device))
    else:
        generator = torch.Generator(device=coarse.device).manual_seed(seed)
        target = torch.randn(coarse.shape, dtype=coarse.dtype, device=coarse.device, generator=generator)
        times = torch.linspace(len(schedule.beta) - 1, 0, sampling_steps).round().long().tolist()
        for index, step in enumerate(times):
            alpha = float(schedule.alpha_bar[step])
            noise = model(target, coarse, condition, torch.full((len(coarse),), step + 1, device=coarse.device))
            clean = (target - math.sqrt(1 - alpha) * noise) / math.sqrt(alpha)
            previous = float(schedule.alpha_bar[times[index + 1]]) if index + 1 < len(times) else 1.
            target = math.sqrt(previous) * clean + math.sqrt(1 - previous) * noise
            if not torch.isfinite(target).all():
                raise ArithmeticError("nonfinite conditional DDIM trajectory")
    image = target if variant == "conditional" else coarse + residual_scale * target
    return decode(image)
