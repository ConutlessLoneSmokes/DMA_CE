"""Exact sample-covariance shrinkage without a dense (KN) by (KJRF) gain."""
import math

import torch

from .structured import _complex


@torch.no_grad()
def fit(training, matrix, variance, strength):
    if (not _complex(training, 3) or len(training) < 2 or not _complex(matrix, 3)
            or training.shape[1:] != (matrix.shape[0], matrix.shape[2])
            or training.device != matrix.device or not math.isfinite(variance) or variance <= 0
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("compatible complex training data and valid noise/shrinkage required")
    h, a = training.to(torch.complex128), matrix.to(torch.complex128)
    mean = h.mean(0)
    x = ((h - mean) / math.sqrt(len(h) - 1)).permute(1, 2, 0).contiguous()
    z = a @ x
    block = strength * (z @ z.mH) + variance * torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    factor = torch.linalg.cholesky(block)
    inverse_z = torch.cholesky_solve(z, factor)
    small = torch.eye(len(h), dtype=h.dtype, device=h.device) + (1 - strength) * (z.mH @ inverse_z).sum(0)
    return {"mean": mean, "observed_mean": (a @ mean[..., None]).squeeze(-1),
        "x": x, "z": z, "factor": factor, "inverse_z": inverse_z,
        "small_factor": torch.linalg.cholesky(small), "strength": float(strength)}


@torch.no_grad()
def apply(statistics, observation):
    if not _complex(observation, 3) or observation.shape[1:] != statistics["observed_mean"].shape:
        raise ValueError("finite compatible observations required")
    s = {k: v.to(observation.device) if isinstance(v, torch.Tensor) else v for k, v in statistics.items()}
    residual = (observation.to(torch.complex128) - s["observed_mean"]).permute(1, 2, 0)
    inverse = torch.cholesky_solve(residual, s["factor"])
    joint = torch.cholesky_solve((s["z"].mH @ inverse).sum(0), s["small_factor"])
    inverse = inverse - (1 - s["strength"]) * (s["inverse_z"] @ joint)
    coordinates = s["z"].mH @ inverse
    weights = s["strength"] * coordinates + (1 - s["strength"]) * coordinates.sum(0, keepdim=True)
    output = s["mean"] + (s["x"] @ weights).permute(2, 0, 1)
    if not torch.isfinite(output).all():
        raise ArithmeticError("nonfinite factored LMMSE estimate")
    return output.to(observation.dtype)
