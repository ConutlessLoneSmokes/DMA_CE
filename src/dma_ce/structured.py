"""Training-only LMMSE statistics and deterministic DDPM/Gaussian alternation."""
import math
from numbers import Real

import torch
from torch import nn

from .data import decode, encode


def _complex(value, dimensions):
    return (isinstance(value, torch.Tensor) and value.ndim == dimensions and value.numel() > 0
        and value.dtype in (torch.complex64, torch.complex128) and torch.isfinite(value).all())


def _positive(value, shape, device):
    return (isinstance(value, torch.Tensor) and value.shape == shape and value.is_floating_point()
        and value.device == device and torch.isfinite(value).all() and (value > 0).all())


def _gaussian_inputs(mean, matrix, observation, variance, diagonal):
    if (not _complex(mean, 3) or not _complex(matrix, 3) or not _complex(observation, 3)
            or matrix.shape[0] != mean.shape[1] or matrix.shape[2] != mean.shape[2]
            or observation.shape != (len(mean), matrix.shape[0], matrix.shape[1])
            or any(x.device != mean.device or x.dtype != mean.dtype for x in (matrix, observation))
            or not _positive(variance, (len(mean),), mean.device)
            or not _positive(diagonal, mean.shape, mean.device)):
        raise ValueError("finite compatible complex Gaussian tensors and positive real variances required")


def gaussian_update(mean, matrix, observation, variance, diagonal):
    """Condition CN(mean, diag(diagonal)) on y=A h+n; variances are complex variances."""
    _gaussian_inputs(mean, matrix, observation, variance, diagonal)
    diagonal, variance = diagonal.to(mean.real), variance.to(mean.real)
    weighted = matrix[None] * diagonal[:, :, None, :]
    system = weighted @ matrix.mH[None] + variance[:, None, None, None] * torch.eye(
        matrix.shape[1], dtype=mean.dtype, device=mean.device)
    factor = torch.linalg.cholesky(system)
    residual = observation - torch.einsum("kjn,bkn->bkj", matrix, mean)
    solved = torch.cholesky_solve(residual[..., None], factor)
    updated = mean + (weighted.mH @ solved).squeeze(-1)
    solved_matrix = torch.cholesky_solve(matrix[None].expand(len(mean), -1, -1, -1), factor)
    curvature = (matrix.conj()[None] * solved_matrix).sum(-2).real
    posterior_diagonal = (diagonal - diagonal.square() * curvature).clamp_min(torch.finfo(diagonal.dtype).tiny)
    if not all(torch.isfinite(x).all() for x in (updated, posterior_diagonal)):
        raise ArithmeticError("nonfinite Gaussian update")
    return updated, posterior_diagonal


@torch.no_grad()
def lmmse_statistics(training, matrix, variance):
    """Joint-frequency unbiased sample covariance; caller supplies only training channels."""
    if (not _complex(training, 3) or len(training) < 2 or not _complex(matrix, 3)
            or matrix.shape[0] != training.shape[1] or matrix.shape[2] != training.shape[2]
            or matrix.device != training.device or isinstance(variance, bool)
            or not isinstance(variance, Real) or not math.isfinite(variance) or variance <= 0):
        raise ValueError("at least two compatible finite training channels and positive noise variance required")
    h = training.to(torch.complex128)
    observed = torch.einsum("kjn,bkn->bkj", matrix.to(torch.complex128), h)
    mean, observed_mean = h.mean(0), observed.mean(0)
    centered = (h - mean).flatten(1)
    centered_y = (observed - observed_mean).flatten(1)
    cross = centered.T @ centered_y.conj() / (len(training) - 1)
    covariance_y = centered_y.T @ centered_y.conj() / (len(training) - 1)
    covariance_y = covariance_y + variance * torch.eye(
        covariance_y.shape[0], dtype=covariance_y.dtype, device=covariance_y.device)
    gain = torch.cholesky_solve(cross.mH, torch.linalg.cholesky(covariance_y)).mH
    diagonal = centered.abs().square().sum(0) / (len(training) - 1)
    diagonal = (diagonal - (gain * cross.conj()).sum(-1).real).clamp_min(0).reshape_as(mean.real)
    if not all(torch.isfinite(x).all() for x in (mean, observed_mean, gain, diagonal)):
        raise ArithmeticError("nonfinite training LMMSE statistics")
    return {"mean": mean, "observed_mean": observed_mean, "gain": gain, "diagonal": diagonal}


def apply_lmmse(statistics, observation):
    """Apply fixed training statistics, preserving the observation dtype and device."""
    if (not isinstance(statistics, dict) or not _complex(observation, 3)
            or any(key not in statistics for key in ("mean", "observed_mean", "gain", "diagonal"))):
        raise ValueError("finite complex observations and complete LMMSE statistics required")
    mean, observed_mean, gain, diagonal = (statistics[key] for key in ("mean", "observed_mean", "gain", "diagonal"))
    if (not _complex(mean, 2) or not _complex(observed_mean, 2) or not _complex(gain, 2)
            or observed_mean.shape != observation.shape[1:] or mean.shape[0] != observation.shape[1]
            or gain.shape != (mean.numel(), observed_mean.numel())
            or not isinstance(diagonal, torch.Tensor) or diagonal.shape != mean.shape
            or not diagonal.is_floating_point() or not torch.isfinite(diagonal).all() or (diagonal < 0).any()):
        raise ValueError("compatible finite joint-frequency LMMSE statistics required")
    mean, observed_mean, gain = (value.to(device=observation.device, dtype=torch.complex128)
        for value in (mean, observed_mean, gain))
    residual = (observation.to(torch.complex128) - observed_mean).flatten(1)
    result = (mean.flatten()[None] + residual @ gain.T).reshape(len(observation), *mean.shape).to(observation.dtype)
    if not torch.isfinite(result).all():
        raise ArithmeticError("nonfinite LMMSE estimate")
    return result


class VarianceUpdate(nn.Module):
    """A shared five-feature CNN changes positive variances, leaving complex means analytic."""
    def __init__(self, width=16):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("positive integer variance network width required")
        self.net = nn.Sequential(nn.Conv2d(5, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mean, diagonal, backprojected, variance):
        if (not _complex(mean, 3) or not _complex(backprojected, 3) or backprojected.shape != mean.shape
                or backprojected.device != mean.device or backprojected.dtype != mean.dtype
                or not _positive(diagonal, mean.shape, mean.device)
                or not _positive(variance, (len(mean),), mean.device)):
            raise ValueError("finite compatible variance features and positive real variances required")
        diagonal, variance = diagonal.to(mean.real), variance.to(mean.real)
        features = torch.stack((mean.real, mean.imag, diagonal.log(),
            torch.log1p(backprojected.abs().square()), variance[:, None, None].log().expand_as(diagonal)), 1)
        log_factor = self.net(features.transpose(-1, -2))[:, 0].transpose(-1, -2).clamp(-4, 4)
        result = (diagonal * log_factor.exp()).clamp_min(torch.finfo(diagonal.dtype).tiny)
        if not torch.isfinite(result).all():
            raise ArithmeticError("nonfinite learned prior variance")
        return result


def structured_recover(prior, schedule, matrix, observation, variance, initial_mean, initial_diagonal,
        *, updater=None, iterations=3, strength=1.):
    """Deterministic plug-and-play alternation; the final diagonal is an algorithmic Gaussian moment."""
    _gaussian_inputs(initial_mean, matrix, observation, variance, initial_diagonal)
    bars = getattr(schedule, "alpha_bar", None)
    if (not isinstance(prior, nn.Module) or any(p.requires_grad for p in prior.parameters())
            or type(iterations) is not int or iterations < 1 or isinstance(strength, bool)
            or not isinstance(strength, Real) or not math.isfinite(strength) or strength <= 0
            or not isinstance(bars, torch.Tensor) or bars.ndim != 1 or len(bars) < 2
            or not torch.isfinite(bars).all() or ((bars <= 0) | (bars >= 1)).any()
            or (bars[1:] >= bars[:-1]).any() or updater is not None and not isinstance(updater, nn.Module)):
        raise ValueError("frozen DDPM, decreasing VP schedule and positive finite iteration settings required")
    bars = bars.to(device=initial_mean.device)
    ratios = (1 - bars) / bars
    current, final_diagonal = initial_mean, initial_diagonal
    prior.eval()
    for iteration in range(iterations):
        target = initial_diagonal * (strength * .5 ** iteration)
        if not _positive(target, initial_mean.shape, initial_mean.device):
            raise ValueError("structured prior variance must stay finite and positive")
        # Real DDPM noise has half the variance of a proper complex channel error.
        real_variance = target.mean((1, 2)) / 2
        indices = (ratios[None] - real_variance[:, None]).abs().argmin(-1)
        alpha = bars[indices].to(current.real)[:, None, None, None]
        state = alpha.sqrt() * encode(current)
        epsilon = prior(state, indices + 1)
        if epsilon.shape != state.shape or epsilon.dtype != state.dtype or not torch.isfinite(epsilon).all():
            raise ValueError("DDPM must predict finite noise with the input image shape and dtype")
        denoised = decode((state - (1 - alpha).sqrt() * epsilon) / alpha.sqrt())
        if updater is not None:
            residual = observation - torch.einsum("kjn,bkn->bkj", matrix, denoised)
            backprojected = torch.einsum("kjn,bkj->bkn", matrix.conj(), residual)
            target = updater(denoised, target, backprojected, variance)
        current, final_diagonal = gaussian_update(denoised, matrix, observation, variance, target)
    return current, final_diagonal
