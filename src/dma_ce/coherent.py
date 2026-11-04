"""Phase-steered coefficient covariances with an unchanged full spatial prior."""
import math

import torch

from .sbl import _validate_inputs


def smooth_envelope(frequencies, weights):
    """Three smooth modes with a constant prior marginal variance at every tone."""
    f = frequencies.to(torch.float64)
    w = torch.as_tensor(weights, dtype=torch.float64, device=f.device)
    if (f.ndim != 1 or len(f) < 3 or not torch.isfinite(f).all() or (f <= 0).any()
            or (f.diff() <= 0).any() or not torch.allclose(f.diff(), f.diff().mean().expand(len(f) - 1))
            or w.shape != (3,) or not torch.isfinite(w).all() or w[0] <= 0 or (w < 0).any()):
        raise ValueError("uniform positive frequencies and three finite envelope weights required")
    coordinate = 2 * (f - f.mean()) / (f[-1] - f[0])
    basis = torch.stack((torch.ones_like(f), coordinate, coordinate.square()), -1) * w
    return basis / basis.norm(dim=-1, keepdim=True)


@torch.no_grad()
def delay_model(coefficients, coefficient_variance, gamma, frequencies, settings):
    """Coefficient-domain empirical scores; diffuse delay mass stays independent."""
    if (coefficients.ndim != 3 or coefficient_variance.shape != coefficients.shape
            or gamma.shape != (len(coefficients), coefficients.shape[-1])
            or len(frequencies) != coefficients.shape[1] or not coefficients.is_complex()
            or not all(torch.isfinite(x).all() for x in (coefficients, coefficient_variance, gamma))
            or (coefficient_variance < 0).any() or (gamma <= 0).any()
            or any(type(settings[key]) is not int or settings[key] < 1
                for key in ("coherent_atoms", "delay_grid", "delay_hypotheses"))
            or settings["delay_grid"] < coefficients.shape[1]
            or settings["delay_hypotheses"] > settings["delay_grid"]
            or type(settings["delay_refinement_steps"]) is not int or settings["delay_refinement_steps"] < 0
            or not math.isfinite(settings["coefficient_variance_floor"]) or settings["coefficient_variance_floor"] <= 0):
        raise ValueError("finite compatible coefficient statistics and positive delay controls required")
    device = coefficients.device
    envelope = smooth_envelope(frequencies.to(device), settings["envelope_weights"])
    count = min(settings["coherent_atoms"], gamma.shape[-1])
    indices = gamma.topk(count, dim=-1).indices
    index = indices[..., None].expand(-1, -1, coefficients.shape[1])
    selected = coefficients.to(torch.complex128).transpose(1, 2).gather(1, index)
    powers = gamma.to(torch.float64).gather(1, indices)
    uncertainty = coefficient_variance.to(torch.float64).transpose(1, 2).gather(1, index)
    uncertainty += settings["coefficient_variance_floor"] * powers[..., None]
    gram = torch.einsum("kp,bakq->bapq", envelope, envelope[None, None] / uncertainty[..., None])
    gram += torch.eye(envelope.shape[-1], device=device) / powers[..., None, None]
    factor = torch.linalg.cholesky(gram.to(torch.complex128))
    values = selected / uncertainty
    derivative = 2j * math.pi * torch.arange(coefficients.shape[1], device=device, dtype=torch.float64)

    def evaluate(locations):
        phase = torch.exp(derivative[None, None, None] * locations[..., None])
        projection = torch.einsum("baqk,bak,kp->baqp", phase, values, envelope.to(torch.complex128))
        first = torch.einsum("baqk,bak,kp->baqp", phase, values * derivative, envelope.to(torch.complex128))
        second = torch.einsum("baqk,bak,kp->baqp", phase, values * derivative.square(), envelope.to(torch.complex128))
        solved = torch.cholesky_solve(projection[..., None], factor[:, :, None]).squeeze(-1)
        solved_first = torch.cholesky_solve(first[..., None], factor[:, :, None]).squeeze(-1)
        return ((projection.conj() * solved).sum(-1).real,
            2 * (first.conj() * solved).sum(-1).real,
            2 * (second.conj() * solved + first.conj() * solved_first).sum(-1).real)

    grid = torch.arange(settings["delay_grid"], dtype=torch.float64, device=device) / settings["delay_grid"]
    grid = grid.expand(len(coefficients), count, -1)
    scores, _, _ = evaluate(grid)
    weights, bins = scores.softmax(-1).topk(settings["delay_hypotheses"], dim=-1)
    origin = grid.gather(-1, bins)
    locations, best = origin.clone(), scores.gather(-1, bins)
    for _ in range(settings["delay_refinement_steps"]):
        _, first, second = evaluate(locations)
        step = torch.where(second < -1e-12, first / second.clamp_max(-1e-12), torch.zeros_like(first))
        proposed = (locations - step).clamp(origin - .5 / settings["delay_grid"], origin + .5 / settings["delay_grid"])
        score, _, _ = evaluate(proposed)
        improved = score > best
        locations = torch.where(improved, proposed, locations)
        best = torch.maximum(best, score)
    return {"indices": indices, "delays": locations.remainder(1), "weights": weights, "envelope": envelope}


@torch.no_grad()
def coherent_posterior(y, matrix, dictionary, variance, gamma, model, strength):
    """Condition once on y; spatial powers and delay weights are empirical Bayes inputs."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("finite correlation strength in [0, 1] required")
    dtype, device = torch.complex128, y.device
    a, d, observation = matrix.to(dtype), dictionary.to(dtype), y.to(dtype)
    noise, powers = variance.to(torch.float64), gamma.to(torch.float64)
    sensing = a @ d
    _validate_inputs(observation, sensing, d, noise, powers)
    indices = model["indices"].to(device)
    delays, weights = (model[key].to(device, torch.float64) for key in ("delays", "weights"))
    envelope = model["envelope"].to(device, torch.float64)
    batch, tones, measurements = y.shape
    if (indices.ndim != 2 or indices.shape[0] != batch or not indices.shape[1]
            or indices.dtype != torch.int64 or (indices < 0).any() or (indices >= d.shape[-1]).any()
            or (indices.sort(-1).values.diff(dim=-1) == 0).any()
            or delays.ndim != 3 or delays.shape[:2] != indices.shape or delays.shape != weights.shape
            or not delays.shape[-1] or envelope.ndim != 2 or envelope.shape[0] != tones
            or not all(torch.isfinite(t).all() for t in (delays, weights, envelope))
            or (weights < 0).any() or (weights.sum(-1) > 1 + 1e-7).any()
            or not torch.allclose(envelope.square().sum(-1), torch.ones(tones, device=device, dtype=torch.float64), rtol=1e-6, atol=1e-8)):
        raise ValueError("distinct spatial indices, bounded delay weights and unit-row envelope required")
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
    selected_powers = powers.gather(1, indices)
    independent = powers.clone()
    independent.scatter_(1, indices, selected_powers * (1 - strength * weights.sum(-1)))
    covariance = torch.einsum("kmg,bg,klg->bkml", sensing, independent, sensing.conj())
    covariance += noise[:, None, None, None] * torch.eye(measurements, dtype=dtype, device=device)
    factor = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(observation[..., None], factor).squeeze(-1)
    logdet = 2 * factor.diagonal(dim1=-2, dim2=-1).real.log().sum((1, 2))
    if strength:
        phase = torch.exp(-2j * math.pi * torch.arange(tones, device=device, dtype=torch.float64)[None, :, None, None] * delays[:, None])
        mixing = (phase[..., None] * envelope[None, :, None, None]
            * (strength * selected_powers[..., None] * weights).sqrt()[:, None, :, :, None])
        selected = sensing[None].expand(batch, -1, -1, -1).gather(3,
            indices[:, None, None].expand(-1, tones, measurements, -1))
        low_rank = (selected[..., None, None] * mixing[:, :, None]).flatten(-3)
        inverse_low_rank = torch.cholesky_solve(low_rank, factor)
        small = torch.eye(low_rank.shape[-1], dtype=dtype, device=device) + (low_rank.mH @ inverse_low_rank).sum(1)
        small_factor = torch.linalg.cholesky((small + small.mH) / 2)
        coordinates = torch.cholesky_solve((low_rank.conj() * solved[..., None]).sum((1, 2))[..., None], small_factor).squeeze(-1)
        solved -= (inverse_low_rank @ coordinates[:, None, :, None]).squeeze(-1)
        logdet += 2 * small_factor.diagonal(dim1=-2, dim2=-1).real.log().sum(-1)
    coefficients = independent[:, None] * torch.einsum("kmg,bkm->bkg", sensing.conj(), solved)
    if strength:
        contribution = (mixing * coordinates.reshape(batch, *mixing.shape[2:])[:, None]).sum((-1, -2))
        coefficients.scatter_add_(2, indices[:, None].expand(-1, tones, -1), contribution)
    mean = torch.einsum("kng,bkg->bkn", d, coefficients)
    objective = (logdet + (observation.conj() * solved).sum((1, 2)).real) / tones
    if not torch.isfinite(mean).all() or not torch.isfinite(objective).all():
        raise ArithmeticError("nonfinite coherent Gaussian posterior")
    return {"mean": mean, "objective_per_tone": objective, "precision_observation": solved}
