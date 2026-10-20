"""Joint-frequency SBL; all statistics are computed from the supplied observations."""
import torch


SOLVER_VERSION = "convex_bound_kkt_v1"


def _validate_inputs(y, sensing, dictionary, variance, gamma):
    if (y.ndim != 3 or not y.numel() or sensing.ndim != 3 or dictionary.ndim != 3
            or sensing.shape[:2] != y.shape[1:] or dictionary.shape[0] != y.shape[1]
            or sensing.shape[-1] != dictionary.shape[-1]
            or gamma.shape != (len(y), sensing.shape[-1]) or variance.shape != (len(y),)
            or not all(torch.isfinite(x).all() for x in (y, sensing, dictionary, variance, gamma))
            or (variance <= 0).any() or (gamma <= 0).any()):
        raise ValueError("finite compatible Gaussian inputs and positive variances required")


def _coefficient_statistics(y, sensing, variance, gamma):
    cov_y = torch.einsum("kjg,bg,klg->bkjl", sensing, gamma, sensing.conj())
    eye = torch.eye(y.shape[-1], dtype=y.dtype, device=y.device)
    cov_y = cov_y + variance[:, None, None, None] * eye
    factor = torch.linalg.cholesky(cov_y)
    solved_y = torch.cholesky_solve(y[..., None], factor).squeeze(-1)
    projected = torch.einsum("kjg,bkj->bkg", sensing.conj(), solved_y)
    solved_b = torch.cholesky_solve(sensing[None].expand(len(y), -1, -1, -1), factor)
    curvature = (sensing.conj()[None] * solved_b).sum(-2).real.clamp_min(0)
    return projected, curvature, factor, solved_y


def posterior(y, sensing, dictionary, variance, gamma, *, channel_moments=True):
    """Return coefficient and channel posterior moments for a proper complex Gaussian model."""
    _validate_inputs(y, sensing, dictionary, variance, gamma)
    projected, curvature, factor, _ = _coefficient_statistics(y, sensing, variance, gamma)
    mean_g = gamma[:, None] * projected
    var_g = (gamma[:, None] - gamma[:, None].square() * curvature).clamp_min(0)
    if not channel_moments:
        return mean_g, var_g, None, None
    mean_h = torch.einsum("kng,bkg->bkn", dictionary, mean_g)
    cross = torch.einsum("kng,bg,kjg->bknj", dictionary, gamma, sensing.conj())
    solved_cross = torch.cholesky_solve(cross.mH, factor)
    var_h = torch.einsum("kng,bg->bkn", dictionary.abs().square(), gamma)
    var_h = var_h - (cross * solved_cross.transpose(-2, -1)).sum(-1).real
    return mean_g, var_g, mean_h, var_h.clamp_min(1e-10)


@torch.no_grad()
def joint_sbl(y, sensing, dictionary, variance, *, iterations=100, tolerance=1e-3, early_stopping=True):
    """Convex-bounding SBL; stop on normalized evidence KKT conditions, not EM step size."""
    if (type(iterations) is not int or iterations < 1 or not torch.isfinite(torch.as_tensor(tolerance))
            or tolerance <= 0 or type(early_stopping) is not bool):
        raise ValueError("positive finite SBL budget/tolerance and boolean stopping policy required")
    if y.ndim != 3 or sensing.ndim != 3 or not sensing.shape[-1]:
        raise ValueError("nonempty batched observations and sensing dictionary required")
    floor = 1e-8 / sensing.shape[-1]
    _validate_inputs(y, sensing, dictionary, variance,
        torch.full((len(y), sensing.shape[-1]), floor, dtype=y.real.dtype, device=y.device))
    atom_power = sensing.abs().square().sum(-1).mean()
    if atom_power <= 0:
        raise ValueError("SBL needs nonzero sensing energy")
    initial = (y.abs().square().mean((1, 2)) - variance).clamp_min(1e-6) / atom_power
    gamma = initial[:, None].expand(-1, sensing.shape[-1]).clone().clamp_min(floor)
    done = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    counts = torch.full((len(y),), iterations, dtype=torch.int64, device=y.device)
    relative = torch.zeros_like(initial)
    for step in range(iterations + 1):
        projected, curvature, factor, solved_y = _coefficient_statistics(y, sensing, variance, gamma)
        curvature = curvature.mean(1)
        ratio = torch.where(curvature > 0, projected.abs().square().mean(1)
            / curvature.clamp_min(torch.finfo(curvature.dtype).tiny), torch.ones_like(curvature))
        error = ratio - 1
        # Dual feasibility and complementarity remain unchanged when all columns are copied/split.
        residual = torch.maximum(error.clamp_min(0),
            (gamma - floor) / gamma.amax(-1, keepdim=True) * error.abs()).amax(-1)
        if early_stopping:
            newly_done = (~done) & (residual <= tolerance)
            counts[newly_done] = step
            done |= newly_done
        if step == iterations or early_stopping and done.all():
            break
        # Minimize the separable upper bound of the same marginal-likelihood objective.
        update = (gamma * ratio.sqrt()).clamp_min(floor)
        change = (update - gamma).norm(dim=-1) / gamma.norm(dim=-1).clamp_min(1e-12)
        relative = torch.where(done, relative, change)
        gamma = torch.where(done[:, None], gamma, update) if early_stopping else update
    objective = (2 * factor.diagonal(dim1=-2, dim2=-1).real.log().sum(-1)
        + (y.conj() * solved_y).sum(-1).real).mean(1)
    _, _, mean_h, var_h = posterior(y, sensing, dictionary, variance, gamma)
    if not all(torch.isfinite(x).all() for x in (mean_h, var_h, residual, objective)):
        raise ArithmeticError("nonfinite SBL output")
    return mean_h, var_h, {"iterations": counts, "converged": residual <= tolerance,
        "relative_change": relative, "optimality_residual": residual, "objective_per_tone": objective}


def consistency(estimate, matrix, observation, variance, prior_variance):
    """One Gaussian proximal correction; prior_variance is a shared scalar, not truth error."""
    if (prior_variance <= 0 or not torch.isfinite(torch.as_tensor(prior_variance))
            or variance.shape != (len(estimate),) or (variance <= 0).any()):
        raise ValueError("positive correction and measurement variances required")
    residual = observation - torch.einsum("kjn,bkn->bkj", matrix, estimate)
    gram = matrix @ matrix.mH
    system = prior_variance * gram[None] + variance[:, None, None, None] * torch.eye(
        matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    solved = torch.cholesky_solve(residual[..., None], torch.linalg.cholesky(system)).squeeze(-1)
    return estimate + prior_variance * torch.einsum("kjn,bkj->bkn", matrix.conj(), solved)
