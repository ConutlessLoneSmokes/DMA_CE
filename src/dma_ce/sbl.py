"""Joint-frequency SBL; all statistics are computed from the supplied observations."""
import torch


def posterior(y, sensing, dictionary, variance, gamma):
    """Return coefficient and channel posterior moments for a proper complex Gaussian model."""
    if (y.ndim != 3 or sensing.ndim != 3 or dictionary.ndim != 3
            or sensing.shape[:2] != y.shape[1:] or dictionary.shape[0] != y.shape[1]
            or sensing.shape[-1] != dictionary.shape[-1]
            or gamma.shape != (len(y), sensing.shape[-1]) or variance.shape != (len(y),)
            or not all(torch.isfinite(x).all() for x in (y, sensing, dictionary, variance, gamma))
            or (variance <= 0).any() or (gamma <= 0).any()):
        raise ValueError("finite compatible Gaussian inputs and positive variances required")
    cov_y = torch.einsum("kjg,bg,klg->bkjl", sensing, gamma, sensing.conj())
    eye = torch.eye(y.shape[-1], dtype=y.dtype, device=y.device)
    cov_y = cov_y + variance[:, None, None, None] * eye
    factor = torch.linalg.cholesky(cov_y)
    solved_y = torch.cholesky_solve(y[..., None], factor).squeeze(-1)
    mean_g = gamma[:, None] * torch.einsum("kjg,bkj->bkg", sensing.conj(), solved_y)
    solved_b = torch.cholesky_solve(sensing[None].expand(len(y), -1, -1, -1), factor)
    var_g = gamma[:, None] - gamma[:, None].square() * (sensing.conj()[None] * solved_b).sum(-2).real
    mean_h = torch.einsum("kng,bkg->bkn", dictionary, mean_g)
    cross = torch.einsum("kng,bg,kjg->bknj", dictionary, gamma, sensing.conj())
    solved_cross = torch.cholesky_solve(cross.mH, factor)
    var_h = torch.einsum("kng,bg->bkn", dictionary.abs().square(), gamma)
    var_h = var_h - (cross * solved_cross.transpose(-2, -1)).sum(-1).real
    return mean_g, var_g.clamp_min(1e-10), mean_h, var_h.clamp_min(1e-10)


@torch.no_grad()
def joint_sbl(y, sensing, dictionary, variance, *, iterations=100, tolerance=1e-3):
    """Shared physical-grid variances; per-tone complex coefficients remain unconstrained."""
    if iterations < 1 or tolerance <= 0:
        raise ValueError("positive SBL iteration budget and tolerance required")
    atom_power = sensing.abs().square().sum(-1).mean()
    initial = (y.abs().square().mean((1, 2)) - variance).clamp_min(1e-6) / atom_power
    gamma = initial[:, None].expand(-1, sensing.shape[-1]).clone().clamp_min(1e-8)
    done = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    counts = torch.full((len(y),), iterations, dtype=torch.int64, device=y.device)
    relative = torch.full_like(initial, float("inf"))
    for step in range(iterations):
        mean, diagonal, _, _ = posterior(y, sensing, dictionary, variance, gamma)
        update = .5 * gamma + .5 * (mean.abs().square() + diagonal).mean(1)
        update = update.clamp_min(1e-8)
        change = (update - gamma).norm(dim=-1) / gamma.norm(dim=-1).clamp_min(1e-12)
        newly_done = (~done) & (change < tolerance) & (step >= 4)
        counts[newly_done] = step + 1
        relative = torch.where(done, relative, change)
        gamma = torch.where(done[:, None], gamma, update)
        done |= newly_done
        if done.all():
            break
    _, _, mean_h, var_h = posterior(y, sensing, dictionary, variance, gamma)
    if not torch.isfinite(mean_h).all() or not torch.isfinite(var_h).all():
        raise ArithmeticError("nonfinite SBL output")
    return mean_h, var_h, {"iterations": counts, "converged": done, "relative_change": relative}


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
