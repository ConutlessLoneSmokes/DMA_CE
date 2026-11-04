"""Non-neural structured-prior probes; all inference inputs are observations or declared priors."""
import math

import torch

from ._upstream.atmosphere import C0_M_PER_S
from .sbl import _coefficient_statistics, _validate_inputs, posterior


@torch.no_grad()
def covariance_sbl(y, matrix, dictionary, variance, training_factor, strength, iterations):
    """SBL with an independent Gaussian component of fixed training covariance."""
    if (not math.isfinite(strength) or strength < 0 or type(iterations) is not int or iterations < 1
            or training_factor.ndim != 3 or training_factor.shape[:2] != dictionary.shape[:2]
            or not training_factor.shape[-1] or not torch.isfinite(training_factor).all()
            or training_factor.dtype != dictionary.dtype or training_factor.device != dictionary.device):
        raise ValueError('finite compatible training factors and bounded solver parameters required')
    sensing = matrix @ dictionary
    atoms = dictionary.shape[-1]
    floor = 1e-8 / atoms
    gamma = torch.full((len(y), atoms), floor, dtype=y.real.dtype, device=y.device)
    _validate_inputs(y, sensing, dictionary, variance, gamma)
    power = sensing.abs().square().sum(-1).mean()
    if power <= 0:
        raise ValueError('nonzero sensing energy required')
    initial = (y.abs().square().mean((1, 2))-variance).clamp_min(1e-6)/power
    gamma = initial[:, None].expand(-1, atoms).clone().clamp_min(floor)
    if strength:
        basis = torch.cat((dictionary, training_factor), -1)
        expanded = torch.cat((sensing, matrix @ training_factor), -1)
        fixed = torch.full((len(y), training_factor.shape[-1]), strength, dtype=y.real.dtype, device=y.device)
    else:
        basis, expanded, fixed = dictionary, sensing, gamma[:, :0]
    for step in range(iterations+1):
        combined = torch.cat((gamma, fixed), -1)
        projected, curvature, factor, solved = _coefficient_statistics(y, expanded, variance, combined)
        projected, curvature = projected[..., :atoms], curvature[..., :atoms].mean(1)
        ratio = torch.where(curvature > 0, projected.abs().square().mean(1)
            / curvature.clamp_min(torch.finfo(curvature.dtype).tiny), torch.ones_like(curvature))
        error = ratio-1
        residual = torch.maximum(error.clamp_min(0),
            (gamma-floor)/gamma.amax(-1, keepdim=True)*error.abs()).amax(-1)
        if step == iterations:
            break
        gamma = (gamma*ratio.sqrt()).clamp_min(floor)
    coefficients, _, _, _ = posterior(y, expanded, basis, variance, combined, channel_moments=False)
    mean = torch.einsum('kng,bkg->bkn', basis, coefficients)
    objective = (2*factor.diagonal(dim1=-2, dim2=-1).real.log().sum(-1)
        +(y.conj()*solved).sum(-1).real).mean(1)
    if not torch.isfinite(mean).all():
        raise ArithmeticError('nonfinite augmented SBL posterior')
    return {'mean': mean, 'gamma': gamma, 'optimality_residual': residual,
        'objective_per_tone': objective, 'iterations': iterations}


def initial_geometry(gamma, dictionary, config, coherence=.9):
    """Select distinct SBL peaks; the strongest candidate is the declared LoS hypothesis."""
    paths, sbl, channel = config['channel']['paths'], config['sbl'], config['channel']
    if (gamma.ndim != 2 or gamma.shape[1] != dictionary.shape[-1] or paths > gamma.shape[1]
            or not 0 < coherence < 1 or not torch.isfinite(gamma).all() or (gamma < 0).any()):
        raise ValueError('finite atom powers and a valid coherence threshold required')
    angles = torch.linspace(-channel['angle_limit_deg'], channel['angle_limit_deg'],
        sbl['angle_points'], dtype=torch.float64, device=gamma.device)*math.pi/180
    ranges = 1/torch.linspace(1/channel['range_max_m'], 1/channel['range_min_m'],
        sbl['range_points'], dtype=torch.float64, device=gamma.device)
    grid = torch.stack(torch.meshgrid(angles, ranges, indexing='ij'), -1).reshape(-1, 2)
    if len(grid) != gamma.shape[1]:
        raise ValueError('dictionary must match the declared angle-range grid')
    tones = torch.linspace(0, len(dictionary)-1, min(4, len(dictionary)), device=gamma.device).long()
    sampled = dictionary[tones]
    selected, blocked = torch.zeros_like(gamma, dtype=torch.bool), torch.zeros_like(gamma, dtype=torch.bool)
    indices = []
    for _ in range(paths):
        mask = torch.where((~blocked).any(1, keepdim=True), blocked, selected)
        index = gamma.masked_fill(mask, -torch.inf).argmax(1)
        indices.append(index)
        selected.scatter_(1, index[:, None], True)
        columns = sampled[:, :, index].permute(2, 0, 1)
        correlation = torch.einsum('kng,bkn->bkg', sampled.conj(), columns).abs().square().mean(1)
        blocked |= (correlation > coherence) | selected
    indices = torch.stack(indices, 1)
    return grid[indices], indices


def propagation_basis(geometry, frequencies, positions, kappa, settings, scale):
    """Full physical paths; a common per-path carrier phase is absorbed in fitted gains."""
    if (geometry.ndim != 3 or geometry.shape[-1] != 2 or geometry.shape[1] != settings['paths']
            or not torch.isfinite(geometry).all() or (geometry[..., 1] <= 0).any()
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('finite angle/range geometry and positive channel scale required')
    angle, radius = geometry.unbind(-1)
    elevation = settings['elevation_deg']*math.pi/180
    points = torch.stack((radius*angle.sin()*math.cos(elevation),
        radius*angle.cos()*math.cos(elevation), radius*math.sin(elevation)), -1)
    before = (points-points[:, :1]).norm(dim=-1)
    if (before[:, 1:] <= torch.finfo(torch.float64).eps).any():
        raise ValueError('a reflector cannot coincide with the transmitter')
    after = (positions[None, :, None, :]-points[:, None]).norm(dim=-1)
    total = before[:, None]+after
    if (total <= 0).any():
        raise ValueError('path cannot coincide with an array element')
    safe_before = before[:, 1:]
    cosine = (safe_before.square()+radius[:, 1:].square()-radius[:, :1].square())/(2*safe_before*radius[:, 1:])
    incidence = .5*torch.acos(cosine.clamp(-1+1e-12, 1-1e-12))
    material = settings['reflection_refractive_index']
    refracted = (1-(incidence.sin()/material).square()).sqrt()
    reflection = ((incidence.cos()-material*refracted)/(incidence.cos()+material*refracted)).abs()
    rough = torch.exp(-8*(math.pi*frequencies[None, :, None]*settings['surface_roughness_m']
        *incidence.cos()[:, None]/C0_M_PER_S).square())
    reflected = reflection[:, None]*rough
    reflection = torch.cat((torch.ones((len(geometry), len(frequencies), 1),
        dtype=geometry.dtype, device=geometry.device), reflected), -1)
    amplitude = C0_M_PER_S/(4*math.pi*frequencies[None, :, None, None]*total[:, None])
    amplitude = amplitude*torch.exp(-.5*kappa[None, :, None, None]*total[:, None])*reflection[:, :, None]
    reference = before+radius
    phase = -2*math.pi/C0_M_PER_S*(frequencies[None, :, None, None]*(after-radius[:, None])[:, None]
        +(frequencies-frequencies.mean())[None, :, None, None]*reference[:, None, None])
    gain = 10**(settings.get('antenna_gain_dbi', 0.)/20)/math.sqrt(settings['paths'])/scale
    return gain*amplitude*torch.exp(1j*phase)


def fit_paths(parts, matrix, observation, shared, ridge=1e-8):
    """Variable projection with a fixed relative ridge, common to both coefficient models."""
    if (type(shared) is not bool or not math.isfinite(ridge) or ridge <= 0
            or parts.ndim != 4 or observation.shape != (len(parts), len(matrix), matrix.shape[1])
            or parts.shape[1:3] != (len(matrix), matrix.shape[2])
            or not torch.isfinite(parts).all() or not torch.isfinite(observation).all()):
        raise ValueError('compatible finite paths/observations and a positive stabilizer required')
    sensing = torch.einsum('kjn,bknl->bkjl', matrix, parts)
    design, target = (sensing.flatten(1, 2), observation.flatten(1)) if shared else (sensing, observation)
    gram = design.mH @ design
    regularizer = ridge*gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(torch.finfo(gram.real.dtype).tiny)
    gram = gram+regularizer[..., None, None]*torch.eye(parts.shape[-1], dtype=gram.dtype, device=gram.device)
    coefficients = torch.linalg.solve(gram, design.mH @ target[..., None]).squeeze(-1)
    channel = (torch.einsum('bknl,bl->bkn', parts, coefficients) if shared
        else torch.einsum('bknl,bkl->bkn', parts, coefficients))
    prediction = torch.einsum('kjn,bkn->bkj', matrix, channel)
    loss = (prediction-observation).abs().square().sum((1, 2))
    return channel, loss, coefficients


def refine_geometry(geometry, matrix, observation, frequencies, positions, kappa, settings, scale,
        *, shared, steps, learning_rate):
    # shortcut: one local initialization, add delay/geometry hypotheses if basin failures dominate.
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('positive bounded refinement budget required')
    angle_limit = settings['angle_limit_deg']*math.pi/180
    lower, width = 1/settings['range_max_m'], 1/settings['range_min_m']-1/settings['range_max_m']
    normalized = torch.stack(((geometry[..., 0]+angle_limit)/(2*angle_limit),
        (1/geometry[..., 1]-lower)/width), -1).detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([normalized], lr=learning_rate)
    best_loss = torch.full((len(geometry),), torch.inf, dtype=geometry.dtype, device=geometry.device)
    best_geometry = geometry.detach().clone()
    energy = observation.abs().square().sum((1, 2)).clamp_min(1e-30)
    initial_loss = None
    for step in range(steps+1):
        current = torch.stack((normalized[..., 0]*(2*angle_limit)-angle_limit,
            1/(lower+normalized[..., 1]*width)), -1)
        parts = propagation_basis(current, frequencies, positions, kappa, settings, scale)
        channel, loss, _ = fit_paths(parts, matrix, observation, shared)
        if initial_loss is None:
            initial_loss = loss.detach().clone()
        improved = loss.detach() < best_loss
        best_geometry = torch.where(improved[:, None, None], current.detach(), best_geometry)
        best_loss = torch.minimum(best_loss, loss.detach())
        if step == steps:
            break
        optimizer.zero_grad()
        (loss/energy).sum().backward()
        if not torch.isfinite(normalized.grad).all():
            raise ArithmeticError('nonfinite physical-geometry gradient')
        optimizer.step()
        with torch.no_grad():
            normalized.clamp_(0, 1)
    with torch.no_grad():
        parts = propagation_basis(best_geometry, frequencies, positions, kappa, settings, scale)
        channel, loss, _ = fit_paths(parts, matrix, observation, shared)
    return {'geometry': best_geometry, 'mean': channel, 'observation_loss': loss,
        'initial_observation_loss': initial_loss, 'steps': steps}
