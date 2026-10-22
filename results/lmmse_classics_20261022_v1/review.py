"""Read-only review of frozen physical LMMSE estimators; no fitting or selection."""
import hashlib
import itertools
import json
import math
import csv
import platform
from pathlib import Path

import torch

torch.set_num_threads(4)
parent = Path('/home/lsy/DMA_CE/runs/physical_baselines_20261021_v2')
case_path = parent / 'cache/physical_absorption.pt'
stats_path = parent / 'cache/statistics.pt'
cache = torch.load(case_path, map_location='cpu', weights_only=True)
statistics = torch.load(stats_path, map_location='cpu', weights_only=True)['physical_absorption']
screen = torch.load(parent / 'predictions/physical_absorption/observations.pt',
                    map_location='cpu', weights_only=True)

# Zero-mean noise with identity covariance gives the same quadratic expectation.
phases = [1, -1, 1j, -1j]
z = torch.tensor(list(itertools.product(phases, repeat=2)), dtype=torch.complex128)
g = torch.tensor([[1+1j, .3], [-.6j, .2-1j]], dtype=torch.complex128)
bias = torch.tensor([.4+.2j, -.9j], dtype=torch.complex128)
assert torch.allclose((bias + math.sqrt(.4)*z@g.T).abs().square().sum(1).mean(),
                      bias.abs().square().sum() + .4*g.abs().square().sum(), atol=1e-12, rtol=1e-12)

rows, budgets, per_scene = [], [], []
dbm = lambda p: 10*math.log10(p/1e-3)
for index, context in enumerate(cache['contexts']):
    band, power, count = (context[k] for k in ('bandwidth_hz', 'transmit_power_dbm', 'configurations'))
    if power == 0:
        raw = context['raw_noise_per_tone_w']
        full = raw.sum(0)
        budgets.append({'bandwidth_hz': band, 'configurations': count,
            'antenna_per_tone_dbm': dbm(context['physical_white_noise_per_tone_w']),
            'antenna_full_band_dbm': dbm(context['thermal_budget']['antenna_noise_full_band_w']),
            'receiver_added_per_tone_dbm': dbm(context['thermal_budget']['receiver_noise_per_tone_w']),
            'raw_per_tone_mean_power_dbm': dbm(raw.mean().item()),
            'raw_full_band_mean_power_dbm': dbm(full.mean().item()),
            'raw_full_band_min_dbm': dbm(full.min().item()),
            'raw_full_band_max_dbm': dbm(full.max().item())})
    if not ((band == 3e9 and power == 0) or (band == 30e9 and power in (10, 20))):
        continue
    band_index = 0 if band == 3e9 else 1
    matrix = context['matrix'].to(torch.complex128)
    for method in ('lmmse_joint', 'lmmse_per_tone'):
        stats = statistics[index][method]
        gain = stats['gain']
        assert gain.dtype == torch.complex128
        assert torch.allclose(torch.einsum('kjn,kn->kj', matrix, stats['mean']),
                              stats['observed_mean'], atol=1e-12, rtol=1e-12)
        noise = context['noise_complex_variance'] * gain.abs().square().sum().item()
        for split, truth in [('train', cache['training'][band_index]),
                             ('validation', cache['validation'][index]['truth']),
                             ('screen', screen['entries'][index]['truth'])]:
            truth = truth.to(torch.complex128)
            clean = torch.einsum('kjn,bkn->bkj', matrix, truth)
            prediction = (stats['mean'].flatten()[None] +
                (clean-stats['observed_mean']).flatten(1)@gain.T).reshape_as(truth)
            bias_energy = (prediction-truth).abs().square().sum((1, 2))
            energy = truth.abs().square().sum((1, 2))
            expected = bias_energy + noise
            assert torch.isfinite(expected).all() and (expected >= 0).all() and (energy > 0).all()
            per_scene.extend({'context': index, 'method': method, 'split': split, 'scene': scene,
                'truth_energy': energy[scene].item(), 'bias_energy': bias_energy[scene].item(),
                'expected_noise_energy': noise, 'expected_nmse': (expected[scene]/energy[scene]).item()}
                for scene in range(len(truth)))
            rows.append({'bandwidth_hz': band, 'configurations': count, 'transmit_power_dbm': power,
                'training_received_snr_db': context['snr_db'], 'method': method, 'split': split,
                'scenes': len(truth),
                'expected_mean_scene_nmse_db': 10*math.log10((expected/energy).mean().item()),
                'expected_energy_ratio_nmse_db': 10*math.log10(expected.sum().item()/energy.sum().item()),
                'expected_bias_nmse': (bias_energy/energy).mean().item(),
                'expected_noise_nmse': (noise/energy).mean().item()})

result = {'parent_run': str(parent), 'case': 'physical_absorption',
    'parent_file_sha256': {str(p.relative_to(parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (parent/'frozen.json', case_path, stats_path)},
    'definition': 'Fixed saved affine estimator; expectation only over measurement noise. '
        'Expected squared error = noiseless squared bias + noise variance * squared Frobenius norm of gain.',
    'self_checks': 'Quadratic expectation and observation-mean consistency passed.',
    'noise_budgets': budgets, 'rows': rows}
frozen = json.loads((parent/'frozen.json').read_text())
assert result['parent_file_sha256']['cache/physical_absorption.pt'] == frozen['cache_sha256']['physical_absorption.pt']
assert result['parent_file_sha256']['cache/statistics.pt'] == frozen['cache_sha256']['statistics.pt']
run = parent.with_name('lmmse_classics_20261022_v1')
run.mkdir(exist_ok=False)
(run/'review.json').write_text(json.dumps(result, indent=2))
(run/'review.py').write_bytes(Path(__file__).read_bytes())
(run/'parent_config.json').write_bytes((parent/'config_resolved.json').read_bytes())
(run/'environment.json').write_text(json.dumps({'python': platform.python_version(),
    'platform': platform.platform(), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
    'new_random_sampling': False}, indent=2))
with (run/'per_scene.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(per_scene[0]))
    writer.writeheader()
    writer.writerows(per_scene)
(run/'status.json').write_text(json.dumps({'status': 'complete', 'scenes_scored': len(per_scene)}))
print(json.dumps(result))
