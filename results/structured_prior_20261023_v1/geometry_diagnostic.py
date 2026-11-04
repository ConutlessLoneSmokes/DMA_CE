"""Post-hoc diagnostic: truth is used only to measure a fixed geometry's CSI subspace."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'src'))
from dma_ce.experiment import write_json
from dma_ce.physics import paper_channel
from dma_ce.physical_study import case_hardware

run = Path(sys.argv[1]).resolve()
config = json.loads((run/'config_resolved.json').read_text())
_, _, hardware = case_hardware(config, config['cases'][0])
cache = torch.load(run/'inputs/physical_parent.pt', weights_only=True)
observed = torch.load(run/'predictions/observations.pt', weights_only=True)
oracle = torch.load(run/'inputs/oracle_geometry.pt', weights_only=True).numpy()
context = config['system']['snrs_db'].index(30)
saved = torch.load(run/f'predictions/context_{context}.pt', weights_only=True)
truth = observed['entries'][context]['truth'].numpy().astype(np.complex128)
rows = []
for stage, geometry in [('initial', saved['states']['initial_geometry'].numpy()),
        ('refined_per_tone', saved['states']['refined']['geometry'].numpy()),
        ('joint_shared', saved['states']['joint']['geometry'].numpy())]:
    gains = np.zeros(oracle.shape[:2], dtype=np.complex128)
    parts = []
    for path in range(config['channel']['paths']):
        gains[:] = 0
        gains[:, path] = 1
        parts.append(paper_channel(hardware[0], (geometry[..., 1], geometry[..., 0], gains))/cache['channel_scale'])
    parts = np.stack(parts, axis=-1)
    floors = {False: [], True: []}
    for scene, (basis, target) in enumerate(zip(parts, truth)):
        common = np.linalg.lstsq(basis.reshape(-1, gains.shape[-1]), target.reshape(-1), rcond=None)[0]
        free = np.stack([np.linalg.lstsq(block, y, rcond=None)[0] for block, y in zip(basis, target)])
        for shared, output in [(True, basis @ common), (False, np.einsum('knl,kl->kn', basis, free))]:
            floors[shared].append(float(np.sum(abs(output-target)**2)/np.sum(abs(target)**2)))
        assert floors[False][-1] <= floors[True][-1]+1e-10
    rows.append({'stage': stage, 'snr_db': 30, 'scenes': len(truth),
        'mean_abs_declared_los_angle_error_deg': float(np.mean(abs(geometry[:, 0, 0]-oracle[:, 0, 0]))*180/np.pi),
        'median_abs_declared_los_range_error_m': float(np.median(abs(geometry[:, 0, 1]-oracle[:, 0, 1]))),
        'fixed_geometry_csi_projection_floor_db': {'per_tone': float(10*np.log10(np.mean(floors[False]))),
            'shared': float(10*np.log10(np.mean(floors[True])))}})
report = {'status': 'passed', 'purpose': 'post_hoc_truth_diagnostic_only',
    'geometry_frozen_to_actual_outputs': True, 'coefficients_fit_to_full_true_csi_not_observations': True,
    'interpretation': 'Minimum full-CSI error within each fixed geometry coefficient subspace; not an attainable observation-only result or a bound for other geometries.',
    'rows': rows}
write_json(run/'export/geometry_diagnostic.json', report)
print(json.dumps(report))
