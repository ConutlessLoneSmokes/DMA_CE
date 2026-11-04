"""Replay saved tensors on CPU; write diagnostic plots without refitting on screening data."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


folder, run = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve()
torch.set_num_threads(4)
read = lambda name: json.loads((run/name).read_text())
load = lambda name: torch.load(run/name, map_location='cpu', weights_only=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
config, frozen, physical = read('config_resolved.json'), read('frozen.json'), load('inputs/physical_parent.pt')
assert read('status.json')['status'] == 'complete'
for name, digest in frozen['files_sha256'].items():
    assert sha(run/name) == digest, name
with zipfile.ZipFile(run/'source.zip') as archive:
    for name, digest in read('source_manifest.json').items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
observed, parts = load('predictions/observations.pt'), load('inputs/oracle_path_basis.pt')
assert observed['created_at_unix'] > frozen['frozen_at_unix']
assert observed['frozen_sha256'] == sha(run/'frozen.json')
ids = observed['geometry_ids']
assert len(set(ids)) == config['data']['screen'] == 32
assert not set(ids) & set(physical['geometry_ids']['train']+physical['geometry_ids']['validation'])
with (folder.parent/config['probe']['parent_run']/'metrics.csv').open() as stream:
    assert not set(ids) & {r['geometry_id'] for r in csv.DictReader(stream)}
assert sha(run/'inputs/physical_parent.pt') == read('parent_provenance.json')['physical_input_sha256']
matrix = physical['contexts'][0]['matrix'].to(torch.complex128)
inverse = matrix.mH @ torch.linalg.inv(matrix @ matrix.mH)
assert torch.allclose(matrix @ inverse, torch.eye(matrix.shape[1], dtype=matrix.dtype), atol=1e-12, rtol=1e-12)
project = lambda value: torch.einsum('knj,bkj->bkn', inverse, torch.einsum('kjn,bkn->bkj', matrix, value))
norm = lambda value: value.abs().square().sum((1, 2))
references = {'fuse_ls': 'ls_min_norm', 'fuse_selected': 'lmmse_selected',
    'fuse_per_tone': 'lmmse_per_tone', 'fuse_selected_null': 'lmmse_selected'}
with (folder/'metrics.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with (folder/'validation.csv').open() as stream:
    validation = list(csv.DictReader(stream))
with (folder/'summary.csv').open() as stream:
    summary = list(csv.DictReader(stream))
with (folder/'paired_effects.csv').open() as stream:
    effects = list(csv.DictReader(stream))
values, maximum_score_relative_error, maximum_output_relative_error = {}, 0., 0.
solver = []
for index, entry in enumerate(observed['entries']):
    truth, y = entry['truth'].to(torch.complex128), entry['observation'].to(torch.complex128)
    energy = norm(truth)
    assert (norm(parts.sum(-1)-truth)/energy).max() < 1e-12
    saved = load(f'predictions/context_{index}.pt')
    assert saved['frozen_sha256'] == sha(run/'frozen.json')
    assert saved['observation_sha256'] == hashlib.sha256(entry['observation'].contiguous().numpy().tobytes()).hexdigest()
    outputs = {name: v.to(torch.complex128) for name, v in saved['outputs'].items()}
    ls = torch.einsum('knj,bkj->bkn', inverse, y).to(torch.complex64).to(torch.complex128)
    assert float((norm(ls-outputs['ls_min_norm'])/norm(ls)).max()) < 1e-12
    val = {name: v.to(torch.complex128) for name, v in load(f'cache/validation_{index}.pt').items()}
    for name, reference in references.items():
        vd = val[reference]-val['sbl100']
        direction = outputs[reference]-outputs['sbl100']
        if name.endswith('_null'):
            vd, direction = vd-project(vd), direction-project(direction)
        numerator = (((val['truth']-val['sbl100']).conj()*vd).real.sum((1, 2))/norm(val['truth'])).mean()
        denominator = (norm(vd)/norm(val['truth'])).mean()
        alpha = float((numerator/denominator).clamp(0, 1)) if denominator > 0 else 0.
        assert abs(alpha-frozen['weights_by_context'][str(index)][name]) < 1e-10
        vr = next(r for r in validation if int(r['context']) == index and r['method'] == name)
        assert abs(float((norm(val['sbl100']+alpha*vd-val['truth'])/norm(val['truth'])).mean())-float(vr['fused_nmse_linear'])) < 1e-10
        estimate = outputs['sbl100']+alpha*direction
        difference = float((norm(estimate.to(torch.complex64).to(torch.complex128)-outputs[name])/norm(outputs[name])).max())
        maximum_output_relative_error = max(maximum_output_relative_error, difference)
    equal_weight = frozen['weights_by_context'][str(index)]['fuse_selected']
    expected = outputs['sbl100']+equal_weight*(outputs['lmmse_per_tone']-outputs['sbl100'])
    assert float((norm(expected.to(torch.complex64).to(torch.complex128)-outputs['fuse_per_tone_same_weight'])/norm(expected)).max()) < 1e-12
    sensing = torch.einsum('kjn,bknl->bkjl', matrix, parts)
    for name, v, target in [('oracle_geometry_per_tone', sensing, y),
            ('oracle_geometry_shared', sensing.flatten(1, 2), y.flatten(1))]:
        coefficients = torch.linalg.solve(v.mH @ v, v.mH @ target[..., None]).squeeze(-1)
        estimate = (torch.einsum('bknl,bkl->bkn', parts, coefficients) if coefficients.ndim == 3
            else torch.einsum('bknl,bl->bkn', parts, coefficients))
        maximum_output_relative_error = max(maximum_output_relative_error,
            float((norm(estimate.to(torch.complex64).to(torch.complex128)-outputs[name])/norm(outputs[name])).max()))
    for name, estimate in outputs.items():
        errors = estimate-truth
        score = (norm(errors)/energy).numpy()
        values[index, name] = score
        local = sorted((r for r in rows if int(r['context']) == index and r['method'] == name), key=lambda r: int(r['scene']))
        assert [r['geometry_id'] for r in local] == ids
        measures = {'nmse_linear': score, 'row_nmse_linear': (norm(project(errors))/energy).numpy(),
            'null_nmse_linear': (norm(errors-project(errors))/energy).numpy(),
            'mse_physical': (norm(errors)/np.prod(errors.shape[1:])*physical['channel_scale']**2).numpy()}
        for key, value in measures.items():
            reported = np.array([float(r[key]) for r in local])
            maximum_score_relative_error = max(maximum_score_relative_error, float(np.max(abs(reported-value)/np.maximum(value, 1e-30))))
        result = next(r for r in summary if float(r['snr_db']) == physical['contexts'][index]['snr_db'] and r['method'] == name)
        assert abs(10*np.log10(score.mean())-float(result['mean_scene_nmse_db'])) < 1e-10
    info = saved['sbl_solver']
    assert (info['sbl_iterations'] == 100).all()
    solver.append({'snr_db': physical['contexts'][index]['snr_db'], 'optimality_max': float(info['sbl_optimality_residual'].max()),
        'reached_stopping_threshold': int(info['sbl_converged'].sum()), 'fixed_iterations': 100})
assert maximum_score_relative_error < 1e-7 and maximum_output_relative_error < 1e-12
assert len(rows) == len(observed['entries'])*len(ids)*len(outputs)
indices = np.random.default_rng(config['probe']['screen_seed']+153).integers(len(ids), size=(config['bootstrap_samples'], len(ids)))
levels = config['system']['snrs_db']
for effect in effects:
    index = levels.index(float(effect['snr_db']))
    a, b = values[index, effect['target']], values[index, effect['reference']]
    interval = np.quantile(10*np.log10(a[indices].mean(1)/b[indices].mean(1)), [.025, .975])
    assert abs(10*np.log10(a.mean()/b.mean())-float(effect['delta_db'])) < 1e-10
    assert np.allclose(interval, [float(effect['ci95_low_db']), float(effect['ci95_high_db'])], atol=1e-10, rtol=0)
verification = {'status': 'passed', 'cpu_double_precision_replay': True,
    'maximum_metric_relative_difference': maximum_score_relative_error,
    'maximum_prediction_relative_squared_difference': maximum_output_relative_error,
    'source_archive_sha256_verified': True, 'new_geometry_disjoint_from_all_parent_splits': True,
    'ls_recomputed_via_full_row_rank_inverse': True, 'validation_weights_refit_on_validation_only': True,
    'oracle_normal_equations_match_saved_pseudoinverse': True, 'paired_intervals_recomputed': True,
    'performance_acceptance_gate': False, 'solver_fixed_budget_diagnostics': solver}
(folder/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
main, control = values[0, 'fuse_selected'], values[0, 'fuse_ls']
comparison = 10*np.log10(main[indices].mean(1)/control[indices].mean(1))
at_low = lambda name, key: np.mean([float(r[key]) for r in rows if int(r['context']) == 0 and r['method'] == name])
total_drop = at_low('sbl100', 'nmse_linear')-at_low('fuse_selected', 'nmse_linear')
null_drop = at_low('sbl100', 'null_nmse_linear')-at_low('fuse_selected', 'null_nmse_linear')
analysis = {'run_id': run.name, 'independent_geometries': len(ids), 'conditional_on_fixed_training_and_hardware': True,
    'minus10_selected_vs_ls_delta_db': float(10*np.log10(main.mean()/control.mean())),
    'minus10_selected_vs_ls_ci95_db': np.quantile(comparison, [.025, .975]).tolist(),
    'minus10_full_fusion_linear_nmse_drop': float(total_drop),
    'minus10_full_fusion_null_nmse_drop': float(null_drop),
    'minus10_fraction_drop_from_null_space': float(null_drop/total_drop),
    'effects': effects, 'summary': summary}
(folder/'analysis_summary.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
names = [('sbl100', 'SBL100', 'o'), ('lmmse_selected', 'LMMSE', 's'), ('fuse_selected', 'SBL + LMMSE', '^'), ('fuse_ls', 'SBL + LS', 'x')]
for name, label, marker in names:
    axes[0].plot(levels, [10*np.log10(values[i, name].mean()) for i in range(len(levels))], marker=marker, label=label)
axes[0].set(title='Deployable estimates', ylabel='Mean scene NMSE (dB)')
for name, label, marker in [('fuse_selected', 'LMMSE fusion', 'o'), ('fuse_ls', 'LS fusion', 'x'), ('fuse_selected_null', 'LMMSE null-space fusion', 's')]:
    records = [next(r for r in effects if float(r['snr_db']) == snr and r['target'] == name and r['reference'] == 'sbl100') for snr in levels]
    delta = np.array([float(r['delta_db']) for r in records])
    low, high = np.array([float(r['ci95_low_db']) for r in records]), np.array([float(r['ci95_high_db']) for r in records])
    axes[1].plot(levels, delta, marker=marker, label=label)
    axes[1].fill_between(levels, low, high, alpha=.12)
axes[1].axhline(0, color='0.4', linewidth=.8)
axes[1].set(title='Paired change; below zero is better', ylabel='NMSE difference from SBL100 (dB)')
for name, label, marker in [('oracle_geometry_per_tone', 'True geometry: free per-tone gains', 's'), ('oracle_geometry_shared', 'True geometry: shared gains', 'o')]:
    axes[2].plot(levels, [10*np.log10(values[i, name].mean()) for i in range(len(levels))], marker=marker, label=label)
axes[2].set(title='Oracle diagnostics: true geometry supplied', ylabel='Mean scene NMSE (dB)')
for axis in axes:
    axis.set_xlabel('Training-average received SNR (dB)')
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
fig.savefig(folder/'complementarity.png', dpi=180)
fig.savefig(folder/'complementarity.svg')
print(json.dumps(verification))
