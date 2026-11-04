"""Replay saved metrics and matched physical fits; run from the repository root."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'src'))
from dma_ce.data import tensor_hash
from dma_ce.experiment import write_csv, write_json
from dma_ce.physics import paper_channel
from dma_ce.physical_study import case_hardware

torch.set_num_threads(4)
run = Path(sys.argv[1]).resolve()
out = run/'export'
config = json.loads((run/'config_resolved.json').read_text())
frozen = json.loads((run/'frozen.json').read_text())
parent = torch.load(run/'inputs/physical_parent.pt', weights_only=True)
observed = torch.load(run/'predictions/observations.pt', weights_only=True)
provenance = json.loads((run/'parent_provenance.json').read_text())
contexts = [parent['contexts'][i] for i in provenance['original_context_indices']]
assert json.loads((run/'status.json').read_text())['status'] == 'complete'
assert observed['created_at_unix'] > frozen['frozen_at_unix']
assert observed['frozen_sha256'] == hashlib.sha256((run/'frozen.json').read_bytes()).hexdigest()
for name, digest in frozen['files_sha256'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest() == digest, name
with zipfile.ZipFile(run/'source.zip') as archive:
    for name, digest in json.loads((run/'source_manifest.json').read_text()).items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
known = set(parent['geometry_ids']['train']+parent['geometry_ids']['validation'])
for name in ('multirf_snr_20261022_v1', 'prior_complementarity_20261023_v1'):
    with (ROOT/'results'/name/'metrics.csv').open() as stream:
        known.update(row['geometry_id'] for row in csv.DictReader(stream))
ids = observed['geometry_ids']
assert len(set(ids)) == len(ids) == 32 and not known.intersection(ids)
assert len({entry['physical_noise_sha256'] for entry in observed['entries']}) == 1
with (out/'metrics.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with (out/'summary.csv').open() as stream:
    summary = list(csv.DictReader(stream))
with (out/'paired_effects.csv').open() as stream:
    effects = list(csv.DictReader(stream))
recomputed, diagnostic = {}, []
metric_error = 0.
matrix = contexts[0]['matrix'].to(torch.complex128)
row_basis = torch.linalg.qr(matrix.mH, mode='reduced').Q
for context, entry in enumerate(observed['entries']):
    saved = torch.load(run/f'predictions/context_{context}.pt', weights_only=True)
    assert saved['observation_sha256'] == tensor_hash(entry['observation'])
    assert saved['frozen_sha256'] == observed['frozen_sha256']
    truth = entry['truth'].numpy().astype(np.complex128)
    energy = np.sum(np.abs(truth)**2, axis=(1, 2))
    for name, output in saved['outputs'].items():
        error = output.numpy().astype(np.complex128)-truth
        nmse = np.sum(np.abs(error)**2, axis=(1, 2))/energy
        mse = np.mean(np.abs(error)**2, axis=(1, 2))*parent['channel_scale']**2
        err = torch.from_numpy(error)
        row = (row_basis @ (row_basis.mH @ err.permute(1, 2, 0))).permute(2, 0, 1)
        row_nmse = row.abs().square().sum((1, 2)).numpy()/energy
        null_nmse = (err-row).abs().square().sum((1, 2)).numpy()/energy
        residual = (matrix @ output.to(torch.complex128).permute(1, 2, 0)).permute(2, 0, 1)-entry['observation']
        observed_error = residual.abs().square().mean((1, 2)).numpy()/contexts[context]['noise_complex_variance']
        local = [r for r in rows if int(r['context']) == context and r['method'] == name]
        assert [r['geometry_id'] for r in local] == ids
        assert all(r['observation_sha256'] == tensor_hash(entry['observation'][scene])
            for scene, r in enumerate(local))
        for field, actual in [('nmse_linear', nmse), ('mse_physical', mse),
                ('row_nmse_linear', row_nmse), ('null_nmse_linear', null_nmse),
                ('normalized_observation_residual', observed_error)]:
            expected = np.array([float(r[field]) for r in local])
            metric_error = max(metric_error, float(np.max(np.abs(actual-expected)/np.maximum(abs(expected), 1e-30))))
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-15)
        recomputed[context, name] = nmse
    strength = frozen['covariance_strength_by_context'][str(context)]
    if not strength:
        assert torch.equal(saved['outputs']['sbl100'], saved['outputs']['covariance_sbl'])
    state = saved['states']
    diagnostic.append({'snr_db': contexts[context]['snr_db'], 'selected_strength': strength,
        'solver': {name: {'fixed_iterations': config['probe']['sbl_iterations'],
            'reached_original_threshold': int((state[name]['optimality_residual'] <= config['sbl']['tolerance']).sum()),
            'optimality_max': float(state[name]['optimality_residual'].max())}
            for name in ('sbl100', 'covariance_sbl')}})
    for stage in ('refined', 'joint'):
        assert (state[stage]['observation_loss'] <= state[stage]['initial_observation_loss']+1e-8).all()
for item in summary:
    actual = recomputed[int(item['context']), item['method']].mean()
    np.testing.assert_allclose(actual, float(item['mean_nmse_linear']), rtol=1e-12)
    np.testing.assert_allclose(10*np.log10(actual), float(item['mean_scene_nmse_db']), atol=1e-10)
draw = np.random.default_rng(config['probe']['screen_seed']+153).integers(len(ids),
    size=(config['bootstrap_samples'], len(ids)))
for item in effects:
    target = recomputed[int(item['context']), item['target']]
    reference = recomputed[int(item['context']), item['reference']]
    delta = 10*np.log10(target[draw].mean(1)/reference[draw].mean(1))
    np.testing.assert_allclose(np.quantile(delta, [.025, .975]),
        [float(item['ci95_low_db']), float(item['ci95_high_db'])], atol=1e-10)
with (out/'validation.csv').open() as stream:
    validation = list(csv.DictReader(stream))
for context in range(len(contexts)):
    entry = torch.load(run/f'cache/validation_{context}.pt', weights_only=True)
    assert entry['geometry_ids'] == frozen['validation_geometry_ids']
    scores = {}
    for strength in config['probe']['covariance_strengths']:
        truth = entry['truth'].numpy().astype(np.complex128)
        output = entry['outputs'][str(strength)].numpy().astype(np.complex128)
        score = float(np.mean(np.sum(abs(output-truth)**2, axis=(1, 2))/np.sum(abs(truth)**2, axis=(1, 2))))
        match = next(r for r in validation if int(r['context']) == context and float(r['strength']) == strength)
        np.testing.assert_allclose(score, float(match['mean_nmse_linear']), rtol=1e-12)
        scores[strength] = score
    assert min(scores, key=lambda x: (scores[x], x)) == frozen['covariance_strength_by_context'][str(context)]
write_json(out/'solver_diagnostics.json', diagnostic)

_, _, hardware = case_hardware(config, config['cases'][0])
hw = hardware[0]
oracle = torch.load(run/'inputs/oracle_geometry.pt', weights_only=True).numpy()
matrix_numpy = matrix.numpy()
prediction_error = 0.
fit_diagnostics = []
for context, entry in enumerate(observed['entries']):
    saved = torch.load(run/f'predictions/context_{context}.pt', weights_only=True)
    groups = [(saved['states']['initial_geometry'].numpy(),
        [('physical_initial_per_tone', False), ('physical_initial_shared', True)]),
        (saved['states']['refined']['geometry'].numpy(),
        [('physical_refined_per_tone', False), ('physical_refined_shared', True)]),
        (saved['states']['joint']['geometry'].numpy(), [('physical_joint_shared', True)]),
        (oracle, [('oracle_geometry_per_tone', False), ('oracle_geometry_shared', True)])]
    for geometry, methods in groups:
        gains = np.zeros((len(geometry), config['channel']['paths']), dtype=np.complex128)
        columns = []
        for path in range(config['channel']['paths']):
            gains[:] = 0
            gains[:, path] = 1
            columns.append(paper_channel(hw, (geometry[..., 1], geometry[..., 0], gains))/parent['channel_scale'])
        basis = np.stack(columns, axis=-1)
        design = np.einsum('kjn,bknl->bkjl', matrix_numpy, basis, optimize=True)
        for name, shared in methods:
            estimates = []
            for scene in range(len(ids)):
                target = entry['observation'][scene].numpy().astype(np.complex128)
                blocks = [(design[scene].reshape(-1, gains.shape[-1]), target.reshape(-1))] if shared else zip(design[scene], target)
                coefficients = []
                for block, y in blocks:
                    regularizer = config['probe']['relative_path_ridge']*np.mean(np.sum(abs(block)**2, axis=0))
                    coefficient = np.linalg.lstsq(np.vstack((block, np.sqrt(regularizer)*np.eye(gains.shape[-1]))),
                        np.r_[y, np.zeros(gains.shape[-1])], rcond=None)[0]
                    coefficients.append(coefficient)
                estimates.append(np.einsum('knl,l->kn', basis[scene], coefficients[0]) if shared else
                    np.einsum('knl,kl->kn', basis[scene], np.stack(coefficients)))
            estimates = np.stack(estimates)
            expected = saved['outputs'][name].numpy().astype(np.complex128)
            relative = np.sum(abs(estimates-expected)**2, axis=(1, 2))/np.sum(abs(expected)**2, axis=(1, 2))
            assert float(relative.max()) < 1e-10, (context, name, relative.max())
            prediction_error = max(prediction_error, float(relative.max()))
    for target, reference in [('physical_refined_per_tone', 'physical_initial_per_tone'),
            ('physical_refined_shared', 'physical_refined_per_tone'),
            ('physical_joint_shared', 'physical_refined_shared')]:
        local = {name: [r for r in rows if int(r['context']) == context and r['method'] == name]
            for name in (target, reference)}
        residual_delta = np.array([float(a['normalized_observation_residual'])-float(b['normalized_observation_residual'])
            for a, b in zip(local[target], local[reference])])
        csi_delta = recomputed[context, target]-recomputed[context, reference]
        fit_diagnostics.append({'snr_db': contexts[context]['snr_db'], 'target': target, 'reference': reference,
            'lower_observation_residual_scenes': int((residual_delta < 0).sum()),
            'better_csi_scenes': int((csi_delta < 0).sum()),
            'lower_residual_but_worse_csi_scenes': int(((residual_delta < 0) & (csi_delta > 0)).sum())})
write_csv(out/'fit_diagnostics.csv', fit_diagnostics)
verification = {'status': 'passed', 'metric_rows_recomputed': len(rows),
    'maximum_metric_relative_difference': metric_error,
    'maximum_physical_prediction_relative_squared_difference': prediction_error,
    'physical_fits_replayed_with_original_numpy_propagation_and_augmented_lstsq': True,
    'row_null_metrics_replayed_with_qr_projection': True,
    'validation_choices_and_paired_intervals_recomputed': True,
    'source_archive_and_frozen_inputs_verified': True,
    'new_screen_geometry_disjoint_from_all_previous_splits': True,
    'frozen_before_new_screen_observations': True,
    'performance_acceptance_gate': False,
    'inference_optimizers_rerun': False}
write_json(out/'verification.json', verification)
print(json.dumps(verification))
