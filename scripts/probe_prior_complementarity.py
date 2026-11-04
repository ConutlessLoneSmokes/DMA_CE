"""Paired fixed-hardware probe of learned covariance and physical frequency coupling."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from dma_ce.data import estimate_entry, sample_scenes, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.factor_lmmse import apply as apply_factor, fit as fit_factor
from dma_ce.physical_study import archive, case_hardware, make_entries
from dma_ce.physics import paper_channel
from dma_ce.research import save_tensor


FUSIONS = {'fuse_ls': 'ls_min_norm', 'fuse_selected': 'lmmse_selected',
    'fuse_per_tone': 'lmmse_per_tone', 'fuse_selected_null': 'lmmse_selected'}
ORACLES = ('oracle_geometry_per_tone', 'oracle_geometry_shared')


def risk(truth, estimate):
    return (estimate-truth).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2))


def weight(truth, base, direction):
    energy = truth.abs().square().sum((1, 2))
    numerator = ((truth-base).conj()*direction).real.sum((1, 2)) / energy
    denominator = direction.abs().square().sum((1, 2)) / energy
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all() or (energy <= 0).any():
        raise ValueError('finite estimates and nonzero truth energy required for validation')
    return float((numerator.mean()/denominator.mean()).clamp(0, 1)) if denominator.mean() > 0 else 0.


def null_part(direction, matrix, inverse):
    observed = torch.einsum('kjn,bkn->bkj', matrix, direction)
    return direction-torch.einsum('knj,bkj->bkn', inverse, observed)


def oracle_outputs(parts, matrix, observation):
    sensing = torch.einsum('kjn,bknl->bkjl', matrix, parts)
    separate = (torch.linalg.pinv(sensing) @ observation[..., None]).squeeze(-1)
    joint = (torch.linalg.pinv(sensing.flatten(1, 2)) @ observation.flatten(1)[..., None]).squeeze(-1)
    return {ORACLES[0]: torch.einsum('bknl,bkl->bkn', parts, separate),
        ORACLES[1]: torch.einsum('bknl,bl->bkn', parts, joint)}


def self_check():
    truth = torch.ones(1, 1, 1, dtype=torch.complex128)
    zero = torch.zeros_like(truth)
    assert weight(truth, zero, 2*truth) == .5
    assert weight(truth, zero, -truth) == 0.
    assert weight(truth, zero, zero) == 0.
    assert weight(3*truth, zero, truth) == 1.
    a = torch.tensor([[[1., 0.]]], dtype=torch.complex128)
    direction = torch.tensor([[[3+1j, 4-2j]]], dtype=torch.complex128)
    expected = torch.tensor([[[0., 4-2j]]], dtype=torch.complex128)
    torch.testing.assert_close(null_part(direction, a, torch.linalg.pinv(a)), expected)
    parts = torch.zeros(1, 2, 3, 2, dtype=torch.complex128)
    parts[0, 0, 0, 0], parts[0, 0, 1, 1] = 1., 1.
    parts[0, 1, 0, 0], parts[0, 1, 2, 1] = 2., 3.
    a = torch.eye(3, dtype=torch.complex128).repeat(2, 1, 1)
    known = torch.tensor([[[1+1j, -.5+.2j, 0.], [2+2j, 0., -1.5+.6j]]], dtype=torch.complex128)
    for value in oracle_outputs(parts, a, known).values():
        torch.testing.assert_close(value, known, rtol=1e-12, atol=1e-12)


def run(settings_path, run_id):
    if re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id) is None:
        raise ValueError('safe independent run ID required')
    settings = json.loads(settings_path.read_text())
    if (re.fullmatch(r'[A-Za-z0-9_-]{1,80}', settings['parent_run']) is None
            or type(settings['screen_geometries']) is not int or settings['screen_geometries'] < 2
            or type(settings['screen_seed']) is not int or settings['screen_seed'] < 0
            or not 0 < settings['timeout_seconds'] <= 900 or settings['bootstrap_samples'] < 2):
        raise ValueError('bounded probe, independent seed and safe parent run required')
    self_check()
    parent = ROOT/'runs'/settings['parent_run']
    if json.loads((parent/'status.json').read_text())['status'] != 'complete':
        raise ValueError('completed parent run required')
    parent_config = json.loads((parent/'config_resolved.json').read_text())
    parent_frozen = json.loads((parent/'frozen.json').read_text())
    manifest = json.loads((parent/'source_manifest.json').read_text())
    for name, digest in manifest.items():
        if name.startswith(('src/', 'scripts/')) and name.endswith('.py') and file_hash(ROOT/name) != digest:
            raise ValueError(f'parent execution source differs: {name}')
    for name, digest in parent_frozen['files_sha256'].items():
        if file_hash(parent/name) != digest:
            raise ValueError(f'parent frozen input differs: {name}')
    config = copy.deepcopy(parent_config)
    config.update(label=settings['label'], device=settings['device'], timeout_seconds=settings['timeout_seconds'],
        bootstrap_samples=settings['bootstrap_samples'], probe=settings)
    config['data'].update(screen=settings['screen_geometries'], screen_seed=settings['screen_seed'])
    path, export = Path(settings['run_storage_root'])/run_id, ROOT/'results'/run_id
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError('existing absolute run storage directory required')
    if path.exists() or export.exists():
        raise ValueError('probe output already exists; use an independent run ID')
    if shutil.disk_usage(path.parent).free < 2_500_000_000:
        raise ValueError('at least 2.5 GB free space required for this bounded probe')
    path.mkdir(parents=True)
    export.mkdir(parents=True)
    for folder in ('inputs', 'cache', 'predictions'):
        (path/folder).mkdir()
    archive(ROOT, path, export, config)
    shutil.copyfile(parent/'cache/physical.pt', path/'inputs/physical_parent.pt')
    provenance = {'parent_run': settings['parent_run'], 'parent_source_sha256': file_hash(parent/'source.zip'),
        'parent_frozen_sha256': file_hash(parent/'frozen.json'), 'physical_input_sha256': file_hash(path/'inputs/physical_parent.pt'),
        'reused_files_sha256': parent_frozen['files_sha256'], 'new_screen_seed': settings['screen_seed']}
    write_json(path/'parent_provenance.json', provenance)
    write_json(export/'parent_provenance.json', provenance)
    def status(value):
        write_json(path/'status.json', value)
        write_json(export/'status.json', value)
    def log(event, **fields):
        record = {'event': event, 'at_unix': time.time(), **fields}
        with (path/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    def timeout():
        status({'status': 'timeout', 'limit_seconds': config['timeout_seconds']})
        os._exit(124)
    timer = threading.Timer(config['timeout_seconds'], timeout)
    timer.daemon = True
    timer.start()
    started = time.time()
    try:
        status({'status': 'validation', 'run_id': run_id})
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(config['device'])
        cache = torch.load(path/'inputs/physical_parent.pt', weights_only=True)
        selected, bank, hardware = case_hardware(config, config['cases'][0])
        assert torch.equal(torch.tensor(bank), cache['state_bank'])
        matrix = cache['contexts'][0]['matrix'].to(device, torch.complex128)
        assert all(torch.equal(c['matrix'], cache['contexts'][0]['matrix']) for c in cache['contexts'])
        assert tensor_hash(hardware[0]['matrix'][:, :matrix.shape[1]]) == tensor_hash(cache['contexts'][0]['matrix'])
        inverse = torch.linalg.pinv(matrix)
        training = cache['training'][0].to(device)
        def linear_outputs(entry, context, index):
            observation = entry['observation'].to(device)
            stats = torch.load(parent/f'cache/lmmse_{index}.pt', weights_only=True)
            selected_output = apply_factor(stats, observation).to(torch.complex128)
            del stats
            stats = fit_factor(training, matrix, context['noise_complex_variance'], 1.)
            per_tone = apply_factor(stats, observation).to(torch.complex128)
            return {'ls_min_norm': torch.einsum('knj,bkj->bkn', inverse, observation.to(torch.complex128)).to(torch.complex64).to(torch.complex128),
                'lmmse_selected': selected_output, 'lmmse_per_tone': per_tone}
        validation, choices = [], {}
        for index, (context, entry) in enumerate(zip(cache['contexts'], cache['validation'])):
            coarse = torch.load(parent/f'cache/coarse_validation_{index}.pt', weights_only=True)
            assert tensor_hash(coarse['observation']) == tensor_hash(entry['observation'])
            assert (coarse['sbl_iterations'] == config['ddpm']['coarse_iterations']).all()
            base, truth = coarse['coarse'].to(device, torch.complex128), entry['truth'].to(device, torch.complex128)
            outputs = linear_outputs(entry, context, index)
            choices[str(index)] = {}
            for name, reference in FUSIONS.items():
                direction = outputs[reference]-base
                if name.endswith('_null'):
                    direction = null_part(direction, matrix, inverse)
                alpha = weight(truth, base, direction)
                choices[str(index)][name] = alpha
                error_a, error_b = base-truth, base+direction-truth
                energy = truth.abs().square().sum((1, 2))
                cross = (error_a.conj()*error_b).real.sum((1, 2))/energy
                a, b = float(risk(truth, base).mean()), float(risk(truth, base+direction).mean())
                validation.append({'context': index, 'snr_db': context['snr_db'], 'method': name, 'weight': alpha,
                    'base_nmse_linear': a, 'other_nmse_linear': b, 'error_cross_mean': float(cross.mean()),
                    'aggregate_error_correlation': float(cross.mean()/np.sqrt(a*b)),
                    'fused_nmse_linear': float(risk(truth, base+alpha*direction).mean())})
            save_tensor(path/f'cache/validation_{index}.pt', {'truth': truth.cpu().to(torch.complex64), 'observation': entry['observation'],
                'sbl100': base.cpu().to(torch.complex64), **{k: v.cpu().to(torch.complex64) for k, v in outputs.items()}})
            log('validation_context', context=index, snr_db=context['snr_db'], weights=choices[str(index)])
        write_csv(export/'validation.csv', validation)
        frozen = {'selection_split': 'validation', 'screening_used': False, 'frozen_at_unix': time.time(),
            'weights_by_context': choices, 'frequency_shrinkage': parent_frozen['frequency_shrinkage'],
            'sbl_iterations': config['ddpm']['coarse_iterations'],
            'files_sha256': {str(p.relative_to(path)): file_hash(p) for p in
                [path/'source.zip', path/'inputs/physical_parent.pt', *sorted((path/'cache').glob('validation_*.pt'))]}}
        write_json(path/'frozen.json', frozen)
        write_json(export/'frozen.json', frozen)
        status({'status': 'screen', 'run_id': run_id})
        entries, ids = make_entries(config, selected, hardware, cache, 'screen')
        old = torch.load(parent/'predictions/observations.pt', weights_only=True)
        known = set(cache['geometry_ids']['train']+cache['geometry_ids']['validation']+old['geometry_ids'])
        assert len(set(ids)) == config['data']['screen'] and not set(ids) & known
        del old
        save_tensor(path/'predictions/observations.pt', {'entries': entries, 'geometry_ids': ids,
            'created_at_unix': time.time(), 'frozen_sha256': file_hash(path/'frozen.json')})
        radii, angles, gains = sample_scenes(config, 2, len(ids))
        parts = []
        for component in range(config['channel']['paths']):
            coefficients = np.zeros_like(gains)
            coefficients[:, component] = gains[:, component]
            parts.append(paper_channel(hardware[0], (radii, angles, coefficients))/cache['channel_scale'])
        parts = torch.tensor(np.stack(parts, axis=-1), dtype=torch.complex128, device=device)
        relative_truth = float((parts.sum(-1)-entries[0]['truth'].to(device)).abs().max()/parts.sum(-1).abs().max())
        assert relative_truth < 1e-6
        save_tensor(path/'inputs/oracle_path_basis.pt', parts.cpu())
        rows, observations_unchanged, maximum_null_residual = [], [], 0.
        dictionary = hardware[0]['dictionary'].to(device)
        sbl_settings = {**config['sbl'], 'iterations': config['ddpm']['coarse_iterations'], 'early_stopping': False}
        for index, (context, entry) in enumerate(zip(cache['contexts'], entries)):
            truth = entry['truth'].to(device, torch.complex128)
            observation = entry['observation'].to(device, torch.complex128)
            observation_hash = tensor_hash(entry['observation'])
            estimated = estimate_entry(entry, context, dictionary, sbl_settings, device, log)
            base = estimated['coarse'].to(device, torch.complex128)
            outputs = {'sbl100': base, **linear_outputs(entry, context, index)}
            for name, reference in FUSIONS.items():
                direction = outputs[reference]-base
                if name.endswith('_null'):
                    direction = null_part(direction, matrix, inverse)
                    error = torch.einsum('kjn,bkn->bkj', matrix, direction).abs().max()
                    scale = direction.abs().max().clamp_min(1e-30)*matrix.abs().sum(-1).max()
                    maximum_null_residual = max(maximum_null_residual, float(error/scale))
                outputs[name] = base+choices[str(index)][name]*direction
            outputs['fuse_per_tone_same_weight'] = base+choices[str(index)]['fuse_selected']*(outputs['lmmse_per_tone']-base)
            outputs.update(oracle_outputs(parts, matrix, observation))
            # Match the parent prediction precision; evaluate saved values in double precision.
            outputs = {name: value.to(torch.complex64).to(torch.complex128) for name, value in outputs.items()}
            energy = truth.abs().square().sum((1, 2))
            for name, value in outputs.items():
                errors = value-truth
                row_error = errors-null_part(errors, matrix, inverse)
                error_nmse = risk(truth, value)
                physical_mse = errors.abs().square().mean((1, 2))*cache['channel_scale']**2
                row_nmse = row_error.abs().square().sum((1, 2))/energy
                null_nmse = (errors-row_error).abs().square().sum((1, 2))/energy
                assert torch.isfinite(error_nmse).all()
                assert torch.allclose(error_nmse, row_nmse+null_nmse, rtol=1e-8, atol=1e-12)
                for scene in range(len(ids)):
                    rows.append({'context': index, 'snr_db': context['snr_db'], 'scene': scene, 'geometry_id': ids[scene],
                        'method': name, 'diagnostic_oracle': name in ORACLES, 'nmse_linear': float(error_nmse[scene]),
                        'mse_physical': float(physical_mse[scene]), 'row_nmse_linear': float(row_nmse[scene]),
                        'null_nmse_linear': float(null_nmse[scene]), 'observation_sha256': tensor_hash(entry['observation'][scene])})
            save_tensor(path/f'predictions/context_{index}.pt', {'outputs': {k: v.cpu().to(torch.complex64) for k, v in outputs.items()},
                'sbl_solver': {k: v for k, v in estimated.items() if k.startswith('sbl_')},
                'observation_sha256': observation_hash, 'frozen_sha256': file_hash(path/'frozen.json')})
            observations_unchanged.append(observation_hash == tensor_hash(entry['observation']))
            log('screen_context', context=index, snr_db=context['snr_db'])
        write_csv(export/'metrics.csv', rows)
        rng = np.random.default_rng(settings['screen_seed']+153)
        indices = rng.integers(len(ids), size=(settings['bootstrap_samples'], len(ids)))
        summary, effects = [], []
        methods = list(outputs)
        for index, context in enumerate(cache['contexts']):
            values = {}
            for name in methods:
                local = [r for r in rows if r['context'] == index and r['method'] == name]
                values[name] = np.array([r['nmse_linear'] for r in local])
                summary.append({'snr_db': context['snr_db'], 'method': name, 'diagnostic_oracle': name in ORACLES,
                    'scenes': len(ids), 'mean_nmse_linear': float(values[name].mean()),
                    'mean_scene_nmse_db': float(10*np.log10(values[name].mean())),
                    'mean_physical_mse': float(np.mean([r['mse_physical'] for r in local]))})
            pairs = [(name, 'sbl100') for name in methods if name.startswith('fuse_')]
            pairs += [('fuse_selected', 'fuse_per_tone_same_weight'), (ORACLES[1], ORACLES[0])]
            for target, reference in pairs:
                draws = 10*np.log10(values[target][indices].mean(1)/values[reference][indices].mean(1))
                low, high = np.quantile(draws, [.025, .975])
                effects.append({'snr_db': context['snr_db'], 'target': target, 'reference': reference,
                    'delta_db': float(10*np.log10(values[target].mean()/values[reference].mean())),
                    'ci95_low_db': float(low), 'ci95_high_db': float(high),
                    'target_better_scenes': int((values[target] < values[reference]).sum())})
        write_csv(export/'summary.csv', summary)
        write_csv(export/'paired_effects.csv', effects)
        assert all(observations_unchanged) and maximum_null_residual < 1e-10
        assert len(rows) == len(entries)*len(ids)*len(methods)
        for name, digest in frozen['files_sha256'].items():
            assert file_hash(path/name) == digest
        observed = torch.load(path/'predictions/observations.pt', weights_only=True)
        assert observed['created_at_unix'] > frozen['frozen_at_unix']
        assert len({e['physical_noise_sha256'] for e in entries}) == 1
        audit = {'status': 'passed', 'run_id': run_id, 'independent_screen_geometries': len(ids),
            'new_screen_disjoint_from_parent': True, 'frozen_before_screen': True,
            'same_physical_noise_across_snr': True, 'same_observation_across_methods': True,
            'metric_rows': len(rows), 'maximum_null_projection_relative_residual': maximum_null_residual,
            'oracle_truth_reconstruction_relative_error': relative_truth, 'oracle_not_a_main_method': True,
            'validation_only_weight_selection': True, 'performance_acceptance_gate': False,
            'elapsed_seconds': time.time()-started}
        write_json(export/'audit.json', audit)
        status({**audit, 'status': 'complete'})
        log('complete', elapsed_seconds=audit['elapsed_seconds'])
        return export
    except BaseException as error:
        status({'status': 'failed', 'error': repr(error), 'elapsed_seconds': time.time()-started})
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    print(run(args.config, args.run_id))
