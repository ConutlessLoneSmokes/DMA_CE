"""Bounded non-neural experiment: internal covariance and estimated physical geometry."""
import argparse
import copy
import csv
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
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch

from dma_ce.data import sample_scenes, tensor_hash
from dma_ce.experiment import file_hash, write_csv, write_json
from dma_ce.physical_study import archive, case_hardware, make_entries
from dma_ce.prior_probe import covariance_sbl, fit_paths, initial_geometry, propagation_basis, refine_geometry
from dma_ce.research import save_tensor


ORACLES = ('oracle_geometry_per_tone', 'oracle_geometry_shared')


def risk(truth, estimate):
    return (truth-estimate).abs().square().sum((1, 2))/truth.abs().square().sum((1, 2))


def run(settings_path, run_id):
    settings = json.loads(settings_path.read_text())
    if (re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id) is None
            or re.fullmatch(r'[A-Za-z0-9_-]{1,80}', settings['parent_run']) is None
            or not 0 < settings['timeout_seconds'] <= 1800
            or settings['validation_geometries'] != 32 or settings['screen_geometries'] != 32
            or settings['sbl_iterations'] != 100 or settings['relative_path_ridge'] != 1e-8
            or settings['snrs_db'] != [-10, 0, 10, 20, 30]
            or settings['covariance_strengths'] != [0, .01, .1, 1]):
        raise ValueError('this probe requires its declared bounded design')
    parent = ROOT/'runs'/settings['parent_run']
    if json.loads((parent/'status.json').read_text())['status'] != 'complete':
        raise ValueError('complete parent required')
    original = json.loads((parent/'config_resolved.json').read_text())
    original_frozen = json.loads((parent/'frozen.json').read_text())
    indices = [original['system']['snrs_db'].index(snr) for snr in settings['snrs_db']]
    used = ['cache/physical.pt', 'source.zip', *[f'cache/coarse_validation_{i}.pt' for i in indices]]
    for name in used:
        if file_hash(parent/name) != original_frozen['files_sha256'][name]:
            raise ValueError(f'parent frozen input changed: {name}')
    for name, digest in json.loads((parent/'source_manifest.json').read_text()).items():
        if name.startswith(('src/', 'scripts/')) and name.endswith('.py') and file_hash(ROOT/name) != digest:
            raise ValueError(f'parent execution source changed: {name}')
    config = copy.deepcopy(original)
    config.update(label=settings['label'], device=settings['device'], timeout_seconds=settings['timeout_seconds'],
        bootstrap_samples=settings['bootstrap_samples'], probe=settings)
    config['system']['snrs_db'] = settings['snrs_db']
    config['data'].update(validation=settings['validation_geometries'], screen=settings['screen_geometries'],
        screen_seed=settings['screen_seed'])
    path = Path(settings['run_storage_root'])/run_id
    export = path/'export'
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ValueError('unused run ID and existing absolute storage directory required')
    if shutil.disk_usage(path.parent).free < 3_000_000_000:
        raise ValueError('3 GB free storage required')
    path.mkdir()
    for name in ('inputs', 'cache', 'predictions', 'export'):
        (path/name).mkdir()
    archive(ROOT, path, export, config)
    shutil.copyfile(parent/'cache/physical.pt', path/'inputs/physical_parent.pt')
    write_json(path/'parent_provenance.json', {'parent_run': settings['parent_run'],
        'used_parent_files_sha256': {name: file_hash(parent/name) for name in used},
        'parent_frozen_sha256': file_hash(parent/'frozen.json'), 'original_context_indices': indices})
    shutil.copyfile(path/'parent_provenance.json', export/'parent_provenance.json')
    def status(value):
        write_json(path/'status.json', value)
        write_json(export/'status.json', value)
    def log(event, **fields):
        item = {'event': event, 'at_unix': time.time(), **fields}
        with (path/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(item)+'\n')
        print(json.dumps(item), flush=True)
    def timeout():
        status({'status': 'timeout', 'limit_seconds': settings['timeout_seconds']})
        os._exit(124)
    timer = threading.Timer(settings['timeout_seconds'], timeout)
    timer.daemon = True
    timer.start()
    started = time.time()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        device = torch.device(settings['device'])
        physical = torch.load(path/'inputs/physical_parent.pt', weights_only=True)
        selected_config, bank, hardware = case_hardware(config, config['cases'][0])
        assert torch.equal(torch.tensor(bank), physical['state_bank'])
        matrix = physical['contexts'][0]['matrix'].to(device)
        assert tensor_hash(hardware[0]['matrix'][:, :matrix.shape[1]]) == tensor_hash(matrix)
        contexts = [physical['contexts'][i] for i in indices]
        assert all(torch.equal(c['matrix'], physical['contexts'][0]['matrix']) for c in contexts)
        training = physical['training'][0].to(device)
        training_factor = ((training-training.mean(0))/np.sqrt(len(training)-1)).permute(1, 2, 0).contiguous()
        dictionary = hardware[0]['dictionary'].to(device)
        batch_size = config['sbl']['batch_size']
        def infer(entry, strength, context):
            chunks = []
            for first in range(0, len(entry['observation']), batch_size):
                result = covariance_sbl(entry['observation'][first:first+batch_size].to(device), matrix, dictionary,
                    entry['variance'][first:first+batch_size].to(device), training_factor, strength,
                    settings['sbl_iterations'])
                chunks.append({name: value.cpu() for name, value in result.items() if isinstance(value, torch.Tensor)})
                log('covariance_batch', context=context, strength=strength,
                    completed=min(first+batch_size, len(entry['observation'])))
            return {name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]}
        status({'status': 'validation', 'run_id': run_id})
        choices, validation = {}, []
        count = settings['validation_geometries']
        for index, original_index in enumerate(indices):
            raw = physical['validation'][original_index]
            entry = {name: value[:count] if isinstance(value, torch.Tensor) and value.ndim else value for name, value in raw.items()}
            saved_base = torch.load(parent/f'cache/coarse_validation_{original_index}.pt', weights_only=True)
            assert tensor_hash(saved_base['observation'][:count]) == tensor_hash(entry['observation'])
            assert (saved_base['sbl_iterations'][:count] == settings['sbl_iterations']).all()
            outputs = {'0': saved_base['coarse'][:count]}
            for strength in settings['covariance_strengths'][1:]:
                outputs[str(strength)] = infer(entry, strength, index)['mean']
            scores = {}
            for strength in settings['covariance_strengths']:
                values = risk(entry['truth'].to(torch.complex128), outputs[str(strength)].to(torch.complex128))
                scores[str(strength)] = float(values.mean())
                validation.append({'context': index, 'snr_db': contexts[index]['snr_db'], 'strength': strength,
                    'mean_nmse_linear': scores[str(strength)], 'scenes': count})
            choices[str(index)] = min(settings['covariance_strengths'], key=lambda value: (scores[str(value)], value))
            save_tensor(path/f'cache/validation_{index}.pt', {'truth': entry['truth'], 'observation': entry['observation'],
                'outputs': outputs, 'geometry_ids': physical['geometry_ids']['validation'][:count]})
            log('validation_context', context=index, selected_strength=choices[str(index)], scores=scores)
        write_csv(export/'validation.csv', validation)
        frozen = {'selection_split': 'validation', 'screening_used': False, 'frozen_at_unix': time.time(),
            'covariance_strength_by_context': choices, 'validation_geometry_ids': physical['geometry_ids']['validation'][:count],
            'physical_settings': {name: settings[name] for name in ('peak_coherence', 'geometry_steps',
                'geometry_learning_rate', 'joint_steps', 'joint_learning_rate', 'relative_path_ridge')},
            'files_sha256': {str(p.relative_to(path)): file_hash(p) for p in
                [path/'source.zip', path/'inputs/physical_parent.pt', *sorted((path/'cache').glob('validation_*.pt'))]}}
        write_json(path/'frozen.json', frozen)
        write_json(export/'frozen.json', frozen)
        status({'status': 'screen', 'run_id': run_id})
        reduced_cache = {**physical, 'contexts': contexts}
        entries, ids = make_entries(config, selected_config, hardware, reduced_cache, 'screen')
        known = set(physical['geometry_ids']['train']+physical['geometry_ids']['validation'])
        for name in ('multirf_snr_20261022_v1', 'prior_complementarity_20261023_v1'):
            with (ROOT/'results'/name/'metrics.csv').open() as stream:
                known.update(row['geometry_id'] for row in csv.DictReader(stream))
        assert len(set(ids)) == len(ids) == settings['screen_geometries'] and not known & set(ids)
        save_tensor(path/'predictions/observations.pt', {'entries': entries, 'geometry_ids': ids,
            'created_at_unix': time.time(), 'frozen_sha256': file_hash(path/'frozen.json')})
        frequencies = torch.tensor(hardware[0]['frequencies'], dtype=torch.float64, device=device)
        positions = torch.tensor(hardware[0]['positions'], dtype=torch.float64, device=device)
        table = hardware[0]['absorption_table']
        kappa = torch.tensor(table.coefficient(hardware[0]['frequencies']) if table is not None
            else np.zeros(len(frequencies)), dtype=torch.float64, device=device)
        matrix_double = matrix.to(torch.complex128)
        inverse = torch.linalg.pinv(matrix_double)
        scale, channel_settings = physical['channel_scale'], config['channel']
        radii, angles, _ = sample_scenes(config, 2, len(ids))
        oracle_geometry = torch.tensor(np.stack((angles, radii), -1), dtype=torch.float64, device=device)
        save_tensor(path/'inputs/oracle_geometry.pt', oracle_geometry.cpu())
        rows = []
        for index, entry in enumerate(entries):
            base = infer(entry, 0, index)
            strength = choices[str(index)]
            statistical = infer(entry, strength, index) if strength else base
            outputs = {'sbl100': base['mean'], 'covariance_sbl': statistical['mean']}
            geometry, peaks = initial_geometry(base['gamma'].to(device), dictionary, config, settings['peak_coherence'])
            observation = entry['observation'].to(device, torch.complex128)
            physical_arguments = (matrix_double, observation, frequencies, positions, kappa, channel_settings, scale)
            with torch.no_grad():
                parts = propagation_basis(geometry, frequencies, positions, kappa, channel_settings, scale)
                outputs['physical_initial_per_tone'] = fit_paths(parts, matrix_double, observation, False)[0].cpu()
                outputs['physical_initial_shared'] = fit_paths(parts, matrix_double, observation, True)[0].cpu()
            log('physical_stage', context=index, stage='initial_geometry')
            refined = refine_geometry(geometry, *physical_arguments, shared=False,
                steps=settings['geometry_steps'], learning_rate=settings['geometry_learning_rate'])
            outputs['physical_refined_per_tone'] = refined['mean'].cpu()
            with torch.no_grad():
                parts = propagation_basis(refined['geometry'], frequencies, positions, kappa, channel_settings, scale)
                outputs['physical_refined_shared'] = fit_paths(parts, matrix_double, observation, True)[0].cpu()
            log('physical_stage', context=index, stage='per_tone_geometry_refined')
            joint = refine_geometry(refined['geometry'], *physical_arguments, shared=True,
                steps=settings['joint_steps'], learning_rate=settings['joint_learning_rate'])
            outputs['physical_joint_shared'] = joint['mean'].cpu()
            log('physical_stage', context=index, stage='joint_geometry_refined')
            with torch.no_grad():
                # Ground-truth geometry enters only these explicitly labelled diagnostics.
                parts = propagation_basis(oracle_geometry, frequencies, positions, kappa, channel_settings, scale)
                outputs[ORACLES[0]] = fit_paths(parts, matrix_double, observation, False)[0].cpu()
                outputs[ORACLES[1]] = fit_paths(parts, matrix_double, observation, True)[0].cpu()
            states = {'initial_geometry': geometry.cpu(), 'initial_peak_indices': peaks.cpu(),
                'refined': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in refined.items() if k != 'mean'},
                'joint': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in joint.items() if k != 'mean'},
                'sbl100': {k: v for k, v in base.items() if k != 'mean'},
                'covariance_sbl': {k: v for k, v in statistical.items() if k != 'mean'}}
            outputs = {name: value.to(torch.complex64) for name, value in outputs.items()}
            truth = entry['truth'].to(device, torch.complex128)
            energy = truth.abs().square().sum((1, 2))
            for name, value in outputs.items():
                error = value.to(device, torch.complex128)-truth
                measured_error = torch.einsum('kjn,bkn->bkj', matrix_double, error)
                row_error = torch.einsum('knj,bkj->bkn', inverse, measured_error)
                nmse = error.abs().square().sum((1, 2))/energy
                row_nmse = row_error.abs().square().sum((1, 2))/energy
                null_nmse = (error-row_error).abs().square().sum((1, 2))/energy
                physical_mse = error.abs().square().mean((1, 2))*scale**2
                observation_error = torch.einsum('kjn,bkn->bkj', matrix_double,
                    value.to(device, torch.complex128))-observation
                observed_residual = observation_error.abs().square().mean((1, 2))/contexts[index]['noise_complex_variance']
                assert torch.isfinite(nmse).all()
                assert torch.allclose(nmse, row_nmse+null_nmse, rtol=1e-8, atol=1e-12)
                for scene in range(len(ids)):
                    rows.append({'context': index, 'snr_db': contexts[index]['snr_db'], 'scene': scene,
                        'geometry_id': ids[scene], 'method': name, 'diagnostic_oracle': name in ORACLES,
                        'nmse_linear': float(nmse[scene]), 'mse_physical': float(physical_mse[scene]),
                        'row_nmse_linear': float(row_nmse[scene]), 'null_nmse_linear': float(null_nmse[scene]),
                        'normalized_observation_residual': float(observed_residual[scene]),
                        'observation_sha256': tensor_hash(entry['observation'][scene])})
            save_tensor(path/f'predictions/context_{index}.pt', {'outputs': outputs, 'states': states,
                'observation_sha256': tensor_hash(entry['observation']), 'frozen_sha256': file_hash(path/'frozen.json')})
            log('screen_context', context=index, snr_db=contexts[index]['snr_db'])
        write_csv(export/'metrics.csv', rows)
        summary, effects = [], []
        indices_bootstrap = np.random.default_rng(settings['screen_seed']+153).integers(len(ids),
            size=(settings['bootstrap_samples'], len(ids)))
        pairs = [(name, 'sbl100') for name in outputs if name != 'sbl100' and name not in ORACLES]
        pairs += [('physical_initial_shared', 'physical_initial_per_tone'),
            ('physical_refined_shared', 'physical_refined_per_tone'),
            ('physical_refined_per_tone', 'physical_initial_per_tone'),
            ('physical_joint_shared', 'physical_refined_shared'), (ORACLES[1], ORACLES[0])]
        for index, context in enumerate(contexts):
            values = {}
            for name in outputs:
                local = [r for r in rows if r['context'] == index and r['method'] == name]
                values[name] = np.array([r['nmse_linear'] for r in local])
                summary.append({'context': index, 'snr_db': context['snr_db'], 'method': name,
                    'diagnostic_oracle': name in ORACLES, 'scenes': len(ids),
                    'mean_nmse_linear': float(values[name].mean()),
                    'mean_scene_nmse_db': float(10*np.log10(values[name].mean())),
                    'mean_physical_mse': float(np.mean([r['mse_physical'] for r in local])),
                    'mean_normalized_observation_residual': float(np.mean([r['normalized_observation_residual'] for r in local]))})
            for target, reference in pairs:
                delta = 10*np.log10(values[target][indices_bootstrap].mean(1)/values[reference][indices_bootstrap].mean(1))
                low, high = np.quantile(delta, [.025, .975])
                effects.append({'context': index, 'snr_db': context['snr_db'], 'target': target, 'reference': reference,
                    'delta_db': float(10*np.log10(values[target].mean()/values[reference].mean())),
                    'ci95_low_db': float(low), 'ci95_high_db': float(high),
                    'target_better_scenes': int((values[target] < values[reference]).sum())})
        write_csv(export/'summary.csv', summary)
        write_csv(export/'paired_effects.csv', effects)
        for name, digest in frozen['files_sha256'].items():
            assert file_hash(path/name) == digest
        observed = torch.load(path/'predictions/observations.pt', weights_only=True)
        assert observed['created_at_unix'] > frozen['frozen_at_unix']
        assert len({entry['physical_noise_sha256'] for entry in entries}) == 1
        audit = {'status': 'passed', 'run_id': run_id, 'independent_screen_geometries': len(ids),
            'new_geometry_disjoint_from_all_previous_splits': True, 'frozen_before_screen': True,
            'same_physical_observation_across_methods': True, 'same_physical_noise_across_snr': True,
            'true_geometry_only_used_in_diagnostics': True, 'performance_acceptance_gate': False,
            'metric_rows': len(rows), 'elapsed_seconds': time.time()-started}
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
