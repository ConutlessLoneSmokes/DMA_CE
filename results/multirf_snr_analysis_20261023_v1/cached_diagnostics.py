"""Read existing server tensors on CPU; no estimation, training or selection."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


root = Path('/home/lsy/DMA_CE')
run = root / 'runs/multirf_snr_20261022_v1'
export = root / 'results/multirf_snr_20261022_v1'
torch.set_num_threads(4)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def errors(truth, prediction):
    return (prediction - truth).abs().square().sum((1, 2)) / truth.abs().square().sum((1, 2))


frozen = json.loads((run / 'frozen.json').read_text())
for name, expected in frozen['files_sha256'].items():
    assert sha(run / name) == expected, name
manifest = json.loads((run / 'source_manifest.json').read_text())
with zipfile.ZipFile(run / 'source.zip') as zipped:
    assert set(zipped.namelist()) == set(manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256(zipped.read(name)).hexdigest() == expected, name
config = json.loads((run / 'config_resolved.json').read_text())
cache = load(run / 'cache/physical.pt')
observations = load(run / 'predictions/observations.pt')
groups = [set(cache['geometry_ids'][split]) for split in ('train', 'validation')]
groups.append(set(observations['geometry_ids']))
assert [len(g) for g in groups] == [config['data'][s] for s in ('train', 'validation', 'screen')]
assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
assert observations['frozen_sha256'] == sha(run / 'frozen.json')
assert observations['created_at_unix'] > frozen['frozen_at_unix']
assert len({e['physical_noise_sha256'] for e in observations['entries']}) == 1
matrices = [c['matrix'] for c in cache['contexts']]
assert all(torch.equal(a, matrices[0]) for a in matrices)
_, singular, vh = torch.linalg.svd(matrices[0].to(torch.complex128), full_matrices=False)
threshold = torch.finfo(torch.float64).eps * max(matrices[0].shape[-2:]) * singular[:, :1]
assert (singular > threshold).all(), 'row-space computation requires full row rank'

with (export / 'metrics.csv').open() as stream:
    metrics = {(int(r['context']), r['method'], int(r['scene'])): r for r in csv.DictReader(stream)}
rows, calibration, max_difference = [], [], 0.
for index, (context, entry) in enumerate(zip(cache['contexts'], observations['entries'])):
    saved = load(run / f'predictions/context_{index}.pt')
    assert saved['frozen_sha256'] == sha(run / 'frozen.json')
    observation_bytes = entry['observation'].contiguous().numpy().tobytes()
    assert saved['observation_sha256'] == hashlib.sha256(observation_bytes).hexdigest()
    truth = entry['truth'].to(torch.complex128)
    energy = truth.abs().square().sum((1, 2))
    coarse = saved['ddpm_coarse'].to(torch.complex128)
    generated = saved['outputs']['ddpm'].to(torch.complex128)
    before, change = coarse - truth, generated - coarse
    change_energy = change.abs().square().sum((1, 2))
    cross = 2 * (before.conj() * change).real.sum((1, 2)) / energy
    cosine = -(before.conj() * change).real.sum((1, 2)) / (
        before.abs().square().sum((1, 2)) * change_energy).sqrt().clamp_min(1e-30)
    coarse_error = errors(truth, coarse)
    generated_error = errors(truth, generated)
    assert torch.allclose(generated_error, coarse_error + change_energy / energy + cross, atol=1e-12, rtol=1e-9)
    coordinates = torch.einsum('krn,bkn->bkr', vh, before)
    row_error = coordinates.abs().square().sum((1, 2)) / energy
    null_error = coarse_error - row_error
    assert (null_error >= -1e-12).all()
    for method, prediction in saved['outputs'].items():
        exact = errors(truth, prediction.to(torch.complex128))
        for scene, value in enumerate(exact.tolist()):
            recorded = float(metrics[index, method, scene]['nmse_linear'])
            max_difference = max(max_difference, abs(value - recorded) / value)
    for scene in range(len(truth)):
        rows.append({'context': index, 'snr_db': context['snr_db'], 'scene': scene,
            'geometry_id': observations['geometry_ids'][scene], 'coarse_nmse_linear': float(coarse_error[scene]),
            'coarse_row_nmse_linear': float(row_error[scene]), 'coarse_null_nmse_linear': float(null_error[scene]),
            'ddpm_nmse_linear': float(generated_error[scene]),
            'correction_nmse_linear': float(change_energy[scene] / energy[scene]),
            'error_correction_cross': float(cross[scene]), 'correction_cosine': float(cosine[scene])})
    for split in ('train', 'validation'):
        item = load(run / f'cache/coarse_{split}_{index}.pt')
        target = item['truth'].to(torch.complex128)
        prediction = item['coarse'].to(torch.complex128)
        calibration.append({'context': index, 'snr_db': context['snr_db'], 'split': split,
            'mean_nmse_linear': float(errors(target, prediction).mean()),
            'residual_complex_mse': float((target - prediction).abs().square().mean()),
            'mean_coarse_ms_per_scene': item['coarse_ms_per_sample']})
    del saved
assert max_difference < 1e-5
training_error = np.mean([r['residual_complex_mse'] for r in calibration if r['split'] == 'train'])
assert abs((training_error / 2) ** .5 - frozen['ddpm']['residual_real_scale']) < 1e-6
events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
result = {'run_id': run.name, 'mode': 'existing tensor CPU diagnostics; no solver or model inference',
    'result_files_sha256': {p.name: sha(p) for p in sorted(export.iterdir()) if p.is_file()},
    'source_archive_files_verified': len(manifest), 'frozen_files_verified': len(frozen['files_sha256']),
    'split_sizes': [len(g) for g in groups], 'frozen_before_screen': True,
    'row_rank_per_tone': [len(s) for s in singular], 'maximum_relative_nmse_difference': max_difference,
    'training_pooled_residual_complex_mse': float(training_error), 'calibration': calibration,
    'screen': rows, 'events_sha256': sha(run / 'events.jsonl'),
    'phase_events': [e for e in events if e['event'] in ('lmmse_validation', 'ddpm_coarse', 'validation', 'frozen', 'screen_context')],
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__}}
json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False)
