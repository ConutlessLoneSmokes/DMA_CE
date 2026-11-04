"""Recompute exported results and paired geometry intervals; run from repository root."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


root = Path.cwd()
folder = Path(__file__).resolve().parent
source = root / 'results/multirf_snr_20261022_v1'
diagnostics = json.loads((folder / 'cached_diagnostics.json').read_text(encoding='utf-8-sig'))
config = json.loads((source / 'config_resolved.json').read_text())
frozen = json.loads((source / 'frozen.json').read_text())
for name, expected in diagnostics['result_files_sha256'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
manifest = json.loads((source / 'source_manifest.json').read_text())
execution = {name: digest for name, digest in manifest.items()
    if name.startswith(('src/', 'scripts/', 'tests/')) and name.endswith('.py')}
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in execution.items())
assert json.loads((root / 'configs/multirf_snr.json').read_text()) == config


def read(name):
    with (source / name).open() as stream:
        return list(csv.DictReader(stream))


def db(value):
    return float(10 * np.log10(value))


rows, summary = read('metrics.csv'), read('summary.csv')
methods = ('ls_min_norm', 'lmmse_selected', 'sbl_dense', 'ddpm')
levels = config['system']['snrs_db']
count = config['data']['screen']
assert len(rows) == len(levels) * len(methods) * count == 4608
assert len(summary) == len(levels) * len(methods)
groups = {(snr, method): sorted((r for r in rows if float(r['snr_db']) == snr and r['method'] == method),
    key=lambda r: int(r['scene'])) for snr in levels for method in methods}
original_indices = np.random.default_rng(config['seed'] + 819).integers(count,
    size=(config['bootstrap_samples'], count))
indices = np.random.default_rng(20261023).integers(count, size=(10000, count))
maximum_summary_difference, maximum_decomposition_difference = 0., 0.
reference_ids = [r['geometry_id'] for r in groups[levels[0], methods[0]]]
assert len(set(reference_ids)) == count
for snr in levels:
    for scene in range(count):
        same = [groups[snr, method][scene] for method in methods]
        assert len({r['observation_sha256'] for r in same}) == 1
    for method in methods:
        selected = groups[snr, method]
        assert [int(r['scene']) for r in selected] == list(range(count))
        assert [r['geometry_id'] for r in selected] == reference_ids
        values = np.array([float(r['nmse_linear']) for r in selected])
        assert np.isfinite(values).all() and (values > 0).all()
        reported = next(r for r in summary if float(r['snr_db']) == snr and r['method'] == method)
        low, high = np.quantile(10 * np.log10(values[original_indices].mean(1)), [.025, .975])
        maximum_summary_difference = max(maximum_summary_difference,
            abs(db(values.mean()) - float(reported['mean_scene_nmse_db'])),
            abs(low - float(reported['nmse_db_ci95_low'])), abs(high - float(reported['nmse_db_ci95_high'])))
        decomposed = np.array([float(r['row_nmse_linear']) + float(r['null_nmse_linear']) for r in selected])
        maximum_decomposition_difference = max(maximum_decomposition_difference, float(np.max(abs(decomposed-values)/values)))
assert maximum_summary_difference < 1e-9
assert maximum_decomposition_difference < 1e-4

lmmse_validation, correction_validation = read('lmmse_validation.csv'), read('ddpm_correction_validation.csv')
choice = min(config['frequency_shrinkage_grid'], key=lambda strength: (
    np.mean([float(r['mean_nmse_linear']) for r in lmmse_validation if float(r['frequency_shrinkage']) == strength]), -strength))
assert choice == frozen['frequency_shrinkage']
assert min(frozen['ddpm']['checkpoint_scores'], key=lambda r: (r['score'], r['step'])) == frozen['ddpm']['selected_checkpoint']
correction = min(config['ddpm']['dc_multipliers'], key=lambda multiplier: (
    np.mean([float(r['mean_nmse_linear']) for r in correction_validation if float(r['multiplier']) == multiplier]), multiplier))
assert correction == frozen['ddpm']['correction_multiplier'] == 0
assert all(min((r for r in correction_validation if int(r['context']) == i),
    key=lambda r: float(r['mean_nmse_linear']))['multiplier'] == '0.0' for i in range(len(levels)))

records = []
for snr in levels:
    values = {method: np.array([float(r['nmse_linear']) for r in groups[snr, method]]) for method in methods}
    cached = sorted((r for r in diagnostics['screen'] if r['snr_db'] == snr), key=lambda r: r['scene'])
    assert [r['geometry_id'] for r in cached] == reference_ids
    coarse = np.array([r['coarse_nmse_linear'] for r in cached])
    selected = groups[snr, 'sbl_dense']
    converged = np.array([r['sbl_converged'] == 'True' for r in selected])
    row = {'snr_db': snr, **{f'{method}_nmse_db': db(value.mean()) for method, value in values.items()},
        'coarse100_nmse_db': db(coarse.mean())}
    for name, baseline in (('sbl', values['sbl_dense']), ('coarse100', coarse)):
        draw = 10 * np.log10(values['ddpm'][indices].mean(1) / baseline[indices].mean(1))
        low, high = np.quantile(draw, [.025, .975])
        row.update({f'ddpm_minus_{name}_db': db(values['ddpm'].mean() / baseline.mean()),
            f'ddpm_minus_{name}_ci95_low': float(low), f'ddpm_minus_{name}_ci95_high': float(high),
            f'ddpm_better_than_{name}_scenes': int((values['ddpm'] < baseline).sum())})
    row.update(coarse100_minus_sbl_db=db(coarse.mean()/values['sbl_dense'].mean()),
        sbl_nonconverged=int((~converged).sum()),
        sbl_iterations_median=float(np.median([int(r['sbl_iterations']) for r in selected])),
        sbl_iterations_max=max(int(r['sbl_iterations']) for r in selected),
        sbl_kkt_max=max(float(r['sbl_optimality_residual']) for r in selected),
        converged_only_ddpm_minus_sbl_db=db(values['ddpm'][converged].mean()/values['sbl_dense'][converged].mean()),
        correction_energy_mean=float(np.mean([r['correction_nmse_linear'] for r in cached])),
        error_correction_cross_mean=float(np.mean([r['error_correction_cross'] for r in cached])),
        correction_cosine_mean=float(np.mean([r['correction_cosine'] for r in cached])),
        coarse100_row_nmse_mean=float(np.mean([r['coarse_row_nmse_linear'] for r in cached])),
        coarse100_null_nmse_mean=float(np.mean([r['coarse_null_nmse_linear'] for r in cached])),
        ls_null_nmse_mean=float(np.mean([float(r['null_nmse_linear']) for r in groups[snr, 'ls_min_norm']])),
        realized_snr_min=min(float(r['realized_snr_db']) for r in selected),
        realized_snr_median=float(np.median([float(r['realized_snr_db']) for r in selected])),
        realized_snr_max=max(float(r['realized_snr_db']) for r in selected))
    for method in ('sbl_dense', 'ddpm'):
        for part in ('row', 'null'):
            row[f'{method}_{part}_nmse_mean'] = float(np.mean([float(r[f'{part}_nmse_linear']) for r in groups[snr, method]]))
    records.append(row)

with (folder / 'diagnostics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
audit = {'raw_exports_byte_identical_to_server': len(diagnostics['result_files_sha256']),
    'source_files_matching_archive': len(execution), 'resolved_configuration_matches': True,
    'metrics_rows': len(rows), 'independent_geometries': count, 'same_observation_across_methods': True,
    'maximum_summary_or_ci_difference_db': maximum_summary_difference,
    'maximum_relative_row_null_decomposition_difference': maximum_decomposition_difference,
    'maximum_relative_tensor_nmse_difference': diagnostics['maximum_relative_nmse_difference'],
    'validation_choices_recomputed': True, 'paired_bootstrap': {'seed': 20261023, 'draws': len(indices),
        'unit': 'geometry; same resampled geometry indices for every SNR and method',
        'interval': 'pointwise 95%; conditional on fixed training, hardware and network; no multiplicity adjustment'},
    'cached_diagnostics_sha256': hashlib.sha256((folder/'cached_diagnostics.json').read_bytes()).hexdigest(),
    'runtime': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
(folder/'analysis_summary.json').write_text(json.dumps({'audit': audit, 'records': records}, indent=2, allow_nan=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
styles = (('ls_min_norm', 'LS', '#888888', '-'), ('lmmse_selected', 'Selected LMMSE', '#d68b00', '-'),
    ('sbl_dense', 'SBL to stopping rule', '#2271b2', '-'), ('coarse100', 'SBL 100 (DDPM input)', '#009e73', '--'),
    ('ddpm', 'DDPM, 16 draws', '#cc79a7', '-'))
for method, label, color, style in styles:
    axes[0].plot(levels, [r[f'{method}_nmse_db'] for r in records], marker='o', ms=4,
        color=color, ls=style, label=label)
for name, label, color, marker in (('sbl', 'DDPM minus SBL to stopping rule', '#2271b2', 'o'),
        ('coarse100', 'DDPM minus its SBL 100 input', '#cc79a7', 's')):
    y = np.array([r[f'ddpm_minus_{name}_db'] for r in records])
    low = np.array([r[f'ddpm_minus_{name}_ci95_low'] for r in records])
    high = np.array([r[f'ddpm_minus_{name}_ci95_high'] for r in records])
    axes[1].errorbar(levels, y, yerr=[y-low, high-y], color=color, marker=marker, ms=4, capsize=3, label=label)
axes[0].set(ylabel='Mean scene NMSE (dB)', title='The missing input baseline')
axes[1].axhline(0, color='#666666', ls=':', lw=1)
axes[1].set(ylabel='Difference in mean scene NMSE (dB)', title='Positive: DDPM is worse')
for ax in axes:
    ax.set_xlabel('Training-average received SNR per measurement (dB)')
    ax.set_xticks(levels)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='upper right' if ax is axes[1] else 'lower left')
fig.suptitle('4 x 256 DMA, 4 RF, 30 GHz, J=32 | 128 paired geometries | one trained model', fontsize=11)
for suffix in ('png', 'svg'):
    fig.savefig(folder/f'coarse_and_ddpm.{suffix}', dpi=190)
plt.close(fig)
print(json.dumps(audit, indent=2))
