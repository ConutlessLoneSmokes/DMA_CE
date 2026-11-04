"""Plot the frozen five-SNR experiment; saved CSV files are the only inputs."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1]).resolve()
with (out/'summary.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with (out/'paired_effects.csv').open() as stream:
    effects = list(csv.DictReader(stream))
snrs = sorted({float(row['snr_db']) for row in rows})
assert snrs == [-10, 0, 10, 20, 30]
colors = ['#222222', '#0072B2', '#E69F00', '#D55E00', '#009E73', '#CC79A7', '#56B4E9']
methods = [('sbl100', 'SBL, 100 iterations'), ('covariance_sbl', 'Covariance SBL'),
    ('physical_initial_per_tone', 'Initial geometry, per-tone gains'),
    ('physical_initial_shared', 'Initial geometry, shared gains'),
    ('physical_refined_per_tone', 'Refined geometry, per-tone gains'),
    ('physical_refined_shared', 'Refined geometry, shared gains'),
    ('physical_joint_shared', 'Joint refinement, shared gains')]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for index, (method, label) in enumerate(methods):
    values = [float(next(r['mean_scene_nmse_db'] for r in rows if r['method'] == method and float(r['snr_db']) == snr)) for snr in snrs]
    axes[0, 0].plot(snrs, values, 'o-' if index % 2 == 0 else 's--', color=colors[index], label=label, ms=4)
axes[0, 0].set(title='Actual estimators: full CSI', ylabel='Mean scene NMSE (dB, lower is better)')
axes[0, 0].legend(fontsize=8)
for index, (target, reference, label) in enumerate([
        ('covariance_sbl', 'sbl100', 'Covariance SBL vs SBL'),
        ('physical_refined_per_tone', 'physical_initial_per_tone', 'Geometry refinement, per-tone gains'),
        ('physical_refined_shared', 'physical_refined_per_tone', 'Shared vs per-tone, same refined geometry'),
        ('physical_joint_shared', 'physical_refined_shared', 'Joint vs fixed refined geometry')]):
    values = [next(r for r in effects if r['target'] == target and r['reference'] == reference and float(r['snr_db']) == snr) for snr in snrs]
    delta = np.array([float(r['delta_db']) for r in values])
    error = np.array([[float(r['ci95_low_db']), float(r['ci95_high_db'])] for r in values]).T
    axes[0, 1].errorbar(np.array(snrs)+(index-1.5)*.45, delta,
        yerr=np.maximum(np.stack((delta-error[0], error[1]-delta)), 0),
        fmt='o-', color=colors[index+1], label=label, ms=4, capsize=2)
axes[0, 1].axhline(0, color='#666666', lw=.8)
axes[0, 1].set(title='Matched changes and paired 95% intervals', ylabel='Target minus reference NMSE (dB)')
axes[0, 1].legend(fontsize=8)
for index, (method, label) in enumerate([('sbl100', 'SBL, 100 iterations'),
        ('physical_joint_shared', 'Estimated geometry, shared gains'),
        ('oracle_geometry_per_tone', 'True geometry, per-tone gains (diagnostic)'),
        ('oracle_geometry_shared', 'True geometry, shared gains (diagnostic)')]):
    values = [float(next(r['mean_scene_nmse_db'] for r in rows if r['method'] == method and float(r['snr_db']) == snr)) for snr in snrs]
    axes[1, 0].plot(snrs, values, 'o--' if index >= 2 else 'o-', color=colors[index], label=label, ms=4)
axes[1, 0].set(title='True geometry provides extra information', ylabel='Mean scene NMSE (dB, lower is better)')
axes[1, 0].legend(fontsize=8)
for index, (method, label) in enumerate(methods):
    if index in (2, 3):
        continue
    values = [float(next(r['mean_normalized_observation_residual'] for r in rows if r['method'] == method and float(r['snr_db']) == snr)) for snr in snrs]
    axes[1, 1].plot(snrs, values, 'o-', color=colors[index], label=label, ms=4)
axes[1, 1].set_yscale('log')
axes[1, 1].set(title='Observation fit: distinct from full CSI error', ylabel='Residual power / known noise variance')
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.set_xlabel('Training-average received SNR (dB)')
    ax.set_xticks(snrs)
    ax.grid(alpha=.2)
fig.suptitle('Structured priors | 32 new geometries | fixed training set and DMA | exploratory', fontsize=13)
for suffix in ('png', 'svg'):
    fig.savefig(out/f'structured_priors.{suffix}', dpi=220)
plt.close(fig)
