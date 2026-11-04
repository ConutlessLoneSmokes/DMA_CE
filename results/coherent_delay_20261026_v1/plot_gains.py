"""Show the paired differences that are small on the full SNR curves."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

out = Path(__file__).resolve().parent
rows = list(csv.DictReader((out / "paired_effects.csv").open()))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for candidate, reference, label, color in (
        ("coherent_selected", "free", "Selected coherent (rho=0.95) vs free", "tab:orange"),
        ("coherent_half", "free", "Coherent (rho=0.5) vs free", "tab:green"),
        ("coherent_half", "shuffled_half", "Original vs shuffled delay pairing (rho=0.5)", "tab:purple")):
    selected = [row for row in rows if row["cohort"] == "new" and row["candidate"] == candidate and row["reference"] == reference]
    x, y = [float(row["snr_db"]) for row in selected], [float(row["improvement_db"]) for row in selected]
    errors = [[value - float(row["ci95_low_db"]) for value, row in zip(y, selected)],
        [float(row["ci95_high_db"]) - value for value, row in zip(y, selected)]]
    ax.errorbar(x, y, yerr=errors, fmt="o-", capsize=4, color=color, label=label)
ax.axhline(0, color="black", linewidth=.8)
ax.set(xlabel="Training-average received SNR (dB)", ylabel="Improvement in mean scene full-CSI NMSE (dB)",
    title="32 new independent geometries; paired scene bootstrap 95% intervals")
ax.set_xticks([-10, 0, 10, 20, 30])
ax.grid(alpha=.25)
ax.legend(fontsize=8.5)
fig.tight_layout()
for suffix in ("png", "svg"):
    fig.savefig(out / ("coherent_delay_gains." + suffix), dpi=180)
plt.close(fig)
