"""Independent Gram projections, CSV identities and paired effect plots."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "runs" / OUT.name
assert json.loads((RUN / "status.json").read_text())["status"] == "complete"
settings = json.loads((RUN / "config_resolved.json").read_text())["mechanism"]
PARENT = ROOT / "runs" / settings["parent_run"]
torch.set_num_threads(4)
rows = list(csv.DictReader((OUT / "decomposition.csv").open()))
groups = {}
for row in rows:
    key = tuple(row[k] for k in ("cohort", "snr_db", "candidate", "reference", "region"))
    groups.setdefault(key, []).append(row)
for key, values in groups.items():
    assert len(values) == (32 if key[0] == "new" else 16)
    for r in values:
        assert abs(float(r["after"]) - float(r["before"]) - float(r["correction"]) + float(r["alignment"])) < 1e-10
    if key[-1] == "full":
        for region in ("row", "null"):
            assert [r["geometry_id"] for r in values] == [r["geometry_id"] for r in groups[(*key[:-1], region)]]
        for i, r in enumerate(values):
            for column in ("before", "after", "correction", "alignment", "gain"):
                assert abs(float(r[column]) - sum(float(groups[(*key[:-1], p)][i][column]) for p in ("row", "null"))) < 1e-10

cache = torch.load(PARENT / "inputs/physical_parent.pt", map_location="cpu", weights_only=True)
a = cache["contexts"][0]["matrix"].numpy().astype(complex)
del cache
observed = torch.load(PARENT / "predictions/new_observations.pt", map_location="cpu", weights_only=True)
checks = []
for context in (0, 4):
    saved = torch.load(PARENT / f"predictions/new_{context}.pt", map_location="cpu", weights_only=True)
    truth = observed["entries"][context]["truth"][0].numpy().astype(complex)
    r = truth - saved["outputs"]["free"][0].numpy().astype(complex)
    d = saved["outputs"]["coherent_selected"][0].numpy().astype(complex) - saved["outputs"]["free"][0].numpy().astype(complex)
    gram = a @ a.conj().transpose(0, 2, 1)
    def project(v):
        return (a.conj().transpose(0, 2, 1) @ np.linalg.solve(gram, (a @ v[..., None])))[..., 0]
    er, dr = project(r), project(d)
    energy = np.sum(abs(truth)**2)
    for region, error, change in (("row", er, dr), ("null", r-er, d-dr)):
        key = ("new", str([-10, 0, 10, 20, 30][context]), "coherent_selected", "free", region)
        recorded = groups[key][0]
        values = {"before": np.sum(abs(error)**2)/energy,
            "after": np.sum(abs(error-change)**2)/energy,
            "correction": np.sum(abs(change)**2)/energy,
            "alignment": 2*np.vdot(error, change).real/energy}
        difference = max(abs(v-float(recorded[k])) for k,v in values.items())
        assert difference < 1e-10
        checks.append({"context": context, "scene": 0, "region": region,
            "maximum_absolute_Gram_vs_SVD_difference": difference})

effects = []
draws = {c: np.random.default_rng(settings["bootstrap_seed"]+i).integers(0,n,(settings["bootstrap_samples"],n))
    for i,(c,n) in enumerate((("new",32),("historical",16)))}
for key, values in groups.items():
    if key[-1] != "full":
        continue
    base, final = (np.array([float(r[c]) for r in values]) for c in ("before", "after"))
    low, high = np.quantile(10*np.log10(base[draws[key[0]]].mean(1)/final[draws[key[0]]].mean(1)),[.025,.975])
    effects.append({"cohort": key[0], "snr_db": float(key[1]), "candidate": key[2], "reference": key[3],
        "reference_nmse_db": 10*np.log10(base.mean()), "candidate_nmse_db": 10*np.log10(final.mean()),
        "improvement_db": 10*np.log10(base.mean()/final.mean()),
        "ci95_low_db": low, "ci95_high_db": high,
        "row_gain_linear": np.mean([float(r["gain"]) for r in groups[(*key[:-1], "row")]]),
        "null_gain_linear": np.mean([float(r["gain"]) for r in groups[(*key[:-1], "null")]])})
with (OUT / "paired_effects.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
    writer.writeheader()
    writer.writerows(effects)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
snrs = [-10, 0, 10, 20, 30]
percent = {}
for region in ("row", "null"):
    percent[region] = []
    for snr in snrs:
        p = ("new", str(snr), "coherent_selected", "free")
        baseline = np.mean([float(r["before"]) for r in groups[(*p,"full")]])
        gain = np.mean([float(r["gain"]) for r in groups[(*p,region)]])
        percent[region].append(100*gain/baseline)
axes[0].bar(range(5), percent["row"], label="Row-space gain")
axes[0].bar(range(5), percent["null"], bottom=percent["row"], label="Null-space gain")
axes[0].set_xticks(range(5),snrs)
axes[0].set_xlabel("Training-average received SNR (dB)")
axes[0].set_ylabel("Reduction / reference full-CSI NMSE (%)")
axes[0].set_title("Selected coherent update vs free posterior")
axes[0].legend()
for snr in snrs:
    values = []
    for name in ("coherent_half", "offset_0.25_bin", "offset_0.5_bin", "offset_1_bin"):
        effect = next(r for r in effects if r["cohort"] == "new" and r["snr_db"] == snr and r["candidate"] == name and r["reference"] == "free")
        values.append(effect["improvement_db"])
    axes[1].plot([0,.25,.5,1], values,"o-", label=f"{snr} dB")
axes[1].axhline(0,color="black",linewidth=.8)
axes[1].set_xlabel("Injected delay offset, units of 1/(K delta-f)")
axes[1].set_ylabel("Improvement vs free posterior (dB)")
axes[1].set_title("Fixed correlation strength = 0.5")
axes[1].legend(ncol=2, fontsize=8)
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle("32 reused geometries; paired post-hoc mechanism diagnosis")
fig.tight_layout()
for extension in ("png","svg"):
    fig.savefig(OUT / f"mechanism.{extension}",dpi=170)
plt.close(fig)
(OUT / "independent_verification.json").write_text(json.dumps({"status":"passed",
    "checked_scene_region_rows":len(rows),"Gram_projection_checks":checks,
    "scope":"CSV identities, independent Gram projection on two saved scenes; no selection"},indent=2)+"\n")
print(json.dumps({"rows":len(rows),"independent_checks":checks}))
