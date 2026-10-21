"""Reproduce one validation overview from the preserved research_002 CSV tables."""
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, font_manager, pyplot as plt
import numpy as np
from PIL import Image


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = Path(sys.argv[1]) if len(sys.argv) == 2 else OUT / "source_tables"
assert len(sys.argv) <= 2, "usage: python plot_results.py [source-table-directory]"
source_files = {}


def read_json(name):
    source_files[name] = {"sha256": sha(SOURCE / name), "bytes": (SOURCE / name).stat().st_size}
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


config = read_json("config_resolved.json")
selection = read_json("diffusion_selection.json")["selection"]
baseline = read_json("base_baseline/selection.json")
audit = read_json("validation_audit.json")
assert audit["status"] == "passed" and audit["maximum_metric_difference"] == 0.
assert config["system"]["antennas"] == config["system"]["tones"] == 32
contexts = list(itertools.product(config["system"]["bandwidths_hz"], config["system"]["configurations"], config["system"]["snrs_db"]))
assert len(contexts) == 8
seeds = config["research"]["training_seeds"]
count = config["data"]["validation"]
groups, observations = collections.defaultdict(list), {}
for name in ("base_baseline/metrics.csv", "reliability/metrics.csv", "diffusion/metrics.csv"):
    source_files[name] = {"sha256": sha(SOURCE / name), "bytes": (SOURCE / name).stat().st_size}
    with (SOURCE / name).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            c, scene = int(row["context"]), int(row["scene"])
            assert 0 <= c < len(contexts) and 0 <= scene < count
            key = c, scene
            assert observations.setdefault(key, row["observation_sha256"]) == row["observation_sha256"]
            if name.startswith("base_baseline"):
                method, seed = f"sbl_{row['grid']}_{row['iterations']}", None
            else:
                method = ("deterministic_" if name.startswith("reliability") else "") + row["method"]
                seed = int(row["seed"])
                assert seed in seeds
                assert (float(row["bandwidth_hz"]), int(row["configurations"]), float(row["snr_db"])) == contexts[c]
            value = float(row["nmse_linear"])
            assert np.isfinite(value) and value > 0
            groups[method, seed, c].append((scene, value))
assert len(observations) == count * len(contexts)
for rows in groups.values():
    assert len(rows) == count and len({scene for scene, _ in rows}) == count

all_values = collections.defaultdict(list)
for (method, seed, c), rows in groups.items():
    all_values[method].extend(v for _, v in rows)
overall = {m: float(np.mean(v)) for m, v in all_values.items()}
for method in overall:
    expected = {(method, seed, c) for seed in ((None,) if method.startswith("sbl_") else seeds) for c in range(len(contexts))}
    assert {key for key in groups if key[0] == method} == expected
if (SOURCE / "download_manifest.json").exists():
    downloaded = read_json("download_manifest.json")["files"]
    assert all(info["sha256"] == downloaded[name]["sha256"] for name, info in source_files.items() if name in downloaded)
db = lambda value: float(10 * np.log10(value))
coarse = f"sbl_coarse_{baseline['coarse']['iterations']}"
dense = f"sbl_dense_{baseline['dense']['iterations']}"
best = selection["selected"]
assert (best["mode"], best["steps"], best["average"]) == ("none", 40000, 16)
best_diff = f"diffusion_{best['mode']}_step{best['steps']}_avg{best['average']}"
assert np.isclose(overall[best_diff], best["mean_nmse_linear"], rtol=1e-12, atol=1e-14)
reliability = read_json("reliability_selection.json")["selection"]
for mode, expected in reliability["validation_mean_nmse_linear"].items():
    assert np.isclose(overall["deterministic_" + mode], expected, rtol=1e-12, atol=1e-14)
for item in selection["candidates"]:
    label = f"diffusion_{item['mode']}_step{item['steps']}_avg{item['average']}"
    assert np.isclose(overall[label], item["mean_nmse_linear"], rtol=1e-12, atol=1e-14)

methods = [coarse, dense, "deterministic_none", "deterministic_map", "deterministic_scalar", best_diff]
labels = ["粗字典 SBL\n49×7，1,000次", "密字典 SBL\n97×13，2,000次", "确定性精修\n无可靠度", "确定性精修\n完整方差图", "确定性精修\n标量可靠度", "扩散精修\n无可靠度，16次平均"]
short = ["粗字典 SBL", "密字典 SBL", "确定性·无可靠度", "确定性·方差图", "确定性·标量", "扩散·无可靠度×16"]
palette = ["#4B5563", "#A23A00", "#0072B2", "#7A5195", "#007F5F", "#B94B00"]
markers = ["o", "s", "o", "D", "D", "^"]


def context_mean(method, context):
    return float(np.mean([v for (m, _, c), rows in groups.items() if m == method and c == context for _, v in rows]))


summary = [{"method": m, "label": label.replace("\n", " "), "mean_nmse_linear": overall[m],
    "mean_nmse_db": db(overall[m]), "improvement_over_coarse_sbl_db": db(overall[coarse] / overall[m]),
    "relative_nmse_reduction_percent": 100 * (1 - overall[m] / overall[coarse]), "metric_rows": len(all_values[m])}
    for m, label in zip(methods, labels)]
write_csv("overall_results.csv", summary)
write_csv("all_method_seed_context_means.csv", [{"method": m, "training_seed": "" if seed is None else seed,
    "context": c, "bandwidth_ghz": contexts[c][0] / 1e9, "dma_configurations": contexts[c][1],
    "nominal_snr_db": contexts[c][2], "geometries": len(rows), "mean_nmse_linear": float(np.mean([v for _, v in rows])),
    "mean_nmse_db": db(np.mean([v for _, v in rows]))}
    for (m, seed, c), rows in groups.items()])
conditions = [{"method": m, "context": c, "bandwidth_ghz": band / 1e9, "dma_configurations": j,
    "nominal_snr_db": snr, "mean_nmse_db": db(context_mean(m, c)),
    "improvement_over_coarse_sbl_db": db(context_mean(coarse, c) / context_mean(m, c))}
    for m in methods for c, (band, j, snr) in enumerate(contexts)]
write_csv("condition_results.csv", conditions)

font = Path("C:/Windows/Fonts/msyh.ttc")
if font.exists():
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
else:
    family = "Noto Sans CJK SC"
style = {"font.family": family, "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "path", "figure.facecolor": "white"}
with plt.rc_context(style):
    fig, axes = plt.subplot_mosaic([["A", "B"], ["C", "D"]], figsize=(16.5, 11), layout="constrained",
        gridspec_kw={"width_ratios": [1., 1.28], "height_ratios": [1.1, 1.]})
    fig.suptitle("同观测宽带 DMA 信道估计｜当前验证集结果", fontsize=22, fontweight="bold")
    fig.supxlabel("32阵元 × 32子载波，300 GHz；512训练几何 / 64验证几何；8种条件；3个训练初始化。\n"
        "NMSE先在线性域平均再转dB；空心点为各初始化的均值；图中为未加末端观测修正的输出。", fontsize=11.5)
    a = axes["A"]
    for index, (method, color, marker) in enumerate(zip(methods, palette, markers)):
        value = db(overall[method])
        if not method.startswith("sbl_"):
            for offset, seed in zip((-.12, 0, .12), seeds):
                samples = [v for (m, s, _), rows in groups.items() if m == method and s == seed for _, v in rows]
                a.scatter(db(np.mean(samples)), index + offset, s=30, facecolors="none", edgecolors=color, linewidths=1.1, zorder=4)
        a.scatter(value, index, s=95, color=color, marker=marker, edgecolors="white", linewidths=.8, zorder=3)
        a.text(value + .12, index, f"{value:.3f}", ha="left", va="center", fontsize=12, color="#202124")
    a.axvline(db(overall[coarse]), color=palette[0], linestyle=":", linewidth=1.2, zorder=0)
    a.set(yticks=np.arange(len(methods)), yticklabels=labels, xlabel="完整 CSI 的 NMSE / dB（越低越好）",
        xlim=(-11., -7.35), ylim=(len(methods) - .5, -.5), xticks=(-11, -10, -9, -8))
    a.set_title("A  总体精度：确定性精修最好", loc="left", pad=13)
    a.grid(axis="x", alpha=.2)

    b = axes["B"]
    gains = np.array([[db(context_mean(coarse, c) / context_mean(m, c)) for c in range(8)] for m in methods[1:]])
    limit = max(1., np.ceil(abs(gains).max() * 2) / 2)
    norm = colors.Normalize(-limit, limit)
    cmap = plt.get_cmap("RdBu")
    heat = b.pcolormesh(np.arange(9), np.arange(6), gains, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.)
    for row in range(5):
        for col in range(8):
            rgb = np.array(cmap(norm(gains[row, col]))[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            ink = "black" if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else "white"
            b.text(col + .5, row + .5, f"{gains[row, col]:+.2f}", ha="center", va="center", color=ink, fontsize=11.5)
    b.axvline(4, color="#374151", linewidth=2)
    b.set(xticks=np.arange(8) + .5, xticklabels=[f"J={j}\n{snr:g} dB" for _, j, snr in contexts],
        yticks=np.arange(5) + .5, yticklabels=short[1:], ylim=(5, -.45), xlim=(0, 8))
    b.text(2, -.2, "带宽 3 GHz", ha="center", va="center", fontweight="bold")
    b.text(6, -.2, "带宽 30 GHz", ha="center", va="center", fontweight="bold")
    b.set_title("B  各条件相对粗字典 SBL 的改善", loc="left", pad=13)
    b.tick_params(length=0)
    for spine in b.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=b, orientation="horizontal", fraction=.055, pad=.025, aspect=35)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("NMSE 改善 / dB：正值更好，负值更差", fontsize=11.5)

    modes = [("none", "无可靠度", "#B94B00", "o", "-"), ("scalar", "标量可靠度", "#7A5195", "s", "--")]
    for key in ("C", "D"):
        axis = axes[key]
        axis.axhline(db(overall[coarse]), color=palette[0], linestyle=":", linewidth=1.3, label="粗字典 SBL")
        axis.axhline(db(overall["deterministic_scalar"]), color=palette[4], linestyle="-.", linewidth=1.5, label="确定性·标量")
        axis.set(ylim=(-10.65, -7.25), ylabel="完整 CSI 的 NMSE / dB（越低越好）")
        axis.grid(alpha=.18)
    c = axes["C"]
    for mode, label, color, marker, line in modes:
        values = [db(overall[f"diffusion_{mode}_step40000_avg{draws}"]) for draws in (1, 4, 16)]
        c.plot(range(3), values, color=color, marker=marker, linestyle=line, linewidth=2, markersize=7, label="扩散·" + label)
    c.set(xticks=range(3), xticklabels=["1次\n20次网络调用", "4次\n80次网络调用", "16次\n320次网络调用"],
        xlim=(-.12, 2.15), xlabel="生成输出的平均次数（均为40,000步模型）")
    c.set_title("C  平均更多生成输出可降低扩散误差", loc="left", pad=13)
    c.legend(loc="upper right", fontsize=11, framealpha=.95)
    d = axes["D"]
    steps = config["research"]["diffusion_checkpoints"]
    d.axvspan(20, 40, color="#F0F1F3", zorder=0)
    d.axvline(20, color="#9CA3AF", linewidth=1)
    for mode, label, color, marker, line in modes:
        values = [db(overall[f"diffusion_{mode}_step{step}_avg16"]) for step in steps]
        d.plot(np.array(steps) / 1000, values, color=color, marker=marker, linestyle=line, linewidth=2, markersize=7, label="扩散·" + label)
    d.set(xticks=(2, 8, 20, 30, 40), xlim=(0, 42), xlabel="扩散训练更新数 / 千步；每个检查点平均16次输出")
    gain = selection["last_interval_improvement_db"]
    gap = db(overall[best_diff] / overall["deterministic_scalar"])
    d.text(.43, .91, f"30k→40k 改善 {gain:.4f} dB\n扩散仍比确定性差 {gap:.3f} dB",
        transform=d.transAxes, va="top", fontsize=12, bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "pad": 6})
    d.text(.74, .14, "灰区：原日程末端学习率 1e-4", transform=d.transAxes, ha="center", fontsize=10.5, color="#4B5563")
    d.set_title("D  续训后改善已很小", loc="left", pad=13)
    d.legend(loc="upper left", fontsize=10.5, framealpha=.95)
    fig.canvas.draw()
    renderer, bounds = fig.canvas.get_renderer(), fig.bbox
    clipped = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0 - 2 or box.x1 > bounds.x1 + 2 or box.y0 < bounds.y0 - 2 or box.y1 > bounds.y1 + 2:
                clipped.append(text.get_text())
    assert not clipped, clipped
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"validation_overview.{extension}", dpi=180, facecolor="white", transparent=False)
    plt.close(fig)

with Image.open(OUT / "validation_overview.png") as image:
    assert image.size == (2970, 1980)
    if image.mode == "RGBA":
        assert image.getchannel("A").getextrema() == (255, 255)
    image_metadata = {"width_pixels": image.width, "height_pixels": image.height, "dpi": image.info.get("dpi"), "mode": image.mode}
for name in ("config_resolved.json", "validation_audit.json"):
    shutil.copy2(SOURCE / name, OUT / name)
shutil.copy2(SOURCE / "environment.json", OUT / "training_environment.json")
write_json("source_data_manifest.json", {"run_id": "research_002", "source_directory": str(SOURCE.resolve()),
    "remote_source_directory": "/home/lsy/DMA_CE/results/research_002", "files": source_files})
write_json("figure_provenance.json", {"scope": "validation_only", "source_run": "research_002", "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "independent_validation_geometries": count,
    "training_geometries": config["data"]["train"], "contexts_per_geometry": len(contexts), "training_seeds": seeds,
    "common_observation_hashes_verified": len(observations), "mean": "10*log10(mean(per-sample linear NMSE)); seeds and contexts equally weighted",
    "hollow_points": "one linear-mean NMSE per training initialization; no confidence interval",
    "condition_gain": "10*log10(coarse-SBL mean NMSE / method mean NMSE) within each condition",
    "diffusion_selection_recomputed_from_csv": True, "smoothing": None, "uncertainty_intervals": None,
    "end_observation_consistency": "not applied in these validation tables", "final_screening_data_used": False,
    "plot_script_sha256": sha(Path(__file__)), "python": platform.python_version(), "matplotlib": matplotlib.__version__,
    "numpy": np.__version__, "font": family, "image": image_metadata,
    "outputs": {f"validation_overview.{ext}": sha(OUT / f"validation_overview.{ext}") for ext in ("png", "pdf", "svg")}})
print(json.dumps({"status": "passed", "output": str(OUT), "observation_pairs": len(observations),
    "summary": summary, "heatmap_range_db": [-float(limit), float(limit)]}, ensure_ascii=False, indent=2))
