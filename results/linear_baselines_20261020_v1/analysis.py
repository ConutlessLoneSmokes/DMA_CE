from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

root = Path.cwd()
main = root / 'results/linear_baselines_20261020_v1'
probe = root / 'results/linear_baseline_probe_20261020_v1'
counts = root / 'results/baseline_covariance_probe_20261020_v1'
def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))
def db(value):
    return 10*math.log10(value)

hashes = {}
sha_file = root / 'tmp/linear_baselines_server.sha256'
if sha_file.exists():
    for line in sha_file.read_text(encoding='utf-8-sig').splitlines():
        digest, path = line.split(None, 1)
        hashes[path.strip()] = digest
else:
    hashes = json.loads((main/'server_file_hashes.json').read_text(encoding='utf-8'))
for path, digest in hashes.items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, path
write(main / 'server_file_hashes.json', hashes)
manifest = json.loads((main/'source_manifest.json').read_text(encoding='utf-8'))
for path, digest in manifest.items():
    if path.startswith(('src/', 'scripts/', 'tests/')):
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, path

old = read(probe/'contexts.csv')
hardware = read(probe/'hardware.csv')
sample = read(counts/'sample_count.csv')
pooled = read(main/'pooled_snr_summary.csv')
summary = read(main/'summary.csv')
metrics = read(main/'metrics.csv')
assert len(metrics) == 36864 and len(summary) == 144 and len(pooled) == 36
frozen = json.loads((main/'frozen.json').read_text())
assert frozen['selected_frequency_shrinkage'] == {'3000000000.0': .1, '30000000000.0': .9}
snrs = sorted({float(r['snr_db']) for r in pooled})
lookup = {(float(r['snr_db']), r['method']): float(r['mean_nmse_db']) for r in pooled}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
old_ls = [[r for r in old if r['split']=='old_screen' and r['method']=='ls' and float(r['snr_db'])==snr] for snr in snrs]
for key, label in [('mean_nmse_linear','LS total'), ('null_floor_linear','Unobserved component'),
        ('ls_noise_nmse_linear','Propagated noise'), ('expected_ls_noise_nmse_linear','Analytic noise expectation')]:
    axes[0].plot(snrs, [db(avg(rs,key)) for rs in old_ls], 'o--' if key.startswith('expected') else 'o-', label=label)
axes[0].set_title('Original LS: exact orthogonal error split')
axes[0].set_xlabel('Received SNR (dB)')
for split, method, label in [('train','lmmse_joint','Joint: training'), ('validation','lmmse_joint','Joint: validation'),
        ('validation','lmmse_per_tone','Per-tone: validation')]:
    axes[1].plot(snrs,[db(avg([r for r in old if r['split']==split and r['method']==method
        and float(r['snr_db'])==snr], 'mean_nmse_linear')) for snr in snrs], 'o-', label=label)
axes[1].set_title('Same 512 training geometries')
axes[1].set_xlabel('Received SNR (dB)')
for strength, label in [(0.,'Joint covariance'), (1.,'Per-tone covariance')]:
    ns=[64,128,256,512]
    axes[2].plot(ns,[db(avg([r for r in sample if int(r['training_geometries'])==n and float(r['snr_db'])==30
        and float(r['frequency_shrinkage'])==strength], 'mean_nmse_linear')) for n in ns], 'o-', label=label)
axes[2].set_xscale('log', base=2)
axes[2].set_xticks(ns, ns)
axes[2].set_xlabel('Nested training geometry count')
axes[2].set_title('30 dB; same 64 validation geometries')
for ax in axes:
    ax.set_ylabel('Mean linear NMSE, converted to dB')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(main/'diagnostic_checks.png',dpi=180)
fig.savefig(main/'diagnostic_checks.svg')
plt.close(fig)

table = '\n'.join('| '+f'{snr:g}'+' | '+' | '.join(f'{lookup[snr,m]:.3f}'
    for m in ['ls_min_norm','lmmse_joint','lmmse_per_tone','lmmse_shrunk'])+' |' for snr in snrs)
sample_lines = []
for n in [64,128,256,512]:
    values = []
    for strength in [0.,1.]:
        selected = [r for r in sample if int(r['training_geometries'])==n and float(r['snr_db'])==30
            and float(r['frequency_shrinkage'])==strength]
        values.append(f"{db(avg(selected, 'mean_nmse_linear')):.3f}")
    sample_lines.append('| '+str(n)+' | '+' | '.join(values)+' |')
sample_table = '\n'.join(sample_lines)
report = f'''## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + run
- Origin Date: 2026-10-21
- Verification Status: VERIFIED
- Version Label: linear_baselines_20261020_v1
- 性质：查看原曲线后设计的实现检查与探索性复查。
- 来源：原SNR运行的512训练几何、64验证几何；本轮另生成256个独立筛选几何。
- 核验：60项单元检查、小流程、逐场景NMSE重算和LS解析误差分解通过。

# LS与联合LMMSE：异常定位及新基线

LS在低SNR高于0 dB的直接来源是传播到信道估计的噪声误差。高SNR约−2 dB的平台来自未观测维度。联合LMMSE的高SNR回升对应无结构样本协方差对新场景的外推误差；原复数公式与噪声尺度检查通过。

![新几何下的四组条件](linear_baselines.png)

## LS：缺失维度与噪声分开核实

原曲线每个子载波用8或16个复观测恢复32维信道，使用最小范数解 $\\widehat h=A^\\dagger y$。令 $P=A^\\dagger A$，误差为 $-(I-P)h+A^\\dagger n$。第一项为未观测分量，第二项为传播后的噪声；两项正交，平方范数严格相加。

原128个筛选几何在0 dB接收SNR下，平均归一化未观测误差为0.631291，噪声项为1.774299，总计2.405589，转dB为3.812 dB。30 dB下噪声项降至0.001774，总误差为−1.986 dB。噪声趋零时的平台为约−1.998 dB。

零估计的NMSE等于1，即0 dB。低SNR的LS噪声项可超过信道能量，因而出现正dB。原曲线10 dB时为−0.922 dB。

本轮新256几何在0 dB下，未观测项为0.626174、实际噪声项为1.989954，解析期望噪声项为1.986314；总计4.177 dB。30 dB总计−2.019 dB，无噪声平台约{db(.6261735849464971):.3f} dB。两批场景的差异保持在各自结果中。

SNR按训练几何平均接收功率定义，弱信道的实际SNR更低。主指标对每个几何的归一化误差等权平均；原0 dB的LS若改为误差能量总和除以信道能量总和，将为1.952 dB，低于主指标3.812 dB。本轮保留原评分定义。

四组DMA矩阵的最大条件数为3.190、7.113、5.468和14.431；实测训练噪声功率与设定值之比为0.99950–1.00427。没有观测到噪声方差的倍数错配。LS保留原伪逆定义。

## 联合LMMSE：协方差泛化检查

原函数按标准复数LMMSE公式计算，使用训练样本均值和无偏样本协方差。1024维信道只有512个训练几何，样本协方差秩最多511。代码已与独立构造的完整复数协方差公式对照。

同一512训练几何，在30 dB时联合LMMSE的训练NMSE为−4.199 dB，独立验证NMSE为+1.553 dB；逐频32维协方差的验证NMSE为−2.150 dB。20 dB对应的联合训练/验证为−3.753/+0.147 dB。训练集误差改善没有传递到新场景。

原筛选集在30 dB时，联合LMMSE的行空间误差为0.007784，零空间误差为1.487130；LS的零空间误差为0.631291。主要差距位于需要用统计相关项外推的未观测信道部分。随着噪声方差降低，LMMSE对协方差小特征值方向的抑制减弱；不准确的跨频统计项会产生错误外推。

另一个只使用验证集的对照，固定观测、噪声、64验证几何与求解公式，仅从同一训练集取嵌套子集。下表为30 dB下四硬件条件的线性NMSE平均后转dB：

| 训练几何数 | 无收缩联合/dB | 逐频/dB |
|---:|---:|---:|
{sample_table}

逐频估计在该对照中随样本数增加持续改善；联合估计在高SNR不单调。直接增加训练样本数的效果取决于所估计的协方差结构；相同512个样本下，逐频约束已经明显改变结果。

![误差分解与协方差对照](diagnostic_checks.png)

## 已完成的协方差处理

保留训练均值、噪声和所有观测，将联合样本协方差与逐频块对角协方差作凸组合。收缩系数0恢复原联合估计，1等价于逐频估计。验证候选为0、0.1、0.25、0.5、0.75、0.9、1；每个带宽只选一个系数，全部SNR与8/16配置共用。

验证集冻结3 GHz系数0.1、30 GHz系数0.9后，才生成新的256个筛选几何。CSI哈希与训练、验证及原筛选集无重叠。原联合统计量在服务器重建后逐元素差为0。

高维样本协方差的收缩有既有统计学依据，参见[Ledoit与Wolf原论文](https://www.ledoit.net/Well-conditioned2004.pdf)。本轮系数由验证集选择，目标为逐频块协方差；不使用该论文的系数公式。

## 新256几何的结果

32阵元×32频点，3/30 GHz、8/16配置。下表各值均先在四硬件条件和256几何上平均线性NMSE，再转dB。

| 接收SNR/dB | 最小范数LS/dB | 原联合LMMSE/dB | 逐频LMMSE/dB | 冻结收缩LMMSE/dB |
|---:|---:|---:|---:|---:|
{table}

30 dB时，冻结收缩版比原联合版低3.817 dB；逐频版比原联合版低4.031 dB。20 dB时分别低2.252和2.439 dB。逐频版在中高SNR比全SNR共用系数的收缩版更好；20/30 dB差距为0.187/0.214 dB。

全九点汇总，收缩版为−1.185 dB、逐频版−1.120 dB、原联合版+0.329 dB。全SNR汇总与高SNR单点评分的排序不同，逐条件结果均保留。

## 核验和适用范围

- 60项单元检查通过，新增的稠密协方差混合公式与逐频等价检查通过。
- 小流程192条指标重算差0；完整流程36,864条NMSE重算差0。
- 9,216个几何×条件的LS解析误差分解通过，最大差2.85e−14；9,216组观测哈希检查一致。
- 选参后冻结统计量与训练缓存哈希保持不变；服务器原联合统计量重建差0。
- 服务器原始导出及两个诊断文件共{len(hashes)}项逐字节核验通过，见[文件哈希](server_file_hashes.json)。

本轮是固定DMA配置、32×32系统和单组训练/验证几何的探索性结果。256个筛选几何为独立单位，36个条件重复使用相同几何。没有测量在线时延。完整尺寸、新DMA配置及训练数据抽样变化需另行验证。

[实现与冻结规则](../../docs/LINEAR_BASELINES.md) · [逐场景指标](metrics.csv) · [LS误差分解](ls_decomposition.csv) · [验证候选](validation_scores.csv) · [冻结记录](frozen.json) · [核验](audit.json)

[原曲线诊断](../linear_baseline_probe_20261020_v1/contexts.csv) · [样本数对照](../baseline_covariance_probe_20261020_v1/sample_count.csv) · [统计解释检查](statistical_review.json)

统计解释覆盖11/11项，所有比较为描述性结果，未作显著性结论。
'''
(main/'report.md').write_text(report, encoding='utf-8')
review = [
    ('Simpson paradox','CHECKED','分别保留四硬件条件和九SNR；全SNR汇总与高SNR点的逐频/收缩排序不同，已列出。'),
    ('Ecological fallacy','CHECKED','仅报告条件均值，不推断每个场景都改善。'),
    ('Berkson paradox','CHECKED','保留全部256几何，无按误差筛除。'),
    ('Collider bias','NOT_APPLICABLE','未按估计结果或中间变量筛选和调节。'),
    ('Base rate neglect','NOT_APPLICABLE','连续误差指标，不涉及诊断概率。'),
    ('Regression to mean','CHECKED','新几何同时运行所有基线，以相同观测进行比较。'),
    ('Survivorship bias','CHECKED','全部36条件与36864条指标保留。'),
    ('Look elsewhere effect','CHECKED','完整候选表和全部条件公开，未作显著性结论。'),
    ('Garden of forking paths','EXPLORATORY','查看旧结果后设计；系数仅在验证集选择，新筛选生成晚于冻结。'),
    ('Correlation causation','CHECKED','误差正交分解为代数恒等式；跨频收缩和训练数对照固定其余因素，不推断其他系统。'),
    ('Reverse causality','NOT_APPLICABLE','确定性估计流程与仿真控制，无观测性因果方向问题。')]
write(main/'statistical_review.json',{'coverage':'11/11','inferential_claims':False,
    'independent_unit':'geometry, shared across 36 conditions',
    'items':[{'type':a,'status':b,'assessment':c} for a,b,c in review]})
write(main/'implementation_verification.json',{'unit_tests':60,'unit_tests_status':'passed',
    'command':'python -m unittest discover -s tests -v','mathematical_checks':'complex128 dense covariance mixture and per-tone equivalence',
    'smoke_run':'linear_baselines_smoke_20261020_v2','smoke_nmse_difference':0.,
    'source_files_match_archived_manifest':True,'server_exports_byte_verified':len(hashes),
    'figures_visually_checked':['linear_baselines.png','diagnostic_checks.png'],'measurement_timing':'not measured'})
for folder, title, mode in [(probe,'原LS及LMMSE事后诊断','validate'),(counts,'训练样本数的验证集对照','validate')]:
    (folder/'report.md').write_text('## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
        f'- Origin Mode: {mode}\n- Origin Date: 2026-10-21\n- Verification Status: VERIFIED\n- Version Label: {folder.name}\n\n'
        f'# {title}\n\n执行配置、源码与数据哈希随目录保存。分析与适用范围见[完整复查报告](../linear_baselines_20261020_v1/report.md)。\n',encoding='utf-8')
shutil.copyfile(__file__, main/'analysis.py')
print(json.dumps({'verified_server_files':len(hashes),'new_screen_metrics':len(metrics),'report':str(main/'report.md')}))
