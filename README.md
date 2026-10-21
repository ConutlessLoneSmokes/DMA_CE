# DMA_CE

物理宽带DMA压缩观测下的信道估计研究。复用[THz_DMA](https://github.com/ConutlessLoneSmokes/THz_DMA)的球面波、Lorentzian/馈线响应与轻量CNN，来源和原始哈希见[复用清单](docs/UPSTREAM_REUSE.json)。

当前结果采用32阵元、32子载波、单微带和单RF链，300 GHz载频、3/30 GHz带宽、8/16个DMA配置，角度±60°、距离0.1–0.4 m。它们属于缩小系统的探索性实验。

## 当前结果

| 实验 | 主要结果 | 记录与数据 |
|---|---|---|
| LS/LMMSE复查 | 30 dB时，原联合LMMSE +1.778 dB、逐频版−2.253 dB、验证冻结收缩版−2.039 dB；256个独立筛选几何 | [报告](results/linear_baselines_20261021_v2/report.md) · [曲线](results/linear_baselines_20261021_v2/linear_baselines.png) |
| −10至30 dB扫描 | 20 dB时，密SBL−17.319 dB、原条件DDPM−14.609 dB、DDPM先验＋学习更新−2.015 dB；128个独立筛选几何 | [报告](results/snr_comparison_20261020_v1/interpretation.md) · [曲线](results/snr_comparison_20261020_v1/snr_nmse_main.png) |
| research_003 | 五初始化匹配评价：密SBL−11.704 dB、确定性精修−11.560 dB、条件DDPM十六次平均−10.627 dB | [核验与比较](results/research_003_review_20261020_v1/report.md) |

各实验的样本与评价条件分别见对应报告。NMSE先按场景在线性尺度平均，再转dB；跨带宽、配置和SNR的汇总规则以各报告为准。

## 实验记录

- [实验经过、失败与版本变更](docs/EXPERIMENT_LOG.md)
- [合并后的实现核验记录](docs/IMPLEMENTATION_CHECKS.md)
- [清理范围与保留位置](docs/CLEANUP_20261021.md)
- [文献比较与候选方向](docs/LITERATURE_SYNTHESIS.md) · [历史证据笔记](docs/literature_notes/README.md)
- [实验设计](docs/EXPERIMENT_PLAN.md) · [SBL求解与收敛](docs/SBL_SOLVER.md)

## 运行与复现

Python 3.10及以上，安装匹配显卡的PyTorch后：

```text
python -m pip install -e .
python -m unittest discover -s tests -v
```

每次实验使用新运行ID。已完成实验保留源码快照和环境，精确重放使用对应`source.zip`中的代码。

- [正式分阶段实验与续训](docs/RESEARCH_STUDY.md)
- [SNR扫描定义和命令](docs/SNR_COMPARISON.md)
- [线性基线定义、冻结规则和命令](docs/LINEAR_BASELINES.md)

GitHub保存代码、配置、实验记录、必要逐场景结果、图和审计。现用完整模型、观测、缓存和预测保存在服务器`/home/lsy/DMA_CE/runs/`的`research_003`、`snr_comparison_20261020_v1`、`linear_baselines_20261021_v2`中。

本地THz_DMA原仓库保留在`upstream/THz_DMA/`并由Git忽略；服务器原仓库保持在`/home/lsy/THz_DMA`。历史smoke输出与过期运行缓存已清理，核验结论和失败记录保留。
