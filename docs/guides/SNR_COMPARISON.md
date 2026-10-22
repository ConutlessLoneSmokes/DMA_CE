# SNR扫描与DDPM先验结构化恢复

本实验在完成的 `research_003` 上新增独立运行；原模型、缓存和结果保持可核验。SNR从−10至30 dB，每5 dB一个实际运行点。系统继续采用32阵元、32频点、单RF链、3/30 GHz带宽、8/16次DMA配置、角度±60°、距离0.1–0.4 m。

## 运行

服务器 `/home/lsy/DMA_CE` 的 `thz_dma` 环境：

```bash
python scripts/run_snr_comparison.py --config configs/snr_comparison.json --parent research_003 --run-id snr_comparison_20261020_v1
```

同ID只在配置与执行源码完全相同时恢复。每个模型保存优化器和随机数状态；结果按条件保存，完成条件可复用。总运行设4小时上限，失败保存错误与现有产物。

本地缩小系统检查使用已完成的实现验证运行：

```bash
python scripts/run_snr_comparison.py --config configs/snr_comparison_smoke.json --parent corrected_sbl_smoke_20261020_v1 --run-id snr_comparison_smoke_20261020_v2
```

## 五条主曲线与消融

| 曲线 | 实际处理 |
|---|---|
| LS | 逐频最小范数伪逆；欠定观测下明确指定该解 |
| LMMSE | 用同一批训练几何估计完整阵元—频率联合均值与协方差，complex128求解 |
| SBL | 已修正的MM/KKT求解器、密字典、最多32000更新；逐场景记录真实迭代数与一阶条件 |
| 最新DDPM | `research_003` 中冻结的SBL条件残差模型，主图为16次输出平均 |
| DDPM先验＋学习结构更新 | 完整干净信道DDPM冻结后，交替去噪、学习正方差、按真实DMA观测计算高斯均值 |

辅助图同时显示固定方差更新、既有确定性精修和单次条件DDPM。固定/学习更新使用同一个DDPM、同一LMMSE初始化、同一轮数及全局强度，分离学习方差更新的作用。最新模型选用预先声明的201、202、203三个种子；新模型使用相同初始化编号。原五种子结果单独保存在 `results/research_003`。

完整CSI DDPM仅从512个训练几何在两种带宽下的信道学习，不输入导频、DMA配置或真实角距；避免把同一信道在多个SNR/导频下的重复作为独立样本。网络沿用仓库已有DenoisingCNN和VP日程，训练20000步并在验证扩散噪声损失上选断点。

学习方差网络训练4000步，带噪训练仍使用父运行10/20 dB和相同的512几何。验证集可选择断点及全局强度，筛选集冻结权重，扫描全部9个SNR。其余SNR用于冻结模型的观测条件迁移评估，旧条件DDPM复用父运行权重。

## 结构化迭代

设初始化为训练统计得到的LMMSE均值及其对角方差。第$t$轮的基本方差为初始方差乘$s\,2^{-t}$，其中$s$只由验证集从0.1、1、10中选取。将复方差除以2，匹配VP日程中的$(1-\bar\alpha)/\bar\alpha$，用冻结DDPM去噪当前均值。

小CNN从去噪均值实虚部、基本方差对数、反投影残差能量及测量噪声方差对数预测有界对数倍率，得到正方差$v$。随后计算

$$
\mu^+ = m + V A^H(A V A^H+\sigma_n^2I)^{-1}(y-Am),\qquad V=\operatorname{diag}(v).
$$

每个频点使用实际DMA矩阵。网络只改变方差，复均值由上述式子给出；三轮共享权重。固定消融去掉方差CNN。零初始化的CNN与固定更新完全相同，检查梯度时允许其穿过冻结DDPM的输入。

这是plug-and-play近似。中间均值已经依赖观测，内部Gaussian矩并非整个算法的精确后验，方差也没有作为真实CSI误差标定。借鉴Li等TCOM2026中“解析后验计算＋学习方差更新”的分工；其原文采用AoSA与SBL，不使用DDPM。

## 配对、评分与归档

- 每个几何共享带宽/导频/SNR条件，同一标准噪声按SNR缩放，所有方法共享完全相同的观测。
- 归一化及接收SNR基准来自父运行训练数据；每个样本实际SNR另存。测试信道能量只用于NMSE评分。
- 新筛选种子为20268939，128个独立几何在模型和超参数冻结后生成。
- 主NMSE先计算逐场景线性误差，再平均几何与初始化，最后转dB。95%区间按几何块bootstrap，先平均初始化；区间不包含初始化随机性。
- 传统基线只记一次，不把重复的训练种子标签作为额外观测。
- 保存所有逐场景指标、行/零空间误差、观测哈希、SBL收敛状态；结果差也保留。主图不使用插值生成额外点。
- 运行目录保存完整配置、源代码压缩包、模型、预测、父运行哈希、冻结参数和审计；大缓存及模型不进入Git。
- 自动重算全部保存预测的NMSE，并对首尾SNR条件重放固定与学习恢复，检查与冻结记录一致。

## 原始参考

- Li et al., *Wideband Hybrid-Field THz UM-MIMO Channel Estimation: A Dual-Attention-Aided Deep-Unfolded Bayesian Learning Approach*, TCOM 74, 6523–6540, 2026. https://doi.org/10.1109/TCOMM.2026.3675428
- Zhou et al., *Generative Diffusion Models for High Dimensional Channel Estimation*, TWC 24(7), 5840–5854, 2025. https://arxiv.org/abs/2408.10501

本实验使用现有训练预算探索该组合；完整尺寸及更多训练初始化依据配对结果再设计。

## 结果与核验

已完成运行 `snr_comparison_20261020_v1` 的主曲线、冻结规则和复算见[结果分析](../../results/snr_comparison_20261020_v1/interpretation.md)；去噪与观测更新的归因见[分步诊断](../../results/snr_comparison_20261020_v1_diagnostic_v2/report.md)。当前状态统一见[实验日志](../EXPERIMENT_LOG.md)。
