# 实验记录

更新：2026-10-21。以下区分探索性方法比较、验证集诊断与实现核验。

## 当前评价

| 日期与运行 | 实际完成内容与结论 | 结果位置 |
|---|---|---|
| 2026-10-21，linear_baselines_20261021_v2 | 512训练、64验证、256独立筛选几何；36组条件。LS低SNR误差分解与协方差对照完成。30 dB下原联合LMMSE +1.778 dB、逐频−2.253 dB、冻结收缩−2.039 dB。收缩系数仅用验证集选择 | [报告](../results/linear_baselines_20261021_v2/report.md) |
| 2026-10-20，snr_comparison_20261020_v1 | 9个SNR点、128个新筛选几何、3个初始化。20 dB下密SBL−17.319 dB、原条件DDPM−14.609 dB、新先验＋学习更新−2.015 dB。三轮新组合未达到中高SNR的密SBL精度 | [报告](../results/snr_comparison_20261020_v1/interpretation.md) |
| 2026-10-20，research_003 | 修正SBL后重新生成缓存并训练，5个新初始化。匹配评价：密SBL−11.704 dB、确定性−11.560 dB、DDPM十六次平均−10.627 dB | [核验](../results/research_003_review_20261020_v1/report.md) |

这些评价使用32×32缩小系统和既定DMA配置。详细训练、验证、筛选划分及平均口径保存在各报告中。

## 诊断与修正

- 原SBL单步变化阈值会提前停止，曾出现密字典弱于粗字典的排序；[原诊断](../results/sbl_dictionary_diagnostic_20261019_v1/report.md)保存失败及收敛证据。
- SBL改为同一边缘似然目标的凸上界更新，并用归一化一阶条件停止。[固定观测复查](../results/sbl_corrected_probe_20261020_v1/report.md)完成后，在research_003重建完整比较。
- [新DDPM分步诊断](../results/snr_comparison_20261020_v1_diagnostic_v2/report.md)保留实际初始化误差、每场景等能量随机方向对照、去噪与观测更新各阶段数据。
- [线性基线原曲线诊断](../results/linear_baseline_probe_20261020_v1/report.md)及[协方差样本数对照](../results/baseline_covariance_probe_20261020_v1/report.md)用于区分观测缺失、噪声与统计估计问题。
- 线性收缩v1存在非整数系数的隐式float32掩码乘法，v2改为双精度。沿用原候选、验证规则和预定筛选观测，选择仍为0.1/0.9，逐场景线性NMSE最大变化2.3842e−7；三位小数结果不变。[修正对照](../results/linear_baselines_20261021_v2/precision_comparison.json)保留旧版本标识与差值，旧输出已合并清理。

## 历史探索

以下使用修正前SBL，保留为研究经过，不与当前正确收敛的基线直接合并排名。

- [首轮筛选](../results/pilot_20261016_v1/report.md)：8条件、128筛选几何。可靠度残差扩散弱于当时100次SBL，后续转向确定性对照。
- [验证集诊断](../results/validation_probe_20261016_v1/report.md)：增加SBL迭代和增加生成平均次数的检查。
- [第二轮可靠度比较](../results/deterministic_20261016_v1/report.md)：可靠度总体额外收益约0.029 dB，区间包含0。
- research_001在20,000步预算边界停止：8k→20k验证改善0.11091 dB超过原0.1 dB稳定阈值。[当时检查记录](RESEARCH_001_INSPECTION.md)及[退役服务器运行记录](RETIRED_SERVER_RUNS.json)保存停止原因。
- [research_002验证总览](../results/research_002_overview_20261019_v1/report.md)：续训至40k，30k→40k改善0.00628 dB；原始评分CSV与绘图输入已集中保留于该结果目录。

## 实现核验与清理

短流程、尺寸检查和真实断点单步恢复已合并到[实现核验记录](IMPLEMENTATION_CHECKS.md)。测试源码和小规模配置保留，生成的smoke模型、缓存、图和重复表已移除。

[清理记录](CLEANUP_20261021.md)说明现用完整运行、历史材料和删除范围。Git保留历史提交；过期的未跟踪训练缓存不再保留。
