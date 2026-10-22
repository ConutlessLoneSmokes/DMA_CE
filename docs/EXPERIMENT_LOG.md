# 实验记录

更新：2026-10-22。这里集中记录实验状态与简要结论；完整条件、数据和科学解释见对应报告。

## 当前状态

密集多 RF 阵列的4×256单元、4RF、30 GHz、J=32四方法SNR扫描已完成实现与短预算核验，正式512/64/128几何性能实验尚未启动。[设计与命令](guides/MULTIRF_SNR.md) · [实现证据](../results/multirf_validation_20261022_v1/verification.json)。

以下已完成比较采用32阵元、32子载波的单 RF 缩小系统，属于探索性实验。NMSE按场景在线性尺度平均后转dB，初始化与条件平均顺序见各报告。

| 日期与运行 | 主要结论 | 证据 |
|---|---|---|
| 2026-10-22，观测数扫描3/30 GHz | 512训练、64验证、128筛选几何；J=4–64。30 dB、J=64时SBL为−36.232/−35.914 dB；−10 dB小J仍弱于LMMSE。LS在J=32放大噪声；30 GHz联合样本LMMSE训练子空间投影下限约−3.58 dB | [分析与完整运行ID](../results/observation_baselines_analysis_20261022_v1/interpretation.md) |
| 2026-10-21，physical_baselines_20261021_v2 | 固定热环境的三种路径模型、五方法，512/64/128几何；0 dBm时SBL随机增益−15.539、物理无吸收−14.395 dB，单独吸收影响≤0.0013 dB；物理修正未消除LS与联合LMMSE异常 | [分析](../results/physical_baselines_20261021_v2/interpretation.md) |
| 2026-10-21，linear_baselines_20261021_v2 | 512/64/256几何；30 dB下联合LMMSE +1.778、逐频−2.253、验证冻结收缩−2.039 dB | [报告](../results/linear_baselines_20261021_v2/report.md) |
| 2026-10-20，snr_comparison_20261020_v1 | 9个SNR、128新筛选几何、3初始化；20 dB下密SBL−17.319、原条件DDPM−14.609、DDPM先验＋学习更新−2.015 dB，三轮新组合未达到中高SNR的SBL精度 | [报告](../results/snr_comparison_20261020_v1/interpretation.md) |
| 2026-10-20，research_003 | 修正SBL后重建缓存并训练；5初始化匹配评价：密SBL−11.704、无可靠度确定性精修−11.560、DDPM十六次平均−10.627 dB | [复核](../results/research_003_review_20261020_v1/report.md) |

## 异常、失败与修正

- 原SBL单步变化阈值提前停止，曾出现密字典弱于粗字典；[诊断](../results/sbl_dictionary_diagnostic_20261019_v1/report.md)保留原失败，改为同一边缘似然目标的凸上界更新和一阶停止条件后完成[固定观测复查](../results/sbl_corrected_probe_20261020_v1/report.md)，再执行research_003。
- [DDPM分步诊断](../results/snr_comparison_20261020_v1_diagnostic_v2/report.md)保留初始化误差、等能量随机方向对照、去噪与观测更新各阶段数据。
- [线性基线诊断](../results/linear_baseline_probe_20261020_v1/report.md)及[协方差样本数对照](../results/baseline_covariance_probe_20261020_v1/report.md)区分观测缺失、噪声和统计估计问题。收缩v1的隐式float32掩码乘法在v2改为双精度；候选、验证规则和筛选观测保持不变，逐场景NMSE最大变化2.3842e−7，见[修正对照](../results/linear_baselines_20261021_v2/precision_comparison.json)。
- 物理模型v1冻结后因缓存NumPy标量不能安全恢复而停止，v2修复缓存类型，科学配置和筛选场景不变；[修正记录](../results/physical_baselines_20261021_v2/implementation_verification.json)保留失败。传播与噪声假设见[物理模型](PHYSICAL_MODEL.md)。
- 观测数扫描8192个SBL输出中1个未达一阶阈值，已保留；统计与排除该点的描述性对照见[分析](../results/observation_baselines_analysis_20261022_v1/interpretation.md)。

## 历史探索

以下使用修正前SBL，保留研究经过，不与修正后的基线合并排名。

- [首轮筛选](../results/pilot_20261016_v1/report.md)：8条件、128几何，可靠度残差扩散弱于当时100次SBL；[验证诊断](../results/validation_probe_20261016_v1/report.md)检查迭代与生成平均预算。
- [第二轮可靠度比较](../results/deterministic_20261016_v1/report.md)：可靠度额外收益约0.029 dB，区间包含0。
- research_001在20,000步预算边界停止：8k→20k验证改善0.11091 dB，超过原0.1 dB稳定阈值；见[当时检查](archive/RESEARCH_001_INSPECTION.md)及[退役元数据](archive/RETIRED_SERVER_RUNS.json)。
- [research_002验证总览](../results/research_002_overview_20261019_v1/report.md)：续训至40k，30k→40k改善0.00628 dB；必要原始评分及来源集中保留在结果目录。

数学、物理检查和复算证据汇总于[实现核验](IMPLEMENTATION_CHECKS.md)。历史缓存清理范围见[清理记录](CLEANUP_20261021.md)；当前完整运行主要保存在服务器 `runs/`，每个结果目录保留运行ID、配置与来源。
