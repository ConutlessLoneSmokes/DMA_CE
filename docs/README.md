# 文档索引

项目结果先看 [截至2026-11-04的阶段总结](EXPERIMENT_LOG.md#阶段总结)，各轮状态与证据看同页实验记录；执行任务看 [工作流程](WORKFLOW.md)，统一用语看 [领域术语](../GLOSSARY.md)。

## 运行与方法

| 主题 | 文档 |
|---|---|
| 时延分数与输出修正幅度标定 | [COHERENT_CALIBRATION](guides/COHERENT_CALIBRATION.md) · [10月28日新几何结果](../results/coherent_calibration_20261028_v1/interpretation.md) |
| 相干增益的行/零空间分解、固定边缘相位扰动 | [COHERENT_MECHANISM](guides/COHERENT_MECHANISM.md) · [10月28日机制结果](../results/coherent_mechanism_20261028_v2/interpretation.md) |
| 保留空间不确定性的相干跨频后验探查 | [COHERENT_DELAY_PROBE](guides/COHERENT_DELAY_PROBE.md) · [10月27日结果：新32几何正增量](../results/coherent_delay_20261026_v1/interpretation.md) |
| 密集多 RF 阵列、四方法 SNR 扫描 | [MULTIRF_SNR](guides/MULTIRF_SNR.md) |
| 固定观测的先验互补与跨频物理诊断 | [PRIOR_COMPLEMENTARITY](guides/PRIOR_COMPLEMENTARITY.md) |
| 模型内协方差与未知几何物理联合恢复 | [STRUCTURED_PRIOR_PROBE](guides/STRUCTURED_PRIOR_PROBE.md) |
| 观测数扫描与固定训练平均接收 SNR | [OBSERVATION_BASELINES](guides/OBSERVATION_BASELINES.md) |
| 物理路径、热噪声、发射功率与吸收假设 | [PHYSICAL_MODEL](PHYSICAL_MODEL.md) |
| LS 分解、联合/逐频/收缩 LMMSE | [LINEAR_BASELINES](LINEAR_BASELINES.md) |
| 复数跨频 SBL 的更新与停止条件 | [SBL_SOLVER](SBL_SOLVER.md) |
| 修正后分阶段方法比较与续训 | [RESEARCH_STUDY](guides/RESEARCH_STUDY.md) |
| 原条件 DDPM 与结构化恢复的 SNR 比较 | [SNR_COMPARISON](guides/SNR_COMPARISON.md) |

配置数值以 `configs/` 与各运行的 `config_resolved.json` 为准；指南解释方法条件、选择理由和复现方式。

## 文献与来源

- [2026-10-26 配置冻结后的CE算法](research/ALGORITHM_NEXT_STEPS_20261026.md)：当前SBL缺少的相干跨频约束、几何/时延参数边界，以及相关SBL、DMA张量CE、THz展开和2026-09顺序离网方法的直接近邻；具体新算法尚待讨论。
- [2026-10-24 DMA跨频硬件响应与信道估计核查](research/DMA_CROSSFREQUENCY_LITERATURE_20261024.md)：相邻原文、实际频响假设及当前实现；10月26日已明确两类比较口径、固定采集与已知硬件边界，短选择保留原随机bank，后续CE方法待定。
- [实现采用的依据](research/REFERENCES.md)；物理传播和多 RF 架构的具体引用随对应指南。
- [2026-10-23研究进程与待讨论方向](research/RESEARCH_PROGRESS_20261023.md)：结合多RF正式结果、100步输入对照及直接相关原始论文。
- [2026-10-23先验利用核查](research/PRIOR_COMPLEMENTARITY_20261023.md)：SBL、LS、LMMSE的实际约束，相关SBL与连续参数估计原文，以及同观测融合的解释条件。
- [2026-10-23结构化结合候选](research/STRUCTURED_PRIOR_OPTIONS_20261023.md)：区分洛伦兹硬件响应与传播幅相，核实SBL现有协方差，并比较物理联合恢复、模型内统计融合及学习更新。
- [2026-10-16 文献综合与候选方向](research/LITERATURE_SYNTHESIS.md)：保留当时的18篇比较与方案推导。
- [历史原文证据笔记](research/literature_notes/README.md) · [论文索引](research/literature_notes/catalog.json) · [原仓库来源](research/literature_notes/github_sources.json)。
- [复用函数与哈希](UPSTREAM_REUSE.json)。

## 实验与核验

- [2026-10-28既有观测融合证据核查](research/EXISTING_FUSION_EVIDENCE_20261028.md)：关联增量、零空间补全、DDPM退化及协方差/几何失配的证据边界。

- [实验日志](EXPERIMENT_LOG.md)：已完成比较、异常修正、失败和未执行状态。
- [实现核验](IMPLEMENTATION_CHECKS.md)：数学/物理检查、小流程与结果复算；[历史原始核验字段](IMPLEMENTATION_CHECKS.json)。
- [2026-11-04同步前核验](archive/SYNC_20261104.json)：现有代码检查、CSV汇总复算、源码归档哈希与文档链接。
- 各实验的详细分析和逐场景结果保留于 `results/<id>/`，从实验日志进入。

## 历史归档

- [首轮预声明](archive/EXPERIMENT_PLAN.md) · [第二轮预声明](SECOND_STAGE_PLAN.md)。
- [research_001 停止与检查记录](archive/RESEARCH_001_INSPECTION.md) · [2026-10-20 备选方向讨论](archive/RESEARCH_DISCUSSION_20261020.md)。
- [2026-10-21 清理记录](CLEANUP_20261021.md)及其[本地清单](archive/CLEANUP_20261021.json)、[服务器清单](archive/CLEANUP_SERVER_20261021.json)、[保留文件哈希](archive/RETAINED_SERVER_FILES.json)。
- [2026-10-22 本地清理清单](archive/CLEANUP_20261022.json)：新增短流程结果、临时文档与缓存的归档及移除范围。
- [本地退役运行](archive/RETIRED_RUNS.json) · [服务器退役运行](archive/RETIRED_SERVER_RUNS.json)。

历史记录按原日期解释，状态以实验日志为准。旧脚本、已冻结报告引用的 `SECOND_STAGE_PLAN.md`、`PHYSICAL_MODEL.md`、`LINEAR_BASELINES.md`、`SBL_SOLVER.md`、`CLEANUP_20261021.md` 和核验/复用清单保留原路径；通过本页统一分类，避免改写历史结果与审计数据。

## Agent 配置

[GitHub Issues](agents/issue-tracker.md) · [任务标签](agents/triage-labels.md) · [领域文档规则](agents/domain.md)。有实际决策需要记录时再建立 `docs/adr/`。
