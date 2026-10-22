# 文档索引

项目状态看 [实验日志](EXPERIMENT_LOG.md)，执行任务看 [工作流程](WORKFLOW.md)，统一用语看 [领域术语](../GLOSSARY.md)。

## 运行与方法

| 主题 | 文档 |
|---|---|
| 密集多 RF 阵列、四方法 SNR 扫描 | [MULTIRF_SNR](guides/MULTIRF_SNR.md) |
| 观测数扫描与固定训练平均接收 SNR | [OBSERVATION_BASELINES](guides/OBSERVATION_BASELINES.md) |
| 物理路径、热噪声、发射功率与吸收假设 | [PHYSICAL_MODEL](PHYSICAL_MODEL.md) |
| LS 分解、联合/逐频/收缩 LMMSE | [LINEAR_BASELINES](LINEAR_BASELINES.md) |
| 复数跨频 SBL 的更新与停止条件 | [SBL_SOLVER](SBL_SOLVER.md) |
| 修正后分阶段方法比较与续训 | [RESEARCH_STUDY](guides/RESEARCH_STUDY.md) |
| 原条件 DDPM 与结构化恢复的 SNR 比较 | [SNR_COMPARISON](guides/SNR_COMPARISON.md) |

配置数值以 `configs/` 与各运行的 `config_resolved.json` 为准；指南解释方法条件、选择理由和复现方式。

## 文献与来源

- [实现采用的依据](research/REFERENCES.md)；物理传播和多 RF 架构的具体引用随对应指南。
- [2026-10-16 文献综合与候选方向](research/LITERATURE_SYNTHESIS.md)：保留当时的18篇比较与方案推导。
- [历史原文证据笔记](research/literature_notes/README.md) · [论文索引](research/literature_notes/catalog.json) · [原仓库来源](research/literature_notes/github_sources.json)。
- [复用函数与哈希](UPSTREAM_REUSE.json)。

## 实验与核验

- [实验日志](EXPERIMENT_LOG.md)：已完成比较、异常修正、失败和未执行状态。
- [实现核验](IMPLEMENTATION_CHECKS.md)：数学/物理检查、小流程与结果复算；[历史原始核验字段](IMPLEMENTATION_CHECKS.json)。
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
