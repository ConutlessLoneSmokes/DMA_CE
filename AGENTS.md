# DMA_CE 工作约定

本仓库由用户于2026-10-16授权新建，先研究候选A：同观测粗估、可靠度条件残差生成、真实DMA观测一致性。允许实现、检查和运行规模可控的仿真实验。

- 保留 THz_DMA 原仓库；复用函数的来源和原始哈希记入 docs/UPSTREAM_REUSE.json。
- 原始PDF、缓存、模型和凭据不提交。每次运行创建独立 runs/<id>，保存完整配置、种子、环境、源码和逐场景结果。
- 所有方法使用相同的物理观测、噪声和信息。真实路径、真值误差只用于训练标签、评分和明确标注的诊断。
- 训练、验证、筛选按独立几何场景划分。同一场景的多个SNR/带宽/配置只能在同一划分中。
- 归一化、可靠度标定只用训练集；选参只用验证集，先冻结再读取筛选集结果。
- 首轮为缩小系统的单种子方法筛选；完整尺寸、多种子和通信闭环在筛选结果基础上设计。
- 如实记录失败和非收敛。候选A是否有效由对照决定，不以更小观测残差代替完整CSI改善。
- 修改新逻辑后运行相关数学/物理检查。中文汇报，区别实现验证、探索性实验和论文主结果。
- 每次答复与实验后同步更新相关项目文档，记录当前结论、已采纳决定、实际执行状态和证据；讨论中的候选与已完成实验分别标明。文档归属见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## Agent skills

### Issue tracker

任务与规格使用 GitHub Issues（ConutlessLoneSmokes/DMA_CE）；操作约定见 [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md)。

### Triage labels

使用默认五个分流标签；含义见 [docs/agents/triage-labels.md](docs/agents/triage-labels.md)。

### Domain docs

采用单一领域布局：根目录 `GLOSSARY.md` 与按需建立的 `docs/adr/`；读取规则见 [docs/agents/domain.md](docs/agents/domain.md)。

开始任务时按 [工作流程](docs/WORKFLOW.md) 选择适用 skill；查找运行指南、实验状态或历史证据时使用 [文档索引](docs/README.md)。
