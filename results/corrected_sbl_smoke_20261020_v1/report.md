# 有界场景方法比较执行记录

运行：corrected_sbl_smoke_20261020_v1。标签：corrected_sbl_single_refinement_smoke_implementation_only。

SBL求解器：convex_bound_kkt_v1。基线检查强制执行完整预算，收敛标记来自边缘似然的一阶条件。
确定性精修只保留无可靠度版本，采用跨频卷积和按条件归一化的训练损失。

各选择只使用训练/验证数据；最终各系统在保存模型和参数哈希后生成新的筛选几何。
独立重复为训练初始化种子和几何场景；同一几何的带宽、配置与SNR按块配对。

| 阶段 | 状态 |
|---|---|
| baseline | complete |
| curves | complete |
| factorial | complete |
| reliability | complete |
| budget | complete |
| diffusion | complete |
| final | complete |

训练、验证曲线和精度—时间图保存在相应阶段目录。主NMSE先在线性尺度汇总再转dB。
修正能量与真实残差对齐只用于评分诊断。可靠度标定和条件损失权重仅来自训练集。
最终通信指标是单用户、单接收RF链、跨频共享DMA状态的接收速率，扣除导频与配置切换开销。
所报时间包含SBL和精修的批量均摊时间，排除编译预热；不等同于单样本时延。

这是小规模流程验证，迭代和训练预算已缩短；数值不能作为方法性能或论文结论。

- [base/matched](final/base/matched/summary.csv)
- [base/paths3](final/base/paths3/summary.csv)
- [base/range_shift](final/base/range_shift/summary.csv)
- [n16_k16/matched](final/n16_k16/matched/summary.csv)
- [n16_k16/paths3](final/n16_k16/paths3/summary.csv)
- [n16_k16/range_shift](final/n16_k16/range_shift/summary.csv)
- [n16_k8/matched](final/n16_k8/matched/summary.csv)
- [n16_k8/paths3](final/n16_k8/paths3/summary.csv)
- [n16_k8/range_shift](final/n16_k8/range_shift/summary.csv)

## 2026-10-20实现核查

39项检查通过，覆盖复高斯后验、独立分量解析最优解、边缘似然下降、四份重复列的结果及停止位置不变性、强制预算计数、旧SBL缓存拒绝、训练/验证独立、CPU/GPU优化器与随机状态续训。

本运行完成全部7阶段，再用同ID恢复，7个阶段均复用。验证审计重放6个扩散检查点、核对288条指标，最大差为零。最终审计核对18个缓存、12个冻结模型、9组场景、4,032条指标和12次模型重放，所有误差与重放差值均为零。

主短系统为8阵元、8子载波，每个划分只有6/4/4个几何，2个训练初始化和2个新的最终初始化，训练检查点为2/4/6步。为覆盖原有通用分支，还执行16阵元的两种极小扩展及路径数/距离偏移。正式研究配置继续固定32×32及原角距范围。

两个基线预算5/10步实际执行完毕。短验证放宽了收敛和稳定阈值，一阶条件通过率为零；这里的 `complete` 只确认流程和记录逻辑。正式配置保留95%总体、90%逐条件通过比例及0.1 dB/5%稳定性要求。

所有确定性训练任务均为无可靠度、跨频卷积、条件归一化损失；不再执行方差图、标量和置换对照。低层历史模型读取接口保留。

[验证审计](validation_audit.json)与[最终审计](audit.json)保存完整核对结果。原39项检查输出、执行日志及补充核查记录保存在本目录；模型和原始缓存保存在独立 `runs/`。
