# 有界场景方法比较执行记录

运行：research_003。标签：bounded_corrected_sbl_single_refinement。

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

- [base/matched](final/base/matched/summary.csv)
- [base/paths5](final/base/paths5/summary.csv)
