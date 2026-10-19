# 五步实验执行记录

运行：continuation_smoke_20261019_v1。标签：five_step_smoke_implementation_only。

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
