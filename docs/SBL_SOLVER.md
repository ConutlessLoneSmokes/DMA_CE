# 复数跨频SBL求解与收敛检查

2026-10-20版本：`convex_bound_kkt_v1`。已知噪声、字典和观测保持原定义；将原阻尼EM更新改为同一边缘似然目标的凸上界更新。真实CSI不参与求解和停止。

## 模型和更新

各频点独立复系数共享非负角距方差 $\gamma_g$。对第 $k$ 个频点，记观测为 $y_k$，压缩字典为 $A_k$，已知复噪声方差为 $\sigma^2$：

$$
C_k=\sigma^2 I+A_k\operatorname{diag}(\gamma)A_k^H,
\qquad
L(\gamma)=\frac{1}{K}\sum_k\left[\log\det C_k+y_k^HC_k^{-1}y_k\right].
$$

计算各列的统计量：

$$
q_g=\frac{1}{K}\sum_k\lvert a_{kg}^HC_k^{-1}y_k\rvert^2,
\qquad
s_g=\frac{1}{K}\sum_k a_{kg}^HC_k^{-1}a_{kg}.
$$

更新为

$$
\gamma_g^+=\max\left(\gamma_{\min},\ \gamma_g\sqrt{q_g/s_g}\right).
$$

其依据是：对凹函数 $\log\det C_k$ 使用当前点的切平面上界，并在恒等式

$$
y_k^HC_k^{-1}y_k
=\min_x\left[\sigma^{-2}\lVert y_k-A_kx\rVert_2^2+x^H\operatorname{diag}(\gamma)^{-1}x\right]
$$

中固定当前最优 $x=\mu_k$。得到的可分离上界包含 $s_g\gamma_g+K^{-1}\sum_k\lvert\mu_{kg}\rvert^2/\gamma_g$，最小化即得上述平方根更新。精确算术下目标不增加。该方法属于标准凸上界SBL，参考 [Hashemi等，2021，式42](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433122/)。这里将统计量按不同频点的压缩字典求和，使用复数模平方。

数值下限为 $\gamma_{\min}=10^{-8}/G$，其中 $G$ 是字典列数。它使整个字典的方差下限之和固定。若某列的全部压缩响应为零，则该列 $q_g=s_g=0$，保留其初始方差。

## 停止条件

目标梯度为 $\partial L/\partial\gamma_g=s_g-q_g$。令 $u_g=q_g/s_g-1$，归一化的一阶最优性残差定义为

$$
R=\max_g\left\{\max(u_g,0),\ \frac{\gamma_g-\gamma_{\min}}{\max_j\gamma_j}\lvert u_g\rvert\right\}.
$$

第一项检查是否存在增加方差即可下降的方向；第二项检查未落在下限的分量是否满足互补条件。$R=0$ 对应当前方差下限约束下的一阶KKT条件。默认阈值为 $10^{-3}$。单步方差相对变化仍记录为诊断字段，但不用于判停。

将全部字典列复制 $m$ 份、对应方差均分后，$C_k$、$q_g/s_g$、总方差更新和 $R$ 都保持不变。下限也按列数同步缩小。因此重复列不再像原EM那样将有效更新缩小到 $1/m$。此性质有CPU双精度回归检查。

## 预算检查和记录

`early_stopping=True` 时，满足 $R$ 阈值的样本停止。`False` 时，所有样本执行给定次数的更新，最后按最终 $R$ 标记是否收敛；曾在中途满足阈值不等于最终满足。

研究配置和基线预算筛选使用 `False`。比较相邻预算时，要求两者均实际执行完毕、总体收敛比例至少95%、每条件至少90%、每条件线性均值转dB的NMSE变化不超过0.1 dB、平均后验方差变化不超过5%。粗、密字典分别判断。未通过就记录预算不足，不进入后续训练和最终筛选。

缓存与逐场景表记录求解器版本、停止策略、实际更新数、$R$ 和每频点目标值。旧SBL缓存不能通过扩散续训入口继承到本版本；需要新运行重算基线、训练标定和精修模型。

一阶条件和预算稳定性针对当前统计模型及局部解。最终方法优劣依据完整CSI误差和总计算量。
