# LS与LMMSE基线复查

本轮在2026-10-20查看已有SNR结果后设计，属于探索性复查。保留旧曲线，先处理线性基线，不改变DDPM模型或训练。

## 固定的观测与评分

每个子载波使用 $y_k=A_kh_k+n_k$，$A_k$ 为8或16行、32列的已知DMA观测矩阵。噪声经过原流程白化，每个复元素的方差为 $\sigma^2$。SNR由512个训练几何的平均接收功率定义；每个场景的实际SNR随信道能量变化。

主指标先对每场景 $\lVert\widehat h-h\rVert^2/\lVert h\rVert^2$ 求均值，再转dB。它不同于误差能量总和除以信道能量总和；后者会让强信道获得更大权重。

## LS的检验

保留最小范数LS $\widehat h_k=A_k^\dagger y_k$。令 $P_k=A_k^\dagger A_k$，则

$$
\widehat h_k-h_k=-(I-P_k)h_k+A_k^\dagger n_k.
$$

两项分别位于观测矩阵的零空间和行空间，平方误差严格相加。条件期望为

$$
E[\lVert\widehat h_k-h_k\rVert^2\mid h_k]
=\lVert(I-P_k)h_k\rVert^2+\sigma^2\lVert A_k^\dagger\rVert_F^2.
$$

逐场景检查该分解与 $A_kA_k^\dagger=I$，并保存实际噪声项与解析期望噪声项。高SNR保留未观测分量的误差；低SNR噪声项可使NMSE大于1。0 dB对应NMSE等于1，即零估计的归一化误差。

## LMMSE的检验与改动

原实现使用512个训练几何，估计32频点×32阵元的1024维协方差。该样本协方差的秩最多511。训练场景与独立验证场景的误差、逐频协方差对照和无噪声偏差分别保留在诊断中。

新增协方差族：

$$
\widehat R_\lambda=(1-\lambda)\widehat R+
\lambda\operatorname{blockdiag}(\widehat R_{11},\ldots,\widehat R_{KK}).
$$

$\lambda=0$精确保留原联合样本协方差；$\lambda=1$等价于逐子载波独立LMMSE。其余值保持子载波内协方差不变，减小跨频项。训练均值、复数共轭、噪声方差与求解精度均保持原定义。

对展平后的信道和块对角观测算子 $B$ 使用

$$
\widehat h=\widehat\mu+\widehat R_\lambda B^H
(B\widehat R_\lambda B^H+\sigma^2I)^{-1}(y-B\widehat\mu).
$$

数学检查将实现与完整稠密协方差公式逐元素比较，并检查 $\lambda=1$ 与逐频实现一致。[Ledoit与Wolf的原始论文](https://www.ledoit.net/Well-conditioned2004.pdf)支持高维样本协方差需要正则化的一般动机；这里使用验证集选择的块协方差收缩，不采用该论文的收缩系数公式。

## 冻结规则

- 继承原512训练几何和64验证几何；所有方法使用相同观测、噪声、DMA状态与训练信息。
- 比较 $\lambda\in\{0,0.1,0.25,0.5,0.75,0.9,1\}$；每个带宽选择一个值，目标是验证集全部SNR和配置数的平均线性NMSE。选定后保存统计量与冻结哈希。
- 冻结后才生成256个新筛选几何，种子20269021；检查其CSI哈希与训练、验证和已有筛选均不重叠。
- 在−10至30 dB的9个SNR点、3/30 GHz和8/16配置下比较最小范数LS、原联合LMMSE、逐频LMMSE和冻结收缩LMMSE，共36个条件。
- 结果不包含在线时延测量。逐场景指标、LS误差分解、预测、源码、环境与配置归档；模型、缓存和PDF不提交。

```text
python scripts/run_linear_baselines.py --config configs/linear_baselines.json --parent snr_comparison_20261020_v1 --run-id linear_baselines_20261021_v2
```

本轮区分的是无结构联合样本协方差与跨频正则化。它不增加训练几何，不等同于已知总体协方差的理想LMMSE。现有统计量的精度和线性估计的误差下限需按实际结果报告。

## 已完成结果

2026-10-21完成上述冻结与新筛选，并将非整数收缩系数的掩码乘法修正为双精度。v2沿用首次冻结后的256筛选几何，验证选择仍为3 GHz系数0.1、30 GHz系数0.9，候选与选择规则保持相同。30 dB时，原联合LMMSE、逐频LMMSE、收缩LMMSE分别为+1.778、−2.253、−2.039 dB。原联合统计量重建逐元素差0，144次预测重放及36,864条五项指标重算差0，9,216次LS误差分解通过。详细归因、全部SNR数据与训练样本数对照见[复查报告](../results/linear_baselines_20261021_v2/report.md)。
