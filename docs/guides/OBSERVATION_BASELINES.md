# 观测数扫描：LS、LMMSE与SBL

2026-10-22设计并完成正式观测数扫描。代码复用 `physical_study.py`，不改变LS、LMMSE或SBL求解器。两带宽结果及独立核验见[分析报告](../../results/observation_baselines_analysis_20261022_v1/interpretation.md)。

## 问题、变量和独立样本

本轮研究在给定平均接收SNR下，增加每个待估信道的DMA观测数量，完整CSI误差如何变化。横轴是每子载波DMA配置数J；32个子载波共提供32J个复观测。512个训练几何用于估计LMMSE统计量，在全部J/SNR下固定，不把训练样本量变化混进观测数曲线。

| 设置 | 本轮设计 |
|---|---|
| 物理系统 | 300 GHz，32阵元×32子载波，单RF链 |
| 带宽 | 3 GHz、30 GHz，分别出图 |
| 每子载波观测数J | 4、8、12、16、24、32、48、64 |
| 训练平均接收SNR | −10、0、15、30 dB |
| 信道 | 既有物理传播含吸收组：1条直达、2条单次反射路径 |
| 几何范围 | 角度±60°，距离0.1–0.4 m |
| 独立几何数 | 训练512，验证64，筛选128 |
| 随机性 | 一组训练/验证抽样、一组64状态DMA调谐序列；筛选种子20269222 |
| SBL | 97×13角距字典，最多32000次更新，KKT阈值10⁻³，批量16 |

相同场景的不同J、SNR、带宽始终留在同一划分中。不同方法使用同一观测、噪声方差、DMA矩阵；真值只用于训练统计量、验证选择和评分。128个筛选几何是独立评价单位，不把各条件的重复使用当作额外独立样本。

J<32时每个频点的线性观测方程欠定；J≥32且矩阵满列秩时不再欠定。真实DMA响应的相关性仍会影响噪声放大，因此不预设LS曲线在32附近一定单调改善。完整矩阵和逐场景行/零空间误差保留在缓存与指标中，可复核异常点。

## 固定SNR与物理噪声

沿用296 K天线温度、290 K接收机参考温度和6 dB噪声系数。对带宽B、K个频点，白化物理噪声每频点为 $N_k=k_BT_aB/K$，接收机附加噪声包含在DMA白化矩阵中。温度及接收机噪声不随J或目标SNR变化。

先仅用训练集计算归一化尺度s，以及每个J下的白化无噪声响应平均功率

$$
q_J=\frac{1}{N_{train}KJ}\sum_{g,k,j}\lvert[A_k^{(J)}h_{g,k}/s]_j\rvert^2.
$$

对目标 $\rho=10^{\mathrm{SNR}/10}$，令算法复噪声方差 $\sigma_J^2=q_J/\rho$，实际每频点导频功率

$$
P_{tone,J}=\frac{N_k}{s^2\sigma_J^2},\qquad P_{tx,J}=K P_{tone,J}.
$$

这通过改变已知导频功率匹配接收SNR，物理接收噪声保持不变。筛选场景自身的真值或接收功率不参与功率校准；不同场景的实际SNR允许不同。每个J使用调谐序列前J个状态，基础噪声也使用对应前缀。因各J的校准导频功率可能不同，不能把不同J归一化后的观测误称为逐位相同的前缀。

**这是固定平均每观测SNR的曲线，不是固定总训练能量的比较。** 增加J同时增加导频符号数。输出记录校准后的发射功率，以及按有效符号时长K/B换算的 $T_{pilot}=JK/B$、$E_{pilot}=P_{tx,J}JK/B$；不计循环前缀和硬件切换开销。

按带宽启动两个独立运行时，每个运行在自己的带宽上计算训练归一化尺度；几何与DMA调谐种子相同。带宽选择写入解析后的配置。双带宽合并成一次调用会使用跨带宽全局尺度及不同的噪声抽样编号，不把两种执行方式宣称为逐位相同的重放。

## 方法与评价

保留三个方法类别、五条曲线：

- LS：每个频点的复数最小范数伪逆解，支持欠定和过定矩阵。
- 联合样本LMMSE：512个训练几何估计完整跨频统计量，作为此前高SNR异常的对照。
- 逐频LMMSE：保留频点内32维空间协方差，跨频块置零。
- 验证收缩LMMSE：在联合与逐频协方差之间收缩。每带宽只从既有7个候选中选一个系数，目标为验证集全部J/SNR的平均线性NMSE；该系数由所有J/SNR共用。
- 密字典SBL：沿用当前凸上界更新和归一KKT判停，已知噪声方差，跨频共享角距方差。训练真值不参与逐场景求解。

联合/逐频/收缩版本使用相同训练信息；它们都不是已知总体协方差的理想LMMSE。收缩版本不是为每个筛选点单独选参。冻结训练缓存、统计量、源码哈希后才生成筛选几何。筛选结果不用于重新选择收缩系数或SBL预算。

对每个独立几何g，计算

$$
\mathrm{NMSE}_g=\frac{\lVert\widehat H_g-H_g\rVert_F^2}{\lVert H_g\rVert_F^2},\qquad
\mathrm{MSE}_{g,physical}=\frac{\lVert\widehat H_g-H_g\rVert_F^2}{KN}.
$$

主图先平均线性NMSE再转dB；另外输出物理信道尺度MSE曲线，纵轴取对数刻度。算法归一化尺度上的MSE也写入逐场景CSV。MMSE是估计准则，不能把所有方法的实测误差都称为MMSE。

每个带宽分别输出NMSE和MSE两张图，每图四个SNR子图，横轴为J。阴影为128个独立几何的配对块bootstrap描述性95%区间，只覆盖固定训练集与DMA状态下的筛选几何变化。SBL非收敛点保留，在图上加x标记；逐场景表保留实际更新数、KKT残差及目标值。

## 服务器运行

工作目录 `/home/lsy/DMA_CE`；运行时使用既有环境，无新增依赖。CPU承担信道生成、4线程线性代数预处理、评分和绘图。两张GPU各负责一个带宽，两个运行目录互相独立，无共享写入或合并脚本。

在服务器终端A运行：

```bash
cd /home/lsy/DMA_CE
CUDA_VISIBLE_DEVICES=0 /home/lsy/miniconda3/envs/thz_dma/bin/python scripts/run_physical_baselines.py \
  --config configs/observation_baselines.json --bandwidth-ghz 3 --device cuda \
  --run-id observation_baselines_20261022_3ghz_v1
```

同时在服务器终端B运行：

```bash
cd /home/lsy/DMA_CE
CUDA_VISIBLE_DEVICES=1 /home/lsy/miniconda3/envs/thz_dma/bin/python scripts/run_physical_baselines.py \
  --config configs/observation_baselines.json --bandwidth-ghz 30 --device cuda \
  --run-id observation_baselines_20261022_30ghz_v1
```

每个带宽运行32个条件，保存20480条逐几何方法指标；两者共40960条。每个进程单次时限8小时，是执行上限，不是预计耗时。中断或超时后，保持源码、配置、运行ID和设备选项不变，重复原命令会校验冻结记录并从已保存的完整条件继续。不要同时对同一个运行ID启动多个进程。要改变样本数、SBL预算或实现，使用新ID。

结果保存在 `results/<run-id>/`：`summary.csv`、`metrics.csv`、`link_budgets.csv`、`validation_scores.csv`、`frozen.json`、`audit.json`、`status.json`、报告及PNG/SVG。完整源码、观测、模型统计量和预测保存在 `runs/<run-id>/`。请以运行状态和SBL收敛记录判断完成情况，不根据图文件出现就认定结果充分收敛。

小流程验证命令仅将配置替换为 `configs/observation_baselines_smoke.json`，并使用新的smoke运行ID；它使用8阵元、4频点、J=2/8/12和12次SBL更新，只验证实现，不能用作正式精度结论。

## 结果与核验

运行前小流程、J=64尺寸、评分/SNR复算和恢复检查见[实现核验](../IMPLEMENTATION_CHECKS.md)。两带宽已完成的观测数比较及其LS噪声放大、联合LMMSE训练子空间、SBL收敛诊断见[完整分析](../../results/observation_baselines_analysis_20261022_v1/interpretation.md)与[独立核验](../../results/observation_baselines_analysis_20261022_v1/verification.json)。

物理传播、材料和吸收表的假设沿用[物理模型](../PHYSICAL_MODEL.md)。本轮没有增加训练几何、训练神经网络，也没有加入固定总能量或多硬件种子的另一套实验。
