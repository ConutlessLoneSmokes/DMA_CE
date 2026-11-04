# 实现核验记录

以下汇总已完成的检查及其证据。实现检查、保存预测复算与方法性能评价分别解释；短预算曲线不作性能排名。

| 日期与范围 | 支持的判断及边界 | 原始证据 |
|---|---|---|
| 2026-11-04，既有实验整理与GitHub同步前核验 | 原实验环境下91项检查全部通过；从12992条逐场景CSV重算313组均值，最大dB差7.11×10⁻¹⁵；机制分解的误差恒等式、行/零空间可加性及均值一致。5份源码归档的557成员与各自清单哈希相同；待提交JSON、Python语法及相对文档链接核对通过。该轮核验对象是代码和已有导出，张量评分及后验重建沿用下列各实验的原日期记录 | [同步前核验](archive/SYNC_20261104.json) |
| 2026-10-28，相干输出步长标定 | 自由协方差重建、已知标定解、截断与二次误差自检通过；独立评分最大相对差6.22×10⁻¹⁶，完整尺寸自由后验方程相对平方差6.31×10⁻¹⁶，独立增广LS系数差1.12×10⁻¹⁵。训练统计、验证选择、冻结与新几何隔离通过；续接保留原冻结/观测，数学函数AST一致。分析脚本标量广播索引错误修正后通过，原预测未修改 | [运行核验](../results/coherent_calibration_20261028_v1/verification.json) · [独立核验](../results/coherent_calibration_20261028_v1/independent_verification.json) · [报告](../results/coherent_calibration_20261028_v1/interpretation.md) |
| 2026-10-28，相干机制分析 | 手算复数分解与共享因子后验自检通过，原相干4项相关数学检查通过；6720条分解可加性与误差恒等式通过，父评分最大相对差6.15×10⁻¹⁶、原后验重放相对平方差7.67×10⁻¹⁶；独立NumPy Gram与SVD投影最大绝对差2.64×10⁻¹⁶，控制边缘差4.57×10⁻¹⁶。父输入前后逐哈希一致；v1吞吐不足主动停止，v2相同控制复用因子完成 | [运行核验](../results/coherent_mechanism_20261028_v2/verification.json) · [独立核验](../results/coherent_mechanism_20261028_v2/independent_verification.json) · [报告](../results/coherent_mechanism_20261028_v2/interpretation.md) |
| 2026-10-27，相干跨频结果与归档 | 1280条保存评分独立NumPy重算，最大相对差5.91×10⁻¹⁶；独立协方差正向作用核对自由/相干/打乱后验，最大相对残差2.50×10⁻¹²，恢复CSI相对平方差≤6.30×10⁻¹⁶。验证选择、冻结顺序及几何隔离核验通过；完整40文件约1.23 GB和三组实施运行逐哈希归档一致。这些检查支持计算与交付，性能结论另看新几何配对结果 | [verification.json](../results/coherent_delay_20261026_v1/verification.json) · [归档核对](../results/coherent_delay_20261026_v1/delivery_verification.json) · [实验报告](../results/coherent_delay_20261026_v1/interpretation.md) |
| 2026-10-26，相干跨频运行前 | 零相关等价、完整复高斯后验/证据与解证书、时延/周期及边界检查通过；91项仓库检查通过；全尺寸v2/v3短流程完成。v1诊断重放抽样数量错误已修正并保留失败。只支持实现与流程，不预设性能收益 | [implementation_checks.json](../results/coherent_delay_20261026_v1/implementation_checks.json) · [规格](guides/COHERENT_DELAY_PROBE.md) |
| 2026-10-26，短配置选择 | 频带覆盖、种子复现、评分与选择自检及相关多RF数学检查通过；硬件/白化噪声重放、LS正规方程重放、576条保存评分与区间/冻结复算通过。评分最大相对差9.97×10⁻¹⁴，LS相对平方差6.43×10⁻¹⁶；27个完整运行文件逐哈希归档一致。检查不证明候选有性能收益 | [自检](../results/configuration_probe_20261026_v1/implementation_checks.json) · [复算](../results/configuration_probe_20261026_v1/verification.json) · [交付](../results/configuration_probe_20261026_v1/delivery_verification.json) |
| 2026-10-23，结构化先验及结果复算 | 零统计项等价、独立复高斯后验、证据下降、传播、梯度及拟合检查通过；86项完整检查和4×256短流程通过，边界修正后受影响10项通过。1440条保存指标最大相对差2.35×10⁻¹⁵，原NumPy传播及增广最小二乘重放物理预测最大相对平方差7.73×10⁻¹⁶；验证选择、区间、划分与冻结哈希一致。真值投影仅为事后诊断，检查不预设性能改善 | [实现](../results/structured_prior_20261023_v1/implementation_checks.json) · [结果复算](../results/structured_prior_20261023_v1/verification.json) |
| 2026-10-23，固定观测先验互补探查 | 凸权重、零空间投影与oracle已知解自检通过；CPU复双精度重放LS、融合、oracle及评分，指标最大相对差1.25×10⁻¹⁵。验证冻结、共享观测、独立几何、源码/输入哈希和配对区间复算通过；不设性能门槛，100步未到停止阈值的输出全部保留 | [verification.json](../results/prior_complementarity_20261023_v1/verification.json) · [运行审计](../results/prior_complementarity_20261023_v1/audit.json) |
| 2026-10-22，多RF实现 | 77项测试通过，覆盖RF块连接、馈线重置、白化、球面距离、资源计数与分解LMMSE等价性。小规模及4×256完整尺寸短流程覆盖9个SNR、4方法，216条指标复算最大相对差2.99×10⁻⁷；同ID恢复的metrics/summary/frozen哈希一致。短预算SBL未收敛均保留 | [verification.json](../results/multirf_validation_20261022_v1/verification.json) |
| 2026-10-22，观测数扫描运行前 | 73项检查、双GPU带宽小流程、CPU功率扫描回归及J=64尺寸检查通过；1160条指标重算MSE一致、NMSE最大相对差2.24×10⁻⁷，SNR校准差≤4.31×10⁻⁷ dB，恢复前后科学数据哈希一致。短预算SBL未收敛均保留 | [verification.json](../results/observation_baselines_validation_20261022_v1/verification.json) |
| 2026-10-22，已保存观测数结果 | 40960条指标复双精度复算，NMSE最大相对差4.45×10⁻⁷、MSE一致；LS重放、联合LMMSE对偶式、划分与配对、源码/冻结哈希及矩阵秩检查通过；SNR差≤4.58×10⁻⁷ dB。8191/8192个SBL输出达一阶阈值，未重新拟合SBL | [verification.json](../results/observation_baselines_analysis_20261022_v1/verification.json) |
| 2026-10-21，物理模型 | 69项检查通过；600条小流程指标重算差0，50步SBL未收敛均保留。v1缓存类型错误及v2安全恢复修正有记录 | [implementation_verification.json](../results/physical_baselines_20261021_v2/implementation_verification.json) |

## 历史核验归档

2026-10-21清理时，13个smoke、尺寸检查和单步续训运行的原报告、配置、环境、源码清单与审计字段已集中到 [IMPLEMENTATION_CHECKS.json](IMPLEMENTATION_CHECKS.json)。覆盖首轮/确定性流程、原research流程、128×128尺寸、SBL修正、线性基线、SNR扫描、独立续训和真实断点单步恢复。原生成缓存、模型和重复图表已按[清理记录](CLEANUP_20261021.md)处理。

测试源码与对应小规模配置保留，新检查使用新运行ID。各正式规模的探索性评价及其独立审计从[实验日志](EXPERIMENT_LOG.md)进入。

2026-10-22补充归档了6组多RF冒烟/尺寸、观测数冒烟/尺寸和物理回归检查。配置、环境、源码清单、冻结选择、审计状态及原报告已合并到上述JSON，现共有19条记录；独立核验报告继续保留。重复图表、临时工作副本和字节码缓存按[本地清理清单](archive/CLEANUP_20261022.json)移出工作区，当前源码及配置与此前77项测试通过时的快照一致。
