# 实现核验记录

以下汇总已完成的检查及其证据。实现检查、保存预测复算与方法性能评价分别解释；短预算曲线不作性能排名。

| 日期与范围 | 支持的判断及边界 | 原始证据 |
|---|---|---|
| 2026-10-22，多RF实现 | 77项测试通过，覆盖RF块连接、馈线重置、白化、球面距离、资源计数与分解LMMSE等价性。小规模及4×256完整尺寸短流程覆盖9个SNR、4方法，216条指标复算最大相对差2.99×10⁻⁷；同ID恢复的metrics/summary/frozen哈希一致。短预算SBL未收敛均保留 | [verification.json](../results/multirf_validation_20261022_v1/verification.json) |
| 2026-10-22，观测数扫描运行前 | 73项检查、双GPU带宽小流程、CPU功率扫描回归及J=64尺寸检查通过；1160条指标重算MSE一致、NMSE最大相对差2.24×10⁻⁷，SNR校准差≤4.31×10⁻⁷ dB，恢复前后科学数据哈希一致。短预算SBL未收敛均保留 | [verification.json](../results/observation_baselines_validation_20261022_v1/verification.json) |
| 2026-10-22，已保存观测数结果 | 40960条指标复双精度复算，NMSE最大相对差4.45×10⁻⁷、MSE一致；LS重放、联合LMMSE对偶式、划分与配对、源码/冻结哈希及矩阵秩检查通过；SNR差≤4.58×10⁻⁷ dB。8191/8192个SBL输出达一阶阈值，未重新拟合SBL | [verification.json](../results/observation_baselines_analysis_20261022_v1/verification.json) |
| 2026-10-21，物理模型 | 69项检查通过；600条小流程指标重算差0，50步SBL未收敛均保留。v1缓存类型错误及v2安全恢复修正有记录 | [implementation_verification.json](../results/physical_baselines_20261021_v2/implementation_verification.json) |

## 历史核验归档

2026-10-21清理时，13个smoke、尺寸检查和单步续训运行的原报告、配置、环境、源码清单与审计字段已集中到 [IMPLEMENTATION_CHECKS.json](IMPLEMENTATION_CHECKS.json)。覆盖首轮/确定性流程、原research流程、128×128尺寸、SBL修正、线性基线、SNR扫描、独立续训和真实断点单步恢复。原生成缓存、模型和重复图表已按[清理记录](CLEANUP_20261021.md)处理。

测试源码与对应小规模配置保留，新检查使用新运行ID。各正式规模的探索性评价及其独立审计从[实验日志](EXPERIMENT_LOG.md)进入。

2026-10-22补充归档了6组多RF冒烟/尺寸、观测数冒烟/尺寸和物理回归检查。配置、环境、源码清单、冻结选择、审计状态及原报告已合并到上述JSON，现共有19条记录；独立核验报告继续保留。重复图表、临时工作副本和字节码缓存按[本地清理清单](archive/CLEANUP_20261022.json)移出工作区，当前源码及配置与此前77项测试通过时的快照一致。
