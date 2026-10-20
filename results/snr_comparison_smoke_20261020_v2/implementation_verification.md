# SNR比较流程实现验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-20
- Verification Status: VERIFIED
- Version Label: snr_comparison_smoke_v2

8阵元、8频点，3/30 GHz、2/4次配置、−10/10/30 dB，共12组条件；6个训练几何、4个验证几何、4个筛选几何、一个初始化。CSI先验训练12步，方差网络训练8步。本运行检查流程与数学实现。

整体56项单元检查通过，包含14项新增结构化数学检查。随后补充的优化器/随机状态恢复与冻结模型哈希检查，所在的5项流程检查全部通过；当前共58项独立检查。

数学检查覆盖：

- 复高斯条件均值与完整矩阵公式一致，方差及复杂梯度正确。
- 联合频率LMMSE保留复交叉协方差，并采用训练样本的无偏协方差。
- DDPM实方差为复误差方差的一半，扩散时间与Tweedie去噪式一致。
- 零初始化方差网络与固定更新逐位一致，训练梯度能够穿过冻结DDPM的输入。

实际短流程完成480条逐场景指标。全部指标由保存预测重算，最大差0；固定与学习结构各两次重放，最大差0。各方法的观测哈希一致，筛选观测在冻结记录之后生成。

同ID恢复复用了全部12个条件，新增训练步数为0；再次重算480条指标与4次结构恢复，差值仍为0。模拟中断检查另验证了优化器、随机状态及断点选择可恢复。

执行源码哈希：

- `snr_study.py`：`4e1d36952d1f89225ddbfddf8e5017c64e013a88840b0be8db1c62b4e9f6cf50`
- `structured.py`：`a95d834c684574fbe7a7b2e7f7734e3d5e58487b5c618f5b02b7d00f1ed4e7c3`

[原始审计](audit.json) · [恢复核验](resume_verification.json) · [配置](config_resolved.json) · [流程输出](report.md)

正式性能比较使用独立的32×32服务器运行。
