# 实现核验记录

2026-10-21清理时，smoke、尺寸检查和单步续训的生成数据已移除，原报告正文、配置、环境、源码清单及审计字段集中保存于[核验记录](IMPLEMENTATION_CHECKS.json)。这些记录支持实现正确性，不作为方法性能或论文结果。

| 历史运行 | 保存的核验信息 |
|---|---|
| `continuation_parent_smoke_20261019_v1` | 原配置、环境及报告已归档 |
| `continuation_smoke_20261019_v1` | audit.json: passed；implementation_verification.json: passed；validation_audit.json: passed |
| `corrected_sbl_smoke_20261020_v1` | audit.json: passed；validation_audit.json: passed |
| `deterministic_smoke_20261016_v1` | audit.json: passed；status.json: complete |
| `linear_baselines_smoke_20261020_v1` | audit.json: recorded；status.json: complete |
| `linear_baselines_smoke_20261020_v2` | audit.json: recorded；status.json: complete |
| `linear_baselines_smoke_20261021_v3` | audit.json: recorded；status.json: complete |
| `research_smoke_20261018_v2` | audit.json: passed |
| `smoke_20261016_v1` | status.json: complete |
| `snr_comparison_smoke_20261020_v1` | audit.json: passed |
| `snr_comparison_smoke_20261020_v2` | audit.json: passed；resume_verification.json: recorded |
| `continuation_state_probe_20261019_v1` | 原配置、环境及报告已归档 |
| `research_shape128_20261018_v2` | status.json: passed |

测试源码与`configs/*smoke.json`配置继续保留，后续检查可在新运行ID下重新生成。当前正式结果各自保留独立审计，见[实验记录](EXPERIMENT_LOG.md)。
