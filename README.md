# DMA_CE

物理宽带DMA压缩观测下的粗估可靠度条件残差信道估计。

本项目从 [THz_DMA](https://github.com/ConutlessLoneSmokes/THz_DMA) 选择复用球面波、Lorentzian/馈线响应与轻量去噪CNN，建立独立的同观测SBL、条件生成和消融实验。来源见 [复用清单](docs/UPSTREAM_REUSE.json)，实验设计见 [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)。

首轮采用单微带32阵元、32子载波，300 GHz、3/30 GHz带宽、8/16配置及10/20 dB训练平均接收SNR。距离0.1–0.4 m位于该缩小孔径的辐射近场；它是方法筛选场景。完整尺寸和通信结论需另行验证。

## 运行

使用 Python 3.10及以上，安装与显卡匹配的PyTorch，再安装本包其余依赖。

```text
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_experiment.py --config configs/smoke.json --run-id smoke_001
python scripts/run_experiment.py --config configs/pilot.json --run-id pilot_001
python scripts/audit_experiment.py pilot_001
python scripts/diagnose_validation.py pilot_001 validation_probe_001
python scripts/select_sbl_budget.py pilot_001 budget_001
python scripts/run_experiment.py --config configs/deterministic.json --run-id deterministic_001
python scripts/audit_experiment.py deterministic_001
```

每个运行ID只能创建一次；新实验使用新ID。运行保存在被Git忽略的 runs/；紧凑逐场景指标、图和报告另保存 results/，可提交到私有研究仓库。

第二轮的控制变量、迭代预算选择规则与新筛选集安排见[实验预声明](docs/SECOND_STAGE_PLAN.md)。

## 结果状态

13项数学/物理/数据边界检查通过。[首轮筛选](results/pilot_20261016_v1/report.md)完成8个条件、128个独立筛选几何和11组方法，共11,264条逐场景指标，GPU用时207秒。全部指标及保存模型重放复核一致。

当前可靠度残差扩散仍弱于SBL；确定性残差网络在3 GHz下改善0.50–1.66 dB，在30 GHz下改善0.02–0.26 dB。可靠度自身的独立收益尚待确定性网络的完整对照。

[验证集诊断](results/validation_probe_20261016_v1/report.md)另完成1,280条指标：增加SBL迭代明显改善粗估；平均16次生成输出降低误差，但仍未超过原100次SBL。诊断在查看主结果后设计，未用于重选主实验参数。训练数据、模型和完整预测保存在本地runs/。
