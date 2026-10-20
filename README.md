# DMA_CE

物理宽带DMA压缩观测下的粗估可靠度条件残差信道估计。

本项目从 [THz_DMA](https://github.com/ConutlessLoneSmokes/THz_DMA) 选择复用球面波、Lorentzian/馈线响应与轻量去噪CNN，建立独立的同观测SBL、条件生成和消融实验。来源见 [复用清单](docs/UPSTREAM_REUSE.json)，实验设计见 [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)。

首轮采用单微带32阵元、32子载波，300 GHz、3/30 GHz带宽、8/16配置及10/20 dB训练平均接收SNR。距离0.1–0.4 m位于该缩小孔径的辐射近场；它是方法筛选场景。完整尺寸和通信结论需另行验证。

## 运行

使用 Python 3.10及以上，安装与显卡匹配的PyTorch，再安装本包其余依赖。

```text
python -m pip install -e .
python -m unittest discover -s tests -v
```

每个运行ID只能创建一次；新实验使用新ID。运行保存在被Git忽略的 runs/；紧凑逐场景指标、图和报告另保存 results/，可提交到私有研究仓库。

第二轮的控制变量、迭代预算选择规则与新筛选集安排见[实验预声明](docs/SECOND_STAGE_PLAN.md)。

当前比较保持±60°和0.1–0.4 m范围，使用修正后的SBL，并只保留无可靠度确定性精修。完整说明与服务器命令见[RESEARCH_STUDY.md](docs/RESEARCH_STUDY.md)，求解与收敛依据见[SBL_SOLVER.md](docs/SBL_SOLVER.md)。

```text
python scripts/run_research.py --config configs/research_smoke.json --run-id server_smoke_20261020
python scripts/audit_research.py server_smoke_20261020
python scripts/run_research.py --config configs/research.json --run-id research_003 --stage baseline
```

基线通过后，同一ID运行完整比较：

```text
python scripts/run_research.py --config configs/research.json --run-id research_003
```

旧 `research_001/002` 使用原SBL，不作为本版本的续训父运行。新 `research_003` 若在扩散预算边界停止，可在新ID恢复它的优化器与随机状态：

```text
python scripts/run_research.py --continue-from research_003 --diffusion-checkpoints 30000 40000 --run-id research_004 --stage diffusion
python scripts/audit_research.py research_004 --validation-only
```

新增步骤使用原余弦日程的末端学习率0.0001。验证审计的 `ready_for_final=true` 后可运行 `final`；完整命令和继续延长的方法见[服务器续训说明](docs/RESEARCH_STUDY.md)。

## 结果状态

2026-10-20：SBL采用同一边缘似然目标的凸上界更新，以归一化一阶条件检查收敛；预算比较强制执行全部更新。确定性只保留无可靠度版本，跳过方差输入变换。[本轮39项检查及完整短流程](results/corrected_sbl_smoke_20261020_v1/report.md)通过，4,032条指标、12次模型重放均一致。正式配置暂缓尺寸扩展和范围偏移，先在原范围重建缓存、训练和比较。

[原SBL排序异常诊断](results/sbl_dictionary_diagnostic_20261019_v1/report.md)确认原单步变化阈值会提前停止。以下历史性能数值均对应原SBL；修正后增益需重新核实。

[修正后的固定观测复查](results/sbl_corrected_probe_20261020_v1/report.md)使用原两个验证几何和四种条件。8,000次更新后全部16个粗/密输出通过一阶条件，密字典比粗字典改善0.455–1.499 dB。4,000到8,000次的条件均值变化最多0.00196 dB。完整64几何和精修模型重训练由新服务器运行核实。

[research_002验证集总览](results/research_002_overview_20261019_v1/report.md)已绘制为一张四联图，提供PNG/PDF/SVG和可复现数据。32×32系统下，标量可靠度确定性精修比粗字典SBL改善0.888 dB；最佳扩散平均16次后改善0.352 dB，但在30 GHz的4种条件均弱于SBL。结果来自64个验证几何和3个训练初始化，最终阶段另用新初始化与筛选几何评价。

[独立续训短验证](results/continuation_smoke_20261019_v1/implementation_verification.md)通过33项检查。父运行6步状态续训到10步，480条验证指标重放、4,608条最终指标重算和15次模型重放的最大差值均为0；CPU/GPU均通过优化器和随机状态的精确续训检查。

[服务器真实断点检查](results/continuation_state_probe_20261019_v1/report.md)成功继承六个扩散状态，仅用一个模型从20,000步更新到20,001步；原运行379个文件哈希保持不变。该记录用于检查恢复接口，完整30,000/40,000步续训按上面的命令另行执行。

[五步流程短验证](results/research_smoke_20261018_v2/report.md)已完成7个阶段、3种小尺寸和9组匹配/失配评价，耗时44秒。29项检查通过，4,608条最终指标重算及15次模型重放完全一致，阶段恢复检查通过。[128×128双样本检查](results/research_shape128_20261018_v2/report.md)也通过两种字典、两种结构及确定性/扩散训练和推理；完整数据量的分位数标定边界已修复并加入回归检查。这些是实现验证，正式多种子实验由服务器运行。

[第二轮可靠度对照](results/deterministic_20261016_v1/report.md)已完成。验证选定SBL上限1000次，新筛选集1013/1024个场景×条件收敛；8条管线共8192条指标，运行763秒。17项检查通过，逐场景指标、来源和模型重放复核一致。

有可靠度确定性残差相对SBL在3 GHz下改善0.75–2.45 dB，30 GHz下改善0.004–0.160 dB。汇总线性NMSE后，无可靠度残差改善0.774 dB，有可靠度改善0.803 dB；可靠度额外收益约0.029 dB，区间包含0。打乱可靠度会使该模型变差，部分高SNR条件的主要作用是避免无效修正。公共观测修正系数在验证集选为0。

[首轮筛选](results/pilot_20261016_v1/report.md)完成8个条件、128个独立筛选几何和11组方法，共11,264条逐场景指标，GPU用时207秒。该轮可靠度残差扩散弱于100次SBL，确定性残差的局部改善促成了上述第二轮完整对照。

[验证集诊断](results/validation_probe_20261016_v1/report.md)另完成1,280条指标：增加SBL迭代明显改善粗估；平均16次生成输出降低误差，但仍未超过原100次SBL。诊断在查看主结果后设计，未用于重选主实验参数。训练数据、模型和完整预测保存在本地runs/。
