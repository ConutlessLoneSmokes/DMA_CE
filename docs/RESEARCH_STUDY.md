# 五步实验与服务器运行

本次实现的是完整实验流程；本地只运行小规模验证。正式配置为 `configs/research.json`，短验证配置为 `configs/research_smoke.json`。

## 运行方法

在服务器的 DMA_CE 仓库目录中，使用已配置好 CUDA 版 PyTorch 的 Python 环境运行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_research.py --config configs/research_smoke.json --run-id server_smoke_001
python scripts/audit_research.py server_smoke_001
python scripts/run_research.py --config configs/research.json --run-id research_001
python scripts/audit_research.py research_001
```

`--stage` 默认是 `all`，也可以依次运行：

```bash
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage baseline
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage curves
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage factorial
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage reliability
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage budget
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage diffusion
python scripts/run_research.py --config configs/research.json --run-id research_001 --stage final
```

同一命令可恢复已完成阶段和训练断点。每个阶段运行前检查所有前置阶段；配置、父运行或 `src/`、`scripts/` 中的执行源码有变化时必须使用新运行 ID。每个阶段的超时上限由 `timeout_seconds` 指定。失败保留状态、错误和最近的优化器、批次及扩散随机状态。

### 从 research_001 延长扩散训练

服务器的 `research_001` 已完成前五阶段，六个扩散模型保存在20,000步。更新代码后，在原仓库、原 CUDA 环境执行：

```bash
git pull --ff-only
python scripts/run_research.py --continue-from research_001 --diffusion-checkpoints 30000 40000 --run-id research_002 --stage diffusion
python scripts/audit_research.py research_002 --validation-only
```

`--diffusion-checkpoints` 只列新增预算。程序从父运行读取配置，逐字节复制训练/验证缓存、模型和前五阶段结果到新 ID，保留原始20,000步优化器与两个随机数生成器的快照。恢复已有新 ID 时使用相同命令。父运行保持原样；新运行保存 `continuation.json`，审计核对继承文件哈希和旧日志前缀。复制中断且尚未生成该来源记录时，改用新的 ID。

原始20000步余弦日程和所有旧检查点保持不变；新增训练在其末端学习率继续。正式配置为 `0.001 × 0.1 = 0.0001`，不重新启动余弦衰减。确定性训练预算保持原值。最终新初始化的扩散模型也按同一原始日程加恒定末端学习率训练到验证选定步数。

`validation_audit.json` 中 `status=passed` 表示来源、断点和验证重放通过；`ready_for_final=true` 表示扩散阶段已通过原稳定性规则。此时继续最终阶段：

```bash
python scripts/run_research.py --continue-from research_001 --diffusion-checkpoints 30000 40000 --run-id research_002 --stage final
python scripts/audit_research.py research_002
```

若最大检查点仍改善超过0.1 dB，程序记录 `budget_limited`、保存验证表格和执行报告并以状态码2停止。验证审计仍可核对已完成产物，完整审计会返回 `not_ready`。需要再次延长时，以尚未开始最终评价的 `research_002` 为父运行，使用更大的新增预算和新 ID。已创建最终冻结记录或筛选预测的运行不能作为续训父运行。

若服务器另有原第二轮的 `runs/deterministic_20261016_v1/data.pt` 和 `config_resolved.json`，可在第一次运行时附加 `--parent deterministic_20261016_v1`，之后每次恢复也保留这个参数。它只取训练和验证观测；物理系统、场景种子和样本数必须完全匹配。Git 不包含这些缓存，未提供父运行时程序会自行生成。

正式配置保留原有512训练、64验证、128筛选几何，以便复用缓存。扩大数据量时修改 `data` 后使用新 ID。配置中的 `learning.steps` 保持兼容旧入口，新流程的实际训练预算来自 `research.checkpoints` 及阶段选择。

## 阶段与控制变量

| 计划 | 阶段 | 实际处理与输出 |
|---|---|---|
| 1. 基线与训练预算 | `baseline` | 同一验证观测比较49×7与97×13字典；逐字典检查1000/2000/4000/8000迭代的收敛率、完整CSI NMSE及平均后验方差稳定性 |
| 1. 基线与训练预算 | `curves` | 3个初始化，分别训练零可靠度与方差图；一条20000步轨迹保存2000/8000/20000步模型，导出训练损失及训练/验证CSI曲线 |
| 2. 宽带瓶颈 | `factorial` | 跨频二维卷积/逐频点一维阵元卷积 × 统一损失/条件残差MSE归一化，每格分别训练无可靠度与完整图；3个初始化 |
| 3. 可靠度形式 | `reliability` | 最佳结构与损失下比较零、样本平均方差、完整图；对完整图做5个固定种子的条件内样本置换 |
| 4. 精度与计算量 | `budget` | 对100/300/500/1000次及充分迭代SBL建立各自匹配的精修器，重算各自训练标定，导出误差—SBL加精修时间图 |
| 4. 生成模型比较 | `diffusion` | 固定相同粗估、表示、结构和损失，比较训练检查点及1/4/16次输出平均；评价精度、调用数和总时间 |
| 5. 扩大与通信 | `final` | 冻结小系统选择，用5个新的初始化复核32×32、128×32、128×128；每个系统重新训练、标定，再冻结模型后生成新筛选几何，评价匹配条件、路径数和距离失配及净速率 |

所有训练曲线采用相同的20000步余弦学习率日程；提前停止保留长训练的相同前缀。初始化种子和抽取批次的种子独立于几何种子。确定性和扩散模型使用相同的批次序列；扩散加噪使用另一个随机数生成器。训练时只把当前批次送到GPU，避免把128×128的整个训练缓存放入显存。

逐频点网络使用3×1卷积，跨频网络使用3×3卷积，输入图像轴为阵元×子载波。逐频点宽度取参数量最接近原网络的整数宽度；`factorial_selection.json` 记录实际参数量。两种可靠度版本在同一结构下参数量完全相同。

“样本平均方差”先对该样本所有阵元和频点的原始后验方差取平均，再应用冻结的训练集标定和对数标准化，并广播回输入尺寸。“完整图”保留逐分量后验方差；“零”使用同尺寸零占位。条件损失权重等于全训练集平均粗估残差MSE除以该条件训练残差MSE，验证与筛选真值不参与标定。

## 选择、停止与新场景

充分迭代SBL要求前一个预算总体收敛率≥95%、每条件≥90%，增加预算后每条件NMSE变化≤0.1 dB、平均方差相对变化≤5%。粗、密两种字典分别判断，未达到阈值就保存表格并停止，不会将预算用完称为收敛。

确定性训练预算按两种可靠度及3种初始化的验证集线性平均NMSE选择。二维对照的结构和损失也按无可靠度、完整图及种子的平均验证误差选择，避免只为某一可靠度版本选结构。可靠度形式随后单独按验证选择。SBL精修预算选择验证误差距最优精修≤0.1 dB的候选中，实测总均摊时间最短者；表格同时保留单独SBL，最终报告可据此判断精修是否有计算价值。

若最优训练检查点落在最大预算，且相比倒数第二检查点仍改善超过 `training_stability_db`，相应阶段保存 `budget_boundary.json` 并记录 `budget_limited`。正式配置阈值为0.1 dB。扩散训练可按上面的独立新 ID 续训入口延长；确定性训练的预算变化需要另建研究配置。最后区间始终指倒数第二检查点到末点，即使验证最优点位于更早位置也按此定义报告。短验证使用放宽阈值和极小迭代数，只检查执行逻辑。

最终5个训练初始化与选择阶段的3个初始化不重合。最终保留充分迭代粗/密字典SBL、预算匹配SBL、无可靠度/完整图确定性网络，以及选定可靠度形式；扩散保留无可靠度与验证选定形式的同检查点模型，并报告1/4/16次平均。观测一致性系数在各系统的无可靠度确定性验证预测上选择，由该系统全部方法共用，原输出和修正输出都记录。

每个系统的 `frozen_validation.json` 在读取筛选真值前保存模型哈希、训练缓存哈希、可靠度标定、方法和筛选种子。失配只改变真实信道生成；估计器继续使用训练系统的字典范围、噪声尺度和标定。原有筛选场景不进入新阶段选择。

阵元增加时，传播距离乘以孔径平方比，保持距离与瑞利距离的比例；128阵元下距离约为1.68–6.71 m。扩展子载波数保持载频和带宽不变。各扩展系统的充分迭代基线仍需在其自身验证集重新通过稳定性检查。

## 指标与输出

`runs/<id>/` 保存完整缓存、断点、预测、源码快照、来源清单和逐阶段状态。`results/<id>/` 保存配置、源码快照、逐场景表、选择结果、图和执行报告；不导出模型及原始训练缓存。

- 主误差先对线性NMSE取平均再转dB。逐条件表保留种子，避免把同一几何的多个条件当作独立重复。
- 每条指标记录同观测哈希、SBL收敛状态、迭代数、观测残差、行空间/零空间误差、修正能量和与真实残差的对齐。
- 最终配对区间同时重抽训练种子和几何场景；同一几何的条件、方法始终一起抽取。有限种子数量会限制区间稳定性。
- 时间为预热后的批量均摊时间，包含SBL及精修的数据传送和计算；字典、已知硬件矩阵及网络预热不计入在线时间。它不是单样本时延。对照应在同一服务器、相同批量大小下顺序运行。
- 训练曲线的训练侧NMSE用每条件固定前128个训练几何，验证侧使用完整验证集；优化损失日志来自实际训练批次。

通信评价对应当前单微带、单RF链模型。每个方法用其估计CSI，在本条件的J个可行DMA状态中选一个状态，整个数据块的全部子载波共用该状态；以真实信道计算该状态的平均 `log2(1+SNR)`。净速率乘以 `max(0, 1 - [J + switch_symbols×(J-1)]/coherence_symbols)`。默认相干块200个OFDM符号，单次切换开销0.1符号，两者都在配置中显式记录。完美CSI状态选择仅为评分上界。

`scripts/audit_research.py` 检查源码和模型哈希、训练集标定、几何划分、同观测、冻结时间，并从保存预测重算完整CSI误差、残差分解、通信速率、配对区间及固定模型重放。短验证通过只证明实现可执行与这些一致性检查通过，不构成论文主结果。

本地已完成29项检查，包含CPU与GPU的训练前缀/续训一致性、父缓存复用及物理配置失配拒绝、超过16,777,216项的可靠度标定。完整短验证通过9组场景、4,608条指标和15次精确重放，记录见 [research_smoke_20261018_v2](../results/research_smoke_20261018_v2/report.md)；完整128×128维度的双样本记录见 [research_shape128_20261018_v2](../results/research_shape128_20261018_v2/report.md)。正式配置的长训练、5种初始化性能和规模扩展比较留待服务器运行。

2026-10-19补充独立续训后，33项检查通过。实际入口完成“父运行6步→新运行10步→3种小系统最终评价→同ID恢复”，480条验证指标、4,608条最终指标和15次模型重放全部核对一致；完整记录见[续训实现验证](../results/continuation_smoke_20261019_v1/implementation_verification.md)。
