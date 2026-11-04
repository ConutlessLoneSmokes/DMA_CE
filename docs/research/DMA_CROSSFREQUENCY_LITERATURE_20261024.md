# DMA跨频硬件响应与信道估计：相邻原文核查

原文核查日期：2026-10-24；讨论更新：2026-10-26。回应用户将研究主线重新聚焦于THz宽带DMA的配置幅相响应。10月24日完成检索、阅读原文和边界讨论；10月26日用户确认将导频阶段的DMA配置设计与信道恢复一起纳入候选范围，并明确按比较目的区分资源口径、优先固定配置和导频、硬件参数设为已知。用户随后采纳先做一次短配置验证并尽快冻结，研究主贡献仍为CE算法。短选择已完成，按预声明保留原随机bank，自检及独立结果复算通过；没有修改物理模型或实现新CE算法。文献部分按 `research` skill 核查一手来源；后续CE设计条件继续由 `grilling` 讨论确定。

## 当前判断

**DMA真实频响、用谐振参数联合控制全频带、以及利用频率选择性加速训练，均已有直接先例。宽带DMA信道估计及DMA专用训练结构也已有工作。** 因此，不能仅将“估计器使用逐频DMA矩阵”“加入Lorentz约束”“配置与估计器联合学习”或“利用频率分集”列为新贡献。

较值得继续核对的交集是：**未知THz近场多径完整CSI，在已标定的真实频变单元与波导模型下，同一组可实现调谐参数必须共同决定全部频点的观测，以明确的训练资源约束联合组织采集与恢复。** 本轮没有确认与这一完整交集完全相同的方案，但也没有完成足以声明新颖性的穷尽检索。特别需要将下面的宽带张量CE、频率选择性单次训练、学习测量设计和真实原型CE放在一起排除重叠。

## 七项直接相关的一手证据

### 1. Ruoyu Zhang等，Tensor-Based Channel Estimation for Extremely Large-Scale MIMO-OFDM With Dynamic Metasurface Antennas

正式版本：IEEE TWC 24(7), 6052–6068, 2025；作者Ruoyu Zhang、Guangyi Chen、Lei Cheng、Xinrong Guan、Qingqing Wu、Wen Wu、Rui Zhang。[正式论文](https://doi.org/10.1109/TWC.2025.3551144)

这是最直接的宽带DMA完整CSI估计近邻。§II式(1)–(8)采用下行发射DMA、常规多天线接收端，空间响应为平面波角度模型。主模型的波导矩阵H与Lorentz圆权重Q没有频率下标。§III-A式(10)–(12)依次激活微带，并重复同一单微带训练矩阵P，形成可分离的训练结构；§III-B式(13)–(20)将多频观测组成四阶CP张量，通过共同角度、时延和路径增益恢复完整G_k。§IV利用Vandermonde结构代数求解，目标是降低完整CSI误差与训练开销。

§VI仿真采用28 GHz载频、320 MHz带宽、128个OFDM子载波；宽带无线信道与DMA主模型的频率平坦假设应分别判断。

关键边界在§VI Fig.11及紧随其后的讨论：作者额外加入频变波导β(f)和互耦，明确指出式(8)仍假定固定β，因模型失配产生高功率误差地板。因此该文已利用DMA配置和多频信道结构，但没有在主恢复器内处理共享谐振调谐产生的真实全带元素响应。这个差异是可定位的研究问题，不足以单独证明新颖性。

获取说明：本轮读取用户已有正式PDF，PDF页序4为式(3)–(8)，5–6为式(10)–(20)，13–14为Fig.11及失配解释。原文件及哈希见[论文目录](literature_notes/catalog.json)中Zhang2025。未找到可信开放arXiv ID；[IEEE页面](https://ieeexplore.ieee.org/document/10938032/)的全文访问可能需要授权，不能把它描述成已核实的开放全文。

### 2. Amarilton L. Magalhães等，Channel Estimation for Downlink Communications Based on Dynamic Metasurface Antennas

采用最新公开v3，2026-03-14；作者Amarilton L. Magalhães、Fazal E-Asim、André L. F. de Almeida、A. Lee Swindlehurst。v1发表于2025-06-14，原题含“Semi-Blind”；当前arXiv说明为Asilomar 2025会议稿。[版本与作者](https://arxiv.org/abs/2506.12639) · [v3全文](https://arxiv.org/html/2506.12639v3)

§II明确假定带内α、β近似不变，波导向量m和DMA波束权重均为频率平坦。其MISO-OFDM观测在频点、数据符号和快速DMA状态间构成PARAFAC模型。§IV式(7)–(12)先用BALS估计无线信道H与X，再以秩一SVD拆出未知波导向量及数据符号。估计目标包含硬件内信道，训练使用符号内多次切换。§III的满列秩F条件涉及P≥N，状态测量数不能与本项目的发送导频RE直接等同。

它是联合无线CE、未知馈线校准与数据检测的直接近邻；OFDM名称并不意味着已处理真实宽带Lorentz谐振共享约束。本项目当前将馈线响应视为已知，若研究未知硬件参数，应把该文作为必要对照。

### 3. Xiangyu Zhang等，Channel Estimation with Dynamic Metasurface Antennas via Model-Based Learning

作者Xiangyu Zhang、Haiyang Zhang、Luxi Yang、Yonina C. Eldar；arXiv:2311.08158v1，2023-11-14。[原文](https://arxiv.org/html/2311.08158v1)

§II使用单个空间信道g、已知波导矩阵H与Lorentz圆权重Q；没有OFDM子载波及共享谐振参数的频率族。§III-B式(16)–(19)和Algorithm 2将Q、稀疏字典及LISTA恢复网络联合训练，DMA层将实相位参数映射到Lorentz可行圆，目标直接使用训练信道恢复误差。部署时先将学到的Q配置到硬件，再用压缩观测推理。

因此“DMA层＋稀疏恢复网络”“训练测量矩阵与估计器联合优化”已被直接提出。可讨论的区别必须落到真实共享调谐的全带可实现性、频变噪声/损耗或采集目标上，不能只增加一个DMA输入层。该文使用单频空间模型，这一分类来自其模型结构，不是将所有DMA学习研究统称为窄带。

### 4. Songjie Yang等，Extremely Large-Scale Dynamic Metasurface Antennas (XL-DMAs): Near-Field Modeling and Channel Estimation

作者Songjie Yang、Wanting Lyu、Boyu Ning、Yue Xiu、Youzhi Xiong、Hua Chen、Chadi Assi、Chau Yuen；arXiv:2407.04954v1，2024-07-06。[原文](https://arxiv.org/html/2407.04954v1)

§II–IV研究接收XL-DMA近场空间CSI，提出Oblong近似、角度解耦及OG-DOLS离网估计；§V专门设计Lorentz约束测量矩阵；§VI采用28 GHz。它已把DMA结构与训练测量设计用于CE，不只是把DMA作为任意已知矩阵。

其主模型为单频近场恢复，没有宽带共享谐振频响族。该文适合作为近场空间处理和测量矩阵优化的对照；“加入Lorentz约束的测量设计”本身不构成当前可认领的新贡献。

### 5. Nitish Vikas Deshpande等，Frequency-selective beamforming and single-shot beam training with dynamic metasurface antennas

作者Nitish Vikas Deshpande、Joseph Carlson、Miguel R. Castellanos、Robert W. Heath Jr.；arXiv:2412.00215v1，2024-11-29。[原文](https://arxiv.org/html/2412.00215v1)

§II–III显式使用频率选择性的单元谐振与波导相位，优化工作频率与各单元谐振配置。§IV-B式(47)–(49)将DMA划分为子阵组，以同一时刻不同子阵的谐振配置同时探测不同角度；接收端从一个OFDM导频中选择最强子载波并反馈，由此估计最优服务方向/工作频率。仿真例为15 GHz、500 m LoS，指标包含角度训练后的波束增益和速率。

它直接证明“利用DMA的频率选择性做并行角度探测”已有先例。主要差异是单用户LoS方向/最佳频率获取，目标并非固定业务频带上的近场多径完整CSI；这一差异需要与其他CE工作共同核对，不能仅换成THz载频就认定为新问题。

### 6. Hanqing Wang等，Dynamic Metasurface Antennas for MIMO-OFDM Receivers With Bit-Limited ADCs

作者Hanqing Wang、Nir Shlezinger、Yonina C. Eldar、Shi Jin、Mohammadreza F. Imani、Insang Yoo、David R. Smith；IEEE TCOMM 69(4), 2643–2659, 2021。[正式论文](https://doi.org/10.1109/TCOMM.2020.3040761) · [作者机构开放PDF](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/ADC.pdf)

§II式(1)–(5)保留元素Lorentz频响及波导滤波；§III-B Algorithm 2联合配置每单元的谐振频率、阻尼/品质因数与振子强度，以同一组物理参数生成全部子载波权重，再联合ADC和数字接收处理降低符号恢复误差。示例为1.9 GHz、40 MHz。

§IV-C明确需要已知多径信道和噪声统计，并把CE阶段的配置研究留作后续。因此真实宽带DMA参数联合控制已有很早的依据，但该算法解决已知CSI下的数据恢复，没有提供未知THz完整CSI估计器。本轮也读取了用户已有正式PDF的§II、Algorithm 2和§IV-C，与开放PDF对应。

### 7. Ioannis Gavras、George C. Alexandropoulos，Wideband Sensing with Dynamic Metasurface Antennas under Realistic Phase Response Modeling

arXiv:2604.11580v1，2026-04-13。[原文](https://arxiv.org/html/2604.11580v1)

§II-B显式保留单元磁极化率、有限谐振调谐和频变波导相位/泄漏；§III–IV推导位置/时延/角度FIM与CRB，分析硬件频响导致的有效信息带宽及孔径变化。§V示例为20 GHz、500 MHz，并明确使用PEB优化的模拟合并器。因此不能把它简化为“只给硬件模型、完全不设计配置”。

其主要结果是宽带参数感知的信息与误差界，空间阵列流形采用平面波响应；不是完整CSI恢复算法。它提示设计时必须同时考虑频率响应差异与有效接收能量，频响变化大并不自动意味着信息更多。

## 另外三个必须知道的边界

- Joseph M. Carlson、Nitish V. Deshpande、Miguel Rodrigo Castellanos、Robert W. Heath Jr.的《Wideband dynamic metasurface antenna performance with practical design characteristics》，arXiv:2510.07827v1（2025-10-09），正式TWC DOI为10.1109/TWC.2025.3613342，已给出有限调谐、频变Lorentz与泄漏模型，并用逐单元谐振配置优化宽带增益/速率。§III、§V已读；该文提供本项目硬件模型依据，不能把真实频响本身再列作新贡献。[原文](https://arxiv.org/html/2510.07827v1)
- Jean Tapie、Bruno Sokal、André L. F. de Almeida、Philipp del Hougne的《Channel Estimation via Tensor Decomposition for Dynamic Metasurface Antennas with Known Mutual Coupling: Algorithms and Experiments》，arXiv:2603.19155v1（2026-03-19），提出已知互耦的BTALS等CE方法。§VI-B明确实验在单一18 GHz进行；因此是校准硬件/互耦感知CE的直接新近邻，尚不等于宽带共享谐振CE。§II–III及§VI-B已读。[原文](https://arxiv.org/html/2603.19155v1)
- Li等《Wideband Hybrid-Field THz UM-MIMO Channel Estimation: A Dual-Attention-Aided Deep-Unfolded Bayesian Learning Approach》，TCOMM 2026，§II与Fig.1明确使用AoSA及每阵元相移器的部分连接混合合并。该文确实处理THz宽带混合场和学习贝叶斯更新，但并非DMA硬件。本轮用用户已有正式PDF核对，不能将它当作共享Lorentz谐振调谐已解决的证据。[正式论文](https://doi.org/10.1109/TCOMM.2026.3675428)

另找到Rezvani与Adve《Channel Estimation for Dynamic Metasurface Antennas》，TWC 23(6), 5832–5846, 2024。IEEE摘要确认其已利用快速DMA切换、长期相关统计和MMSE设计采集，再投影到DMA约束；本轮未取得并核对全文，因此不判断其是否覆盖真实宽带共享谐振条件。[IEEE页面](https://ieeexplore.ieee.org/document/10310000/)

## 本项目讨论时可采用的边界

下述是对论文集合的推断，不是已完成的新颖性证明：最直接的CE近邻具有可分离的频率平坦DMA模型；最直接的真实频变配置训练近邻主要做LoS角度/最佳工作频率；最直接的全带物理参数优化近邻主要假定CSI已知。这为研究它们的交集提供理由，但不能把不同限制拼在一起就自动认领创新。

该候选也有本项目自己的历史：[2026-10-16文献综合](LITERATURE_SYNTHESIS.md)已列出“有限切换下的DMA配置—频点联合分配”，并讨论共享调谐、Deshpande方向训练与固定业务频带多径CSI的区别，以及功率与噪声口径。它一直是待讨论的候选，未被前轮统计融合或几何恢复实验实施，本次重新聚焦不能写成新发现或已确认原创。

还要区分“硬件共享参数”与“信道跨频先验”。若调谐参数u已知、每频信道h_k被作为相互独立的未知向量，且各频噪声条件独立，观测似然仍可按频率分解。u在各频相同，本身不会额外提供未知信道之间的随机相关信息。其可利用价值在于约束可实现观测族、设计互补配置，以及与有依据的信道跨频模型一起组织恢复。

因此，可讨论的具体问题应包含控制变量和评估口径：在比较配置设计时固定总导频RE/能量、RF链数、共享调谐可行域、波导损耗与噪声模型，检验设计出的配置族是否更有利于完整CSI恢复。用户已同意将配置设计纳入范围，但要求快速完成前置选择；本次已冻结原配置，后续CE算法和实验规格尚未确定。

本轮为定向近邻核查，检索词覆盖“dynamic metasurface channel estimation wideband/frequency-dependent/Lorentzian/THz”“tensor CE”“model-based learning”“single-shot frequency-selective beam training”，并对既有18篇目录追原文。未复现论文性能、未完成系统性检索或优先权穷尽审查；IEEE摘要与未取得的全文均明确分开。

## 当前实现的事实对照

当前正式运行并未忽略DMA的跨频幅相响应。[元件与馈线函数](../../src/dma_ce/_upstream/dma.py)对每个状态、每个单元的一组谐振频率，在全部32频点计算Lorentz响应与馈线传播；[观测矩阵构造](../../src/dma_ce/data.py)形成4RF块连接的逐频矩阵，并按状态/RF输出噪声白化。同一调谐状态共同决定整个频带的矩阵，频点权重没有被独立随机化。

当前每个状态的可调变量为各单元谐振频率；品质因数Q=50和馈线参数固定。配置向量u_j同时生成全部频点A_{j,k}(u_j)，不能将各A_{j,k}当成可独立任意设计的复矩阵。

| 实现对象 | 当前已经使用 | 当前没有进行 |
|---|---|---|
| 逐频DMA矩阵 | LS、LMMSE、SBL及物理拟合均使用真实A_k；SBL有效字典为A_kD_k | 没有比较频率平坦近似与真实矩阵的同观测恢复差异，不能量化频响建模本身的收益 |
| 硬件调谐状态 | 固定种子均匀抽样谐振频率，跨频点/场景/SNR共用 | 没有根据宽带CSI误差、观测互补性或资源代价优化状态库/状态选择 |
| 跨频恢复 | SBL共享原子方差；LMMSE有训练联合协方差；DDPM通过粗估与末端一致性使用A_k，CNN可跨频卷积 | DDPM网络没有显式的矩阵或谐振参数输入，也未验证跨配置适配 |
| 上轮物理结合 | 拟合共同传播几何/路径系数，真实A_k固定 | 没有重新设计DMA配置，不能当作已检验硬件频率编码价值 |

源码定位：[随机状态库](../../src/dma_ce/physical_study.py#L74)、[逐频SBL后验](../../src/dma_ce/sbl.py#L18)、[共享方差更新](../../src/dma_ce/sbl.py#L65)、[联合LMMSE](../../src/dma_ce/factor_lmmse.py#L16)、[网络条件接口](../../src/dma_ce/conditional.py#L30)。本轮只读审计还确认DMA复用文件与[来源清单](../UPSTREAM_REUSE.json)的目标哈希一致。

硬件假设也应限定：当前馈线采用常数有效折射率1.4，因此保留随频率线性变化的相位，固定末端功率比例决定频率不变衰减；没有加入实测频变折射率、频变馈线损耗或互耦。不能把“已使用频变矩阵”扩写成全部真实硬件效应均已覆盖。

若后续改变配置，公平性还有一个必须处理的细节：[现有接收SNR校准](../../src/dma_ce/physical_study.py#L99)会用配置对应的训练平均接收功率反推导频发射功率。不同配置若重新对齐相同接收SNR，发送能量一般也会变化。2026-10-26用户明确：口径取决于比较目的，不在全项目统一二选一。比较配置设计时，在同一资源点固定发送能量、导频符号和频点；配置固定后比较CE方法时，可以改变共同发射功率扫描接收SNR。后者是正常的方法比较，不限于附加诊断。沿用现有定义时，SNR由训练平均接收功率标定，各方法共用对应功率、观测和噪声，不利用筛选场景真值逐场景调整功率。

## Grilling讨论树与本轮边界

- 用户已明确的主线：研究THz频段的DMA信道估计，传统LS/LMMSE/SBL保留为有意义基线，围绕DMA配置的跨频幅相响应核查可研究的问题。
- 本轮已完成的事实调查：现有实现如何使用频响；同类训练/估计/波束/感知工作是否存在；最相近原文的假设和目标差异。
- 已采纳Q1，2026-10-26用户回答“是”：将导频阶段可实现的DMA配置设计与信道恢复一起纳入候选范围。估计器形式未预定，该回答不代表采纳具体算法或启动实验。
- 已采纳Q2：按比较目的分别设定口径。配置设计比较固定发送能量、导频符号与频点；固定配置的CE方法比较允许改变发射功率绘制接收SNR曲线。在每个配置/SNR点内，各CE方法共用同一组物理观测、噪声与已知信息。不同采集方案按相同发送资源比较，可以产生不同观测。
- 已采纳Q3：优先固定DMA配置序列与导频，对不同几何场景保持一致，不根据当前场景重新选择配置；场景变化体现为信道变化。各方法的场景保持配对。首轮沿用每个OFDM导频符号内固定一个状态、同一调谐参数共同生成全频响应的讨论范围；具体设计出的状态序列未定。
- 已采纳Q4：单元与波导等硬件频响参数设为已知，暂不联合估计未知硬件参数，以控制问题复杂度。这是仿真中的已知模型条件，不代表完成真实设备标定，也不代表全部频变损耗或互耦已被建模。
- 已采纳Q5：用户赞同先单独检验配置设计的价值，但要求尽快确定配置，主线保留为信道估计算法。配置探查只作有预算上限的前置选择，不扩展为配置优化研究，也不把一组固定数值配置列作论文主贡献。产生配置的通用设计规则与其输出配置属于不同对象；本轮仅使用简单规则完成选择。
- 下一阶段讨论信道估计的核心机制及与近邻工作的差异；具体估计器、训练方式与性能范围尚未预定。

## 2026-10-26一次短配置选择：运行前约定

用户已同意先验证配置，要求较快冻结。本轮限定两个候选：原固定随机bank；每个状态、每条微带用已有 `frequency_tiled_resonances()` 均匀覆盖285–315 GHz并按固定种子置乱阵元对应的bank。它是无学习的简单覆盖候选，不宣称最优性或新颖性；不追加配置种子、候选或调参扫描。

- 复用父运行 `multirf_snr_20261022_v1` 的4×256/4RF、J=32、K=32、512训练CSI、前32个验证几何与训练归一化；不读取筛选集输出用于配置选择。
- 取原−10/10/30 dB三个点已标定的发射功率，两配置共用这些发送功率、1024导频RE与物理噪声模型。候选实际训练平均接收SNR作为结果记录，不重新对齐。
- 真实逐频矩阵及噪声增益随bank重建。用相同标准化白化噪声实现配对；这不表示两种模拟合并器共用同一组阵元噪声潜变量。LS伪逆与LMMSE因子随矩阵重算，LMMSE收缩沿用父运行已冻结的0.9，不为新配置另扫参数。SBL统一固定100步，未达一阶阈值如实保留。
- 选择依据是完整CSI的mean scene NMSE。保守规则：只有均衡覆盖候选在三个功率点的SBL100配对改善量，其场景bootstrap描述性95%区间下界均大于零，才替换原随机配置；否则保留原配置。LS与LMMSE结果作为辅助对照。该规则服务于快速冻结，不是论文性能验收门槛或新颖性证明。
- 验证选择仅限原验证集，描述性区间不构成独立确认。后续论文性能须在冻结配置后以独立数据评价。本轮不新建筛选集，不将已参与选择的验证误差当测试结果。
- 单次计算上限600秒，超时保留原配置并标明比较未完成。运行ID `configuration_probe_20261026_v1`，配置 `configs/configuration_probe.json`，入口 `scripts/probe_configurations.py`；完整配置、源码、种子、环境、输入来源与逐场景结果存入独立运行目录。

## 2026-10-26短选择结果

运行 `configuration_probe_20261026_v1` 完成，计算155.33秒。均衡覆盖候选在原−10/10/30 dB所对应的共同发送功率点，SBL100平均改善0.0407/0.0421/0.0225 dB，但配对描述性95%区间均跨零；按预声明保留原随机bank，结束配置搜索。后续CE实验采用原4×256/4RF、J=32、K=32及同一状态库，配置只作共同实验设置。[报告、完整口径及限制](../../results/configuration_probe_20261026_v1/interpretation.md) · [冻结与状态哈希](../../results/configuration_probe_20261026_v1/frozen.json)。本轮只用验证数据作选择，不构成独立性能确认或对全部配置设计方法的否定。

上述讨论树记录研究边界；本次限定的一次短配置选择已经完成，后续CE算法仍待讨论。仅改变载频、再次写入已知A_k、或在现有估计器前增加DMA层，都不足以由本轮证据确认为新贡献。
