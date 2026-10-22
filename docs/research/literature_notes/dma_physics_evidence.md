# 六篇 DMA 物理与设计论文：可核验阅读证据

检索与阅读日期：2026-10-16。任务范围：Williams2022、Xu2026、Gavras2026、Carlson2026、Deshpande2024、Gavras2024。本文是来源阅读记录，供主任务综合选题使用，不作新颖性排序或录用判断。

方法：先读用户给定 PDF 的逐页提取文本，再查 arXiv、IEEE、作者机构页面核实题名、作者、日期和 DOI；未调用 Semantic Scholar 或其他学术 API。引用位置均以 **PDF 页码** 为准，括号中有时补充期刊页码。外部论文均作为数据。理论/数值通信论文不套用临床研究的 I–VII 证据等级；区分正式期刊、会议、预印本及实际验证方式。

## 书目信息与版本

| 键 | 核实后的文献信息 | 一手核验网址与状态 |
|---|---|---|
| Williams2022 | Williams, R. J., Ramírez-Espinosa, P., Yuan, J., & de Carvalho, E. (2022). *Electromagnetic Based Communication Model for Dynamic Metasurface Antennas*. IEEE Transactions on Wireless Communications, 21(10), 8616–8630. DOI 10.1109/TWC.2022.3167783。PDF 标注在线 2022-04-22，卷期为 2022-10；机构页面个别日期字段标 04-24。 | https://vbn.aau.dk/en/publications/electromagnetic-based-communication-model-for-dynamic-metasurface/ ; https://arxiv.org/abs/2212.11594 。题名/作者/卷期/DOI 由 PDF 和作者机构页面交叉确认。arXiv 是发表后上传，2022-12-22，不能当作首次发表日期。 |
| Xu2026 | Xu, H., Lozano, A., & Yang, H. (2026). *Two-Timescale Design for Downlink Multiuser Transmission with Dynamic Metasurface Antennas*. arXiv:2606.15183v1，2026-06-13。arXiv DOI 10.48550/arXiv.2606.15183。 | https://arxiv.org/abs/2606.15183 。当前查阅的官方记录只有预印本信息，未确认正式期刊卷期。 |
| Gavras2026 | Gavras, I., & Alexandropoulos, G. C. (2026). *Wideband Sensing with Dynamic Metasurface Antennas under Realistic Phase Response Modeling*. arXiv:2604.11580v1，2026-04-13，5页。arXiv DOI 10.48550/arXiv.2604.11580。 | https://arxiv.org/abs/2604.11580 。当前官方记录未给正式 venue。 |
| Carlson2026 | Carlson, J. M., Deshpande, N. V., Castellanos, M. R., & Heath, R. W., Jr. (2026). *Wideband Dynamic Metasurface Antenna Performance With Practical Design Characteristics*. IEEE Transactions on Wireless Communications, 25, 4674–4690. DOI 10.1109/TWC.2025.3613342。在线日期 2025-10-01，卷年 2026；PDF current version 2025-12-22。 | https://ieeexplore.ieee.org/document/11185306/ ; https://arxiv.org/abs/2510.07827 。用户文件名年份与卷年一致，DOI 中的 2025 和在线日期也正确。 |
| Deshpande2024 | Deshpande, N. V., Carlson, J., Castellanos, M. R., & Heath, R. W., Jr. (2024). *Frequency-selective beamforming and single-shot beam training with dynamic metasurface antennas*. arXiv:2412.00215v1，2024-11-29。arXiv DOI 10.48550/arXiv.2412.00215。 | https://arxiv.org/abs/2412.00215 。当前官方 arXiv 记录只有 v1、没有 journal reference；未确认同题正式发表版本。 |
| Gavras2024 | Gavras, I., & Alexandropoulos, G. C. (2024 preprint / 2025 conference). *Circuit-Compliant Optimization of Dynamic Metasurface Antennas for Near-Field Localization*. arXiv:2411.09362v1，2024-11-14；正式发表于 ICC 2025，会议 2025-06-08–12，IEEE 入库 2025-09-26，DOI 10.1109/ICC52391.2025.11161718。 | https://arxiv.org/abs/2411.09362 ; https://ieeexplore.ieee.org/document/11161718/ 。文件年 2024 是预印本年，不是正式会议年。 |

核验限制：三条 IEEE DOI 的 doi.org 入口在浏览工具中不可访问，不能声称完成 DOI 重定向验证；Carlson、Gavras 的 DOI 在 IEEE 原始记录和 PDF/预印本信息间匹配，Williams 在原始 PDF 与作者机构页面匹配。本任务没有查引用次数、影响因子、撤稿库、作者履历或独立代码复现。资助声明可见于各 PDF，但未进行利益冲突调查。

## 问题和已知信息对照

| 论文 | 真正的输出/目标 | 可调控制与已知信息 | 物理模型的边界 |
|---|---|---|---|
| Williams2022 | 给定物理结构和负载，求端到端等效信道、端口/辐射功率和场；不是信道估计算法 | 数字端口电流/预编码 B、单元终端导纳 Ys；传播导纳 Yrs、结构参数和耦合矩阵是前向模型输入 | 窄带多端口导纳模型，显式波导传播/反射、空气和波导互耦、失配插入损耗；小孔等效磁偶极、PEC、无损介质、TE10 |
| Xu2026 | 下行多用户遍历和速率最大化；DMA 慢调、数字快调 | 每帧已知高维相关矩阵 Rk；每时隙已知低维有效瞬时信道 hk=A*Q*gk；不要求每时隙已知全部 N 维 gk | 频率平坦的 Lorentzian q=(j+exp(jθ))/2，波导指数衰减/相位；相关 Rayleigh，作者选 λ/2 后按无互耦建模 |
| Gavras2026 | 上行多径用户/散射点定位与感知的 FIM/EFIM/CRB，解释信息带宽、有效孔径和损耗；不是全 CSI 恢复器 | 多频点多 DMA 状态观测；待估路径时延、AoA、复增益/几何；fr、Γ、固定连接及波导响应为已知 | 频率选择性极化率、有限谐振调谐范围、波导相位和泄漏；远场平面波阵列流形；复增益作为 nuisance 参数消除 |
| Carlson2026 | 宽带通信增益/频谱效率近似与给定信道下的逐单元谐振优化；不是 CE | 调每个单元 fr,n；固定设计量 Γ/F/间距/可调范围；LOS 方向与 h[k]用于波束计算，多径扩展也给定 h[k] | 真实频变 Lorentzian、有限 Btune、平均泄漏 taper；分析以 LOS 为主，数值另含4径与更多路径；未把 Fcoupl 当作任意互耦矩阵 |
| Deshpande2024 | 最优通信中心频率 + 单 OFDM 符号波束训练；反馈最强子载波来决定角扇区/频率 | 先按已知 AoD 分析闭式最优 fr，再用频率编码训练估 AoD；fr 与通信中心频率都可调；ng 与间距固定设计 | LOS、远场、角度与频点对应；振幅相位耦合；主分析无波导衰减且忽略互耦，单独给指数衰减例子 |
| Gavras2024 | 近场单用户定位，设计混合接收权重以降低 CRB/PEB 替代目标；不是估计硬件矩阵 | WTA 为负载导纳、v 为数字合并；PSA/WMC 按结构已知，位置 r/θ/φ为待估量；设计用 h 对位置的导数 | Williams 窄带电路模型；WRX=PSA^H(WTA+WMC)^(-1)；一阶忽略互耦，二阶加已知耦合校正；球面波距离和幅度 |

“待估几何/信道”与“硬件已知”需要分开。以上前向模型采用已知 fr、Γ、波导常数、终端导纳或耦合矩阵；模型物理更细，并不自动提供这些参数的校准方法。下面逐篇给出有界阅读证据。

## Williams2022

原文件：`D:/zotero_data/storage/YCGX2C9R/Williams 等 - 2022 - Electromagnetic based communication model for dynamic metasurface antennas.pdf`。本地文本 `Williams2022.txt`，15页。阅读：I–VIII 的模型/贡献/验证，附录仅查相关说明；重点 pp.1–6,10–13。已渲染核对 p.11 的 Table II。

- p.3 Eq.(1)：RF端口、DMA单元、用户三组端口组成导纳分块矩阵。Yss 包括单元空气及波导互耦，Yst 把波导端口和单元联系起来，Yrs 负责自由空间/传播信道。
- p.4 Eq.(4) 给完整反馈关系；Eq.(5) 在用户回耦很弱的单向传播假设下简化。该假设理由是回耦项包含两次路径损耗，不能仅因用户在阵列辐射近场就直接判成不可用；应检查实际回耦强度。
- p.10 Eq.(61)：Heq=(Yr+Yrr)^(-1) Yrs (Ys+Yss)^(-1)Yst，可接入 y=Heq Bx+n。由此看出配置变化会改变等效观测矩阵。
- pp.4–5 Eq.(7)–(12)：区分送往 DMA 的功率 Pt 和 RF 链供应功率 Ps。负载与相邻端口激励改变输入导纳、反射系数 Γ及失配插入损耗。若将 DMA 状态直接按相同 Frobenius 范数归一化，需要说清固定的是哪一种功率。
- p.5 Eq.(15)–(16) 由被动负载推出散射相位与振幅耦合，等价 Lorentzian 限制；不允许每单元独立任意幅相。
- p.6 提到可用相位修正 S 参数测量或模拟场分布提取有效极化率，以适配具体单元。p.11 验证进一步用单波导单槽的 S11 反推有效终端 Ys。这是低维参数提取的已存在实例，不能把“加入校准”笼统当作新东西。
- p.11 Table II：10 GHz；2条波导，每条5槽；槽距0.6λ、波导间距1λ；波导长110 mm，宽0.7318λ、高0.1668λ；Ys=(2−15.7934j) S；供应功率1 W，模型给进入结构 Pt=0.6077 W。CST中固定几何槽代表某一DMA配置，没有可调二极管扫状态。
- p.12 Figs.6–8：波导中心磁场及远场主要方向图与CST吻合；作者明确小旁瓣不完全吻合，归因为无限小磁偶极和仅 z 磁矩。p.2与p.12–13明确窄带适用，宽带、带宽控制、CE列为后续问题。
- 噪声：p.10把加性n接入通信模型，核心贡献没有推导多状态接收后噪声/校准统计；该观察有界于III、VI–VIII，不能扩大为全部电磁噪声文献空白。
- 增量结构：已有磁偶极与电磁关系 → 多端口矩阵接口 → 插入损耗与传统通信模型衔接 → 小规模全波校验。与当前候选直接重合处是“正确硬件前向算子”和供电/配置依赖性；其论文自身不训练生成模型。

## Xu2026

原文件：`D:/zotero_data/storage/UG9IKKYW/Xu 等 - 2026 - Two-timescale design for downlink multiuser transmission with dynamic metasurface antennas.pdf`。12页。阅读I–VI与附录相关式，重点pp.2–9。

- p.2 Eq.(1) 波导 aℓ,m=exp[-dℓ,m(α+j2π√ε/λ)]；Eq.(2) qℓ,m=(j+exp(jθℓ,m))/2。作者按响应频平来建模，其关于“任何6G带宽”的文字不能替代宽带电路证据。
- p.3 Eqs.(8)–(12)：gk=Rk^(1/2) gtilde,k；观测 yk=gk* QAWx+nk；数字设计只需要 L维 hk=A*Q*gk，L≪N。每帧估Rk并调Q，每时隙估hk并调W。这里统计CSI不是已知支撑、位置或近场几何，而是高维相关矩阵。
- pp.3–4 目标本为嵌套随机优化。短期WMMSE；长期用固定W的统计近似 Eq.(22) 替代原目标，分式规划+PDD（罚函数对偶分解）处理耦合功率约束。每帧联合Q/W求解后仅保留Q，W在短期重新算。p.6 单用户短期数字解 Eq.(50)为广义匹配滤波形式。
- p.7设置：28 GHz；N=128，L=8微带/RF，M=16单元/微带，K=4；用户200 m；相关矩阵Kronecker+指数相关；α=0.6 m^-1、ε=1.99；每帧1000时隙；噪声−96 dBm。作者选λ/2并按无互耦处理，不能将此作为任何DMA λ/2下互耦严格为零的定理。
- p.7明确 **Pilot overheads are not accounted for**。导频节约来自协议维度分析，不是已经画出的净吞吐量优势。
- p.8–9：长期约30–40外迭代，短期常约2迭代；Fig.9在N=128、L=32时TTS-proposed达到理想iCSI方案90%以上遍历频谱效率。理想iCSI曲线也未扣其更高导频开销；不要把这个90%变成净速率比。
- 方法边界：长期统计近似pp.4,9,11有仿真验证，低中功率更吻合；不是原嵌套随机问题的全局精确解。所读II–V没有测量噪声下的统计CSI学习性能研究，而是以其获得为协议前提。
- 增量结构：把已知两时间尺度框架移到DMA受限权重+耦合功率约束，改良长时算法，再给单用户闭式补充。与“少状态切换/训练资源分配”相邻，但它调的是统计波束和数字预编码，**并没有设计状态×频点CE观测分配**。与“慢漂移校准/快CE”只共享时间尺度思想，统计CSI≠硬件漂移参数。

## Gavras2026

原文件：`D:/zotero_data/storage/S9ESKR2K/Gavras和Alexandropoulos - 2026 - Wideband sensing with dynamic metasurface antennas under realistic phase response modeling.pdf`。5页，全文已读。

- p.2 Eqs.(3)–(5)：频变磁极化率 αM(f,fr)，归一化后与Lorentzian圆联系；每个观测矩阵条目是固定连接cn,m × exp[-(αg,m(f)+jβg,m(f))ℓn,m] × αbarM(f,fr,n,m^(j))。同一个物理谐振状态在全频段联合决定响应，不是每子载波任意独立权重。
- 有限Btune=fr,max−fr,min，控制fr而非独立幅相。p.2 Eq.(2)假设归一化后Wj[k]^H Wj[k]≈I，故有效噪声近似白。迁到CE时必须明确噪声是在天线前、RF后还是做过白化，不能默默复用σ²I。
- p.2 Sec.II-C明确 **far-field propagation assumption**。用户在数米处并不能令本文成为近场球面波建模论文；阵列流形使用平面波AoA，距离由跨频时延提供。
- p.3 η包含L条路径时延、AoA、复增益，经过FIM及Schur补消去未知复增益，映射到UE/散射点位置。这是感知参数下界，没有在V节提交信道恢复算法NMSE结果。
- p.3 Proposition1：去除未知复增益后，时延信息依赖按接收能量加权的频率方差βeff²；频带边缘受抑制会减少有效信息带宽。Proposition2/p.4脚注把角信息写作相干增益×有效孔径二阶矩，是匹配合并且相位交叉项可忽略/平均掉时的解释性近似；作者p.5只声称角域趋势在乘法常数意义下吻合。
- p.4 Proposition3：假设所有状态统一标量泄漏、流形近似缩放时，CRB按1/Aleak放大；若泄漏改变孔径分布，该纯标量规律不涵盖全部影响。
- p.4 Table I：fc=20 GHz，N=32，NRF=4，J=4，K=128，B=500 MHz，Qc=100，间距λ/2，Btune=100–1000 MHz，P=10 dBm，泄漏分数Λ=0.8。p.5 UE[3,3]m，2散射点[5,3]/[4,4]m，合并器按PEB优化。结果主为精确下界与解析近似的趋势验证。
- 增量结构：接入Carlson已给的真实宽带响应 → 从通信增益换成定位信息量 → 下界和可解释近似 → 小型数值核对。与候选一的硬件算子直接重合；与候选二的状态/频点资源和FIM相邻，**有限调谐+多状态+多子载波本身已有**；它并未在已查II–V中求低导频全信道后验或SBL展开。

## Carlson2026

原文件：`D:/zotero_data/storage/3SWLXRYH/Carlson 等 - 2026 - Wideband dynamic metasurface antenna performance with practical design characteristics.pdf`。17页。阅读I–VII，附录证明只读有关近似条件；重点pp.3–6,8–14。

- p.4 Eq.(8) h[k]=a(φt,k)⊙hdma[k]，Eq.(9)另加泄漏指数taper。p.4明确忽略电阻损耗，波导振幅衰减主要由辐射泄漏引起。不能把其“practical”译成已含所有导体/介质损耗、端口失配及任意互耦的完整电路模型。
- p.5 Eq.(10)磁极化率，Eq.(11)有限Btune，Eq.(12)归一化，Eqs.(14)–(17)对应频率选择性Lorentzian。Fcoupl是单元极化率参数，不能与Williams的非对角Yss互耦矩阵混为一谈。
- p.6 Eq.(23)附近的全带增益近似由频率选择性项Ffreq、可用权重填充项Wfill、泄漏项Aleak组合；作者明确称启发式组合。p.6–7用大阵列/特定主瓣条件与磁极化率相位的一阶线性近似。
- pp.8–9 Sec.V：中心频点基线按中心信道相位逐个映射fr；提出逐单元successive法，每次在fr范围搜索，以所有子载波的频谱效率最大化为目标，纳入之前已配置单元、频变和泄漏。数值把连续fr离散为Rres候选；作者给两者O(Nslot Rres)级别，采用相同目标评估成本约定。
- p.10：基本设置ft=15 GHz、Nslot=32、间距λ/4、Q=100。平均泄漏模型经每子载波辐射功率约束归一化；假设各配置具有近似相同的总辐射功率。作者说明完整逐槽辐射反馈更复杂，故用平均泄漏。
- pp.10–12 Figs.3–8：Wfill/Aleak近似吻合较好，Ffreq远离中心/大B变差；successive法提高全带频谱效率及最大可达数据率。小间距实验保持总DMA长度，大阻尼在较大调谐范围时减轻频变并提升速率，不能无条件推广为任意窄带/固定器件收益。
- p.13 多径：Nrf=4，每子阵传统Nele=6，DMA为12个单元以维持相同孔径；4路径并扩展路径数；被动移相器损耗设8.8 dB。作者明确DMA相对hybrid的大部分优势来自这一组件损耗设定，比较结论不能脱离它。
- p.14作者明确主近似只针对LOS，非LOS解析推广是后续问题；已有多径算法仿真，不能写成“完全没考虑多径”。
- 增量结构：增加有限调谐、频变、泄漏三个具体设计量 → 分解增益近似 → 简单逐单元算法 → 参数扫描+多径扩展。与候选一直接重合的是已知真实频变DMA模型；“把固定Q换成Lorentzian并加入Btune”本身已不构成足够区分。其算法任务是已知信道下发射波束优化，和从压缩观测估未知信道不同。

## Deshpande2024

原文件：`D:/zotero_data/storage/KUYW67HC/Deshpande 等 - 2024 - Frequency-selective beamforming and single-shot beam training with dynamic metasurface antennas.pdf`。13页。重点阅读I–V、pp.2–4,7–12，闭式推导按相关引理读取。作者代码链接p.2： https://github.com/nvdeshpa/FreqSelBeamformingBeamtrainingDMAs ，本任务未复现。

- p.3 Eq.(1)为频率相关Lorentzian极化率；Eq.(3)明确幅相耦合，F和Γ固定由单元设计决定。波导固有相位−ng 2πf(n−1)dy/c，主分析令幅度不衰减。p.4 Eq.(8)明确忽略单元互耦，外部传播为LOS远场。
- 研究中的频率重构是 **数据通信中心频率ft也可随用户角度变化**，频谱可用范围必须足以支持。不是在固定已分配OFDM带宽里逐个子载波独立设DMA状态。
- p.2两阶段：给ft时闭式求最优fr，再优化ft。p.7设计ng、dy让目标角域的可选ft实现N²中心频点最大增益；p.8把单元谐振与阵列波束的频率响应分开，3 dB带宽近似Γ/(2π)。
- p.8示例：15 GHz中心、可操作12–18 GHz、N=8、dy=0.42λc、ng=2.5、Γ=2πfc/50；角−18°最优16.43 GHz，3 dB带宽300 MHz。p.9–10只在一次数值比较加α=6的指数衰减，后续回到理想模型。
- p.10–11 Sec.IV：Nz条DMA划成L组，组ℓ用一个fr=f*t(φℓ)对准不同扇区；一次OFDM导频同时发不同频点。接收者选择最大增益子载波k*并反馈，整阵重配到fk*，由频率角度映射得角估计。输出是最佳扇区/频点及波束；没有估各路径复系数、近场距离、全H[k]。
- p.11–12：L=4，Ny=8/Nz=4时所选码本增益在±30°内距理想≤3 dB；Ny=4/Nz=8对应≤0.6 dB示例。p.12速率设置250 mW、500 m、15 GHz、64数据子载波、6 GHz中心频率调谐范围；与TTD速率接近到约300 MHz，此后谐振频变令差距增大。
- p.10作者承认二维方位+俯仰训练是后续问题；p.9–10、IV–V读取范围内主要是理想响应/方向扫描，未给出噪声导致子载波选错概率的误差曲线，故“single-shot有效”不能等同任意SNR或多径下完整CE可靠。
- 增量结构：把既有谐振频率从配置参数变成中心频率设计变量 → 闭式增益/带宽/覆盖关系 → 频率角度编码的单次训练 → 有限状态码本。与候选二的有限状态/频点思路有明确直接重合；若讨论CE分配，应把未知目标、固定可用频带、观测预算和误差目标与这里区分，不能只换“beam training”为“channel estimation”。

## Gavras2024 / ICC2025

原文件：`D:/zotero_data/storage/4XHZGDHG/Gavras和Alexandropoulos - 2024 - Circuit-compliant optimization of dynamic metasurface antennas for near-field localization.pdf`。6页，全文已读。p.5已渲染核对。

- p.2 Eq.(1)：WRX=PSA^H(WTA+WMC)^(-1)。PSA来自波导Green函数含传播/反射，WMC来自空气Green函数+同波导耦合，WTA对角负载；Eq.(2)令逆终端导纳满足Lorentzian集合。这里硬件矩阵由结构给定，是已知前向系统，未把任意WMC作为盲估未知量。
- pp.2–3展开矩阵逆：一阶WRX≈PSA^H WTA^(-1)忽略互耦；二阶多−PSA^H WTA^(-1)WMC WTA^(-1)。Fig.1用4RF×32单元、λ/5单元距、λ微带距验证截断误差。不能只因一阶二阶常够用，就声称强耦合任意负载也必然收敛；这是读者推断的适用条件，Neumann/Taylor近似应检查相应矩阵谱半径/误差。
- p.3 Eqs.(3)–(5)：近场每单元不同距离ri,n和角度/辐射幅度。Eq.(6)噪声在N单元输入端，经过WRX和v，p.4协方差为σ² v^H WRX WRX^H v IT，保留权重相关噪声增益。
- p.4：由位置参数ζ=[r,θ,φ]求CRB/PEB，再用Tr(I^(-1))的下界转成Rayleigh商，求理想合并向量后最小二乘分解为v、Lorentzian权重。p.5二阶权重相互耦合，作者明确不能保证最优。故准确称“PEB替代目标+闭式分块更新”，不要说全局精确最小化PEB。
- 读者推断：上述A包含h对r/θ/φ的导数，优化需名义位置/当前估计来计算这些导数；不是对所有未知位置自动适用的无先验训练器。II–IV未评估位置先验偏差下的鲁棒性。
- p.5 Sec.IV：28 GHz、带宽150 kHz、r∈[1,30]m、θ=30°、φ∈[0°,180°]、500 Monte Carlo、T=200导频；dRF=λ，dE=λ/5。使用最大似然估计并与界比较。图2caption写2RF×256，正文写2RF×128，渲染证实矛盾确实在原文，不应选一个数假装已确定。
- p.5–6 Figs.2–3：已知耦合模型设计与理想DMA、HBF作定位性能比较；增加RF链时二阶模型PEB靠近理想DMA。不同架构元素间距λ/5与λ/2不同，不宜用曲线直接推出同孔径/同噪声下“耦合总会带来增益”。
- 增量结构：直接复用Williams电路模型 → 一/二阶逆矩阵近似 → 定位CRB替代目标 → 闭式混合权重更新和ML仿真。与候选一的耦合已知硬件算子、与储备方向的低维硬件参数鲁棒性相邻；它本身并未在所查II–IV建立少量谐振/传播参数漂移的CE问题。

## 可供主任务使用的边界判断

1. 六篇说明“小幅度改变问题条件并把模型—方法—仿真闭合”是常见论文构造：已知框架放入DMA约束、已知宽带响应换成信息量任务、已知电路模型截断后做可算优化。不能据此保证同类组合一定有新颖性或达到TWC标准。
2. 候选一直接重合：真实频变Lorentzian、有限Btune、波导频变/泄漏本身已由Carlson与Gavras2026明确使用；Williams和Gavras2024已有已知互耦矩阵。可区分与否应看“未知H的结构/观测是否改变、求解机制与低导频误差是否实质改善”，不能只看模型更真实。
3. 候选二直接重合：Deshpande已完成有限谐振状态、频率编码角域、单OFDM训练，Gavras2026已堆叠J状态×K频点并算信息界，Xu已做DMA慢调与数字快调。父任务应避免把这三句并列当创新点；本任务只指出重合，未完成全领域优先权检索。
4. 候选三：本六篇没有条件生成算法证据。若粗估来自这里的波束训练/定位，粗估带来的已知信息量及其错误分布要计入比较；完美真实AoD、位置或支撑不是免费侧信息。
5. 储备方向：Williams已有用S11提取单元导纳，Gavras2024已有已知互耦修正，不能把“校准”本身概括为缺口。限定每微带少数谐振频偏/传播常数漂移，并说明哪些已知、哪些由观测估计，才可具体比较。Tapie互耦tensor与2512.22607多端口校准已由父任务补查，本文件不重复。

## 验证台账

- 已渲染并查看：`dma_physics_Williams2022_p11.png`，确认Table II及静态槽/S11验证叙述；`dma_physics_Gavras2024_p5.png`，确认图2单元数在图注和正文确实矛盾。
- 文件安全：仅写本证据文件及上述许可前缀临时图片，原PDF、代码、模型未改。
- 未执行：数值复现、训练或任意硬件实验；未逐点数字化提取图曲线。物理参数跨器件迁移与上述近似适用域仍需在主任务实验阶段检查。
