# 生成模型与 FPN 论文证据摘记

检索日期：2026-10-16。范围：用户语料六篇 Feng2026、Jin2026、Chen2025、Arvinte2023、Zhou2025、Yu2023，全部纳入；没有外扩为系统综述，也没有候选方向排名。以 catalog.json 指向的原 PDF 为事实依据，浏览器只核对公开原文/作者页面及出版信息。本文的 p. 为 PDF 页码，不是期刊印刷页码。

阅读范围：Feng p.1–12；Jin p.1、3–14；Chen p.1、3–14；Arvinte p.1–14；Zhou p.1–13；Yu p.1–13。读过所列模型、方法、实验及结论正文；没有逐式重推附录证明或审计作者实现。已渲染并目视核对 Jin p.10 Table I、Chen p.8 Table I/Algorithm 2、Yu p.10 Table II、Arvinte p.8 Table I，图保存为 generative_*.png。

核查性质：六篇均是无线信道的计算/仿真研究；医学 I–VII 证据等级不适配。已核实模型、输入输出、训练条件、基线设置、相应消融/扩展与主要限制。未核查真实硬件、代码逐行一致性或证明代数。下面每条限制均注明“作者承认”或“阅读推断”；阅读推断不是已验证机制。

## 1. Feng2026

Feng, Q., Fang, Y., Hua, M., Zhang, C., Li, C., Huang, Y., & Yang, L. (2026). *Near-field wideband channel estimation for XL-MIMO systems via denoising diffusion model*. arXiv:2604.20494. https://arxiv.org/abs/2604.20494

书目：公开 arXiv v1 提交日期 2026-04-22；DOI 10.48550/arXiv.2604.20494。所给 PDF 首页及当前 arXiv 记录没有给出已刊期刊信息，因此此处按预印本标记。

- 问题与观测：单天线上行用户，N 元 ULA、M 个 OFDM 子载波；h[m] 是频变球面波多径信道，角度、距离、频率共同决定阵列相位。观测 Y=HS+N，S 是 M×M 对角矩阵，每个子载波导频功率非零且相同；等价 y=Bh+n，B=S^T⊗I_N（p.3 §II，p.4 eqs.6–7）。未知是完整 N×M CSI，不是角距参数。
- 关键信息条件：这里直接观测每个天线、每个子载波的含噪信号；B 是方阵、满秩的已知导频变换。因此在此信号模型下，恢复等价于先除导频再去噪。它没有 Jin/Yu 的 RF 链压缩或 Np<Nt 的欠定性。此为由 eqs.6–7 推出的阅读判断，不能把其结果直接等同于 DMA 压缩反演。
- 方法：离线只用干净 H 加人为扩散噪声训练，无 y、B、实际噪声统计；线上以学习的先验 score 加近似似然 score 进行确定性更新（p.6–7 eqs.34–44，Algorithm 1）。似然采用非信息先验近似 p(h0|ht)，并保留 BB^T（p.7 eqs.40–43）。网络采用 3×3、5×5、7×7 多尺度卷积、feature attention、spatial attention 和 timestep 调制（p.8–9，Figs.4–5）。
- 建模证据：§III 分别推导天线相关与子载波相关依赖绝对天线位置（p.4–6 eqs.12–33）。文中的空间非平稳主要由此相关性含位置解释；§II-A eqs.1–4 没有显式可见区域开关，不能自动等同于路径只在部分子阵可见的 VR 模型。后一句为检查该模型后的阅读判断。
- 仿真：N=128、fc=60 GHz、M=64，角度[-π/3,π/3]、训练用户距离[5,40] m；100000/2000/2000 训练/验证/测试样本；T=100，C=32，3 个多尺度块（p.9 §V-A）。Fig.6/7 比较 L=4/6，Fig.8 在 SNR=10 dB 下扫距离 5–200 m，Fig.9 扫带宽 2–10 GHz（p.10–11）。图与文字支持其优于所列 LMMSE、POMP、PSOMP、BSPD、BPD、CNN，但这里不从未目视的曲线估数值。
- 贡献组合：具体近场宽带统计分析 + 通用 posterior diffusion 恢复 + 多尺度注意力网络 + SNR/距离/带宽验证。核心反演结构与 Zhou 的“先验 score + 非信息先验近似的似然 + 确定性更新”同型；这是对照 eqs.38–44 与 Zhou eqs.15–23 得出的阅读判断，不是作者原创性裁定。
- 限制：阅读推断，检查 §V-A/B p.9–11，仅比较既有传统/CNN 基线，所列实验没有通过相同 diffusion 框架去掉 attention 的控制组独立量化 attention 增益；跨带宽/距离实验也未在这些段落中明确是否始终复用同一模型权重。

## 2. Jin2026

Jin, Z., You, L., Ng, D. W. K., Xia, X.-G., & Gao, X. (2026). Near-field channel estimation for XL-MIMO: A deep generative model guided by side information. *IEEE Transactions on Cognitive Communications and Networking, 12*, 628–643. https://doi.org/10.1109/TCCN.2025.3566047

公开原文：https://arxiv.org/abs/2505.06900 。首页显示 2025-05-01 online、2025-12-19 current version、正式卷年为 2026，不能把文件年和首次在线年混为一谈。

- 模型：ULA 混合接收，C_q 为 NRF×N 随机恒模合并；r_k=C P h_AD,k+n_k。噪声来自 C_q n，先按分块协方差作 Cholesky 白化，再用极域字典 P 和 SOMP 做共同支撑稀疏重建（p.4–5 eqs.5–11）。最终目标 H=P H_AD，是完整 CSI。
- 频率条件：阵列 steering 在模型中用 λc，视为频率无关，所有子载波共用角距稀疏支撑（p.4 eq.2、p.5 eq.9）。因此这是窄相对带宽 OFDM 条件，不是显著 beam split 下的频变共同支撑修复。
- side information 来源：先用同一批 r_k、C、字典 P 得到初估 Ĥ=P Ĥ_AD=H+e（p.5 eq.12）；Ĥ 的 Re/Im 两通道与当前 H_t 沿特征维拼接，输入条件去噪网络 εθ(Ĥ*,H_t,t)（p.4 Fig.2、p.5 §III-B、p.7 Algorithms 1–2、p.9 Fig.3）。线上没有引入额外一轮导频，也没有给方法输入真实角度/距离。
- 离线/在线区别：训练加载 (Ĥ*,H0) 配对数据，人工加噪到真实 H0 形成 H_t；推理加载 Ĥ*，从随机 H_T 开始生成精修信道（p.7 Alg.1/2）。所以是学习 p(H|Ĥ*) 的条件生成精修，区别于 prior-only 的 Arvinte/Chen/Zhou/Feng。真实 H 只在离线训练及评分中使用。真实角度和距离只供 Genie-aided LS 基线（p.12 §V-C）。第二阶段所列更新显式输入只有初估、扩散状态及 t，没有另列 y/A 数据一致性步；检查 p.7 Alg.2 与 p.8 eqs.37–40。
- 小增量算法结构：已有极域 SOMP 初估 + 条件 GDM 精修及条件 ELBO + 非 Markov/DDIM 式采样缩步，共享训练目标（p.3 contributions、p.6–8）。作者明确讨论 SOMP 网格角距与连续真实值不一致，这是引入精修的动机（p.5 §III-B；p.14 conclusion）。
- 仿真：N=256、NRF=16、16 单天线用户、fc=28 GHz、B=100 MHz、K=256、L=6；70000 生成配对样本按5:1:1划分；T=1000、默认采样 S=100（p.10 Table I、p.10–11 §V-A）。Q=32 时每子载波 NRF Q=512≥N；Q=8 时128<N。维数计数为阅读推断，不能把主图 Q=32 误称为“空间未知数多于观测数”。
- 支持结果：p.12 Fig.7，在 SNR=5 dB、用户15–25 m、Q=32 时，NM-GDM 100 步与 GDM 1000 步 NMSE 近似，约10倍采样步数缩减。p.13 Figs.8–9 覆盖近场15–25 m、远场350–400 m，Q32/Q8。
- 限制：作者承认，p.13 §V-C：高 SNR 下增益缩小且存在学习泛化误差；p.14 conclusion：轻量化与采样效率仍需优化。阅读推断：所检 §V-A/C 未充分交代每条跨距离/天线曲线是否固定同一组权重，不把图中“robust”自动提升为严格同模型零重训结论。

## 3. Chen2025

Chen, Z., Shin, H., & Nallanathan, A. (2025). Generative diffusion model-based variational inference for MIMO channel estimation. *IEEE Transactions on Communications, 73*(10), 9254–9269. https://doi.org/10.1109/TCOMM.2025.3556753

作者机构 accepted manuscript：https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/106727/Nallanathan%20Generative%20Diffusion%20Model-based%20Variational%20Inference%20for%20MIMO%20Channel%20Estimation%202025%20Accepted.pdf?isAllowed=y&sequence=2 。出版 DOI 的浏览器解析失败；元数据由用户正式 PDF p.1 及机构稿标题作者交叉确认。正式 PDF 显示2025-04-01 online、2025-10刊期。

- 问题：全数字接收 W=I，Y=HP+N，P 为随机 QPSK beamformers，ω=Np/Nt<1 时欠定；未知 H 是完整 Nr×Nt CSI（p.3 eqs.1–2）。
- 直接学习 H 的先验：s(H)=∇H log p_H(H)，训练输入为 H+Z_t，目标负缩放噪声（p.4 eq.3、p.5 eq.11、p.6 Alg.1）。不是学习稀疏方差超参数或 hyperprior。
- 方法增量：保持已有 generative score prior，把 annealed Langevin posterior sampling 换成 VI；把 q(H0|Y) 取为 δ(H0−μ)，从数据匹配项加 score 正则优化 μ；λ_t=t α_t²/T，避免穿过 score 网络反传，每扩散级仅一次更新（p.7 eqs.18–24、p.8 Alg.2/Theorem2）。最终输出 H=μ，是点估计形式；不能因为文章用 posterior sampling 一词就声称它拟合了一般完整后验协方差。
- 已知信息：推理 P、Y、实际噪声功率、预训练 score、扩散噪声表、步长；不输入真实路径角距。训练只用干净 H，不依赖 pilot/SNR（p.6；p.8 Alg.2）。
- 仿真：MATLAB 3GPP CDL A/B/C/D；fc=40 GHz，ULA(Nt,Nr)=(128,32)，UPA=(256,64)，主 pilot density0.6。每类10000 train、100 test，T=2000。虽然先生成 OFDM 数据，训练/测试仅取第一子载波第一符号（p.8 Table I、§IV）；不能当作宽带联合估计。
- 模型复用：每种 CDL 独立训练，另有每种2500组成的10000混合训练；SNR与pilot density变化不 fine-tune（p.9）。天线尺寸变化则每尺寸单独训练（p.12 §IV-C），不等于一组权重跨尺寸泛化。
- 支持结果：ULA/CDL-A/ω=.6/SNR=-10 dB，对 Langevin L=3 的 NMSE 改善6.3 dB；以每级一次更新对每级三次更新实现约3倍 inference 加速（p.9–10 Fig.2）。
- 限制：作者承认，CDL-C→CDL-A 时效果差（p.9、p.12）；推理延迟/FLOPs仍大于LDAMP/ML（p.13 Table II讨论）；非线性放大器失真和量化导频留作未来工作（p.14 conclusion）。阅读推断：其主证据在单子载波、全数字接收、已知白噪声下，不能直接外推到频变 DMA 和合并后的有色噪声。

## 4. Arvinte2023

Arvinte, M., & Tamir, J. I. (2023). MIMO channel estimation using score-based generative models. *IEEE Transactions on Wireless Communications, 22*(6), 3698–3713. https://doi.org/10.1109/TWC.2022.3220784

公开原文：https://arxiv.org/abs/2204.07122 ；作者实现：https://github.com/utcsilab/score-based-channels 。2022-11-21 online，2023-06刊期（PDF p.1；作者代码引用确认）。

- 模型：窄带、全数字接收W=I，Y=HP+N，随机QPSK P，α=Np/Nt<1；最终H完整CSI（p.3 eqs.1–2）。离线用干净H训练RefineNet score，不使用P、Y、实际pilot噪声（p.4 eq.10、p.5 eq.14）。线上使用P、Y、噪声功率，prior score、pilot一致性梯度和随机扰动迭代，默认每个噪声级3次（p.5–6 Alg.1）。没有真实几何输入。
- 相对既有工作增量：把score-based prior/posterior sampling适配MIMO、给SISO tapped-delay场景的分布失配理论、用MIMO仿真及编码BER和复杂度验证（p.2–3 contributions）。理论条件是线性Gaussian测量和简化SISO，不是对任意MIMO/DMA观测的统一保证（p.6–7 §IV）。
- 实验：40 GHz CDL模型，ULA Nr×Nt=16×64、32×128，UPA64×256；训练CDL-B/C，测试A/B/C/D；10000训练样本，每分布100测试样本；只取生成OFDM信道第一子载波第一符号（p.7–8、Table I）。混合模型每类10000，与单类10000数据预算不相同（p.8 §V-C）。
- 同权重情况：CDL-B/C→不同CDL测试不重训、不适配（p.7–8）；天线尺寸扩展每尺寸单独训练（p.12 §V-E）。
- 支持结果：CDL-C训练→CDL-D测试、α=.6、16QAM、rate1/2 LDPC时，同BER比L-DAMP节省约5dB Eb/N0（p.11 Fig.6a）。这是通信链路BER收益，不是5dB NMSE。高维64×256在α≈.25以下NMSE快速劣化（p.12 Fig.8）。
- 限制：作者承认，训练CDL-B→CDL-A时高SNR error floor（p.9–10）；Langevin推理昂贵，16×64经结构优化仍约1.5秒（p.14 conclusion）。作者明确因此适用低移动性固定接入/回传等场景。

## 5. Zhou2025

Zhou, X., Liang, L., Zhang, J., Jiang, P., Li, Y., & Jin, S. (2025). Generative diffusion models for high dimensional channel estimation. *IEEE Transactions on Wireless Communications, 24*(7), 5840–5854. https://doi.org/10.1109/TWC.2025.3549592

公开原文：https://arxiv.org/abs/2408.10501 ；作者书目确认卷期页：https://www.liang-seu.net/publications/ 。PDF p.1显示2025-03-17 online、2025-07-14 current version。arXiv当前v3更新于2026-03-09，不应据此改写正式刊期。

- 问题：窄带全数字MIMO，Y=HP+N，先转角域再转实数，y=Ah+n；A包含pilot与DFT，目标最终完整CSI（p.3 eqs.1–6）。先验是角域h的DM，非hyperprior。训练与A/噪声统计解耦；推理输入A,y,σ_n²和gradient scale（p.4、p.5 Alg.1）。
- 增量方法：轻量CNN + 确定性逐级posterior mean更新；非信息先验近似得到含AA^T的噪声级相关似然，SVD预计算减少矩阵逆（p.5 eqs.15–23）。进一步换成量化似然；采用SURE先去噪、再以去噪样本训练DM（p.6–7）。这说明“一个主要恢复算法 + 一个实际限制扩展 + 一个数据获取扩展”是已发表的贡献组合。
- 量化的确切条件：闭式近似进一步假设A行正交，使有效噪声协方差为对角（p.6 Proposition2/Remark3）；实验仍用不严格行正交QPSK并展示经验稳健（p.10 §IV-C）。该闭式不能未经验证地视为任意DMA矩阵的精确似然。
- SURE训练信息：假设已有LMMSE型完整估计 h~=h+w，w为已知方差的白Gaussian误差（p.7 eqs.31–33）。它不直接从任意欠定压缩y学习完整先验；将这一假设搬到DMA需先处理估计偏差/有色误差，后半句为阅读推断。
- 仿真：QuaDRiGa UMa-LOS，40GHz，Nt=64/Nr=16，半波ULA，100000train、100validation、100test；BS高25m、UT距35–500m，pilot随机QPSK（p.8）。T100-enhanced在后半程每级3次迭代，之后作为主variant（p.9）。LOS训练→NLOS测试明确无重训（p.10）。
- 支持结果：3bit ADC、SNR10dB，Fig.9中作者报告用少于基线50%导频仍优于基线（p.11）；1bit专用似然相比基线超过1dB NMSE（p.10 Fig.8a）。不要把其摘要“10倍latency”当成所有条件统一常数，文中还存在不同矩阵尺寸和基线的比较。
- 限制：作者承认，OoD稳健性理论、少数据快适配、gradient scaling/后半程重复更新的理论依据仍待研究（p.13 conclusion）。阅读推断：近似似然的行相关性和SURE的白误差条件是适配DMA时的具体差异，非“扩散本身不适合”的结论。

## 6. Yu2023

Yu, W., Shen, Y., He, H., Yu, X., Song, S., Zhang, J., & Letaief, K. B. (2023). An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation. *IEEE Journal of Selected Topics in Signal Processing, 17*(4), 761–776. https://doi.org/10.1109/JSTSP.2023.3282832

公开原文：https://arxiv.org/abs/2211.15939 ；作者实现：https://github.com/wyuaq/FPN-OAMP-THz-Channel-Estimation 。PDF p.1显示2023-06-05 online，卷期为July2023，current version2023-10-13。

- 模型：4个子阵RF链，AoSA非均匀阵列，近/远场路径混合；分块恒模混合合并，y=Mh+n，M已知、h为信道系数实数表示，最终可还原完整CSI；不输入真实路径角距（p.3–5 §II，eqs.10–12）。DL方法用DFT字典，CS基线额外做dictionary learning以匹配该信道（p.5、p.10）。
- 方法增量：保留OAMP线性观测一致性更新，非线性去噪器换成跨迭代共享ResNet；训练以不动点为目标，用one-step gradient降低存储，约束/检查收缩性，用迭代容差控制精度和耗时；加入无标签数据一致性loss作为线上微调后备（p.6–9 eqs.15–23，Alg.1）。这是OAMP+FPN的具体结合，不要求重新发明估计任务或全新网络原理。
- 证明精度边界：Theorem7/Corollary8保证的是到网络不动点h*的收敛及距离收缩，不直接等于每步都向真实h_gt降低NMSE；后者在p.11仿真曲线中验证。此处是按定理变量作出的阅读判断。
- 仿真：TableII p.10：S=4、每SA256天线、总1024、fc=300GHz、da=λc/2、dsub=56λc；Q=128，采样比ρ=50%；L5，LoS距30m、NLoS散射距10–25m。80000/5000/5000样本，训练SNR0–20dB，单组参数mixed-SNR训练（p.10）。p.11宽带扩展B15GHz/K32。
- 宽带机制：两种方案都逐子载波并行估计；一种直接复用中心频点窄带模型；一种把各子载波当独立训练样本重训，再逐频推理（p.9 §IV-A/B）。作者明确第一种忽略跨频相关性；第二种覆盖多频训练分布，但推理输入仍不联合跨频。
- 支持结果：p.10–11 Fig.5–8显示一般2–5次迭代收敛、相对base OAMP可达10dB NMSE增益。p.13 TableV讨论：ρ从训练.5降低到.3/.1时直接泛化明显失败，少量线上自适配恢复91%/98%的性能差距；提高至.7通常无损。
- 限制：作者承认，p.13 §VI-C，少导频失配不能直接泛化，逐测试样本自适配不经济；p.9 §IV-A，窄带复用忽略跨频相关。阅读推断：对频变DMA需要替换每频M[k]及其对应线性步，而不是默认原文频率不变M继续成立。

## 有助于父任务比对的证据边界

1. 六篇最终任务都是完整CSI恢复；Jin中间用角距稀疏系数，Feng/Jin不是以准确角距定位作最终指标。没有一篇给所提估计器oracle真角距；Jin的Genie LS基线单独给出。
2. 同样称“diffusion channel estimation”，信息条件不同：Feng完整N×M含噪观测；Jin同观测SOMP初估作条件；Arvinte/Chen/Zhou是pilot欠定反演的prior-only训练；Yu是已知压缩M的一致性线性步加训练去噪器。由这些不同条件产生的误差曲线不宜直接对照为算法优劣。
3. Jin“初估→条件精修→加速采样”、Chen“既有score prior→VI→权重机制”、Zhou“后验恢复→量化似然→带噪训练”、Yu“OAMP→共享不动点网络→自适配/宽带扩展”都是原文可核查的增量贡献组织。这里只总结这些已发表/预印本工作怎样构成贡献，不据此承诺另一组合的新颖性或录用。
4. 跨SNR/导频复用、跨信道分布复用、跨天线尺寸复用是三件不同的事。Arvinte/Chen明确尺寸变化分别重训；Jin配对条件训练不能自动拥有prior-only方法的任意观测模型适配。

未执行训练、数值复现、证明重推及实现审计；因此数值为原文作者报告，未独立复验。
