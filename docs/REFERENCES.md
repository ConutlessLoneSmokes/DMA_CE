# 方法与物理依据

- Jin等，Near-Field Channel Estimation for XL-MIMO: A Deep Generative Model Guided by Side Information，TCCN 2026：[公开原文](https://arxiv.org/abs/2505.06900)。借鉴同观测粗估条件生成；本仓库轻量CNN不是该文U-Net的逐式复现。
- Zhou等，Generative Diffusion Models for High Dimensional Channel Estimation，TWC 2025：[公开原文](https://arxiv.org/abs/2408.10501)。复用既有THz_DMA实现的CNN和VP时序，新增条件接口与DDIM推理。
- Williams等，Electromagnetic Based Communication Model for Dynamic Metasurface Antennas，TWC 2022：[公开原文](https://arxiv.org/abs/2212.11594)。DMA电磁和电路建模依据。
- Carlson等，Wideband Dynamic Metasurface Antenna Performance With Practical Design Characteristics，TWC 2026：[公开原文](https://arxiv.org/abs/2510.07827)。有限谐振调谐和频变馈线模型依据。
- Peng等，Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance，ICML 2024：[正式论文](https://proceedings.mlr.press/v235/peng24h.html)。高斯一致性和协方差处理的通用先例；本仓库不将该通用公式列为原创。

- Hashemi等，Unification of sparse Bayesian learning algorithms for electromagnetic brain imaging with the majorization minimization framework，NeuroImage 2021：[公开原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8433122/)。标准SBL凸上界更新与下降性质的依据；本仓库采用复数、跨频字典共享方差版本。

SBL粗估采用常规复高斯后验和跨频共享方差。2026-10-20起使用凸上界更新及归一化一阶条件，推导与预算检查见 [SBL_SOLVER.md](SBL_SOLVER.md)。可靠度是该近似模型给出的不确定度代理；当前比较只保留无可靠度确定性精修，旧可靠度对照保留在历史结果中。
