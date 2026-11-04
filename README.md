# DMA_CE

物理宽带 DMA 压缩观测下的完整信道估计研究。复用 [THz_DMA](https://github.com/ConutlessLoneSmokes/THz_DMA) 的球面波、Lorentzian/馈线响应与轻量 CNN，来源及原始哈希见 [复用清单](docs/UPSTREAM_REUSE.json)。

已完成密集多 RF 阵列上的 LS、LMMSE、SBL 与条件 DDPM 比较，以及同观测下的相干跨频后验、机制对照和输出步长标定。当前结果与已采纳决定见 [阶段总结](docs/EXPERIMENT_LOG.md#阶段总结)；各轮状态、单 RF 探索及失败记录保留在同一实验日志。

| 需要做什么 | 入口 |
|---|---|
| 找运行指南、方法依据或历史记录 | [文档索引](docs/README.md) |
| 理解观测、SNR 与评价口径 | [领域术语](GLOSSARY.md) |
| 组织任务、选用 skills、准备交付 | [工作流程](docs/WORKFLOW.md) |
| 查实现检查与复算证据 | [核验记录](docs/IMPLEMENTATION_CHECKS.md) |

## 运行

Python 3.10 及以上，先安装匹配显卡的 PyTorch，再运行：

```text
python -m pip install -e .
python -m unittest discover -s tests -v
```

具体实验使用对应指南中的命令与配置。每个新实验使用独立运行 ID；精确复现历史结果使用该运行保存的源码与环境。

Git 保存代码、配置、必要结果和证据；模型、观测缓存、PDF 与凭据由 `.gitignore` 排除。完整运行主要保存在服务器 `/home/lsy/DMA_CE/runs/`。THz_DMA 原仓库保留在本地 `upstream/THz_DMA/` 和服务器 `/home/lsy/THz_DMA`。
