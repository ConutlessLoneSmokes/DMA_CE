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
```

每个运行ID只能创建一次；新实验使用新ID。运行保存在被Git忽略的 runs/；紧凑逐场景指标、图和报告另保存 results/，可提交到私有研究仓库。

## 结果状态

13项数学/物理/数据边界检查通过。[GPU短训练](results/smoke_20261016_v1/report.md)完整跑通SBL、四个网络、验证冻结及11组评价，共176条逐场景指标、约12秒。它只验证连通性；候选A的首轮方法筛选继续进行。
