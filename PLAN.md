# 基于自适应时空图神经网络的海温场多步预测 — 实施计划

> 依据：`SST_opening_presentation.pdf` + Ning (2023) 图构造方法 + Bai (AGCRN, 2020) 模型方法

---

## 0. 总体思路

将 **Ning 的物理/数据驱动先验图 $A_{prior}$** 与 **Bai 的 AGCRN (DAGG + NAPL)** 结合：
- Ning 负责"从网格构造带远程连通（teleconnection）的稀疏图"。
- AGCRN 负责"在图上做节点自适应的时空建模与多步预测"。
- 融合策略：$\tilde{A} = \lambda A_{prior} + (1-\lambda) A_{adapt}$（MVP 用固定 λ，后续可学习）。

数据流：OISST 日数据 → 去季节化（SSTA）→ 陆地掩膜+节点索引 → 滞后/Pearson 相关 → top-k 稀疏化 → $A_{prior}$ + AGCRN 训练 → 递归多步预测 → 评估。

---

## 1. 代码仓库结构（建议）

```
Temperature_predictor/
├── data/
│   ├── raw/                    # OISST .nc 原始
│   └── processed/              # SSTA tensor [T, N] + mask + coords
├── src/
│   ├── data/
│   │   ├── download.py         # 已有 download_oisst.py 迁入
│   │   ├── preprocess.py       # 去季节化、掩膜、滑窗、归一化
│   │   └── dataset.py          # PyTorch Dataset / DataLoader
│   ├── graph/
│   │   ├── build_prior.py      # Ning: Pearson/lagged corr + top-k
│   │   └── utils.py            # 归一化、度矩阵、可视化
│   ├── models/
│   │   ├── agcrn_cell.py       # NAPL-GCN + GRU (Bai Eq.7)
│   │   ├── agcrn.py            # 多层 encoder + 线性投影多步输出
│   │   └── baselines.py        # Persistence / Climatology / ConvLSTM
│   ├── train/
│   │   ├── train.py            # 训练循环、早停、ckpt
│   │   ├── evaluate.py         # RMSE/MAE/Pearson/SSIM + 误差热图
│   │   └── config.py           # 超参 (dataclass/yaml)
│   └── viz/
│       └── plots.py            # lead-time vs RMSE、空间误差图、时序对比
├── experiments/
│   └── <exp_name>/             # 每次实验独立目录：config + ckpt + logs + figs
└── notebooks/                  # 探索性分析
```

---

## 2. 分阶段任务（与开题 Timeline 对齐）

### Stage A：数据与预处理（第 1–4 周）✅ 部分已完成
- [x] OISST 下载脚本（`download_oisst.py`）
- [ ] **选定空间范围与分辨率**：建议先做区域子集（如 北太平洋 / 热带太平洋 0.25° 降采样到 1°）以降低 N
- [ ] **去季节化（SSTA）**：逐格 climatology (1991–2020 基准)，$\text{SSTA}_{x,t} = \text{SST}_{x,t} - \bar{\text{SST}}_x^{doy}$
- [ ] **陆地掩膜**：保留海洋节点 → 扁平化为 $[T, N]$
- [ ] **滑窗切分**：$T_{in}$（历史）/ $T_{out}$（预测），按时间顺序 6:2:2 划分
- [ ] **归一化**：min-max 或 z-score（记录参数反变换）
- [ ] **输出**：`processed/ssta.npy`, `mask.npy`, `coords.npy`, `splits.json`

### Stage B：图构造（Ning，第 5–8 周）
- [ ] `build_prior.py`：
  - 输入：训练集 SSTA `[T_train, N]`
  - 对每对节点计算 Pearson（可选带 lag τ∈{0,1,...,L} 的 lagged correlation，取最大 |ρ|）
  - 阈值 c 或 **top-k 邻居**（Ning Table 1：k 控制平均度；建议初版 top-k=30）
  - 对称化（无向）或保留方向（有向，lead→lag）
  - 对称归一化 $\tilde{A} = D^{-1/2}(A+I)D^{-1/2}$
- [ ] 保存 `A_prior.npz`（稀疏 COO），可视化：邻接矩阵热图 + 若干节点的远程连接地理图（呼应开题 slide 8）

### Stage C：模型实现（AGCRN，第 9–12 周）
- [ ] **NAPL-GCN**（Bai Eq.4）：$Z = (I + \tilde{A}) X E W + E b$，其中 $\Theta = E W_{pool}$
- [ ] **DAGG**（Bai Eq.5）：$A_{adapt} = \text{softmax}(\text{ReLU}(E_A E_A^T))$
- [ ] **融合**：$\tilde{A}_{used} = \lambda A_{prior} + (1-\lambda) A_{adapt}$（初版 λ=0.5 固定；消融对比 λ∈{0,0.3,0.5,0.7,1}）
- [ ] **AGCRN Cell**（Bai Eq.7）：用 NAPL-GCN 替换 GRU 中的 MLP
- [ ] **多步输出**：堆叠 2 层 AGCRN，末端线性层 $\mathbb{R}^{N\times d_o}\to \mathbb{R}^{N\times \tau}$（避免自回归开销）
- [ ] **损失**：L1；优化器 Adam；早停 patience=15

### Stage D：基线与评估（第 11–14 周）
- [ ] **基线**：
  - Persistence：$\hat{X}_{t+k}=X_t$
  - Climatology：$\hat{X}_{t+k}=\bar{X}^{doy(t+k)}$
  - ConvLSTM（2D 网格版，需保留 grid 形态的一路预处理）
  - 可选：DCRNN / STGCN（对比 GCN 类）
- [ ] **指标**：RMSE、MAE、Pearson r（每节点）、SSIM（重构为 2D 场后）
- [ ] **结果呈现**：
  - 表：lead-time = 1/7/14/30 天的 RMSE 对比
  - 图：lead-time vs RMSE 曲线、空间误差热图、10 个典型点位时序对比（呼应 Ning Fig.3）

### Stage E：消融与分析（第 13–16 周）
- [ ] 消融维度：
  - w/o NAPL（共享参数 GCN）
  - w/o DAGG（仅 $A_{prior}$）
  - w/o $A_{prior}$（仅 DAGG，即原始 AGCRN）
  - 不同 top-k ∈ {10, 30, 50, 100}
  - 节点嵌入维度 d ∈ {2, 5, 10, 20}
  - 融合 λ 扫描
- [ ] 可解释性：可视化学习到的 $E$（t-SNE）、$A_{adapt}$ 与 $A_{prior}$ 差异地图

### Stage F：论文与答辩（第 17–28 周）
- [ ] 论文初稿（方法 / 实验 / 讨论）
- [ ] 复现脚本与 README
- [ ] 答辩 PPT（复用开题视觉）

---

## 3. 关键超参（初始建议值）

| 项 | 值 |
|---|---|
| 空间范围 | 先做区域（如 (−30°–30°N, 120°E–80°W) 热带太平洋），N ≈ 3000–6000 |
| 时间范围 | 2015–2024，其中 2015–2021 训练 / 2022 验证 / 2023–2024 测试 |
| $T_{in}$ / $T_{out}$ | 30 / 30 天（对齐开题"未来 30 天"） |
| 节点嵌入 d | 10 |
| top-k | 30 |
| 隐藏维 | 64 |
| AGCRN 层数 | 2 |
| batch / lr | 32 / 1e-3 |
| 融合 λ | 0.5（初版） |

---

## 4. 风险与对策（扩展开题 slide 14）

| 风险 | 对策 |
|---|---|
| 季节性伪相关 → $A_{prior}$ 虚假边 | 先做 SSTA 去季节化；用 lagged corr 而非同期 corr |
| N 过大导致 DAGG 的 $E_A E_A^T$ 爆显存 | 区域子集 + 降采样；或分块/采样（GraphSAGE 式邻居采样） |
| 训练不稳定 | 梯度裁剪、L1 损失、节点嵌入共享（Bai 原文 tricks） |
| 多步误差累积 | 直接多步输出（非自回归）；或 scheduled sampling |
| 陆地/缺测像素 | 掩膜后仅保留海洋节点参与图与损失 |

---

## 5. 近期（未来 2 周）优先级 TODO

1. 清理 `Temperature_predictor/data/raw/`，确认 2015–2024 OISST 完整性
2. 实现 `preprocess.py`：climatology + SSTA + 掩膜 + 保存 `[T, N]` 张量
3. 实现 `build_prior.py` 的 Pearson + top-k，产出首版 $A_{prior}$ 并可视化
4. 搭 AGCRN 最小可跑版本（Bai 官方仓库：https://github.com/LeiBAI/AGCRN 作参考）
5. 跑通 persistence / climatology 两个零成本基线，建立评估脚手架

