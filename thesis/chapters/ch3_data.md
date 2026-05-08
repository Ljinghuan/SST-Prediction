# 第 3 章 数据与预处理

> 草稿（markdown + KaTeX）。最终粘到 Word 模板时按规范调字号（宋体小四）、行距（固定 24）、公式用 MathType 重排。
> 公式编号 (3-x)、图编号 图 3-x、表编号 表 3-x。

---

## 3.1 数据来源

本研究采用美国国家海洋和大气管理局（NOAA）发布的 **OISST v2.1**（Optimum Interpolation Sea Surface Temperature, version 2.1）逐日海表温度再分析产品 [Reynolds 2007]。该数据集融合卫星（AVHRR）、船舶、浮标等多源观测，经最优插值得到全球均匀网格场，已成为短期 SST 研究中事实上的基准数据集。

主要技术参数如表 3-1 所示。

**表 3-1 OISST v2.1 数据集主要参数**

| 项目 | 取值 |
|---|---|
| 空间分辨率 | 0.25° × 0.25° |
| 时间分辨率 | 日均 |
| 全球网格规模 | 720 × 1440 |
| 变量 | sst, anom, err, ice |
| 经度范围 | 0°–360°（步长 0.25°） |
| 纬度范围 | −89.875°–89.875° |
| 文件格式 | NetCDF4 |
| 时间维度结构 | (time, zlev, lat, lon)，zlev=0 |

考虑到全球海域中纬向尺度最大、信号最强的物理过程为 ENSO 等热带太平洋海气相互作用，且 ENSO 信号能够通过遥相关（teleconnection）影响全球气候系统，本研究将数据空间范围限定为**热带太平洋**：

$$
\mathcal{R} = \{(\phi,\lambda) \mid -30° \le \phi \le 30°,\, 120° \le \lambda \le 280°\}\tag{3-1}
$$

时间范围选择 2015 年 1 月 1 日至 2024 年 12 月 31 日共 10 年，覆盖 2015/16 强 El Niño、2020–2022 三重 La Niña 等多种 ENSO 位相，便于模型学习不同情形下的时空演化规律。

经过下载与文件可读性检查（剔除 14 个 HDF 解码失败的 .nc 文件），最终可用样本数 $T_0 = 3639$ 天。

## 3.2 数据流总览

数据预处理整体流程如图 3-1 所示，依次包括：①区域子集与空间降采样，②去季节化（生成 SSTA），③海陆掩膜，④节点扁平化，⑤时间切分，⑥归一化。每一步的设计动机与公式将在 3.3–3.7 节展开。

![](D:\Graduate-Deisgn\thesis\figures\fig3-1_pipeline.png)

> 图 3-1 数据预处理流程图（待绘制：原始 .nc → 区域裁剪 → 降采样 → climatology → SSTA → 掩膜 → 扁平化 → 切分 → 归一化）

## 3.3 区域子集与空间降采样

原始 0.25° 网格在区域 $\mathcal{R}$ 内含 240×640 ≈ 1.5×10⁵ 个像素，直接将每个像素视为一个图节点会导致后续自适应邻接矩阵 $A_{adapt} \in \mathbb{R}^{N\times N}$ 在显存中无法承受（约 9×10¹⁰ 元素）。

为兼顾空间细节与计算可行性，采用 4×4 平均池化将分辨率降至约 1°：

$$
\widehat{\mathrm{SST}}_{i,j,t} = \frac{1}{16}\sum_{u=0}^{3}\sum_{v=0}^{3} \mathrm{SST}_{4i+u,\,4j+v,\,t}\tag{3-2}
$$

降采样后网格规模为 $H=60,\ W=160$，像素数降至 $H\times W = 9600$。该尺度仍能分辨 ENSO 中尺度结构（典型空间尺度 ≥1000 km），同时使后续节点数控制在可处理范围。

## 3.4 去季节化：异常场 SSTA

SST 时间序列由季节性循环、年际/年代际变化与短期噪声共同组成。其中季节项幅度可达 5–10 °C，远大于 ENSO 异常（典型 ±2–3 °C）。若直接以原始 SST 作为预测目标，模型学习容量将被季节性"占据"，而难以捕捉到本研究真正关注的 ENSO 等异常信号；同时季节项也会在节点间引入虚假同步性，干扰第 4 章中基于相关的图构造。

因此，本研究采用经典的"逐 day-of-year（DOY）扣除气候态"方案 [Reynolds 2007] 计算海表温度异常（Sea Surface Temperature Anomaly, SSTA）。

定义参考期 $\mathcal{T}_{\mathrm{clim}} = [2015,2021]$（与训练集时段一致，避免标签泄漏），逐格点的气候态为：

$$
\overline{\mathrm{SST}}_{i,j}^{(d)} = \frac{1}{|\mathcal{T}_{i,j}^{(d)}|}\sum_{t\in \mathcal{T}_{i,j}^{(d)}} \mathrm{SST}_{i,j,t},\quad d \in \{1,\dots,366\}\tag{3-3}
$$

其中 $\mathcal{T}_{i,j}^{(d)}$ 为参考期内 DOY 等于 $d$ 的所有样本时间戳集合。对未出现的 DOY（如非闰年的 366 日）采用沿 DOY 轴的线性插值与首尾填充补全。

SSTA 由原始 SST 减去对应日期的气候态得到：

$$
\mathrm{SSTA}_{i,j,t} = \mathrm{SST}_{i,j,t} - \overline{\mathrm{SST}}_{i,j}^{(\mathrm{doy}(t))}\tag{3-4}
$$

去季节化后 SSTA 的全样本均值接近 0，标准差约 0.63 °C（见 3.7 节），季节信号被有效剥离。

![](D:\Graduate-Deisgn\thesis\figures\fig3-2_ssta_timeseries.png)

> 图 3-2 去季节化前后某代表点（赤道中太平洋, ≈0°N, 200°E, Niño3.4 区）SST 与 SSTA 时间序列对比

## 3.5 海陆掩膜与节点扁平化

OISST 在陆地像素填充 NaN。本研究通过"全时段恒为有效值"判据构建海洋掩膜，避免任何含 NaN 的格点进入图与损失计算：

$$
\mathcal{M}_{i,j} = \prod_{t=1}^{T_0} \mathbb{1}\!\left[\mathrm{SST}_{i,j,t}\in\mathbb{R}\right]\tag{3-5}
$$

其中 $\mathbb{1}[\cdot]$ 为指示函数。$\mathcal{M}_{i,j}=1$ 表示该格点为始终可观测的海洋节点。

降采样后 $H\times W = 9600$ 个像素中，掩膜保留 $N = 8977$ 个海洋节点，占比约 93.5%；为进一步控制图的规模并保留典型海盆结构，本研究在主实验中采用 $\text{coarsen}=8$ 的更激进降采样（H=30, W=80），此时 $N = 2276$（详见表 3-2）。该规模既能完整呈现热带太平洋纬向条带结构，又使第 4 章中 $A_{prior}\in\mathbb{R}^{N\times N}$ 占用内存仅约 20 MB，便于训练阶段加载到显存。

将三维数组 $[T_0, H, W]$ 按掩膜筛选后得到二维节点矩阵：

$$
\mathbf{X} \in \mathbb{R}^{T_0 \times N},\quad \mathbf{X}_{t,n} = \mathrm{SSTA}_{i_n,j_n,t}\tag{3-6}
$$

其中 $(i_n, j_n)$ 为第 $n$ 个海洋节点在 2D 网格上的索引。同时记录每个节点的地理坐标 $\mathbf{C}\in\mathbb{R}^{N\times 2}$（纬度、经度），供第 4 章构图与可视化使用。

![](D:\Graduate-Deisgn\thesis\figures\fig3-3_ocean_mask.png)

> 图 3-3 海洋掩膜 $\mathcal{M}$ 在热带太平洋区域的 2D 分布（白：海洋节点；黑：陆地/缺测）

## 3.6 时间切分

按时间顺序将样本划分为训练、验证、测试三段，确保模型评估发生在严格未来时段，避免数据泄漏：

**表 3-2 数据集时间切分（$\text{coarsen}=8$, $N=2276$）**

| 子集 | 时段 | 样本数 |
|---|---|---|
| 训练 | 2015-01-01 — 2021-12-31 | 2544 |
| 验证 | 2022-01-01 — 2022-12-31 | 365 |
| 测试 | 2023-01-01 — 2024-12-31 | 730 |
| **合计** | 2015-01-01 — 2024-12-31 | **3639** |

特别地，3.4 节气候态参考期与训练集时段保持一致（2015–2021），避免在计算 SSTA 时引入测试期信息。

## 3.7 归一化

不同节点的 SSTA 方差差异较大（赤道中太平洋方差约 1 °C²，副热带海域方差约 0.1 °C²）。为使深度模型优化器（Adam）的初始学习率对各节点等效，并控制损失函数尺度，对 SSTA 做全局 z-score 归一化：

$$
\widetilde{\mathbf{X}}_{t,n} = \frac{\mathbf{X}_{t,n}-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}\tag{3-7}
$$

其中均值与标准差**仅在训练集 SSTA 上估计**：

$$
\mu_{\mathrm{train}} = \frac{1}{|S_{tr}|N}\sum_{t\in S_{tr}}\sum_{n=1}^{N} \mathbf{X}_{t,n},\quad
\sigma_{\mathrm{train}} = \sqrt{\frac{1}{|S_{tr}|N}\sum_{t\in S_{tr}}\sum_{n=1}^{N}\left(\mathbf{X}_{t,n}-\mu_{\mathrm{train}}\right)^2}\tag{3-8}
$$

在 $S_{tr}=[0,2544)$ 上实测得 $\mu_{\mathrm{train}}\approx 1.46\times 10^{-9}\approx 0$（去季节化已使 SSTA 接近零均值），$\sigma_{\mathrm{train}}=0.6320\,°\mathrm{C}$。该统计量同时用于验证与测试集，不再重新估计。

模型评估阶段，将归一化输出依次反变换可还原为原始 SST：

$$
\widehat{\mathrm{SST}}_{i,j,t} = \widetilde{\mathbf{X}}_{t,n}\cdot\sigma_{\mathrm{train}} + \mu_{\mathrm{train}} + \overline{\mathrm{SST}}_{i,j}^{(\mathrm{doy}(t))}\tag{3-9}
$$

## 3.8 预处理产物

预处理脚本 `preprocess.py` 输出至 `Temperature_predictor/data/processed/`，主要文件如表 3-3 所示。

**表 3-3 预处理产物**

| 文件 | 形状 / 类型 | 说明 |
|---|---|---|
| `sst_raw.npy` | $[T_0, N]$ float32 | 原始 SST，反算评估用 |
| `climatology.npy` | $[366, N]$ float32 | 逐 DOY 气候态 |
| `ssta.npy` | $[T_0, N]$ float32 | 去季节化 SSTA（°C） |
| `ssta_norm.npy` | $[T_0, N]$ float32 | 归一化 SSTA，模型主输入 |
| `mask_2d.npy` | $[H, W]$ bool | 海陆掩膜，可视化复原用 |
| `coords.npy` | $[N, 2]$ float32 | 节点 (lat, lon)，构图用 |
| `dates.npy` | $[T_0]$ datetime64 | 时间轴 |
| `splits.json` | dict | 训/验/测半开区间索引 |
| `norm_stats.json` | dict | $\mu_{tr},\sigma_{tr}$，反归一化用 |
| `meta.json` | dict | 全部预处理参数与形状信息 |

## 3.9 本章小结

本章设计并实现了从原始 OISST .nc 文件到模型可读张量的完整预处理流水线，主要工作包括：

1. 限定热带太平洋区域并采用 4×4（主实验 8×8）平均池化将网格降至 ~1°（~2°），把节点规模从 1.5×10⁵ 降至 2276，使后续基于全连接邻接矩阵的图模型在显存上可行；
2. 采用逐 DOY 气候态扣除生成 SSTA，剥离了量级远大于异常信号的季节项，并避免季节项在图构造中引入虚假同步性；
3. 通过"全时段有限值"判据构建严格海陆掩膜，扁平化为 $[T, N]$ 节点矩阵；
4. 按时间顺序进行 train/val/test 切分（2015–2021 / 2022 / 2023–2024），气候态参考期与训练集对齐，归一化统计量仅在训练集上估计，从根本上杜绝标签泄漏；
5. 输出 9 类标准化产物（含模型输入、反算所需统计量、可视化元数据），为第 4 章先验图构造与第 5 章 AGCRN 训练奠定基础。

---

## TODO（写作时再补）

- [ ] Reynolds 2007 等参考文献完整引用（暂用占位符）
- [x] 图 3-1 流程图（建议用 draw.io 或 mermaid 重画，统一中文字体）
- [x] 图 3-2 SSTA 时间序列对比（写一个小脚本从 `sst_raw.npy` + `ssta.npy` 取赤道中太点画出，300 dpi）
- [x] 图 3-3 掩膜 2D 图（`mask_2d.npy` + `cmap='gray'`，加经纬度刻度）
- [ ] 把海洋占比、最终 N=2276 等数字与 `meta.json` 对齐再校一遍
