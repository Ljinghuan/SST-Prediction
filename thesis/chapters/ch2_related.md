# 第 2 章 相关工作

本章从三个相互关联的方向梳理与本文研究密切相关的已有工作：海表温度预测的传统与深度学习方法（2.1 节）、时空图神经网络的发展脉络（2.2 节）、以及面向 SST/海洋场预测的图方法最新进展（2.3 节）；最后总结现有工作的优势与不足，指出本文的研究切入点（2.4 节）。

## 2.1 海表温度预测方法

### 2.1.1 传统统计与数值模式方法

经典 SST 预测方法主要分为两类。一是基于动力学方程的数值模式，如 NOAA CFSv2、ECMWF SEAS5 等耦合海气模式，通过求解 Navier–Stokes 方程与海洋热力学方程组得到未来海温场。该类方法物理含义清晰、可外推到训练域之外的极端事件，但计算代价巨大、依赖参数化方案，且对初值场质量敏感。二是基于统计的方法，如 ARIMA、经验正交函数（EOF）回归、典型相关分析（CCA）等，将 SST 时空场降维后做线性外推；这类方法计算量小，但难以处理 SST 演化中的强非线性。NOAA OISST v2.1 作为本文使用的数据源，由 Reynolds 等 [@reynoldsDailyHighresolutionblendedAnalyses2007] 通过最优插值融合卫星 AVHRR、船舶与浮标观测得到，已成为 SST 短期研究的事实基准。

### 2.1.2 基于深度学习的方法

近年来，深度学习方法以其强非线性拟合能力与端到端可训练性，在 SST 预测领域取得了一系列进展。在序列建模方面，Hochreiter 与 Schmidhuber 提出的 LSTM [@hochreiterLongShorttermMemory1997] 能够缓解长序列梯度消失问题，被广泛用于格点级 SST 时序外推；在场预测方面，Shi 等提出的 ConvLSTM [@shiConvolutionalLSTMNetwork2015] 在传统 LSTM 单元中引入 2D 卷积，可同时建模空间局部相关与时间长依赖，在降水临近预报、SST 短期预测等任务中得到广泛应用。然而 ConvLSTM 受限于卷积核的局部感受野，难以刻画 ENSO 期间赤道太平洋的跨海盆同相关变化。Ham 等 [@hamDeepLearningMultiyear2019] 进一步证明了深度学习方法在多年 ENSO 预测上可超越经典动力学模式的技巧上限，但其使用的 CNN 仍然受限于规则网格表征。GraphCast [@lamLearningSkillfulMediumrange2023] 则首次将图神经网络系统性应用于全球中期天气预测，在多项指标上超越 ECMWF 高分辨率模式，标志着图方法在地球科学场预测中的巨大潜力，启发了本文将图思想引入 SST 场预测的研究路径。

## 2.2 时空图神经网络

### 2.2.1 图卷积网络基础

Kipf 与 Welling [@kipfSemisupervisedClassificationGraph2017] 提出的图卷积网络（GCN）通过对图 Laplacian 进行一阶切比雪夫近似，给出了简洁高效的层级消息传递公式

$$
H^{(l+1)}=\sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right),
$$

其中 $\tilde{A}=A+I$。该形式简洁、可微、并行高效，已成为图神经网络的基础构件。其后 GAT、GraphSAGE 等变体引入注意力机制与邻居采样，进一步扩展了图卷积的表达能力。

### 2.2.2 时空图卷积网络

为同时建模空间图依赖与时间动态，研究者提出了多种时空图神经网络架构。Yu 等提出的 STGCN [@yuSpatioTemporalGraphConvolutional2018] 采用"图卷积 + 时间一维卷积"的全卷积级联结构，在交通流预测任务中相比 LSTM 类基线取得显著改进；Yan 等的 ST-GCN [@yanSpatialTemporalGraph2018] 将时空图卷积应用于骨架动作识别，验证了图结构对非欧几里得时空数据的天然适配性。然而上述方法均依赖事先给定的固定邻接矩阵（如交通路网、人体骨架），无法适应 SST 等无明确物理图结构的场景。

### 2.2.3 自适应邻接矩阵

针对邻接矩阵难以预定义的场景，Bai 等提出的自适应图卷积循环网络（AGCRN [@baiAdaptiveGraphConvolutional2020]）通过两个低秩节点嵌入矩阵 $E_1, E_2\in\mathbb{R}^{N\times d_e}$ 动态构造邻接

$$
A_{\mathrm{adapt}}=\mathrm{softmax}\!\left(\mathrm{ReLU}(E_1 E_2^\top)\right),
$$

让模型在训练过程中自动学习最优图结构；同时引入节点自适应参数（NAPL）以建模节点级异质动态。AGCRN 在交通流预测多个基准上达到 SOTA，并在不依赖任何先验图的情况下展现出良好泛化能力，是本文图模型的主要参考。

## 2.3 基于图神经网络的 SST 与海洋场预测

将图思想应用于 SST 等海洋场预测是近三年才兴起的方向。Liang 等 [@liangGraphMemoryNeural2023] 提出图记忆神经网络（GMNN），将每个海域视为节点，并以记忆模块缓存长期依赖，在 SST 月度预测上取得改进；其图结构基于地理近邻人工构造，未充分挖掘远程相关。Li 等 [@liSpatiotemporalSeaSurface2023] 提出静态-动态可学习个性化图卷积网络，将固定图与样本级动态图相结合，在 SST 预测上取得 SOTA，但模型复杂度较高。Ning 等 [@ningGraphBasedDeepLearning2023] 系统研究了基于 SST 相关性的先验图构造方法，提出 Top-$k$ 稀疏化与可视化 ENSO teleconnection 的范式，是本文 4.2–4.3 节先验图构造方法的直接参考；其工作主要集中在月度 ENSO 预测，未结合自适应图。Taylor 与 Feng [@taylorDeepLearningModel2022] 则探讨了纯 CNN 在月度 SST 异常预测上的能力，间接验证了远程依赖建模的重要性。

## 2.4 现有工作小结与本文切入点

综上，现有 SST 与海洋场预测研究存在以下不足：

1. 经典 ConvLSTM 等局部模型受限于卷积感受野，难以刻画 ENSO 等远程相关；
2. 已有 GNN 工作多采用固定地理近邻图或单一可学习图，缺乏对"物理先验 + 数据驱动"的系统融合；
3. 既有先验图构造较少考虑波动传播时延（如赤道 Kelvin 波），多以同期 Pearson 相关为主，遗漏了部分物理边。

针对上述问题，本文以热带太平洋日尺度 SST 场预测为目标，提出基于"滞后最大绝对相关 + Top-$k$ 稀疏化 + 自适应图融合"的 AGCRN 增强方案：以滞后相关挖掘的先验图为物理先验，以可学习自适应图表达数据驱动模式，并以统一时间切分与多基线对比验证有效性。具体方法将在第 3–6 章详细展开。
