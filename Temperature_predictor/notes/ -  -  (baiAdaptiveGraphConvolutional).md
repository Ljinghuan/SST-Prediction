---
type: literature
title: "Adaptive graph convolutional recurrent network for traffic forecasting"
author: ""
year: 
venue: ""
doi: ""
bibkey: "baiAdaptiveGraphConvolutional"
pdf: "lit_1/pdfs/baiAdaptiveGraphConvolutional.pdf"
tags: []
status: "not_read"
rating: 0
summary: ""
keywords: []
---

# 一行总结
<!-- 一句概括论文做了什么 -->
本研究提出了 <方法名>，通过 <关键机制> 来解决 <问题>，在 <数据集> 上将 RMSE 从 <> 降低到 <>，显示出优于传统 ConvLSTM 的性能。

# 贡献点
1. 
2. 

# 方法要点（画个小图或伪代码）
- 模型结构：
- 输入/输出维度：
- 关键公式/伪代码：

# 图构造（如果是 GNN）
- 邻接定义： e.g., distance kNN / correlation / learned (AGCRN)
- edge weight 公式 / threshold

# 数据集与预处理
- 数据来源： e.g., NOAA OISST, GHRSST, ERA5
- 网格/时间范围：
- 关键预处理（掩膜/插值/normalization）

# 实验设置与结果（关键数字）
- baseline：
- 评价指标（RMSE/MAE/SSIM/Correlation）：
- 关键结果（把表格/数字抄下来）：

# 可借鉴点（写到你实验的 TODO）
- 
- 

# 缺点 / 风险（会影响复现或迁移到 SST）
- 
- 

# 复现清单（要做的 5 个步骤）
1. 下载数据 / 时间段：
2. 导出样本并做 normalization：
3. 实现最小模型（tiny run）：
4. 训练超参（试验哪些）：
5. 生成对比图（persistence, ConvLSTM, AGCRN）：

# 高亮 / 直接引用（引号 + 页码）
> "引用短句" (p. 3)

# 相关笔记链接
-![[lit_1/pdfs/baiAdaptiveGraphConvolutional.pdf]][[Bai  - Adaptive Graph Convolutional Recurrent Network for.pdf]]

# 个人短评 / 下一步

