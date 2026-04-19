---
type: literature
title: Spatial temporal graph convolutional networks for skeleton-based action recognition
author: Sijie Yan
year: 2018
venue: ""
doi: ""
bibkey: yanSpatialTemporalGraph2018a
pdf: lit_1/pdfs/yanSpatialTemporalGraph2018a.pdf
tags: []
status: skimmed
rating: 2
summary: 用ST-GCN对人体骨骼建模，学习时间和空间的模式，对动作进行识别
keywords:
  - ST-GCN
---

# 一行总结
<!-- 一句概括论文做了什么 -->
本研究提出了Spatial Temporal Graph Convolutional Networks（ST-GCN）方法，通过引入图神经网络中的卷积操作来处理人体骨骼数据，并利用这些操作进行深度学习以识别人体动作。在Kinetics和NTU-RGBD这两个大规模数据集上，该方法显著提高了基于骨架的动作识别性能，相较于传统的基于手工特征或遍历规则的方法，实现了更好的泛化能力。
# 贡献点
1. 
2. 

# 方法要点（画个小图或伪代码）
- 模型结构：
- 输入/输出维度：
- 关键公式/伪代码：
这个文献的输入是一个由骨架序列构成的空间时间图，其中每个节点代表人体关节，边则根据人体结构中的自然连接和连续帧之间的连接来定义。具体来说，输入数据是由图节点上的关节坐标向量构成的。

输出则是通过应用多层空间时间图卷积操作生成的更高层次的特征图，然后通过标准的Softmax分类器对这些特征图进行分类，以确定相应的动作类别。整个模型采用端到端的方式进行训练，使用反向传播算法进行优化。
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

 1. 分区策略可增强模型的表达能力：海温场中可根据距离分区（如近岸与远海）或根据海洋环流模式（类似空间配置分区中的向心/离心组）来定义子图，以捕捉区域特异性。
![[Pasted image 20251206221922.png]]
2. 动态学习边的重要性权重：海温场中不同区域（如上升流区）重要性不同，可类似加入注意力机制（如AGCRN中的自适应图卷积）来动态学习边权重，提升预测精度。
3. 对不同数据集作交叉验证：文献中使用PyTorch实现（第3.6节），并提供了数据增强（如随机仿射变换）和训练细节（如学习率调度），本题目的技术栈（PyTorch Geometric Temporal）可参考这些实践。此外，实验部分（第4节）在多个数据集上的评估方法（如交叉验证）可借鉴到海温场预测的基准测试中。
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
-![[lit_1/pdfs/yanSpatialTemporalGraph2018a.pdf]] 

# 个人短评 / 下一步

