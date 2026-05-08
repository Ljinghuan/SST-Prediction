# 毕业论文

> 题目：基于自适应时空图神经网络的海温场多步预测

## 目录结构

```
thesis/
├── chapters/   分章节 markdown 草稿（写作时用，最后粘进 Word）
├── figures/    论文用图（300 dpi，统一样式；源图见 data/processed/graph/figs/）
├── tables/     表格源数据（.xlsx / .md）
├── refs/       参考文献（.bib + GB/T 7714 格式化结果）
├── forms/      海大附件表格（封面/任务书/开题/中期/进度/答辩）★ 不进 git
└── defense/    答辩 PPT 与脚本
```

## 进度

- [x] Stage A 数据预处理（见 `Temperature_predictor/src/data/preprocess.py`）
- [x] Stage B 先验图构造（见 `Temperature_predictor/src/graph/build_prior.py`）
- [ ] Stage C AGCRN 模型
- [ ] Stage D 基线 + 评估
- [ ] Stage E 消融
- [ ] 第 3、4 章正文撰写（数据已稳定，可优先动笔）

## 写作约定

- 字号：正文宋体小四，固定行距 24（海大规范）
- 公式/图/表编号按一级标题：图 3-1, 表 4-2, 公式 (3-1)
- 参考文献：GB/T 7714—2015，理工科 ≥ 10 篇，外文 ≥ 2
- markdown 草稿用 KaTeX 写公式，最后转 MathType 进 Word
