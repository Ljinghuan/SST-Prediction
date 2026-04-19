# -*- coding: utf-8 -*-
"""
PA-AGCRN 流程图一键生成器
运行前请确保已安装 graphviz：
    pip install graphviz
    Windows 用户还需把 graphviz/bin 加入系统 PATH
"""
from graphviz import Digraph

# 1. 全局画布设置
dot = Digraph(
    name='PA-AGCRN_flow',
    graph_attr={
        'rankdir': 'LR',          # 从左到右
        'bgcolor': 'white',
        'fontname': 'Microsoft YaHei',
        'fontsize': '12'
    },
    node_attr={
        'fontname': 'Microsoft YaHei',
        'fontsize': '11'
    },
    edge_attr={
        'fontname': 'Microsoft YaHei',
        'fontsize': '10'
    }
)

# 2. 定义颜色与样式
COLOR_DATA   = '#F5F5F5'   # 浅灰 – 数据/结果
COLOR_PROC   = '#E6F2FF'   # 淡蓝 – 处理矩形
COLOR_PARAM  = '#FFF2E6'   # 淡橙 – 可学习参数
COLOR_CORE   = '#F0E6FF'   # 淡紫 – 核心单元
COLOR_DECIDE = '#FFE6E6'   # 淡红 – 判断/过滤

# 3. 节点注册函数（简化重复代码）
def add_node(id, label, shape, color, **kw):
    dot.node(id, label=label, shape=shape, style='filled', fillcolor=color, **kw)

# 4. =================  第一阶段：数据输入与预处理  =================
with dot.subgraph(name='cluster_pre') as c:
    c.attr(label='第一阶段：数据输入与预处理', style='rounded,filled', fillcolor='#F0F0F0')
    add_node('raw',  '原始 ERA5-SST\n经纬度网格数据',    'parallelogram', COLOR_DATA)
    add_node('des',  '去季节化处理\n（提取 SSTA 异常场）', 'rect',          COLOR_PROC)
    add_node('mask', '陆地掩膜 &\n节点索引化',           'rect',          COLOR_PROC)
    c.edge('raw', 'des',  label='X∈ℝ^{N×T}', color='blue')
    c.edge('des', 'mask', color='blue')

# 5. =================  第二阶段：双轨图拓扑构建  =================
with dot.subgraph(name='cluster_graph') as c:
    c.attr(label='第二阶段：双轨图拓扑构建', style='rounded,filled', fillcolor='#F0F0F0')
    # 路径 A – 物理先验
    add_node('lag',  '滞后相关性分析\n(Lagged Correlation)', 'rect', COLOR_PROC)
    add_node('topk', 'Top-K 稀疏化\n阈值过滤',          'diamond', COLOR_DECIDE)
    add_node('Ap',   '有向先验邻接矩阵\nÃ_prior',        'box3d', COLOR_DATA)
    c.edge('lag', 'topk', color='blue')
    c.edge('topk', 'Ap', color='blue')
    # 路径 B – 自适应
    add_node('emb',  '可学习节点\n嵌入矩阵 E',         'hexagon', COLOR_PARAM)
    add_node('gen',  '矩阵乘法自生成\n(E·E^T)',          'rect',    COLOR_PROC)
    add_node('Aa',   '自适应邻接矩阵\nA_adaptive',       'box3d',   COLOR_DATA)
    c.edge('emb', 'gen', color='red', style='dashed')
    c.edge('gen', 'Aa', color='red', style='dashed')
    # 融合
    add_node('fuse', '矩阵融合\n(Ã = αÃ_prior + βA_adaptive)', 'circle', COLOR_PROC)
    c.edge('Ap', 'fuse', color='blue')
    c.edge('Aa', 'fuse', color='red', style='dashed')
    # 创新点气泡
    c.node('bubble', '物理先验与\n数据驱动融合', shape='note', fillcolor='#FFFFCC')

# 6. =================  第三阶段：AGCRN 核心单元  =================
with dot.subgraph(name='cluster_core') as c:
    c.attr(label='第三阶段：核心时空计算单元 (AGCRN Cell)', style='rounded,dashed', fillcolor=COLOR_CORE)
    add_node('napl', 'NAPL\n(节点自适应参数学习)', 'hexagon', COLOR_PARAM)
    add_node('gcn',  '图卷积层\n(GCN)',            'rect',    COLOR_PROC)
    add_node('gru',  'GRU 门控单元\n(重置门/更新门)', 'rect',    COLOR_PROC)
    c.edge('napl', 'gcn', label='W=E·𝒲', color='red', style='dashed')
    c.edge('gcn', 'gru', color='blue')
    c.edge('gru', 'gcn', label='递归', color='blue', style='dashed', constraint='false')

# 7. =================  第四阶段：预测输出与评估  =================
with dot.subgraph(name='cluster_out') as c:
    c.attr(label='第四阶段：预测输出与评估', style='rounded,filled', fillcolor='#F0F0F0')
    add_node('fc',   '全连接输出层', 'rect', COLOR_PROC)
    add_node('out',  '未来 30 天\nSSTA 预测场', 'parallelogram', COLOR_DATA)
    add_node('loss', '损失函数 & 指标\n(L1/L2/RMSE/ACC)', 'rect', COLOR_PROC)
    c.edge('fc', 'out', color='blue')
    c.edge('out', 'loss', color='blue')
    # 递归反馈
    c.edge('out', 'fc', label='递归反馈\n(Recursive Forecasting)', color='blue', style='dashed', constraint='false')

# 8. 跨阶段连线
dot.edge('mask', 'lag', color='blue')
dot.edge('mask', 'emb', color='blue')
dot.edge('fuse', 'gcn', color='blue')
dot.edge('gru', 'fc', color='blue')

# 9. 一键渲染
dot.render('PA_AGCRN_flow', format='png', cleanup=True)
dot.render('PA_AGCRN_flow', format='pdf', cleanup=True)
print('流程图已生成：PA_AGCRN_flow.png / .pdf')