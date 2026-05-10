"""
生成 AGCRN 模型架构示意图：
  输入序列 → 2×AGCRNCell（NAPL-GCN + GRU）→ 输出头 → 预测序列
图中同时标注邻接融合 A_prior + A_adapt → A_used 的侧边。
输出：fig_model_arch.png（300 dpi）
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

# 使用 Windows 自带中文字体，避免乱码
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

OUT = pathlib.Path(__file__).parent / "fig_model_arch.png"

# ── 颜色 ──────────────────────────────────────────────────────────────────
C_INPUT   = "#D6EAF8"
C_CELL    = "#D5F5E3"
C_GRAPH   = "#FAD7A0"
C_FUSE    = "#F9E79F"
C_HEAD    = "#E8DAEF"
C_OUTPUT  = "#FDFEFE"
C_ARROW   = "#2C3E50"
C_BORDER  = "#2C3E50"

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")

# ── 辅助函数 ──────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel="", fontsize=9, radius=0.25):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.2, edgecolor=C_BORDER, facecolor=color, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.18, label,   ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555555", zorder=4)
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)

def arrow(ax, x0, y0, x1, y1, color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.4, mutation_scale=12), zorder=5)

def dashed_arrow(ax, x0, y0, x1, y1, label="", color="#7F8C8D"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                linestyle="dashed", mutation_scale=11), zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.08, my, label, fontsize=7.5, color=color, va="center", zorder=6)

# ══ 布局常量 ══════════════════════════════════════════════════════════════
Y0 = 1.5     # 主干 y
BH = 2.0     # box 高
BW_IN  = 1.2
BW_EMB = 1.4
BW_CELL= 2.2
BW_HEAD= 1.4
BW_OUT = 1.2

X_IN   = 0.3
X_EMB  = X_IN + BW_IN + 0.4
X_C1   = X_EMB + BW_EMB + 0.5
X_C2   = X_C1 + BW_CELL + 0.5
X_HEAD = X_C2 + BW_CELL + 0.5
X_OUT  = X_HEAD + BW_HEAD + 0.4

# ── 图融合（顶部）────────────────────────────────────────────────────────
Y_TOP = 4.2
box(ax, 0.3,  Y_TOP, 1.3, 0.7, C_GRAPH, "$A_{prior}$", fontsize=9)
box(ax, 2.1,  Y_TOP, 1.3, 0.7, C_GRAPH, "$A_{adapt}$",
    sublabel=r"softmax(ReLU($E_a E_a^\top$))", fontsize=8.5)
box(ax, 4.15, Y_TOP, 1.5, 0.7, C_FUSE,
    r"$A_{used}$",
    sublabel=r"$\lambda A_{prior}+(1{-}\lambda)A_{adapt}$", fontsize=8)

# 融合箭头
arrow(ax, 0.3+1.3,  Y_TOP+0.35, 4.15,          Y_TOP+0.35)
arrow(ax, 2.1+1.3,  Y_TOP+0.35, 4.15,          Y_TOP+0.35)
ax.text(0.3+1.3+0.5, Y_TOP+0.5, r"$\lambda$", fontsize=8, color="#555", zorder=6)

# A_used → Cell1, Cell2（虚线向下）
for xc in [X_C1 + BW_CELL/2, X_C2 + BW_CELL/2]:
    dashed_arrow(ax, 4.9, Y_TOP, xc, Y0 + BH, color="#E67E22")

# ── 主干 ──────────────────────────────────────────────────────────────────
# 输入
box(ax, X_IN, Y0, BW_IN, BH, C_INPUT,
    "输入序列", sublabel=r"$X\in\mathbb{R}^{B\times T_{in}\times N}$", fontsize=8.5)

# 节点嵌入 E
box(ax, X_EMB, Y0, BW_EMB, BH, C_GRAPH,
    "节点嵌入", sublabel=r"$E\in\mathbb{R}^{N\times d_e}$", fontsize=8.5)

# Cell 1
box(ax, X_C1, Y0, BW_CELL, BH, C_CELL,
    "AGCRNCell × T_in", sublabel="NAPL-GCN + GRU（第 1 层）", fontsize=8.5)

# Cell 2
box(ax, X_C2, Y0, BW_CELL, BH, C_CELL,
    "AGCRNCell × T_in", sublabel="NAPL-GCN + GRU（第 2 层）", fontsize=8.5)

# 输出头
box(ax, X_HEAD, Y0, BW_HEAD, BH, C_HEAD,
    "输出头", sublabel=r"Conv$_{1\times1}$" + "\n" + r"$h\to T_{out}$", fontsize=8.5)

# 输出
box(ax, X_OUT, Y0, BW_OUT, BH, C_OUTPUT,
    "预测序列", sublabel=r"$\hat{Y}\in\mathbb{R}^{B\times T_{out}\times N}$", fontsize=8.5)

# ── 主干箭头 ─────────────────────────────────────────────────────────────
arrow(ax, X_IN + BW_IN,   Y0 + BH/2, X_EMB,            Y0 + BH/2)
arrow(ax, X_EMB + BW_EMB, Y0 + BH/2, X_C1,             Y0 + BH/2)
arrow(ax, X_C1 + BW_CELL, Y0 + BH/2, X_C2,             Y0 + BH/2)
arrow(ax, X_C2 + BW_CELL, Y0 + BH/2, X_HEAD,           Y0 + BH/2)
arrow(ax, X_HEAD+BW_HEAD, Y0 + BH/2, X_OUT,            Y0 + BH/2)

# E → Cell1, Cell2（虚线斜向）
for xc in [X_C1 + BW_CELL/2, X_C2 + BW_CELL/2]:
    dashed_arrow(ax, X_EMB + BW_EMB/2, Y0, xc, Y0 - 0.1 + 0.12, color="#2980B9")

# ── 图例 ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=C_BORDER, label="输入 / 输出"),
    mpatches.Patch(facecolor=C_CELL,   edgecolor=C_BORDER, label="AGCRNCell（NAPL-GCN+GRU）"),
    mpatches.Patch(facecolor=C_GRAPH,  edgecolor=C_BORDER, label="图 / 嵌入"),
    mpatches.Patch(facecolor=C_FUSE,   edgecolor=C_BORDER, label="邻接融合"),
    mpatches.Patch(facecolor=C_HEAD,   edgecolor=C_BORDER, label="输出头"),
]
ax.legend(handles=legend_items, loc="lower center",
          ncol=5, fontsize=7.8, framealpha=0.85,
          bbox_to_anchor=(0.5, -0.02))

ax.set_title("AGCRN 模型整体结构（$L=2$ 层，$K=2$ 阶，直接多步输出头）",
             fontsize=10, pad=8)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved → {OUT}")
