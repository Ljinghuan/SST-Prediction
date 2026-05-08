"""
生成第 4 章用图（300 dpi，统一中文字体），输出到 thesis/figures/。

图 4-1: 先验图构造流程图（相关 → Top-k → 对称化 → 归一化）
图 4-2: 先验图邻接矩阵 A^sym 二值热图
图 4-3: 高度锚点节点及其 Top-k 邻居的地理连边图

用法：
    & 'd:\\Graduate-Deisgn\\Gradu-Design\\Scripts\\python.exe' \
        d:\\Graduate-Deisgn\\thesis\\scripts\\make_ch4_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ------------------------------------------------------------
# 中文字体配置 —— 与第 3 章一致
# ------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 150

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
PROC = _REPO / "Temperature_predictor" / "data" / "processed"
GRAPH = PROC / "graph"
FIG = _REPO / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 图 4-1 先验图构造流程图
# ------------------------------------------------------------
def fig_4_1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # (text, x, y, w, color)
    boxes = [
        ("训练集 SSTA\n$\\mathbf{X}_{tr}\\in\\mathbb{R}^{2544\\times 2276}$",
         0.3, 1.5, 2.4, "#cfe2f3"),
        ("逐节点 z-score\n$\\mathbf{Z}=(\\mathbf{X}-\\mu)/\\sigma$",
         3.0, 1.5, 2.2, "#d9ead3"),
        ("滞后相关\n$\\rho_{ij}(\\tau),\\ \\tau\\in[0,5]$\n取 $\\hat\\rho_{ij}=\\max_\\tau|\\rho|$",
         5.5, 1.5, 2.4, "#fff2cc"),
        ("Top-k 稀疏化\n$k=30$ 邻居/节点\n68,280 条有向边",
         8.2, 1.5, 2.4, "#f4cccc"),
        ("Max 对称化\n$A^{sym}=\\max(A, A^\\top)$\n85,586 条无向边",
         10.9, 1.5, 2.4, "#ead1dc"),
        ("加自环 + 归一化\n$\\tilde A=D^{-1/2}(A^{sym}+I)D^{-1/2}$",
         13.5, 1.5, 1.3, "#d0e0e3"),
    ]
    # 调整最后一个使其合理放置；改为换行布局
    boxes = [
        ("训练集 SSTA\n$\\mathbf{X}_{tr}\\in\\mathbb{R}^{2544\\times 2276}$",
         0.2, 1.7, 2.3, "#cfe2f3"),
        ("逐节点 z-score\n$\\mathbf{Z}=(\\mathbf{X}-\\mu)/\\sigma$",
         2.8, 1.7, 2.1, "#d9ead3"),
        ("滞后相关\n$\\rho_{ij}(\\tau),\\,\\tau\\in[0,5]$\n$\\hat\\rho_{ij}=\\max_\\tau|\\rho|$",
         5.2, 1.7, 2.4, "#fff2cc"),
        ("Top-k 稀疏化\n$k=30$\n68,280 有向边",
         7.9, 1.7, 2.1, "#f4cccc"),
        ("Max 对称化\n$A^{sym}=\\max(A,A^\\top)$\n85,586 无向边",
         10.3, 1.7, 2.4, "#ead1dc"),
        ("加自环 + 归一化\n$\\tilde A=D^{-1/2}(A^{sym}+I)D^{-1/2}$\n87,862 非零元",
         13.0, 1.7, 1.9, "#d0e0e3"),
    ]
    for text, x, y, w, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, 1.4,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.7, text, ha="center", va="center", fontsize=8.5)

    # 顺序连接：右端 -> 下个左端
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + boxes[i][3]
        x1 = boxes[i + 1][1]
        y = boxes[i][2] + 0.7
        ax.annotate(
            "", xy=(x1, y), xytext=(x0, y),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )

    ax.text(7.5, 4.3, "图 4-1 先验图 $A_{prior}$ 构造流程", ha="center", fontsize=11)
    out = FIG / "fig4-1_graph_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4-1] -> {out}")


# ------------------------------------------------------------
# 图 4-2 邻接矩阵热图
# ------------------------------------------------------------
def fig_4_2_adjacency() -> None:
    A = np.load(GRAPH / "A_raw_dense.npy")          # 二值，对称化后
    coords = np.load(PROC / "coords.npy")           # [N, 2] (lat, lon)

    # 节点排序：先纬度再经度（让赤道带靠中部、对角带更清晰）
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    A_ord = A[np.ix_(order, order)]

    N = A.shape[0]
    nnz = int((A > 0).sum())
    avg_deg = nnz / N

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(A_ord > 0, cmap="Blues", aspect="equal", interpolation="nearest")
    ax.set_xlabel("节点索引 $j$（按纬度→经度排序）")
    ax.set_ylabel("节点索引 $i$")
    ax.set_title(
        f"图 4-2 先验图邻接矩阵 $A^{{sym}}$ 二值热图\n"
        f"N={N}, 非零元={nnz} ({nnz//2} 无向边), 平均度={avg_deg:.1f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG / "fig4-2_adjacency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4-2] -> {out}, nnz={nnz}, avg_deg={avg_deg:.2f}")


# ------------------------------------------------------------
# 图 4-3 远程连边地理图
# ------------------------------------------------------------
def fig_4_3_teleconnections(n_anchors: int = 6, seed: int = 0) -> None:
    A = np.load(GRAPH / "A_raw_dense.npy")
    coords = np.load(PROC / "coords.npy")
    lats, lons = coords[:, 0], coords[:, 1]

    rng = np.random.default_rng(seed)
    deg = A.sum(axis=1)
    candidate = np.argsort(-deg)[: max(n_anchors * 3, 20)]
    anchors = rng.choice(candidate, size=min(n_anchors, len(candidate)), replace=False)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.scatter(lons, lats, s=3, c="lightgray", label="海洋节点", zorder=1)
    cmap = plt.get_cmap("tab10")

    legend_handles = []
    for k, a in enumerate(anchors):
        nbrs = np.where(A[a] > 0)[0]
        if nbrs.size == 0:
            continue
        c = cmap(k % 10)
        # 连边
        for j in nbrs:
            ax.plot([lons[a], lons[j]], [lats[a], lats[j]],
                    color=c, alpha=0.35, linewidth=0.5, zorder=2)
        # 邻居散点
        ax.scatter(lons[nbrs], lats[nbrs], s=10, color=c, alpha=0.8, zorder=3)
        # 锚点星
        h = ax.scatter(
            lons[a], lats[a], s=110, color=c, marker="*",
            edgecolors="k", linewidths=0.6, zorder=4,
            label=f"锚点 {k+1} ({lats[a]:+.1f}°, {lons[a]:.1f}°E)",
        )
        legend_handles.append(h)

    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°)")
    ax.set_xlim(lons.min() - 2, lons.max() + 2)
    ax.set_ylim(lats.min() - 2, lats.max() + 2)
    ax.axhline(0, color="#888", lw=0.6, ls="--", alpha=0.6)
    ax.grid(alpha=0.3, ls=":")
    ax.set_title(
        f"图 4-3 度数最高的 {len(anchors)} 个锚点及其 Top-k 邻居的地理连边",
        fontsize=11,
    )
    ax.legend(
        handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.85,
        ncol=2,
    )
    fig.tight_layout()
    out = FIG / "fig4-3_teleconnections.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4-3] -> {out}, 锚点 idx={anchors.tolist()}")


# ------------------------------------------------------------
def main() -> None:
    fig_4_1_pipeline()
    fig_4_2_adjacency()
    fig_4_3_teleconnections()
    print(f"\n[done] 全部图保存至 {FIG}")


if __name__ == "__main__":
    main()
