"""
生成第 3 章用图（300 dpi，统一中文字体），输出到 thesis/figures/。

图 3-1: 数据预处理流程图（matplotlib 流程框图）
图 3-2: 赤道中太平洋代表点 SST 与 SSTA 时间序列对比
图 3-3: 海洋掩膜 2D 分布

用法：
    python -m thesis.scripts.make_ch3_figures
或：
    & 'd:\\Graduate-Deisgn\\Gradu-Design\\Scripts\\python.exe' \
        d:\\Graduate-Deisgn\\thesis\\scripts\\make_ch3_figures.py
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
# 中文字体配置 —— Microsoft YaHei (Windows 自带) 同时含中英文与 ASCII '-'
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
FIG = _REPO / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 图 3-1 预处理流程图
# ------------------------------------------------------------
def fig_3_1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        ("OISST .nc\n(2015–2024)\n3639 天", 0.5, 2.0, 1.8, "#cfe2f3"),
        ("区域子集\n30°S–30°N\n120°E–280°E", 2.6, 2.0, 1.8, "#d9ead3"),
        ("空间降采样\ncoarsen=8\n(0.25°→2°)", 4.7, 2.0, 1.6, "#d9ead3"),
        ("Climatology\n(逐 DOY 均值,\n2015–2021)", 6.6, 4.0, 1.8, "#fff2cc"),
        ("去季节化\nSSTA = SST − climatology", 6.6, 0.5, 1.8, "#fff2cc"),
        ("海陆掩膜\n$\\mathcal{M}$ + 扁平化\n[T, N=2276]", 9.0, 2.0, 1.8, "#f4cccc"),
        ("时间切分\ntrain/val/test\n2544/365/730", 11.1, 4.0, 1.8, "#ead1dc"),
        ("Z-score 归一化\n($\\mu_{tr},\\sigma_{tr}$)", 11.1, 0.5, 1.8, "#ead1dc"),
    ]
    for text, x, y, w, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, 1.6,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.2, edgecolor="#333", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.8, text, ha="center", va="center", fontsize=9)

    arrows = [
        # (x0, y0, x1, y1)
        (2.3, 2.8, 2.6, 2.8),     # raw -> region
        (4.4, 2.8, 4.7, 2.8),     # region -> coarsen
        (6.3, 3.1, 6.6, 4.5),     # coarsen -> climatology
        (6.3, 2.5, 6.6, 1.1),     # coarsen -> SSTA(SST 输入)
        (7.5, 4.0, 7.5, 2.1),     # climatology -> SSTA
        (8.4, 1.1, 9.0, 2.5),     # SSTA -> mask
        (10.8, 3.1, 11.1, 4.5),   # flatten -> split
        (10.8, 2.5, 11.1, 1.1),   # flatten -> norm
        (12.0, 4.0, 12.0, 2.1),   # split -> norm (split 决定 train_slice)
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )

    ax.text(7.0, 5.6, "图 3-1 数据预处理流程", ha="center", fontsize=11)
    out = FIG / "fig3-1_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3-1] -> {out}")


# ------------------------------------------------------------
# 图 3-2 SST vs SSTA 时间序列
# ------------------------------------------------------------
def fig_3_2_timeseries() -> None:
    sst = np.load(PROC / "sst_raw.npy")        # [T, N]
    ssta = np.load(PROC / "ssta.npy")          # [T, N]
    coords = np.load(PROC / "coords.npy")      # [N, 2]
    dates = np.load(PROC / "dates.npy")        # [T]

    # 选赤道中太平洋（≈0°N, 200°E）的最近节点
    target = np.array([0.0, 200.0])
    dist = np.linalg.norm(coords - target, axis=1)
    idx = int(np.argmin(dist))
    lat0, lon0 = coords[idx]
    print(f"[fig3-2] 选取节点 idx={idx}, lat={lat0:.2f}, lon={lon0:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(dates, sst[:, idx], color="#cc4125", lw=0.7)
    axes[0].set_ylabel("SST (°C)")
    axes[0].set_title(
        f"代表点 (lat={lat0:.1f}°, lon={lon0:.1f}°) 的原始 SST 与去季节化 SSTA",
        fontsize=11,
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(dates, ssta[:, idx], color="#1f4e79", lw=0.7)
    axes[1].axhline(0, color="#888", lw=0.5, ls="--")
    axes[1].set_ylabel("SSTA (°C)")
    axes[1].set_xlabel("日期")
    axes[1].grid(alpha=0.3)

    fig.text(0.5, -0.02, "图 3-2 去季节化前后的 SST/SSTA 时间序列对比", ha="center", fontsize=11)
    fig.tight_layout()
    out = FIG / "fig3-2_ssta_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3-2] -> {out}")


# ------------------------------------------------------------
# 图 3-3 海洋掩膜
# ------------------------------------------------------------
def fig_3_3_mask() -> None:
    mask = np.load(PROC / "mask_2d.npy")            # [H, W]
    with open(PROC / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    lats = np.array(meta["lat_values"])
    lons = np.array(meta["lon_values"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    ax.imshow(
        mask, cmap="Blues", origin="lower",
        extent=extent, aspect="auto", interpolation="nearest",
    )
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°)")
    n_ocean = int(mask.sum())
    n_total = int(mask.size)
    ax.set_title(
        f"图 3-3 海洋掩膜（蓝=海洋节点 N={n_ocean}, 白=陆地; "
        f"占比 {n_ocean/n_total*100:.1f}%）",
        fontsize=11,
    )
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    out = FIG / "fig3-3_ocean_mask.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3-3] -> {out}")


# ------------------------------------------------------------
def main() -> None:
    fig_3_1_pipeline()
    fig_3_2_timeseries()
    fig_3_3_mask()
    print(f"\n[done] 全部图保存至 {FIG}")


if __name__ == "__main__":
    main()
