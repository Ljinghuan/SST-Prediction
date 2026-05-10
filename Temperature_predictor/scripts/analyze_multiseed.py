"""
多种子验证后处理：
1) 读取 multiseed_runs.csv，对 lam=0.5 vs lam=1.0 进行配对 t 检验（按 seed 配对）
2) 输出统计表 multiseed_stats.csv（mean / std / paired-t / p-value，按 lead）
3) 绘制 fig_multiseed_bar.png：4 个 lead 天数下两种 λ 的 RMSE/MAE 均值±标准差对比
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT  = pathlib.Path(__file__).resolve().parents[2]
EXP   = ROOT / "Temperature_predictor" / "experiments" / "multiseed"
FIGS  = ROOT / "Temperature_predictor" / "figures"
RUNS  = pd.read_csv(EXP / "multiseed_runs.csv")

LEADS   = [1, 7, 14, 30]
LAMBDAS = [0.5, 1.0]
METRICS = ["rmse", "mae", "pearson", "ssim"]

# ── 配对 t 检验：对每个 (metric, lead)，按 seed 配对 ───────────────────────
rows = []
for met in METRICS:
    for ld in LEADS:
        a = RUNS[(RUNS["lambda"] == 0.5) & (RUNS["lead_day"] == ld)] \
            .sort_values("seed")[met].to_numpy()
        b = RUNS[(RUNS["lambda"] == 1.0) & (RUNS["lead_day"] == ld)] \
            .sort_values("seed")[met].to_numpy()
        t, p = stats.ttest_rel(a, b)
        rows.append({
            "metric": met, "lead": ld,
            "mean@λ=0.5": a.mean(), "std@λ=0.5": a.std(ddof=1),
            "mean@λ=1.0": b.mean(), "std@λ=1.0": b.std(ddof=1),
            "Δ(0.5−1.0)": a.mean() - b.mean(),
            "t":  t, "p": p,
            "sig": "*" if p < 0.05 else "ns",
        })
stats_df = pd.DataFrame(rows)
out_csv = EXP / "multiseed_stats.csv"
stats_df.to_csv(out_csv, index=False, float_format="%.4f")
print(f"saved → {out_csv}")
print(stats_df.to_string(index=False))

# ── 图：RMSE 与 MAE 在 4 个 lead 上的均值±std 柱状图 ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x      = np.arange(len(LEADS))
width  = 0.35
colors = {"0.5": "#3498DB", "1.0": "#E67E22"}

for ax, met, ylabel in [
    (axes[0], "rmse", "RMSE (°C)"),
    (axes[1], "mae",  "MAE (°C)"),
]:
    sub = stats_df[stats_df["metric"] == met].sort_values("lead")
    m05 = sub["mean@λ=0.5"].to_numpy(); s05 = sub["std@λ=0.5"].to_numpy()
    m10 = sub["mean@λ=1.0"].to_numpy(); s10 = sub["std@λ=1.0"].to_numpy()
    ax.bar(x - width/2, m05, width, yerr=s05, capsize=4,
           color=colors["0.5"], edgecolor="black", linewidth=0.5,
           label=r"$\lambda=0.5$（融合）")
    ax.bar(x + width/2, m10, width, yerr=s10, capsize=4,
           color=colors["1.0"], edgecolor="black", linewidth=0.5,
           label=r"$\lambda=1.0$（仅先验）")
    # 显著性标注
    for i, ld in enumerate(LEADS):
        p = sub[sub["lead"] == ld]["p"].iloc[0]
        mark = "*" if p < 0.05 else "ns"
        y_top = max(m05[i] + s05[i], m10[i] + s10[i]) * 1.04
        ax.text(i, y_top, f"{mark}\n(p={p:.2g})", ha="center", va="bottom",
                fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ld} d" for ld in LEADS])
    ax.set_xlabel("预测提前期 (lead day)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"3 种子均值 ± 标准差：{met.upper()}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(m05.max(), m10.max()) * 1.25)

plt.tight_layout()
out_png = FIGS / "fig_multiseed_bar.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"saved → {out_png}")
