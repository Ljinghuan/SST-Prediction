"""
绘制 λ ∈ {0.0, 0.5, 1.0} 三种配置下训练损失与验证损失（L1）曲线对比，
用于判断 λ=0.5 是否过拟合。

数据源（重跑后含 val_loss）：
  experiments/agcrn_lambda0.0/history.json
  experiments/agcrn_lambda0.5/history.json
  experiments/agcrn_lambda1.0/history.json

输出：figures/fig_lambda_traincurve.png
判据：
  过拟合 ⇔ train_loss 持续下降 而 val_loss 同时上升（gap 拉大）
  正则化 ⇔ train_loss 略高 但 val_loss 同样降低或更低
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP  = ROOT / "Temperature_predictor" / "experiments"
FIGS = ROOT / "Temperature_predictor" / "figures"

LAMS   = [0.0, 0.5, 1.0]
COLORS = {0.0: "#E74C3C", 0.5: "#3498DB", 1.0: "#E67E22"}
LABELS = {0.0: r"$\lambda=0.0$（仅自适应图）",
          0.5: r"$\lambda=0.5$（等权融合）",
          1.0: r"$\lambda=1.0$（仅先验图）"}

def load(lam):
    p = EXP / f"agcrn_lambda{lam}" / "history.json"
    h = json.loads(p.read_text(encoding="utf-8"))
    tl = np.array(h["train_loss"])
    vl = np.array(h.get("val_loss", []))
    if vl.size == 0:
        # 回退：旧 history 没记录 val_loss → 用 val_rmse_mean 替代并提示
        vl = np.array(h["val_rmse_mean"])
        print(f"[WARN] λ={lam}: history.val_loss 为空，回退到 val_rmse_mean")
    return tl, vl

# ── 收集 ─────────────────────────────────────────────────────────────────
hist = {lam: load(lam) for lam in LAMS}

# ── 绘图 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# 左：训练损失
ax = axes[0]
for lam in LAMS:
    tl, _ = hist[lam]
    x = np.arange(1, len(tl) + 1)
    ax.plot(x, tl, color=COLORS[lam], lw=1.7, label=LABELS[lam])
    ax.scatter([x[-1]], [tl[-1]], color=COLORS[lam], s=28, zorder=5)
    ax.annotate(f"{tl[-1]:.3f}", (x[-1], tl[-1]),
                textcoords="offset points", xytext=(6, 0),
                fontsize=8.5, color=COLORS[lam], va="center")
ax.set_xlabel("Epoch")
ax.set_ylabel("训练损失 (L1)")
ax.set_title("训练损失曲线")
ax.grid(linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper right")

# 右：验证损失
ax = axes[1]
for lam in LAMS:
    _, vl = hist[lam]
    x = np.arange(1, len(vl) + 1)
    ax.plot(x, vl, color=COLORS[lam], lw=1.7, label=LABELS[lam])
    i_min = int(np.argmin(vl))
    ax.scatter([x[i_min]], [vl[i_min]], color=COLORS[lam], s=40,
               marker="v", zorder=5, edgecolor="black", linewidth=0.6)
    ax.annotate(f"min={vl[i_min]:.4f}\n@ep{x[i_min]}",
                (x[i_min], vl[i_min]),
                textcoords="offset points", xytext=(6, -14),
                fontsize=8, color=COLORS[lam])
ax.set_xlabel("Epoch")
ax.set_ylabel("验证损失 (L1)")
ax.set_title("验证损失曲线（▼ 标记最低点）")
ax.grid(linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
out = FIGS / "fig_lambda_traincurve.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved → {out}")

# ── 量化诊断打印 ──────────────────────────────────────────────────────────
print("\n=== 过拟合诊断 ===")
print(f"{'λ':>5} | {'train_min':>9} {'train_last':>10} | "
      f"{'val_min':>8} {'val_last':>9} | {'val 是否回升':>14} | {'最大 gap':>9}")
for lam in LAMS:
    tl, vl = hist[lam]
    train_min, train_last = tl.min(), tl[-1]
    val_min,  val_last    = vl.min(), vl[-1]
    rebound = val_last - val_min
    flag = f"+{rebound:.4f}" if rebound > 0 else f"{rebound:.4f}"
    L = min(len(tl), len(vl))
    gap_max = float((vl[:L] - tl[:L]).max())
    print(f"{lam:>5.1f} | {train_min:>9.4f} {train_last:>10.4f} | "
          f"{val_min:>8.4f} {val_last:>9.4f} | {flag:>14} | {gap_max:>9.4f}")

print("\n判据：若某 λ 出现 train_loss 持续下降 + val_loss 在最低点之后持续上升，")
print("      则该配置为过拟合；若 val_loss 同步下降或仅微弱回升，则为正常收敛。")

