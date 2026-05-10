"""
基于 multiseed 6 个 run 的 history.json 与 *_test.npz / multiseed_runs.csv，
量化 λ=0.5（含自适应分支）vs λ=1.0（仅先验）的 过拟合 / 微弱贡献 证据。

输出：
  - experiments/multiseed/multiseed_traingap.csv
  - figures/fig_multiseed_traincurve.png   两个 λ 的 train_loss / val_rmse 曲线（3 种子均值±std）
"""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP  = ROOT / "Temperature_predictor" / "experiments" / "multiseed"
FIGS = ROOT / "Temperature_predictor" / "figures"

LAMBDAS = [0.5, 1.0]
SEEDS   = [42, 123, 2024]

def load_hist(lam, seed):
    p = EXP / f"lam{lam}_seed{seed}" / "history.json"
    return json.loads(p.read_text(encoding="utf-8"))

# ── 收集训练/验证曲线（早停导致 epoch 数不同，按最短长度截断对齐） ──────
curves = {}
for lam in LAMBDAS:
    tl_list = [np.array(load_hist(lam, s)["train_loss"]) for s in SEEDS]
    vr_list = [np.array(load_hist(lam, s)["val_rmse_mean"]) for s in SEEDS]
    L = min(min(len(a) for a in tl_list), min(len(a) for a in vr_list))
    tl = np.stack([a[:L] for a in tl_list])
    vr = np.stack([a[:L] for a in vr_list])
    curves[lam] = {
        "epochs":          L,
        "train_loss_mean": tl.mean(0), "train_loss_std": tl.std(0, ddof=1),
        "val_rmse_mean":   vr.mean(0), "val_rmse_std":   vr.std(0, ddof=1),
        "train_loss_last": np.array([a[-1] for a in tl_list]),
        "val_rmse_min":    np.array([a.min() for a in vr_list]),
        "val_rmse_last":   np.array([a[-1] for a in vr_list]),
        "n_epoch_per_seed":np.array([len(a) for a in tl_list]),
    }
    print(f"λ={lam}: per-seed epochs = {curves[lam]['n_epoch_per_seed']}, aligned to L={L}")

# ── train-test gap 表 ────────────────────────────────────────────────────
rows = []
for lam in LAMBDAS:
    c = curves[lam]
    for i, s in enumerate(SEEDS):
        rows.append({
            "lambda": lam, "seed": s,
            "train_loss_last": c["train_loss_last"][i],
            "val_rmse_min":    c["val_rmse_min"][i],
            "val_rmse_last":   c["val_rmse_last"][i],
        })
gap_df = pd.DataFrame(rows)

# 聚合
agg = gap_df.groupby("lambda").agg(["mean", "std"]).round(4)
print(agg.to_string())

out_csv = EXP / "multiseed_traingap.csv"
gap_df.to_csv(out_csv, index=False, float_format="%.4f")
print(f"\nsaved → {out_csv}")

# ── 图：训练损失 + 验证 RMSE ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = {0.5: "#3498DB", 1.0: "#E67E22"}
labels = {0.5: r"$\lambda=0.5$（融合，含自适应）", 1.0: r"$\lambda=1.0$（仅先验）"}

for ax, key_mean, key_std, ylabel, title in [
    (axes[0], "train_loss_mean", "train_loss_std", "训练损失 (L1)", "训练损失曲线（3 种子均值±std）"),
    (axes[1], "val_rmse_mean",   "val_rmse_std",   "验证 RMSE (°C)", "验证 RMSE 曲线（3 种子均值±std）"),
]:
    for lam in LAMBDAS:
        c = curves[lam]
        x = np.arange(1, len(c[key_mean]) + 1)
        ax.plot(x, c[key_mean], color=colors[lam], lw=1.6, label=labels[lam])
        ax.fill_between(x, c[key_mean] - c[key_std], c[key_mean] + c[key_std],
                        color=colors[lam], alpha=0.18)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

plt.tight_layout()
out_png = FIGS / "fig_multiseed_traincurve.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"saved → {out_png}")

# ── 关键统计打印 ─────────────────────────────────────────────────────────
print("\n=== 过拟合 / 贡献证据 ===")
for lam in LAMBDAS:
    c = curves[lam]
    print(f"λ={lam}: train_loss_last = {c['train_loss_last'].mean():.4f} ± {c['train_loss_last'].std(ddof=1):.4f}, "
          f"val_rmse_min = {c['val_rmse_min'].mean():.4f} ± {c['val_rmse_min'].std(ddof=1):.4f}, "
          f"val_rmse_last = {c['val_rmse_last'].mean():.4f}")

dt = curves[0.5]['train_loss_last'].mean() - curves[1.0]['train_loss_last'].mean()
dv = curves[0.5]['val_rmse_min'].mean() - curves[1.0]['val_rmse_min'].mean()
print(f"\nΔtrain_loss(0.5−1.0) = {dt:+.4f}  →  λ=0.5 训练损失{'更低' if dt<0 else '更高'}")
print(f"Δval_rmse_min(0.5−1.0) = {dv:+.4f}  →  λ=0.5 最优验证{'更低' if dv<0 else '更高'}")
