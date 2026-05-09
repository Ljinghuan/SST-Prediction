"""
论文第 6 章用图表生成脚本（汇总数据版，无需重跑模型）。

输入:
    Temperature_predictor/experiments/
        agcrn_main_seed42/{agcrn_test.npz, history.json, config.json}
        agcrn_lambda{0.0, 0.5, 1.0}/{agcrn_test.npz, history.json}
        baselines/{persistence_test.npz, climatology_test.npz}
        ablation_lambda.csv
    Temperature_predictor/data/processed/{meta.json, mask_2d.npy}

输出:
    Temperature_predictor/figures/
        fig_region_map.png            研究区域 + 海陆掩膜
        fig_lead_decay_baseline.png   AGCRN vs Persistence vs Climatology, 4 指标 vs lead
        fig_lead_decay_ablation.png   λ ∈ {0.0, 0.5, 1.0} 4 指标 vs lead
        fig_train_curves.png          main + 3 ablation 的 train_loss / val_rmse_mean
        fig_ablation_bar.png          λ ablation: RMSE@1/7/14/30 柱状图
        fig_baseline_bar.png          baseline vs AGCRN: RMSE@1/7/14/30 柱状图

用法:
    python Temperature_predictor/scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------- 路径 ----------
_HERE = Path(__file__).resolve()
_TP_ROOT = _HERE.parents[1]
EXP = _TP_ROOT / 'experiments'
DATA = _TP_ROOT / 'data' / 'processed'
FIG = _TP_ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# ---------- 全局风格 ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 130,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

LEAD_DAYS = [1, 7, 14, 30]


def _load_npz(p: Path) -> dict:
    return {k: np.asarray(v) for k, v in np.load(p, allow_pickle=False).items()}


# ============================================================
# Figure 1: 研究区域
# ============================================================
def fig_region_map() -> None:
    meta = json.load(open(DATA / 'meta.json', 'r', encoding='utf-8'))
    mask = np.load(DATA / 'mask_2d.npy').astype(bool)
    lats = np.asarray(meta['lat_values'])
    lons = np.asarray(meta['lon_values'])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    # 海洋 = 1, 陆地 = 0
    im = ax.pcolormesh(lons, lats, mask.astype(float), cmap='Blues', shading='auto', vmin=0, vmax=1.2)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Study region: {meta["lon_min"]:.0f}°–{meta["lon_max"]:.0f}°E, '
                 f'{meta["lat_min"]:.0f}°–{meta["lat_max"]:.0f}°N  '
                 f'(grid {meta["H"]}×{meta["W"]}, N={meta["N"]} ocean nodes)')
    ax.grid(alpha=0.3, linestyle='--')
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 1])
    cb.set_ticklabels(['Land', 'Ocean'])
    plt.savefig(FIG / 'fig_region_map.png')
    plt.close()
    print('  wrote fig_region_map.png')


# ============================================================
# Figure 2: lead-day decay (baseline 对比)
# ============================================================
def fig_lead_decay_baseline() -> None:
    main = _load_npz(EXP / 'agcrn_main_seed42' / 'agcrn_test.npz')
    pers = _load_npz(EXP / 'baselines' / 'persistence_test.npz')
    clim = _load_npz(EXP / 'baselines' / 'climatology_test.npz')

    lead = np.arange(1, 31)
    metrics = [('rmse', 'RMSE (°C)', 'lower'),
               ('mae',  'MAE (°C)',  'lower'),
               ('pearson', 'Pearson r', 'higher'),
               ('ssim', 'SSIM', 'higher')]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, ylabel, _) in zip(axes.flat, metrics):
        ax.plot(lead, main[key], label='AGCRN (proposed)', color='C3', lw=2)
        if key in pers:
            ax.plot(lead, pers[key], label='Persistence', color='C0', lw=1.5, ls='--')
        if key in clim:
            ax.plot(lead, clim[key], label='Climatology', color='C7', lw=1.5, ls=':')
        ax.set_xlabel('Lead time (days)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
    fig.suptitle('AGCRN vs. Baselines on test set (per lead)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_lead_decay_baseline.png')
    plt.close()
    print('  wrote fig_lead_decay_baseline.png')


# ============================================================
# Figure 3: lead-day decay (λ ablation)
# ============================================================
def fig_lead_decay_ablation() -> None:
    runs = {}
    for lam in (0.0, 0.5, 1.0):
        runs[lam] = _load_npz(EXP / f'agcrn_lambda{lam}' / 'agcrn_test.npz')

    lead = np.arange(1, 31)
    metrics = [('rmse', 'RMSE (°C)'),
               ('mae',  'MAE (°C)'),
               ('pearson', 'Pearson r'),
               ('ssim', 'SSIM')]
    colors = {0.0: 'C2', 0.5: 'C1', 1.0: 'C3'}
    labels = {0.0: r'$\lambda=0.0$ (pure adaptive)',
              0.5: r'$\lambda=0.5$ (fusion)',
              1.0: r'$\lambda=1.0$ (pure prior)'}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        for lam, m in runs.items():
            ax.plot(lead, m[key], label=labels[lam], color=colors[lam], lw=1.8)
        ax.set_xlabel('Lead time (days)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
    fig.suptitle(r'Effect of graph fusion weight $\lambda$ on test-set metrics')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_lead_decay_ablation.png')
    plt.close()
    print('  wrote fig_lead_decay_ablation.png')


# ============================================================
# Figure 4: 训练曲线
# ============================================================
def fig_train_curves() -> None:
    runs = [
        ('main (λ=0.5, seed 42)', EXP / 'agcrn_main_seed42' / 'history.json', 'C3', '-'),
        ('λ=0.0', EXP / 'agcrn_lambda0.0' / 'history.json', 'C2', '--'),
        ('λ=0.5', EXP / 'agcrn_lambda0.5' / 'history.json', 'C1', '--'),
        ('λ=1.0', EXP / 'agcrn_lambda1.0' / 'history.json', 'C0', '--'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, path, color, ls in runs:
        if not path.exists():
            continue
        h = json.load(open(path, 'r', encoding='utf-8'))
        tl = np.asarray(h.get('train_loss', []))
        vr = np.asarray(h.get('val_rmse_mean', []))
        if tl.size:
            ax1.plot(np.arange(1, len(tl) + 1), tl, label=label, color=color, ls=ls, lw=1.6)
        if vr.size:
            ax2.plot(np.arange(1, len(vr) + 1), vr, label=label, color=color, ls=ls, lw=1.6)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train loss (MAE)')
    ax1.set_title('Training loss')
    ax1.grid(alpha=0.3); ax1.legend(loc='best')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val RMSE (mean over leads)')
    ax2.set_title('Validation RMSE')
    ax2.grid(alpha=0.3); ax2.legend(loc='best')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_train_curves.png')
    plt.close()
    print('  wrote fig_train_curves.png')


# ============================================================
# Figure 5: λ ablation 柱状图（4 个 lead 的 RMSE）
# ============================================================
def fig_ablation_bar() -> None:
    runs = {lam: _load_npz(EXP / f'agcrn_lambda{lam}' / 'agcrn_test.npz')
            for lam in (0.0, 0.5, 1.0)}
    leads = LEAD_DAYS
    x = np.arange(len(leads))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = {0.0: 'C2', 0.5: 'C1', 1.0: 'C3'}
    labels = {0.0: r'$\lambda=0.0$', 0.5: r'$\lambda=0.5$', 1.0: r'$\lambda=1.0$'}
    for i, lam in enumerate((0.0, 0.5, 1.0)):
        vals = [runs[lam]['rmse'][d - 1] for d in leads]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=labels[lam], color=colors[lam])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'{d}d' for d in leads])
    ax.set_xlabel('Lead time'); ax.set_ylabel('Test RMSE (°C)')
    ax.set_title(r'$\lambda$-ablation: test RMSE at four lead horizons')
    ax.grid(alpha=0.3, axis='y'); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'fig_ablation_bar.png')
    plt.close()
    print('  wrote fig_ablation_bar.png')


# ============================================================
# Figure 6: baseline vs AGCRN 柱状图
# ============================================================
def fig_baseline_bar() -> None:
    main = _load_npz(EXP / 'agcrn_main_seed42' / 'agcrn_test.npz')
    pers = _load_npz(EXP / 'baselines' / 'persistence_test.npz')
    clim = _load_npz(EXP / 'baselines' / 'climatology_test.npz')

    leads = LEAD_DAYS
    x = np.arange(len(leads))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [
        ('Persistence', pers, 'C0'),
        ('Climatology', clim, 'C7'),
        ('AGCRN', main, 'C3'),
    ]
    for i, (name, m, c) in enumerate(series):
        vals = [float(m['rmse'][d - 1]) for d in leads]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'{d}d' for d in leads])
    ax.set_xlabel('Lead time'); ax.set_ylabel('Test RMSE (°C)')
    ax.set_title('AGCRN vs. baselines: test RMSE')
    ax.grid(alpha=0.3, axis='y'); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'fig_baseline_bar.png')
    plt.close()
    print('  wrote fig_baseline_bar.png')


def main() -> None:
    print(f'figures -> {FIG}')
    fig_region_map()
    fig_lead_decay_baseline()
    fig_lead_decay_ablation()
    fig_train_curves()
    fig_ablation_bar()
    fig_baseline_bar()
    print('done.')


if __name__ == '__main__':
    main()
