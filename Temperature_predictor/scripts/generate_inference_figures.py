"""
基于 best.pt 重新推理, 生成空间误差图与个例图。

输出 (Temperature_predictor/figures/):
    fig_spatial_rmse.png        4 个 lead 的逐像素 RMSE 空间分布
    fig_case_study.png          一个测试样本的真值/预测/误差 (lead 1/7/14/30)
    cache/test_preds_main.npz   缓存推理结果, 后续重画无需再跑

用法:
    python Temperature_predictor/scripts/generate_inference_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = Path(__file__).resolve()
_TP_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_TP_ROOT.parent))

from Temperature_predictor.src.data.dataset import get_dataloaders
from Temperature_predictor.src.models.agcrn import AGCRN, load_prior_dense
from Temperature_predictor.src.train.config import Config

EXP = _TP_ROOT / 'experiments'
DATA = _TP_ROOT / 'data' / 'processed'
FIG = _TP_ROOT / 'figures'
CACHE = FIG / 'cache'
FIG.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 130,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

LEAD_DAYS = [1, 7, 14, 30]
MAIN_CKPT = EXP / 'agcrn_main_seed42' / 'best.pt'
CACHE_NPZ = CACHE / 'test_preds_main.npz'


def run_inference() -> dict:
    """加载 best.pt, 在 test loader 上做一次推理, 返回 (y, y_hat) 反归一化后结果。"""
    cfg = Config()
    cfg.batch_size = 4  # 本地 4GB GPU 友好
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'inference device={device}, batch_size={cfg.batch_size}')

    _, _, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        T_in=cfg.T_in, T_out=cfg.T_out,
        batch_size=cfg.batch_size, num_workers=0,
    )

    meta = json.load(open(DATA / 'meta.json', 'r', encoding='utf-8'))
    H, W, N = int(meta['H']), int(meta['W']), int(meta['N'])
    norm = json.load(open(DATA / 'norm_stats.json', 'r', encoding='utf-8'))
    std = float(norm['std'])  # mean ~ 0

    A_prior = load_prior_dense(_TP_ROOT / 'data' / 'processed' / 'graph' / 'A_prior.npz')

    model = AGCRN(
        num_nodes=N, T_in=cfg.T_in, T_out=cfg.T_out,
        c_in=1, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        cheb_k=cfg.cheb_k, embed_dim=cfg.embed_dim, adapt_dim=cfg.adapt_dim,
        A_prior=A_prior, lambda_fuse=cfg.lambda_fuse,
        use_checkpoint=False,
    ).to(device)
    sd = torch.load(MAIN_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    ys, yhs = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device); y = y.to(device)
            yh = model(x)
            ys.append(y.cpu().numpy())
            yhs.append(yh.cpu().numpy())
            if (i + 1) % 5 == 0:
                print(f'  batch {i+1}/{len(test_loader)}')
    y = np.concatenate(ys, axis=0)        # (S, T_out, N)
    yh = np.concatenate(yhs, axis=0)
    # 反归一化到 SSTA (摄氏度异常)
    y = y * std
    yh = yh * std
    print(f'shapes: y={y.shape}  yh={yh.shape}')
    return {'y': y, 'yhat': yh, 'std': std}


def get_or_run() -> dict:
    if CACHE_NPZ.exists():
        print(f'load cached {CACHE_NPZ}')
        d = np.load(CACHE_NPZ, allow_pickle=False)
        return {'y': d['y'], 'yhat': d['yhat'], 'std': float(d['std'])}
    out = run_inference()
    np.savez(CACHE_NPZ, **out)
    print(f'cached -> {CACHE_NPZ}')
    return out


def _to_grid(arr_n: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把 (..., N) 还原成 (..., H, W), 陆地填 NaN。"""
    H, W = mask.shape
    out_shape = arr_n.shape[:-1] + (H, W)
    grid = np.full(out_shape, np.nan, dtype=np.float32)
    grid[..., mask] = arr_n
    return grid


# ============================================================
# Figure: 空间 RMSE 图
# ============================================================
def fig_spatial_rmse(data: dict) -> None:
    meta = json.load(open(DATA / 'meta.json', 'r', encoding='utf-8'))
    mask = np.load(DATA / 'mask_2d.npy').astype(bool)
    lats = np.asarray(meta['lat_values']); lons = np.asarray(meta['lon_values'])

    y = data['y']; yh = data['yhat']  # (S, T, N)
    err2 = (yh - y) ** 2  # (S, T, N)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    vmax = 0
    grids = []
    for d in LEAD_DAYS:
        rmse_n = np.sqrt(err2[:, d - 1, :].mean(axis=0))   # (N,)
        g = _to_grid(rmse_n, mask)
        grids.append((d, g))
        vmax = max(vmax, np.nanpercentile(g, 99))
    vmax = float(np.ceil(vmax * 10) / 10)

    for ax, (d, g) in zip(axes.flat, grids):
        im = ax.pcolormesh(lons, lats, g, cmap='magma_r', shading='auto',
                           vmin=0, vmax=vmax)
        ax.set_title(f'Lead {d} day  (mean RMSE={np.nanmean(g):.3f} °C)')
        ax.set_xlabel('Lon (°E)'); ax.set_ylabel('Lat (°N)')
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='RMSE (°C)')
    fig.suptitle('Per-pixel RMSE on test set (AGCRN main run, seed 42)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_spatial_rmse.png')
    plt.close()
    print('  wrote fig_spatial_rmse.png')


# ============================================================
# Figure: 个例
# ============================================================
def fig_case_study(data: dict, sample_idx: int = 200) -> None:
    meta = json.load(open(DATA / 'meta.json', 'r', encoding='utf-8'))
    mask = np.load(DATA / 'mask_2d.npy').astype(bool)
    lats = np.asarray(meta['lat_values']); lons = np.asarray(meta['lon_values'])

    y = data['y'][sample_idx]   # (T, N)
    yh = data['yhat'][sample_idx]

    rows = LEAD_DAYS
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.6 * len(rows)))
    # 颜色范围: 用真值与预测的最大|值| 决定
    vabs = float(np.nanpercentile(np.abs(y), 99))
    for r, d in enumerate(rows):
        gt = _to_grid(y[d - 1], mask)
        pr = _to_grid(yh[d - 1], mask)
        er = pr - gt
        for c, (g, title, cmap, vmin, vmax) in enumerate([
            (gt, f'Truth (lead {d}d)', 'RdBu_r', -vabs, vabs),
            (pr, f'AGCRN pred (lead {d}d)', 'RdBu_r', -vabs, vabs),
            (er, f'Error (pred - truth)', 'PuOr_r', -vabs / 2, vabs / 2),
        ]):
            ax = axes[r, c]
            im = ax.pcolormesh(lons, lats, g, cmap=cmap, shading='auto',
                               vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel('Lat (°N)')
            if r == len(rows) - 1:
                ax.set_xlabel('Lon (°E)')
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(f'Case study: test sample #{sample_idx} (SST anomaly, °C)', y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / 'fig_case_study.png')
    plt.close()
    print(f'  wrote fig_case_study.png  (sample_idx={sample_idx})')


def main() -> None:
    if not MAIN_CKPT.exists():
        print(f'ERROR: best.pt not found at {MAIN_CKPT}')
        sys.exit(1)
    data = get_or_run()
    fig_spatial_rmse(data)
    fig_case_study(data, sample_idx=200)
    print('done.')


if __name__ == '__main__':
    main()
