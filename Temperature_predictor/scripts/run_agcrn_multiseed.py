"""
多 seed × 多 lambda_fuse 验证脚本。

用法:
    python Temperature_predictor/scripts/run_agcrn_multiseed.py \
        --lambdas 0.5 1.0 --seeds 42 123 2024 \
        --out-root experiments/multiseed

为每个 (lambda, seed) 组合训练一次 AGCRN, 保存到独立子目录,
最后聚合 RMSE / MAE / Pearson / SSIM 在各 lead 上的均值与标准差,
输出 CSV 便于直接粘进论文。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_TP_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_TP_ROOT.parent))

from Temperature_predictor.src.data.dataset import get_dataloaders
from Temperature_predictor.src.models.agcrn import AGCRN, load_prior_dense
from Temperature_predictor.src.train.config import Config
from Temperature_predictor.src.train.evaluate import evaluate_loader, format_lead_table
from Temperature_predictor.src.train.train import train_model


def run_one(lam: float, seed: int, out_dir: Path, base_cfg: Config) -> dict:
    """跑单个 (lambda, seed) 组合, 返回 metrics dict。"""
    cfg = Config(**base_cfg.to_dict())
    cfg.lambda_fuse = lam
    cfg.seed = seed
    cfg.exp_dir = str(out_dir.parent)  # 使保存路径与 out_dir 对齐
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        T_in=cfg.T_in, T_out=cfg.T_out,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    with open(Path(cfg.data_dir) / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    H, W, N = int(meta['H']), int(meta['W']), int(meta['N'])
    mask_2d = np.load(Path(cfg.data_dir) / 'mask_2d.npy').astype(bool)

    A_prior = load_prior_dense(Path(cfg.graph_dir) / 'A_prior.npz')

    model = AGCRN(
        num_nodes=N, T_in=cfg.T_in, T_out=cfg.T_out,
        c_in=1, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        cheb_k=cfg.cheb_k, embed_dim=cfg.embed_dim, adapt_dim=cfg.adapt_dim,
        A_prior=A_prior, lambda_fuse=cfg.lambda_fuse,
        use_checkpoint=cfg.use_checkpoint,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[lam={lam} seed={seed}] params={n_params:,}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        model, train_loader, val_loader, cfg,
        save_dir=out_dir,
        mask_2d=mask_2d, H=H, W=W,
    )

    metrics = evaluate_loader(
        model, test_loader, device=device,
        mask_2d=mask_2d, H=H, W=W, compute_ssim=True,
    )
    print(f'\n[lam={lam} seed={seed}] test set')
    print(format_lead_table(metrics, cfg.lead_days))

    np.savez(out_dir / 'agcrn_test.npz', **metrics)
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return metrics


def aggregate(results: list, lead_days: List[int], out_csv: Path) -> None:
    """results: list of dict { 'lambda':..., 'seed':..., 'metrics': {...} }
    输出 CSV: 每个 (lambda) 一行均值, 一行 std, 列为 metric@lead。
    """
    import csv
    metric_keys = ['rmse', 'mae', 'pearson', 'ssim']
    lead_idx = [d - 1 for d in lead_days]

    # 按 lambda 分组
    by_lam: dict = {}
    for r in results:
        by_lam.setdefault(r['lambda'], []).append(r)

    rows = []
    header = ['lambda', 'agg', 'n_seeds']
    for mk in metric_keys:
        for d in lead_days:
            header.append(f'{mk}@{d}d')

    for lam, runs in sorted(by_lam.items()):
        # stack: shape (n_seeds, len(lead_days)) per metric
        stacks = {mk: [] for mk in metric_keys}
        for r in runs:
            for mk in metric_keys:
                arr = np.asarray(r['metrics'][mk])  # full T_out length
                stacks[mk].append(arr[lead_idx])
        for agg in ['mean', 'std']:
            row = [lam, agg, len(runs)]
            for mk in metric_keys:
                vals = np.stack(stacks[mk], axis=0)  # (n_seeds, len(lead_days))
                aggv = vals.mean(0) if agg == 'mean' else vals.std(0, ddof=0)
                row.extend([f'{v:.4f}' for v in aggv])
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'\n[aggregate] wrote {out_csv}')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--lambdas', type=float, nargs='+', default=[0.5, 1.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    p.add_argument('--out-root', type=str, default=None,
                   help='结果根目录, 默认 Temperature_predictor/experiments/multiseed')
    args = p.parse_args()

    base_cfg = Config()
    out_root = Path(args.out_root) if args.out_root else Path(base_cfg.exp_dir) / 'multiseed'
    out_root.mkdir(parents=True, exist_ok=True)

    print(f'lambdas={args.lambdas}  seeds={args.seeds}  out_root={out_root}', flush=True)
    results = []
    t0 = time.time()
    for lam in args.lambdas:
        for sd in args.seeds:
            tag = f'lam{lam}_seed{sd}'
            sub = out_root / tag
            if (sub / 'agcrn_test.npz').exists():
                print(f'[skip] {tag} already done', flush=True)
                m = dict(np.load(sub / 'agcrn_test.npz', allow_pickle=False))
            else:
                m = run_one(lam, sd, sub, base_cfg)
            results.append({'lambda': lam, 'seed': sd, 'metrics': m})
            print(f'  elapsed: {(time.time()-t0)/60:.1f} min', flush=True)

    aggregate(results, base_cfg.lead_days, out_root / 'multiseed_summary.csv')

    # 同时把每个 run 的 lead 指标存成 long format 便于 t-test
    import csv
    long_csv = out_root / 'multiseed_runs.csv'
    with open(long_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lambda', 'seed', 'lead_day', 'rmse', 'mae', 'pearson', 'ssim'])
        for r in results:
            for d in base_cfg.lead_days:
                idx = d - 1
                w.writerow([
                    r['lambda'], r['seed'], d,
                    f"{float(r['metrics']['rmse'][idx]):.4f}",
                    f"{float(r['metrics']['mae'][idx]):.4f}",
                    f"{float(r['metrics']['pearson'][idx]):.4f}",
                    f"{float(r['metrics']['ssim'][idx]):.4f}",
                ])
    print(f'[aggregate] wrote {long_csv}')
    print(f'\nALL DONE  total elapsed: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
