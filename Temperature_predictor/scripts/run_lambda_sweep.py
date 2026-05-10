"""
重跑 λ ∈ {0.0, 0.5, 1.0} 三个配置（带 val_loss 记录），用于绘制训练/验证 loss 曲线。

云端用法：
    python Temperature_predictor/scripts/run_lambda_sweep.py

每个 λ 训练 ~25 epoch（早停 patience=15），耗时约 40 min × 3 ≈ 2 小时。
输出：
    experiments/agcrn_lambda{0.0,0.5,1.0}/history.json   （含 val_loss）
    experiments/agcrn_lambda{0.0,0.5,1.0}/best.pt
    experiments/agcrn_lambda{0.0,0.5,1.0}/agcrn_test.npz

完成后本地拉回 history.json，运行 plot_lambda_traincurve.py 生成图。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

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


LAMBDAS = [0.0, 0.5, 1.0]


def run_one(lam: float) -> None:
    cfg = Config()
    cfg.lambda_fuse = lam
    cfg.exp_dir = f'experiments/agcrn_lambda{lam}'
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
    print(f'\n========== λ={lam} ==========')
    print(f'model params: {n_params:,}')

    save_dir = Path(cfg.exp_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_model(
        model, train_loader, val_loader, cfg,
        save_dir=save_dir,
        mask_2d=mask_2d, H=H, W=W,
    )

    metrics = evaluate_loader(
        model, test_loader, device=device,
        mask_2d=mask_2d, H=H, W=W, compute_ssim=True,
    )
    print(f'\n[AGCRN λ={lam}] test set')
    print(format_lead_table(metrics, cfg.lead_days))
    np.savez(save_dir / 'agcrn_test.npz', **metrics)


def main() -> None:
    for lam in LAMBDAS:
        run_one(lam)
    print('\n=== all 3 λ runs finished ===')


if __name__ == '__main__':
    main()
