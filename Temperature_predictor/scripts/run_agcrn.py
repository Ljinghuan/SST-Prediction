"""
训练 + 评估 AGCRN。使用：
    python Temperature_predictor/scripts/run_agcrn.py
"""

from __future__ import annotations

import json
import sys
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


def main() -> None:
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        T_in=cfg.T_in, T_out=cfg.T_out,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # meta / mask
    with open(Path(cfg.data_dir) / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    H, W, N = int(meta['H']), int(meta['W']), int(meta['N'])
    mask_2d = np.load(Path(cfg.data_dir) / 'mask_2d.npy').astype(bool)

    # 加载先验图
    A_prior = load_prior_dense(Path(cfg.graph_dir) / 'A_prior.npz')

    model = AGCRN(
        num_nodes=N, T_in=cfg.T_in, T_out=cfg.T_out,
        c_in=1, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        cheb_k=cfg.cheb_k, embed_dim=cfg.embed_dim, adapt_dim=cfg.adapt_dim,
        A_prior=A_prior, lambda_fuse=cfg.lambda_fuse,
        use_checkpoint=cfg.use_checkpoint,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model params: {n_params:,}')

    save_dir = Path(cfg.exp_dir) / 'agcrn'
    train_model(
        model, train_loader, val_loader, cfg,
        save_dir=save_dir,
        mask_2d=mask_2d, H=H, W=W,
    )

    # 测试集评估（含 SSIM）
    metrics = evaluate_loader(
        model, test_loader, device=device,
        mask_2d=mask_2d, H=H, W=W, compute_ssim=True,
    )
    print('\n[AGCRN] test set')
    print(format_lead_table(metrics, cfg.lead_days))
    np.savez(save_dir / 'agcrn_test.npz', **metrics)
    print(f'\nresults under: {save_dir}')


if __name__ == '__main__':
    main()
