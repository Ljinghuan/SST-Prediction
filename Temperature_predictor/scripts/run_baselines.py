"""
评估零参数基线。

用法：
    python -m Temperature_predictor.scripts.run_baselines
或：
    python Temperature_predictor/scripts/run_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# 使本脚本可以独立运行（不需 ``-m``）
_HERE = Path(__file__).resolve()
_TP_ROOT = _HERE.parents[1]                # Temperature_predictor/
sys.path.insert(0, str(_TP_ROOT.parent))   # 让 ``Temperature_predictor.xxx`` 可导入

from Temperature_predictor.src.data.dataset import get_dataloaders
from Temperature_predictor.src.models.baselines import Climatology, Persistence
from Temperature_predictor.src.train.config import Config
from Temperature_predictor.src.train.evaluate import evaluate_loader, format_lead_table


def main() -> None:
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        T_in=cfg.T_in, T_out=cfg.T_out,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    # mask_2d / H / W 为 SSIM 重拼提供
    mask_2d = np.load(Path(cfg.data_dir) / 'mask_2d.npy').astype(bool)
    with open(Path(cfg.data_dir) / 'meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    H, W = int(meta['H']), int(meta['W'])

    out_dir = Path(cfg.exp_dir) / 'baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, Model in [('persistence', Persistence), ('climatology', Climatology)]:
        model = Model(T_out=cfg.T_out).to(device)
        metrics = evaluate_loader(
            model, test_loader, device=device,
            mask_2d=mask_2d, H=H, W=W, compute_ssim=True,
        )
        print(f'\n[{name}] test set')
        print(format_lead_table(metrics, cfg.lead_days))
        np.savez(out_dir / f'{name}_test.npz', **metrics)
    print(f'\nMetrics saved under: {out_dir}')


if __name__ == '__main__':
    main()
