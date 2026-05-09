"""
训练循环：Adam + L1 + early stopping + ckpt。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .evaluate import evaluate_loader, format_lead_table


__all__ = ['train_model']


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    save_dir: str | Path,
    log_print: Callable[[str], None] = print,
    mask_2d: Optional[np.ndarray] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
) -> Dict:
    """训练并返回训练过程记录。

    会在 ``save_dir/best.pt`` 处保存验证集 RMSE 最小的权重。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.L1Loss()

    # AMP：仅在 CUDA 上启用；CPU 时即使 cfg.use_amp=True 也退化为常规精度
    use_amp = bool(getattr(cfg, 'use_amp', False)) and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse_mean': []}
    best_val = float('inf')
    patience_cnt = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # ---------- train ----------
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
            if use_amp:
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        # ---------- validate ----------
        val_metrics = evaluate_loader(
            model, val_loader, device=device,
            mask_2d=mask_2d, H=H, W=W,
            compute_ssim=False,
        )
        val_rmse_mean = float(val_metrics['rmse'].mean())
        # 使用 RMSE 均值作为 early stop 准则（与论文主表一致）

        history['train_loss'].append(train_loss)
        history['val_rmse_mean'].append(val_rmse_mean)

        improved = val_rmse_mean < best_val - 1e-6
        if improved:
            best_val = val_rmse_mean
            patience_cnt = 0
            torch.save(model.state_dict(), save_dir / 'best.pt')
        else:
            patience_cnt += 1

        msg = (f'epoch {epoch:3d}  train_L1={train_loss:.4f}  '
               f'val_RMSE_mean={val_rmse_mean:.4f}  '
               f'{"*" if improved else " "}  '
               f'patience={patience_cnt}/{cfg.patience}')
        log_print(msg)

        if patience_cnt >= cfg.patience:
            log_print(f'early stop at epoch {epoch}')
            break

    elapsed = time.time() - t_start
    log_print(f'training finished in {elapsed:.1f}s, best val RMSE = {best_val:.4f}')

    # 保存 history 与 config
    with open(save_dir / 'history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    # 加载最佳权重
    state = torch.load(save_dir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    return {'history': history, 'best_val_rmse': best_val, 'elapsed': elapsed}
