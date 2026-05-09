"""
评估指标与 lead-time 报告。

所有指标均在 *归一化后* 的 SSTA 空间上计算（与训练损失一致），
如需转回原始 \xb0C 尺度，请调用 ``denormalize`` 后再调本模块。

给定预测与真值张量 ``y_pred, y_true`` (形状 ``[B, T_out, N]``)，
本模块提供：

* ``rmse_per_lead``       逐 lead-time 的 RMSE，返回 ``[T_out]``
* ``mae_per_lead``        MAE，返回 ``[T_out]``
* ``pearson_per_lead``    节点平均 Pearson 相关（逐 lead），返回 ``[T_out]``
* ``ssim_per_lead``       将 ``[B, N]`` 节点向量重拼为 2D 场 ``[B, H, W]``后计算 SSIM
* ``evaluate_loader``     高层入口，方便训练/测试脚手架
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


__all__ = [
    'rmse_per_lead', 'mae_per_lead', 'pearson_per_lead', 'ssim_per_lead',
    'evaluate_loader', 'format_lead_table',
]


# ============================================================
# 逐 lead-time 指标（输入为节点向量形式 [B, T_out, N]）
# ============================================================

def rmse_per_lead(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """逐 lead-time RMSE。返回形状 ``[T_out]``。"""
    sq = (y_pred - y_true).pow(2)
    return sq.mean(dim=(0, 2)).sqrt()


def mae_per_lead(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """逐 lead-time MAE。返回 ``[T_out]``。"""
    return (y_pred - y_true).abs().mean(dim=(0, 2))


def pearson_per_lead(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """逐 lead-time 、逐节点计算 Pearson r 后在节点维取均值。

    以 batch 维作为\"时间采样\"：对于 lead k 与节点 i，求
    ``r(y_pred[:, k, i], y_true[:, k, i])``。返回形状 ``[T_out]``。
    """
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    num = (yp * yt).sum(dim=0)                                  # [T_out, N]
    den = yp.pow(2).sum(dim=0).sqrt() * yt.pow(2).sum(dim=0).sqrt()
    r = num / den.clamp_min(1e-12)
    return r.mean(dim=-1)                                       # [T_out]


def _gaussian_kernel(window: int = 11, sigma: float = 1.5,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    k2d = g.unsqueeze(0) * g.unsqueeze(1)                       # [W, W]
    return k2d.unsqueeze(0).unsqueeze(0)                        # [1, 1, W, W]


def _ssim_2d(a: torch.Tensor, b: torch.Tensor,
             window: int = 11, sigma: float = 1.5,
             data_range: float = 1.0) -> torch.Tensor:
    """对 ``[B, 1, H, W]`` 形式计算 SSIM，返回 batch 均值标量。"""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    kernel = _gaussian_kernel(window, sigma, device=a.device, dtype=a.dtype)
    pad = window // 2
    mu_a = torch.nn.functional.conv2d(a, kernel, padding=pad)
    mu_b = torch.nn.functional.conv2d(b, kernel, padding=pad)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sa2 = torch.nn.functional.conv2d(a * a, kernel, padding=pad) - mu_a2
    sb2 = torch.nn.functional.conv2d(b * b, kernel, padding=pad) - mu_b2
    sab = torch.nn.functional.conv2d(a * b, kernel, padding=pad) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa2 + sb2 + C2)
    return (num / den).mean()


def ssim_per_lead(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask_2d: np.ndarray,
    H: int,
    W: int,
    data_range: float = 6.0,
) -> torch.Tensor:
    """将节点向量 ``[B, T_out, N]`` 重拼为 ``[B, T_out, H, W]`` 后逐 lead 计算 SSIM。

    参数
    ----
    mask_2d : np.ndarray, shape ``[H, W]``, bool
        ``True`` 为有效海洋节点。需 ``mask_2d.sum() == N``。
    data_range : float
        归一化后 SSTA 的动态范围估计（默认 6σ）。

    返回
    ----
    torch.Tensor, shape ``[T_out]``。
    """
    B, T_out, N = y_pred.shape
    if int(mask_2d.sum()) != N:
        raise ValueError(f"mask_2d.sum()={int(mask_2d.sum())} != N={N}")
    mask_flat = torch.as_tensor(mask_2d.reshape(-1), device=y_pred.device)
    out = torch.zeros(T_out, device=y_pred.device, dtype=y_pred.dtype)
    for k in range(T_out):
        a = torch.zeros(B, H * W, device=y_pred.device, dtype=y_pred.dtype)
        b = torch.zeros(B, H * W, device=y_pred.device, dtype=y_pred.dtype)
        a[:, mask_flat] = y_pred[:, k, :]
        b[:, mask_flat] = y_true[:, k, :]
        a = a.view(B, 1, H, W)
        b = b.view(B, 1, H, W)
        out[k] = _ssim_2d(a, b, data_range=data_range)
    return out


# ============================================================
# 高层入口
# ============================================================

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda',
    mask_2d: Optional[np.ndarray] = None,
    H: Optional[int] = None,
    W: Optional[int] = None,
    compute_ssim: bool = True,
) -> Dict[str, np.ndarray]:
    """遍历 ``loader`` 计算 RMSE/MAE/Pearson/SSIM。

    返回
    ----
    dict，key 为 ``rmse``/``mae``/``pearson``/``ssim``，value 为形状 ``[T_out]`` 的数组。
    如果 ``compute_ssim=False`` 或 ``mask_2d/H/W`` 未提供，则 ``ssim`` 不包含在返回字典中。
    """
    model.eval()
    rmse_acc, mae_acc, pearson_acc, ssim_acc = [], [], [], []
    do_ssim = compute_ssim and (mask_2d is not None) and (H is not None) and (W is not None)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat = model(x)
        rmse_acc.append(rmse_per_lead(y_hat, y).cpu())
        mae_acc.append(mae_per_lead(y_hat, y).cpu())
        pearson_acc.append(pearson_per_lead(y_hat, y).cpu())
        if do_ssim:
            ssim_acc.append(ssim_per_lead(y_hat, y, mask_2d, H, W).cpu())
    out: Dict[str, np.ndarray] = {
        'rmse': torch.stack(rmse_acc).mean(0).numpy(),
        'mae':  torch.stack(mae_acc).mean(0).numpy(),
        'pearson': torch.stack(pearson_acc).mean(0).numpy(),
    }
    if do_ssim and ssim_acc:
        out['ssim'] = torch.stack(ssim_acc).mean(0).numpy()
    return out


def format_lead_table(metrics: Dict[str, np.ndarray],
                      lead_days: List[int]) -> str:
    """生成一个 ASCII 表：行为指标，列为指定 lead 天数。

    ``lead_days`` 为 1-based，下标会自动 -1。
    """
    cols = lead_days
    keys = [k for k in ('rmse', 'mae', 'pearson', 'ssim') if k in metrics]
    head = 'lead(d) ' + '  '.join(f'{c:>7d}' for c in cols)
    lines = [head]
    for k in keys:
        vals = metrics[k]
        row = f'{k:<7s} ' + '  '.join(f'{vals[c-1]:>7.4f}' for c in cols)
        lines.append(row)
    return '\n'.join(lines)
