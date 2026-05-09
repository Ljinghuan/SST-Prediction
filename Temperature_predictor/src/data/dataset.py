"""
SSTA 滑窗 Dataset
================

加载预处理后的归一化海温异常张量 ``ssta_norm.npy``（shape ``[T, N]``），
按 ``splits.json`` 定义的训练/验证/测试时间区间生成 ``(T_in, T_out)``
滑窗样本，供 PyTorch ``DataLoader`` 使用。

样本定义
--------
对于一个起始下标 ``s``：

* 输入  ``x = ssta_norm[s : s + T_in]``                shape ``(T_in, N)``
* 目标  ``y = ssta_norm[s + T_in : s + T_in + T_out]``  shape ``(T_out, N)``

其中 ``s`` 取自所选 split ``[lo, hi)`` 的 *起始下标合法集*：
``s ∈ [lo, hi - T_in - T_out]``，保证 ``s + T_in + T_out <= hi``，
即整个窗口落在该 split 内、不跨 split。

文件路径
--------
默认从 ``Temperature_predictor/data/processed/`` 目录加载，
若调用者从其它工作目录运行，请显式传入 ``data_dir``。

用法
----
>>> from src.data.dataset import SSTADataset, get_dataloaders
>>> train_loader, val_loader, test_loader = get_dataloaders(
...     data_dir='data/processed', T_in=30, T_out=30, batch_size=32)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


__all__ = ['SSTADataset', 'get_dataloaders']


class SSTADataset(Dataset):
    """SSTA 滑窗数据集。

    Parameters
    ----------
    data_dir : str | Path
        预处理输出目录，需包含 ``ssta_norm.npy`` 与 ``splits.json``。
    split : {'train', 'val', 'test'}
        所选时间切分。
    T_in : int, default 30
        历史窗口长度（天）。
    T_out : int, default 30
        预测窗口长度（天）。
    data : np.ndarray, optional
        直接传入已加载的 ``[T, N]`` 数组以避免重复 IO；若提供则忽略
        ``data_dir`` 中的 ``ssta_norm.npy``，但仍读取 ``splits.json``。
    dtype : torch.dtype, default torch.float32
        输出张量数据类型。
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = 'train',
        T_in: int = 30,
        T_out: int = 30,
        data: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        if T_in < 1 or T_out < 1:
            raise ValueError("T_in and T_out must be >= 1")

        data_dir = Path(data_dir)
        if data is None:
            data = np.load(data_dir / 'ssta_norm.npy')
        if data.ndim != 2:
            raise ValueError(f"expected ssta of shape [T, N], got {data.shape}")
        with open(data_dir / 'splits.json', 'r', encoding='utf-8') as f:
            splits = json.load(f)
        lo, hi = splits[split]

        T_total = data.shape[0]
        if hi > T_total:
            raise ValueError(f"split[{split}].hi={hi} exceeds T={T_total}")
        if hi - lo < T_in + T_out:
            raise ValueError(
                f"split {split!r} length {hi - lo} < T_in+T_out={T_in + T_out}"
            )

        self.data = data
        self.T_in = T_in
        self.T_out = T_out
        self.split = split
        self.dtype = dtype
        # 起始下标合法集：闭区间 [lo, hi - T_in - T_out]
        self.starts = np.arange(lo, hi - T_in - T_out + 1, dtype=np.int64)

    @property
    def N(self) -> int:
        """节点数。"""
        return int(self.data.shape[1])

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = self.data[s : s + self.T_in]                          # (T_in, N)
        y = self.data[s + self.T_in : s + self.T_in + self.T_out]  # (T_out, N)
        return (
            torch.as_tensor(x, dtype=self.dtype),
            torch.as_tensor(y, dtype=self.dtype),
        )


def get_dataloaders(
    data_dir: str | Path,
    T_in: int = 30,
    T_out: int = 30,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train/val/test 三个 ``DataLoader``。

    ``ssta_norm.npy`` 仅加载一次并在三个数据集间共享，节省内存。

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    data_dir = Path(data_dir)
    data = np.load(data_dir / 'ssta_norm.npy')

    common = dict(data_dir=data_dir, T_in=T_in, T_out=T_out, data=data)
    train_set = SSTADataset(split='train', **common)
    val_set = SSTADataset(split='val', **common)
    test_set = SSTADataset(split='test', **common)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kw)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 自检：打印各 split 长度与样本形状
    import os
    here = Path(__file__).resolve().parents[2]  # Temperature_predictor/
    data_dir = here / 'data' / 'processed'
    if not (data_dir / 'ssta_norm.npy').exists():
        raise SystemExit(f"ssta_norm.npy not found under {data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, T_in=30, T_out=30, batch_size=4, num_workers=0
    )
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        x, y = next(iter(loader))
        print(f'[{name}] dataset_size={len(loader.dataset):>5d}  '
              f'x={tuple(x.shape)}  y={tuple(y.shape)}  '
              f'dtype={x.dtype}')
