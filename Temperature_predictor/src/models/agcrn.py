"""
完整 AGCRN 模型：2 层 AGCRN 编码器 + 多步线性预测头。

邀请 ``A_prior`` 作为物理先验，与可学习的自适应邻接 ``A_adapt`` 融合：

.. math::
    \\tilde A = \\lambda A_{prior} + (1-\\lambda) A_{adapt}

其中 ``A_adapt = softmax(ReLU(E_a E_a^\\top))``（Bai Eq.5）。随后以
如下支撑集供给 NAPL-GCN：

* ``cheb_k=2``：``[I, \\tilde A]``  (默认)
* ``cheb_k=k``：使用切比雪夫递推 ``T_0=I, T_1=\\tilde A, T_k=2\\tilde A T_{k-1}-T_{k-2}``

输入 / 输出
-----------
``forward(x)``:
    x  : ``(B, T_in, N)``     SSTA 序列
    y  : ``(B, T_out, N)``    多步 SSTA 预测
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from .agcrn_cell import AGCRNCell


__all__ = ['AGCRN', 'load_prior_dense']


def load_prior_dense(npz_path: str | Path) -> torch.Tensor:
    """从 ``A_prior.npz`` 加载稀疏附接，返回 dense ``torch.Tensor`` ``(N, N)``。"""
    arr = np.load(npz_path)
    A = sp.coo_matrix((arr['data'], (arr['row'], arr['col'])),
                      shape=tuple(arr['shape'])).toarray().astype(np.float32)
    return torch.from_numpy(A)


def _build_supports(A: torch.Tensor, cheb_k: int) -> List[torch.Tensor]:
    """由 ``A`` 生成 ``cheb_k`` 个支撑集。

    cheb_k = 1 → [I]；cheb_k = 2 → [I, A]；更高阶使用 Cheb 递推。
    调用者应保证 ``A`` 为对称归一化形式。
    """
    if cheb_k < 1:
        raise ValueError('cheb_k must be >= 1')
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    supports = [I]
    if cheb_k >= 2:
        supports.append(A)
    for _ in range(cheb_k - 2):
        T_prev2, T_prev = supports[-2], supports[-1]
        supports.append(2.0 * (A @ T_prev) - T_prev2)
    return supports


def _sym_normalize(A: torch.Tensor, add_self_loop: bool = True,
                   eps: float = 1e-12) -> torch.Tensor:
    """:math:`D^{-1/2} (A+I) D^{-1/2}` 。在 dense 张量上运算。"""
    if add_self_loop:
        A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    deg = A.sum(dim=1).clamp_min(eps)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class AGCRN(nn.Module):
    """面向 SST 多步预测的融合先验图 + 自适应图的 AGCRN。

    Parameters
    ----------
    num_nodes : int
        节点数 N。
    T_in / T_out : int
        输入 / 输出窗口长度。
    c_in : int, default 1
        每个节点输入特征维，SST 单变量设为 1。
    hidden_dim : int, default 64
    num_layers : int, default 2
    cheb_k : int, default 2
    embed_dim : int, default 10
        节点嵌入维库 (供 NAPL 使用)。
    adapt_dim : int, default 10
        自适应邻接嵌入维 (供 A_adapt 生成)。
    A_prior : torch.Tensor, optional
        形状 ``(N, N)`` 的先验邻接。可以在事后调用 :py:meth:`set_prior` 设置。
    lambda_fuse : float, default 0.5
        融合系数 λ。
    """

    def __init__(
        self,
        num_nodes: int,
        T_in: int,
        T_out: int,
        c_in: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cheb_k: int = 2,
        embed_dim: int = 10,
        adapt_dim: int = 10,
        A_prior: Optional[torch.Tensor] = None,
        lambda_fuse: float = 0.5,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.c_in = int(c_in)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.cheb_k = int(cheb_k)
        self.lambda_fuse = float(lambda_fuse)
        self.use_checkpoint = bool(use_checkpoint)

        # 节点嵌入（NAPL）
        self.node_emb = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # 自适应邻接嵌入
        self.adapt_emb = nn.Parameter(torch.empty(num_nodes, adapt_dim))
        nn.init.xavier_uniform_(self.adapt_emb)

        # 堆叠 AGCRN cell
        cells = []
        for layer in range(self.num_layers):
            cells.append(AGCRNCell(
                c_in=c_in if layer == 0 else hidden_dim,
                hidden=hidden_dim, cheb_k=cheb_k, embed_dim=embed_dim,
            ))
        self.cells = nn.ModuleList(cells)

        # 多步输出头：从 (B, N, hidden) 映射到 (B, N, T_out)
        self.head = nn.Conv2d(
            in_channels=hidden_dim, out_channels=T_out,
            kernel_size=(1, 1), bias=True,
        )

        # 先验邻接 buffer（不参与梯度，但会随 model.to(device) 迁移）
        if A_prior is not None:
            self.set_prior(A_prior)
        else:
            self.register_buffer(
                'A_prior_norm',
                torch.eye(num_nodes, dtype=torch.float32),
                persistent=False,
            )
            self._has_prior = False

    # ------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------
    def set_prior(self, A: torch.Tensor) -> None:
        """设置先验图，需 dense ``(N, N)``。会做对称归一化（含自环）。"""
        if A.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f'A_prior shape {tuple(A.shape)} mismatch '
                             f'num_nodes={self.num_nodes}')
        A_norm = _sym_normalize(A.float(), add_self_loop=True)
        # 如果该 buffer 已存在则赋值，否则注册
        if hasattr(self, 'A_prior_norm') and isinstance(
            getattr(self, 'A_prior_norm'), torch.Tensor
        ):
            self.A_prior_norm = A_norm.to(self.A_prior_norm.device)
        else:
            self.register_buffer('A_prior_norm', A_norm, persistent=False)
        self._has_prior = True

    def _build_adapt(self) -> torch.Tensor:
        """A_adapt = softmax(ReLU(E_a E_a^T))。返回 ``(N, N)``，已行归一化。"""
        return F.softmax(F.relu(self.adapt_emb @ self.adapt_emb.t()), dim=-1)

    def _fuse_supports(self) -> List[torch.Tensor]:
        """生成当前 batch 可以复用的 Cheb 支撑集。"""
        A_adapt = self._build_adapt()
        if self._has_prior:
            A_used = self.lambda_fuse * self.A_prior_norm \
                     + (1.0 - self.lambda_fuse) * A_adapt
        else:
            A_used = A_adapt
        return _build_supports(A_used, self.cheb_k)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 ``(B, T_in, N)``（单特征，内部会添加 c_in 维），输出 ``(B, T_out, N)``。"""
        if x.dim() == 3:
            x = x.unsqueeze(-1)              # (B, T_in, N, 1)
        elif x.dim() != 4:
            raise ValueError(f'unexpected x.dim()={x.dim()}, expected 3 or 4')
        B, T_in, N, C = x.shape
        if N != self.num_nodes:
            raise ValueError(f'N={N} != num_nodes={self.num_nodes}')
        if C != self.c_in:
            raise ValueError(f'c_in={C} != model.c_in={self.c_in}')

        supports = self._fuse_supports()
        device, dtype = x.device, x.dtype

        # 初始隐状态
        h_list = [cell.init_hidden(B, N, device, dtype) for cell in self.cells]

        # 逐时间步展开；训练时对每个时间步使用 gradient checkpointing，
        # 在反向传播时重算激活以大幅降低显存（3~5×）。
        use_ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()

        def step(inp_t, *h_in):
            h_out = []
            cur = inp_t
            for li, cell in enumerate(self.cells):
                cur = cell(cur, h_in[li], self.node_emb, supports)
                h_out.append(cur)
            return tuple(h_out)

        for t in range(T_in):
            inp = x[:, t]                                    # (B, N, c_in)
            if use_ckpt:
                h_tuple = _ckpt.checkpoint(
                    step, inp, *h_list, use_reentrant=False
                )
                h_list = list(h_tuple)
            else:
                h_list = list(step(inp, *h_list))

        # 多步预测：用末状 (B, N, hidden) 过卷积头
        h_top = h_list[-1]                                   # (B, N, H)
        h_top = h_top.permute(0, 2, 1).unsqueeze(-1)          # (B, H, N, 1)
        y = self.head(h_top)                                 # (B, T_out, N, 1)
        y = y.squeeze(-1)                                    # (B, T_out, N)
        return y
