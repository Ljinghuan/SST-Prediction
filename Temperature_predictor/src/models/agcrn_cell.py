"""
AGCRN 单元：NAPL-GCN + GRU。

本文件实现 AGCRN (Bai et al., NeurIPS 2020) 的两个核心组件：

* :class:`AVWGCN`  节点自适应参数学习的图卷积（NAPL-GCN, Bai Eq.4 与 Eq.6）
* :class:`AGCRNCell` 用 NAPL-GCN 替换 GRU 中 MLP 的计算单元（Bai Eq.7）

记号
----
* ``B`` batch
* ``N`` 节点数
* ``E`` 节点嵌入维 ``embed_dim``
* ``K`` Cheb 阶数
* ``c_in`` / ``c_out`` 输入 / 输出通道维

NAPL-GCN 计算
--------------
给定节点表示 ``X ∈ R^{B × N × c_in}`` 与节点嵌入 ``E ∈ R^{N × d_e}``，
在 ``support_list = [I, P_1, ..., P_{K-1}]`` 上依次作图卷积，拼接得到
``X_g ∈ R^{B × N × K × c_in}``；随后以节点嵌入从参数池中生成逐节点参数：

    W_i = E_i \\cdot W_{pool},  W_{pool} ∈ R^{d_e × K × c_in × c_out}
    b_i = E_i \\cdot b_{pool},  b_{pool} ∈ R^{d_e × c_out}

并为每个节点应用自己的 ``W_i, b_i``，得到 ``Z ∈ R^{B × N × c_out}``。

AGCRN cell 结构（与 GRU 类似）
----------------------------
.. math::

    z_t = \\sigma(\\text{GCN}_z([x_t, h_{t-1}]))                              \\\\
    r_t = \\sigma(\\text{GCN}_r([x_t, h_{t-1}]))                              \\\\
    \\tilde h_t = \\tanh(\\text{GCN}_h([x_t, r_t \\odot h_{t-1}]))             \\\\
    h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde h_t

其中 ``z_t`` 与 ``r_t`` 一起预测，因此代码中合并为一个大 GCN。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['AVWGCN', 'AGCRNCell']


class AVWGCN(nn.Module):
    """Adaptive Vertex-Wise GCN (Bai Eq.4 + Eq.6)。

    Parameters
    ----------
    c_in, c_out : int
        输入 / 输出通道维。
    cheb_k : int
        支撑集数量（包含 ``I``，默认 2：即 [I, P]）。
    embed_dim : int
        节点嵌入维。
    """

    def __init__(self, c_in: int, c_out: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.cheb_k = int(cheb_k)
        # 参数池，逐节点生成 W_i / b_i
        self.W_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, c_in, c_out))
        self.b_pool = nn.Parameter(torch.empty(embed_dim, c_out))
        nn.init.xavier_uniform_(self.W_pool)
        nn.init.zeros_(self.b_pool)

    def forward(
        self,
        x: torch.Tensor,                  # (B, N, c_in)
        node_emb: torch.Tensor,           # (N, embed_dim)
        supports: List[torch.Tensor],     # [I, P, ...] 每个 (N, N)，长度 == cheb_k
    ) -> torch.Tensor:
        """返回 ``(B, N, c_out)``。"""
        if len(supports) != self.cheb_k:
            raise ValueError(
                f'len(supports)={len(supports)} != cheb_k={self.cheb_k}'
            )
        # 堆叠为 (K, N, N) 与 (K, N, M) · (B, M, c_in) → (K, B, N, c_in)
        support_stack = torch.stack(supports, dim=0)
        x_g = torch.einsum('knm,bmc->kbnc', support_stack, x)
        x_g = x_g.permute(1, 2, 0, 3).contiguous()   # (B, N, K, c_in)

        # 三项一次性 einsum，由 PyTorch 选择优收缩顺序，
        # 避免显式生成 (N, K, c_in, c_out) 的逐节点 W 大张量。
        out = torch.einsum('bnki,nd,dkio->bno', x_g, node_emb, self.W_pool)
        out = out + node_emb @ self.b_pool         # (B, N, c_out)
        return out


class AGCRNCell(nn.Module):
    """AGCRN GRU 单元（Bai Eq.7）。

    Parameters
    ----------
    c_in : int
        输入特征维（在 SST 任务中为 1）。
    hidden : int
        隐状态维。
    cheb_k : int
        Cheb 阶数（默认 2）。
    embed_dim : int
        节点嵌入维。
    """

    def __init__(self, c_in: int, hidden: int,
                 cheb_k: int = 2, embed_dim: int = 10) -> None:
        super().__init__()
        self.hidden = int(hidden)
        # gates_gcn: 同时输出 update + reset gate
        self.gates_gcn = AVWGCN(c_in + hidden, 2 * hidden, cheb_k, embed_dim)
        # cand_gcn: 输出候选隐状态
        self.cand_gcn  = AVWGCN(c_in + hidden,     hidden, cheb_k, embed_dim)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, c_in)
        h: torch.Tensor,              # (B, N, hidden)
        node_emb: torch.Tensor,       # (N, embed_dim)
        supports: List[torch.Tensor], # [I, P]
    ) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)                       # (B, N, c_in + hidden)
        zr = torch.sigmoid(self.gates_gcn(xh, node_emb, supports))  # (B, N, 2H)
        z, r = torch.split(zr, self.hidden, dim=-1)
        xr = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.cand_gcn(xr, node_emb, supports)) # (B, N, H)
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new

    def init_hidden(self, batch_size: int, num_nodes: int,
                    device: torch.device, dtype: torch.dtype = torch.float32
                    ) -> torch.Tensor:
        return torch.zeros(batch_size, num_nodes, self.hidden,
                           device=device, dtype=dtype)
