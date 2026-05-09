"""
零参数基线。

提供两个与 AGCRN 接口一致的零参数预报器：

* :class:`Persistence`  持续性预报：以历史窗口最后一天重复作为未来预测。
* :class:`Climatology`  气候态预报：在 SSTA 空间上总是输出零。

> 说明：本项目预处理已在节点维减去逐 day-of-year 气候态并 z-score 归一化，
> 所以在 SSTA 空间里，\"气候态\"就是零。这是一个极弱但合法的下界。
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ['Persistence', 'Climatology']


class Persistence(nn.Module):
    """:math:`\\hat X_{t+k}=X_t` 与 ``T_out`` 无关。"""

    def __init__(self, T_out: int) -> None:
        super().__init__()
        self.T_out = int(T_out)
        # 让 ``model.parameters()`` 返回不为空（避免 Adam 报错）
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B, T_in, N)
        last = x[:, -1:, :]                                # (B, 1, N)
        return last.expand(-1, self.T_out, -1).contiguous()


class Climatology(nn.Module):
    """SSTA 空间上的气候态预报始终输出零。"""

    def __init__(self, T_out: int) -> None:
        super().__init__()
        self.T_out = int(T_out)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N = x.shape
        return torch.zeros(B, self.T_out, N, device=x.device, dtype=x.dtype)
