"""
超参与路径配置。

使用 ``dataclass`` 集中管理 AGCRN 训练/评估需要的参数，
避免硬编码散落在多个脚本中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


__all__ = ['Config']


# Temperature_predictor/ 根目录（config.py 在 src/train/ 下，向上 2 层）
_PROJ_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """训练/评估超参。"""

    # ---------- 路径 ----------
    data_dir: str = str(_PROJ_ROOT / 'data' / 'processed')
    graph_dir: str = str(_PROJ_ROOT / 'data' / 'processed' / 'graph')
    exp_dir: str = str(_PROJ_ROOT / 'experiments')

    # ---------- 窗口 ----------
    T_in: int = 30
    T_out: int = 30

    # ---------- 模型 ----------
    embed_dim: int = 10           # 节点嵌入 d (NAPL)
    adapt_dim: int = 10           # 自适应邻接嵌入 d_a
    hidden_dim: int = 64
    num_layers: int = 2
    cheb_k: int = 2               # NAPL-GCN 切比雪夫阶数
    lambda_fuse: float = 0.5      # A_used = lambda * A_prior + (1-lambda) * A_adapt

    # ---------- 训练 ----------
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 15
    grad_clip: float = 5.0
    num_workers: int = 0
    seed: int = 42

    # ---------- 加速开关（云端推荐 use_amp=True / use_checkpoint=False） ----------
    use_amp: bool = True              # 训练时启用混合精度 (autocast + GradScaler)
    use_checkpoint: bool = False      # 模型内部时间步 gradient checkpointing；4GB 显卡时改 True

    # ---------- 评估 ----------
    # 以 1-based 天数报告；evaluate.py 会取 lead_days[i]-1 作为下标
    lead_days: List[int] = field(default_factory=lambda: [1, 7, 14, 30])

    # ---------- 设备 ----------
    device: str = 'cuda'          # 'cuda' | 'cpu'；train.py 会自动 fallback

    def to_dict(self) -> dict:
        return asdict(self)
