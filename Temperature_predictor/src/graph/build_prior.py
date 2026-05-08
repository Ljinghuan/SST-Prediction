"""
Stage B：基于 Ning (2023) 的先验图构造
========================================================================

思路（呼应开题 PPT slide 8 与 PLAN Stage B）：
  1. 仅用训练集 SSTA `[T_train, N]`（已去季节化），避免标签泄漏；
  2. 对每对节点 (i, j) 计算 Pearson 相关 ρ_ij；可选启用 lagged corr
     τ ∈ {0,1,...,L}，取 max_τ |ρ_ij(τ)|，以捕获 ENSO 等遥相关的滞后传播；
  3. 阈值化：对每个节点取 |ρ| 最大的 top-k 邻居（k 控制平均度，k≈30）；
     可选叠加全局阈值 |ρ| ≥ c；
  4. 对称化：A = max(A, A.T)（边的并集，无向）；
  5. 加自环 + 对称归一化：Ã = D^{-1/2} (A + I) D^{-1/2}（GCN 标准形式）；
  6. 输出稀疏 COO 与密集矩阵，并可视化邻接热图与若干"高远程连通"节点的地理连边图。

输入：
  Temperature_predictor/data/processed/
    - ssta.npy        [T, N]   去季节化 SSTA（°C，未归一化，量纲一致）
    - splits.json     训练切片
    - coords.npy      [N, 2]   (lat, lon)
    - mask_2d.npy     [H, W]   仅用于复原网格画图（可选）
    - meta.json       lat/lon 数组（画地理图时用）

输出：
  Temperature_predictor/data/processed/graph/
    - A_prior.npz       稀疏 COO: row, col, data, shape, meta
    - A_prior_dense.npy [N, N] float32 已归一化邻接 Ã
    - A_raw_dense.npy   [N, N] float32 二值邻接（top-k 后、未归一化、未加自环）
    - corr.npy          [N, N] float32 (max_τ |ρ|) 相关矩阵（diag 置 0）
    - graph_meta.json   构造参数 + 平均度等统计
    - figs/adjacency.png            邻接矩阵热图（呼应 PPT 右图）
    - figs/teleconnections.png      地理图：top-k 节点 + 若干远程边
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
PROC_DIR = _REPO_ROOT / "Temperature_predictor" / "data" / "processed"
GRAPH_DIR = PROC_DIR / "graph"
FIG_DIR = GRAPH_DIR / "figs"


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage B: 构造先验图 A_prior (Ning 2023 风格)")
    p.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    p.add_argument("--out-dir", type=Path, default=GRAPH_DIR)
    p.add_argument("--top-k", type=int, default=30, help="每个节点保留 |ρ| 最大的 k 个邻居")
    p.add_argument("--corr-threshold", type=float, default=0.0,
                   help="附加阈值: 仅保留 |ρ|>=c 的边 (0 表示仅依赖 top-k)")
    p.add_argument("--max-lag", type=int, default=0,
                   help="lagged correlation 最大滞后天数 L (0 关闭，仅同期)")
    p.add_argument("--symmetrize", choices=["max", "mean"], default="max",
                   help="对称化方式: max=取两向较大 (并集), mean=平均")
    p.add_argument("--use-abs", action="store_true", default=True,
                   help="按 |ρ| 选 top-k；负相关也算邻居 (默认开)")
    p.add_argument("--no-self-loop", action="store_true", help="不加 I (默认会加)")
    p.add_argument("--no-fig", action="store_true", help="跳过画图")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------
def load_training_ssta(proc_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """读取 ssta.npy 并按 splits.json 切出训练段；同时返回 coords 与 meta。"""
    ssta = np.load(proc_dir / "ssta.npy")  # [T, N]
    coords = np.load(proc_dir / "coords.npy")
    with open(proc_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    s0, s1 = splits["train"]
    train = ssta[s0:s1].astype(np.float32, copy=False)  # [T_train, N]
    print(f"[load] ssta full={ssta.shape}, train_slice=[{s0}:{s1}] -> {train.shape}")
    return train, coords, splits


# ------------------------------------------------------------
# 相关矩阵
# ------------------------------------------------------------
def _zscore_along_time(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """逐节点 z-score 标准化时间序列：[T, N] -> [T, N]，便于 X.T @ X / T = corr."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mu) / sd


def pearson_corr(x: np.ndarray) -> np.ndarray:
    """[T, N] -> [N, N] Pearson 相关 (同期)。"""
    z = _zscore_along_time(x)  # [T, N]
    T = z.shape[0]
    # 用 float32 矩阵乘，N≈数千以内可承受
    corr = (z.T @ z) / np.float32(T)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32, copy=False)


def lagged_max_abs_corr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """带滞后的 |ρ| 最大值矩阵：max_τ∈[0..L] |corr(X_i[t], X_j[t+τ])|。

    返回 [N, N] (带符号: 取得 max|ρ| 时对应那个 τ 的 ρ)。
    复杂度 O((L+1)·T·N^2)，N 不大时直接做。
    """
    T, N = x.shape
    z = _zscore_along_time(x)
    best = np.zeros((N, N), dtype=np.float32)
    best_abs = np.zeros((N, N), dtype=np.float32)
    for tau in range(max_lag + 1):
        if tau == 0:
            a = z
            b = z
        else:
            a = z[:-tau]      # 对应 t
            b = z[tau:]       # 对应 t+τ
        Teff = a.shape[0]
        # corr(i, j; τ) = <a[:,i], b[:,j]> / Teff
        c = (a.T @ b) / np.float32(Teff)  # [N, N]
        ac = np.abs(c)
        mask = ac > best_abs
        best = np.where(mask, c, best)
        best_abs = np.where(mask, ac, best_abs)
    np.fill_diagonal(best, 0.0)
    return best


# ------------------------------------------------------------
# 稀疏化
# ------------------------------------------------------------
def topk_sparsify(
    corr: np.ndarray, k: int, threshold: float, use_abs: bool
) -> np.ndarray:
    """每行保留 |ρ| 前 k 大的位置 -> 二值邻接 [N, N] (float32, 0/1)。
    可选叠加 |ρ| >= threshold 的硬阈值。
    """
    N = corr.shape[0]
    score = np.abs(corr) if use_abs else corr
    # argpartition 取 top-k（不保证顺序，足够）
    k = min(k, N - 1)
    idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]  # [N, k]
    A = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k)
    cols = idx.ravel()
    A[rows, cols] = 1.0
    np.fill_diagonal(A, 0.0)
    if threshold > 0.0:
        A = A * (np.abs(corr) >= threshold).astype(np.float32)
    return A


def symmetrize(A: np.ndarray, mode: str = "max") -> np.ndarray:
    if mode == "max":
        return np.maximum(A, A.T)
    elif mode == "mean":
        return 0.5 * (A + A.T)
    raise ValueError(mode)


def normalize_adj(A: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """对称归一化 D^{-1/2} (A + I) D^{-1/2}。"""
    A = A.astype(np.float32, copy=True)
    if add_self_loop:
        A = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    return (A * d_inv_sqrt[None, :]) * d_inv_sqrt[:, None]


# ------------------------------------------------------------
# 保存
# ------------------------------------------------------------
def save_sparse_coo(A: np.ndarray, path: Path) -> tuple[int, float]:
    """保存为 npz: row, col, data, shape。返回 (边数, 平均度)."""
    rows, cols = np.nonzero(A)
    data = A[rows, cols].astype(np.float32)
    np.savez(
        path,
        row=rows.astype(np.int32),
        col=cols.astype(np.int32),
        data=data,
        shape=np.array(A.shape, dtype=np.int64),
    )
    edges = int(rows.size)
    avg_deg = edges / float(A.shape[0])
    return edges, avg_deg


# ------------------------------------------------------------
# 可视化（matplotlib 可选）
# ------------------------------------------------------------
def _try_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print(f"[viz] matplotlib 不可用，跳过画图: {e}")
        return None


def plot_adjacency(A: np.ndarray, out_path: Path, title: str = "A_prior") -> None:
    plt = _try_import_mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(A > 0, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Node Index i")
    ax.set_ylabel("Node Index j")
    ax.set_title(f"Sparse Adjacency Matrix {title}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_teleconnections(
    A_raw: np.ndarray,
    coords: np.ndarray,
    corr: np.ndarray,
    out_path: Path,
    n_anchors: int = 6,
    seed: int = 0,
) -> None:
    """画几个锚点节点及其远程邻居（地理空间投影）。"""
    plt = _try_import_mpl()
    if plt is None:
        return
    rng = np.random.default_rng(seed)
    N = A_raw.shape[0]
    # 选取边数最多的若干节点作为锚点（更可能含远程边）
    deg = A_raw.sum(axis=1)
    candidate = np.argsort(-deg)[: max(n_anchors * 3, 20)]
    anchors = rng.choice(candidate, size=min(n_anchors, len(candidate)), replace=False)

    lats, lons = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.scatter(lons, lats, s=2, c="lightgray", label="nodes")
    cmap = plt.get_cmap("tab10")
    for k, a in enumerate(anchors):
        nbrs = np.where(A_raw[a] > 0)[0]
        if nbrs.size == 0:
            continue
        c = cmap(k % 10)
        ax.scatter(lons[a], lats[a], s=40, color=c, marker="*",
                   edgecolors="k", linewidths=0.5, zorder=3)
        for j in nbrs:
            ax.plot([lons[a], lons[j]], [lats[a], lats[j]],
                    color=c, alpha=0.35, linewidth=0.5)
        ax.scatter(lons[nbrs], lats[nbrs], s=8, color=c, alpha=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Prior Graph: anchor nodes & their teleconnections")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 训练集 SSTA
    train, coords, splits = load_training_ssta(args.proc_dir)
    T, N = train.shape

    # 2) 相关矩阵（同期 / 滞后取 max|ρ|）
    if args.max_lag > 0:
        print(f"[corr] lagged correlation, L={args.max_lag} (取 max_τ |ρ|)")
        corr = lagged_max_abs_corr(train, args.max_lag)
    else:
        print(f"[corr] 同期 Pearson 相关")
        corr = pearson_corr(train)
    print(f"[corr] |ρ| 分布: mean={np.abs(corr).mean():.3f}, "
          f"p95={np.quantile(np.abs(corr), 0.95):.3f}, max={np.abs(corr).max():.3f}")

    # 3) Top-k 稀疏化（带阈值）
    A_raw = topk_sparsify(corr, k=args.top_k,
                          threshold=args.corr_threshold,
                          use_abs=args.use_abs)
    in_deg_before = A_raw.sum(axis=1).mean()

    # 4) 对称化
    A_sym = symmetrize(A_raw, mode=args.symmetrize)
    A_sym = (A_sym > 0).astype(np.float32)  # 保持 0/1

    # 5) 自环 + 对称归一化
    A_norm = normalize_adj(A_sym, add_self_loop=not args.no_self_loop)

    # 6) 保存
    edges, avg_deg = save_sparse_coo(A_norm, args.out_dir / "A_prior.npz")
    np.save(args.out_dir / "A_prior_dense.npy", A_norm.astype(np.float32))
    np.save(args.out_dir / "A_raw_dense.npy", A_sym.astype(np.float32))
    np.save(args.out_dir / "corr.npy", corr.astype(np.float32))

    stats = {
        "N": int(N), "T_train": int(T),
        "top_k": int(args.top_k),
        "corr_threshold": float(args.corr_threshold),
        "max_lag": int(args.max_lag),
        "symmetrize": args.symmetrize,
        "use_abs": bool(args.use_abs),
        "self_loop": not args.no_self_loop,
        "edges_directed_topk": int((A_raw > 0).sum()),
        "avg_in_deg_before_sym": float(in_deg_before),
        "edges_after_sym": int((A_sym > 0).sum()),
        "avg_deg_after_sym": float(A_sym.sum(axis=1).mean()),
        "edges_normalized_nnz": int(edges),
        "avg_deg_normalized": float(avg_deg),
        "abs_corr_p95": float(np.quantile(np.abs(corr), 0.95)),
        "abs_corr_max": float(np.abs(corr).max()),
    }
    with open(args.out_dir / "graph_meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[save] -> {args.out_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 7) 画图
    if not args.no_fig:
        plot_adjacency(A_sym, FIG_DIR / "adjacency.png", title="A_prior (binary, sym)")
        plot_teleconnections(A_sym, coords, corr,
                             FIG_DIR / "teleconnections.png",
                             n_anchors=6, seed=args.seed)
        print(f"[viz] figs -> {FIG_DIR}")

    print("[done]")


if __name__ == "__main__":
    main()
