"""
训练性能微基准：跡期、指标估计、显存占用。
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import json
from Temperature_predictor.src.data.dataset import get_dataloaders
from Temperature_predictor.src.models.agcrn import AGCRN, load_prior_dense
from Temperature_predictor.src.train.config import Config


def main() -> None:
    cfg = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg.data_dir, T_in=cfg.T_in, T_out=cfg.T_out,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )
    print(f'device={device}  batch_size={cfg.batch_size}  '
          f'train_batches={len(train_loader)}  val_batches={len(val_loader)}')

    with open(Path(cfg.data_dir) / 'meta.json', 'r', encoding='utf-8') as f:
        N = int(json.load(f)['N'])
    A_prior = load_prior_dense(Path(cfg.graph_dir) / 'A_prior.npz')
    model = AGCRN(
        num_nodes=N, T_in=cfg.T_in, T_out=cfg.T_out,
        c_in=1, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        cheb_k=cfg.cheb_k, embed_dim=cfg.embed_dim, adapt_dim=cfg.adapt_dim,
        A_prior=A_prior, lambda_fuse=cfg.lambda_fuse,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'AGCRN params = {n_params/1e6:.3f} M')

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.L1Loss()

    # 热身 3 个 batch
    model.train()
    it = iter(train_loader)
    for _ in range(3):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 计时 20 个 batch
    N_BENCH = 20
    t0 = time.perf_counter()
    for i in range(N_BENCH):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    s_per_batch = dt / N_BENCH

    # 测 val 同时占用（推理模式）
    model.eval()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_val = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _ = model(x)
            n_val += 1
    if device == 'cuda':
        torch.cuda.synchronize()
    dt_val = time.perf_counter() - t1

    epoch_train = s_per_batch * len(train_loader)
    epoch_val = dt_val
    epoch_total = epoch_train + epoch_val

    print(f'\n--- per-batch timing ({N_BENCH} batches) ---')
    print(f'train:  {s_per_batch*1000:.1f} ms / batch')
    print(f'val:    {dt_val/n_val*1000:.1f} ms / batch')
    print(f'\n--- per-epoch ---')
    print(f'train phase   = {epoch_train:>6.1f} s  ({len(train_loader)} batches)')
    print(f'val phase     = {epoch_val:>6.1f} s  ({n_val} batches)')
    print(f'1 epoch total = {epoch_total:>6.1f} s  = {epoch_total/60:.2f} min')

    print(f'\n--- estimated total training (with early stop ~50 epochs) ---')
    print(f'50 epochs  ~ {50*epoch_total/60:.1f} min  = {50*epoch_total/3600:.2f} h')
    print(f'100 epochs ~ {100*epoch_total/60:.1f} min = {100*epoch_total/3600:.2f} h')

    if device == 'cuda':
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f'\nGPU peak memory = {peak_gb:.2f} GB')


if __name__ == '__main__':
    main()
