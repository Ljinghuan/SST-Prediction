"""
OISST 预处理：去季节化(SSTA) + 陆地掩膜 + 节点扁平化 + 归一化 + 时间切分
========================================================================

输入：Temperature_predictor/data/raw/oisst-avhrr-v02r01.YYYYMMDD.nc
输出：Temperature_predictor/data/processed/
    - sst_raw.npy        [T, N]  原始 SST (K/°C，保留以便反算)
    - climatology.npy    [366, N] 逐 DOY 均值 (参考期内)
    - ssta.npy           [T, N]  去季节化异常 (°C)
    - ssta_norm.npy      [T, N]  训练集 z-score 归一化后的 SSTA
    - mask_2d.npy        [H, W]  海洋=1 / 陆地=0 布尔掩膜 (原始网格上)
    - coords.npy         [N, 2]  每个节点的 (lat, lon)
    - dates.npy          [T]     np.datetime64[D]
    - splits.json        {train:[i0,i1], val:[...], test:[...]}
    - norm_stats.json    {mean, std}  (标量，基于训练集 SSTA)
    - meta.json          配置参数与形状信息

使用：
    python -m Temperature_predictor.src.data.preprocess \
        --start 2015 --end 2024 \
        --lat-min -30 --lat-max 30 --lon-min 120 --lon-max 280 \
        --coarsen 4 \
        --clim-start 2015 --clim-end 2021 \
        --train-end 2021 --val-end 2022
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr


# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _REPO_ROOT / "Temperature_predictor" / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "Temperature_predictor" / "data" / "processed"


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OISST 预处理: SSTA + 掩膜 + 扁平化 + 归一化")
    # 时间范围
    p.add_argument("--start", type=int, default=2015, help="起始年份（含）")
    p.add_argument("--end", type=int, default=2024, help="结束年份（含）")
    # 区域 (OISST 经度为 0-360)
    p.add_argument("--lat-min", type=float, default=-30.0)
    p.add_argument("--lat-max", type=float, default=30.0)
    p.add_argument("--lon-min", type=float, default=120.0, help="0-360 经度")
    p.add_argument("--lon-max", type=float, default=280.0, help="0-360 经度")
    # 空间降采样
    p.add_argument("--coarsen", type=int, default=4,
                   help="沿 lat/lon 聚合的因子 (4 → 0.25°变 1°)")
    # climatology 参考期
    p.add_argument("--clim-start", type=int, default=2015)
    p.add_argument("--clim-end", type=int, default=2021)
    # 切分 (按年份闭区间上界)
    p.add_argument("--train-end", type=int, default=2021,
                   help="训练集最后一年（含）")
    p.add_argument("--val-end", type=int, default=2022,
                   help="验证集最后一年（含），之后直到 end 为测试集")
    # I/O
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--no-float32", action="store_true", help="保存为 float64（默认 float32）")
    return p.parse_args()


# ------------------------------------------------------------
# 加载
# ------------------------------------------------------------
def list_files(raw_dir: Path, start: int, end: int) -> list[Path]:
    files = []
    d = datetime.date(start, 1, 1)
    last = datetime.date(end, 12, 31)
    while d <= last:
        f = raw_dir / f"oisst-avhrr-v02r01.{d.strftime('%Y%m%d')}.nc"
        if f.exists():
            files.append(f)
        d += datetime.timedelta(days=1)
    return files


def filter_valid_files(files: list[Path]) -> list[Path]:
    """尝试逐个打开，过滤掉损坏的 nc 文件（NetCDF/HDF 错误）。"""
    good, bad = [], []
    for f in files:
        try:
            with xr.open_dataset(f) as _:
                pass
            good.append(f)
        except Exception as e:
            bad.append((f.name, type(e).__name__))
    if bad:
        print(f"[filter] 跳过 {len(bad)} 个损坏文件，例如: {bad[:5]}")
    return good


def load_subset(
    files: list[Path],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    coarsen: int,
) -> xr.Dataset:
    """懒加载 + 区域子集 + 空间降采样 + 丢掉 zlev 维。"""
    print(f"[load] 打开 {len(files)} 个文件 (lazy)...")
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        parallel=False,
        chunks={"time": 64},
    )[["sst"]]
    # 去掉 zlev=0 维
    if "zlev" in ds.dims:
        ds = ds.isel(zlev=0).drop_vars("zlev", errors="ignore")

    # 区域子集 (OISST 经度单调递增 0.125..359.875)
    ds = ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )

    # 空间降采样（boundary=trim 丢掉不足一个窗口的边缘）
    if coarsen > 1:
        ds = ds.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()

    print(f"[load] 计算中… shape(time,lat,lon)={ds.sst.shape}")
    ds = ds.load()
    print(f"[load] done.")
    return ds


# ------------------------------------------------------------
# climatology (逐 DOY)
# ------------------------------------------------------------
def compute_climatology(
    sst: xr.DataArray, clim_start: int, clim_end: int
) -> xr.DataArray:
    """参考期内按 dayofyear 平均，返回 [366, lat, lon]。
    未出现的 DOY (如部分年份无 366) 用线性插值补全。"""
    ref = sst.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))
    clim = ref.groupby("time.dayofyear").mean("time", skipna=True)
    # 确保 1..366 齐全
    clim = clim.reindex(dayofyear=np.arange(1, 367))
    # 线性插值缺失的 DOY；首尾 NaN 用 numpy 前后向填充
    clim = clim.interpolate_na(dim="dayofyear", method="linear")
    arr = clim.values  # [366, H, W]
    # 沿 dayofyear 轴做前向/后向填充
    for axis_slice in (slice(None, None, 1), slice(None, None, -1)):
        # forward then backward
        cur = arr[axis_slice]
        for i in range(1, cur.shape[0]):
            mask = np.isnan(cur[i])
            cur[i] = np.where(mask, cur[i - 1], cur[i])
        arr[axis_slice] = cur
    clim = clim.copy(data=arr)
    return clim


def subtract_climatology(sst: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    doy = sst["time"].dt.dayofyear
    # 对齐广播
    clim_aligned = clim.sel(dayofyear=doy)
    ssta = sst - clim_aligned.drop_vars("dayofyear")
    ssta.name = "ssta"
    return ssta


# ------------------------------------------------------------
# 掩膜 + 扁平化
# ------------------------------------------------------------
def build_ocean_mask(sst: xr.DataArray) -> np.ndarray:
    """任意时刻出现 NaN 的格点视为陆地/缺测 → 排除。
    返回 [H, W] 布尔：True=海洋节点保留。"""
    finite_always = np.isfinite(sst.values).all(axis=0)
    return finite_always


def flatten_nodes(
    arr: xr.DataArray, mask_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """[T, H, W] → [T, N] + coords [N, 2] (lat, lon)."""
    vals = arr.values  # [T, H, W]
    T = vals.shape[0]
    flat = vals.reshape(T, -1)[:, mask_2d.ravel()]  # [T, N]
    lat_v = arr["lat"].values
    lon_v = arr["lon"].values
    LAT, LON = np.meshgrid(lat_v, lon_v, indexing="ij")
    coords = np.stack([LAT.ravel()[mask_2d.ravel()],
                       LON.ravel()[mask_2d.ravel()]], axis=1)
    return flat, coords


# ------------------------------------------------------------
# 时间切分 + 归一化
# ------------------------------------------------------------
def compute_splits(
    dates: np.ndarray, train_end: int, val_end: int
) -> dict[str, list[int]]:
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    idx_train = np.where(years <= train_end)[0]
    idx_val = np.where((years > train_end) & (years <= val_end))[0]
    idx_test = np.where(years > val_end)[0]
    return {
        "train": [int(idx_train[0]), int(idx_train[-1]) + 1] if len(idx_train) else [0, 0],
        "val":   [int(idx_val[0]),   int(idx_val[-1]) + 1]   if len(idx_val)   else [0, 0],
        "test":  [int(idx_test[0]),  int(idx_test[-1]) + 1]  if len(idx_test)  else [0, 0],
    }


def zscore_normalize(
    ssta: np.ndarray, train_slice: tuple[int, int]
) -> tuple[np.ndarray, float, float]:
    s0, s1 = train_slice
    train = ssta[s0:s1]
    mean = float(np.nanmean(train))
    std = float(np.nanstd(train))
    if std < 1e-8:
        std = 1.0
    return (ssta - mean) / std, mean, std


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 文件列表
    files = list_files(args.raw_dir, args.start, args.end)
    if not files:
        raise FileNotFoundError(f"在 {args.raw_dir} 未找到 {args.start}-{args.end} 的 .nc 文件")
    print(f"[main] 找到文件: {len(files)}; 验证可读性…")
    files = filter_valid_files(files)
    print(f"[main] 可用文件: {len(files)} / 理论 {(args.end - args.start + 1) * 365}+")

    # 2) 加载 + 区域子集 + 降采样
    ds = load_subset(
        files,
        args.lat_min, args.lat_max,
        args.lon_min, args.lon_max,
        args.coarsen,
    )
    sst = ds["sst"]  # [T, H, W]
    dates = sst["time"].values.astype("datetime64[D]")
    H, W = sst.sizes["lat"], sst.sizes["lon"]
    T = sst.sizes["time"]
    print(f"[main] 形状: T={T}, H={H}, W={W}")

    # 3) climatology & SSTA
    print(f"[clim] 参考期 {args.clim_start}-{args.clim_end}")
    clim = compute_climatology(sst, args.clim_start, args.clim_end)  # [366,H,W]
    ssta = subtract_climatology(sst, clim)

    # 4) 海洋掩膜
    mask_2d = build_ocean_mask(sst)
    N = int(mask_2d.sum())
    print(f"[mask] 海洋节点 N = {N} (占 {N/(H*W)*100:.1f}%)")
    if N == 0:
        raise RuntimeError("掩膜后无有效节点，检查经纬度范围")

    # 5) 扁平化
    sst_flat, coords = flatten_nodes(sst, mask_2d)
    ssta_flat, _ = flatten_nodes(ssta, mask_2d)
    clim_flat_T_H_W = clim.values.reshape(366, -1)[:, mask_2d.ravel()]  # [366,N]

    # 6) 切分
    splits = compute_splits(dates, args.train_end, args.val_end)
    print(f"[split] train={splits['train']} val={splits['val']} test={splits['test']}")

    # 7) 归一化 (基于训练集 SSTA)
    ssta_norm, mu, sd = zscore_normalize(ssta_flat, tuple(splits["train"]))
    print(f"[norm] train SSTA mean={mu:.4f}, std={sd:.4f}")

    # 8) 保存
    dtype = np.float64 if args.no_float32 else np.float32
    out = args.out_dir
    np.save(out / "sst_raw.npy", sst_flat.astype(dtype))
    np.save(out / "climatology.npy", clim_flat_T_H_W.astype(dtype))
    np.save(out / "ssta.npy", ssta_flat.astype(dtype))
    np.save(out / "ssta_norm.npy", ssta_norm.astype(dtype))
    np.save(out / "mask_2d.npy", mask_2d)
    np.save(out / "coords.npy", coords.astype(np.float32))
    np.save(out / "dates.npy", dates)

    with open(out / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    with open(out / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump({"mean": mu, "std": sd, "method": "zscore_on_train_ssta"}, f, indent=2)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "start": args.start, "end": args.end,
            "lat_min": args.lat_min, "lat_max": args.lat_max,
            "lon_min": args.lon_min, "lon_max": args.lon_max,
            "coarsen": args.coarsen,
            "clim_start": args.clim_start, "clim_end": args.clim_end,
            "train_end": args.train_end, "val_end": args.val_end,
            "T": int(T), "H": int(H), "W": int(W), "N": int(N),
            "lat_values": sst["lat"].values.tolist(),
            "lon_values": sst["lon"].values.tolist(),
        }, f, indent=2)

    print(f"\n[done] 输出保存到 {out}")
    print(f"       ssta.npy shape = ({T}, {N})")


if __name__ == "__main__":
    main()
