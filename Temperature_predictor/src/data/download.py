"""
NOAA OISST v2.1 批量下载脚本
==============================
数据来源: https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/
分辨率: 0.25° × 0.25°，全球日均 SST
文件大小: 每天约 1.5 MB

使用方法 (从仓库根目录)：
    python -m Temperature_predictor.src.data.download
    python -m Temperature_predictor.src.data.download --start 2010 --end 2024
    python -m Temperature_predictor.src.data.download --start 2020 --end 2024 --workers 8
"""

import os
import argparse
import datetime
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================
BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "sea-surface-temperature-optimum-interpolation/v2.1/"
    "access/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
)
# 仓库根目录 = 本文件上溯 4 级 (src/data/download.py -> Temperature_predictor -> repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAVE_DIR = os.path.join(_REPO_ROOT, "Temperature_predictor", "data", "raw")


def build_file_list(start_year: int, end_year: int):
    """生成日期列表与对应 URL。"""
    files = []
    d = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    while d <= end:
        yyyymm = d.strftime("%Y%m")
        yyyymmdd = d.strftime("%Y%m%d")
        url = BASE_URL.format(yyyymm=yyyymm, yyyymmdd=yyyymmdd)
        fname = f"oisst-avhrr-v02r01.{yyyymmdd}.nc"
        files.append((url, fname, d))
        d += datetime.timedelta(days=1)
    return files


def download_one(url: str, save_path: str, retries: int = 3) -> str:
    """下载单个文件，支持重试，已存在则跳过。"""
    if os.path.exists(save_path):
        return f"[跳过] {os.path.basename(save_path)}"
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, save_path)
            return f"[完成] {os.path.basename(save_path)}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"[404 ] {os.path.basename(save_path)} (文件不存在)"
            if attempt == retries:
                return f"[失败] {os.path.basename(save_path)}: {e}"
        except Exception as e:
            if attempt == retries:
                return f"[失败] {os.path.basename(save_path)}: {e}"
    return f"[失败] {os.path.basename(save_path)}"


def main():
    parser = argparse.ArgumentParser(description="NOAA OISST v2.1 批量下载")
    parser.add_argument("--start", type=int, default=2015, help="起始年份 (含)")
    parser.add_argument("--end", type=int, default=2024, help="结束年份 (含)")
    parser.add_argument("--workers", type=int, default=4, help="并发下载线程数")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="保存目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    file_list = build_file_list(args.start, args.end)
    total = len(file_list)

    print(f"========================================")
    print(f"  NOAA OISST v2.1 批量下载")
    print(f"  年份范围: {args.start} – {args.end}")
    print(f"  文件总数: {total} 天")
    print(f"  预估大小: ~{total * 1.5 / 1024:.1f} GB")
    print(f"  保存目录: {args.save_dir}")
    print(f"  并发线程: {args.workers}")
    print(f"========================================\n")

    done = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                url,
                os.path.join(args.save_dir, fname),
            ): fname
            for url, fname, _ in file_list
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if "[失败]" in result:
                failed += 1
            if done % 50 == 0 or done == total:
                print(f"  进度: {done}/{total}  ({done/total*100:.1f}%)")
            if "[失败]" in result or "[404 ]" in result:
                print(f"  {result}")

    print(f"\n下载完成！成功: {done - failed}, 失败: {failed}, 共: {total}")


if __name__ == "__main__":
    main()
