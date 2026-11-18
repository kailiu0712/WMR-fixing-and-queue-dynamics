"""Replay the expanded 6E L3 days in bounded parallel workers for EDA caching.

Run from project with its virtual environment after all 92 L3 days exist.
The final `wmr_eda.py` run reads these identical daily cache files and creates
the combined tables, plots, and report.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

PROJECT = Path(r"D:\lkh\Cornell\26 Fall\5257\project")
sys.path.insert(0, str(PROJECT))
from wmr_eda import process_day, tick_size


def cache_paths(path: Path) -> tuple[list[Path], Path]:
    folder = PROJECT / "outputs" / "wmr_eda" / "daily_cache_v1"
    cache = [folder / f"{path.stem}_{kind}.parquet" for kind in
             ("samples", "shape", "candidates", "add_sizes")]
    return cache, folder / f"{path.stem}_diagnostics.json"


def replay(path: Path, tick: float) -> tuple[str, int, int]:
    cache, diag_path = cache_paths(path)
    if all(p.is_file() for p in cache) and diag_path.is_file():
        return path.name, -1, -1
    samples, shape, candidates, add_sizes, diag = process_day(path, PROJECT, tick)
    for frame, dest in zip((samples, shape, candidates, add_sizes), cache):
        frame.to_parquet(dest, index=False)
    diag_path.write_text(json.dumps(dict(diag)), encoding="utf-8")
    return path.name, len(samples), len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--available", action="store_true",
                        help="Cache completed L3 files already on disk while builds continue")
    parser.add_argument("--start-day", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument("--end-day", type=date.fromisoformat, default=date(2026, 10, 14),
                        help="exclusive UTC day")
    args = parser.parse_args()
    folder = PROJECT / "outputs" / "wmr_eda" / "daily_cache_v1"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    day, stop = args.start_day, args.end_day
    while day < stop:
        symbol = "6EU6" if day < date(2026, 9, 11) else "6EZ6"
        paths.append(PROJECT / "data_6e" / "l3" / f"{symbol}_mbo_{day}.parquet")
        day += timedelta(days=1)
    missing = [p.name for p in paths if not p.is_file()]
    if missing and not args.available:
        raise FileNotFoundError(f"Missing {len(missing)} daily L3 files: {missing[:5]}")
    if args.available:
        paths = [p for p in paths if p.is_file() and time.time() - p.stat().st_mtime > 120]
    todo = [p for p in paths if not (all(x.is_file() for x in cache_paths(p)[0])
                                      and cache_paths(p)[1].is_file())]
    print(f"Caching {len(todo)} of {len(paths)} UTC days with {args.workers} workers", flush=True)
    tick = tick_size(PROJECT)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(replay, p, tick): p for p in todo}
        for future in as_completed(futures):
            name, samples, candidates = future.result()
            print(f"cached {name}: {samples:,} samples; {candidates:,} candidates", flush=True)


if __name__ == "__main__":
    main()
