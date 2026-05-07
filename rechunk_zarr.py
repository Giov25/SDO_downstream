"""
Rechunk zarr from (15, 1024, 1024) to (1, 1024, 1024).

This reduces per-sample disk I/O from ~440 MB to ~30 MB (15x) when using
random access (shuffle=True). Runs each year-channel pair in a separate
process to fully saturate disk bandwidth.

Usage:
    conda run -n SDOenv python rechunk_zarr.py [--workers N]

Or via SLURM:
    sbatch rechunk_zarr.sbatch
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zarr
from numcodecs import Blosc

SRC = '/home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr'
DST = '/home/gpatane/Dataset/zarr_file_magnetogram_1024_rechunked.zarr'
NEW_CHUNKS = (1, 1024, 1024)
COMPRESSOR = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)


def rechunk_array(task):
    """Copy one (year, channel) array with new chunk size."""
    year, channel, src_path, dst_path = task
    t0 = time.time()
    try:
        src_root = zarr.open(src_path, mode='r')
        dst_root = zarr.open(dst_path, mode='a')

        src_arr = src_root[year][channel]
        n = src_arr.shape[0]

        dst_arr = dst_root.require_dataset(
            f'{year}/{channel}',
            shape=src_arr.shape,
            dtype=src_arr.dtype,
            chunks=NEW_CHUNKS,
            compressor=COMPRESSOR,
            overwrite=False,
        )

        # Copy array-level attrs (e.g. T_OBS per channel)
        dst_arr.attrs.update(dict(src_arr.attrs))

        # Copy group-level attrs for the year group (T_OBS timestamps, etc.)
        src_year_grp = src_root[year]
        dst_year_grp = dst_root.require_group(year)
        dst_year_grp.attrs.update(dict(src_year_grp.attrs))

        # Copy data in batches of 30
        batch = 30
        for i in range(0, n, batch):
            end = min(i + batch, n)
            dst_arr[i:end] = src_arr[i:end]

        elapsed = time.time() - t0
        return year, channel, n, elapsed, None
    except Exception as e:
        return year, channel, 0, 0.0, str(e)


def build_tasks(src_path):
    z = zarr.open(src_path, mode='r')
    tasks = []
    for year in sorted(z.keys()):
        for channel in sorted(z[year].keys()):
            tasks.append((year, channel, src_path, DST))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16,
                        help='Parallel workers (default: 16)')
    args = parser.parse_args()

    tasks = build_tasks(SRC)
    total = len(tasks)
    print(f'Tasks: {total}  ({args.workers} workers)')
    print(f'Source : {SRC}')
    print(f'Dest   : {DST}')
    print(f'Chunks : {NEW_CHUNKS}')
    print()

    t_start = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rechunk_array, t): t for t in tasks}
        for fut in as_completed(futures):
            year, channel, n, elapsed, err = fut.result()
            done += 1
            if err:
                print(f'  [ERROR] {year}/{channel}: {err}', flush=True)
            else:
                rate = n / elapsed if elapsed > 0 else 0
                print(
                    f'  [{done:3d}/{total}] {year}/{channel}: {n} imgs in {elapsed:.1f}s ({rate:.1f} img/s)',
                    flush=True,
                )

    total_elapsed = time.time() - t_start
    print(f'\nDone in {total_elapsed/60:.1f} min')
    print(f'Output zarr: {DST}')


if __name__ == '__main__':
    main()
