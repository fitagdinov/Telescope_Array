"""
Add dt_params_rubsov_approximation to a normalized phd_work h5 file.

For each detector hit:
  cols 0-2: copied from dt_params (normalized, unchanged)
  cols 3-5: forward model from recos + geometry (iterate.cpp physics via data_reader)

Usage:
  python make_rubsov_approx_h5.py [--input PATH] [--output PATH] [--step N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import h5py as h5
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reconstruction"))
from data_reader import reconstruct_dt_params  # noqa: E402

SPLITS = ("train", "test", "val")
NEW_KEY = "dt_params_rubsov_approximation"


def _default_paths() -> tuple[str, str]:
    base = "/home3/rfit/Telescope_Array/phd_work/data/normed"
    if not os.path.isdir(base):
        base = "/home/rfit/Telescope_Array/phd_work/data/normed"
    src = os.path.join(base, "pr_q4_1895_no_sat_no_geo_0110_bundled_one_work.h5")
    dst = os.path.join(base, "pr_q4_1895_no_sat_no_geo_0110_bundled_one_work_rubsov_approx.h5")
    return src, dst


def _fill_split(
    hf: h5.File,
    split: str,
    dt_mean: np.ndarray,
    dt_std: np.ndarray,
    recos_mean: np.ndarray,
    recos_std: np.ndarray,
    step: int,
) -> None:
    grp = hf[split]
    dt = grp["dt_params"]
    recos = grp["recos"]
    ev_starts = grp["ev_starts"][()]
    out = grp[NEW_KEY]

    n_events = len(ev_starts) - 1
    if n_events == 0:
        return

    for i0 in tqdm(range(0, n_events, step), desc=split, leave=True):
        i1 = min(i0 + step, n_events)
        for ev in range(i0, i1):
            st, fn = int(ev_starts[ev]), int(ev_starts[ev + 1])
            if st == fn:
                continue

            dt_norm = dt[st:fn]
            # Denorm for forward model (physical units), then re-norm cols 3-5 for storage.
            dt_phys = dt_norm * dt_std + dt_mean
            recos_phys = recos[ev] * recos_std + recos_mean

            pred_phys = reconstruct_dt_params(
                recos_phys,
                dt_phys[:, :3],
                observed=dt_phys,
            )

            block = np.empty_like(dt_norm, dtype=np.float32)
            block[:, :3] = dt_norm[:, :3]
            dt_std_safe = np.where(np.abs(dt_std) < 1e-12, 1.0, dt_std)
            block[:, 3:6] = (
                (pred_phys[:, 3:6] - dt_mean[3:6]) / dt_std_safe[3:6]
            ).astype(np.float32)
            out[st:fn] = block


def build(output: str, source: str, step: int, skip_copy: bool = False) -> None:
    if not os.path.exists(source):
        raise FileNotFoundError(source)

    if os.path.exists(output) and not skip_copy:
        raise FileExistsError(f"Output already exists: {output}")

    if not skip_copy:
        print(f"Copying {source} -> {output} ...")
        shutil.copy2(source, output)
        print("Copy done.")

    dt_mean = None
    dt_std = None
    recos_mean = None
    recos_std = None

    with h5.File(output, "r+") as hf:
        dt_np = hf["norm_param/dt_params"]
        dt_mean = np.asarray(dt_np["mean"][()], dtype=np.float64)
        dt_std = np.asarray(dt_np["std"][()], dtype=np.float64)
        rec_np = hf["norm_param/recos"]
        recos_mean = np.asarray(rec_np["mean"][()], dtype=np.float64)
        recos_std = np.asarray(rec_np["std"][()], dtype=np.float64)

        for split in SPLITS:
            if split not in hf:
                continue
            n_hits = hf[split]["dt_params"].shape[0]
            if NEW_KEY not in hf[split]:
                if n_hits == 0:
                    hf[split].create_dataset(NEW_KEY, shape=(0, 6), dtype=np.float32)
                else:
                    hf[split].create_dataset(
                        NEW_KEY,
                        shape=(n_hits, 6),
                        dtype=np.float32,
                        chunks=(min(65536, n_hits), 6),
                    )

        if NEW_KEY not in hf["norm_param"]:
            norm_grp = hf["norm_param"].create_group(NEW_KEY)
            norm_grp.create_dataset("mean", data=dt_mean.astype(np.float32))
            norm_grp.create_dataset("std", data=dt_std.astype(np.float32))

        for split in SPLITS:
            if split not in hf:
                continue
            _fill_split(hf, split, dt_mean, dt_std, recos_mean, recos_std, step)

    print(f"Written: {output}")


def main() -> None:
    default_in, default_out = _default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=default_in)
    parser.add_argument("--output", default=default_out)
    parser.add_argument("--step", type=int, default=5000, help="events per tqdm step")
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="output file is already a copy of input; only fill the new dataset",
    )
    args = parser.parse_args()
    build(args.output, args.input, args.step, skip_copy=args.skip_copy)


if __name__ == "__main__":
    main()
