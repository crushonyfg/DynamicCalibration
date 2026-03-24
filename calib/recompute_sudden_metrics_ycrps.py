"""
Recompute sudden-change y_crps metrics from saved experiment files.

Motivation:
- Some methods (e.g., BPC / BOCPD-BPC) may not populate `crps_hist` in their saved results,
  which makes `y_crps` aggregation become nan.
- However, per-step `crps_sim` is often stored inside `others` entries.

This script:
1. Loads all `sudden_*_results.pt` under --out_dir
2. For each (combo, method):
   - y_crps = mean(crps_hist) if available and finite
   - else fallback to mean of `others[*].get("crps_sim")` if any finite values exist
3. Aggregates across all combos and prints a summary table.

It does NOT rerun simulations; it only recomputes metrics from saved files.

python -m calib.recompute_sudden_metrics_ycrps --out_dir figs/suddenCmp_tryThm_v6 --exclude_delta_mag 5.0
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _mean_finite(x: List[float]) -> float:
    if len(x) == 0:
        return float("nan")
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _compute_y_crps_from_result(method_data: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Returns:
      (y_crps, used_fallback)
    """
    crps_hist = method_data.get("crps_hist", None)
    if crps_hist is not None:
        arr = np.asarray(crps_hist, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            return float(arr.mean()), False

    # fallback: others[*].get("crps_sim")
    others = method_data.get("others", None)
    if others is not None and isinstance(others, list):
        vals: List[float] = []
        for it in others:
            if not isinstance(it, dict):
                continue
            v = it.get("crps_sim", None)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
        y_crps = _mean_finite(vals)
        return y_crps, True

    return float("nan"), True


def _parse_combo_from_filename(path: str) -> Dict[str, Any]:
    """
    expected: sudden_L{seg_len}_delta{delta}_bs{batch}_seed{seed}_results.pt
    """
    name = os.path.basename(path)
    # Use a permissive regex (delta_mag is float in tag with :g formatting).
    m = re.match(r"sudden_L(?P<L>[-0-9\.eE]+)_delta(?P<delta>[-0-9\.eE]+)_bs(?P<bs>\d+)_seed(?P<seed>\d+)_results\.pt", name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "seg_len_L": float(d["L"]),
        "delta_mag": float(d["delta"]),
        "batch_size": int(d["bs"]),
        "seed": int(d["seed"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--pattern",
        type=str,
        default="sudden_*_results.pt",
        help="Glob pattern for saved results files.",
    )
    ap.add_argument(
        "--exclude_delta_mag",
        type=float,
        default=None,
        help="If set, skip all combinations whose parsed delta_mag equals this value (e.g., 5.0).",
    )
    ap.add_argument("--print_problem_files", action="store_true", default=False)
    args = ap.parse_args()

    out_dir = args.out_dir
    paths = sorted(glob.glob(os.path.join(out_dir, args.pattern)))
    if len(paths) == 0:
        raise RuntimeError(f"No files found under {out_dir} matching {args.pattern}")

    # per-method per-combo metrics
    per_combo_rows: List[Dict[str, Any]] = []

    for p in paths:
        combo = _parse_combo_from_filename(p)
        if args.exclude_delta_mag is not None and "delta_mag" in combo:
            if abs(float(combo["delta_mag"]) - float(args.exclude_delta_mag)) < 1e-9:
                continue
        res = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(res, dict):
            continue
        for method_name, data in res.items():
            theta = np.asarray(data["theta"], dtype=float)
            theta_oracle = np.asarray(data["theta_oracle"], dtype=float)
            theta_rmse = float(np.sqrt(np.mean((theta - theta_oracle) ** 2)))

            rmse_hist = np.asarray(data.get("rmse", []), dtype=float).reshape(-1)
            rmse_hist_finite = rmse_hist[np.isfinite(rmse_hist)]
            y_rmse_mean = float(rmse_hist_finite.mean()) if rmse_hist_finite.size > 0 else float("nan")

            y_crps, used_fallback = _compute_y_crps_from_result(data)

            per_combo_rows.append(
                dict(
                    combo_file=os.path.basename(p),
                    **combo,
                    method=method_name,
                    theta_rmse=theta_rmse,
                    y_rmse=y_rmse_mean,
                    y_crps=y_crps,
                    used_fallback=used_fallback,
                )
            )

    # aggregate across combos
    methods = sorted(set(r["method"] for r in per_combo_rows))

    def summarize(metric: str) -> Dict[str, Tuple[float, float]]:
        out = {}
        for m in methods:
            vals = [r[metric] for r in per_combo_rows if r["method"] == m and np.isfinite(r[metric])]
            if len(vals) == 0:
                out[m] = (float("nan"), float("nan"))
            else:
                arr = np.asarray(vals, dtype=float)
                out[m] = (float(arr.mean()), float(arr.std(ddof=0)))
        return out

    theta_rmse_stats = summarize("theta_rmse")
    y_rmse_stats = summarize("y_rmse")
    y_crps_stats = summarize("y_crps")

    print("\nRecomputed sudden metrics (y_crps uses fallback from others[*].crps_sim when needed)")
    print("=" * 100)
    print(f"{'Method':<35} | {'theta_rmse':>12} | {'y_rmse':>10} | {'y_crps':>10} | {'fallback%':>10}")
    print("-" * 100)
    for m in methods:
        # fallback ratio
        vals = [r for r in per_combo_rows if r["method"] == m]
        fallback_ratio = 0.0
        if len(vals) > 0:
            fallback_ratio = float(np.mean([bool(r["used_fallback"]) for r in vals]))
        tmean, tstd = theta_rmse_stats[m]
        ymean, ystd = y_rmse_stats[m]
        cmean, cstd = y_crps_stats[m]
        print(
            f"{m:<35} | {tmean:.6f} ± {tstd:.6f} | {ymean:.6f} ± {ystd:.6f} | "
            f"{cmean:.6f} ± {cstd:.6f} | {fallback_ratio*100:.1f}%"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()

