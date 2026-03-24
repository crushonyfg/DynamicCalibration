"""
Print aggregated y-scale statistics for RCAM hybrid-rolled run.

Usage:
  python -m calib.print_rcam6d_std_y_agg --normalize --max_steps 200
"""

from __future__ import annotations

import argparse
import numpy as np

from .run_rcam6d_hybrid_rolled import run_rcam6d_hybrid_rolled


def main():
    parser = argparse.ArgumentParser(description="Report std_y_agg for RCAM hybrid-rolled.")
    parser.add_argument("--data_csv", type=str, default="C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data_windjump.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=256)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--bias_scale", type=float, default=3.0)
    parser.add_argument("--bias_rw_std", type=float, default=0.1)
    args = parser.parse_args()

    out = run_rcam6d_hybrid_rolled(
        data_csv=args.data_csv,
        seed=args.seed,
        use_cuda=args.use_cuda,
        num_particles=args.num_particles,
        dt=args.dt,
        plot_dir=None,  # only report stats
        max_steps=args.max_steps,
        normalize=args.normalize,
        bias_scale=args.bias_scale,
        bias_rw_std=args.bias_rw_std,
    )

    y_obs = np.asarray(out["y_obs"], dtype=float)  # physical space
    y_std_per_dim = np.std(y_obs, axis=0)
    y_std_per_dim = np.where(y_std_per_dim < 1e-8, 1.0, y_std_per_dim)
    std_y_agg = float(np.sqrt(np.sum(y_std_per_dim ** 2)))

    y_rmse_overall = float(out["y_rmse_overall"])
    ratio = y_rmse_overall / std_y_agg if std_y_agg > 0 else float("nan")

    print("=== RCAM6D std_y report ===")
    print(f"y_std_per_dim           : {y_std_per_dim}")
    print(f"std_y_agg               : {std_y_agg:.6f}")
    print(f"y_rmse_overall          : {y_rmse_overall:.6f}")
    print(f"y_rmse_overall/std_y_agg: {ratio:.6f}")
    print(f"y_nrmse_overall(script) : {float(out['y_nrmse_overall']):.6f}")
    print("")
    print("Note: ratio above is aggregated-L2 normalization;")
    print("      y_nrmse_overall is mean of per-dimension normalized RMSE.")


if __name__ == "__main__":
    main()

