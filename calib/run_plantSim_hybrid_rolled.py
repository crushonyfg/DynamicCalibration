"""
Run PlantSim only for "Ours-hybrid-rolled" (vs Ground Truth only).

This is a lightweight companion to `run_plantSim_comparison.py`:
- no DA/BC curves
- only Ours-hybrid-rolled vs GT

Metrics (per mode):
- theta_rmse, theta_crps (MAE surrogate)
- y_rmse, y_crps   (MAE surrogate)

python -m calib.run_plantSim_hybrid_rolled --csv physical_data.csv --modes 0 1 2 --n_particles 1024 --out_dir "figs/plantSim_hybrid_rolled"
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from .configs import CalibrationConfig
from .online_calibrator import OnlineBayesCalibrator
from .v3_utils import StreamClass, JumpPlan
from .run_plantSim_v3_std import (
    init_pipeline,
    batch_X_base_to_s,
    batch_y_to_s,
    prior_sampler,
)


def _rmse(err: np.ndarray) -> float:
    arr = np.asarray(err, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr ** 2)))


def _crps_simple(mu: np.ndarray, y: np.ndarray) -> float:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if mu.size == 0:
        return float("nan")
    return float(np.mean(np.abs(mu - y)))


def _iter_batches(stream: StreamClass, batch_size: int):
    while True:
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break


def _make_stream(mode: int, data_dir: str, csv_path: str):
    """Keep consistent with `run_plantSim_comparison.py` stream creation."""
    if mode == 2:
        jp = JumpPlan(
            max_jumps=5,
            min_gap_theta=500.0,
            min_interval=180,
            max_interval=320,
            min_jump_span=40,
            seed=7,
        )
        return StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
    return StreamClass(mode, folder=data_dir, csv_path=csv_path)


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="PlantSim: Ours-hybrid-rolled only (vs GT).",
    )
    parser.add_argument("--out_dir", type=str, default="figs/plantSim_hybrid_rolled")
    parser.add_argument("--data_dir", type=str, default=None, help="PhysicalData_v3 directory")
    parser.add_argument("--csv", type=str, default=None, help="Aggregated physical-data CSV path")
    parser.add_argument("--npz", type=str, default=None, help="Computer-data NPZ path")
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_particles", type=int, default=1024)
    parser.add_argument("--out_plots", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_tf, nn_model, emu, a_s, b_s = init_pipeline(npz_path=args.npz)
    sigma_obs_s = float(gt_tf.y_scaler.scale_[0])
    print(f"sigma_obs_s = {sigma_obs_s:.4f}")

    MODE_NAMES = {
        0: "Gradual (mode 0)",
        1: "Sudden Jump (mode 1)",
        2: "Mixed (mode 2)",
    }

    metrics_rows: List[Dict[str, Any]] = []

    # Optional: one combined theta figure
    if args.out_plots:
        fig, axes = plt.subplots(
            1,
            len(args.modes),
            figsize=(6 * len(args.modes), 5),
            squeeze=False,
        )
        axes = axes.ravel()

    for mi, mode in enumerate(args.modes):
        mode_label = MODE_NAMES.get(mode, f"mode {mode}")
        print(f"\n=== Mode {mode}: {mode_label} ===")

        stream = _make_stream(mode, args.data_dir, args.csv)

        cfg = CalibrationConfig()
        cfg.bocpd.bocpd_mode = "restart"
        cfg.bocpd.use_restart = True

        # half-discrepancy style: PF likelihood uses no discrepancy mean; BOCPD discrepancy is used for prediction/state
        cfg.model.use_discrepancy = False
        cfg.model.bocpd_use_discrepancy = True

        # hybrid restart implementation
        cfg.bocpd.restart_impl = "hybrid_260319"
        cfg.bocpd.hybrid_partial_restart = True
        cfg.bocpd.hybrid_pf_sigma_mode = "rolled"
        cfg.bocpd.hybrid_sigma_ema_beta = 0.98
        cfg.bocpd.hybrid_sigma_min = 1e-4
        cfg.bocpd.hybrid_sigma_max = 10.0
        cfg.bocpd.hybrid_tau_delta = 0.05
        cfg.bocpd.hybrid_tau_theta = 0.05
        cfg.bocpd.hybrid_tau_full = 0.05
        cfg.bocpd.hybrid_delta_share_rho = 0.75
        cfg.bocpd.hybrid_sigma_delta_alpha = 1.0

        calib = OnlineBayesCalibrator(cfg, emu, prior_sampler)

        ours_theta_raw: List[float] = []
        gt_theta_raw: List[float] = []

        ours_y_pred_raw_batches: List[np.ndarray] = []
        ours_y_true_raw_batches: List[np.ndarray] = []

        for Xb, yb, thb in tqdm(
            _iter_batches(stream, args.batch_size),
            desc=f"mode {mode_label}",
            unit="batch",
        ):
            newX = batch_X_base_to_s(gt_tf, Xb)
            newY = batch_y_to_s(gt_tf, yb)

            calib.step_batch(newX, newY, verbose=False)

            mean_theta_s, _, _, _ = calib._aggregate_particles(0.9)
            mean_theta_raw = gt_tf.theta_s_to_raw(float(mean_theta_s[0]))
            ours_theta_raw.append(float(mean_theta_raw.item()))
            gt_theta_raw.append(float(np.mean(thb)))

            pred = calib.predict_batch(newX)
            mu_s = pred.get("mu_sim", pred.get("mu"))
            mu_s = mu_s.detach().cpu().numpy().reshape(-1)
            mu_raw = gt_tf.y_s_to_raw(mu_s).reshape(-1)
            ours_y_pred_raw_batches.append(np.asarray(mu_raw, dtype=float).reshape(-1))
            ours_y_true_raw_batches.append(np.asarray(yb, dtype=float).reshape(-1))

        ours_theta_raw = np.asarray(ours_theta_raw, dtype=float)
        gt_theta_raw = np.asarray(gt_theta_raw, dtype=float)

        theta_rmse = _rmse(ours_theta_raw - gt_theta_raw)
        theta_crps = _crps_simple(ours_theta_raw, gt_theta_raw)

        y_pred = np.concatenate(ours_y_pred_raw_batches, axis=0) if len(ours_y_pred_raw_batches) > 0 else np.asarray([], dtype=float)
        y_true = np.concatenate(ours_y_true_raw_batches, axis=0) if len(ours_y_true_raw_batches) > 0 else np.asarray([], dtype=float)

        y_rmse = _rmse(y_pred - y_true)
        y_crps = _crps_simple(y_pred, y_true)

        metrics_rows.append(
            dict(
                mode=mode,
                mode_name=mode_label,
                method="Ours-hybrid-rolled",
                theta_rmse=theta_rmse,
                theta_crps=theta_crps,
                y_rmse=y_rmse,
                y_crps=y_crps,
                n_steps=int(ours_theta_raw.shape[0]),
            )
        )

        if args.out_plots:
            ax = axes[mi]
            bidx = np.arange(len(gt_theta_raw))
            ax.plot(bidx, gt_theta_raw, "k--", lw=2, label="Ground Truth", zorder=5)
            ax.plot(bidx, ours_theta_raw, "b-", lw=2, label="Ours-hybrid-rolled", zorder=6)
            ax.set_title(mode_label, fontsize=12)
            ax.set_xlabel("Batch Index", fontsize=10)
            # Matplotlib mathtext: use single backslash for \theta
            ax.set_ylabel(r"$\theta$ (minutes)", fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=9)

    if args.out_plots:
        fig.suptitle("PlantSim: Ours-hybrid-rolled vs GT", fontsize=14)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "plantSim_hybrid_rolled_theta_vs_gt.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {fig_path}")

    df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"[Saved] {metrics_csv}")

    # Print quick summary
    print("\nSummary:")
    for row in metrics_rows:
        print(
            f"  mode={row['mode']} | theta_rmse={row['theta_rmse']:.4f} theta_crps={row['theta_crps']:.4f} | "
            f"y_rmse={row['y_rmse']:.4f} y_crps={row['y_crps']:.4f}"
        )


if __name__ == "__main__":
    main()

