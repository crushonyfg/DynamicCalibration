"""
RCAM 6D experiment: only run hybrid-rolled variant.

Variant:
  - half-discrepancy BOCPD setting (PF no discrepancy mean, BOCPD uses discrepancy)
  - restart implementation = hybrid_260319
  - PF sigma mode = rolled

Outputs:
  - theta tracking figure
  - y tracking figure
  - RMSE / CRPS-like metrics (absolute + normalized)

python -m calib.run_rcam6d_hybrid_rolled --normalize --max_steps 1000 --num_particles 1024
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from .configs import CalibrationConfig, BOCPDConfig, ModelConfig, PFConfig
from .online_calibrator import OnlineBayesCalibrator
from .run_rcam6d_bocpd_pf import (
    build_rcam6d_stream,
    make_rcam6d_emulator,
    make_prior_sampler_state,
    f_nav_disc,
)


def run_rcam6d_hybrid_rolled(
    data_csv: str,
    seed: int = 0,
    use_cuda: bool = False,
    num_particles: int = 256,
    dt: float = 0.1,
    plot_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    normalize: bool = False,
    bias_scale: float = 3.0,
    bias_rw_std: float = 0.1,
) -> Dict[str, Any]:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    Va_seq, y_seq, y_seq_physical, x0, t_seq, scaler = build_rcam6d_stream(data_csv, normalize=normalize)
    T = Va_seq.shape[0]
    if max_steps is not None and max_steps < T:
        T = max_steps
        print(f"[RCAM6D-hybrid-rolled] Limiting to {T} steps")

    model_cfg = ModelConfig()
    model_cfg.rho = 1.0
    model_cfg.sigma_eps = 0.5 if not normalize else 0.2
    model_cfg.emulator_type = "deterministic"
    model_cfg.device = device
    model_cfg.dtype = dtype
    model_cfg.use_discrepancy = False
    setattr(model_cfg, "bocpd_use_discrepancy", True)

    pf_cfg = PFConfig()
    pf_cfg.num_particles = num_particles
    pf_cfg.resample_ess_ratio = 0.5
    pf_cfg.move_strategy = "none"

    bocpd_cfg = BOCPDConfig()
    bocpd_cfg.hazard_lambda = 600.0
    bocpd_cfg.hazard_type = "geometric"
    bocpd_cfg.bocpd_mode = "restart"
    bocpd_cfg.max_experts = 5
    bocpd_cfg.use_restart = True
    bocpd_cfg.restart_margin = 1.0
    bocpd_cfg.restart_cooldown = 20
    bocpd_cfg.restart_criteria = "rank_change"
    bocpd_cfg.restart_impl = "hybrid_260319"
    bocpd_cfg.hybrid_partial_restart = True
    bocpd_cfg.hybrid_pf_sigma_mode = "rolled"
    bocpd_cfg.hybrid_sigma_ema_beta = 0.98
    bocpd_cfg.hybrid_sigma_min = 1e-4
    bocpd_cfg.hybrid_sigma_max = 10.0
    bocpd_cfg.hybrid_tau_delta = 0.05
    bocpd_cfg.hybrid_tau_theta = 0.05
    bocpd_cfg.hybrid_tau_full = 0.05
    bocpd_cfg.hybrid_delta_share_rho = 0.75

    cfg = CalibrationConfig(model=model_cfg, pf=pf_cfg, bocpd=bocpd_cfg)
    emulator = make_rcam6d_emulator(scaler=scaler)
    prior_sampler = make_prior_sampler_state(x0, bias_scale=bias_scale)

    restart_times: List[float] = []
    current_step = [0]

    def on_restart_callback(t_now, r_new, s_star, anchor_rl, p_anchor, best_other):
        restart_times.append(t_seq[current_step[0]] if current_step[0] < len(t_seq) else t_seq[-1])

    calibrator = OnlineBayesCalibrator(
        calib_cfg=cfg,
        emulator=emulator,
        prior_sampler=prior_sampler,
        on_restart=on_restart_callback,
        notify_on_restart=True,
    )

    X_all = torch.from_numpy(Va_seq).to(device=device, dtype=dtype)
    Y_all = torch.from_numpy(y_seq).to(device=device, dtype=dtype)

    theta_est_list, theta_true_list = [], []
    y_pred_list, y_obs_list = [], []
    theta_true_seq = y_seq_physical[:, 3:6] - Va_seq

    from tqdm import tqdm
    for k in tqdm(range(T)):
        current_step[0] = k
        xk = X_all[k: k + 1]
        yk = Y_all[k: k + 1]

        for e in calibrator.bocpd.experts:
            ps = e.pf.particles
            theta_state = ps.theta.detach().cpu().numpy()
            N_particles = theta_state.shape[0]
            theta_next = []
            for n in range(N_particles):
                s_next = f_nav_disc(theta_state[n], Va_seq[k], dt)
                theta_next.append(s_next)
            theta_next = np.stack(theta_next, axis=0)
            theta_next[:, 3:6] += np.random.normal(0.0, bias_rw_std, size=(N_particles, 3))
            ps.theta = torch.from_numpy(theta_next).to(device=device, dtype=dtype)

        if len(calibrator.bocpd.experts) > 0:
            pred = calibrator.predict_batch(xk)
            mu_np = pred["mu"].detach().cpu().numpy()
            if mu_np.ndim == 1:
                mu_np = mu_np[None, :]
            y_pred_list.append(mu_np)
        else:
            y_pred_list.append(np.full((1, 6), np.nan))
        y_obs_list.append(yk.detach().cpu().numpy())

        calibrator.step_batch(xk, yk, verbose=False)
        mean_theta, _ = calibrator._aggregate_particles(quantile=None)
        if mean_theta is not None:
            theta_est_list.append(mean_theta.detach().cpu().numpy()[3:6].copy())
        else:
            theta_est_list.append(np.full(3, np.nan))
        theta_true_list.append(theta_true_seq[k, :].copy())

    theta_est_arr = np.stack(theta_est_list, axis=0)
    theta_true_arr = np.stack(theta_true_list, axis=0)
    y_pred_flat = np.concatenate(y_pred_list, axis=0)
    y_obs_flat = np.concatenate(y_obs_list, axis=0)
    t_flat = t_seq[:T]

    if scaler is not None:
        y_pred_phys = scaler.inverse_transform(y_pred_flat)
        y_obs_phys = scaler.inverse_transform(y_obs_flat)
    else:
        y_pred_phys, y_obs_phys = y_pred_flat, y_obs_flat

    theta_std = np.where(np.std(theta_true_arr, axis=0) < 1e-8, 1.0, np.std(theta_true_arr, axis=0))
    theta_rmse_dim = np.sqrt(np.mean((theta_est_arr - theta_true_arr) ** 2, axis=0))
    theta_crps_dim = np.mean(np.abs(theta_est_arr - theta_true_arr), axis=0)
    theta_nrmse_dim = theta_rmse_dim / theta_std
    theta_ncrps_dim = theta_crps_dim / theta_std

    theta_rmse = float(np.sqrt(np.mean(np.sum((theta_est_arr - theta_true_arr) ** 2, axis=1))))
    theta_crps = float(np.mean(np.linalg.norm(theta_est_arr - theta_true_arr, ord=1, axis=1)))
    theta_nrmse = float(np.mean(theta_nrmse_dim))
    theta_ncrps = float(np.mean(theta_ncrps_dim))

    valid = ~np.isnan(y_pred_phys[:, 0])
    if np.any(valid):
        ypv, yov = y_pred_phys[valid], y_obs_phys[valid]
        y_std = np.where(np.std(yov, axis=0) < 1e-8, 1.0, np.std(yov, axis=0))
        y_rmse_dim = np.sqrt(np.mean((ypv - yov) ** 2, axis=0))
        y_crps_dim = np.mean(np.abs(ypv - yov), axis=0)
        y_nrmse_dim = y_rmse_dim / y_std
        y_ncrps_dim = y_crps_dim / y_std
        y_rmse = float(np.sqrt(np.mean(np.sum((ypv - yov) ** 2, axis=1))))
        y_crps = float(np.mean(np.linalg.norm(ypv - yov, ord=1, axis=1)))
        y_nrmse = float(np.mean(y_nrmse_dim))
        y_ncrps = float(np.mean(y_ncrps_dim))
    else:
        y_rmse = y_crps = y_nrmse = y_ncrps = float("nan")
        y_rmse_dim = y_crps_dim = y_nrmse_dim = y_ncrps_dim = np.full(6, np.nan)

    fig_theta_path, fig_y_path = None, None
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig1, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["bN", "bE", "bD"]
        for j in range(3):
            axes[j].plot(t_flat, theta_true_arr[:, j], "k-", alpha=0.8, label="theta_true")
            axes[j].plot(t_flat, theta_est_arr[:, j], "b-", alpha=0.8, label="theta_est")
            for rt in restart_times:
                axes[j].axvline(rt, color="r", linestyle="--", alpha=0.4, lw=0.8)
            axes[j].set_ylabel(labels[j])
            axes[j].grid(True, alpha=0.3)
            axes[j].legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig1.suptitle("RCAM hybrid-rolled: theta tracking")
        fig1.tight_layout()
        fig_theta_path = os.path.join(plot_dir, "rcam6d_hybrid_rolled_theta_tracking.png")
        fig1.savefig(fig_theta_path, dpi=150)
        plt.close(fig1)

        fig2, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
        y_labels = ["lat", "lon", "h", "V_N", "V_E", "V_D"]
        for j in range(6):
            axes[j].plot(t_flat, y_obs_phys[:, j], "k-", alpha=0.6, label="y_obs")
            axes[j].plot(t_flat, y_pred_phys[:, j], "b-", alpha=0.7, label="y_pred")
            for rt in restart_times:
                axes[j].axvline(rt, color="r", linestyle="--", alpha=0.4, lw=0.8)
            axes[j].set_ylabel(y_labels[j])
            axes[j].grid(True, alpha=0.3)
            axes[j].legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig2.suptitle("RCAM hybrid-rolled: y tracking")
        fig2.tight_layout()
        fig_y_path = os.path.join(plot_dir, "rcam6d_hybrid_rolled_y_tracking.png")
        fig2.savefig(fig_y_path, dpi=150)
        plt.close(fig2)

    return {
        "method": "R-BOCPD-PF-halfdiscrepancy-hybrid-rolled",
        "theta_est": theta_est_arr,
        "theta_true": theta_true_arr,
        "y_pred": y_pred_phys,
        "y_obs": y_obs_phys,
        "t": t_flat,
        "num_restarts": len(restart_times),
        "restart_times": restart_times,
        "theta_rmse_overall": theta_rmse,
        "theta_crps_overall": theta_crps,
        "theta_nrmse_overall": theta_nrmse,
        "theta_ncrps_overall": theta_ncrps,
        "theta_rmse_per_dim": theta_rmse_dim,
        "theta_crps_per_dim": theta_crps_dim,
        "y_rmse_overall": y_rmse,
        "y_crps_overall": y_crps,
        "y_nrmse_overall": y_nrmse,
        "y_ncrps_overall": y_ncrps,
        "y_rmse_per_dim": y_rmse_dim,
        "y_crps_per_dim": y_crps_dim,
        "fig_theta_path": fig_theta_path,
        "fig_y_path": fig_y_path,
    }


def main():
    parser = argparse.ArgumentParser(description="RCAM 6D: hybrid-rolled only.")
    parser.add_argument("--data_csv", type=str, default="C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data_windjump.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=256)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--plot_dir", type=str, default="figs/rcam6d_hybrid_rolled")
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
        plot_dir=args.plot_dir,
        max_steps=args.max_steps,
        normalize=args.normalize,
        bias_scale=args.bias_scale,
        bias_rw_std=args.bias_rw_std,
    )

    print("\n=== RCAM6D hybrid-rolled results ===")
    print(f"num_restarts        : {out['num_restarts']}")
    print(f"theta_nrmse_overall : {out['theta_nrmse_overall']:.4f}")
    print(f"theta_ncrps_overall : {out['theta_ncrps_overall']:.4f}")
    print(f"y_nrmse_overall     : {out['y_nrmse_overall']:.4f}")
    print(f"y_ncrps_overall     : {out['y_ncrps_overall']:.4f}")
    if out.get("fig_theta_path"):
        print(f"theta tracking fig  : {out['fig_theta_path']}")
    if out.get("fig_y_path"):
        print(f"y tracking fig      : {out['fig_y_path']}")


if __name__ == "__main__":
    main()

