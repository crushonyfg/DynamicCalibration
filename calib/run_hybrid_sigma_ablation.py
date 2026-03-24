"""
Hybrid PF-likelihood sigma-mode ablation on both synthetic tasks.

Runs:
1) slope task    (run_synthetic_slope_deltaCmp.run_one_slope)
2) sudden task   (run_synthetic_suddenCmp_tryThm.run_one_sudden)

Compares hybrid modes:
  - fixed    : default fixed sigma_eps
  - var_only : sigma^2 = sigma_eps^2 + alpha * E[var_delta]
  - rolled   : sigma^2 = EMA(residual^2)

python -m calib.run_hybrid_sigma_ablation --debug --out_dir figs/hybrid_sigma_ablation_debug
python -m calib.run_hybrid_sigma_ablation --out_dir figs/hybrid_sigma_ablation
"""

import os
import itertools
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .run_synthetic_slope_deltaCmp import run_one_slope, build_phi2_from_theta_star
from .run_synthetic_suddenCmp_tryThm import run_one_sudden


def _build_methods(sigma_modes):
    methods = {}
    for m in sigma_modes:
        methods[f"R-BOCPD-PF-halfdiscrepancy-hybrid-{m}"] = dict(
            type="bocpd",
            mode="restart",
            use_discrepancy=False,
            bocpd_use_discrepancy=True,
            restart_impl="hybrid_260319",
            hybrid_partial_restart=True,
            hybrid_tau_delta=0.05,
            hybrid_tau_theta=0.05,
            hybrid_tau_full=0.05,
            hybrid_delta_share_rho=0.75,
            hybrid_pf_sigma_mode=m,
            hybrid_sigma_delta_alpha=1.0,
            hybrid_sigma_ema_beta=0.98,
            hybrid_sigma_min=1e-4,
            hybrid_sigma_max=10.0,
        )
    return methods


def _extract_metrics(result_dict, task_name, config_dict):
    rows = []
    for method_name, data in result_dict.items():
        theta = np.asarray(data["theta"], dtype=float)
        theta_oracle = np.asarray(data["theta_oracle"], dtype=float)
        theta_rmse = float(np.sqrt(np.mean((theta - theta_oracle) ** 2)))
        y_rmse = float(np.mean(np.asarray(data.get("rmse", []), dtype=float))) if len(data.get("rmse", [])) > 0 else np.nan
        y_crps = float(np.mean(np.asarray(data.get("crps_hist", []), dtype=float))) if len(data.get("crps_hist", [])) > 0 else np.nan

        rm_hist = data.get("restart_mode_hist", [])
        n_steps = len(rm_hist)
        n_none = sum(1 for v in rm_hist if v == "none")
        n_delta = sum(1 for v in rm_hist if v == "delta_only")
        n_full = sum(1 for v in rm_hist if v not in ("none", "delta_only"))
        n_restart = n_delta + n_full
        delta_ratio_given_restart = (n_delta / n_restart) if n_restart > 0 else np.nan
        full_ratio_given_restart = (n_full / n_restart) if n_restart > 0 else np.nan

        row = dict(
            task=task_name,
            method=method_name,
            theta_rmse=theta_rmse,
            y_rmse=y_rmse,
            y_crps=y_crps,
            n_steps=n_steps,
            n_none=n_none,
            n_delta_only=n_delta,
            n_full=n_full,
            restart_rate=(n_restart / n_steps) if n_steps > 0 else np.nan,
            delta_ratio_given_restart=delta_ratio_given_restart,
            full_ratio_given_restart=full_ratio_given_restart,
        )
        row.update(config_dict)
        rows.append(row)
    return rows


def _save_theta_tracking_plot(res, oracle_hist, save_path, title):
    plt.figure(figsize=(10, 5))
    for method_name, data in res.items():
        theta = np.asarray(data.get("theta", []), dtype=float)
        if theta.size == 0:
            continue
        plt.plot(theta, label=method_name, alpha=0.9)
    if oracle_hist is not None:
        plt.plot(np.asarray(oracle_hist, dtype=float), "k--", lw=2, label="oracle theta*")
    plt.title(title)
    plt.xlabel("batch index")
    plt.ylabel("theta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/hybrid_sigma_ablation")
    parser.add_argument("--sigma_modes", nargs="+", default=["fixed", "var_only", "rolled"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    methods = _build_methods(args.sigma_modes)

    all_rows = []

    # ---------- slope task ----------
    if args.debug:
        slopes = [0.001]
        batch_sizes = [10]
        seeds = [456]
    else:
        slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
        batch_sizes = [20]
        seeds = [101, 202, 303, 404, 505]

    phi2_grid = np.linspace(3.0, 12.0, 300)
    theta_grid = np.linspace(0.0, 3.0, 600)
    phi2_of_theta, _ = build_phi2_from_theta_star(phi2_grid=phi2_grid, theta_grid=theta_grid)

    for slope, batch_size, seed in itertools.product(slopes, batch_sizes, seeds):
        res, phi_hist, oracle_hist = run_one_slope(
            slope=slope,
            methods=methods,
            total_T=600 if not args.debug else 200,
            batch_size=batch_size,
            seed=seed,
            phi2_of_theta=phi2_of_theta,
            mode=1,
        )
        all_rows.extend(
            _extract_metrics(
                res,
                task_name="slope",
                config_dict=dict(slope=slope, batch_size=batch_size, seed=seed),
            )
        )
        tag = f"slope_s{slope}_b{batch_size}_seed{seed}"
        torch.save(res, os.path.join(args.out_dir, f"{tag}.pt"))
        _save_theta_tracking_plot(
            res=res,
            oracle_hist=oracle_hist,
            save_path=os.path.join(args.out_dir, f"{tag}_theta.png"),
            title=f"Slope theta tracking (slope={slope}, batch={batch_size}, seed={seed})",
        )

    # ---------- sudden task ----------
    if args.debug:
        seg_lens = [120]
        magnitudes = [2.0]
        batch_sizes = [20]
        seeds = [456]
    else:
        seg_lens = [80, 120, 200]
        magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0]
        batch_sizes = [20]
        seeds = [101, 202, 303, 404, 505]

    for seg_len, delta_mag, batch_size, seed in itertools.product(seg_lens, magnitudes, batch_sizes, seeds):
        if seg_len % batch_size != 0:
            continue
        res, phi_hist, oracle_hist = run_one_sudden(
            seg_len_L=seg_len,
            delta_mag=delta_mag,
            methods=methods,
            batch_size=batch_size,
            seed=seed,
            noise_sd=0.2,
            phi_center=7.5,
            out_dir=args.out_dir,
        )
        all_rows.extend(
            _extract_metrics(
                res,
                task_name="sudden",
                config_dict=dict(seg_len_L=seg_len, delta_mag=delta_mag, batch_size=batch_size, seed=seed),
            )
        )
        tag = f"sudden_L{seg_len}_d{delta_mag}_b{batch_size}_seed{seed}"
        torch.save(res, os.path.join(args.out_dir, f"{tag}.pt"))
        _save_theta_tracking_plot(
            res=res,
            oracle_hist=oracle_hist,
            save_path=os.path.join(args.out_dir, f"{tag}_theta.png"),
            title=f"Sudden theta tracking (L={seg_len}, delta={delta_mag}, batch={batch_size}, seed={seed})",
        )

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out_dir, "hybrid_sigma_ablation_metrics.csv")
    xlsx_path = os.path.join(args.out_dir, "hybrid_sigma_ablation_metrics.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    summary = (
        df.groupby(["task", "method"], as_index=False)[
            ["theta_rmse", "y_rmse", "y_crps", "restart_rate", "delta_ratio_given_restart", "full_ratio_given_restart"]
        ]
        .mean()
        .sort_values(["task", "method"])
    )
    summary_path = os.path.join(args.out_dir, "hybrid_sigma_ablation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {xlsx_path}")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()

