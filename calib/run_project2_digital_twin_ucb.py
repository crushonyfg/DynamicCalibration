from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from calib.project2_common import (
        batch_X_base_to_s,
        batch_y_to_s,
        build_restart_bocpd_calibrator,
        theta_prior_sampler_factory,
        train_or_load_standardized_emulator,
    )
    from calib.project2_contextual_ucb import (
        DecisionSpace,
        DiscreteContextualUCBOptimizer,
        PlatformThetaSchedule,
        SinThetaSchedule,
    )
    from calib.project2_plantsim import EmulatorProxyRunner, PlantSimConfig, PlantSimulationRunner
else:
    from .project2_common import (
        batch_X_base_to_s,
        batch_y_to_s,
        build_restart_bocpd_calibrator,
        theta_prior_sampler_factory,
        train_or_load_standardized_emulator,
    )
    from .project2_contextual_ucb import (
        DecisionSpace,
        DiscreteContextualUCBOptimizer,
        PlatformThetaSchedule,
        SinThetaSchedule,
    )
    from .project2_plantsim import EmulatorProxyRunner, PlantSimConfig, PlantSimulationRunner


def make_nn_raw_scorer(decision_space, nn_std, gt):
    def scorer(X_decision: np.ndarray, theta_raw: float):
        X_model = decision_space.decision_to_model(X_decision)
        X_s = gt.X_base_to_s(X_model)
        theta_s = gt.theta_raw_to_s(np.full(len(X_s), theta_raw, dtype=np.float64)).reshape(-1, 1)
        X_full_s = np.concatenate([X_s, theta_s], axis=1)
        mu_s = nn_std.predict_y_s_from_Xfull_s(X_full_s)
        mu_raw = gt.y_s_to_raw(mu_s)
        sigma_raw = np.zeros_like(mu_raw)
        return mu_raw, sigma_raw

    return scorer


def log_msg(msg: str) -> None:
    print(msg, flush=True)


def fmt_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    return f"{seconds:.2f} s"


def save_plots(out_dir: Path, batch_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(batch_df["batch_idx"], batch_df["theta_true"], label="true theta")
    plt.plot(batch_df["batch_idx"], batch_df["theta_est"], label="estimated theta")
    plt.fill_between(
        batch_df["batch_idx"],
        batch_df["theta_est"] - batch_df["theta_est_std"],
        batch_df["theta_est"] + batch_df["theta_est_std"],
        alpha=0.2,
        label="est +/- 1 std",
    )
    plt.xlabel("batch")
    plt.ylabel("theta (minutes)")
    plt.title("Theta Tracking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "theta_tracking.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_df["batch_idx"], batch_df["cum_gap_mean"], label="cumulative mean gap")
    plt.plot(batch_df["batch_idx"], batch_df["cum_gap_best"], label="cumulative max gap")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("batch")
    plt.ylabel("cumulative reward gap")
    plt.title("Cumulative Improvement vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_gap.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_df["batch_idx"], batch_df["pred_selected_mean"], label="predicted batch mean")
    plt.plot(batch_df["batch_idx"], batch_df["y_selected_mean"], label="actual batch mean")
    plt.plot(batch_df["batch_idx"], batch_df["y_base"], label="baseline batch mean")
    plt.xlabel("batch")
    plt.ylabel("net revenue")
    plt.title("Reality Gap and Batch Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reality_gap.png", dpi=250)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_df["batch_idx"], batch_df["y_selected_best"], label="candidate max")
    plt.plot(batch_df["batch_idx"], batch_df["y_base_best"], label="baseline max")
    plt.xlabel("batch")
    plt.ylabel("net revenue")
    plt.title("Candidate Max vs Baseline Max")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "max_reward_comparison.png", dpi=250)
    plt.close()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/project2_ucb")
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"C:\Users\yxu59\files\winter2026\park\simulation\ComputerData_v3\factory_aggregated.npz",
    )
    parser.add_argument("--bundle_dir", type=str, default="figs/project2_ucb/bundles")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_batches", type=int, default=80)
    parser.add_argument("--candidate_top_k", type=int, default=5)
    parser.add_argument("--baseline_top_k", type=int, default=5)
    parser.add_argument("--hold_batches", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--num_particles", type=int, default=512)
    parser.add_argument("--random_pool_size", type=int, default=4096)
    parser.add_argument("--local_pool_size", type=int, default=1024)
    parser.add_argument("--eval_chunk_size", type=int, default=512)
    parser.add_argument("--n_replications", type=int, default=1)
    parser.add_argument("--backend", choices=["plantsim", "emulator"], default="plantsim")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\yxu59\files\winter2026\park\simulation\DBCsystem_v3.spp",
    )
    parser.add_argument("--prog_id", type=str, default="Tecnomatix.PlantSimulation.RemoteControl.24.4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jump_batch", type=int, default=-1)
    parser.add_argument("--jump_target_theta", type=float, default=None)
    parser.add_argument("--theta_schedule_mode", choices=["sin", "platform"], default="platform")
    parser.add_argument("--platform_levels", type=str, default="3,11.5,20,30")
    parser.add_argument("--platform_block_batches", type=int, default=20)
    parser.add_argument("--platform_noise_std", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    log_msg("[Init] loading or training digital twin emulator ...")
    init_t0 = time.perf_counter()
    gt, nn_std, emulator = train_or_load_standardized_emulator(
        data_path=args.data_path,
        bundle_dir=str(bundle_dir),
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
    )
    log_msg(f"[Init] emulator ready in {fmt_seconds(time.perf_counter() - init_t0)}")

    decision_space = DecisionSpace()
    prior_sampler = theta_prior_sampler_factory(gt, theta_bounds_raw=(1.0, 30.0))
    log_msg("[Init] building R-BOCPD-PF-nodiscrepancy calibrator ...")
    calib = build_restart_bocpd_calibrator(
        emulator=emulator,
        prior_sampler=prior_sampler,
        gt=gt,
        device=args.device,
        use_discrepancy=False,
        num_particles=args.num_particles,
    )

    optimizer = DiscreteContextualUCBOptimizer(
        decision_space=decision_space,
        top_k=args.candidate_top_k,
        beta=args.beta,
        random_pool_size=args.random_pool_size,
        local_pool_size=args.local_pool_size,
        eval_chunk_size=args.eval_chunk_size,
        seed=args.seed,
    )
    if args.theta_schedule_mode == "platform":
        platform_levels = tuple(float(x.strip()) for x in args.platform_levels.split(",") if x.strip())
        theta_schedule = PlatformThetaSchedule(
            levels=platform_levels,
            block_batches=args.platform_block_batches,
            noise_std=args.platform_noise_std,
            theta_min=0.1,
            theta_max=30.0,
            seed=args.seed,
        )
        log_msg(
            f"[Init] theta schedule = platform | levels={list(platform_levels)} | "
            f"block_batches={args.platform_block_batches} | noise_std={args.platform_noise_std}"
        )
    else:
        theta_schedule = SinThetaSchedule(
            num_batches=args.num_batches,
            hold_batches=args.hold_batches,
            jump_batch=(args.jump_batch if args.jump_batch >= 0 else None),
            jump_target=args.jump_target_theta,
            theta_min=1.0,
            theta_max=30.0,
        )
        log_msg(
            f"[Init] theta schedule = sin | hold_batches={args.hold_batches} | "
            f"jump_batch={(args.jump_batch if args.jump_batch >= 0 else None)} | "
            f"jump_target={args.jump_target_theta}"
        )

    log_msg("[Init] computing baseline X0 with theta=11.5 ...")
    baseline_theta = 11.5
    baseline_pick = optimizer.propose_with_nn(
        nn_std=nn_std,
        gt=gt,
        theta_raw=baseline_theta,
        include_full_grid=False,
    )
    baseline_top = {
        "X_decision": baseline_pick["X_decision"][: args.baseline_top_k].copy(),
        "mu_raw": baseline_pick["mu_raw"][: args.baseline_top_k].copy(),
        "sigma_raw": baseline_pick["sigma_raw"][: args.baseline_top_k].copy(),
        "ucb": baseline_pick["ucb"][: args.baseline_top_k].copy(),
    }
    log_msg(
        f"[Init] baseline top{args.baseline_top_k} ready | "
        f"best baseline X={baseline_top['X_decision'][0].tolist()}"
    )

    log_msg("[Init] computing initial contextual UCB candidate batch ...")
    current_pick = optimizer.propose_with_nn(
        nn_std=nn_std,
        gt=gt,
        theta_raw=baseline_theta,
        previous_top=None,
        include_full_grid=False,
    )

    scorer = make_nn_raw_scorer(decision_space, nn_std, gt)
    if args.backend == "plantsim":
        runner_cm = PlantSimulationRunner(
            PlantSimConfig(model_path=args.model_path, prog_id=args.prog_id, visible=False)
        )
    else:
        runner_cm = EmulatorProxyRunner(scorer)

    batch_rows: List[Dict] = []
    candidate_rows: List[Dict] = []

    with runner_cm as runner:
        for batch_idx in range(args.num_batches):
            batch_t0 = time.perf_counter()
            theta_true = theta_schedule.theta_at(batch_idx)
            log_msg("")
            log_msg(f"[Batch {batch_idx + 1}/{args.num_batches}] theta_true={theta_true:.4f} min")

            sim_t0 = time.perf_counter()
            log_msg(
                f"[Batch {batch_idx + 1}] running physical evaluations for "
                f"{len(current_pick['X_decision'])} candidate points ..."
            )
            cand_eval = runner.evaluate_batch(
                current_pick["X_decision"],
                customer_lbd_min=theta_true,
                n_replications=args.n_replications,
            )
            log_msg(
                f"[Batch {batch_idx + 1}] running physical evaluations for "
                f"{len(baseline_top['X_decision'])} baseline points ..."
            )
            base_eval = runner.evaluate_batch(
                baseline_top["X_decision"],
                customer_lbd_min=theta_true,
                n_replications=args.n_replications,
            )
            sim_elapsed = time.perf_counter() - sim_t0
            log_msg(f"[Batch {batch_idx + 1}] PlantSimulation/backend finished in {fmt_seconds(sim_elapsed)}")

            calib_t0 = time.perf_counter()
            log_msg(
                f"[Batch {batch_idx + 1}] running continuous calibration on "
                f"{len(cand_eval['X_decision']) + len(base_eval['X_decision'])} evaluated points ..."
            )
            calib_X_decision = np.vstack([cand_eval["X_decision"], base_eval["X_decision"]])
            calib_y = np.concatenate([cand_eval["y_mean"], base_eval["y_mean"]])
            X_model = decision_space.decision_to_model(calib_X_decision)
            newX = batch_X_base_to_s(gt, X_model)
            newY = batch_y_to_s(gt, calib_y)
            rec = calib.step_batch(newX, newY, verbose=False)

            theta_mean_s, theta_var_s, _, _ = calib._aggregate_particles(0.9)
            theta_est = float(gt.theta_s_to_raw(np.array([theta_mean_s.item()]))[0])
            theta_var_raw = float(theta_var_s[0, 0].item()) * (gt.theta_sd ** 2)
            theta_std_raw = float(np.sqrt(max(theta_var_raw, 0.0)))
            calib_elapsed = time.perf_counter() - calib_t0
            log_msg(
                f"[Batch {batch_idx + 1}] calibration finished in {fmt_seconds(calib_elapsed)} | "
                f"theta_est={theta_est:.4f} +/- {theta_std_raw:.4f} | restart={bool(rec['did_restart'])}"
            )
            log_msg(
                f"[Batch {batch_idx + 1}] theta tracking | "
                f"theta_true={theta_true:.4f} | theta_est={theta_est:.4f} | "
                f"abs_error={abs(theta_est - theta_true):.4f}"
            )

            pred_selected_mean = float(np.mean(current_pick["mu_raw"]))
            pred_selected_best = float(np.max(current_pick["mu_raw"]))
            pred_base_mean = float(np.mean(baseline_top["mu_raw"]))
            pred_base_best = float(np.max(baseline_top["mu_raw"]))
            y_selected_mean = float(np.mean(cand_eval["y_mean"]))
            y_selected_best = float(np.max(cand_eval["y_mean"]))
            y_base = float(np.mean(base_eval["y_mean"]))
            y_base_best = float(np.max(base_eval["y_mean"]))

            batch_rows.append(
                {
                    "batch_idx": batch_idx,
                    "theta_true": theta_true,
                    "theta_est": theta_est,
                    "theta_est_std": theta_std_raw,
                    "did_restart": bool(rec["did_restart"]),
                    "pred_selected_mean": pred_selected_mean,
                    "pred_selected_best": pred_selected_best,
                    "pred_base_mean": pred_base_mean,
                    "pred_base_best": pred_base_best,
                    "y_selected_mean": y_selected_mean,
                    "y_selected_best": y_selected_best,
                    "y_base": y_base,
                    "y_base_best": y_base_best,
                    "reality_gap_mean": y_selected_mean - pred_selected_mean,
                    "reality_gap_best": y_selected_best - pred_selected_best,
                }
            )

            for rank_idx, (x_decision, pred_mu, pred_sigma, actual_y) in enumerate(
                zip(
                    current_pick["X_decision"],
                    current_pick["mu_raw"],
                    current_pick["sigma_raw"],
                    cand_eval["y_mean"],
                ),
                start=1,
            ):
                candidate_rows.append(
                    {
                        "batch_idx": batch_idx,
                        "group": "candidate",
                        "rank": rank_idx,
                        "Q": int(x_decision[0]),
                        "R": int(x_decision[1]),
                        "W": int(x_decision[2]),
                        "M1": int(x_decision[3]),
                        "M2": int(x_decision[4]),
                        "theta_true": theta_true,
                        "theta_est": theta_est,
                        "pred_mu": float(pred_mu),
                        "pred_sigma": float(pred_sigma),
                        "actual_y": float(actual_y),
                    }
                )

            for rank_idx, (x_decision, pred_mu, pred_sigma, actual_y) in enumerate(
                zip(
                    baseline_top["X_decision"],
                    baseline_top["mu_raw"],
                    baseline_top["sigma_raw"],
                    base_eval["y_mean"],
                ),
                start=1,
            ):
                candidate_rows.append(
                    {
                        "batch_idx": batch_idx,
                        "group": "baseline",
                        "rank": rank_idx,
                        "Q": int(x_decision[0]),
                        "R": int(x_decision[1]),
                        "W": int(x_decision[2]),
                        "M1": int(x_decision[3]),
                        "M2": int(x_decision[4]),
                        "theta_true": theta_true,
                        "theta_est": theta_est,
                        "pred_mu": float(pred_mu),
                        "pred_sigma": float(pred_sigma),
                        "actual_y": float(actual_y),
                    }
                )

            ucb_t0 = time.perf_counter()
            log_msg(
                f"[Batch {batch_idx + 1}] computing next contextual UCB top{args.candidate_top_k} candidate set ..."
            )
            current_pick = optimizer.propose_with_calibrator(
                calib=calib,
                nn_std=nn_std,
                gt=gt,
                context_theta_raw=theta_est,
                previous_top=current_pick["X_decision"],
                include_full_grid=False,
            )
            ucb_elapsed = time.perf_counter() - ucb_t0
            log_msg(
                f"[Batch {batch_idx + 1}] UCB finished in {fmt_seconds(ucb_elapsed)} | "
                f"best_next_X={current_pick['X_decision'][0].tolist()} | "
                f"best_next_ucb={float(current_pick['ucb'][0]):.4f}"
            )
            log_msg(
                f"[Batch {batch_idx + 1}] summary | "
                f"candidate_top{args.candidate_top_k}_mean={y_selected_mean:.4f} | "
                f"baseline_top{args.baseline_top_k}_mean={y_base:.4f} | "
                f"mean_gap={y_selected_mean - y_base:.4f} | "
                f"candidate_max={y_selected_best:.4f} | baseline_max={y_base_best:.4f} | "
                f"max_gap={y_selected_best - y_base_best:.4f} | total={fmt_seconds(time.perf_counter() - batch_t0)}"
            )

    batch_df = pd.DataFrame(batch_rows)
    batch_df["cum_gap_mean"] = (batch_df["y_selected_mean"] - batch_df["y_base"]).cumsum()
    batch_df["cum_gap_best"] = (batch_df["y_selected_best"] - batch_df["y_base_best"]).cumsum()
    batch_df["theta_abs_error"] = np.abs(batch_df["theta_est"] - batch_df["theta_true"])
    cand_df = pd.DataFrame(candidate_rows)

    log_msg("[Finalize] saving csv/json/plots ...")
    batch_df.to_csv(out_dir / "project2_batch_summary.csv", index=False)
    cand_df.to_csv(out_dir / "project2_candidate_history.csv", index=False)
    save_plots(out_dir, batch_df)

    summary = {
        "backend": args.backend,
        "baseline_top_q_r_w_m1_m2": baseline_top["X_decision"].tolist(),
        "num_batches": int(args.num_batches),
        "candidate_top_k": int(args.candidate_top_k),
        "baseline_top_k": int(args.baseline_top_k),
        "calibration_points_per_batch": int(args.candidate_top_k + args.baseline_top_k),
        "hold_batches": int(args.hold_batches),
        "theta_schedule_mode": args.theta_schedule_mode,
        "platform_levels": args.platform_levels,
        "platform_block_batches": int(args.platform_block_batches),
        "platform_noise_std": float(args.platform_noise_std),
        "jump_batch": (int(args.jump_batch) if args.jump_batch >= 0 else None),
        "jump_target_theta": args.jump_target_theta,
        "beta": float(args.beta),
        "num_particles": int(args.num_particles),
        "cum_gap_mean_final": float(batch_df["cum_gap_mean"].iloc[-1]),
        "cum_gap_best_final": float(batch_df["cum_gap_best"].iloc[-1]),
        "avg_theta_abs_error": float(batch_df["theta_abs_error"].mean()),
        "avg_reality_gap_mean": float(batch_df["reality_gap_mean"].mean()),
        "avg_reality_gap_best": float(batch_df["reality_gap_best"].mean()),
        "avg_candidate_max_minus_baseline_max": float((batch_df["y_selected_best"] - batch_df["y_base_best"]).mean()),
        "final_candidate_max_minus_baseline_max": float(batch_df["y_selected_best"].iloc[-1] - batch_df["y_base_best"].iloc[-1]),
        "restart_count": int(batch_df["did_restart"].sum()),
    }
    with open(out_dir / "project2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_msg("[Done] final summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

# python -m calib.run_project2_digital_twin_ucb --backend plantsim --device cpu --out_dir figs/project2_ucb
# python -m calib.run_project2_digital_twin_ucb --backend plantsim --beta 0.3 --jump_batch 20 --jump_target_theta 19.5 --out_dir figs/project2_ucb_jump
# python -m calib.run_project2_digital_twin_ucb --backend plantsim --theta_schedule_mode platform --platform_levels "3,11.5,20,30" --platform_block_batches 20 --platform_noise_std 0.8 --out_dir figs/project2_platform
