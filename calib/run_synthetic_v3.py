# =============================================================
# file: calib/run_synthetic.py (Refactored for clarity)
# =============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator
from .koh_calibrator import KOHCalibrator
from .enhanced_data import create_config2_config, EnhancedSyntheticDataStream1, EnhancedChangepointConfig
from .projected_calibrator import BOCPDProjectedCalibrator
from .koh_batch_calibrator import KOHBatchCalibrator

from time import time

# -------------------------------------------------------------
# Global Matplotlib Setup
# -------------------------------------------------------------
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def theta_true_schedule(t):
    """
    t: observation index (0 ... T)
    returns true a2 parameter
    """
    if t < 1600:
        # 7.5 -> 12 (linear drift)
        return 7.5 + (12.0 - 7.5) * (t / 1600)
    elif t < 2000:
        # abrupt drop to 5
        return 5.0
    else:
        # 5 -> 7.5 (slow drift)
        return 5.0 + (7.5 - 5.0) * ((t - 2000) / (4000 - 2000))

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------
def calculate_crps(y_true: torch.Tensor, mu_pred: torch.Tensor, var_pred: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS for Gaussian predictive distribution."""
    sigma = torch.sqrt(var_pred.clamp_min(1e-12))
    z = (y_true - mu_pred) / sigma
    c1 = z * torch.erf(z / torch.sqrt(torch.tensor(2.0)))
    c2 = torch.sqrt(torch.tensor(2.0 / np.pi)) * torch.exp(-0.5 * z**2)
    c3 = torch.sqrt(torch.tensor(1.0 / np.pi))
    return sigma * (c1 + c2 - c3)


def computer_model_config2(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Config2 computer model: y*(x, θ) = sin(5 θ x) + 5x."""
    if x.dim() == 1:
        x = x[None, :]
    if theta.dim() == 1:
        theta = theta[None, :]
    th, xx = theta[:, 0:1], x[:, 0:1]
    return torch.sin(5.0 * th * xx) + 5.0 * xx

def computer_model_config2_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Config2 computer model (NumPy version):
    y*(x, θ) = sin(5 θ x) + 5x
    - x: shape [n] or [n, 1]
    - theta: shape [p] or [1, p]
    Returns: shape [n, 1]
    """
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    th = theta[:, [0]]
    xx = x[:, [0]]
    return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)


# -------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------
def plot_results(results: dict, prefix: str = "cfg2"):
    """Plot RMSE and CRPS curves for all methods in results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- RMSE ----------
    ax1 = axes[0]
    for name, data in results.items():
        xs = data["times_rmse"]
        ys = data["rmse_history"]
        if len(xs) > 0:
            ax1.plot(xs, ys, label=name, linewidth=2)
    cp_times = list(results.values())[0]["changepoint_times"]
    for cp in cp_times:
        ax1.axvline(cp, color="red", linestyle="--", alpha=0.6)
    ax1.set_title("Batch RMSE Comparison")
    ax1.set_xlabel("Observation Time (t)")
    ax1.set_ylabel("RMSE")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- CRPS ----------
    ax2 = axes[1]
    for name, data in results.items():
        xs = data["times_rmse"]
        ys = data["crps_history"]
        if len(xs) > 0:
            ax2.plot(xs, ys, label=name, linewidth=2)
    for cp in cp_times:
        ax2.axvline(cp, color="red", linestyle="--", alpha=0.6)
    ax2.set_title("Batch CRPS Comparison")
    ax2.set_xlabel("Observation Time (t)")
    ax2.set_ylabel("CRPS")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{prefix}_rmse_crps_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_history(history, prefix: str = "cfg2"):
    times = []
    run_length_0 = []
    run_length_1 = []
    mass_0 = []
    mass_1 = []
    theta_0 = []
    theta_1 = []

    for rec in history:
        times.append(rec["time"])

        e0 = rec["experts"][0]
        run_length_0.append(e0["run_length"])
        mass_0.append(e0["mass"])
        theta_0.append(e0["theta_mean"][0])

        try:
            e1 = rec["experts"][1]
            run_length_1.append(e1["run_length"])
            mass_1.append(e1["mass"])
            theta_1.append(e1["theta_mean"][0])
        except:
            run_length_1.append(math.nan)
            mass_1.append(0.0)         
            theta_1.append(math.nan)

    import matplotlib.pyplot as plt
    import numpy as np

    t = np.array(times)
    m0 = np.array(mass_0)
    m1 = np.array(mass_1)
    m1 = np.ma.masked_invalid(np.array(mass_1))

    fig, ax1 = plt.subplots(figsize=(12,6))
    rl1 = np.ma.masked_invalid(np.array(run_length_1))

    # --------------------------
    # 1) 左轴：run_length 折线
    # --------------------------
    ax1.plot(t, run_length_0, color="green", linewidth=2, label="Expert 0 run_length")
    ax1.plot(t, rl1, color="red", linewidth=2, label="Expert 1 run_length")
    ax1.set_ylabel("Run Length")
    ax1.set_xlabel("Time step")
    ax1.legend(loc="upper left")

    # --------------------------
    # 2) 右轴：mass 堆叠柱状图
    # --------------------------
    ax2 = ax1.twinx()
    bar_width = (t[1] - t[0]) * 0.8 if len(t) > 1 else 0.8

    ax2.bar(t, m0, width=bar_width, color="green", alpha=0.3, label="mass expert 0")
    ax2.bar(t, m1, width=bar_width, bottom=m0, color="red", alpha=0.3, label="mass expert 1")

    ax2.set_ylabel("Mass (stacked)")

    plt.title("Experts run-length evolution + mass proportion")
    plt.savefig(f"{prefix}_experts_history.png", dpi=300, bbox_inches="tight")
    plt.show()    


# -------------------------------------------------------------
# Main Experiment Function
# -------------------------------------------------------------
def run_config2_experiment(prefix: str = "cfg2"):
    """
    Config2 experiment comparing:
        - Standard BOCPD
        - Restart BOCPD
        - KOH Calibration (explicit GP discrepancy)
    """
    # -------- Experiment setup --------
    target_observations = 4000
    batch_size = 40
    # batch_size = 40
    assert target_observations % batch_size == 0
    seed_fixed = 123

    calib_cfg = CalibrationConfig()
    device, dtype = calib_cfg.model.device, calib_cfg.model.dtype

    # Emulator
    emulator = DeterministicSimulator(func=computer_model_config2, enable_autograd=True)

    cfg2 = create_config2_config(
        n_observations=target_observations,
        noise_variance=0.04,               # 0.2^2
        batch_size_range=(batch_size, batch_size),
    )
    cp_times = [800, 1600, 2400, 3200]
    # cfg2.changepoints = [
    #     EnhancedChangepointConfig(time=cp_times[0], phys_param_new=torch.tensor([5, 5, 5], dtype=dtype, device=device)),
    #     EnhancedChangepointConfig(time=cp_times[1], phys_param_new=torch.tensor([5, 12, 5], dtype=dtype, device=device)),
    #     EnhancedChangepointConfig(time=cp_times[2], phys_param_new=torch.tensor([5, 7.0, 5], dtype=dtype, device=device)),
    #     EnhancedChangepointConfig(time=cp_times[3], phys_param_new=torch.tensor([5, 11, 5], dtype=dtype, device=device)),
    # ]
    cfg2.changepoints = [
        EnhancedChangepointConfig(time=1600, phys_param_new=torch.tensor([5, 5.0, 5])),
        EnhancedChangepointConfig(time=2000, phys_param_new=None),
    ]

    seed_fixed = 123

    # --------- 先验 θ ∈ [0,3] ----------
    theta_dim = 1
    def prior_sampler(N: int) -> torch.Tensor:
        lo = torch.full((theta_dim,), 0.0, dtype=dtype, device=device)
        hi = torch.full((theta_dim,), 3.0, dtype=dtype, device=device)
        u = torch.rand(N, theta_dim, dtype=dtype, device=device)
        return lo + (hi - lo) * u

    # -------- Methods definition --------
    methods = {
        "KOH": {
            "type": "koh",
            "params": dict(
                update_mode="full",
                window_length=200,
                lengthscale=0.3,
                variance=1.0,
                noise_var=0.04,
                optimize_theta=True,
                optimize_hypers=True,
                max_opt_steps=200,
            ),
        },
        "Standard": {
            "type": "bocpd",
            "bocpd_mode": "standard",
        },
        "Restart": {
            "type": "bocpd",
            "bocpd_mode": "restart",
        },
        # "Proj+BOCPD+δGP": {
        #     "type": "proj_bocpd",
        #     "params": dict(
        #         topk=5,
        #         hazard_lambda=800.0,
        #         yhat_update_mode="window",
        #         yhat_window_length=800,
        #         yhat_fit_iters=200,
        #         theta_solver_kwargs=dict(
        #             n_theta_samples=64,
        #             n_restart=5,
        #             proj_on="history",      # 或 "grid"; grid 需指定 x_range/n_grid
        #             delta_update_mode="full",   # 或 "window"
        #             delta_window_length=800,
        #             delta_fit_iters=200,
        #         )
        #     ),
        # },
    }

    results = {}

    # -----------------------------------------------------
    # Loop over methods
    # -----------------------------------------------------
    for method_name, meta in methods.items():
        start_time = time()
        print(f"\n=== Running {method_name} ===")
        stream = EnhancedSyntheticDataStream1(cfg2, seed=seed_fixed)

        prediction_errors = []
        rmse_history = []
        crps_history = []
        batch_times_all = []
        total_observations = 0
        theta_history = []

        # ----------- BOCPD methods -----------
        if meta["type"] == "bocpd":
            calib_cfg.bocpd.bocpd_mode = meta["bocpd_mode"]
            if meta["bocpd_mode"] == "restart":
                calib_cfg.bocpd.use_backdated_restart = False
                # calib_cfg.bocpd.restart_margin = 0.2
                calib_cfg.bocpd.restart_margin = 1
                calib_cfg.bocpd.restart_cooldown = 2
                calib_cfg.bocpd.use_restart = True
                calib_cfg.bocpd.restart_threshold = 0.85
                history = []
            else:
                calib_cfg.bocpd.use_restart = True
                calib_cfg.bocpd.restart_threshold = 0.85

            calibrator = OnlineBayesCalibrator(calib_cfg, emulator, prior_sampler)

            while total_observations < target_observations:
                if total_observations % 100 == 0:
                    print(f"{method_name} Total observations: {total_observations}")
                X_batch, Y_batch = stream.next(batch_size)
                batch_times_all.append(total_observations)

                if total_observations > 0:
                    pred = calibrator.predict_batch(X_batch)
                    mu, var = pred["mu"], pred["var"]

                    # --- batch RMSE / CRPS ---
                    rmse_batch = float(torch.sqrt(torch.mean((Y_batch - mu) ** 2)))
                    rmse_history.append(rmse_batch)
                    crps_batch = float(torch.mean(calculate_crps(Y_batch, mu, var)))
                    crps_history.append(crps_batch)

                rec = calibrator.step_batch(X_batch, Y_batch, verbose=True)
                if meta["bocpd_mode"] == "restart":
                    record = {"time": calibrator.bocpd.t, "experts": [{"run_length": e["run_length"], "mass": e["mass"], "theta_mean": e["theta_mean"]} for e in rec["experts_debug"]]}
                    history.append(record)


                mean_theta, _ = calibrator._aggregate_particles()
                theta_history.append(float(mean_theta.cpu().numpy()))

                total_observations += X_batch.shape[0]

            results[method_name] = dict(
                rmse_history=rmse_history,
                crps_history=crps_history,
                times_rmse=batch_times_all[1:1 + len(rmse_history)],
                changepoint_times=cp_times,
                theta_history=theta_history,
            )

        # ----------- KOH method -----------
        elif meta["type"] == "koh":
            koh = KOHBatchCalibrator(
                simulator=computer_model_config2_np,
                theta_init=1.0,
                theta_bounds=(0.05, 3.0),
                window_size=800,
            )

            while total_observations < target_observations:
                if total_observations % 100 == 0:
                    print(f"{method_name} Total observations: {total_observations}")
                X_batch, Y_batch = stream.next(batch_size)
                X_batch = X_batch.detach().cpu().numpy()
                Y_batch = Y_batch.detach().cpu().numpy()
                batch_times_all.append(total_observations)

                if total_observations > 0:
                    out = koh.predict(X_batch)
                    mu, var = out["mu"], out["var"]

                    rmse_batch = koh.rmse(Y_batch, mu)
                    crps_batch = koh.crps_gaussian(Y_batch, mu, var)
                    rmse_history.append(rmse_batch)
                    crps_history.append(crps_batch)

                    if total_observations % 100==0:
                        print(f"{method_name} Total observations: {total_observations}, RMSE: {rmse_batch:.4f}, CRPS: {crps_batch:.4f}, theta: {koh.theta_:.4f}")

                    # x_train, y_train = X_pres.reshape(-1), y_pres.reshape(-1)
                    # x_test, y_test_true = X_batch.reshape(-1), Y_batch.reshape(-1)
                    # n_train = x_train.shape[0]
                    # # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
                    # import pandas as pd
                    # from scipy.optimize import minimize
                    # from numpy.linalg import cholesky, solve
                    # def y_sim(x, theta):
                    #     return np.sin(5 * theta * x) + 5 * x
                    # def rbf_kernel(X, Y=None, ell=0.2, sf2=1.0):
                    #     X = X.reshape(-1,1)
                    #     Y = X if Y is None else Y.reshape(-1,1)
                    #     d2 = (X - Y.T)**2
                    #     return sf2 * np.exp(-0.5 * d2 / (ell**2))

                    # def neg_log_marginal_likelihood(params):
                    #     # params = [theta, log_ell, log_sf2, log_sn2]
                    #     theta, log_ell, log_sf2, log_sn2 = params
                    #     ell = np.exp(log_ell)
                    #     sf2 = np.exp(log_sf2)
                    #     sn2 = np.exp(log_sn2)

                    #     r = y_train - y_sim(x_train, theta)  # the discrepancy GP explains this
                    #     K = rbf_kernel(x_train, ell=ell, sf2=sf2) + sn2 * np.eye(n_train)

                    #     # Cholesky factorization
                    #     try:
                    #         L = cholesky(K)
                    #     except np.linalg.LinAlgError:
                    #         return 1e30

                    #     # r^T K^{-1} r and log|K|
                    #     v = solve(L, r)
                    #     alpha = solve(L.T, v)
                    #     nll = 0.5 * np.dot(r, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n_train * np.log(2*np.pi)
                    #     return float(nll)

                    # # init and bounds
                    # init_theta   = 1.0
                    # init_log_ell = np.log(0.2)
                    # init_log_sf2 = np.log(1.0)
                    # init_log_sn2 = np.log(1e-6)

                    # bounds = [
                    #     (0.05, 3.0),                  # theta
                    #     (np.log(1e-3), np.log(10.0)), # log_ell
                    #     (np.log(1e-6), np.log(10.0)), # log_sf2
                    #     (np.log(1e-9), np.log(1e-1))  # log_sn2
                    # ]

                    # res_koh = minimize(
                    #     neg_log_marginal_likelihood,
                    #     x0=np.array([init_theta, init_log_ell, init_log_sf2, init_log_sn2]),
                    #     bounds=bounds,
                    #     method='L-BFGS-B'
                    # )

                    # theta_hat_koh, log_ell_hat, log_sf2_hat, log_sn2_hat = res_koh.x
                    # ell_hat = float(np.exp(log_ell_hat))
                    # sf2_hat = float(np.exp(log_sf2_hat))
                    # sn2_hat = float(np.exp(log_sn2_hat))

                    # # GP posterior of discrepancy at test
                    # r_train = y_train - y_sim(x_train, theta_hat_koh)
                    # K  = rbf_kernel(x_train, ell=ell_hat, sf2=sf2_hat) + sn2_hat * np.eye(n_train)
                    # Ks = rbf_kernel(x_train, x_test, ell=ell_hat, sf2=sf2_hat)
                    # L  = cholesky(K)
                    # alpha = solve(L.T, solve(L, r_train))
                    # mu_delta_test = Ks.T @ alpha

                    # # final KOH prediction on test
                    # y_pred_test_koh = y_sim(x_test, theta_hat_koh) + mu_delta_test
                    # rmse_koh = float(np.sqrt(np.mean((y_pred_test_koh - y_test_true)**2)))
                    # print(f"RMSE of KOH: {rmse_koh:.4f}")
                # X_pres, y_pres = X_batch, Y_batch

                koh.update(X_batch, Y_batch)
                koh.fit()
                theta_history.append(koh.theta_)
                total_observations += X_batch.shape[0]

            results[method_name] = dict(
                rmse_history=rmse_history,
                crps_history=crps_history,
                times_rmse=batch_times_all[1:1 + len(rmse_history)],
                changepoint_times=cp_times,
                theta_history=theta_history,
            )
            
        elif meta["type"] == "proj_bocpd":
            calib = BOCPDProjectedCalibrator(
                simulator=emulator.func,
                theta_bounds=(torch.tensor([0.0], dtype=dtype, device=device),
                            torch.tensor([3.0], dtype=dtype, device=device)),
                device=device, dtype=dtype,
                **meta["params"]
            )
            rmse_history, crps_history, batch_times = [], [], []
            total = 0
            while total < target_observations:
                Xb, Yb = stream.next()
                batch_times.append(total)

                if total > 0:
                    out = calib.predict(Xb)
                    mu, var = out["mu"], out["var"]
                    rmse_history.append(float(torch.sqrt(torch.mean((Yb - mu) ** 2))))
                    # 你的 CRPS 函数
                    crps = calculate_crps(Yb, mu, var).mean().item()
                    crps_history.append(crps)

                calib.update(Xb, Yb)
                total += Xb.shape[0]

            results["Proj+BOCPD+δGP"] = dict(
                rmse_history=rmse_history,
                crps_history=crps_history,
                times_rmse=batch_times[1:1+len(rmse_history)],
                changepoint_times=cp_times,
            )

        end_time = time()
        print(f"{method_name} Time taken: {end_time - start_time:.2f} seconds")
        if meta["type"] == "bocpd" and meta["bocpd_mode"] == "restart":
            plot_history(history, prefix=f"{prefix}_{method_name}")

    # -----------------------------------------------------
    # Plot and Save
    # -----------------------------------------------------
    plot_results(results, prefix=prefix)

    np.savez_compressed(
        f"{prefix}_results_summary.npz",
        **{f"{name}_rmse": np.array(data["rmse_history"]) for name, data in results.items()},
        **{f"{name}_crps": np.array(data["crps_history"]) for name, data in results.items()},
        **{f"{name}_times": np.array(data["times_rmse"]) for name, data in results.items()},
        **{f"{name}_theta": np.array(data["theta_history"]) for name, data in results.items()},

    )

    print("\nExperiment complete. Results saved and plotted.")
    for name, data in results.items():
        print(f"  {name}: Final RMSE={data['rmse_history'][-1]:.4f}, "
              f"Final CRPS={data['crps_history'][-1]:.4f}")

    return results


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    run_config2_experiment(prefix="cfg2_t4000_debug_26010803_40_batch_compareBatch")
