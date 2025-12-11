# # =============================================================
# # file: calib/run_bocpd_hazard_sensitivity.py
# # =============================================================
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# from .configs import CalibrationConfig
# from .emulator import DeterministicSimulator
# from .online_calibrator import OnlineBayesCalibrator
# from .enhanced_data import create_config2_config, EnhancedSyntheticDataStream, EnhancedChangepointConfig
# from .run_synthetic_v1 import calculate_crps, computer_model_config2

# rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
# rcParams['axes.unicode_minus'] = False


# def run_restart_bocpd_sensitivity(hazard_lambdas=None, prefix="hazard_sensitivity"):
#     """
#     Sensitivity analysis on hazard function λ for Restart BOCPD.

#     For each λ, run the experiment, record restart times and RMSE evolution.
#     """
#     if hazard_lambdas is None:
#         hazard_lambdas = [100.0, 200.0, 400.0, 800.0, 1600.0]

#     # ---------------- Experiment setup ----------------
#     target_observations = 4000
#     batch_size = 40
#     seed_fixed = 123

#     calib_cfg = CalibrationConfig()
#     device, dtype = calib_cfg.model.device, calib_cfg.model.dtype

#     emulator = DeterministicSimulator(func=computer_model_config2, enable_autograd=True)

#     cfg2 = create_config2_config(
#         n_observations=target_observations,
#         noise_variance=0.04,
#         batch_size_range=(batch_size, batch_size),
#     )
#     cp_times = [800, 1600, 2400, 3200]
#     cfg2.changepoints = [
#         EnhancedChangepointConfig(time=cp_times[0], phys_param_new=torch.tensor([5, 5, 5], dtype=dtype, device=device)),
#         EnhancedChangepointConfig(time=cp_times[1], phys_param_new=torch.tensor([5, 12, 5], dtype=dtype, device=device)),
#         EnhancedChangepointConfig(time=cp_times[2], phys_param_new=torch.tensor([5, 7.0, 5], dtype=dtype, device=device)),
#         EnhancedChangepointConfig(time=cp_times[3], phys_param_new=torch.tensor([5, 11, 5], dtype=dtype, device=device)),
#     ]

#     theta_dim = 1
#     def prior_sampler(N: int) -> torch.Tensor:
#         lo = torch.full((theta_dim,), 0.0, dtype=dtype, device=device)
#         hi = torch.full((theta_dim,), 3.0, dtype=dtype, device=device)
#         u = torch.rand(N, theta_dim, dtype=dtype, device=device)
#         return lo + (hi - lo) * u

#     # ---------------- Run experiments ----------------
#     sensitivity_results = {}

#     for lam in hazard_lambdas:
#         print(f"\n=== Running Restart BOCPD with hazard λ={lam:.1f} ===")

#         calib_cfg.bocpd.bocpd_mode = "restart"
#         calib_cfg.bocpd.use_restart = True
#         calib_cfg.bocpd.restart_threshold = 0.85
#         calib_cfg.bocpd.hazard_lambda = lam
#         calib_cfg.bocpd.use_backdated_restart = False
#         calib_cfg.bocpd.restart_margin = 0.2
#         calib_cfg.bocpd.restart_cooldown = 2

#         calibrator = OnlineBayesCalibrator(calib_cfg, emulator, prior_sampler)
#         stream = EnhancedSyntheticDataStream(cfg2, seed=seed_fixed)

#         rmse_history = []
#         restart_times = []
#         batch_times = []
#         total_obs = 0

#         while total_obs < target_observations:
#             X_batch, Y_batch = stream.next()
#             batch_times.append(total_obs)

#             if total_obs > 0:
#                 pred = calibrator.predict_batch(X_batch)
#                 mu, var = pred["mu"], pred["var"]
#                 rmse_batch = float(torch.sqrt(torch.mean((Y_batch - mu) ** 2)))
#                 rmse_history.append(rmse_batch)

#             out = calibrator.step_batch(X_batch, Y_batch, verbose=False)
#             if out.get("did_restart", False):
#                 restart_times.append(total_obs)

#             total_obs += X_batch.shape[0]

#         sensitivity_results[lam] = dict(
#             rmse_history=rmse_history,
#             restart_times=restart_times,
#             times=batch_times[1:1 + len(rmse_history)],
#         )

#     # ---------------- Plot results ----------------
#     plot_hazard_sensitivity(sensitivity_results, hazard_lambdas, prefix)
#     return sensitivity_results


# def plot_hazard_sensitivity(sensitivity_results, hazard_lambdas, prefix="hazard_sensitivity"):
#     """Plot RMSE curves and restart event markers for each hazard λ."""
#     plt.figure(figsize=(12, 7))
#     for lam in hazard_lambdas:
#         data = sensitivity_results[lam]
#         times = data["times"]
#         rmse = data["rmse_history"]
#         plt.plot(times, rmse, label=f"λ={lam}", linewidth=2)
#         for rt in data["restart_times"]:
#             plt.axvline(rt, color="red", linestyle="--", alpha=0.3)

#     plt.title("Restart BOCPD Sensitivity to Hazard λ")
#     plt.xlabel("Observation Time (t)")
#     plt.ylabel("RMSE")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{prefix}_rmse_restarts.png", dpi=300, bbox_inches="tight")
#     plt.show()

#     # Summary plot of restart counts vs hazard λ
#     plt.figure(figsize=(8, 5))
#     counts = [len(sensitivity_results[lam]["restart_times"]) for lam in hazard_lambdas]
#     plt.plot(hazard_lambdas, counts, "o-", linewidth=2)
#     plt.title("Restart Count vs Hazard λ")
#     plt.xlabel("Hazard λ")
#     plt.ylabel("Number of Restarts")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{prefix}_restart_counts.png", dpi=300, bbox_inches="tight")
#     plt.show()

#     print("\nHazard Sensitivity Summary:")
#     for lam in hazard_lambdas:
#         data = sensitivity_results[lam]
#         print(f"  λ={lam:<6}: Restarts={len(data['restart_times'])}, Final RMSE={data['rmse_history'][-1]:.4f}")


# # -------------------------------------------------------------
# # Entry point
# # -------------------------------------------------------------
# if __name__ == "__main__":
#     run_restart_bocpd_sensitivity()

# =============================================================
# file: calib/run_bocpd_hazard_form_comparison.py
# =============================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator
from .enhanced_data import create_config2_config, EnhancedSyntheticDataStream, EnhancedChangepointConfig
from .run_synthetic_v1 import calculate_crps, computer_model_config2

# Matplotlib setup
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


# -------------------------------------------------------------
# Hazard Function Definitions
# -------------------------------------------------------------
def make_hazard_function(hazard_type: str, lam: float = 400.0, k: float = 1.5):
    """
    Returns a callable hazard function h(r) for the given type.
    hazard_type ∈ {'constant', 'geometric', 'linear', 'weibull'}
    """
    def hazard_constant(r: torch.Tensor):
        return torch.full_like(r, 1.0 / lam)

    def hazard_geometric(r: torch.Tensor):
        return 1.0 / (lam + r)

    def hazard_linear(r: torch.Tensor):
        return torch.clamp(r / lam, max=1.0)

    def hazard_weibull(r: torch.Tensor):
        return (k / lam) * torch.pow(r / lam, k - 1)

    if hazard_type == "constant":
        return hazard_constant
    elif hazard_type == "geometric":
        return hazard_geometric
    elif hazard_type == "linear":
        return hazard_linear
    elif hazard_type == "weibull":
        return hazard_weibull
    else:
        raise ValueError(f"Unknown hazard_type: {hazard_type}")


# -------------------------------------------------------------
# Main Comparison Function
# -------------------------------------------------------------
def run_hazard_form_comparison(hazard_types=None, prefix="hazard_form_cmp"):
    """
    Compare different hazard function forms (constant / geometric / linear / weibull)
    for Restart BOCPD.
    """
    if hazard_types is None:
        hazard_types = ["constant", "geometric", "linear", "weibull"]

    lam_default = 400.0
    weibull_k = 1.5

    # ---------------- Experiment setup ----------------
    target_observations = 4000
    batch_size = 40
    seed_fixed = 123

    calib_cfg = CalibrationConfig()
    device, dtype = calib_cfg.model.device, calib_cfg.model.dtype

    emulator = DeterministicSimulator(func=computer_model_config2, enable_autograd=True)

    cfg2 = create_config2_config(
        n_observations=target_observations,
        noise_variance=0.04,
        batch_size_range=(batch_size, batch_size),
    )
    cp_times = [800, 1600, 2400, 3200]
    cfg2.changepoints = [
        EnhancedChangepointConfig(time=cp_times[0], phys_param_new=torch.tensor([5, 5, 5], dtype=dtype, device=device)),
        EnhancedChangepointConfig(time=cp_times[1], phys_param_new=torch.tensor([5, 12, 5], dtype=dtype, device=device)),
        EnhancedChangepointConfig(time=cp_times[2], phys_param_new=torch.tensor([5, 7.0, 5], dtype=dtype, device=device)),
        EnhancedChangepointConfig(time=cp_times[3], phys_param_new=torch.tensor([5, 11, 5], dtype=dtype, device=device)),
    ]

    theta_dim = 1
    def prior_sampler(N: int) -> torch.Tensor:
        lo = torch.full((theta_dim,), 0.0, dtype=dtype, device=device)
        hi = torch.full((theta_dim,), 3.0, dtype=dtype, device=device)
        u = torch.rand(N, theta_dim, dtype=dtype, device=device)
        return lo + (hi - lo) * u

    # ---------------- Run experiments ----------------
    comparison_results = {}

    for htype in hazard_types:
        print(f"\n=== Running Restart BOCPD with hazard form: {htype} ===")

        calib_cfg.bocpd.bocpd_mode = "restart"
        calib_cfg.bocpd.use_restart = True
        calib_cfg.bocpd.restart_threshold = 0.85
        calib_cfg.bocpd.hazard_lambda = lam_default
        calib_cfg.bocpd.use_backdated_restart = False
        calib_cfg.bocpd.restart_margin = 0.2
        calib_cfg.bocpd.restart_cooldown = 2

        # Assign custom hazard function dynamically
        calib_cfg.bocpd.hazard = make_hazard_function(htype, lam=lam_default, k=weibull_k)

        calibrator = OnlineBayesCalibrator(calib_cfg, emulator, prior_sampler)
        stream = EnhancedSyntheticDataStream(cfg2, seed=seed_fixed)

        rmse_history = []
        restart_times = []
        batch_times = []
        total_obs = 0

        while total_obs < target_observations:
            X_batch, Y_batch = stream.next()
            batch_times.append(total_obs)

            if total_obs > 0:
                pred = calibrator.predict_batch(X_batch)
                mu, var = pred["mu"], pred["var"]
                rmse_batch = float(torch.sqrt(torch.mean((Y_batch - mu) ** 2)))
                rmse_history.append(rmse_batch)

            out = calibrator.step_batch(X_batch, Y_batch, verbose=False)
            if out.get("did_restart", False):
                restart_times.append(total_obs)

            total_obs += X_batch.shape[0]

        comparison_results[htype] = dict(
            rmse_history=rmse_history,
            restart_times=restart_times,
            times=batch_times[1:1 + len(rmse_history)],
        )

    # ---------------- Plot results ----------------
    plot_hazard_form_comparison(comparison_results, hazard_types, prefix)
    return comparison_results


# -------------------------------------------------------------
# Plotting
# -------------------------------------------------------------
def plot_hazard_form_comparison(results, hazard_types, prefix="hazard_form_cmp"):
    """Plot RMSE and restart comparisons across hazard function types."""
    plt.figure(figsize=(12, 7))
    for htype in hazard_types:
        data = results[htype]
        plt.plot(data["times"], data["rmse_history"], label=htype.capitalize(), linewidth=2)
        for rt in data["restart_times"]:
            plt.axvline(rt, color="red", linestyle="--", alpha=0.2)
    plt.title("Restart BOCPD Comparison Across Hazard Function Forms")
    plt.xlabel("Observation Time (t)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_rmse_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Restart count summary
    plt.figure(figsize=(8, 5))
    counts = [len(results[htype]["restart_times"]) for htype in hazard_types]
    plt.bar(hazard_types, counts, color="steelblue")
    plt.title("Restart Count Across Hazard Function Forms")
    plt.xlabel("Hazard Function Type")
    plt.ylabel("Number of Restarts")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_restart_counts.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nHazard Function Comparison Summary:")
    for htype in hazard_types:
        data = results[htype]
        print(f"  {htype.capitalize():<10}: Restarts={len(data['restart_times'])}, "
              f"Final RMSE={data['rmse_history'][-1]:.4f}")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    run_hazard_form_comparison()
