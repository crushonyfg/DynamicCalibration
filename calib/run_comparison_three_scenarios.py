# =============================================================
# run_comparison_three_scenarios.py
# =============================================================
"""
Compare three calibration methods across three θ* change scenarios.

Methods
-------
  BC  : Batch Calibration (KOH with sliding window of 20 batches)
  DA  : Dynamic Adaptation (PF-NoDiscrepancy)
  Ours: R-BOCPD-PF-NoDiscrepancy

Scenarios
---------
  1. Gradual drift   – θ*(t) increases linearly
  2. Sudden jump      – θ*(t) has discrete jumps (3 segments)
  3. Gradual + Sudden – linear drift within segments + sudden jumps

Output
------
  Figure 1: 3 subplots (one per scenario), BC + DA vs ground truth
  Figure 2: 3 subplots (one per scenario), BC + DA + Ours vs ground truth

Usage
-----
  python -m calib.run_comparison_three_scenarios [--out_dir DIR] [--seed N]
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from time import time as timer
from tqdm import tqdm

# ---- calib package imports (R-BOCPD-PF) ----
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator


# =================================================================
# Physical system  &  Simulator   (Config-2, same as existing exps)
# =================================================================
def computer_model_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """y_sim(x, θ) = sin(5θx) + 5x   (numpy, batched)"""
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    th = theta[:, [0]]
    xx = x[:, [0]]
    return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)


def computer_model_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """y_sim(x, θ) = sin(5θx) + 5x   (torch)"""
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    return torch.sin(5.0 * theta[:, 0:1] * x[:, 0:1]) + 5.0 * x[:, 0:1]


def physical_system(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """η(x; φ) = a1·x·cos(a2·x) + a3·x,   φ = [a1, a2, a3]"""
    x = x.reshape(-1)
    a1, a2, a3 = phi
    return a1 * x * np.cos(a2 * x) + a3 * x


def oracle_theta(phi: np.ndarray, grid: np.ndarray) -> float:
    """θ*(φ) = argmin_θ ∫ (η(x;φ) − y_sim(x,θ))² dx  (dense-grid approx)"""
    x = np.linspace(0, 1, 400).reshape(-1, 1)
    eta = physical_system(x, phi)
    errs = []
    for th in grid:
        ys = computer_model_np(x, np.array([th]))
        errs.append(np.mean((eta - ys) ** 2))
    return float(grid[int(np.argmin(errs))])


# =================================================================
# φ2 ↔ θ* mapping  (inverse oracle)
# =================================================================
def build_phi2_from_theta_star(
    phi2_grid: np.ndarray,
    theta_grid: np.ndarray,
    a1: float = 5.0,
    a3: float = 5.0,
):
    """Build interpolant φ2 = f(θ*) so we can prescribe θ*(t) and recover φ(t)."""
    theta_star_vals = []
    for phi2 in tqdm(phi2_grid, desc="  Oracle θ*(φ2)", unit="pt"):
        phi = np.array([a1, phi2, a3])
        theta_star = oracle_theta(phi, theta_grid)
        theta_star_vals.append(theta_star)
    theta_star_vals = np.asarray(theta_star_vals)

    # ---- CRITICAL: sort by θ* for a valid interpolant ----
    order = np.argsort(theta_star_vals)
    theta_sorted = theta_star_vals[order]
    phi2_sorted = phi2_grid[order]

    # Remove near-duplicate θ* values (keeps first occurrence)
    # to prevent divide-by-zero in interp1d slope computation
    _, unique_idx = np.unique(np.round(theta_sorted, decimals=8), return_index=True)
    theta_sorted = theta_sorted[unique_idx]
    phi2_sorted = phi2_sorted[unique_idx]

    phi2_of_theta = interp1d(
        theta_sorted, phi2_sorted,
        kind="linear", fill_value="extrapolate",
        bounds_error=False, assume_sorted=True,
    )
    return phi2_of_theta, theta_star_vals


# =================================================================
# Unified data stream driven by θ*(t) schedule
# =================================================================
class ThetaScheduleDataStream:
    """
    Generate physical observations where the ground-truth best-fit θ*(t)
    follows an arbitrary user-defined schedule.

    θ*(t) → φ2(t) via interpolation → physical system generates y.
    """

    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        theta_schedule: Callable[[int], float],
        phi2_of_theta,
        phi_base=np.array([5.0, 0.0, 5.0]),
        seed: int = 0,
    ):
        self.T = total_T
        self.bs = batch_size
        self.noise_sd = noise_sd
        self.theta_schedule = theta_schedule
        self.phi2_of_theta = phi2_of_theta
        self.phi_base = phi_base.copy()
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.theta_star_history: List[float] = []
        self.phi_history: List[np.ndarray] = []

    def true_theta_star(self, t: int) -> float:
        return self.theta_schedule(t)

    def true_phi(self, t: int) -> np.ndarray:
        theta_star = self.true_theta_star(t)
        phi = self.phi_base.copy()
        phi[1] = float(self.phi2_of_theta(theta_star))
        return phi

    def next(self):
        if self.t >= self.T:
            raise StopIteration
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]
        phi_t = self.true_phi(self.t)
        theta_star_t = self.true_theta_star(self.t)
        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)
        self.phi_history.append(phi_t.copy())
        self.theta_star_history.append(theta_star_t)
        self.t += self.bs
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =================================================================
# Method DA : PF-NoDiscrepancy  (Particle Filter, no GP discrepancy)
# =================================================================
class PFNoDiscrepancy:
    """
    Batch particle filter on θ only (no GP discrepancy).
    Likelihood per observation:  N(y_t | y_sim(x_t, θ), σ²)
    """

    def __init__(
        self,
        n_particles: int = 1024,
        theta_lo: float = 0.0,
        theta_hi: float = 3.0,
        sigma_obs: float = 0.2,
        resample_ess_ratio: float = 0.5,
        theta_move_std: float = 0.03,
        seed: int = 42,
    ):
        self.N = n_particles
        self.lo, self.hi = theta_lo, theta_hi
        self.sigma2 = sigma_obs ** 2
        self.ess_ratio = resample_ess_ratio
        self.move_std = theta_move_std
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(self.lo, self.hi, size=self.N)
        self.w = np.ones(self.N) / self.N

    def _normalize_logw(self, logw: np.ndarray) -> np.ndarray:
        logw = logw - logsumexp(logw)
        return np.exp(logw)

    def _ess(self) -> float:
        return 1.0 / np.sum(self.w ** 2)

    def _systematic_resample(self):
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        cumsum = np.cumsum(self.w)
        idx = np.searchsorted(cumsum, positions, side="left")
        self.theta = self.theta[idx]
        self.w[:] = 1.0 / self.N

    def _rejuvenate(self):
        self.theta += self.rng.normal(0.0, self.move_std, size=self.N)
        self.theta = np.clip(self.theta, self.lo, self.hi)

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        """Xb: (bs, 1) or (bs,), Yb: (bs,)"""
        logw = np.log(self.w + 1e-300)
        for x_t, y_t in zip(Xb.ravel(), Yb.ravel()):
            ys = np.sin(5.0 * self.theta * x_t) + 5.0 * x_t
            logw += -0.5 * (np.log(2.0 * np.pi * self.sigma2)
                            + (y_t - ys) ** 2 / self.sigma2)
        self.w = self._normalize_logw(logw)
        if self._ess() < self.ess_ratio * self.N:
            self._systematic_resample()
            self._rejuvenate()

    def mean_theta(self) -> float:
        return float(np.sum(self.w * self.theta))


# =================================================================
# Method BC : KOH with Sliding Window  (batch calibration)
# =================================================================
class KOHSlidingWindow:
    """
    KOH-style batch calibration using profile marginal likelihood:
      y = y_sim(x, θ) + δ(x) + ε,   δ ~ GP(0, K_δ),  ε ~ N(0, σ²)

    For a grid of θ values, compute  log p(r | K)  where  r = Y − y_sim(X, θ)
    and  K = K_δ + σ²I.  Posterior ∝ prior(θ) × p(r | K).
    Uses a sliding window of the last `window_batches` batches.
    """

    def __init__(
        self,
        window_batches: int = 20,
        batch_size: int = 20,
        theta_grid: Optional[np.ndarray] = None,
        sigma_obs: float = 0.2,
        gp_lengthscale: float = 0.3,
        gp_signal_var: float = 1.0,
    ):
        self.W = window_batches * batch_size  # window in observations
        self.sigma2 = sigma_obs ** 2
        self.theta_grid = (
            theta_grid if theta_grid is not None
            else np.linspace(0.0, 3.0, 200)
        )
        self.ls = gp_lengthscale
        self.sv = gp_signal_var
        self.X_hist: List[float] = []
        self.Y_hist: List[float] = []
        self.current_theta = 1.5  # initial (before any data)

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        self.X_hist.extend(Xb.ravel().tolist())
        self.Y_hist.extend(Yb.ravel().tolist())
        # sliding window
        if len(self.X_hist) > self.W:
            self.X_hist = self.X_hist[-self.W:]
            self.Y_hist = self.Y_hist[-self.W:]

        n = len(self.X_hist)
        if n < 10:
            return  # not enough data yet

        X = np.array(self.X_hist).reshape(-1, 1)
        Y = np.array(self.Y_hist)

        # Covariance:  K = sv · exp(−||x−x'||²/(2 ls²))  +  σ² I
        dist_sq = (X - X.T) ** 2
        K = self.sv * np.exp(-0.5 * dist_sq / self.ls ** 2)
        K += self.sigma2 * np.eye(n) + 1e-8 * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)

        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        const = n * np.log(2.0 * np.pi)

        # Profile marginal log-likelihood for each θ (K is independent of θ)
        log_ml = np.empty(len(self.theta_grid))
        for i, th in enumerate(self.theta_grid):
            ys = np.sin(5.0 * th * X.ravel()) + 5.0 * X.ravel()
            r = Y - ys
            alpha = np.linalg.solve(L, r)  # L α = r  →  r^T K^{-1} r = α^T α
            log_ml[i] = -0.5 * (np.dot(alpha, alpha) + logdet + const)

        # Posterior (uniform prior on grid)
        w = np.exp(log_ml - logsumexp(log_ml))
        self.current_theta = float(np.sum(w * self.theta_grid))

    def mean_theta(self) -> float:
        return self.current_theta


# =================================================================
# Method Ours : R-BOCPD-PF-NoDiscrepancy  (wrapper)
# =================================================================
def run_rbocpd_pf(
    stream: ThetaScheduleDataStream,
    total_T: int,
    batch_size: int,
    scenario_name: str = "",
) -> np.ndarray:
    """Run the R-BOCPD-PF (restart BOCPD + particle filter, no discrepancy)."""

    def prior_sampler(N):
        return torch.rand(N, 1) * 3.0

    cfg = CalibrationConfig()
    cfg.bocpd.bocpd_mode = "restart"
    cfg.bocpd.use_restart = True
    cfg.model.use_discrepancy = False
    cfg.model.refit_delta_every_batch = False   # no y-prediction needed here; skip delta refit
    # dynamic attribute expected by restart_bocpd_debug_260115_gpytorch
    cfg.model.bocpd_use_discrepancy = False

    emulator = DeterministicSimulator(
        func=computer_model_torch,
        enable_autograd=True,
    )

    calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

    n_batches = total_T // batch_size
    theta_hist: List[float] = []
    pbar = tqdm(total=n_batches, desc=f"  Ours ({scenario_name})", unit="batch")
    total_obs = 0
    while total_obs < total_T:
        Xb, Yb = stream.next()
        calib.step_batch(Xb, Yb, verbose=False)
        mean_theta, _, _, _ = calib._aggregate_particles(0.9)
        theta_hist.append(float(mean_theta[0]))
        total_obs += batch_size
        n_experts = len(calib.bocpd.experts)
        pbar.set_postfix({"θ̂": f"{theta_hist[-1]:.3f}", "experts": n_experts})
        pbar.update(1)
    pbar.close()

    return np.array(theta_hist)


# =================================================================
# θ*(t) schedule factories
# =================================================================
def make_gradual_schedule(theta0: float = 1.2, slope: float = 0.001):
    """Gradual linear drift:  θ*(t) = θ0 + slope · t"""
    return lambda t: theta0 + slope * t


def make_sudden_schedule(seg_len: int, theta_values: List[float]):
    """Piece-wise constant θ*(t) with jumps at segment boundaries."""
    def schedule(t):
        seg = min(t // seg_len, len(theta_values) - 1)
        return theta_values[seg]
    return schedule


def make_mixed_schedule(
    seg_len: int,
    theta_bases: List[float],
    slopes: List[float],
):
    """Gradual drift within each segment + sudden jumps at boundaries."""
    def schedule(t):
        seg = min(t // seg_len, len(theta_bases) - 1)
        t_local = t - seg * seg_len
        return theta_bases[seg] + slopes[seg] * t_local
    return schedule


# =================================================================
# Plotting helper
# =================================================================
COLORS = {"BC": "#e74c3c", "DA": "#2980b9", "Ours": "#27ae60"}
MARKERS = {"BC": "o", "DA": "s", "Ours": "^"}


def _plot_scenario(
    ax,
    batches: np.ndarray,
    gt: np.ndarray,
    method_results: Dict[str, np.ndarray],
    scenario_name: str,
    cp_batch_indices: Optional[List[int]] = None,
):
    """Draw one subplot for a single scenario."""
    ax.plot(batches, gt, "k--", lw=2.0, label="Ground Truth", zorder=5)

    for label in ["BC", "DA", "Ours"]:
        if label not in method_results:
            continue
        theta_arr = method_results[label]
        ax.plot(
            batches, theta_arr,
            color=COLORS.get(label, "gray"),
            marker=MARKERS.get(label, "."),
            markersize=3,
            linewidth=1.3,
            alpha=0.85,
            label=label,
        )

    if cp_batch_indices is not None:
        for cp in cp_batch_indices:
            ax.axvline(x=cp, color="red", linestyle=":", alpha=0.45, lw=1.0)

    ax.set_title(scenario_name, fontsize=13)
    ax.set_xlabel("Batch Index", fontsize=11)
    ax.set_ylabel(r"$\theta$", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)


# =================================================================
# Main
# =================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Three-scenario comparison: BC / DA / Ours")
    parser.add_argument("--out_dir", type=str, default="figs/comparison_three_scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_T", type=int, default=1800)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    seed = args.seed
    total_T = args.total_T
    batch_size = 20
    n_batches = total_T // batch_size
    noise_sd = 0.2

    # ----------------------------------------------------------------
    # Build φ2 ↔ θ* mapping  (one-time, reused by all scenarios)
    # ----------------------------------------------------------------
    print("Building φ2 ↔ θ* inverse-oracle mapping ...")
    phi2_grid = np.linspace(3.0, 12.0, 300)
    theta_grid_oracle = np.linspace(0.0, 3.0, 600)
    phi2_of_theta, theta_star_vals = build_phi2_from_theta_star(phi2_grid, theta_grid_oracle)
    print(f"  Oracle θ* range: [{theta_star_vals.min():.4f}, {theta_star_vals.max():.4f}]")
    print("  done.\n")

    # ----------------------------------------------------------------
    # Define the three scenarios
    # ----------------------------------------------------------------
    seg_len = total_T // 3  # e.g. 600 obs per segment (= 30 batches at bs=20)

    scenarios = {
        "Gradual Drift": {
            "schedule": make_gradual_schedule(theta0=1.2, slope=0.001),
            "cp_batches": None,
        },
        "Sudden Jump": {
            "schedule": make_sudden_schedule(seg_len, [1.2, 2.0, 1.2]),
            "cp_batches": [seg_len // batch_size, 2 * seg_len // batch_size],
        },
        "Gradual + Sudden": {
            "schedule": make_mixed_schedule(
                seg_len,
                theta_bases=[1.0, 2.0, 1.2],
                slopes=[0.002, 0.0005, 0.001],
            ),
            "cp_batches": [seg_len // batch_size, 2 * seg_len // batch_size],
        },
    }

    # ----------------------------------------------------------------
    # Run each scenario  ×  each method
    # ----------------------------------------------------------------
    all_results: Dict[str, dict] = {}

    for scenario_name, scenario_cfg in scenarios.items():
        print(f"{'=' * 60}")
        print(f"Scenario : {scenario_name}")
        print(f"{'=' * 60}")

        theta_schedule = scenario_cfg["schedule"]

        # Ground truth θ* per batch
        gt_theta = np.array([theta_schedule(b * batch_size) for b in range(n_batches)])
        print(f"  θ* range in scenario: [{gt_theta.min():.4f}, {gt_theta.max():.4f}]")

        # ---- DA  (PF-NoDiscrepancy) ----
        t0 = timer()
        stream_da = ThetaScheduleDataStream(
            total_T, batch_size, noise_sd, theta_schedule, phi2_of_theta, seed=seed,
        )
        pf = PFNoDiscrepancy(n_particles=1024, sigma_obs=noise_sd, seed=seed + 1)
        da_theta: List[float] = []
        for _ in tqdm(range(n_batches), desc=f"  DA  ({scenario_name})", unit="batch"):
            Xb, Yb = stream_da.next()
            pf.update_batch(Xb.numpy(), Yb.numpy())
            da_theta.append(pf.mean_theta())
        da_theta_arr = np.array(da_theta)
        print(f"  DA  (PF-NoDisc)    finished in {timer() - t0:.1f}s")

        # ---- BC  (KOH Sliding Window) ----
        t0 = timer()
        stream_bc = ThetaScheduleDataStream(
            total_T, batch_size, noise_sd, theta_schedule, phi2_of_theta, seed=seed,
        )
        koh = KOHSlidingWindow(
            window_batches=20,
            batch_size=batch_size,
            sigma_obs=noise_sd,
        )
        bc_theta: List[float] = []
        for _ in tqdm(range(n_batches), desc=f"  BC  ({scenario_name})", unit="batch"):
            Xb, Yb = stream_bc.next()
            koh.update_batch(Xb.numpy(), Yb.numpy())
            bc_theta.append(koh.mean_theta())
        bc_theta_arr = np.array(bc_theta)
        print(f"  BC  (KOH window)   finished in {timer() - t0:.1f}s")

        # ---- Ours (R-BOCPD-PF-NoDiscrepancy) ----
        t0 = timer()
        stream_ours = ThetaScheduleDataStream(
            total_T, batch_size, noise_sd, theta_schedule, phi2_of_theta, seed=seed,
        )
        ours_theta_arr = run_rbocpd_pf(stream_ours, total_T, batch_size, scenario_name=scenario_name)
        print(f"  Ours (R-BOCPD-PF)  finished in {timer() - t0:.1f}s")

        all_results[scenario_name] = {
            "gt": gt_theta,
            "DA": da_theta_arr,
            "BC": bc_theta_arr,
            "Ours": ours_theta_arr,
            "cp_batches": scenario_cfg["cp_batches"],
        }

        # quick RMSE summary
        for lbl in ["DA", "BC", "Ours"]:
            rmse_val = np.sqrt(np.mean((all_results[scenario_name][lbl] - gt_theta) ** 2))
            print(f"    {lbl:>5s}  θ-RMSE = {rmse_val:.4f}")
        print()

    # ==================================================================
    # Figure 1 : BC + DA  vs  Ground Truth   (3 subplots side by side)
    # ==================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    for i, (scenario_name, res) in enumerate(all_results.items()):
        batches = np.arange(len(res["gt"]))
        _plot_scenario(
            axes1[i], batches, res["gt"],
            {"BC": res["BC"], "DA": res["DA"]},
            scenario_name,
            cp_batch_indices=res["cp_batches"],
        )
    fig1.suptitle("BC (KOH) and DA (PF-NoDisc) vs Ground Truth", fontsize=15, y=1.02)
    plt.tight_layout()
    fig1.savefig(os.path.join(out_dir, "figure1_BC_DA.pdf"), bbox_inches="tight")
    fig1.savefig(os.path.join(out_dir, "figure1_BC_DA.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Saved] Figure 1 → {out_dir}/figure1_BC_DA.{{pdf,png}}")

    # ==================================================================
    # Figure 2 : BC + DA + Ours  vs  Ground Truth  (3 subplots side by side)
    # ==================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, (scenario_name, res) in enumerate(all_results.items()):
        batches = np.arange(len(res["gt"]))
        _plot_scenario(
            axes2[i], batches, res["gt"],
            {"BC": res["BC"], "DA": res["DA"], "Ours": res["Ours"]},
            scenario_name,
            cp_batch_indices=res["cp_batches"],
        )
    fig2.suptitle(
        "BC (KOH), DA (PF-NoDisc), and Ours (R-BOCPD-PF) vs Ground Truth",
        fontsize=15, y=1.02,
    )
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "figure2_BC_DA_Ours.pdf"), bbox_inches="tight")
    fig2.savefig(os.path.join(out_dir, "figure2_BC_DA_Ours.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Saved] Figure 2 → {out_dir}/figure2_BC_DA_Ours.{{pdf,png}}")

    # Save raw results
    torch.save(all_results, os.path.join(out_dir, "all_results.pt"))
    print(f"[Saved] Results  → {out_dir}/all_results.pt")
    print("\nAll done!")


if __name__ == "__main__":
    main()

'''
python -m calib.run_comparison_three_scenarios [--out_dir DIR] [--seed 42] [--total_T 1800]
'''

