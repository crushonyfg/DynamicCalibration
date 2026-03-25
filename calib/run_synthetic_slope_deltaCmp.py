# =============================================================
# run_synthetic_slopecomp.py
# Gradual-drift slope comparison experiment
# =============================================================

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *
from .restart_bocpd_debug_260115_gpytorch import RollingStats
# from .restart_bocpd_ogp import (
#     BOCPD_OGP, OGPPFConfig,
#     RollingStats as OGPRollingStats,
#     make_grad_func_from_emulator,
# )
from .restart_bocpd_ogp_gpytorch import (
    BOCPD_OGP, OGPPFConfig, OGPParticleFilter,
    RollingStats as OGPRollingStats, make_fast_batched_grad_func,
)
from .configs import BOCPDConfig, ModelConfig
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

# -------------------------------------------------------------
# Simulator (Config2)
# -------------------------------------------------------------
def computer_model_config2_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    th = theta[:, [0]]
    xx = x[:, [0]]
    return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)


def computer_model_config2_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    return torch.sin(5.0 * theta[:, 0:1] * x[:, 0:1]) + 5.0 * x[:, 0:1]

from scipy.interpolate import interp1d

def build_phi2_of_theta_interp(theta_grid: np.ndarray):
    """
    Build interpolation phi2(theta) by inverting oracle_theta on a phi2 grid.
    This mirrors the logic in your slope synthetic script: ensure theta*(t) corresponds
    to a realizable physical phi2(t).

    Returns: callable phi2_of_theta(theta) -> float
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # choose a phi2 grid (wide enough)
    phi2_grid = np.linspace(2.0, 12.0, 400)
    phi_base = np.array([5.0, 0.0, 5.0], dtype=float)

    # map phi2 -> theta*(phi)
    theta_star_list = []
    for phi2 in phi2_grid:
        phi = phi_base.copy()
        phi[1] = float(phi2)
        th = oracle_theta(phi, theta_grid)
        theta_star_list.append(th)

    theta_star_arr = np.asarray(theta_star_list, dtype=float)

    # theta_star_arr should be monotone-ish; if not, sort by theta for safe inversion
    order = np.argsort(theta_star_arr)
    theta_sorted = theta_star_arr[order]
    phi2_sorted = phi2_grid[order]

    # Invert by interpolation
    f = interp1d(theta_sorted, phi2_sorted, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return lambda th: float(f(float(th)))

def build_phi2_from_theta_star(
    phi2_grid: np.ndarray,
    theta_grid: np.ndarray,
    a1: float = 5.0,
    a3: float = 5.0,
):
    """
    构造 φ2 = f(θ*) 的插值函数
    """

    theta_star_vals = []

    for phi2 in phi2_grid:
        phi = np.array([a1, phi2, a3])
        theta_star = oracle_theta(phi, theta_grid)
        theta_star_vals.append(theta_star)

    theta_star_vals = np.asarray(theta_star_vals)

    phi2_of_theta = interp1d(
        theta_star_vals,
        phi2_grid,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return phi2_of_theta, theta_star_vals
# -------------------------------------------------------------
# True physical system η(x; φ(t))
# -------------------------------------------------------------
def physical_system(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    φ = [a1, a2, a3]
    η(x) = a1 * x * cos(a2 * x) + a3 * x
    """
    x = x.reshape(-1)
    a1, a2, a3 = phi
    return a1 * x * np.cos(a2 * x) + a3 * x


# -------------------------------------------------------------
# Data stream with explicit slope drift
# -------------------------------------------------------------
class SlopeDriftDataStream:
    def __init__(
        self,
        total_T: int = 800,
        batch_size: int = 20,
        noise_sd: float = 0.2,
        slope: float = 0.002,
        phi0 = np.array([5.0, 5.0, 5.0]),
        seed: int = 0,
    ):
        self.T = total_T
        self.bs = batch_size
        self.noise_sd = noise_sd
        self.slope = slope
        self.phi0 = phi0
        self.rng = np.random.RandomState(seed)

        self.t = 0
        self.phi_history = []

    def true_phi(self, t: int) -> np.ndarray:
        phi = self.phi0.copy()
        phi[1] = self.phi0[1] + self.slope * t
        return phi

    def next(self):
        if self.t >= self.T:
            raise StopIteration

        # X = self.rng.rand(self.bs, 1)
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs     # 每个区间一个点
        X = u[:, None]
        # self.rng.shuffle(X)  # 可选

        phi_t = self.true_phi(self.t)
        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.t += self.bs

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

class ThetaDrivenSlopeDataStream:
    """
    Ground truth: θ*(t) 线性变化
    Physical parameter φ2(t) 由数值反推得到
    """

    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        theta0: float,
        theta_slope: float,
        phi2_of_theta,           # 上一步构造的插值函数
        phi_base = np.array([5.0, 0.0, 5.0]),
        seed: int = 0,
    ):
        self.T = total_T
        self.bs = batch_size
        self.noise_sd = noise_sd
        self.theta0 = theta0
        self.theta_slope = theta_slope
        self.phi_base = phi_base.copy()
        self.phi2_of_theta = phi2_of_theta

        self.rng = np.random.RandomState(seed)
        self.t = 0

        self.theta_star_history = []
        self.phi_history = []

    def true_theta_star(self, t: int) -> float:
        return self.theta0 + self.theta_slope * t

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
# -------------------------------------------------------------
# Oracle θ*(φ) via dense grid search
# -------------------------------------------------------------
def oracle_theta(phi: np.ndarray, grid: np.ndarray) -> float:
    """
    θ* = argmin || η(x;φ) - y_s(x,θ) ||^2
    """
    x = np.linspace(0, 1, 400).reshape(-1, 1)
    eta = physical_system(x, phi)

    errs = []
    for th in grid:
        ys = computer_model_config2_np(x, np.array([th]))
        errs.append(np.mean((eta - ys) ** 2))

    return grid[np.argmin(errs)]


# =====================================================================
# DA: PF (simple likelihood) + GP discrepancy for y-prediction only
# =====================================================================
class PFWithGPPrediction:
    """
    DA method.
    PF update: p(y|x,θ) = N(y; sim(x,θ), σ²)  — no GP in likelihood.
    y-prediction: y_pred = sim(x, θ_mean) + GP_discrepancy(x), with GP variance.
    GP is refitted every batch on residuals r = y - sim(x, θ_mean).
    """

    def __init__(self, sim_func_np, n_particles=1024,
                 theta_lo=0.0, theta_hi=3.0, sigma_obs=0.2,
                 resample_ess_ratio=0.5, theta_move_std=0.05,
                 window_size=80, gp_lengthscale=0.3, gp_signal_var=1.0,
                 seed=42):
        self.sim = sim_func_np
        self.N = n_particles
        self.lo, self.hi = theta_lo, theta_hi
        self.sigma2 = sigma_obs ** 2
        self.ess_ratio = resample_ess_ratio
        self.move_std = theta_move_std
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(self.lo, self.hi, size=self.N)
        self.logw = np.zeros(self.N) - np.log(self.N)

        self.W = window_size
        self.X_buf, self.Y_buf = [], []
        self.ls = gp_lengthscale
        self.sv = gp_signal_var

        self._gp_L = None
        self._gp_alpha = None
        self._gp_X = None

    def _normalize_logw(self):
        self.logw -= logsumexp(self.logw)

    def _ess(self):
        return 1.0 / np.sum(np.exp(self.logw) ** 2)

    def _systematic_resample(self):
        w = np.exp(self.logw)
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        idx = np.searchsorted(np.cumsum(w), positions, side="left")
        idx = np.clip(idx, 0, self.N - 1)
        self.theta = self.theta[idx]
        self.logw[:] = -np.log(self.N)

    def _rejuvenate(self):
        self.theta += self.rng.normal(0.0, self.move_std, size=self.N)
        self.theta = np.clip(self.theta, self.lo, self.hi)

    def mean_theta(self):
        return float(np.sum(np.exp(self.logw) * self.theta))

    def _fit_gp(self):
        X_all = np.concatenate(self.X_buf, axis=0)
        Y_all = np.concatenate(self.Y_buf, axis=0)
        if len(X_all) > self.W:
            X_all, Y_all = X_all[-self.W:], Y_all[-self.W:]
            self.X_buf, self.Y_buf = [X_all], [Y_all]
        n = len(X_all)
        if n < 3:
            self._gp_L = None
            return
        th_mean = self.mean_theta()
        sim_pred = self.sim(X_all, np.full((n, 1), th_mean))
        residuals = Y_all - sim_pred
        dist_sq = cdist(X_all, X_all, metric="sqeuclidean")
        K = self.sv * np.exp(-0.5 * dist_sq / self.ls ** 2) + self.sigma2 * np.eye(n) + 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                self._gp_L = None
                return
        self._gp_L = L
        self._gp_alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))
        self._gp_X = X_all

    def predict(self, X_new):
        n_new = X_new.shape[0]
        th_mean = self.mean_theta()
        sim_pred = self.sim(X_new, np.full((n_new, 1), th_mean))
        if self._gp_L is None or self._gp_X is None:
            return sim_pred, np.full(n_new, self.sigma2)
        k_star = self.sv * np.exp(-0.5 * cdist(X_new, self._gp_X, metric="sqeuclidean") / self.ls ** 2)
        gp_mu = k_star @ self._gp_alpha
        v = np.linalg.solve(self._gp_L, k_star.T)
        gp_var = np.maximum(self.sv + self.sigma2 - np.sum(v ** 2, axis=0), 1e-8)
        return sim_pred + gp_mu, gp_var

    def update_batch(self, Xb_np, Yb_np):
        self.X_buf.append(Xb_np.copy())
        self.Y_buf.append(Yb_np.copy())

        B = Xb_np.shape[0]
        X_rep = np.tile(Xb_np, (self.N, 1))
        th_rep = np.repeat(self.theta, B)[:, None]
        pred_all = self.sim(X_rep, th_rep).reshape(self.N, B)
        resid = Yb_np[None, :] - pred_all
        loglik = np.sum(
            -0.5 * (resid ** 2 / self.sigma2 + np.log(2 * np.pi * self.sigma2)),
            axis=1,
        )

        self.logw += loglik
        self._normalize_logw()
        if self._ess() < self.ess_ratio * self.N:
            self._systematic_resample()
            self._rejuvenate()

        self._fit_gp()


# =====================================================================
# BC: KOH Sliding Window
# =====================================================================
class KOHSlidingWindow:
    """
    BC method.
    KOH-style batch calibration: profile GP marginal log-likelihood
    over a theta grid, with a sliding window of observations.
    Also fits a prediction GP for computing RMSE / CRPS.
    """

    def __init__(self, sim_func_np, theta_grid, window_size=80,
                 sigma_obs=0.2, gp_lengthscale=0.3, gp_signal_var=1.0):
        self.sim = sim_func_np
        self.theta_grid = theta_grid
        self.W = window_size
        self.sigma2 = sigma_obs ** 2
        self.ls = gp_lengthscale
        self.sv = gp_signal_var
        self.X_buf, self.Y_buf = [], []
        self.current_theta = float(np.median(theta_grid))

        self._gp_L = None
        self._gp_alpha = None
        self._gp_X = None

    def _fit_prediction_gp(self, X_all, Y_all):
        n = len(X_all)
        if n < 3:
            self._gp_L = None
            return
        sim_pred = self.sim(X_all, np.full((n, 1), self.current_theta))
        residuals = Y_all - sim_pred
        dist_sq = cdist(X_all, X_all, metric="sqeuclidean")
        K = self.sv * np.exp(-0.5 * dist_sq / self.ls ** 2) + self.sigma2 * np.eye(n) + 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                self._gp_L = None
                return
        self._gp_L = L
        self._gp_alpha = np.linalg.solve(L.T, np.linalg.solve(L, residuals))
        self._gp_X = X_all

    def predict(self, X_new):
        n_new = X_new.shape[0]
        sim_pred = self.sim(X_new, np.full((n_new, 1), self.current_theta))
        if self._gp_L is None or self._gp_X is None:
            return sim_pred, np.full(n_new, self.sigma2)
        k_star = self.sv * np.exp(-0.5 * cdist(X_new, self._gp_X, metric="sqeuclidean") / self.ls ** 2)
        gp_mu = k_star @ self._gp_alpha
        v = np.linalg.solve(self._gp_L, k_star.T)
        gp_var = np.maximum(self.sv + self.sigma2 - np.sum(v ** 2, axis=0), 1e-8)
        return sim_pred + gp_mu, gp_var

    def update_batch(self, Xb_np, Yb_np):
        self.X_buf.append(Xb_np.copy())
        self.Y_buf.append(Yb_np.copy())
        X_all = np.concatenate(self.X_buf, axis=0)
        Y_all = np.concatenate(self.Y_buf, axis=0)
        if len(X_all) > self.W:
            X_all, Y_all = X_all[-self.W:], Y_all[-self.W:]
            self.X_buf, self.Y_buf = [X_all], [Y_all]
        n = len(X_all)
        if n < 5:
            return
        dist_sq = cdist(X_all, X_all, metric="sqeuclidean")
        K = self.sv * np.exp(-0.5 * dist_sq / self.ls ** 2) + self.sigma2 * np.eye(n) + 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        const = n * np.log(2.0 * np.pi)
        log_ml = np.empty(len(self.theta_grid))
        for i, th in enumerate(self.theta_grid):
            ys = self.sim(X_all, np.full((n, 1), th))
            r = Y_all - ys
            alpha = np.linalg.solve(L, r)
            log_ml[i] = -0.5 * (np.dot(alpha, alpha) + logdet + const)
        w = np.exp(log_ml - logsumexp(log_ml))
        self.current_theta = float(np.sum(w * self.theta_grid))
        self._fit_prediction_gp(X_all, Y_all)

    def mean_theta(self):
        return self.current_theta


# -------------------------------------------------------------
# Aggregate OGP-BOCPD particles across experts (mass-weighted)
# -------------------------------------------------------------
def _aggregate_ogp_particles(bocpd, quantile=0.9):
    experts = bocpd.experts
    if len(experts) == 0:
        return None, None, None, None

    d = experts[0].pf.theta.shape[1]
    device = experts[0].pf.theta.device
    dtype = experts[0].pf.theta.dtype

    mean = torch.zeros(d, dtype=dtype, device=device)
    cov = torch.zeros(d, d, dtype=dtype, device=device)
    theta_list, weight_list = [], []

    for e in experts:
        w_e = math.exp(e.log_mass)
        w = e.pf.weights()
        th = e.pf.theta
        m = (w[:, None] * th).sum(0)
        C = ((th - m) * w[:, None]).T @ (th - m)
        mean = mean + w_e * m
        cov = cov + w_e * (C + (m - mean)[:, None] @ (m - mean)[None, :])
        theta_list.append(th)
        weight_list.append(w_e * w)

    theta_all = torch.cat(theta_list, dim=0)
    weight_all = torch.cat(weight_list, dim=0)
    weight_all = weight_all / weight_all.sum()
    var = torch.diag(cov)

    def weighted_quantile_1d(x, w, q):
        idx = torch.argsort(x)
        x, w = x[idx], w[idx]
        cw = torch.cumsum(w, dim=0)
        return x[cw >= q][0]

    alpha = (1.0 - quantile) / 2.0
    lo = torch.zeros(d, dtype=dtype, device=device)
    hi = torch.zeros(d, dtype=dtype, device=device)
    for j in range(d):
        lo[j] = weighted_quantile_1d(theta_all[:, j], weight_all, alpha)
        hi[j] = weighted_quantile_1d(theta_all[:, j], weight_all, 1.0 - alpha)

    return mean, var, lo, hi


# -------------------------------------------------------------
# Run one slope experiment
# -------------------------------------------------------------
def run_one_slope(
    slope: float,
    methods: Dict,
    total_T: int = 600,
    batch_size: int = 20,
    seed: int = 123,
    phi2_of_theta: callable = None,
    mode: int = 0,#0: slope origin, 1: slope inverse, 2: sudden origin
):
    print(f"\n=== Running slope={slope:.4f} ===")

    # if mode == 0:
    #     stream = SlopeDriftDataStream(
    #         total_T=total_T,
    #         batch_size=batch_size,
    #         slope=slope,
    #         seed=seed,
    #     )
    # elif mode == 1:
    #     stream = ThetaDrivenSlopeDataStream(
    #     total_T=total_T,
    #     batch_size=batch_size,
    #     noise_sd=0.2,
    #     theta0=1.6,                 # 起始 θ*
    #     theta_slope=slope,          # 你想测试的 drift
    #     phi2_of_theta=phi2_of_theta,
    #     seed=seed,
    # )

    # theta prior
    def prior_sampler(N):
        return torch.rand(N, 1) * 3.0
    def prior_sampler1(N, theta_anchor=None, sigma_local=0.2, p_global=0.2):
        """
        theta_anchor: 上一个 anchor / expert 的 posterior mean
        """
        N_global = int(p_global * N)
        N_local  = N - N_global

        samples = []

        # global prior (handle sudden change)
        if N_global > 0:
            samples.append(torch.rand(N_global, 1) * 3.0)

        # local prior (handle gradual drift)
        if theta_anchor is not None:
            local = theta_anchor + sigma_local * torch.randn(N_local, 1)
            local = torch.clamp(local, 0.0, 3.0)
            samples.append(local)
        else:
            samples.append(torch.rand(N_local, 1) * 3.0)

        return torch.cat(samples, dim=0)

    use_sampler1 = False
    # oracle
    theta_grid = np.linspace(0, 3, 400)

    results = {}

    for name, meta in methods.items():
        print(f"  -> {name}")
        t0 = time()

        theta_hist, rmse_hist, crps_hist = [], [], []
        total_obs = 0
        others_hist = []
        report_sub_hist = []
        theta_var_hist = []

        dll_hist = []
        mu_hist = []
        sig_hist = []
        h_hist = []
        odds_hist = []
        anchor_rl_hist = []
        cand_rl_hist = []

        top0_particles_hist = []
        restart_mode_hist = []

        if mode == 0:
            stream2 = SlopeDriftDataStream(
                total_T=total_T,
                batch_size=batch_size,
                slope=slope,
                seed=seed,
            )
        elif mode == 1:
            stream2 = ThetaDrivenSlopeDataStream(
                total_T=total_T,
                batch_size=batch_size,
                noise_sd=0.2,
                theta0=1.5,                 # 起始 θ*
                theta_slope=slope,          # 你想测试的 drift
                phi2_of_theta=phi2_of_theta,
                seed=seed,
            )
        

        # ---------- R-BOCPD-OGP ----------
        if name == "R-BOCPD-PF-OGP":
            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            grad_func = make_fast_batched_grad_func(
                computer_model_config2_torch, device="cuda", dtype=torch.float64,
            )

            ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=(0.0, 1.0),
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([3.0]),
                particle_chunk_size=256,  # 新增：控制 GPU 内存
            )

            
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            roll = OGPRollingStats(window=50)

            # bocpd = BOCPD_OGP(
            #     config=bocpd_cfg,
            #     ogp_pf_cfg=ogp_cfg,
            #     grad_func=grad_func,
            # )
            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,  # 注意：参数名变了
                device="cuda",                # GPU
            )

            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=f"{name}", unit="obs")
            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()

                ogp_dev = bocpd.device
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)

                if total_obs > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(batch_size, device=ogp_dev, dtype=torch.float64)
                    mix_var = torch.zeros(batch_size, device=ogp_dev, dtype=torch.float64)
                    Z = 0.0
                    for e in bocpd.experts:
                        w_e = math.exp(e.log_mass)
                        e_X_hist = e.X_hist if e.X_hist.numel() > 0 else None
                        e_y_hist = e.y_hist if e.y_hist.numel() > 0 else None
                        mu_mix_e, var_mix_e = e.pf.predict_batch(
                            Xb64, e_X_hist, e_y_hist,
                            emulator, model_cfg.rho, model_cfg.sigma_eps,
                        )
                        mix_mu += w_e * mu_mix_e
                        mix_var += w_e * var_mix_e
                        Z += w_e
                    mix_mu = mix_mu / max(Z, 1e-12)
                    mix_var = mix_var / max(Z, 1e-12)

                    mix_mu_cpu = mix_mu.cpu()
                    mix_var_cpu = mix_var.cpu()
                    Yb_cpu = Yb64.cpu()
                    rmse_hist.append(float(torch.sqrt(((mix_mu_cpu - Yb_cpu) ** 2).mean())))
                    crps = crps_gaussian(mix_mu_cpu, mix_var_cpu, Yb_cpu).mean()
                    crps_hist.append(crps.item())

                rec = bocpd.update_batch(
                    Xb64, Yb64, emulator, model_cfg, None, prior_sampler,
                    verbose=False,
                )

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mu_hat = roll.mean()
                sig_hat = roll.std()
                h_log = rec.get("h_log", None)
                log_odds = rec.get("log_odds_mass", None)

                dll_hist.append(dll)
                mu_hist.append(mu_hat)
                sig_hist.append(sig_hat)
                h_hist.append(h_log)
                odds_hist.append(log_odds)

                anchor_rl_hist.append(rec.get("anchor_rl", None))
                cand_rl_hist.append(rec.get("cand_rl", None))

                mean_theta, var_theta, lo_theta, hi_theta = _aggregate_ogp_particles(
                    bocpd, 0.9,
                )
                theta_hist.append(float(mean_theta[0]))

                batch_particles = []
                batch_weights = []
                batch_logmass = []
                for e in bocpd.experts:
                    batch_particles.append(e.pf.theta.squeeze(-1).detach().cpu())
                    batch_weights.append(e.pf.weights().detach().cpu())
                    batch_logmass.append(float(e.log_mass))

                top0_particles_hist.append(dict(
                    particles=batch_particles,
                    weights=batch_weights,
                    log_mass=torch.tensor(batch_logmass),
                ))

                others_hist.append({
                    "did_restart": rec["did_restart"],
                    "var": float(var_theta[0]),
                    "lo": float(lo_theta[0]),
                    "hi": float(hi_theta[0]),
                    "pf_info": rec["pf_diags"],
                })

                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- Standalone PF-OGP (no BOCPD) ----------
        elif name == "PF-OGP":
            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            pf_grad_func = make_fast_batched_grad_func(
                computer_model_config2_torch, device="cuda", dtype=torch.float64,
            )
            pf_ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=(0.0, 1.0),
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([3.0]),
                theta_move_std=0.02,
                particle_chunk_size=256,
            )
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            ogp_dev = "cuda"

            pf = OGPParticleFilter(
                ogp_cfg=pf_ogp_cfg,
                prior_sampler=prior_sampler,
                batched_grad_func=pf_grad_func,
                device=ogp_dev,
                dtype=torch.float64,
            )

            pf_X_hist = torch.empty(
                0, 1, dtype=torch.float64, device=ogp_dev,
            )
            pf_y_hist = torch.empty(0, dtype=torch.float64, device=ogp_dev)
            pf_ogp_max_hist = 200

            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream2.next()
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)

                if total_obs > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(
                        Xb64, pf_Xh, pf_yh,
                        emulator, pf_model_cfg.rho, pf_model_cfg.sigma_eps,
                    )
                    rmse_hist.append(
                        float(torch.sqrt(((mu_mix.cpu() - Yb64.cpu()) ** 2).mean()))
                    )
                    crps = crps_gaussian(mu_mix.cpu(), var_mix.cpu(), Yb64.cpu()).mean()
                    crps_hist.append(crps.item())

                pf.step_batch(
                    Xb64, Yb64,
                    pf_X_hist if pf_X_hist.numel() > 0 else None,
                    pf_y_hist if pf_y_hist.numel() > 0 else None,
                    emulator,
                    pf_model_cfg.rho,
                    pf_model_cfg.sigma_eps,
                )

                if pf_X_hist.numel() == 0:
                    pf_X_hist = Xb64.clone()
                    pf_y_hist = Yb64.clone()
                else:
                    pf_X_hist = torch.cat([pf_X_hist, Xb64], dim=0)
                    pf_y_hist = torch.cat([pf_y_hist, Yb64], dim=0)
                if pf_X_hist.shape[0] > pf_ogp_max_hist:
                    pf_X_hist = pf_X_hist[-pf_ogp_max_hist:]
                    pf_y_hist = pf_y_hist[-pf_ogp_max_hist:]

                w = pf.weights().view(-1, 1)
                mean_theta = (w * pf.theta).sum(dim=0)
                theta_hist.append(float(mean_theta[0]))

                top0_particles_hist.append(dict(
                    particles=[pf.theta.squeeze(-1).detach().cpu()],
                    weights=[pf.weights().detach().cpu()],
                    log_mass=torch.tensor([0.0]),
                ))

                others_hist.append({
                    "did_restart": False,
                    "var": float(
                        (w * (pf.theta - mean_theta).pow(2)).sum(dim=0)[0]
                    ),
                })

                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- BOCPD ----------
        elif meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True
            # Keep backward compatibility by default; only enable hybrid when requested in methods.
            cfg.bocpd.restart_impl = meta.get("restart_impl", "debug_260115")
            cfg.bocpd.hybrid_partial_restart = bool(meta.get("use_dual_restart", meta.get("hybrid_partial_restart", True)))
            cfg.bocpd.hybrid_tau_delta = float(meta.get("hybrid_tau_delta", 0.05))
            cfg.bocpd.hybrid_tau_theta = float(meta.get("hybrid_tau_theta", 0.05))
            cfg.bocpd.hybrid_tau_full = float(meta.get("hybrid_tau_full", 0.05))
            cfg.bocpd.hybrid_delta_share_rho = float(meta.get("hybrid_delta_share_rho", 0.75))
            cfg.bocpd.hybrid_pf_sigma_mode = str(meta.get("hybrid_pf_sigma_mode", "fixed"))
            cfg.bocpd.hybrid_sigma_delta_alpha = float(meta.get("hybrid_sigma_delta_alpha", 1.0))
            cfg.bocpd.hybrid_sigma_ema_beta = float(meta.get("hybrid_sigma_ema_beta", 0.98))
            cfg.bocpd.hybrid_sigma_min = float(meta.get("hybrid_sigma_min", 1e-4))
            cfg.bocpd.hybrid_sigma_max = float(meta.get("hybrid_sigma_max", 10.0))
            cfg.bocpd.use_cusum = bool(meta.get("use_cusum", False))
            cfg.bocpd.cusum_threshold = float(meta.get("cusum_threshold", 10.0))
            cfg.bocpd.cusum_recent_obs = int(meta.get("cusum_recent_obs", 20))
            cfg.bocpd.cusum_cov_eps = float(meta.get("cusum_cov_eps", 1e-6))
            cfg.bocpd.cusum_mode = str(meta.get("cusum_mode", "cumulative"))
            cfg.bocpd.standardized_gate_threshold = float(meta.get("standardized_gate_threshold", 3.0))
            cfg.bocpd.standardized_gate_consecutive = int(meta.get("standardized_gate_consecutive", 1))
            roll = RollingStats(window=50)

            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]
                cfg.model.bocpd_use_discrepancy = meta.get("bocpd_use_discrepancy", cfg.model.use_discrepancy)

            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )

            if use_sampler1:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler1)
            else:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()

                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    pred_comp = calib.predict_complete(Xb, Yb)
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    # print(name, total_obs, report_hist[-1])
                    rmse_hist.append(
                        float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()))
                    )
                    crps = crps_gaussian(pred["mu"], pred["var"], Yb).mean()
                    # print(crps)
                    crps_hist.append(crps.item())
                    # rmse_hist.append(
                    #     float(torch.sqrt(((pred["mu_sim"] - Yb) ** 2).mean()))
                    # )

                rec = calib.step_batch(Xb, Yb, verbose=False)
                rm = rec.get("restart_mode", None)
                if rm is None:
                    rm = "full" if bool(rec.get("did_restart", False)) else "none"
                restart_mode_hist.append(rm)

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mu_hat = roll.mean()
                sig_hat = roll.std()
                h_log = rec.get("h_log", None)
                log_odds = rec.get("log_odds_mass", None)
                # print("debug: dll, mu_hat, sig_hat, h_log, log_odds",dll, mu_hat, sig_hat, h_log, log_odds)

                dll_hist.append(dll)
                mu_hist.append(mu_hat)
                sig_hist.append(sig_hat)
                h_hist.append(h_log)
                odds_hist.append(log_odds)

                anchor_rl_hist.append(rec.get("anchor_rl", None))
                cand_rl_hist.append(rec.get("cand_rl", None))

                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
                theta_hist.append(float(mean_theta[0]))

                experts = calib.bocpd.experts

                batch_particles = []
                batch_weights = []
                batch_logmass = []

                for e in experts:
                    # particles
                    particles = e.pf.particles.theta          # (N,1)
                    particles_1d = particles.squeeze(-1).detach().cpu()

                    # weights
                    pw = e.pf.particles.weights()             # (N,)
                    pw_1d = pw.squeeze(-1).detach().cpu()

                    # log mass
                    log_mass = float(e.log_mass)

                    batch_particles.append(particles_1d)
                    batch_weights.append(pw_1d)
                    batch_logmass.append(log_mass)

                batch_dict = dict(
                    particles=batch_particles,      # list length E
                    weights=batch_weights,          # list length E
                    log_mass=torch.tensor(batch_logmass)  # (E,)
                )

                top0_particles_hist.append(batch_dict)

                ess_gini_info = []
                for ei, e in enumerate(calib.bocpd.experts):
                    ps = e.pf.particles
                    unique_ratio = float(ps.unique_ratio())
                    entropy_1d_histogram = float(ps.entropy_1d_histogram())
                    # print(ei, unique_ratio, entropy_1d_histogram)
                    ess_gini_info.append({"expert_id": ei, "unique_ratio": unique_ratio, "entropy_1d_histogram": entropy_1d_histogram})
                others_hist.append({"did_restart": rec["did_restart"],"var": float(var_theta[0]), "lo": float(lo_theta[0]), "hi": float(hi_theta[0]), "pf_info": rec["pf_diags"], "report_sub_hist": report_sub_hist, "pf_health_info": ess_gini_info})

                if use_sampler1:
                    calib.theta_anchor = mean_theta[0]

                total_obs += batch_size

        # ---------- BPC ----------
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
                crps_sim = None
                if X_hist is None:
                    X_hist, y_hist = Xb.numpy(), Yb.numpy()
                else:
                    X_hist = np.concatenate([X_hist, Xb.numpy()], axis=0)
                    y_hist = np.concatenate([y_hist, Yb.numpy()], axis=0)
                if X_hist.shape[0] >= W:
                    X_hist = X_hist[-W:]
                    y_hist = y_hist[-W:]
                # X_hist.append(Xb.numpy())
                # y_hist.append(Yb.numpy())
                if total_obs > 0 and bpc is not None:
                    # mu_np, var_np = bpc.predict_sim(Xb.detach().cpu().numpy())
                    mu_np, var_np = bpc.predict(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device), torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device) 
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(crps_sim.mean().item())
                    # print("bpc crps sim:", crps_sim)

                X_all, y_all = X_hist, y_hist

                bpc = BayesianProjectedCalibration(
                    theta_lo=np.array([0.0]),
                    theta_hi=np.array([3.0]),
                    noise_var=0.04,
                    y_sim=computer_model_config2_np,
                )

                X_grid = np.linspace(0, 1, 300).reshape(-1, 1)
                bpc.fit(X_all, y_all, X_grid, n_eta_draws=500, n_restart=10, gp_fit_iters=200)

                theta_hist.append(float(bpc.theta_mean[0]))
                # print("bpc theta var:", bpc.theta_var[0])
                entropy_info = bpc.entropy_theta()
                # print("bpc theta entropy:", entropy_info)
                total_obs += batch_size
                others_hist.append({"var": float(bpc.theta_var[0]), "entropy": entropy_info, "crps_sim": crps_sim})

                theta_samples_bpc = torch.tensor(bpc.theta_samples).squeeze(-1)
                top0_particles_hist.append(theta_samples_bpc)
                batch_dict = dict(
                    particles=[theta_samples_bpc],      # list length E
                    weights=None,          # list length E
                    log_mass=torch.tensor([0.0])  # (E,)
                )

                top0_particles_hist.append(batch_dict)

        # ---------- BPC + BOCPD ----------
        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(
                theta_lo=np.array([0.0]),
                theta_hi=np.array([3.0]),
                noise_var=0.04,
                y_sim=computer_model_config2_np,
                X_grid=np.linspace(0, 1, 300).reshape(-1, 1),
            )

            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
                crps_sim = None
                if total_obs > 0:
                    # mu, var = calib.predict_sim(Xb)
                    mu, var = calib.predict(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device), torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(crps_sim.mean().item())
                    # print("bocpd-bpc crps sim:", crps_sim)
                info = calib.step_batch(Xb.detach().cpu().numpy(), Yb.detach().cpu().numpy())

                masses = np.asarray(info["masses"])
                thetas = np.asarray(info["theta_means"])

                # if masses.sum() > 0:
                #     w = masses / masses.sum()
                #     theta_hat = float((w * thetas[:, 0]).sum())
                # else:
                #     theta_hat = np.nan

                # theta_hist.append(theta_hat)
                total_obs += batch_size

                theta_mean, theta_var, theta_lo, theta_hi = calib._aggregate_particles(0.9)
                theta_hist.append(float(theta_mean[0]))
                # print("bocpd-bpc theta var:", theta_var[0])
                others_hist.append({"did_restart": info["did_restart"], "var": theta_var[0], "lo": theta_lo, "hi": theta_hi, "crps_sim": crps_sim})

                batch_particles = []
                batch_weights = []
                batch_logmass = []
                for e in calib.experts:
                    particles = torch.tensor(e.bpc.theta_samples).squeeze(-1)
                    batch_logmass.append(e.logw)          # (N,1)
                    batch_particles.append(particles)

                batch_dict = dict(
                    particles=batch_particles,      # list length E
                    weights=batch_weights,          # list length E
                    log_mass=torch.tensor(batch_logmass)  # (E,)
                )

                top0_particles_hist.append(batch_dict)
                # print(theta_mean, theta_var[0], theta_lo, theta_hi)

        # ---------- DA (PF + GP prediction) ----------
        elif name == "DA":
            da = PFWithGPPrediction(
                sim_func_np=computer_model_config2_np,
                n_particles=1024, theta_lo=0.0, theta_hi=3.0,
                sigma_obs=0.2, resample_ess_ratio=0.5,
                theta_move_std=0.05, window_size=80,
                gp_lengthscale=0.3, gp_signal_var=1.0, seed=seed,
            )
            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream2.next()
                Xb_np, Yb_np = Xb.numpy(), Yb.numpy()

                if total_obs > 0:
                    mu_pred, var_pred = da.predict(Xb_np)
                    mu_t = torch.tensor(mu_pred, dtype=Yb.dtype)
                    var_t = torch.tensor(var_pred, dtype=Yb.dtype)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps = crps_gaussian(mu_t, var_t, Yb).mean()
                    crps_hist.append(crps.item())

                da.update_batch(Xb_np, Yb_np)
                theta_hist.append(da.mean_theta())

                top0_particles_hist.append(dict(
                    particles=[torch.tensor(da.theta.copy())],
                    weights=[torch.tensor(np.exp(da.logw.copy()))],
                    log_mass=torch.tensor([0.0]),
                ))
                others_hist.append({"did_restart": False, "var": 0.0})
                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- BC (KOH Sliding Window) ----------
        elif name == "BC":
            bc_theta_grid = np.linspace(0.0, 3.0, 200)
            bc = KOHSlidingWindow(
                sim_func_np=computer_model_config2_np,
                theta_grid=bc_theta_grid, window_size=80,
                sigma_obs=0.2, gp_lengthscale=0.3, gp_signal_var=1.0,
            )
            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream2.next()
                Xb_np, Yb_np = Xb.numpy(), Yb.numpy()

                if total_obs > 0:
                    mu_pred, var_pred = bc.predict(Xb_np)
                    mu_t = torch.tensor(mu_pred, dtype=Yb.dtype)
                    var_t = torch.tensor(var_pred, dtype=Yb.dtype)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps = crps_gaussian(mu_t, var_t, Yb).mean()
                    crps_hist.append(crps.item())

                bc.update_batch(Xb_np, Yb_np)
                theta_hist.append(bc.mean_theta())

                top0_particles_hist.append(dict(
                    particles=[], weights=[], log_mass=torch.tensor([0.0]),
                ))
                others_hist.append({"did_restart": False, "var": 0.0})
                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- oracle ----------
        phi_hist = stream2.phi_history[: len(theta_hist)]
        oracle_hist = [
            oracle_theta(phi, theta_grid) for phi in phi_hist
        ]
        if meta["type"] == "bocpd":
            results[name] = dict(
                    theta=np.array(theta_hist),
                    theta_oracle=np.array(oracle_hist),
                    others=others_hist,
                    rmse=np.array(rmse_hist),
                    top0_particles_hist=top0_particles_hist,
                    seed=seed,
                    batch_size=batch_size,
                    slope=slope,
                    mode=mode,
                    oracle_hist=oracle_hist,
                    phi_hist=phi_hist,
                    delta_ll_hist=np.array(dll_hist),
                    mu_hat_hist=np.array(mu_hist),
                    sigma_hat_hist=np.array(sig_hist),
                    h_log_hist=np.array(h_hist),
                    log_odds_hist=np.array(odds_hist),
                    anchor_rl_hist=np.array(anchor_rl_hist),
                    cand_rl_hist=np.array(cand_rl_hist),
                    crps_hist=np.array(crps_hist),
                    restart_mode_hist=restart_mode_hist,
                )
        else:
            results[name] = dict(
                    theta=np.array(theta_hist),
                    theta_oracle=np.array(oracle_hist),
                    others=others_hist,
                    rmse=np.array(rmse_hist),
                    top0_particles_hist=top0_particles_hist,
                    seed=seed,
                    batch_size=batch_size,
                    slope=slope,
                    mode=mode,
                    oracle_hist=oracle_hist,
                    phi_hist=phi_hist,
                    crps_hist=np.array(crps_hist),
                )

        if mode == 1:
            results[name]["theta_star_true"] = np.array(stream2.theta_star_history)

        print(f"     done in {time() - t0:.1f}s")

    K = len(theta_hist)
    phi_hist = [stream2.true_phi(k*batch_size) for k in range(K)]
    oracle_hist = [oracle_theta(phi, theta_grid) for phi in phi_hist]

    return results, phi_hist, oracle_hist


# -------------------------------------------------------------
# Main: multiple slopes
# -------------------------------------------------------------
def main():
    # seeds = [0, 123, 456, 789]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/slope_deltaCmp_v2")
    args = parser.parse_args()

    if args.debug:
        seeds = [456]
        batch_sizes = [10]
        slopes = [0.001]
    else:
        seeds = [101, 202, 303, 404, 505]
        batch_sizes = [20]
        slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
    # batch_sizes = [20]
    # slopes = [0.003]
    store_dir = args.out_dir
    import os
    os.makedirs(store_dir, exist_ok=True)

    mode = 1
    if mode == 1:
        phi2_grid = np.linspace(3.0, 12.0, 300)
        theta_grid = np.linspace(0.0, 3.0, 600)

        phi2_of_theta, _ = build_phi2_from_theta_star(
            phi2_grid=phi2_grid,
            theta_grid=theta_grid,
        )
    else:
        phi2_of_theta = None

    methods = {
        # "DA": dict(type="da"),
        # "BC": dict(type="bc"),
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        # "R-BOCPD-PF-OGP": dict(type="bocpd", mode="restart"),
        # "PF-OGP": dict(type="pf_ogp"),
        # "BOCPD-PF": dict(type="bocpd", mode="standard"),
        # "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True, bocpd_use_discrepancy=True),
        # "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
        # "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
        # "R-BOCPD-PF-halfdiscrepancy-hybrid": dict(
        #     type="bocpd",
        #     mode="restart",
        #     use_discrepancy=False,
        #     bocpd_use_discrepancy=True,
        #     restart_impl="hybrid_260319",
        #     hybrid_partial_restart=True,
        #     hybrid_tau_delta=0.05,
        #     hybrid_tau_theta=0.05,
        # ),
        "RBOCPD_half_STDGate": dict(
            type="bocpd",
            mode="restart",
            use_discrepancy=False,
            bocpd_use_discrepancy=True,
            restart_impl="rolled_cusum_260324",
            use_dual_restart=False,
            use_cusum=True,
            cusum_mode="standardized_gate",
            standardized_gate_threshold=3.0,
            standardized_gate_consecutive=1,
            cusum_recent_obs=20,
            hybrid_tau_delta=0.05,
            hybrid_tau_theta=0.05,
        ),
    }

    all_results = {}
    import itertools

    for s,batch_size,seed in itertools.product(slopes, batch_sizes, seeds):
        res, phi_hist, oracle_hist = run_one_slope(s, methods, batch_size=batch_size, seed=seed, phi2_of_theta=phi2_of_theta, mode=mode)
        all_results[(s,batch_size,seed)] = res

        # ---------- plot ----------
        plt.figure(figsize=(10, 5))
        for name, d in res.items():
            plt.plot(d["theta"], label=name)
        plt.plot(d["theta_oracle"], "k--", lw=2, label="oracle θ*")
        plt.title(f"Theta tracking (slope={s})")
        plt.xlabel("batch index")
        plt.ylabel("theta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_theta.png", dpi=300)
        plt.close()

        # np.savez(
        #     f"slope_{s}_results.npz",
        #     **{f"{k}_theta": v["theta"] for k, v in res.items()},
        #     oracle_theta=res["Standard-BOCPD"]["theta_oracle"],
        # )
        torch.save(res, f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_results.pt")
        torch.save(dict(phi_hist=phi_hist, oracle_hist=oracle_hist), f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_phi_oracle_hist.pt")

    # ========== 收集所有组合的 metrics ==========
    all_metrics = []
    restart_mode_rows = []

    for s, batch_size, seed in itertools.product(slopes, batch_sizes, seeds):
        # 从 all_results 获取对应的 res
        res = all_results[(s, batch_size, seed)]  # 注意：当前代码中 all_results[s] 会被覆盖，需要改为 all_results[(s, batch_size, seed)]
        
        for method_name, data in res.items():
            # 计算 theta_rmse: sqrt(mean((theta_pred - theta_oracle)^2))
            theta_rmse = np.sqrt(np.mean((data["theta"] - data["theta_oracle"]) ** 2))
            
            # y_rmse 已经存在于 data["rmse"] 中
            y_rmse_mean = np.mean(data["rmse"])
            
            # y_crps 已经存在于 data["crps_hist"] 中
            y_crps_mean = np.mean(data["crps_hist"])
            
            all_metrics.append({
                "method": method_name,
                "slope": s,
                "batch_size": batch_size,
                "seed": seed,
                "theta_rmse": theta_rmse,
                "y_rmse": y_rmse_mean,
                "y_crps": y_crps_mean,
            })

            if "restart_mode_hist" in data:
                rm = data["restart_mode_hist"]
                n = max(len(rm), 1)
                n_none = sum(1 for v in rm if v == "none")
                n_delta = sum(1 for v in rm if v == "delta_only")
                n_full = sum(1 for v in rm if v not in ("none", "delta_only"))
                restart_mode_rows.append({
                    "method": method_name,
                    "slope": s,
                    "batch_size": batch_size,
                    "seed": seed,
                    "n_steps": len(rm),
                    "none_ratio": n_none / n,
                    "delta_only_ratio": n_delta / n,
                    "full_ratio": n_full / n,
                    "n_none": n_none,
                    "n_delta_only": n_delta,
                    "n_full": n_full,
                })

    # 转换为 DataFrame 并保存
    import pandas as pd
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(f"{store_dir}/all_metrics.csv", index=False)
    df_metrics.to_excel(f"{store_dir}/all_metrics.xlsx", index=False)

    if len(restart_mode_rows) > 0:
        df_restart = pd.DataFrame(restart_mode_rows)
        df_restart.to_csv(f"{store_dir}/restart_mode_stats.csv", index=False)
        df_restart.to_excel(f"{store_dir}/restart_mode_stats.xlsx", index=False)

        # aggregated plot: by method and slope, averaged across seeds/batch sizes
        df_plot = (
            df_restart.groupby(["method", "slope"], as_index=False)[
                ["none_ratio", "delta_only_ratio", "full_ratio"]
            ]
            .mean()
            .sort_values(["method", "slope"])
        )
        for method in df_plot["method"].unique():
            sub = df_plot[df_plot["method"] == method].copy()
            x = np.arange(len(sub))
            plt.figure(figsize=(9, 4.8))
            plt.stackplot(
                x,
                sub["none_ratio"].values,
                sub["delta_only_ratio"].values,
                sub["full_ratio"].values,
                labels=["none", "delta_only", "full"],
                alpha=0.9,
            )
            plt.xticks(x, [f"{v:.4g}" for v in sub["slope"].values], rotation=0)
            plt.ylim(0.0, 1.0)
            plt.xlabel("slope")
            plt.ylabel("ratio")
            plt.title(f"Restart mode ratio vs slope ({method})")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"{store_dir}/restart_mode_ratio_{method}.png", dpi=300)
            plt.close()

    # ========== 打印每个 method 的平均 metrics ==========
    print("\n" + "="*70)
    print("Average Metrics Across All Combinations (slopes × batch_sizes × seeds):")
    print("="*70)

    grouped = df_metrics.groupby("method").agg({
        "theta_rmse": ["mean", "std"],
        "y_rmse": ["mean", "std"],
        "y_crps": ["mean", "std"],
    })

    for method in df_metrics["method"].unique():
        print(f"\n{method}:")
        stats = grouped.loc[method]
        print(f"  theta_rmse: {stats[('theta_rmse', 'mean')]:.6f} ± {stats[('theta_rmse', 'std')]:.6f}")
        print(f"  y_rmse:     {stats[('y_rmse', 'mean')]:.6f} ± {stats[('y_rmse', 'std')]:.6f}")
        print(f"  y_crps:     {stats[('y_crps', 'mean')]:.6f} ± {stats[('y_crps', 'std')]:.6f}")

    print("\n" + "="*70)

    print("All slope experiments finished.")


if __name__ == "__main__":
    main()

