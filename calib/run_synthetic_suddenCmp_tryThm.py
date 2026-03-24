# =============================================================
# run_synthetic_suddencomp.py
# Sudden-change (3 changepoints) magnitude/frequency grid experiment
# =============================================================

import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from time import time
import itertools

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig, BOCPDConfig, ModelConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *  # StandardBOCPD_BPC
from .restart_bocpd_debug_260115_gpytorch import RollingStats
from .restart_bocpd_ogp_gpytorch import (
    BOCPD_OGP, OGPPFConfig, OGPParticleFilter,
    RollingStats as OGPRollingStats,
    make_fast_batched_grad_func,
)
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


# -------------------------------------------------------------
# True physical system η(x; φ)
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
# Data stream with 3 sudden changepoints
# -------------------------------------------------------------
class SuddenChangeDataStream:
    """
    total_T observations, generated in batches.
    changepoints are in observation-time units (same as t counter).
    """

    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        cp_times: List[int],
        phi_segments: List[np.ndarray],  # length = len(cp_times)+1
        seed: int,
    ):
        assert len(phi_segments) == len(cp_times) + 1
        self.T = int(total_T)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)
        self.cp_times = [int(t) for t in cp_times]
        self.phi_segments = [np.asarray(p, dtype=float) for p in phi_segments]
        self.rng = np.random.RandomState(int(seed))

        self.t = 0
        self.phi_history = []  # per-batch phi (at batch start)
        self.seg_history = []  # per-batch segment id

    def _seg_id(self, t: int) -> int:
        # number of cps with time <= t
        k = 0
        for c in self.cp_times:
            if t >= c:
                k += 1
            else:
                break
        return k

    def true_phi(self, t: int) -> np.ndarray:
        return self.phi_segments[self._seg_id(t)].copy()

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.t >= self.T:
            raise StopIteration

        # X = self.rng.rand(self.bs, 1)
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs     
        X = u[:, None]
        self.rng.shuffle(X)

        phi_t = self.true_phi(self.t)

        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.seg_history.append(self._seg_id(self.t))

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
    θ* = argmin || η(x;φ) - y_s(x,θ) ||^2  (approximated by dense x-grid)
    """
    x = np.linspace(0, 1, 400).reshape(-1, 1)
    eta = physical_system(x, phi)

    errs = []
    for th in grid:
        ys = computer_model_config2_np(x, np.array([th]))
        errs.append(np.mean((eta - ys) ** 2))

    return float(grid[int(np.argmin(errs))])


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
# Build 4 segment-phis (3 changepoints), centered around phi2=7.5
# magnitude controls how far we move phi[1] (the "a2" term)
# -------------------------------------------------------------
def build_phi_segments_centered(delta: float, center: float = 7.5):
    """
    4 regimes, 3 changepoints
    phi[1] strictly increasing
    mean(phi[1]) = center
    adjacent jump size = delta
    """
    a1, a3 = 5.0, 5.0

    phi2_vals = np.array([
        center - 1.5 * delta,
        center - 0.5 * delta,
        center + 0.5 * delta,
        center + 1.5 * delta,
    ])

    return [
        np.array([a1, v, a3], dtype=float)
        for v in phi2_vals
    ]



# -------------------------------------------------------------
# Run one (frequency, magnitude) experiment
# frequency here is segment length L (in observation-time units)
# 3 CPs => total_T = 4*L
# -------------------------------------------------------------
def run_one_sudden(
    seg_len_L: int,
    delta_mag: float,
    methods: Dict,
    batch_size: int,
    seed: int,
    noise_sd: float = 0.2,
    phi_center: float = 7.5,
    out_dir: str = ".",
):
    # Ensure CP times and total_T align with batch size to avoid partial batch around CP
    seg_len_L = int(seg_len_L)
    bs = int(batch_size)
    if seg_len_L % bs != 0:
        raise ValueError(f"seg_len_L ({seg_len_L}) must be divisible by batch_size ({bs})")

    total_T = 4 * seg_len_L
    cp_times = [seg_len_L, 2 * seg_len_L, 3 * seg_len_L]
    phi_segments = build_phi_segments_centered(delta=delta_mag, center=phi_center)

    print(f"\n=== Sudden experiment: L={seg_len_L}, delta={delta_mag:.4f}, bs={bs}, seed={seed} ===")
    print(f"    cp_times={cp_times}, total_T={total_T}")
    print(f"    phi[1] values={[p[1] for p in phi_segments]} (center={phi_center})")

    # oracle grid
    theta_grid = np.linspace(0, 3, 600)

    # shared stream for oracle phi_history reference (per-batch)
    stream_ref = SuddenChangeDataStream(
        total_T=total_T,
        batch_size=bs,
        noise_sd=noise_sd,
        cp_times=cp_times,
        phi_segments=phi_segments,
        seed=seed,
    )

    # prior
    def prior_sampler(N: int):
        return torch.rand(N, 1) * 3.0

    results = {}

    for name, meta in methods.items():
        print(f"  -> {name}")
        t0 = time()

        theta_hist: List[float] = []
        rmse_hist: List[float] = []
        others_hist: List[dict] = []
        total_obs = 0
        top0_particles_hist = []
        crps_hist = []
        restart_mode_hist: List[str] = []


        # fresh stream per method (same data given same seed)
        stream = SuddenChangeDataStream(
            total_T=total_T,
            batch_size=bs,
            noise_sd=noise_sd,
            cp_times=cp_times,
            phi_segments=phi_segments,
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
                particle_chunk_size=256,
            )
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            roll = OGPRollingStats(window=50)

            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,
                device="cuda",
            )

            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream.next()
                ogp_dev = bocpd.device
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)

                if total_obs > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(bs, device=ogp_dev, dtype=torch.float64)
                    mix_var = torch.zeros(bs, device=ogp_dev, dtype=torch.float64)
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
                    rmse_hist.append(
                        float(torch.sqrt(((mix_mu.cpu() - Yb64.cpu()) ** 2).mean()))
                    )

                rec = bocpd.update_batch(
                    Xb64, Yb64, emulator, model_cfg, None, prior_sampler,
                    verbose=False,
                )

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

                others_hist.append(dict(
                    did_restart=rec["did_restart"],
                    var=float(var_theta[0]),
                    lo=float(lo_theta[0]),
                    hi=float(hi_theta[0]),
                    seg_id=int(stream.seg_history[-1]),
                    t=int(total_obs),
                    pf_info=rec["pf_diags"],
                ))

                total_obs += bs
                pbar.update(bs)
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
                Xb, Yb = stream.next()
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

                others_hist.append(dict(
                    did_restart=False,
                    var=float(
                        (w * (pf.theta - mean_theta).pow(2)).sum(dim=0)[0]
                    ),
                    seg_id=int(stream.seg_history[-1]),
                    t=int(total_obs),
                ))

                total_obs += bs
                pbar.update(bs)
            pbar.close()

        # ---------- BOCPD ----------
        elif meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True
            # Keep old restart by default; opt-in hybrid via methods entry.
            cfg.bocpd.restart_impl = meta.get("restart_impl", "debug_260115")
            cfg.bocpd.hybrid_partial_restart = bool(meta.get("hybrid_partial_restart", True))
            cfg.bocpd.hybrid_tau_delta = float(meta.get("hybrid_tau_delta", 0.05))
            cfg.bocpd.hybrid_tau_theta = float(meta.get("hybrid_tau_theta", 0.05))
            cfg.bocpd.hybrid_tau_full = float(meta.get("hybrid_tau_full", 0.05))
            cfg.bocpd.hybrid_delta_share_rho = float(meta.get("hybrid_delta_share_rho", 0.75))
            cfg.bocpd.hybrid_pf_sigma_mode = str(meta.get("hybrid_pf_sigma_mode", "fixed"))
            cfg.bocpd.hybrid_sigma_delta_alpha = float(meta.get("hybrid_sigma_delta_alpha", 1.0))
            cfg.bocpd.hybrid_sigma_ema_beta = float(meta.get("hybrid_sigma_ema_beta", 0.98))
            cfg.bocpd.hybrid_sigma_min = float(meta.get("hybrid_sigma_min", 1e-4))
            cfg.bocpd.hybrid_sigma_max = float(meta.get("hybrid_sigma_max", 10.0))

            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]

            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
            roll = RollingStats(window=50)

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
                    print(f"     {name}: total_obs={total_obs}")

                Xb, Yb = stream.next()
                report_sub_hist = None

                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    pred_comp = calib.predict_complete(Xb, Yb)
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    # print(name, total_obs, report_hist[-1])
                    rmse_hist.append(
                        float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()))
                    )
                    # rmse_hist.append(
                    #     float(torch.sqrt(((pred["mu_sim"] - Yb) ** 2).mean()))
                    # )
                    crps = crps_gaussian(pred["mu"], pred["var"], Yb).mean()
                    # print(crps)
                    crps_hist.append(crps.item())

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

                # NOTE: assumes your OnlineBayesCalibrator._aggregate_particles(q)
                # returns (mean, var_or_cov, lo, hi) where mean/lo/hi are vectors.
                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)

                theta_hist.append(float(mean_theta[0]))
                # var_theta may be scalar/vec/cov; keep first dim as scalar for logging
                v0 = float(var_theta[0]) if np.ndim(var_theta) >= 1 else float(var_theta)

                ess_gini_info = []
                for ei, e in enumerate(calib.bocpd.experts):
                    ps = e.pf.particles
                    unique_ratio = float(ps.unique_ratio())
                    entropy_1d_histogram = float(ps.entropy_1d_histogram())
                    # print(ei, unique_ratio, entropy_1d_histogram)
                    ess_gini_info.append({"expert_id": ei, "unique_ratio": unique_ratio, "entropy_1d_histogram": entropy_1d_histogram})

                # experts = calib.bocpd.experts

                # weights = torch.tensor([e.log_mass for e in experts])
                # top_idx = torch.argmax(weights).item()
                # top_expert = experts[top_idx]

                # particles = top_expert.pf.particles.theta  
                # pw = top_expert.pf.particles.weights()

                # particles_1d = particles.squeeze(-1).detach().cpu()
                # pw_1d = pw.squeeze(-1).detach().cpu()

                # top0_particles_hist.append((particles_1d, pw_1d))

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

                others_hist.append(
                    dict(
                        did_restart=bool(rec.get("did_restart", False)),
                        var=v0,
                        lo=float(lo_theta[0]),
                        hi=float(hi_theta[0]),
                        ess_gini_info=ess_gini_info,
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        pf_info=rec["pf_diags"],
                        report_sub_hist=report_sub_hist,
                        pf_health_info=ess_gini_info,
                        delta_ll_pair=dll,
                        mu_hat=mu_hat,
                        sigma_hat=sig_hat,
                        h_log=h_log,
                        log_odds_mass=log_odds,
                        anchor_idx=rec.get("anchor_idx", None),
                        cand_idx=rec.get("cand_idx", None),
                        anchor_rl=rec.get("anchor_rl", None),
                        cand_rl=rec.get("cand_rl", None),
                    )
                )

                total_obs += bs

        # ---------- BPC (batch refit each step) ----------
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            bpc = None

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
                    print(f"     {name}: total_obs={total_obs}")

                Xb, Yb = stream.next()
                crps_sim = None
                if X_hist is None:
                    X_hist, y_hist = Xb.numpy(), Yb.numpy()
                else:
                    X_hist = np.concatenate([X_hist, Xb.numpy()], axis=0)
                    y_hist = np.concatenate([y_hist, Yb.numpy()], axis=0)
                if X_hist.shape[0] >= W:
                    X_hist = X_hist[-W:]
                    y_hist = y_hist[-W:]

                if total_obs > 0 and bpc is not None:
                    mu_np, var_np = bpc.predict(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device), torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device) 
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)

                X_all, y_all = X_hist, y_hist

                bpc = BayesianProjectedCalibration(
                    theta_lo=np.array([0.0]),
                    theta_hi=np.array([3.0]),
                    noise_var=float(noise_sd ** 2),
                    y_sim=computer_model_config2_np,
                )
                X_grid = np.linspace(0, 1, 400).reshape(-1, 1)
                bpc.fit(X_all, y_all, X_grid, n_eta_draws=500, n_restart=10, gp_fit_iters=200)

                theta_hist.append(float(bpc.theta_mean[0]))
                entropy_info = bpc.entropy_theta()
                others_hist.append(
                    dict(
                        did_restart=False,
                        var=float(bpc.theta_var[0]) if bpc.theta_var is not None else float("nan"),
                        lo=float("nan"),
                        hi=float("nan"),
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        entropy=entropy_info,
                        crps_sim=crps_sim,
                    )
                )

                theta_samples_bpc = torch.tensor(bpc.theta_samples).squeeze(-1)
                top0_particles_hist.append(theta_samples_bpc)
                batch_dict = dict(
                    particles=[theta_samples_bpc],      # list length E
                    weights=None,          # list length E
                    log_mass=torch.tensor([0.0])  # (E,)
                )

                top0_particles_hist.append(batch_dict)

                total_obs += bs

        # ---------- BPC + BOCPD ----------
        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(
                theta_lo=np.array([0.0]),
                theta_hi=np.array([3.0]),
                noise_var=float(noise_sd ** 2),
                y_sim=computer_model_config2_np,
                X_grid=np.linspace(0, 1, 400).reshape(-1, 1),
                # if your class supports: hazard_h/topk/etc, put them in meta["params"]
                **meta.get("params", {}),
            )

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
                    print(f"     {name}: total_obs={total_obs}")

                Xb, Yb = stream.next()
                crps_sim = None

                if total_obs > 0:
                    mu, var = calib.predict(Xb)
                    mu_t, var_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device), torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)

                info = calib.step_batch(Xb.detach().cpu().numpy(), Yb.detach().cpu().numpy())

                # assumes your StandardBOCPD_BPC._aggregate_particles(q) exists and returns (mean, cov/var, lo, hi)
                theta_mean, theta_var, theta_lo, theta_hi = calib._aggregate_particles(0.9)

                theta_hist.append(float(theta_mean[0]))
                # theta_var could be vector or cov; keep first dim scalar
                try:
                    v0 = float(theta_var[0][0]) if np.ndim(theta_var) >= 1 else float(theta_var)
                except:
                    v0 = theta_var

                others_hist.append(
                    dict(
                        did_restart=bool(info.get("did_restart", False)),
                        var=v0,
                        lo=float(theta_lo[0]) if np.ndim(theta_lo) >= 1 else float(theta_lo),
                        hi=float(theta_hi[0]) if np.ndim(theta_hi) >= 1 else float(theta_hi),
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        crps_sim=crps_sim,
                    )
                )

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

                total_obs += bs

        # ---------- DA (PF + GP prediction) ----------
        elif name == "DA":
            da = PFWithGPPrediction(
                sim_func_np=computer_model_config2_np,
                n_particles=1024, theta_lo=0.0, theta_hi=3.0,
                sigma_obs=noise_sd, resample_ess_ratio=0.5,
                theta_move_std=0.05, window_size=80,
                gp_lengthscale=0.3, gp_signal_var=1.0, seed=seed,
            )
            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream.next()
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
                others_hist.append(dict(
                    did_restart=False, var=0.0,
                    seg_id=int(stream.seg_history[-1]),
                    t=int(total_obs),
                ))
                total_obs += bs
                pbar.update(bs)
            pbar.close()

        # ---------- BC (KOH Sliding Window) ----------
        elif name == "BC":
            bc_theta_grid = np.linspace(0.0, 3.0, 200)
            bc = KOHSlidingWindow(
                sim_func_np=computer_model_config2_np,
                theta_grid=bc_theta_grid, window_size=80,
                sigma_obs=noise_sd, gp_lengthscale=0.3, gp_signal_var=1.0,
            )
            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream.next()
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
                others_hist.append(dict(
                    did_restart=False, var=0.0,
                    seg_id=int(stream.seg_history[-1]),
                    t=int(total_obs),
                ))
                total_obs += bs
                pbar.update(bs)
            pbar.close()

        else:
            raise ValueError(f"Unknown method type: {meta['type']}")

        # ---------- oracle (aligned by batch index) ----------
        # Use the *reference* stream to define "true phi per batch" for oracle
        # but both streams are deterministic under same seed anyway.
        K = len(theta_hist)
        # Make sure stream_ref advanced to K batches:
        while len(stream_ref.phi_history) < K:
            stream_ref.next()

        phi_hist = stream_ref.phi_history[:K]
        oracle_hist = [oracle_theta(phi, theta_grid) for phi in phi_hist]

        results[name] = dict(
            theta=np.asarray(theta_hist, dtype=float),
            theta_oracle=np.asarray(oracle_hist, dtype=float),
            others=others_hist,
            rmse=np.asarray(rmse_hist, dtype=float),
            cp_times=cp_times,
            seg_len_L=seg_len_L,
            delta_mag=float(delta_mag),
            batch_size=bs,
            seed=int(seed),
            top0_particles_hist=top0_particles_hist,
            crps_hist=np.array(crps_hist),
            restart_mode_hist=restart_mode_hist,
        )

        print(f"     done in {time() - t0:.1f}s")

    # Also return the phi/oracle series for external plotting
    phi_hist = stream_ref.phi_history[: len(results[list(results.keys())[0]]["theta"])]
    oracle_hist = results[list(results.keys())[0]]["theta_oracle"]
    return results, phi_hist, oracle_hist


# -------------------------------------------------------------
# Plotting helper
# -------------------------------------------------------------
def plot_theta_tracking(
    res: Dict,
    oracle_hist: np.ndarray,
    cp_times: List[int],
    batch_size: int,
    title: str,
    save_path: str,
):
    plt.figure(figsize=(12, 5))
    for name, d in res.items():
        plt.plot(d["theta"], label=name, alpha=0.9)

    plt.plot(np.asarray(oracle_hist), "k--", lw=2, label="oracle θ*")

    # CP vertical lines in batch-index coordinates
    for c in cp_times:
        x = c // batch_size
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.35)

    plt.title(title)
    plt.xlabel("batch index")
    plt.ylabel("theta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------------------------------------
# Main: traverse (frequency, magnitude) + (seed, batch_size)
# 3 changepoints per run, phi centered around 7.5
# -------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="./figs/sudden_grid_outputs/v1")
    args = parser.parse_args()

    out_dir = args.out_dir
    store_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- experimental grid ---
    if args.debug:
        seeds = [456]               # you can add more
        batch_sizes = [20]      # you can add more
        seg_lens = [120]
        magnitudes = [2.0] 
    else:
        magnitudes = [0.5, 1.0, 2.0, 3.0]
        seeds = [101, 202, 303]               # you can add more
        batch_sizes = [20]      # you can add more

        # frequency: segment length L in observation-time units
        # NOTE: must be divisible by batch_size (enforced in run_one_sudden)
        seg_lens = [80, 120, 200]  # frequency: smaller => more frequent CPs

    # methods
    methods = {
        # "DA": dict(type="da"),
        # "BC": dict(type="bc"),
        "BOCPD-PF": dict(type="bocpd", mode="standard"),
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        # "R-BOCPD-PF-OGP": dict(type="bocpd", mode="restart"),
        # "PF-OGP": dict(type="pf_ogp"),
        "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
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
        "R-BOCPD-PF-halfdiscrepancy-hybrid-rolled": dict(
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
            hybrid_pf_sigma_mode="rolled",
            hybrid_sigma_delta_alpha=1.0,
            hybrid_sigma_ema_beta=0.98,
            hybrid_sigma_min=1e-4,
            hybrid_sigma_max=10.0,
        ),
        # "BPC-80": dict(type="bpc"),
    }

    # run grid
    all_results = {}
    for seg_len_L, delta_mag, batch_size, seed in itertools.product(seg_lens, magnitudes, batch_sizes, seeds):
        # skip invalid combos early (L must be divisible by batch_size)
        if seg_len_L % batch_size != 0:
            continue

        res, phi_hist, oracle_hist = run_one_sudden(
            seg_len_L=seg_len_L,
            delta_mag=delta_mag,
            methods=methods,
            batch_size=batch_size,
            seed=seed,
            noise_sd=0.2,
            phi_center=7.5,
            out_dir=out_dir,
        )
        all_results[(seg_len_L, delta_mag, batch_size, seed)] = res

        tag = f"L{seg_len_L}_delta{delta_mag:g}_bs{batch_size}_seed{seed}"
        save_pt = os.path.join(out_dir, f"sudden_{tag}_results.pt")
        torch.save(res, save_pt)

        save_meta_pt = os.path.join(out_dir, f"sudden_{tag}_phi_oracle.pt")
        # store phi per batch + oracle theta*
        torch.save(dict(phi_hist=phi_hist, oracle_hist=oracle_hist), save_meta_pt)

        # plot
        cp_times = res[list(res.keys())[0]]["cp_times"]
        save_png = os.path.join(out_dir, f"sudden_{tag}_theta.png")
        plot_theta_tracking(
            res=res,
            oracle_hist=oracle_hist,
            cp_times=cp_times,
            batch_size=batch_size,
            title=f"Sudden-change theta tracking (L={seg_len_L}, Δphi2={delta_mag}, bs={batch_size}, seed={seed})",
            save_path=save_png,
        )

        print(f"[Saved] {save_pt}")
        print(f"[Saved] {save_meta_pt}")
        print(f"[Saved] {save_png}")

    all_metrics = []
    restart_mode_rows = []

    for seg_len_L, delta_mag, batch_size, seed in itertools.product(seg_lens, magnitudes, batch_sizes, seeds):
        # 从 all_results 获取对应的 res
        res = all_results[(seg_len_L, delta_mag, batch_size, seed)]  # 注意：当前代码中 all_results[s] 会被覆盖，需要改为 all_results[(s, batch_size, seed)]
        
        for method_name, data in res.items():
            # 计算 theta_rmse: sqrt(mean((theta_pred - theta_oracle)^2))
            theta_rmse = np.sqrt(np.mean((data["theta"] - data["theta_oracle"]) ** 2))
            
            # y_rmse 已经存在于 data["rmse"] 中
            y_rmse_mean = np.mean(data["rmse"])
            
            # y_crps 已经存在于 data["crps_hist"] 中
            y_crps_mean = np.mean(data["crps_hist"])
            
            all_metrics.append({
                "method": method_name,
                "seg_len_L": seg_len_L,
                "delta_mag": delta_mag,
                "batch_size": batch_size,
                "seed": seed,
                "theta_rmse": theta_rmse,
                "y_rmse": y_rmse_mean,
                "y_crps": y_crps_mean,
            })

            if "restart_mode_hist" in data and len(data["restart_mode_hist"]) > 0:
                rm = data["restart_mode_hist"]
                n = len(rm)
                n_none = sum(1 for v in rm if v == "none")
                n_delta = sum(1 for v in rm if v == "delta_only")
                n_full = sum(1 for v in rm if v not in ("none", "delta_only"))
                restart_mode_rows.append({
                    "method": method_name,
                    "seg_len_L": seg_len_L,
                    "delta_mag": delta_mag,
                    "batch_size": batch_size,
                    "seed": seed,
                    "n_steps": n,
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

        # Plot 1: ratio vs delta_mag (group by method, seg_len_L)
        df_plot_mag = (
            df_restart.groupby(["method", "seg_len_L", "delta_mag"], as_index=False)[
                ["none_ratio", "delta_only_ratio", "full_ratio"]
            ]
            .mean()
            .sort_values(["method", "seg_len_L", "delta_mag"])
        )
        for method in df_plot_mag["method"].unique():
            for seg_len in sorted(df_plot_mag[df_plot_mag["method"] == method]["seg_len_L"].unique()):
                sub = df_plot_mag[
                    (df_plot_mag["method"] == method) & (df_plot_mag["seg_len_L"] == seg_len)
                ].copy()
                if len(sub) == 0:
                    continue
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
                plt.xticks(x, [f"{v:.4g}" for v in sub["delta_mag"].values], rotation=0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("delta_mag")
                plt.ylabel("ratio")
                plt.title(f"Restart mode ratio vs delta_mag ({method}, L={seg_len})")
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(f"{store_dir}/restart_mode_ratio_{method}_L{seg_len}.png", dpi=300)
                plt.close()

    # ========== 打印每个 method 的平均 metrics ==========
    print("\n" + "="*70)
    print("Average Metrics Across All Combinations (seg_len_L × delta_mag × batch_sizes × seeds):")
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

    print("All sudden-change grid experiments finished.")


if __name__ == "__main__":
    main()
