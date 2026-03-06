import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Optional

from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor, cho_solve

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

import os
import pickle

from tqdm import tqdm


# =========================================================
# Plot colors
# =========================================================
colors = {
    "PF-1(NoDisc)": "#1f77b4",
    "PF-2(GlobalGP-train)": "#ff7f0e",
    "PF-3(PerParticleGP+hyp)": "#2ca02c",
    "PF-4(TrueOGP@theta_i)": "#d62728",
}


# =========================================================
# Basic metrics
# =========================================================
def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    x = samples.reshape(-1)
    term1 = np.mean(np.abs(x - y))
    term2 = 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    return float(term1 - term2)

def make_phi_jump_schedule(values: List[float], n_batches: int) -> Callable[[int], float]:
    """
    把 n_batches 尽量均匀地分成 len(values) 段，每段取一个常数值。
    例如 values=[4.5, 7.5, 10.5, 7.5], n_batches=20
    -> 每段 5 个 batch.
    """
    n_segments = len(values)
    edges = np.linspace(0, n_batches, n_segments + 1, dtype=int)

    def phi_schedule(b: int) -> float:
        for s in range(n_segments):
            if edges[s] <= b < edges[s + 1]:
                return float(values[s])
        return float(values[-1])

    return phi_schedule


# =========================================================
# Physical vs simulator
# =========================================================
def y_physical(x: np.ndarray, phi: float) -> np.ndarray:
    # physical truth
    return 5.0 * x * np.cos(phi * x) + 5.0 * x


def y_simulator(x: np.ndarray, theta: float) -> np.ndarray:
    # simulator
    return np.sin(5.0 * theta * x) + 5.0 * x


def dy_sim_dtheta(x: np.ndarray, theta: float) -> np.ndarray:
    # derivative wrt theta
    # y_sim = sin(5 theta x) + 5x
    # d/dtheta = 5x cos(5 theta x)
    return 5.0 * x * np.cos(5.0 * theta * x)


# =========================================================
# Utilities
# =========================================================
def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    idx = np.searchsorted(cumsum, positions, side="left")
    return idx


def weighted_mean_and_var(particles: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    m = float(np.sum(weights * particles))
    v = float(np.sum(weights * (particles - m) ** 2))
    return m, v


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def log_normal_pdf_vec(y: float, mean: np.ndarray, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var)


def log_normal_pdf_vec_var(y: float, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mean) ** 2 / var)


def ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights ** 2))


def theta_l2_optimal(phi: float,
                     theta_bounds: Tuple[float, float],
                     x_bounds: Tuple[float, float],
                     n_grid: int = 4000) -> float:
    """
    theta*(phi) = argmin_theta ∫ (y_phys(x;phi) - y_sim(x;theta))^2 dx
    """
    a, b = x_bounds
    xg = np.linspace(a, b, n_grid)
    y_true = y_physical(xg, phi)

    def obj(theta: float) -> float:
        ys = y_simulator(xg, theta)
        return float(np.trapz((y_true - ys) ** 2, xg))

    res = minimize_scalar(
        obj,
        bounds=theta_bounds,
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 200},
    )
    return float(res.x)


# =========================================================
# Configs
# =========================================================
@dataclass
class PFConfig:
    n_particles: int = 400
    theta_prior: Tuple[float, float] = (0.0, 3.0)
    sigma_obs: float = 0.2
    resample_ess_ratio: float = 0.5

    theta_move_std: float = 0.03

    log_ls_move_std: float = 0.05
    log_sv_move_std: float = 0.05
    log_nv_move_std: float = 0.05

    ls_bounds: Tuple[float, float] = (0.03, 1.5)
    sv_bounds: Tuple[float, float] = (1e-3, 50.0)
    nv_bounds: Tuple[float, float] = (1e-4, 5.0)


@dataclass
class StreamConfig:
    n_batches: int = 20
    batch_size: int = 10
    x_low: float = 0.0
    x_high: float = 1.0
    stratified_x: bool = True


# =========================================================
# Standard GP helpers
# =========================================================
def default_gp_hyperparams() -> Dict[str, float]:
    return {"length_scale": 0.2, "signal_var": 5.0, "noise_var": 0.2}


def make_gp_fixed(kernel_params: Dict[str, float],
                  normalize_y: bool = True) -> GaussianProcessRegressor:
    length_scale = float(kernel_params["length_scale"])
    signal_var = float(kernel_params["signal_var"])
    noise_var = float(kernel_params["noise_var"])

    kernel = (
        C(signal_var, constant_value_bounds="fixed")
        * RBF(length_scale, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=noise_var, noise_level_bounds="fixed")
    )

    return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, optimizer=None)


def make_gp_trainable(init_params: Dict[str, float],
                      normalize_y: bool = True,
                      n_restarts_optimizer: int = 0) -> GaussianProcessRegressor:
    ls0 = float(init_params["length_scale"])
    sv0 = float(init_params["signal_var"])
    nv0 = float(init_params["noise_var"])

    kernel = (
        C(sv0, (1e-3, 1e2))
        * RBF(ls0, (1e-3, 3.0))
        + WhiteKernel(noise_level=nv0, noise_level_bounds=(1e-6, 5.0))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=normalize_y,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=0.0,
    )


# =========================================================
# True kernel-orthogonal OGP
# =========================================================
def rbf_kernel_1d(X1: np.ndarray, X2: np.ndarray, length_scale: float, signal_var: float) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float).reshape(-1)
    X2 = np.asarray(X2, dtype=float).reshape(-1)
    sqdist = (X1[:, None] - X2[None, :]) ** 2
    return signal_var * np.exp(-0.5 * sqdist / (length_scale ** 2))


def make_trapz_grid(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(a, b, n)
    w = np.ones(n) * ((b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


class OrthogonalRBFGP1D:
    """
    True kernel-orthogonal GP for scalar theta case.

    Base kernel:
        k(x,x') = sigma_f^2 exp(-(x-x')^2/(2 l^2))

    Orthogonalized wrt g_theta(x) = d y_sim(x,theta) / d theta:

        k_perp(x,x') = k(x,x') - h(x) H^{-1} h(x')

    where
        h(x) = ∫ k(x,xi) g(xi) dxi
        H    = ∫∫ g(xi) k(xi,xj) g(xj) dxi dxj

    We approximate both integrals numerically on a 1D quadrature grid.
    """

    def __init__(self,
                 length_scale: float,
                 signal_var: float,
                 noise_var: float,
                 x_domain: Tuple[float, float],
                 quad_n: int = 201,
                 normalize_y: bool = True,
                 jitter: float = 1e-8):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.x_domain = x_domain
        self.quad_n = int(quad_n)
        self.normalize_y = bool(normalize_y)
        self.jitter = float(jitter)

        self.is_fit = False

    def _prepare_quadrature(self, theta: float):
        a, b = self.x_domain
        xq, wq = make_trapz_grid(a, b, self.quad_n)
        gq = dy_sim_dtheta(xq, theta)

        Kqq = rbf_kernel_1d(xq, xq, self.length_scale, self.signal_var)

        gw = gq * wq
        H = float(gw @ Kqq @ gw)

        self.xq = xq
        self.wq = wq
        self.gq = gq
        self.gw = gw
        self.Kqq = Kqq
        self.H = max(H, 1e-12)

    def _h_vec(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1)
        Kxq = rbf_kernel_1d(X, self.xq, self.length_scale, self.signal_var)
        return Kxq @ self.gw

    def _k_perp(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        K = rbf_kernel_1d(X1, X2, self.length_scale, self.signal_var)
        h1 = self._h_vec(X1)
        h2 = self._h_vec(X2)
        return K - np.outer(h1, h2) / self.H

    def _k_perp_diag(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1)
        base_diag = np.full(len(X), self.signal_var, dtype=float)
        h = self._h_vec(X)
        diag = base_diag - (h ** 2) / self.H
        return np.maximum(diag, 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray, theta: float):
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        self.theta = float(theta)
        self._prepare_quadrature(self.theta)

        self.X_train = X.copy()

        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y))
            if self.y_std < 1e-12:
                self.y_std = 1.0
            y_norm = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_norm = y.copy()

        K = self._k_perp(X, X)
        K = K + (self.noise_var + self.jitter) * np.eye(len(X))

        self.cF = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve(self.cF, y_norm, check_finite=False)

        self.is_fit = True
        return self

    def predict(self, Xtest: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fit:
            raise RuntimeError("OrthogonalRBFGP1D must be fit before predict.")

        Xtest = np.asarray(Xtest, dtype=float).reshape(-1)

        Kxs = self._k_perp(Xtest, self.X_train)
        mean_norm = Kxs @ self.alpha
        mean = self.y_mean + self.y_std * mean_norm

        if not return_std:
            return mean, None

        v = cho_solve(self.cF, Kxs.T, check_finite=False)
        var_norm = self._k_perp_diag(Xtest) - np.sum(Kxs * v.T, axis=1)
        var_norm = np.maximum(var_norm, 1e-12)

        var = (self.y_std ** 2) * var_norm
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean, std


# =========================================================
# Base PF skeleton
# =========================================================
class PFBase:
    def __init__(self, cfg: PFConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _normalize_from_logw(self, logw: np.ndarray) -> np.ndarray:
        logw = logw - logsumexp(logw)
        return np.exp(logw)

    def _maybe_resample(self) -> bool:
        e = ess(self.w)
        if e < self.cfg.resample_ess_ratio * self.cfg.n_particles:
            idx = systematic_resample(self.w, self.rng)
            self._apply_resample_idx(idx)
            self.w[:] = 1.0 / self.cfg.n_particles
            return True
        return False

    def _apply_resample_idx(self, idx: np.ndarray):
        raise NotImplementedError

    def _move_if_resampled(self):
        raise NotImplementedError

    def summarize(self) -> Tuple[float, float]:
        return weighted_mean_and_var(self.theta, self.w)

    def get_theta_samples(self) -> np.ndarray:
        idx = systematic_resample(self.w, self.rng)
        return self.theta[idx].copy()


# =========================================================
# Method 1: no discrepancy
# =========================================================
class PFMethod1_NoDiscrepancy(PFBase):
    def __init__(self, cfg: PFConfig, rng: np.random.Generator):
        super().__init__(cfg, rng)
        lo, hi = cfg.theta_prior
        self.theta = rng.uniform(lo, hi, size=cfg.n_particles)
        self.w = np.ones(cfg.n_particles) / cfg.n_particles

        self.X_hist: List[float] = []
        self.Y_hist: List[float] = []

    def _apply_resample_idx(self, idx: np.ndarray):
        self.theta = self.theta[idx]

    def _move_if_resampled(self):
        self.theta = self.theta + self.rng.normal(0.0, self.cfg.theta_move_std, size=self.cfg.n_particles)
        self.theta = clamp(self.theta, self.cfg.theta_prior[0], self.cfg.theta_prior[1])

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        sigma2 = self.cfg.sigma_obs ** 2
        eps = 1e-300
        logw = np.log(self.w + eps)

        for x_t, y_t in zip(Xb, Yb):
            x_t = float(x_t)
            y_t = float(y_t)
            ys = y_simulator(np.array([x_t]), self.theta).reshape(-1)
            logw += log_normal_pdf_vec(y_t, ys, sigma2)

        self.w = self._normalize_from_logw(logw)

        did_resample = self._maybe_resample()
        if did_resample:
            self._move_if_resampled()

        for x_t, y_t in zip(Xb, Yb):
            self.X_hist.append(float(x_t))
            self.Y_hist.append(float(y_t))


# =========================================================
# Method 2: global GP discrepancy
# =========================================================
class PFMethod2_GPDiscrepancy_Global(PFBase):
    def __init__(self, cfg: PFConfig, rng: np.random.Generator,
                 gp_init: Optional[Dict[str, float]] = None,
                 n_restarts_optimizer: int = 0):
        super().__init__(cfg, rng)
        lo, hi = cfg.theta_prior
        self.theta = rng.uniform(lo, hi, size=cfg.n_particles)
        self.w = np.ones(cfg.n_particles) / cfg.n_particles

        self.gp_init = gp_init if gp_init is not None else default_gp_hyperparams()
        self.n_restarts_optimizer = n_restarts_optimizer

        self.gp: Optional[GaussianProcessRegressor] = None
        self.has_gp = False

        self.X_hist: List[float] = []
        self.Y_hist: List[float] = []

    def _apply_resample_idx(self, idx: np.ndarray):
        self.theta = self.theta[idx]

    def _move_if_resampled(self):
        self.theta = self.theta + self.rng.normal(0.0, self.cfg.theta_move_std, size=self.cfg.n_particles)
        self.theta = clamp(self.theta, self.cfg.theta_prior[0], self.cfg.theta_prior[1])

    def _gp_predict(self, x: float) -> Tuple[float, float]:
        if not self.has_gp or self.gp is None:
            return 0.0, 0.0
        m, std = self.gp.predict(np.array([[x]]), return_std=True)
        return float(m[0]), float(std[0] ** 2)

    def _refit_gp_on_history(self):
        if len(self.X_hist) < 5:
            self.gp = None
            self.has_gp = False
            return

        X = np.array(self.X_hist, dtype=float).reshape(-1, 1)
        Y = np.array(self.Y_hist, dtype=float)
        x = X[:, 0]

        ys_mat = np.sin(5.0 * np.outer(self.theta, x)) + 5.0 * x
        ys_mix = np.sum(self.w[:, None] * ys_mat, axis=0)
        resid = Y - ys_mix

        gp = make_gp_trainable(self.gp_init, n_restarts_optimizer=self.n_restarts_optimizer)
        gp.fit(X, resid)
        self.gp = gp
        self.has_gp = True

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        sigma2 = self.cfg.sigma_obs ** 2
        eps = 1e-300
        logw = np.log(self.w + eps)

        for x_t, y_t in zip(Xb, Yb):
            x_t = float(x_t)
            y_t = float(y_t)
            m_d, v_d = self._gp_predict(x_t)

            ys = y_simulator(np.array([x_t]), self.theta).reshape(-1)
            mean = ys + m_d
            var = v_d + sigma2
            logw += log_normal_pdf_vec_var(y_t, mean, np.full_like(mean, var))

        self.w = self._normalize_from_logw(logw)

        did_resample = self._maybe_resample()
        if did_resample:
            self._move_if_resampled()

        for x_t, y_t in zip(Xb, Yb):
            self.X_hist.append(float(x_t))
            self.Y_hist.append(float(y_t))

        self._refit_gp_on_history()


# =========================================================
# Method 3: per-particle GP discrepancy
# =========================================================
class PFMethod3_GPDiscrepancy_PerParticleHypers(PFBase):
    def __init__(self, cfg: PFConfig, rng: np.random.Generator):
        super().__init__(cfg, rng)
        N = cfg.n_particles
        lo, hi = cfg.theta_prior
        self.theta = rng.uniform(lo, hi, size=N)
        self.w = np.ones(N) / N

        base = default_gp_hyperparams()
        self.length_scale = clamp(
            np.exp(np.log(base["length_scale"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.ls_bounds[0], cfg.ls_bounds[1]
        )
        self.signal_var = clamp(
            np.exp(np.log(base["signal_var"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.sv_bounds[0], cfg.sv_bounds[1]
        )
        self.noise_var = clamp(
            np.exp(np.log(base["noise_var"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.nv_bounds[0], cfg.nv_bounds[1]
        )

        self.X_hist: List[float] = []
        self.Y_hist: List[float] = []

    def _apply_resample_idx(self, idx: np.ndarray):
        self.theta = self.theta[idx]
        self.length_scale = self.length_scale[idx]
        self.signal_var = self.signal_var[idx]
        self.noise_var = self.noise_var[idx]

    def _move_if_resampled(self):
        self.theta = self.theta + self.rng.normal(0.0, self.cfg.theta_move_std, size=self.cfg.n_particles)
        self.theta = clamp(self.theta, self.cfg.theta_prior[0], self.cfg.theta_prior[1])

        self.length_scale = np.exp(np.log(self.length_scale) + self.rng.normal(0.0, self.cfg.log_ls_move_std, size=self.cfg.n_particles))
        self.signal_var = np.exp(np.log(self.signal_var) + self.rng.normal(0.0, self.cfg.log_sv_move_std, size=self.cfg.n_particles))
        self.noise_var = np.exp(np.log(self.noise_var) + self.rng.normal(0.0, self.cfg.log_nv_move_std, size=self.cfg.n_particles))

        self.length_scale = clamp(self.length_scale, self.cfg.ls_bounds[0], self.cfg.ls_bounds[1])
        self.signal_var = clamp(self.signal_var, self.cfg.sv_bounds[0], self.cfg.sv_bounds[1])
        self.noise_var = clamp(self.noise_var, self.cfg.nv_bounds[0], self.cfg.nv_bounds[1])

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        sigma2 = self.cfg.sigma_obs ** 2
        eps = 1e-300
        logw = np.log(self.w + eps)

        has_hist = len(self.X_hist) >= 5
        if has_hist:
            Xhist = np.array(self.X_hist, dtype=float).reshape(-1, 1)
            Yhist = np.array(self.Y_hist, dtype=float)
            xhist = Xhist[:, 0]

        N = self.cfg.n_particles
        for i in range(N):
            if not has_hist:
                ll = 0.0
                for x_t, y_t in zip(Xb, Yb):
                    ys_i = y_simulator(np.array([x_t]), float(self.theta[i]))[0]
                    ll += float(log_normal_pdf_vec(y_t, np.array([ys_i]), sigma2)[0])
                logw[i] += ll
                continue

            hyp = {
                "length_scale": float(self.length_scale[i]),
                "signal_var": float(self.signal_var[i]),
                "noise_var": float(self.noise_var[i]),
            }
            gp = make_gp_fixed(hyp, normalize_y=True)

            ys_hist = y_simulator(xhist, float(self.theta[i]))
            resid = Yhist - ys_hist
            gp.fit(Xhist, resid)

            ll = 0.0
            for x_t, y_t in zip(Xb, Yb):
                m, std = gp.predict(np.array([[float(x_t)]]), return_std=True)
                m_d = float(m[0])
                v_d = float(std[0] ** 2)

                ys_i = y_simulator(np.array([x_t]), float(self.theta[i]))[0]
                mean = ys_i + m_d
                var = v_d + sigma2
                ll += float(log_normal_pdf_vec_var(y_t, np.array([mean]), np.array([var]))[0])

            logw[i] += ll

        self.w = self._normalize_from_logw(logw)

        did_resample = self._maybe_resample()
        if did_resample:
            self._move_if_resampled()

        for x_t, y_t in zip(Xb, Yb):
            self.X_hist.append(float(x_t))
            self.Y_hist.append(float(y_t))


# =========================================================
# Method 4: TRUE kernel-orthogonal OGP per particle
# =========================================================
class PFMethod4_TrueOrthogonalGP_PerParticleHypers(PFBase):
    """
    True kernel-orthogonal OGP.

    For each particle i:
      discrepancy GP prior is orthogonalized wrt g_theta(x)=d y_sim(x,theta_i)/dtheta
      via kernel:
          k_perp(x,x') = k(x,x') - h(x) H^{-1} h(x')

    This is a true kernel-level orthogonalization, not residual projection.
    """

    def __init__(self,
                 cfg: PFConfig,
                 rng: np.random.Generator,
                 x_domain: Tuple[float, float],
                 ogp_quad_n: int = 201):
        super().__init__(cfg, rng)
        N = cfg.n_particles
        lo, hi = cfg.theta_prior
        self.theta = rng.uniform(lo, hi, size=N)
        self.w = np.ones(N) / N

        base = default_gp_hyperparams()
        self.length_scale = clamp(
            np.exp(np.log(base["length_scale"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.ls_bounds[0], cfg.ls_bounds[1]
        )
        self.signal_var = clamp(
            np.exp(np.log(base["signal_var"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.sv_bounds[0], cfg.sv_bounds[1]
        )
        self.noise_var = clamp(
            np.exp(np.log(base["noise_var"]) + rng.normal(0.0, 0.2, size=N)),
            cfg.nv_bounds[0], cfg.nv_bounds[1]
        )

        self.X_hist: List[float] = []
        self.Y_hist: List[float] = []

        self.x_domain = x_domain
        self.ogp_quad_n = int(ogp_quad_n)

    def _apply_resample_idx(self, idx: np.ndarray):
        self.theta = self.theta[idx]
        self.length_scale = self.length_scale[idx]
        self.signal_var = self.signal_var[idx]
        self.noise_var = self.noise_var[idx]

    def _move_if_resampled(self):
        self.theta = self.theta + self.rng.normal(0.0, self.cfg.theta_move_std, size=self.cfg.n_particles)
        self.theta = clamp(self.theta, self.cfg.theta_prior[0], self.cfg.theta_prior[1])

        self.length_scale = np.exp(np.log(self.length_scale) + self.rng.normal(0.0, self.cfg.log_ls_move_std, size=self.cfg.n_particles))
        self.signal_var = np.exp(np.log(self.signal_var) + self.rng.normal(0.0, self.cfg.log_sv_move_std, size=self.cfg.n_particles))
        self.noise_var = np.exp(np.log(self.noise_var) + self.rng.normal(0.0, self.cfg.log_nv_move_std, size=self.cfg.n_particles))

        self.length_scale = clamp(self.length_scale, self.cfg.ls_bounds[0], self.cfg.ls_bounds[1])
        self.signal_var = clamp(self.signal_var, self.cfg.sv_bounds[0], self.cfg.sv_bounds[1])
        self.noise_var = clamp(self.noise_var, self.cfg.nv_bounds[0], self.cfg.nv_bounds[1])

    def _build_ogp(self, i: int, Xhist: np.ndarray, Yhist: np.ndarray) -> OrthogonalRBFGP1D:
        theta_i = float(self.theta[i])

        ys_hist = y_simulator(Xhist[:, 0], theta_i)
        resid = Yhist - ys_hist

        ogp = OrthogonalRBFGP1D(
            length_scale=float(self.length_scale[i]),
            signal_var=float(self.signal_var[i]),
            noise_var=float(self.noise_var[i]),
            x_domain=self.x_domain,
            quad_n=self.ogp_quad_n,
            normalize_y=True,
            jitter=1e-8,
        )
        ogp.fit(Xhist[:, 0], resid, theta=theta_i)
        return ogp

    def update_batch(self, Xb: np.ndarray, Yb: np.ndarray):
        sigma2 = self.cfg.sigma_obs ** 2
        eps = 1e-300
        logw = np.log(self.w + eps)

        has_hist = len(self.X_hist) >= 5
        if has_hist:
            Xhist = np.array(self.X_hist, dtype=float).reshape(-1, 1)
            Yhist = np.array(self.Y_hist, dtype=float)

        N = self.cfg.n_particles

        for i in range(N):
            if not has_hist:
                ll = 0.0
                for x_t, y_t in zip(Xb, Yb):
                    ys_i = y_simulator(np.array([x_t]), float(self.theta[i]))[0]
                    ll += float(log_normal_pdf_vec(y_t, np.array([ys_i]), sigma2)[0])
                logw[i] += ll
                continue

            ogp = self._build_ogp(i, Xhist, Yhist)

            ll = 0.0
            for x_t, y_t in zip(Xb, Yb):
                m, std = ogp.predict(np.array([float(x_t)]), return_std=True)
                m_d = float(m[0])
                v_d = float(std[0] ** 2)

                ys_i = y_simulator(np.array([x_t]), float(self.theta[i]))[0]
                mean = ys_i + m_d
                var = v_d + sigma2
                ll += float(log_normal_pdf_vec_var(y_t, np.array([mean]), np.array([var]))[0])

            logw[i] += ll

        self.w = self._normalize_from_logw(logw)

        did_resample = self._maybe_resample()
        if did_resample:
            self._move_if_resampled()

        for x_t, y_t in zip(Xb, Yb):
            self.X_hist.append(float(x_t))
            self.Y_hist.append(float(y_t))


# =========================================================
# Data generation
# =========================================================
def generate_stream(stream_cfg: StreamConfig,
                    phi_schedule: Callable[[int], float],
                    sigma_obs: float,
                    rng: np.random.Generator,
                    theta_bounds: Tuple[float, float]) -> Dict[str, np.ndarray]:
    B, K = stream_cfg.n_batches, stream_cfg.batch_size
    x_low, x_high = stream_cfg.x_low, stream_cfg.x_high

    X = np.empty((B, K), dtype=float)
    for b in range(B):
        if stream_cfg.stratified_x:
            u = (np.arange(K) + rng.random(K)) / K
            xb = x_low + (x_high - x_low) * u
            rng.shuffle(xb)
            X[b] = xb
        else:
            X[b] = rng.uniform(x_low, x_high, size=K)

    phi = np.array([phi_schedule(b) for b in range(B)], dtype=float)

    theta_true = np.array([
        theta_l2_optimal(float(phi[b]), theta_bounds=theta_bounds, x_bounds=(x_low, x_high), n_grid=4000)
        for b in range(B)
    ], dtype=float)

    Y = np.empty((B, K), dtype=float)
    for b in range(B):
        mu = y_physical(X[b], phi[b])
        Y[b] = mu + rng.normal(0.0, sigma_obs, size=K)

    return {"X": X, "Y": Y, "phi": phi, "theta_true": theta_true}


# =========================================================
# Evaluation helpers
# =========================================================
def weighted_theta_mean(method) -> float:
    return float(np.sum(method.w * method.theta))


def fit_shared_gp_residual_from_theta_mean(
    X_hist: np.ndarray,
    Y_hist: np.ndarray,
    theta_mean: float,
    gp_init: Optional[Dict[str, float]] = None,
    n_restarts_optimizer: int = 0,
) -> Optional[GaussianProcessRegressor]:
    if len(X_hist) < 5:
        return None

    gp_init = gp_init if gp_init is not None else default_gp_hyperparams()

    X_hist = np.asarray(X_hist, dtype=float).reshape(-1, 1)
    x = X_hist[:, 0]
    Y_hist = np.asarray(Y_hist, dtype=float)

    resid = Y_hist - y_simulator(x, theta_mean)

    gp = make_gp_trainable(gp_init, n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(X_hist, resid)
    return gp


def predict_y_for_method_on_batch(
    method_name: str,
    method,
    Xb: np.ndarray,
    pf_cfg: PFConfig,
    eval_rng: np.random.Generator,
    gp_init: Optional[Dict[str, float]] = None,
    n_restarts_optimizer: int = 0,
    n_sample_cap: Optional[int] = 200,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    Xb = np.asarray(Xb, dtype=float)
    K = len(Xb)

    theta_samples = method.get_theta_samples()
    if n_sample_cap is not None and len(theta_samples) > n_sample_cap:
        idx = eval_rng.choice(len(theta_samples), size=n_sample_cap, replace=False)
        theta_samples = theta_samples[idx]

    pred_mean = np.zeros(K, dtype=float)
    pred_var = np.zeros(K, dtype=float)
    pred_samples_list: List[np.ndarray] = []

    if method_name == "PF-1(NoDisc)":
        theta_bar = weighted_theta_mean(method)

        gp = fit_shared_gp_residual_from_theta_mean(
            np.array(method.X_hist, dtype=float),
            np.array(method.Y_hist, dtype=float),
            theta_bar,
            gp_init=gp_init,
            n_restarts_optimizer=n_restarts_optimizer,
        )

        for j, x in enumerate(Xb):
            base = y_simulator(np.array([x]), theta_bar)[0]

            if gp is None:
                m_d, v_d = 0.0, 0.0
            else:
                m, std = gp.predict(np.array([[x]]), return_std=True)
                m_d, v_d = float(m[0]), float(std[0] ** 2)

            mu = base + m_d
            samp = mu + eval_rng.normal(
                0.0, np.sqrt(v_d + pf_cfg.sigma_obs ** 2), size=len(theta_samples)
            )

            pred_mean[j] = mu
            pred_var[j] = np.var(samp, ddof=1) if len(samp) > 1 else 0.0
            pred_samples_list.append(samp)

        return pred_mean, pred_samples_list, pred_var

    elif method_name == "PF-2(GlobalGP-train)":
        for j, x in enumerate(Xb):
            ys = y_simulator(np.full_like(theta_samples, x), theta_samples)

            if method.has_gp and method.gp is not None:
                m, std = method.gp.predict(np.array([[x]]), return_std=True)
                m_d, v_d = float(m[0]), float(std[0] ** 2)
            else:
                m_d, v_d = 0.0, 0.0

            samp = ys + m_d + eval_rng.normal(
                0.0, np.sqrt(v_d + pf_cfg.sigma_obs ** 2), size=len(theta_samples)
            )

            pred_mean[j] = np.mean(samp)
            pred_var[j] = np.var(samp, ddof=1) if len(samp) > 1 else 0.0
            pred_samples_list.append(samp)

        return pred_mean, pred_samples_list, pred_var

    elif method_name == "PF-3(PerParticleGP+hyp)":
        has_hist = len(method.X_hist) >= 5
        if has_hist:
            Xhist = np.array(method.X_hist, dtype=float).reshape(-1, 1)
            Yhist = np.array(method.Y_hist, dtype=float)
            xhist = Xhist[:, 0]

        for j, x in enumerate(Xb):
            samp = np.zeros(len(theta_samples), dtype=float)

            for i, th in enumerate(theta_samples):
                base = y_simulator(np.array([x]), th)[0]

                if not has_hist:
                    m_d, v_d = 0.0, 0.0
                else:
                    idx0 = int(np.argmin(np.abs(method.theta - th)))
                    hyp = {
                        "length_scale": float(method.length_scale[idx0]),
                        "signal_var": float(method.signal_var[idx0]),
                        "noise_var": float(method.noise_var[idx0]),
                    }
                    gp = make_gp_fixed(hyp, normalize_y=True)

                    ys_hist = y_simulator(xhist, th)
                    resid = Yhist - ys_hist
                    gp.fit(Xhist, resid)

                    m, std = gp.predict(np.array([[x]]), return_std=True)
                    m_d = float(m[0])
                    v_d = float(std[0] ** 2)

                samp[i] = base + m_d + eval_rng.normal(0.0, np.sqrt(v_d + pf_cfg.sigma_obs ** 2))

            pred_mean[j] = np.mean(samp)
            pred_var[j] = np.var(samp, ddof=1) if len(samp) > 1 else 0.0
            pred_samples_list.append(samp)

        return pred_mean, pred_samples_list, pred_var

    elif method_name == "PF-4(TrueOGP@theta_i)":
        has_hist = len(method.X_hist) >= 5
        if has_hist:
            Xhist = np.array(method.X_hist, dtype=float).reshape(-1, 1)
            Yhist = np.array(method.Y_hist, dtype=float)

        for j, x in enumerate(Xb):
            samp = np.zeros(len(theta_samples), dtype=float)

            for i, th in enumerate(theta_samples):
                base = y_simulator(np.array([x]), th)[0]

                if not has_hist:
                    m_d, v_d = 0.0, 0.0
                else:
                    idx0 = int(np.argmin(np.abs(method.theta - th)))
                    ogp = OrthogonalRBFGP1D(
                        length_scale=float(method.length_scale[idx0]),
                        signal_var=float(method.signal_var[idx0]),
                        noise_var=float(method.noise_var[idx0]),
                        x_domain=method.x_domain,
                        quad_n=method.ogp_quad_n,
                        normalize_y=True,
                        jitter=1e-8,
                    )

                    ys_hist = y_simulator(Xhist[:, 0], th)
                    resid = Yhist - ys_hist
                    ogp.fit(Xhist[:, 0], resid, theta=th)

                    m, std = ogp.predict(np.array([x]), return_std=True)
                    m_d = float(m[0])
                    v_d = float(std[0] ** 2)

                samp[i] = base + m_d + eval_rng.normal(0.0, np.sqrt(v_d + pf_cfg.sigma_obs ** 2))

            pred_mean[j] = np.mean(samp)
            pred_var[j] = np.var(samp, ddof=1) if len(samp) > 1 else 0.0
            pred_samples_list.append(samp)

        return pred_mean, pred_samples_list, pred_var

    else:
        raise ValueError(f"Unknown method_name: {method_name}")


def evaluate_theta_metrics(results: dict, theta_true: np.ndarray, n_batches: int) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in results.keys():
        theta_mean = results[k]["mean"]
        theta_rmse = rmse(theta_mean, theta_true)

        crps_list = []
        for b in range(n_batches):
            samp = results[k]["samples"][b]
            crps_list.append(crps_ensemble(samp, float(theta_true[b])))

        out[k] = {
            "theta_rmse": float(theta_rmse),
            "theta_crps": float(np.mean(crps_list)),
        }
    return out


# =========================================================
# Plot helpers
# =========================================================
# def save_scenario_plots(name: str,
#                         stream_cfg: StreamConfig,
#                         theta_true: np.ndarray,
#                         results: dict,
#                         methods: dict,
#                         out_dir: str = "figs"):
#     os.makedirs(out_dir, exist_ok=True)

#     safe_name = (
#         name.replace(" ", "_")
#             .replace("(", "")
#             .replace(")", "")
#             .replace("+", "plus")
#             .replace("*", "star")
#     )

#     fig1 = plt.figure(figsize=(10, 4))
#     ax = fig1.gca()
#     ax.plot(np.arange(1, len(theta_true) + 1), theta_true, label="theta_true (L2-opt)", linewidth=2, color="black")
#     for k in methods.keys():
#         ax.plot(np.arange(1, len(results[k]["mean"]) + 1), results[k]["mean"], label=f"{k} mean",
#                 marker="o", markersize=3, color=colors[k])
#     ax.set_title(f"{name}: posterior mean per batch")
#     ax.set_xlabel("batch index")
#     ax.set_ylabel("theta")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     plt.tight_layout()
#     fig1.savefig(f"{out_dir}/{safe_name}_mean.pdf", bbox_inches="tight")
#     plt.close(fig1)

#     fig2 = plt.figure(figsize=(10, 4))
#     ax = fig2.gca()
#     for k in methods.keys():
#         ax.plot(np.arange(1, len(results[k]["var"]) + 1), results[k]["var"], label=f"{k} var",
#                 marker="o", markersize=3, color=colors[k])
#     ax.set_title(f"{name}: posterior variance per batch")
#     ax.set_xlabel("batch index")
#     ax.set_ylabel("Var(theta)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     plt.tight_layout()
#     fig2.savefig(f"{out_dir}/{safe_name}_var.pdf", bbox_inches="tight")
#     plt.close(fig2)

#     fig3 = plt.figure(figsize=(14, 5))
#     ax = fig3.gca()

#     method_list = list(methods.keys())
#     n_methods = len(method_list)
#     B = stream_cfg.n_batches
#     batch_centers = np.arange(1, B + 1)

#     positions = []
#     box_data = []
#     width = 0.25
#     offsets = np.linspace(-width, width, n_methods)

#     for b in range(B):
#         for mi, mk in enumerate(method_list):
#             positions.append((b + 1) + offsets[mi])
#             box_data.append(results[mk]["samples"][b])

#     bp = ax.boxplot(
#         box_data,
#         positions=positions,
#         widths=0.18,
#         patch_artist=True,
#         showfliers=False
#     )

#     for i, patch in enumerate(bp["boxes"]):
#         mk = method_list[i % n_methods]
#         patch.set_facecolor(colors[mk])
#         patch.set_alpha(0.35)
#         patch.set_edgecolor(colors[mk])

#     for med in bp["medians"]:
#         med.set_color("black")
#         med.set_linewidth(1.2)

#     ax.plot(np.arange(1, B + 1), theta_true, label="theta_true (L2-opt)", linewidth=2, color="black")
#     ax.set_title(f"{name}: grouped boxplots of theta particles (per batch)")
#     ax.set_xticks(batch_centers)
#     ax.set_xlim(0.5, B + 0.5)
#     ax.set_xlabel("batch index")
#     ax.set_ylabel("theta")
#     ax.grid(True, alpha=0.25)

#     from matplotlib.patches import Patch
#     handles = [Patch(facecolor=colors[k], edgecolor=colors[k], alpha=0.35, label=k) for k in method_list]
#     handles.append(plt.Line2D([0], [0], color="black", lw=2, label="theta_true (L2-opt)"))
#     ax.legend(handles=handles, loc="best")

#     plt.tight_layout()
#     fig3.savefig(f"{out_dir}/{safe_name}_boxplot.pdf", bbox_inches="tight")
#     plt.close(fig3)

def save_scenario_plots(name: str,
                        stream_cfg: StreamConfig,
                        theta_true: np.ndarray,
                        results: dict,
                        methods: dict,
                        X: np.ndarray,
                        Y_target_all: np.ndarray,
                        out_dir: str = "figs",
                        y_plot_batches: Optional[List[int]] = None):
    """
    Y_target_all: shape (B, K)
        你用于评估 y 的 ground truth。
        如果 eval_on_observation_y=True，就传 noisy Y；
        否则传 noiseless physical y.
    y_plot_batches:
        想画哪些 batch 的 y prediction vs truth。
        传 None 时默认画 [0, B//2, B-1]
    """
    os.makedirs(out_dir, exist_ok=True)

    safe_name = (
        name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
            .replace("*", "star")
    )

    # -------------------------------------------------
    # 1) theta posterior mean
    # -------------------------------------------------
    fig1 = plt.figure(figsize=(10, 4))
    ax = fig1.gca()
    ax.plot(np.arange(1, len(theta_true) + 1), theta_true,
            label="theta_true (L2-opt)", linewidth=2, color="black")
    for k in methods.keys():
        ax.plot(np.arange(1, len(results[k]["mean"]) + 1), results[k]["mean"],
                label=f"{k} mean", marker="o", markersize=3, color=colors[k])
    ax.set_title(f"{name}: posterior mean per batch")
    ax.set_xlabel("batch index")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig1.savefig(f"{out_dir}/{safe_name}_mean.pdf", bbox_inches="tight")
    plt.close(fig1)

    # -------------------------------------------------
    # 2) theta posterior variance
    # -------------------------------------------------
    fig2 = plt.figure(figsize=(10, 4))
    ax = fig2.gca()
    for k in methods.keys():
        ax.plot(np.arange(1, len(results[k]["var"]) + 1), results[k]["var"],
                label=f"{k} var", marker="o", markersize=3, color=colors[k])
    ax.set_title(f"{name}: posterior variance per batch")
    ax.set_xlabel("batch index")
    ax.set_ylabel("Var(theta)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig2.savefig(f"{out_dir}/{safe_name}_var.pdf", bbox_inches="tight")
    plt.close(fig2)

    # -------------------------------------------------
    # 3) theta boxplot
    # -------------------------------------------------
    fig3 = plt.figure(figsize=(14, 5))
    ax = fig3.gca()

    method_list = list(methods.keys())
    n_methods = len(method_list)
    B = stream_cfg.n_batches
    batch_centers = np.arange(1, B + 1)

    positions = []
    box_data = []
    width = 0.25
    offsets = np.linspace(-width, width, n_methods)

    for b in range(B):
        for mi, mk in enumerate(method_list):
            positions.append((b + 1) + offsets[mi])
            box_data.append(results[mk]["samples"][b])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False
    )

    for i, patch in enumerate(bp["boxes"]):
        mk = method_list[i % n_methods]
        patch.set_facecolor(colors[mk])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors[mk])

    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.2)

    ax.plot(np.arange(1, B + 1), theta_true,
            label="theta_true (L2-opt)", linewidth=2, color="black")
    ax.set_title(f"{name}: grouped boxplots of theta particles (per batch)")
    ax.set_xticks(batch_centers)
    ax.set_xlim(0.5, B + 0.5)
    ax.set_xlabel("batch index")
    ax.set_ylabel("theta")
    ax.grid(True, alpha=0.25)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[k], edgecolor=colors[k], alpha=0.35, label=k)
               for k in method_list]
    handles.append(plt.Line2D([0], [0], color="black", lw=2, label="theta_true (L2-opt)"))
    ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    fig3.savefig(f"{out_dir}/{safe_name}_boxplot.pdf", bbox_inches="tight")
    plt.close(fig3)

    # -------------------------------------------------
    # 4) y_RMSE per batch
    # -------------------------------------------------
    fig4 = plt.figure(figsize=(10, 4))
    ax = fig4.gca()
    for k in methods.keys():
        ax.plot(np.arange(1, len(results[k]["y_rmse_batches"]) + 1),
                results[k]["y_rmse_batches"],
                label=f"{k} y_RMSE", marker="o", markersize=3, color=colors[k])
    ax.set_title(f"{name}: y RMSE per batch")
    ax.set_xlabel("batch index")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig4.savefig(f"{out_dir}/{safe_name}_y_rmse.pdf", bbox_inches="tight")
    plt.close(fig4)

    # -------------------------------------------------
    # 5) y_CRPS per batch
    # -------------------------------------------------
    fig5 = plt.figure(figsize=(10, 4))
    ax = fig5.gca()
    for k in methods.keys():
        ax.plot(np.arange(1, len(results[k]["y_crps_batches"]) + 1),
                results[k]["y_crps_batches"],
                label=f"{k} y_CRPS", marker="o", markersize=3, color=colors[k])
    ax.set_title(f"{name}: y CRPS per batch")
    ax.set_xlabel("batch index")
    ax.set_ylabel("CRPS")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig5.savefig(f"{out_dir}/{safe_name}_y_crps.pdf", bbox_inches="tight")
    plt.close(fig5)

    # -------------------------------------------------
    # 6) y prediction mean vs ground truth on selected batches
    # -------------------------------------------------
    if y_plot_batches is None:
        y_plot_batches = sorted(set([0, B // 2, B - 1]))

    for b in y_plot_batches:
        if b < 0 or b >= B:
            continue

        xb = np.asarray(X[b], dtype=float)
        yt = np.asarray(Y_target_all[b], dtype=float)

        order = np.argsort(xb)
        xb_sorted = xb[order]
        yt_sorted = yt[order]

        figb = plt.figure(figsize=(10, 5))
        ax = figb.gca()

        ax.plot(xb_sorted, yt_sorted, color="black", linewidth=2, marker="o",
                label="ground truth")

        for k in methods.keys():
            ypm = np.asarray(results[k]["y_pred_mean_batches"][b], dtype=float)
            ypm_sorted = ypm[order]
            ax.plot(xb_sorted, ypm_sorted, color=colors[k], linewidth=1.8,
                    marker="o", markersize=4, label=f"{k} pred mean")

        ax.set_title(f"{name}: y prediction vs truth (batch {b+1})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        figb.savefig(f"{out_dir}/{safe_name}_y_pred_batch{b+1}.pdf", bbox_inches="tight")
        plt.close(figb)


# =========================================================
# Experiment runner
# =========================================================
def run_one_scenario(name: str,
                     stream_cfg: StreamConfig,
                     pf_cfg: PFConfig,
                     phi_schedule: Callable[[int], float],
                     seed: int = 0,
                     method2_gp_restarts: int = 0,
                     save_artifacts: bool = True,
                     eval_on_observation_y: bool = True,
                     ogp_quad_n: int = 201):
    rng = np.random.default_rng(seed)
    data = generate_stream(stream_cfg, phi_schedule, pf_cfg.sigma_obs, rng, theta_bounds=pf_cfg.theta_prior)
    X, Y = data["X"], data["Y"]
    theta_true = data["theta_true"]
    phi = data["phi"]

    rng1 = np.random.default_rng(seed + 1)
    rng2 = np.random.default_rng(seed + 2)
    rng3 = np.random.default_rng(seed + 3)
    rng4 = np.random.default_rng(seed + 4)
    eval_rng = np.random.default_rng(seed + 999)

    m1 = PFMethod1_NoDiscrepancy(pf_cfg, rng1)
    m2 = PFMethod2_GPDiscrepancy_Global(pf_cfg, rng2, n_restarts_optimizer=method2_gp_restarts)
    m3 = PFMethod3_GPDiscrepancy_PerParticleHypers(pf_cfg, rng3)
    m4 = PFMethod4_TrueOrthogonalGP_PerParticleHypers(
        pf_cfg,
        rng4,
        x_domain=(stream_cfg.x_low, stream_cfg.x_high),
        ogp_quad_n=ogp_quad_n,
    )

    methods = {
        "PF-1(NoDisc)": m1,
        "PF-2(GlobalGP-train)": m2,
        "PF-3(PerParticleGP+hyp)": m3,
        "PF-4(TrueOGP@theta_i)": m4,
    }

    results = {
        k: {
            "mean": [],
            "var": [],
            "samples": [],
            "y_pred_mean_batches": [],
            "y_pred_samples_batches": [],
            "y_rmse_batches": [],
            "y_crps_batches": [],
        } for k in methods.keys()
    }
    times = {k: 0.0 for k in methods.keys()}

    for b in tqdm(range(stream_cfg.n_batches), desc=f"Running batches [{name}] seed={seed}"):
        Xb = X[b]
        Yb = Y[b]

        Y_target = Yb if eval_on_observation_y else y_physical(Xb, float(phi[b]))

        for k, m in methods.items():
            t0 = time.perf_counter()
            m.update_batch(Xb, Yb)
            t1 = time.perf_counter()
            times[k] += (t1 - t0)

            mu, va = m.summarize()
            results[k]["mean"].append(mu)
            results[k]["var"].append(va)
            results[k]["samples"].append(m.get_theta_samples())

            y_pred_mean, y_pred_samples_list, y_pred_var = predict_y_for_method_on_batch(
                method_name=k,
                method=m,
                Xb=Xb,
                pf_cfg=pf_cfg,
                eval_rng=eval_rng,
                gp_init=default_gp_hyperparams(),
                n_restarts_optimizer=method2_gp_restarts,
                n_sample_cap=200,
            )

            y_rmse_b = rmse(y_pred_mean, Y_target)
            y_crps_b = float(np.mean([
                crps_ensemble(y_pred_samples_list[j], float(Y_target[j]))
                for j in range(len(Y_target))
            ]))

            results[k]["y_pred_mean_batches"].append(y_pred_mean)
            results[k]["y_pred_samples_batches"].append(y_pred_samples_list)
            results[k]["y_rmse_batches"].append(y_rmse_b)
            results[k]["y_crps_batches"].append(y_crps_b)

    for k in results.keys():
        results[k]["mean"] = np.array(results[k]["mean"], dtype=float)
        results[k]["var"] = np.array(results[k]["var"], dtype=float)
        results[k]["y_rmse_batches"] = np.array(results[k]["y_rmse_batches"], dtype=float)
        results[k]["y_crps_batches"] = np.array(results[k]["y_crps_batches"], dtype=float)

    theta_metrics = evaluate_theta_metrics(results, theta_true, stream_cfg.n_batches)

    y_metrics = {}
    for k in methods.keys():
        y_metrics[k] = {
            "y_rmse": float(np.mean(results[k]["y_rmse_batches"])),
            "y_crps": float(np.mean(results[k]["y_crps_batches"])),
        }

    metrics = {}
    for k in methods.keys():
        metrics[k] = {
            **theta_metrics[k],
            **y_metrics[k],
            "time_sec": float(times[k]),
        }

    if save_artifacts:
        # save_scenario_plots(name, stream_cfg, theta_true, results, methods, out_dir="figs")
        if eval_on_observation_y:
            Y_target_all = Y
        else:
            Y_target_all = np.array([
                y_physical(X[b], float(phi[b]))
                for b in range(stream_cfg.n_batches)
            ], dtype=float)

        save_scenario_plots(
            name=name,
            stream_cfg=stream_cfg,
            theta_true=theta_true,
            results=results,
            methods=methods,
            X=X,
            Y_target_all=Y_target_all,
            out_dir="figs",
            y_plot_batches=[0, stream_cfg.n_batches // 2, stream_cfg.n_batches - 1],
        )

        os.makedirs("outputs", exist_ok=True)
        safe_name = (
            name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("+", "plus")
                .replace("*", "star")
        )

        save_obj = {
            "scenario": name,
            "seed": seed,
            "theta_true": theta_true,
            "phi": phi,
            "results": {
                k: {
                    "theta_mean": results[k]["mean"],
                    "theta_var": results[k]["var"],
                    "theta_samples": results[k]["samples"],
                    "y_rmse_batches": results[k]["y_rmse_batches"],
                    "y_crps_batches": results[k]["y_crps_batches"],
                }
                for k in results.keys()
            },
            "metrics": metrics,
            "times_sec": times,
            "stream_cfg": stream_cfg,
            "pf_cfg": pf_cfg,
            "ogp_quad_n": ogp_quad_n,
        }

        with open(f"outputs/{safe_name}_seed{seed}_pf_results.pkl", "wb") as f:
            pickle.dump(save_obj, f)

    print(f"\n=== Scenario: {name}, seed={seed} ===")
    for k, tsec in times.items():
        print(f"{k:>24s}: {tsec:.3f} sec total")

    print("\nMetrics:")
    for k in methods.keys():
        print(
            f"{k:>24s}  "
            f"theta_RMSE={metrics[k]['theta_rmse']:.4f}  "
            f"theta_CRPS={metrics[k]['theta_crps']:.4f}  "
            f"y_RMSE={metrics[k]['y_rmse']:.4f}  "
            f"y_CRPS={metrics[k]['y_crps']:.4f}"
        )

    return data, results, times, metrics


def run_many_seeds(name: str,
                   stream_cfg: StreamConfig,
                   pf_cfg: PFConfig,
                   phi_schedule: Callable[[int], float],
                   seeds: List[int],
                   method2_gp_restarts: int = 0,
                   save_each_seed: bool = True,
                   eval_on_observation_y: bool = True,
                   ogp_quad_n: int = 201) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for seed in seeds:
        _, _, _, metrics = run_one_scenario(
            name=name,
            stream_cfg=stream_cfg,
            pf_cfg=pf_cfg,
            phi_schedule=phi_schedule,
            seed=seed,
            method2_gp_restarts=method2_gp_restarts,
            save_artifacts=save_each_seed,
            eval_on_observation_y=eval_on_observation_y,
            ogp_quad_n=ogp_quad_n,
        )

        for method_name, mm in metrics.items():
            rows.append({
                "scenario": name,
                "seed": seed,
                "method": method_name,
                "theta_rmse": mm["theta_rmse"],
                "theta_crps": mm["theta_crps"],
                "y_rmse": mm["y_rmse"],
                "y_crps": mm["y_crps"],
                "time_sec": mm["time_sec"],
            })

    raw_df = pd.DataFrame(rows)

    summary_df = (
        raw_df
        .groupby(["scenario", "method"], as_index=False)
        .agg(
            theta_rmse_mean=("theta_rmse", "mean"),
            theta_rmse_std=("theta_rmse", "std"),
            theta_crps_mean=("theta_crps", "mean"),
            theta_crps_std=("theta_crps", "std"),
            y_rmse_mean=("y_rmse", "mean"),
            y_rmse_std=("y_rmse", "std"),
            y_crps_mean=("y_crps", "mean"),
            y_crps_std=("y_crps", "std"),
            time_sec_mean=("time_sec", "mean"),
            time_sec_std=("time_sec", "std"),
        )
    )

    return raw_df, summary_df


def make_mean_std_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()

    for metric in ["theta_rmse", "theta_crps", "y_rmse", "y_crps", "time_sec"]:
        out[metric] = (
            out[f"{metric}_mean"].map(lambda x: f"{x:.4f}")
            + " ± "
            + out[f"{metric}_std"].fillna(0.0).map(lambda x: f"{x:.4f}")
        )

    out = out[["scenario", "method", "theta_rmse", "theta_crps", "y_rmse", "y_crps", "time_sec"]]
    return out


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    stream_cfg = StreamConfig(
        n_batches=20,
        batch_size=10,
        x_low=0.0,
        x_high=1.0,
        stratified_x=True,
    )

    pf_cfg = PFConfig(
        n_particles=400,
        theta_prior=(0.0, 3.0),
        sigma_obs=0.2,
        resample_ess_ratio=0.5,
        theta_move_std=0.02,
        log_ls_move_std=0.05,
        log_sv_move_std=0.05,
        log_nv_move_std=0.05,
        ls_bounds=(0.03, 1.5),
        sv_bounds=(1e-3, 50.0),
        nv_bounds=(1e-4, 5.0),
    )

    seeds = [101, 202, 303, 404, 505]

    phi_jump = make_phi_jump_schedule(
        values=[4.5, 7.5, 10.5, 7.5],
        n_batches=stream_cfg.n_batches
    )

    raw_jump, summary_jump = run_many_seeds(
        name="phi sudden change (4.5 -> 7.5 -> 10.5 -> 7.5)",
        stream_cfg=stream_cfg,
        pf_cfg=pf_cfg,
        phi_schedule=phi_jump,
        seeds=seeds,
        method2_gp_restarts=0,
        save_each_seed=True,
        eval_on_observation_y=True,
        ogp_quad_n=201,
    )

    phi_const = lambda b: 7.5
    raw_const, summary_const = run_many_seeds(
        name="phi constant (7.5)",
        stream_cfg=stream_cfg,
        pf_cfg=pf_cfg,
        phi_schedule=phi_const,
        seeds=seeds,
        method2_gp_restarts=0,
        save_each_seed=True,
        eval_on_observation_y=True,   # False -> use noiseless physical y
        ogp_quad_n=201,
    )

    phi_drift = lambda b: 4.0 + 0.1 * b
    raw_drift, summary_drift = run_many_seeds(
        name="phi drifting (4 + 0.1*batch)",
        stream_cfg=stream_cfg,
        pf_cfg=pf_cfg,
        phi_schedule=phi_drift,
        seeds=seeds,
        method2_gp_restarts=0,
        save_each_seed=True,
        eval_on_observation_y=True,
        ogp_quad_n=201,
    )

    # raw_all = pd.concat([raw_const, raw_drift], axis=0, ignore_index=True)
    # summary_all = pd.concat([summary_const, summary_drift], axis=0, ignore_index=True)
    raw_all = pd.concat([raw_const, raw_drift, raw_jump], axis=0, ignore_index=True)
    summary_all = pd.concat([summary_const, summary_drift, summary_jump], axis=0, ignore_index=True)
    mean_std_table = make_mean_std_table(summary_all)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 100)

    print("\n=== Per-seed raw metrics ===")
    print(raw_all.round(4).to_string(index=False))

    print("\n=== Aggregated summary ===")
    print(summary_all.round(4).to_string(index=False))

    print("\n=== Mean ± Std table ===")
    print(mean_std_table.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    raw_all.to_csv("outputs/multi_seed_raw_metrics.csv", index=False)
    summary_all.to_csv("outputs/multi_seed_summary_metrics.csv", index=False)
    mean_std_table.to_csv("outputs/multi_seed_mean_std_table.csv", index=False)

    print("\nSaved:")
    print("  outputs/multi_seed_raw_metrics.csv")
    print("  outputs/multi_seed_summary_metrics.csv")
    print("  outputs/multi_seed_mean_std_table.csv")