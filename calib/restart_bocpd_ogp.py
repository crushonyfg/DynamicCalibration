# =============================================================
# file: calib/restart_bocpd_ogp.py
# R-BOCPD with per-particle Orthogonal GP (OGP) discrepancy
#
# Supports arbitrary-dimensional x (dx) and theta (d_theta).
#
# Key changes from restart_bocpd_debug_260115_gpytorch.py:
# - Each expert's PF uses per-particle OGP discrepancy
#   (generalised from Method 4 in PF_ablation_exp.py)
# - Particles contain theta + GP hyperparameters
#   (length_scale, signal_var, noise_var)
# - Particles random-walk in both theta and log-hyperparameter space
# - Both PF weight update and BOCPD UMP use OGP-based likelihood
# - No separate delta_state; OGP is built on-the-fly per particle

# def grad_func(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # X     : [M, dx]     — 输入点
    # theta : [d_theta]    — 参数向量
    # 返回  : [M, d_theta] — 每个 x 处 η 对 θ 的 Jacobian
# =============================================================
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from .configs import BOCPDConfig, ModelConfig, PFConfig
from .emulator import Emulator
from .particles import ParticleSet
from .resampling import resample_indices


# =============================================================
# Multi-dimensional RBF kernel and quadrature grid (numpy)
# =============================================================

def rbf_kernel_nd(
    X1: np.ndarray,
    X2: np.ndarray,
    length_scale: float,
    signal_var: float,
) -> np.ndarray:
    """
    Isotropic RBF kernel for arbitrary dx.

    Parameters
    ----------
    X1 : [M1, dx]
    X2 : [M2, dx]

    Returns
    -------
    K : [M1, M2]
    """
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    diff = X1[:, None, :] - X2[None, :, :]          # [M1, M2, dx]
    sqdist = (diff ** 2).sum(axis=-1)                # [M1, M2]
    return signal_var * np.exp(-0.5 * sqdist / (length_scale ** 2))


def _parse_x_bounds(
    x_domain,
    dx: int,
) -> List[Tuple[float, float]]:
    """
    Normalise *x_domain* into a list of ``(lo, hi)`` per dimension.

    Accepts
    -------
    - ``(lo, hi)``              → broadcast to every dim
    - ``[(lo0,hi0), ...]``      → per-dim (len must equal dx)
    """
    if isinstance(x_domain, np.ndarray):
        x_domain = x_domain.tolist()
    if isinstance(x_domain, (list, tuple)) and len(x_domain) > 0:
        first = x_domain[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            assert len(x_domain) == dx, (
                f"x_domain has {len(x_domain)} entries but dx={dx}"
            )
            return [(float(lo), float(hi)) for lo, hi in x_domain]
        else:
            lo, hi = float(x_domain[0]), float(x_domain[1])
            return [(lo, hi)] * dx
    raise ValueError(f"Cannot interpret x_domain={x_domain!r}")


def make_quadrature_grid(
    x_bounds: List[Tuple[float, float]],
    quad_n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tensor-product trapezoidal quadrature grid in dx dimensions.

    Parameters
    ----------
    x_bounds : list of (lo, hi) per dimension  (length = dx)
    quad_n   : *approximate* total number of quadrature points.
               For 1-D this is exact; for multi-D we use
               ``ceil(quad_n^(1/dx))`` points per dimension.

    Returns
    -------
    X_quad : [n_total, dx]
    W_quad : [n_total]          (tensor-product trapezoidal weights)
    """
    dx = len(x_bounds)
    n_per_dim = max(int(round(quad_n ** (1.0 / dx))), 3)

    grids_1d: List[np.ndarray] = []
    weights_1d: List[np.ndarray] = []
    for lo, hi in x_bounds:
        g = np.linspace(lo, hi, n_per_dim)
        w = np.full(n_per_dim, (hi - lo) / max(n_per_dim - 1, 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        grids_1d.append(g)
        weights_1d.append(w)

    meshes = np.meshgrid(*grids_1d, indexing="ij")
    X_quad = np.column_stack([m.ravel() for m in meshes])   # [n_total, dx]

    w_meshes = np.meshgrid(*weights_1d, indexing="ij")
    W = w_meshes[0].copy()
    for wm in w_meshes[1:]:
        W = W * wm
    W_quad = W.ravel()                                       # [n_total]

    return X_quad, W_quad


# =============================================================
# Kernel-orthogonal GP  (multi-dim x & theta)
# =============================================================

class OrthogonalRBFGP:
    """
    Kernel-orthogonal GP discrepancy model.

    Supports **arbitrary** ``dx`` (dimension of x) and ``d_theta``
    (dimension of θ).

    Base kernel (isotropic RBF on x-space):
        k(x,x') = σ_f² exp( −‖x−x'‖²/(2 l²) )

    Given the simulator Jacobian
        g(x) = ∂η(x,θ)/∂θ ∈ ℝ^{d_theta}
    the kernel is orthogonalised w.r.t. the column space of g:

        k⊥(x,x') = k(x,x') − h(x)ᵀ H⁻¹ h(x')

    where
        h_j(x)  = ∫ k(x,ξ) g_j(ξ) dξ            ∈ ℝ^{d_theta}
        H_{jk}  = ∫∫ g_j(ξ) k(ξ,ξ') g_k(ξ') dξ dξ'  ∈ ℝ^{d_theta×d_theta}

    Integrals are approximated via tensor-product trapezoidal quadrature.

    Parameters
    ----------
    grad_func : callable(X, theta) -> np.ndarray
        X     : [M, dx]
        theta : [d_theta]
        Returns ∂η/∂θ of shape [M, d_theta].
    x_bounds : list of (lo, hi) per x-dimension.
    """

    def __init__(
        self,
        length_scale: float,
        signal_var: float,
        noise_var: float,
        x_bounds: List[Tuple[float, float]],
        grad_func: Callable,
        quad_n: int = 201,
        normalize_y: bool = True,
        jitter: float = 1e-8,
    ):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.x_bounds = x_bounds
        self.grad_func = grad_func
        self.quad_n = int(quad_n)
        self.normalize_y = bool(normalize_y)
        self.jitter = float(jitter)
        self.is_fit = False

    # ----- quadrature setup -----

    def _prepare_quadrature(self, theta: np.ndarray):
        """
        theta : [d_theta]  (1-D numpy array)
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        Xq, wq = make_quadrature_grid(self.x_bounds, self.quad_n)
        # Xq: [n_quad, dx],  wq: [n_quad]

        Gq = np.asarray(self.grad_func(Xq, theta), dtype=float)
        if Gq.ndim == 1:
            Gq = Gq.reshape(-1, 1)  # scalar theta → [n_quad, 1]
        d_theta = Gq.shape[1]

        Kqq = rbf_kernel_nd(Xq, Xq, self.length_scale, self.signal_var)
        # [n_quad, n_quad]

        Gw = Gq * wq[:, None]                       # [n_quad, d_theta]
        H = Gw.T @ Kqq @ Gw                         # [d_theta, d_theta]
        H += self.jitter * np.eye(d_theta)           # regularise

        self.Xq = Xq
        self.wq = wq
        self.Gw = Gw
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.d_theta = d_theta

    # ----- orthogonal-kernel helpers -----

    def _h_mat(self, X: np.ndarray) -> np.ndarray:
        """
        X : [M, dx]
        Returns h(X) : [M, d_theta]
        """
        X = np.atleast_2d(X)
        Kxq = rbf_kernel_nd(X, self.Xq, self.length_scale, self.signal_var)
        return Kxq @ self.Gw                         # [M, d_theta]

    def _k_perp(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """k⊥(X1, X2) : [M1, M2]"""
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        K = rbf_kernel_nd(X1, X2, self.length_scale, self.signal_var)
        h1 = self._h_mat(X1)                        # [M1, d_theta]
        h2 = self._h_mat(X2)                        # [M2, d_theta]
        correction = h1 @ self.H_inv @ h2.T         # [M1, M2]
        return K - correction

    def _k_perp_diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal of k⊥(X, X) : [M]"""
        X = np.atleast_2d(X)
        base_diag = np.full(X.shape[0], self.signal_var, dtype=float)
        h = self._h_mat(X)                           # [M, d_theta]
        # diag( h H⁻¹ hᵀ ) = rowwise dot of (h H⁻¹) and h
        correction = (h @ self.H_inv * h).sum(axis=1)
        return np.maximum(base_diag - correction, 1e-12)

    # ----- fit / predict -----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
    ):
        """
        Parameters
        ----------
        X     : [M, dx]
        y     : [M]
        theta : [d_theta]
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        self.theta = theta
        self._prepare_quadrature(theta)
        self.X_train = X.copy()

        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = max(float(np.std(y)), 1e-12)
            y_norm = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_norm = y.copy()

        K = self._k_perp(X, X)
        K += (self.noise_var + self.jitter) * np.eye(len(X))

        self.cF = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve(self.cF, y_norm, check_finite=False)
        self.is_fit = True
        return self

    def predict(
        self,
        Xtest: np.ndarray,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fit:
            raise RuntimeError("OrthogonalRBFGP must be fit before predict.")
        Xtest = np.atleast_2d(np.asarray(Xtest, dtype=float))

        Kxs = self._k_perp(Xtest, self.X_train)
        mean_norm = Kxs @ self.alpha
        mean = self.y_mean + self.y_std * mean_norm

        if not return_std:
            return mean, None

        v = cho_solve(self.cF, Kxs.T, check_finite=False)
        var_norm = self._k_perp_diag(Xtest) - np.sum(Kxs * v.T, axis=1)
        var_norm = np.maximum(var_norm, 1e-12)
        std = np.sqrt(np.maximum(self.y_std ** 2 * var_norm, 1e-12))
        return mean, std


# =============================================================
# Configuration for the OGP-based particle filter
# =============================================================

@dataclass
class OGPPFConfig:
    num_particles: int = 400
    resample_ess_ratio: float = 0.5
    resample_scheme: str = "systematic"

    # random-walk std
    theta_move_std: float = 0.03
    log_ls_move_std: float = 0.05
    log_sv_move_std: float = 0.05
    log_nv_move_std: float = 0.05

    # GP hyper bounds (in natural space)
    ls_bounds: Tuple[float, float] = (0.03, 1.5)
    sv_bounds: Tuple[float, float] = (1e-3, 50.0)
    nv_bounds: Tuple[float, float] = (1e-4, 5.0)

    # GP hyper initialisation (natural space)
    init_ls: float = 0.2
    init_sv: float = 5.0
    init_nv: float = 0.2
    init_hyp_spread: float = 0.2

    # OGP settings
    ogp_quad_n: int = 201
    # (lo, hi) → same bounds for every x-dim.
    # [(lo0,hi0), (lo1,hi1), ...] → per-dim bounds.
    x_domain: Any = (0.0, 1.0)
    min_hist_for_ogp: int = 5
    ogp_jitter: float = 1e-8
    ogp_normalize_y: bool = True

    # theta bounds (optional, applied element-wise after random walk)
    theta_lo: Optional[torch.Tensor] = None
    theta_hi: Optional[torch.Tensor] = None


# =============================================================
# OGP Particle Filter  (multi-dim x & theta)
# =============================================================

class OGPParticleFilter:
    """
    Particle filter whose particles carry (θ, ls, sv, nv).

    Discrepancy is modelled per-particle using a kernel-orthogonal GP
    (``OrthogonalRBFGP``).  Both the PF weight update and the BOCPD UMP
    computation call ``compute_loglik_batch`` which includes OGP
    discrepancy.

    Works with **arbitrary** ``dx`` and ``d_theta``.
    """

    def __init__(
        self,
        ogp_cfg: OGPPFConfig,
        prior_sampler: Callable,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        theta_anchor=None,
    ):
        self.cfg = ogp_cfg
        self.device = device
        self.dtype = dtype
        N = ogp_cfg.num_particles

        # --- theta  [N, d_theta] ---
        try:
            self.theta = prior_sampler(N, theta_anchor=theta_anchor).to(device, dtype)
        except TypeError:
            self.theta = prior_sampler(N).to(device, dtype)

        # --- log GP hypers  [N] ---
        self.log_ls = (
            math.log(ogp_cfg.init_ls)
            + ogp_cfg.init_hyp_spread * torch.randn(N, device=device, dtype=dtype)
        )
        self.log_sv = (
            math.log(ogp_cfg.init_sv)
            + ogp_cfg.init_hyp_spread * torch.randn(N, device=device, dtype=dtype)
        )
        self.log_nv = (
            math.log(ogp_cfg.init_nv)
            + ogp_cfg.init_hyp_spread * torch.randn(N, device=device, dtype=dtype)
        )
        self._clamp_hypers()

        # --- weights ---
        self.logw = torch.full((N,), -math.log(N), device=device, dtype=dtype)

    # ---- helpers ---------------------------------------------------------

    def _clamp_hypers(self):
        self.log_ls.clamp_(math.log(self.cfg.ls_bounds[0]),
                           math.log(self.cfg.ls_bounds[1]))
        self.log_sv.clamp_(math.log(self.cfg.sv_bounds[0]),
                           math.log(self.cfg.sv_bounds[1]))
        self.log_nv.clamp_(math.log(self.cfg.nv_bounds[0]),
                           math.log(self.cfg.nv_bounds[1]))

    @property
    def particles(self) -> ParticleSet:
        """Lightweight view for compatibility with BOCPD theta tests."""
        return ParticleSet(theta=self.theta, logw=self.logw)

    def normalize_(self):
        self.logw = self.logw - torch.logsumexp(self.logw, dim=0)

    def weights(self) -> torch.Tensor:
        self.normalize_()
        return torch.exp(self.logw)

    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / (w.pow(2).sum() + 1e-16))

    def gini(self) -> float:
        w = self.weights()
        N = w.numel()
        if N == 0:
            return 0.0
        w_sorted, _ = torch.sort(w)
        idx = torch.arange(1, N + 1, device=w.device, dtype=w.dtype)
        g = 1.0 - 2.0 * torch.sum(w_sorted * (N - idx + 0.5)) / N
        return float(g.clamp(0.0, 1.0))

    def theta_mean(self) -> torch.Tensor:
        w = self.weights().view(-1, 1)
        return (w * self.theta).sum(dim=0)

    # ---- OGP-based likelihood -------------------------------------------

    def compute_loglik_batch(
        self,
        X_batch: torch.Tensor,       # [B, dx]
        Y_batch: torch.Tensor,       # [B]
        X_hist: Optional[torch.Tensor],    # [M, dx] or None
        y_hist: Optional[torch.Tensor],    # [M] or None
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
        grad_func: Callable,
    ) -> torch.Tensor:
        """
        Per-particle log-likelihood for a batch, **including** OGP
        discrepancy.

        grad_func signature
        -------------------
        grad_func(X, theta) -> np.ndarray
            X     : [M, dx]
            theta : [d_theta]
            returns : [M, d_theta]

        Returns
        -------
        loglik : Tensor [N]
            loglik[i] = Σ_j log p(y_j | x_j, θ_i, OGP_i)
        """
        N = self.theta.shape[0]
        B = X_batch.shape[0]
        dx = X_batch.shape[1]
        sigma2 = sigma_eps ** 2
        loglik = torch.zeros(N, device=self.device, dtype=self.dtype)

        has_hist = (
            X_hist is not None
            and X_hist.numel() > 0
            and X_hist.shape[0] >= self.cfg.min_hist_for_ogp
        )

        # batch emulator predict for all particles at once
        mu_eta_all, var_eta_all = emulator.predict(X_batch, self.theta)  # [B, N]

        if not has_hist:
            mean_all = rho * mu_eta_all
            var_all = (rho ** 2 * var_eta_all + sigma2).clamp_min(1e-12)
            Y_col = Y_batch.view(-1, 1)
            loglik_bn = -0.5 * (
                torch.log(2 * math.pi * var_all)
                + (Y_col - mean_all) ** 2 / var_all
            )
            return loglik_bn.sum(dim=0)

        # ---- convert to numpy for OGP ----
        X_hist_np = X_hist.detach().cpu().numpy()        # [M, dx]
        y_hist_np = y_hist.detach().cpu().numpy()        # [M]
        X_batch_np = X_batch.detach().cpu().numpy()      # [B, dx]
        Y_batch_np = Y_batch.detach().cpu().numpy()      # [B]

        mu_eta_all_np = mu_eta_all.detach().cpu().numpy()   # [B, N]
        var_eta_all_np = var_eta_all.detach().cpu().numpy()

        mu_eta_hist_all, _ = emulator.predict(X_hist, self.theta)  # [M, N]
        mu_eta_hist_np = mu_eta_hist_all.detach().cpu().numpy()

        x_bounds = _parse_x_bounds(self.cfg.x_domain, dx)

        LOG2PI = math.log(2.0 * math.pi)

        for i in range(N):
            theta_i_np = self.theta[i].detach().cpu().numpy()   # [d_theta]
            resid = y_hist_np - rho * mu_eta_hist_np[:, i]      # [M]

            try:
                ogp = OrthogonalRBFGP(
                    length_scale=math.exp(float(self.log_ls[i])),
                    signal_var=math.exp(float(self.log_sv[i])),
                    noise_var=math.exp(float(self.log_nv[i])),
                    x_bounds=x_bounds,
                    grad_func=grad_func,
                    quad_n=self.cfg.ogp_quad_n,
                    normalize_y=self.cfg.ogp_normalize_y,
                    jitter=self.cfg.ogp_jitter,
                )
                ogp.fit(X_hist_np, resid, theta=theta_i_np)
                m_d, std_d = ogp.predict(X_batch_np, return_std=True)
                v_d = std_d ** 2
            except Exception:
                m_d = np.zeros(B)
                v_d = np.zeros(B)

            mean = rho * mu_eta_all_np[:, i] + m_d
            var = rho ** 2 * var_eta_all_np[:, i] + v_d + sigma2
            var = np.maximum(var, 1e-12)
            ll = -0.5 * (LOG2PI + np.log(var) + (Y_batch_np - mean) ** 2 / var)
            loglik[i] = float(ll.sum())

        return loglik

    # ---- step (weight update + resample + move) --------------------------

    def step_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        X_hist: Optional[torch.Tensor],
        y_hist: Optional[torch.Tensor],
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
        grad_func: Callable,
    ) -> Dict[str, Any]:
        loglik = self.compute_loglik_batch(
            X_batch, Y_batch, X_hist, y_hist,
            emulator, rho, sigma_eps, grad_func,
        )

        self.logw = self.logw + loglik
        self.normalize_()

        e = self.ess()
        g = self.gini()
        N = self.theta.shape[0]
        resampled = False

        if e < self.cfg.resample_ess_ratio * N:
            idx = resample_indices(self.weights(), scheme=self.cfg.resample_scheme)
            self.theta = self.theta[idx]
            self.log_ls = self.log_ls[idx]
            self.log_sv = self.log_sv[idx]
            self.log_nv = self.log_nv[idx]
            self.logw = torch.full_like(self.logw, -math.log(N))
            resampled = True
            self._move()

        return {"ess": e, "gini": g, "resampled": resampled}

    # ---- move (random walk) ----------------------------------------------

    def _move(self):
        self.theta = self.theta + self.cfg.theta_move_std * torch.randn_like(self.theta)
        if self.cfg.theta_lo is not None:
            lo = self.cfg.theta_lo.to(self.theta.device, self.theta.dtype)
            self.theta = torch.max(self.theta, lo.expand_as(self.theta))
        if self.cfg.theta_hi is not None:
            hi = self.cfg.theta_hi.to(self.theta.device, self.theta.dtype)
            self.theta = torch.min(self.theta, hi.expand_as(self.theta))

        self.log_ls += self.cfg.log_ls_move_std * torch.randn_like(self.log_ls)
        self.log_sv += self.cfg.log_sv_move_std * torch.randn_like(self.log_sv)
        self.log_nv += self.cfg.log_nv_move_std * torch.randn_like(self.log_nv)
        self._clamp_hypers()


# =============================================================
# Expert dataclass
# =============================================================

@dataclass
class Expert:
    run_length: int
    pf: OGPParticleFilter
    log_mass: float
    X_hist: torch.Tensor   # [M, dx]
    y_hist: torch.Tensor   # [M]


# =============================================================
# Utility: rolling statistics for LLR diagnostics
# =============================================================

class RollingStats:
    def __init__(self, window: int = 50):
        self.window = int(window)
        self.buf: deque = deque(maxlen=self.window)

    def update(self, x: float):
        self.buf.append(float(x))

    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else float("nan")

    def std(self) -> float:
        if len(self.buf) < 2:
            return 0.0
        return float(np.std(self.buf, ddof=1))

    def n(self) -> int:
        return len(self.buf)


# =============================================================
# Utility: build grad_func from an Emulator (autograd fallback)
# =============================================================

def make_grad_func_from_emulator(
    emulator: Emulator,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable:
    """
    Return a numpy-level ``grad_func(X, theta) -> [M, d_theta]``
    that wraps ``emulator.grad_theta`` (works for any dx / d_theta).
    """
    def grad_func(X_np: np.ndarray, theta_np: np.ndarray) -> np.ndarray:
        X = torch.from_numpy(np.atleast_2d(np.asarray(X_np, dtype=np.float64))).to(
            device, dtype
        )
        th = torch.from_numpy(
            np.atleast_1d(np.asarray(theta_np, dtype=np.float64)).reshape(1, -1)
        ).to(device, dtype)
        dmu, _ = emulator.grad_theta(X, th)          # [M, 1, d_theta]
        return dmu[:, 0, :].detach().cpu().numpy()    # [M, d_theta]

    return grad_func


# =============================================================
# BOCPD with OGP particle filters
# =============================================================

class BOCPD_OGP:
    """
    Restart-BOCPD whose experts use ``OGPParticleFilter``.

    Compared with the standard restart BOCPD:
    * No separate ``delta_state`` per expert.
    * Per-particle OGP discrepancy is built on-the-fly from expert history.
    * Both ``ump_batch`` (for mass update) and ``step_batch`` (PF weight
      update) compute likelihoods that include OGP discrepancy.
    * Supports arbitrary-dimensional x and theta.
    """

    def __init__(
        self,
        config: BOCPDConfig,
        ogp_pf_cfg: OGPPFConfig,
        grad_func: Callable,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        on_restart: Optional[Callable] = None,
        notify_on_restart: bool = True,
        **kwargs,
    ):
        self.config: BOCPDConfig = config
        self.ogp_pf_cfg: OGPPFConfig = ogp_pf_cfg
        self.grad_func: Callable = grad_func
        self.device = device
        self.dtype = dtype

        self.experts: List[Expert] = []
        self.t: int = 0
        self.restart_start_time: int = 0
        self._last_restart_t: int = -(10 ** 9)
        self.prev_max_ump: float = 0.0

        self.restart_margin: float = getattr(config, "restart_margin", 0.05)
        self.restart_cooldown: int = getattr(config, "restart_cooldown", 10)
        self.restart_criteria: str = getattr(config, "restart_criteria", "rank_change")

        self.on_restart = on_restart
        self.notify_on_restart = notify_on_restart

        self.theta_anchor = None

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _spawn_new_expert(
        self,
        prior_sampler: Callable,
        dx: int,
        log_mass: float,
    ) -> Expert:
        pf = OGPParticleFilter(
            ogp_cfg=self.ogp_pf_cfg,
            prior_sampler=prior_sampler,
            device=self.device,
            dtype=self.dtype,
            theta_anchor=self.theta_anchor,
        )
        return Expert(
            run_length=0,
            pf=pf,
            log_mass=log_mass,
            X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
        )

    def _append_hist_batch(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        max_len: int,
    ) -> None:
        if e.X_hist.numel() == 0:
            e.X_hist = X_batch.clone()
            e.y_hist = Y_batch.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, X_batch], dim=0)
            e.y_hist = torch.cat([e.y_hist, Y_batch], dim=0)

    def _expert_theta_mean(self, e: Expert) -> torch.Tensor:
        ps = e.pf.particles
        w = ps.weights().view(-1, 1)
        return (w * ps.theta).sum(dim=0)

    # ------------------------------------------------------------------
    # hazard helper
    # ------------------------------------------------------------------

    def _hazard(self, rl: int) -> float:
        if hasattr(self.config, "hazard_rate"):
            return float(getattr(self.config, "hazard_rate"))
        r_tensor = torch.tensor([rl], dtype=self.dtype, device=self.device)
        val = self.config.hazard(r_tensor)[0].item()
        return float(max(min(val, 1.0 - 1e-12), 1e-12))

    # ------------------------------------------------------------------
    # UMP (log predictive) using OGP likelihood
    # ------------------------------------------------------------------

    def ump_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> List[float]:
        out: List[float] = []
        for e in self.experts:
            loglik_n = e.pf.compute_loglik_batch(
                X_batch, Y_batch, e.X_hist, e.y_hist,
                emulator, model_cfg.rho, model_cfg.sigma_eps,
                self.grad_func,
            )
            logmix = torch.logsumexp(e.pf.logw + loglik_n, dim=0)
            out.append(float(logmix))
        return out

    # ------------------------------------------------------------------
    # prune: keep anchor + top mass experts
    # ------------------------------------------------------------------

    def _prune_keep_anchor(self, anchor_run_length: int, max_experts: int) -> None:
        if len(self.experts) <= max_experts:
            return

        anchor_idx: Optional[int] = None
        best_diff = 10 ** 9
        for i, e in enumerate(self.experts):
            diff = abs(e.run_length - anchor_run_length)
            if diff < best_diff:
                best_diff = diff
                anchor_idx = i

        sorted_idx = sorted(
            range(len(self.experts)),
            key=lambda i: self.experts[i].log_mass,
            reverse=True,
        )

        kept: List[int] = []
        for idx in sorted_idx:
            if idx == anchor_idx or len(kept) < max_experts - 1:
                kept.append(idx)
            if len(kept) >= max_experts:
                break
        if anchor_idx is not None and anchor_idx not in kept:
            kept[-1] = anchor_idx

        kept = sorted(set(kept))
        self.experts = [self.experts[i] for i in kept]

    # ------------------------------------------------------------------
    # batch update (main entry point)
    # ------------------------------------------------------------------

    def update_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable,
        verbose: bool = False,
    ) -> Dict[str, Any]:

        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0]
        dx = X_batch.shape[1]

        # ---- initialise first expert if needed ----
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(prior_sampler, dx, log_mass=0.0)
            self.experts.append(e0)
            self.restart_start_time = 0
            self.t = 0
            self._last_restart_t = -(10 ** 9)

        # ---- 1) per-expert log UMP ----
        log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
        log_umps_t = torch.tensor(log_umps, device=self.device, dtype=self.dtype)

        experts_pre = list(self.experts)
        log_umps_pre = list(log_umps)
        idx_pre = {id(e): i for i, e in enumerate(experts_pre)}

        def get_pre_log_ump(e):
            j = idx_pre.get(id(e), None)
            return None if j is None else float(log_umps_pre[j])

        # ---- 2) mass update ----
        prev_log_mass = torch.tensor(
            [e.log_mass for e in self.experts], device=self.device, dtype=self.dtype,
        )
        hazards = torch.tensor(
            [self._hazard(e.run_length) for e in self.experts],
            device=self.device, dtype=self.dtype,
        )
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        for i, e in enumerate(self.experts):
            e.run_length += batch_size
            e.log_mass = float(growth_log_mass[i])

        new_e = self._spawn_new_expert(prior_sampler, dx, log_mass=float(cp_log_mass))
        self.experts.append(new_e)

        masses = torch.tensor(
            [e.log_mass for e in self.experts], device=self.device, dtype=self.dtype,
        )
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # ---- 3) restart decision ----
        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)
        anchor_e = self._closest_by_run_length(anchor_rl)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        best_other_mass = 0.0
        s_star: Optional[int] = None
        cand_e: Optional[Expert] = None
        for e in self.experts:
            rl = e.run_length
            s = t_now - rl
            if s > r_old:
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = int(s)
                    cand_e = e

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

        did_restart = False
        msg_mode = "NO_RESTART"

        theta_pass = False
        theta_stat = None

        if (
            self.restart_criteria == "theta_test"
            and anchor_e is not None
            and cand_e is not None
        ):
            method = getattr(self.config, "restart_theta_test", "energy")
            if method == "credible":
                z = float(getattr(self.config, "restart_cred_z", 2.0))
                frac = float(getattr(self.config, "restart_cred_frac", 0.5))
                rate, theta_pass = self._credible_nonoverlap(anchor_e, cand_e, z, frac)
                theta_stat = rate
            elif method == "sw":
                n_proj = int(getattr(self.config, "restart_sw_proj", 32))
                theta_stat = self._sliced_wasserstein(anchor_e, cand_e, n_proj)
                tau = float(getattr(self.config, "restart_theta_tau", 0.1))
                theta_pass = theta_stat > tau
            else:
                theta_stat = self._energy_distance(anchor_e, cand_e)
                tau = float(getattr(self.config, "restart_theta_tau", 0.1))
                theta_pass = theta_stat > tau
        elif self.restart_criteria != "theta_test":
            theta_pass = True
            theta_stat = 0.0

        if (
            getattr(self.config, "use_restart", True)
            and s_star is not None
            and theta_pass
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        ):
            did_restart = True
            print(
                f"\n[DEBUG BEFORE RESTART] t={t_now}, r_old={r_old}, s_star={s_star}"
            )
            print(
                f"  p_anchor={p_anchor:.6f}, best_other_mass={best_other_mass:.6f}, "
                f"p_cp={p_cp:.6f}"
            )
            print(
                f"  theta_test="
                f"{getattr(self.config, 'restart_theta_test', 'energy')} "
                f"stat={theta_stat} pass={theta_pass}"
            )
            sorted_experts = sorted(
                self.experts, key=lambda e: e.log_mass, reverse=True,
            )
            for ii, ee in enumerate(sorted_experts[:5]):
                tm = self._expert_theta_mean(ee).detach().cpu().numpy()
                print(
                    f"    Expert[{ii}] rl={ee.run_length}, "
                    f"mass={math.exp(ee.log_mass):.6f}, theta_mean={tm}"
                )
            print("--------------------------------------------------------\n")
            self._last_restart_t = t_now

            if getattr(self.config, "use_backdated_restart", False):
                self.restart_start_time = int(s_star)
                new_anchor_rl = max(t_now - self.restart_start_time, 0)
                keep_e = next(
                    (e for e in self.experts if e.run_length == new_anchor_rl),
                    None,
                )
                if keep_e is None and len(self.experts) > 0:
                    keep_e = min(
                        self.experts,
                        key=lambda e: abs(e.run_length - new_anchor_rl),
                    )
                    keep_e.run_length = new_anchor_rl
                self.experts = (
                    [keep_e] if keep_e is not None else self.experts[:1]
                )
                msg_mode = "BACKDATED r←s*"
            else:
                self.restart_start_time = t_now
                r0 = self._spawn_new_expert(prior_sampler, dx, log_mass=0.0)
                self.experts = [r0]
                msg_mode = "ALGO2 r←t+1"

            if self.on_restart is not None and self.notify_on_restart:
                self.on_restart(
                    int(t_now),
                    int(self.restart_start_time),
                    int(s_star) if s_star is not None else None,
                    int(anchor_rl),
                    float(p_anchor),
                    float(best_other_mass),
                )

        if not did_restart:
            anchor_run_length = max(t_now - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        # ---- 4) PF step (BEFORE appending, so OGP uses OLD history) ----
        pf_diags: List[Dict[str, Any]] = []
        for e in self.experts:
            diag = e.pf.step_batch(
                X_batch, Y_batch,
                e.X_hist, e.y_hist,
                emulator, model_cfg.rho, model_cfg.sigma_eps,
                self.grad_func,
            )
            pf_diags.append(diag)

        # ---- 5) append to history (AFTER PF step) ----
        for e in self.experts:
            self._append_hist_batch(
                e, X_batch, Y_batch, self.config.max_run_length,
            )

        # ---- 6) advance clock ----
        self.t += batch_size

        # ---- 7) diagnostics ----
        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)
        if log_umps:
            self.prev_max_ump = max(log_umps)

        experts_debug: List[Dict[str, Any]] = []
        for idx_e, e in enumerate(self.experts):
            try:
                tm = self._expert_theta_mean(e).detach().cpu().tolist()
            except Exception:
                tm = None
            mass = math.exp(e.log_mass)
            lu = float(log_umps[idx_e]) if idx_e < len(log_umps) else None
            experts_debug.append({
                "index": idx_e,
                "run_length": int(e.run_length),
                "start_time": int(t_now - e.run_length),
                "log_mass": float(e.log_mass),
                "mass": float(mass),
                "theta_mean": tm,
                "log_ump": lu,
            })

        if did_restart:
            print(
                f"[R-BOCPD-OGP][batch] Restart at t={t_now}: "
                f"mode={msg_mode}, r_old={r_old}, "
                f"r_new={self.restart_start_time}, "
                f"s_star={s_star}, p_anchor={p_anchor:.4g}, p_cp={p_cp:.4g}"
            )
            for info in experts_debug:
                print(
                    f"  expert#{info['index']}: rl={info['run_length']}, "
                    f"start={info['start_time']}, mass={info['mass']:.4g}, "
                    f"log_ump={info['log_ump']}"
                )

        # ---- LLR diagnostics ----
        def _single_expert_log_ump(e_: Expert) -> float:
            ll_n = e_.pf.compute_loglik_batch(
                X_batch, Y_batch, e_.X_hist, e_.y_hist,
                emulator, model_cfg.rho, model_cfg.sigma_eps,
                self.grad_func,
            )
            return float(torch.logsumexp(e_.pf.logw + ll_n, dim=0))

        log_ump_anchor = None
        log_ump_cand = None
        if anchor_e is not None:
            log_ump_anchor = get_pre_log_ump(anchor_e)
            if log_ump_anchor is None:
                log_ump_anchor = _single_expert_log_ump(anchor_e)
        if cand_e is not None:
            log_ump_cand = get_pre_log_ump(cand_e)
            if log_ump_cand is None:
                log_ump_cand = _single_expert_log_ump(cand_e)

        delta_ll_pair = None
        if log_ump_anchor is not None and log_ump_cand is not None:
            delta_ll_pair = float(log_ump_cand - log_ump_anchor)

        log_odds_mass = None
        if anchor_e is not None and cand_e is not None:
            log_odds_mass = float(cand_e.log_mass - anchor_e.log_mass)

        h_log = float(math.log1p(self.restart_margin))

        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": pf_diags,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
            "experts_debug": experts_debug,
            "anchor_rl": (
                int(anchor_e.run_length) if anchor_e is not None else None
            ),
            "cand_rl": (
                int(cand_e.run_length) if cand_e is not None else None
            ),
            "delta_ll_pair": delta_ll_pair,
            "log_odds_mass": log_odds_mass,
            "h_log": h_log,
            "log_ump_anchor": log_ump_anchor,
            "log_ump_cand": log_ump_cand,
        }

    # ------------------------------------------------------------------
    # theta test helpers (identical to original BOCPD)
    # ------------------------------------------------------------------

    def _theta_particles(self, e: Expert):
        ps = e.pf.particles
        theta = ps.theta.detach()
        w = ps.weights().detach()
        w = w / (w.sum() + 1e-12)
        return theta, w

    def _weighted_mean_var(self, theta: torch.Tensor, w: torch.Tensor):
        w = w.view(-1, 1)
        mu = (w * theta).sum(dim=0)
        diff2 = (theta - mu).pow(2)
        var = (w * diff2).sum(dim=0)
        return mu, var

    def _credible_nonoverlap(
        self, e1: Expert, e2: Expert, z: float = 2.0, frac: float = 0.5,
    ):
        th1, w1 = self._theta_particles(e1)
        th2, w2 = self._theta_particles(e2)
        mu1, var1 = self._weighted_mean_var(th1, w1)
        mu2, var2 = self._weighted_mean_var(th2, w2)
        sd1 = torch.sqrt(var1 + 1e-12)
        sd2 = torch.sqrt(var2 + 1e-12)
        lo1, hi1 = mu1 - z * sd1, mu1 + z * sd1
        lo2, hi2 = mu2 - z * sd2, mu2 + z * sd2
        nonoverlap = (hi1 < lo2) | (hi2 < lo1)
        rate = nonoverlap.float().mean().item()
        return rate, rate >= frac

    def _energy_distance(self, e1: Expert, e2: Expert):
        X, a = self._theta_particles(e1)
        Y, b = self._theta_particles(e2)
        XY = torch.cdist(X, Y, p=2)
        XX = torch.cdist(X, X, p=2)
        YY = torch.cdist(Y, Y, p=2)
        a2, b2 = a.view(-1, 1), b.view(-1, 1)
        Exy = (a2 * b.view(1, -1) * XY).sum()
        Exx = (a2 * a.view(1, -1) * XX).sum()
        Eyy = (b2 * b.view(1, -1) * YY).sum()
        D = 2.0 * Exy - Exx - Eyy
        return float(D.clamp_min(0.0).item())

    def _sliced_wasserstein(self, e1: Expert, e2: Expert, n_proj: int = 32):
        X, a = self._theta_particles(e1)
        Y, b = self._theta_particles(e2)
        d = X.shape[1]
        U = torch.randn(n_proj, d, device=X.device, dtype=X.dtype)
        U = U / (U.norm(dim=1, keepdim=True) + 1e-12)

        def w1_1d(x, w, y, v):
            xs, ix = torch.sort(x)
            ws = w[ix]
            ys, iy = torch.sort(y)
            vs = v[iy]
            cwx = torch.cumsum(ws, dim=0)
            cwy = torch.cumsum(vs, dim=0)
            grid = torch.unique(torch.cat([cwx, cwy], dim=0))
            ixg = torch.searchsorted(cwx, grid, right=True).clamp(
                0, len(xs) - 1,
            )
            iyg = torch.searchsorted(cwy, grid, right=True).clamp(
                0, len(ys) - 1,
            )
            qx, qy = xs[ixg], ys[iyg]
            du = torch.cat([grid[:1], grid[1:] - grid[:-1]], dim=0)
            return (du * (qx - qy).abs()).sum()

        W = 0.0
        for p in range(n_proj):
            x1 = X @ U[p]
            x2 = Y @ U[p]
            W += w1_1d(x1, a, x2, b)
        return float((W / n_proj).item())

    def _closest_by_run_length(self, target_rl: int) -> Optional[Expert]:
        if len(self.experts) == 0:
            return None
        return min(self.experts, key=lambda e: abs(e.run_length - target_rl))
