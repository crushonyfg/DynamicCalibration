# =============================================================
# file: calib/restart_bocpd_ogp_gpytorch.py
# R-BOCPD with per-particle Orthogonal GP — PyTorch batched + GPU
#
# Key improvements over restart_bocpd_ogp.py:
#  1. All N particles processed simultaneously via batched linear algebra
#     (torch.linalg.cholesky, torch.cholesky_solve, torch.bmm)
#  2. GPU-accelerated: all OGP computation stays on device
#  3. Chunked processing to control memory (particle_chunk_size)
#  4. Cached loglik: UMP and PF step share the same loglik_n (exact 2x)
#  5. make_fast_batched_grad_func: single forward+backward for all particles
# =============================================================
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .configs import BOCPDConfig, ModelConfig, PFConfig
from .emulator import Emulator
from .particles import ParticleSet
from .resampling import resample_indices


# =============================================================
# Helpers
# =============================================================

def _parse_x_bounds(x_domain, dx: int) -> List[Tuple[float, float]]:
    if isinstance(x_domain, np.ndarray):
        x_domain = x_domain.tolist()
    if isinstance(x_domain, (list, tuple)) and len(x_domain) > 0:
        first = x_domain[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            assert len(x_domain) == dx
            return [(float(lo), float(hi)) for lo, hi in x_domain]
        else:
            lo, hi = float(x_domain[0]), float(x_domain[1])
            return [(lo, hi)] * dx
    raise ValueError(f"Cannot interpret x_domain={x_domain!r}")


def _make_quadrature_grid_torch(
    x_bounds: List[Tuple[float, float]],
    quad_n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = len(x_bounds)
    n_per_dim = max(int(round(quad_n ** (1.0 / dx))), 3)

    grids_1d, weights_1d = [], []
    for lo, hi in x_bounds:
        g = np.linspace(lo, hi, n_per_dim)
        w = np.full(n_per_dim, (hi - lo) / max(n_per_dim - 1, 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        grids_1d.append(g)
        weights_1d.append(w)

    meshes = np.meshgrid(*grids_1d, indexing="ij")
    X_quad = np.column_stack([m.ravel() for m in meshes])

    w_meshes = np.meshgrid(*weights_1d, indexing="ij")
    W = w_meshes[0].copy()
    for wm in w_meshes[1:]:
        W = W * wm

    return (
        torch.tensor(X_quad, device=device, dtype=dtype),
        torch.tensor(W.ravel(), device=device, dtype=dtype),
    )


# =============================================================
# Batched Orthogonal RBF GP
# =============================================================

class BatchedOrthogonalRBFGP:
    """Process all N particles' OGP fit+predict in one batched call."""

    def __init__(
        self,
        x_bounds: List[Tuple[float, float]],
        quad_n: int,
        jitter: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.Xq, self.wq = _make_quadrature_grid_torch(
            x_bounds, quad_n, device, dtype,
        )
        self.jitter = jitter
        self.device = device
        self.dtype = dtype
        self._sqdist_qq: Optional[torch.Tensor] = None

    @property
    def sqdist_qq(self) -> torch.Tensor:
        if self._sqdist_qq is None:
            self._sqdist_qq = torch.cdist(
                self.Xq, self.Xq, p=2,
            ).pow(2)
        return self._sqdist_qq

    def fit_and_predict(
        self,
        X_train: torch.Tensor,      # [M, dx]
        resid_all: torch.Tensor,     # [Nc, M]
        X_test: torch.Tensor,        # [B, dx]
        ls: torch.Tensor,            # [Nc]
        sv: torch.Tensor,            # [Nc]
        nv: torch.Tensor,            # [Nc]
        Gq_all: torch.Tensor,        # [Nc, n_quad, d_theta]
        normalize_y: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mean_d : [Nc, B]  OGP discrepancy mean
        var_d  : [Nc, B]  OGP discrepancy variance
        """
        Nc, M = resid_all.shape
        B = X_test.shape[0]
        d_theta = Gq_all.shape[2]

        # --- normalise residuals ---
        if normalize_y and M > 1:
            y_mean = resid_all.mean(dim=1, keepdim=True)
            y_std = resid_all.std(dim=1, keepdim=True).clamp_min(1e-12)
            resid_norm = (resid_all - y_mean) / y_std
        else:
            y_mean = torch.zeros(Nc, 1, device=self.device, dtype=self.dtype)
            y_std = torch.ones(Nc, 1, device=self.device, dtype=self.dtype)
            resid_norm = resid_all

        # --- shared squared distances (independent of hypers) ---
        sqdist_tq = torch.cdist(X_train, self.Xq, p=2).pow(2)   # [M, nq]
        sqdist_bq = torch.cdist(X_test, self.Xq, p=2).pow(2)    # [B, nq]
        sqdist_tt = torch.cdist(X_train, X_train, p=2).pow(2)    # [M, M]
        sqdist_bt = torch.cdist(X_test, X_train, p=2).pow(2)     # [B, M]

        ls2 = ls.pow(2)[:, None, None]
        sv3 = sv[:, None, None]

        # --- H and H_inv ---
        Kqq = sv3 * torch.exp(-0.5 * self.sqdist_qq.unsqueeze(0) / ls2)
        Gw = Gq_all * self.wq[None, :, None]
        H = torch.bmm(Gw.transpose(1, 2), torch.bmm(Kqq, Gw))
        H = H + self.jitter * torch.eye(
            d_theta, device=self.device, dtype=self.dtype,
        ).unsqueeze(0)
        H_inv = torch.linalg.inv(H)

        # --- h matrices ---
        K_tq = sv3 * torch.exp(-0.5 * sqdist_tq.unsqueeze(0) / ls2)
        h_train = torch.bmm(K_tq, Gw)                  # [Nc, M, dth]

        K_bq = sv3 * torch.exp(-0.5 * sqdist_bq.unsqueeze(0) / ls2)
        h_test = torch.bmm(K_bq, Gw)                   # [Nc, B, dth]

        # --- k_perp(train, train) + Cholesky ---
        K_tt = sv3 * torch.exp(-0.5 * sqdist_tt.unsqueeze(0) / ls2)
        corr_tt = torch.bmm(
            torch.bmm(h_train, H_inv), h_train.transpose(1, 2),
        )
        K_tt_perp = K_tt - corr_tt
        eye_M = torch.eye(M, device=self.device, dtype=self.dtype).unsqueeze(0)
        K_tt_perp = K_tt_perp + (nv[:, None, None] + self.jitter) * eye_M

        L, info = torch.linalg.cholesky_ex(K_tt_perp)
        failed = info != 0
        if failed.any():
            K_retry = K_tt_perp[failed] + 1e-4 * eye_M
            L_retry, info2 = torch.linalg.cholesky_ex(K_retry)
            L[failed] = L_retry
            still = torch.zeros(Nc, dtype=torch.bool, device=self.device)
            still[failed] = info2 != 0
            failed = still

        alpha = torch.cholesky_solve(resid_norm.unsqueeze(2), L)  # [Nc,M,1]

        # --- k_perp(test, train) ---
        K_bt = sv3 * torch.exp(-0.5 * sqdist_bt.unsqueeze(0) / ls2)
        corr_bt = torch.bmm(
            torch.bmm(h_test, H_inv), h_train.transpose(1, 2),
        )
        K_bt_perp = K_bt - corr_bt                      # [Nc, B, M]

        # --- predictive mean ---
        mean_norm = torch.bmm(K_bt_perp, alpha).squeeze(2)   # [Nc, B]
        mean_d = y_mean + y_std * mean_norm

        # --- predictive variance ---
        v = torch.cholesky_solve(K_bt_perp.transpose(1, 2), L)  # [Nc, M, B]
        kp_diag = sv.unsqueeze(1) - (
            torch.bmm(h_test, H_inv) * h_test
        ).sum(dim=2)
        kp_diag = kp_diag.clamp_min(1e-12)
        var_norm = kp_diag - (K_bt_perp * v.transpose(1, 2)).sum(dim=2)
        var_norm = var_norm.clamp_min(1e-12)
        var_d = y_std.pow(2) * var_norm

        if failed.any():
            mean_d[failed] = 0.0
            var_d[failed] = 0.0

        return mean_d, var_d


# =============================================================
# Config
# =============================================================

@dataclass
class OGPPFConfig:
    num_particles: int = 400
    resample_ess_ratio: float = 0.5
    resample_scheme: str = "systematic"

    theta_move_std: float = 0.03
    log_ls_move_std: float = 0.05
    log_sv_move_std: float = 0.05
    log_nv_move_std: float = 0.05

    ls_bounds: Tuple[float, float] = (0.03, 1.5)
    sv_bounds: Tuple[float, float] = (1e-3, 50.0)
    nv_bounds: Tuple[float, float] = (1e-4, 5.0)

    init_ls: float = 0.2
    init_sv: float = 5.0
    init_nv: float = 0.2
    init_hyp_spread: float = 0.2

    ogp_quad_n: int = 201
    x_domain: Any = (0.0, 1.0)
    min_hist_for_ogp: int = 5
    ogp_jitter: float = 1e-8
    ogp_normalize_y: bool = True

    theta_lo: Optional[torch.Tensor] = None
    theta_hi: Optional[torch.Tensor] = None

    particle_chunk_size: int = 256
    max_hist: int = 0


# =============================================================
# OGP Particle Filter  (batched)
# =============================================================

class OGPParticleFilter:

    def __init__(
        self,
        ogp_cfg: OGPPFConfig,
        prior_sampler: Callable,
        batched_grad_func: Callable,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        theta_anchor=None,
    ):
        self.cfg = ogp_cfg
        self.device = device
        self.dtype = dtype
        self.batched_grad_func = batched_grad_func
        N = ogp_cfg.num_particles

        try:
            self.theta = prior_sampler(N, theta_anchor=theta_anchor).to(device, dtype)
        except TypeError:
            self.theta = prior_sampler(N).to(device, dtype)

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

        self.logw = torch.full((N,), -math.log(N), device=device, dtype=dtype)

        self._batched_ogp: Optional[BatchedOrthogonalRBFGP] = None

    def _get_batched_ogp(self, dx: int) -> BatchedOrthogonalRBFGP:
        if self._batched_ogp is None:
            x_bounds = _parse_x_bounds(self.cfg.x_domain, dx)
            self._batched_ogp = BatchedOrthogonalRBFGP(
                x_bounds=x_bounds,
                quad_n=self.cfg.ogp_quad_n,
                jitter=self.cfg.ogp_jitter,
                device=self.device,
                dtype=self.dtype,
            )
        return self._batched_ogp

    def _clamp_hypers(self):
        self.log_ls.clamp_(
            math.log(self.cfg.ls_bounds[0]), math.log(self.cfg.ls_bounds[1]),
        )
        self.log_sv.clamp_(
            math.log(self.cfg.sv_bounds[0]), math.log(self.cfg.sv_bounds[1]),
        )
        self.log_nv.clamp_(
            math.log(self.cfg.nv_bounds[0]), math.log(self.cfg.nv_bounds[1]),
        )

    @property
    def particles(self) -> ParticleSet:
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

    # ---- batched predict (with OGP discrepancy) ------------------------

    def predict_batch(
        self,
        X_batch: torch.Tensor,
        X_hist: Optional[torch.Tensor],
        y_hist: Optional[torch.Tensor],
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Particle-weighted predictive mean and variance **including**
        OGP discrepancy.

        Returns
        -------
        mu_mix : [B]   weighted mixture mean
        var_mix : [B]  weighted mixture variance
        """
        N = self.theta.shape[0]
        B = X_batch.shape[0]
        dx = X_batch.shape[1]
        sigma2 = sigma_eps ** 2

        mu_eta, var_eta = emulator.predict(X_batch, self.theta)  # [B, N]

        has_hist = (
            X_hist is not None
            and X_hist.numel() > 0
            and X_hist.shape[0] >= self.cfg.min_hist_for_ogp
        )

        if not has_hist:
            mean_all = rho * mu_eta
            var_all = (rho ** 2 * var_eta + sigma2).clamp_min(1e-12)
        else:
            mu_eta_hist, _ = emulator.predict(X_hist, self.theta)
            bogp = self._get_batched_ogp(dx)
            chunk = self.cfg.particle_chunk_size

            mean_all = torch.zeros(
                B, N, device=self.device, dtype=self.dtype,
            )
            var_all = torch.zeros(
                B, N, device=self.device, dtype=self.dtype,
            )

            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                resid_c = (
                    y_hist.unsqueeze(0).expand(c1 - c0, -1)
                    - rho * mu_eta_hist[:, c0:c1].T
                )
                Gq_c = self.batched_grad_func(
                    bogp.Xq, self.theta[c0:c1],
                )
                mean_d, var_d = bogp.fit_and_predict(
                    X_hist, resid_c, X_batch,
                    self.log_ls[c0:c1].exp(),
                    self.log_sv[c0:c1].exp(),
                    self.log_nv[c0:c1].exp(),
                    Gq_c,
                    normalize_y=self.cfg.ogp_normalize_y,
                )
                mean_all[:, c0:c1] = (
                    rho * mu_eta[:, c0:c1] + mean_d.T
                )
                var_all[:, c0:c1] = (
                    rho ** 2 * var_eta[:, c0:c1] + var_d.T + sigma2
                )

            var_all = var_all.clamp_min(1e-12)

        w = self.weights().unsqueeze(0)                      # [1, N]
        mu_mix = (w * mean_all).sum(dim=1)                   # [B]
        var_mix = (
            (w * (var_all + mean_all ** 2)).sum(dim=1)
            - mu_mix ** 2
        ).clamp_min(1e-12)
        return mu_mix, var_mix

    # ---- batched loglik --------------------------------------------------

    def compute_loglik_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        X_hist: Optional[torch.Tensor],
        y_hist: Optional[torch.Tensor],
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
    ) -> torch.Tensor:
        """Returns loglik [N]."""
        N = self.theta.shape[0]
        B = X_batch.shape[0]
        dx = X_batch.shape[1]
        sigma2 = sigma_eps ** 2

        mu_eta_batch, var_eta_batch = emulator.predict(X_batch, self.theta)

        has_hist = (
            X_hist is not None
            and X_hist.numel() > 0
            and X_hist.shape[0] >= self.cfg.min_hist_for_ogp
        )

        if not has_hist:
            mean_all = rho * mu_eta_batch
            var_all = (rho ** 2 * var_eta_batch + sigma2).clamp_min(1e-12)
            Y_col = Y_batch.view(-1, 1)
            ll = -0.5 * (
                torch.log(2 * math.pi * var_all)
                + (Y_col - mean_all) ** 2 / var_all
            )
            return ll.sum(dim=0)

        mu_eta_hist, _ = emulator.predict(X_hist, self.theta)

        bogp = self._get_batched_ogp(dx)
        chunk = self.cfg.particle_chunk_size
        loglik = torch.zeros(N, device=self.device, dtype=self.dtype)

        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)

            resid_c = (
                y_hist.unsqueeze(0).expand(c1 - c0, -1)
                - rho * mu_eta_hist[:, c0:c1].T
            )

            Gq_c = self.batched_grad_func(bogp.Xq, self.theta[c0:c1])

            mean_d, var_d = bogp.fit_and_predict(
                X_hist,
                resid_c,
                X_batch,
                self.log_ls[c0:c1].exp(),
                self.log_sv[c0:c1].exp(),
                self.log_nv[c0:c1].exp(),
                Gq_c,
                normalize_y=self.cfg.ogp_normalize_y,
            )

            mean_c = rho * mu_eta_batch[:, c0:c1].T + mean_d
            var_c = (
                rho ** 2 * var_eta_batch[:, c0:c1].T
                + var_d + sigma2
            ).clamp_min(1e-12)

            ll = -0.5 * (
                torch.log(2 * math.pi * var_c)
                + (Y_batch.unsqueeze(0) - mean_c) ** 2 / var_c
            )
            loglik[c0:c1] = ll.sum(dim=1)

        return loglik

    # ---- step ------------------------------------------------------------

    def step_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        X_hist: Optional[torch.Tensor],
        y_hist: Optional[torch.Tensor],
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
        cached_loglik: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if cached_loglik is not None:
            loglik = cached_loglik
        else:
            loglik = self.compute_loglik_batch(
                X_batch, Y_batch, X_hist, y_hist,
                emulator, rho, sigma_eps,
            )

        self.logw = self.logw + loglik
        self.normalize_()

        e = self.ess()
        g = self.gini()
        N = self.theta.shape[0]
        resampled = False

        if e < self.cfg.resample_ess_ratio * N:
            idx = resample_indices(
                self.weights(), scheme=self.cfg.resample_scheme,
            )
            self.theta = self.theta[idx]
            self.log_ls = self.log_ls[idx]
            self.log_sv = self.log_sv[idx]
            self.log_nv = self.log_nv[idx]
            self.logw = torch.full_like(self.logw, -math.log(N))
            resampled = True
            self._move()

        return {"ess": e, "gini": g, "resampled": resampled}

    def _move(self):
        self.theta = (
            self.theta
            + self.cfg.theta_move_std * torch.randn_like(self.theta)
        )
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
# Expert
# =============================================================

@dataclass
class Expert:
    run_length: int
    pf: OGPParticleFilter
    log_mass: float
    X_hist: torch.Tensor
    y_hist: torch.Tensor


# =============================================================
# RollingStats
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
# Gradient utilities
# =============================================================

def make_batched_grad_func_from_emulator(
    emulator: Emulator,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable:
    """
    Wraps ``emulator.grad_theta`` into the batched interface.

    Slow for DeterministicSimulator because grad_theta loops internally.
    Prefer ``make_fast_batched_grad_func`` when the simulator function
    is a pure torch function that supports arbitrary batch size.
    """
    def batched_grad_func(
        X: torch.Tensor, thetas: torch.Tensor,
    ) -> torch.Tensor:
        X = X.to(device, dtype)
        thetas = thetas.to(device, dtype)
        dmu, _ = emulator.grad_theta(X, thetas)   # [M, N, d_theta]
        return dmu.permute(1, 0, 2).contiguous()   # [N, M, d_theta]

    return batched_grad_func


def make_fast_batched_grad_func(
    sim_func: Callable,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable:
    """
    Single forward+backward for all (particle, quadrature-point) pairs.

    Works when ``sim_func(X, theta)`` operates row-wise so that
    ``y[k]`` depends only on ``(X[k], theta[k])``.  This is true for
    ``DeterministicSimulator.func``.  ~100-1000x faster than the
    per-element autograd in ``DeterministicSimulator.grad_theta``.
    """
    def batched_grad_func(
        X: torch.Tensor, thetas: torch.Tensor,
    ) -> torch.Tensor:
        N, d_theta = thetas.shape
        M = X.shape[0]
        with torch.enable_grad():
            X_flat = X.unsqueeze(0).expand(N, -1, -1).reshape(N * M, -1)
            th_flat = (
                thetas.unsqueeze(1)
                .expand(-1, M, -1)
                .reshape(N * M, -1)
                .detach()
                .requires_grad_(True)
            )
            y = sim_func(X_flat.to(device, dtype), th_flat.to(device, dtype))
            if y.dim() > 1:
                y = y.view(-1)
            grad = torch.autograd.grad(
                y.sum(), th_flat, create_graph=False,
            )[0]
        return grad.reshape(N, M, d_theta)

    return batched_grad_func


# =============================================================
# BOCPD with batched OGP particle filters
# =============================================================

class BOCPD_OGP:

    def __init__(
        self,
        config: BOCPDConfig,
        ogp_pf_cfg: OGPPFConfig,
        batched_grad_func: Callable,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        on_restart: Optional[Callable] = None,
        notify_on_restart: bool = True,
        **kwargs,
    ):
        self.config = config
        self.ogp_pf_cfg = ogp_pf_cfg
        self.batched_grad_func = batched_grad_func
        self.device = device
        self.dtype = dtype

        self.experts: List[Expert] = []
        self.t: int = 0
        self.restart_start_time: int = 0
        self._last_restart_t: int = -(10 ** 9)
        self.prev_max_ump: float = 0.0

        self.restart_margin = float(getattr(config, "restart_margin", 0.05))
        self.restart_cooldown = int(getattr(config, "restart_cooldown", 10))
        self.restart_criteria = str(
            getattr(config, "restart_criteria", "rank_change"),
        )

        self.on_restart = on_restart
        self.notify_on_restart = notify_on_restart
        self.theta_anchor = None

    # ---- internal helpers ------------------------------------------------

    def _spawn_new_expert(
        self, prior_sampler: Callable, dx: int, log_mass: float,
    ) -> Expert:
        pf = OGPParticleFilter(
            ogp_cfg=self.ogp_pf_cfg,
            prior_sampler=prior_sampler,
            batched_grad_func=self.batched_grad_func,
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
        self, e: Expert, X_batch: torch.Tensor, Y_batch: torch.Tensor,
        max_len: int,
    ) -> None:
        if e.X_hist.numel() == 0:
            e.X_hist = X_batch.clone()
            e.y_hist = Y_batch.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, X_batch], dim=0)
            e.y_hist = torch.cat([e.y_hist, Y_batch], dim=0)
        mh = self.ogp_pf_cfg.max_hist
        if mh > 0 and e.X_hist.shape[0] > mh:
            e.X_hist = e.X_hist[-mh:]
            e.y_hist = e.y_hist[-mh:]

    def _expert_theta_mean(self, e: Expert) -> torch.Tensor:
        ps = e.pf.particles
        w = ps.weights().view(-1, 1)
        return (w * ps.theta).sum(dim=0)

    def _hazard(self, rl: int) -> float:
        if hasattr(self.config, "hazard_rate"):
            return float(getattr(self.config, "hazard_rate"))
        r = torch.tensor([rl], dtype=self.dtype, device=self.device)
        val = self.config.hazard(r)[0].item()
        return float(max(min(val, 1.0 - 1e-12), 1e-12))

    # ---- UMP (with cache) ------------------------------------------------

    def ump_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> Tuple[List[float], Dict[int, torch.Tensor]]:
        out: List[float] = []
        cache: Dict[int, torch.Tensor] = {}
        for e in self.experts:
            loglik_n = e.pf.compute_loglik_batch(
                X_batch, Y_batch, e.X_hist, e.y_hist,
                emulator, model_cfg.rho, model_cfg.sigma_eps,
            )
            logmix = torch.logsumexp(e.pf.logw + loglik_n, dim=0)
            out.append(float(logmix))
            cache[id(e)] = loglik_n
        return out, cache

    # ---- prune -----------------------------------------------------------

    def _prune_keep_anchor(
        self, anchor_run_length: int, max_experts: int,
    ) -> None:
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

    # ---- update ----------------------------------------------------------

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

        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(prior_sampler, dx, log_mass=0.0)
            self.experts.append(e0)
            self.restart_start_time = 0
            self.t = 0
            self._last_restart_t = -(10 ** 9)

        # 1) UMP + cache
        log_umps, loglik_cache = self.ump_batch(
            X_batch, Y_batch, emulator, model_cfg,
        )
        log_umps_t = torch.tensor(
            log_umps, device=self.device, dtype=self.dtype,
        )

        experts_pre = list(self.experts)
        log_umps_pre = list(log_umps)
        idx_pre = {id(e): i for i, e in enumerate(experts_pre)}

        def get_pre_log_ump(e):
            j = idx_pre.get(id(e), None)
            return None if j is None else float(log_umps_pre[j])

        # 2) mass update
        prev_log_mass = torch.tensor(
            [e.log_mass for e in self.experts],
            device=self.device, dtype=self.dtype,
        )
        hazards = torch.tensor(
            [self._hazard(e.run_length) for e in self.experts],
            device=self.device, dtype=self.dtype,
        )
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth = prev_log_mass + log_1mh + log_umps_t
        cp = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        for i, e in enumerate(self.experts):
            e.run_length += batch_size
            e.log_mass = float(growth[i])

        new_e = self._spawn_new_expert(
            prior_sampler, dx, log_mass=float(cp),
        )
        self.experts.append(new_e)

        masses = torch.tensor(
            [e.log_mass for e in self.experts],
            device=self.device, dtype=self.dtype,
        )
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # 3) restart decision
        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)
        anchor_e = self._closest_by_run_length(anchor_rl)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e else 0.0

        best_other_mass = 0.0
        s_star: Optional[int] = None
        cand_e: Optional[Expert] = None
        for e in self.experts:
            s = t_now - e.run_length
            if s > r_old:
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = int(s)
                    cand_e = e

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e else 0.0

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
                rate, theta_pass = self._credible_nonoverlap(
                    anchor_e, cand_e, z, frac,
                )
                theta_stat = rate
            elif method == "sw":
                n_proj = int(getattr(self.config, "restart_sw_proj", 32))
                theta_stat = self._sliced_wasserstein(
                    anchor_e, cand_e, n_proj,
                )
                tau = float(
                    getattr(self.config, "restart_theta_tau", 0.1),
                )
                theta_pass = theta_stat > tau
            else:
                theta_stat = self._energy_distance(anchor_e, cand_e)
                tau = float(
                    getattr(self.config, "restart_theta_tau", 0.1),
                )
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
                f"\n[DEBUG BEFORE RESTART] t={t_now}, r_old={r_old}, "
                f"s_star={s_star}"
            )
            print(
                f"  p_anchor={p_anchor:.6f}, "
                f"best_other={best_other_mass:.6f}, p_cp={p_cp:.6f}"
            )
            sorted_e = sorted(
                self.experts, key=lambda e: e.log_mass, reverse=True,
            )
            for ii, ee in enumerate(sorted_e[:5]):
                tm = self._expert_theta_mean(ee).detach().cpu().numpy()
                print(
                    f"    Expert[{ii}] rl={ee.run_length}, "
                    f"mass={math.exp(ee.log_mass):.6f}, θ_mean={tm}"
                )
            print("----------------------------------------------------\n")

            self._last_restart_t = t_now

            if getattr(self.config, "use_backdated_restart", False):
                self.restart_start_time = int(s_star)
                new_anchor_rl = max(
                    t_now - self.restart_start_time, 0,
                )
                keep_e = next(
                    (
                        e for e in self.experts
                        if e.run_length == new_anchor_rl
                    ),
                    None,
                )
                if keep_e is None and self.experts:
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
                r0 = self._spawn_new_expert(
                    prior_sampler, dx, log_mass=0.0,
                )
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
            self._prune_keep_anchor(
                anchor_run_length, self.config.max_experts,
            )

        # 4) PF step — reuse cached loglik where possible
        pf_diags: List[Dict[str, Any]] = []
        for e in self.experts:
            cached = loglik_cache.get(id(e), None)
            diag = e.pf.step_batch(
                X_batch, Y_batch,
                e.X_hist, e.y_hist,
                emulator, model_cfg.rho, model_cfg.sigma_eps,
                cached_loglik=cached,
            )
            pf_diags.append(diag)

        # 5) append history
        for e in self.experts:
            self._append_hist_batch(
                e, X_batch, Y_batch, self.config.max_run_length,
            )

        # 6) advance clock
        self.t += batch_size

        # 7) diagnostics
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
            lu = (
                float(log_umps[idx_e])
                if idx_e < len(log_umps)
                else None
            )
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
                f"[R-BOCPD-OGP-GPU][batch] Restart t={t_now}: "
                f"mode={msg_mode}, r_old={r_old}, "
                f"r_new={self.restart_start_time}, "
                f"s_star={s_star}, p_anchor={p_anchor:.4g}, "
                f"p_cp={p_cp:.4g}"
            )

        # LLR diagnostics
        log_ump_anchor = None
        log_ump_cand = None
        if anchor_e is not None:
            log_ump_anchor = get_pre_log_ump(anchor_e)
        if cand_e is not None:
            log_ump_cand = get_pre_log_ump(cand_e)

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
                int(anchor_e.run_length) if anchor_e else None
            ),
            "cand_rl": (
                int(cand_e.run_length) if cand_e else None
            ),
            "delta_ll_pair": delta_ll_pair,
            "log_odds_mass": log_odds_mass,
            "h_log": h_log,
            "log_ump_anchor": log_ump_anchor,
            "log_ump_cand": log_ump_cand,
        }

    # ---- theta tests (same as original) ----------------------------------

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
        self, e1: Expert, e2: Expert,
        z: float = 2.0, frac: float = 0.5,
    ):
        th1, w1 = self._theta_particles(e1)
        th2, w2 = self._theta_particles(e2)
        mu1, var1 = self._weighted_mean_var(th1, w1)
        mu2, var2 = self._weighted_mean_var(th2, w2)
        sd1, sd2 = torch.sqrt(var1 + 1e-12), torch.sqrt(var2 + 1e-12)
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

    def _sliced_wasserstein(
        self, e1: Expert, e2: Expert, n_proj: int = 32,
    ):
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

    def _closest_by_run_length(
        self, target_rl: int,
    ) -> Optional[Expert]:
        if not self.experts:
            return None
        return min(
            self.experts, key=lambda e: abs(e.run_length - target_rl),
        )
