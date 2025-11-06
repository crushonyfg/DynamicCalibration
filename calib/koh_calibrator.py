# =============================================================
# file: calib/koh_calibrator.py
# KOH (Kennedy–O'Hagan) Bayesian calibration with explicit GP discrepancy δ(x)
# - Model: y(x) = y_s(x, θ) + δ(x) + ε,  δ ~ GP(0, Kδ),  ε ~ N(0, σ_n^2)
# - Online update with "full" or "window" memory
# - Predicts mean & variance of reality: μ* = y_s(x*, θ̂) + k_*^T (K + σ_n^2 I)^(-1) r
#   where r = y - y_s(x, θ̂),  K = Kδ(X, X)
# References:
#   Kennedy & O'Hagan (2001, 2002): Bayesian calibration with GP discrepancy.
#   Spitieris et al. (2023) JMLR Appendix A.2: predictive equations with discrepancy.
# =============================================================

from __future__ import annotations
import torch
import numpy as np
from typing import Callable, Dict, Literal, Optional, Tuple

Tensor = torch.Tensor


def rbf_kernel(
    X: Tensor,
    Y: Optional[Tensor] = None,
    lengthscale: float | Tensor = 0.3,
    variance: float | Tensor = 1.0,
) -> Tensor:
    """
    RBF kernel for discrepancy GP (scalar y).
    - Accepts lengthscale / variance as torch.Tensor or Python float.
    - If Tensor is provided, preserves gradient for autograd.
    """
    X = X.reshape(X.shape[0], -1)
    Y = X if Y is None else Y.reshape(Y.shape[0], -1)

    # ensure ls, var are tensors on the same device/dtype; preserve requires_grad if they are tensors
    if isinstance(lengthscale, torch.Tensor):
        ls = lengthscale
        if ls.device != X.device:
            ls = ls.to(X.device)
        if ls.dtype != X.dtype:
            ls = ls.to(X.dtype)
    else:
        ls = torch.tensor(lengthscale, device=X.device, dtype=X.dtype)

    if isinstance(variance, torch.Tensor):
        var = variance
        if var.device != X.device:
            var = var.to(X.device)
        if var.dtype != X.dtype:
            var = var.to(X.dtype)
    else:
        var = torch.tensor(variance, device=X.device, dtype=X.dtype)

    # ||x - y||^2
    XX = (X**2).sum(1, keepdim=True)
    YY = (Y**2).sum(1, keepdim=True)
    d2 = XX - 2.0 * (X @ Y.T) + YY.T
    K = var * torch.exp(-0.5 * d2 / (ls**2))
    return K


class KOHCalibrator:
    """
    Kennedy–O'Hagan Bayesian calibration with explicit GP discrepancy δ(x).
    - Supports 'full' history or fixed-length sliding 'window'.
    - θ and kernel hyperparameters can be optimized by marginal likelihood (optional).
    """

    def __init__(
        self,
        simulator: Callable[[Tensor, Tensor], Tensor],  # y_s(X, theta) -> shape [B,1]
        theta_init: Tensor,                             # shape [p] or [1,p]
        theta_bounds: Optional[Tuple[Tensor, Tensor]] = None,  # (lo[p], hi[p])
        update_mode: Literal["full", "window"] = "full",
        window_length: int = 1000,
        # Discrepancy GP hyperparams
        lengthscale: float = 0.3,
        variance: float = 1.0,
        noise_var: float = 1e-6,
        # Optimization toggles
        optimize_theta: bool = True,
        optimize_hypers: bool = False,                  # set True to also fit (lengthscale, variance, noise_var)
        max_opt_steps: int = 200,
        # Torch context
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.simulator = simulator
        self.theta = theta_init.detach().flatten().to(device, dtype)  # current estimate
        self.theta_bounds = theta_bounds
        self.update_mode = update_mode
        self.window_length = int(window_length)
        self.l = float(lengthscale)
        self.sf2 = float(variance)
        self.sn2 = float(noise_var)
        self.optimize_theta = optimize_theta
        self.optimize_hypers = optimize_hypers
        self.max_opt_steps = max_opt_steps
        self.device = device
        self.dtype = dtype

        self.X_hist: Optional[Tensor] = None
        self.Y_hist: Optional[Tensor] = None

        # caches
        self._K: Optional[Tensor] = None            # Kδ(X,X)
        self._Kreg_chol: Optional[Tensor] = None    # chol(K + sn2 I)
        self._alpha: Optional[Tensor] = None        # (K + sn2 I)^(-1) r for current θ

    # ---------- utilities ----------
    def _ensure_2d(self, X: Tensor) -> Tensor:
        return X if X.ndim == 2 else X.reshape(X.shape[0], -1)

    def _y_sim(self, X: Tensor, theta: Tensor) -> Tensor:
        if theta.ndim == 1:
            theta = theta[None, :]
        return self.simulator(X, theta)  # shape [B,1]

    def _build_kernel(self, X: Tensor) -> Tensor:
        return rbf_kernel(X, None, self.l, self.sf2)

    def _form_r(self, X: Tensor, Y: Tensor, theta: Tensor) -> Tensor:
        return Y - self._y_sim(X, theta)

    def _apply_window(self):
        if self.update_mode == "window" and self.X_hist is not None:
            n = self.X_hist.shape[0]
            if n > self.window_length:
                self.X_hist = self.X_hist[-self.window_length:]
                self.Y_hist = self.Y_hist[-self.window_length:]

    def _recompute_cache(self):
        """Compute K, chol(K+sn2 I), alpha = solve(K+sn2 I, r(θ))."""
        X, Y = self.X_hist, self.Y_hist
        if X is None:
            self._K = self._Kreg_chol = self._alpha = None
            return
        K = self._build_kernel(X)
        Kreg = K + self.sn2 * torch.eye(K.shape[0], device=self.device, dtype=self.dtype)
        L = torch.linalg.cholesky(Kreg)
        r = self._form_r(X, Y, self.theta)
        alpha = torch.cholesky_solve(r, L)  # (K+sn2I)^(-1) r
        self._K, self._Kreg_chol, self._alpha = K, L, alpha

    # ---------- public API ----------
    @torch.no_grad()
    def update(self, X_batch: Tensor, Y_batch: Tensor):
        """Append data then (optionally) re-fit θ and/or GP hyperparameters."""
        X_batch = self._ensure_2d(X_batch.detach().to(self.device, self.dtype))
        Y_batch = Y_batch.detach().to(self.device, self.dtype).reshape(-1, 1)

        if self.X_hist is None:
            self.X_hist, self.Y_hist = X_batch, Y_batch
        else:
            self.X_hist = torch.cat([self.X_hist, X_batch], dim=0)
            self.Y_hist = torch.cat([self.Y_hist, Y_batch], dim=0)

        self._apply_window()

        # (Optional) re-optimize θ / hypers by marginal likelihood
        if self.optimize_theta or self.optimize_hypers:
            self._fit_marginal_likelihood()

        # refresh cache with latest θ/hypers
        self._recompute_cache()

    @torch.no_grad()
    def predict(self, X_new: Tensor) -> Dict[str, Tensor]:
        """
        Predictive distribution of reality y at X_new:
        μ* = y_s(X*, θ̂) + K(X*,X) (K+sn2I)^(-1) r
        Σ* = K(X*,X*) - K(X*,X) (K+sn2I)^(-1) K(X,X*)
        (return only marginal variances as a vector)
        """
        X_new = self._ensure_2d(X_new.detach().to(self.device, self.dtype))

        # if nothing yet, return simulator + large variance
        if self.X_hist is None or self._Kreg_chol is None:
            mu = self._y_sim(X_new, self.theta)
            var = torch.full_like(mu, 1.0)
            return {"mu": mu, "var": var}

        KxX = rbf_kernel(X_new, self.X_hist, self.l, self.sf2)
        # posterior mean of δ:
        delta_mean = KxX @ self._alpha
        mu = self._y_sim(X_new, self.theta) + delta_mean

        # marginal variance of δ:
        # v = solve(Kreg, K(X,X*)) using chol
        v = torch.cholesky_solve(KxX.T, self._Kreg_chol)  # shape [N_hist, N*]
        # diag( K** - KxX @ v )
        Kxx_diag = torch.full((X_new.shape[0], 1), self.sf2, device=self.device, dtype=self.dtype)  # since RBF k(x*,x*)=sf2
        var = Kxx_diag - (KxX * v.T).sum(dim=1, keepdim=True)
        var = torch.clamp(var, min=1e-10)
        return {"mu": mu, "var": var}

    # ---------- objective & optimizer ----------
    def _nll(self, theta_and_hypers: Tensor) -> Tensor:
        """
        Negative log marginal likelihood:
        0.5 * r^T (K+sn2 I)^{-1} r + 0.5 log|K+sn2 I| + const (const ignored).
        """
        p = self.theta.numel()
        theta = theta_and_hypers[:p]
        if self.optimize_hypers:
            log_l, log_sf2, log_sn2 = theta_and_hypers[p:]
            l = torch.exp(log_l)
            sf2 = torch.exp(log_sf2)
            sn2 = torch.exp(log_sn2)
        else:
            l = torch.tensor(self.l, device=self.device, dtype=self.dtype)
            sf2 = torch.tensor(self.sf2, device=self.device, dtype=self.dtype)
            sn2 = torch.tensor(self.sn2, device=self.device, dtype=self.dtype)

        X, Y = self.X_hist, self.Y_hist
        if X is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # build Kreg and r (重要：不要把超参转成 float，保持 autograd)
        K = rbf_kernel(X, None, l, sf2)
        Kreg = K + sn2 * torch.eye(K.shape[0], device=self.device, dtype=self.dtype)
        try:
            L = torch.linalg.cholesky(Kreg)
        except RuntimeError:
            # jitter
            jitter = 1e-6
            L = torch.linalg.cholesky(Kreg + jitter * torch.eye(Kreg.shape[0], device=self.device, dtype=self.dtype))
        r = self._form_r(X, Y, theta)
        # 0.5 r^T Kreg^{-1} r
        alpha = torch.cholesky_solve(r, L)
        data_fit = 0.5 * (r * alpha).sum()
        # 0.5 log |Kreg| = sum(log(diag(L)))
        logdet = torch.sum(torch.log(torch.diag(L)))
        nll = data_fit + logdet
        return nll

    def _fit_marginal_likelihood(self):
        """Optimize θ (and optionally hypers) by minimizing NLL."""
        if self.X_hist is None:
            return

        p = self.theta.numel()
        params = [self.theta.clone().requires_grad_(True)]
        if self.optimize_hypers:
            log_l = torch.tensor(np.log(max(self.l, 1e-6)), device=self.device, dtype=self.dtype, requires_grad=True)
            log_sf2 = torch.tensor(np.log(max(self.sf2, 1e-8)), device=self.device, dtype=self.dtype, requires_grad=True)
            log_sn2 = torch.tensor(np.log(max(self.sn2, 1e-12)), device=self.device, dtype=self.dtype, requires_grad=True)
            params += [log_l, log_sf2, log_sn2]

        opt = torch.optim.LBFGS(params, max_iter=self.max_opt_steps, line_search_fn="strong_wolfe")

        # bounds for θ if provided (implemented by clamping after step)
        lo, hi = None, None
        if self.theta_bounds is not None:
            lo = self.theta_bounds[0].to(self.device, self.dtype).flatten()
            hi = self.theta_bounds[1].to(self.device, self.dtype).flatten()

        def closure():
            opt.zero_grad(set_to_none=True)
            # 修复：把所有参与 cat 的张量变成 1D，避免 0 维拼接
            packed = torch.cat(
                [params[0].view(-1)] +
                ([] if not self.optimize_hypers else [p.view(-1) for p in params[1:]])
            )
            loss = self._nll(packed)
            loss.backward()
            return loss

        prev_loss = None
        for _ in range(10):  # a few outer restarts for robustness
            loss = opt.step(closure)
            if lo is not None:
                with torch.no_grad():
                    params[0][:] = torch.max(torch.min(params[0], hi), lo)
            if prev_loss is not None and abs(prev_loss.item() - loss.item()) < 1e-7:
                break
            prev_loss = loss

        # unpack back
        with torch.no_grad():
            self.theta = params[0].detach()
            if self.optimize_hypers:
                self.l = float(torch.exp(params[1]).item())
                self.sf2 = float(torch.exp(params[2]).item())
                self.sn2 = float(torch.exp(params[3]).item())

    # ---------- metrics ----------
    @staticmethod
    def rmse(y_true: Tensor, y_pred: Tensor) -> float:
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

    @staticmethod
    def crps_gaussian(y_true: Tensor, mu: Tensor, var: Tensor) -> float:
        """Closed-form CRPS for Gaussian predictive (per-point avg)."""
        sigma = torch.sqrt(var.clamp_min(1e-12))
        z = (y_true - mu) / sigma
        c1 = z * torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype)))
        c2 = torch.sqrt(torch.tensor(2.0 / np.pi, device=z.device, dtype=z.dtype)) * torch.exp(-0.5 * z**2)
        c3 = torch.sqrt(torch.tensor(1.0 / np.pi, device=z.device, dtype=z.dtype))
        crps = sigma * (c1 + c2 - c3)
        return float(torch.mean(crps))
