# ============================================================
# file: koh_batch_calibrator.py
# Incremental Kennedy–O'Hagan Bayesian calibration (batch re-fit)
# ============================================================

import numpy as np
from typing import Optional, Dict, Callable
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
from scipy.special import erf


class KOHBatchCalibrator:
    def __init__(
        self,
        simulator: Callable[[np.ndarray, float], np.ndarray],  # y_s(x, theta)
        theta_init: float = 1.0,
        theta_bounds=(0.05, 3.0),
        lengthscale_init: float = 0.2,
        variance_init: float = 1.0,
        noise_var_init: float = 1e-6,
        random_seed: int = 0,
        window_size: Optional[int] = None,
    ):
        """Batch re-fit KOH that supports streaming updates via a sliding window."""
        np.random.seed(random_seed)
        self._sim_user = simulator          # 原始用户提供的模拟器
        self.theta_init = float(theta_init)
        self.theta_bounds = tuple(theta_bounds)
        self.l_init = float(lengthscale_init)
        self.sf2_init = float(variance_init)
        self.sn2_init = float(noise_var_init)

        # buffers
        self.X_hist: Optional[np.ndarray] = None
        self.Y_hist: Optional[np.ndarray] = None
        self.window_size = window_size

        # fitted params (running estimates)
        self.theta_ = float(theta_init)
        self.l_ = float(lengthscale_init)
        self.sf2_ = float(variance_init)
        self.sn2_ = float(noise_var_init)

    # ------------------ helpers ------------------
    def _sim(self, x: np.ndarray, theta: float) -> np.ndarray:
        """Wrapper to ensure simulator output is 1D (n,)."""
        out = self._sim_user(x, theta)
        return np.asarray(out).reshape(-1)

    @staticmethod
    def rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None, ell: float = 0.2, sf2: float = 1.0) -> np.ndarray:
        X = np.asarray(X).reshape(-1, 1)
        Y = X if Y is None else np.asarray(Y).reshape(-1, 1)
        d2 = (X - Y.T) ** 2
        return sf2 * np.exp(-0.5 * d2 / (ell ** 2))

    # ------------------ objective ------------------
    def _neg_log_marginal_likelihood(self, params: np.ndarray) -> float:
        # params = [theta, log_ell, log_sf2, log_sn2]
        theta, log_ell, log_sf2, log_sn2 = params
        ell, sf2, sn2 = float(np.exp(log_ell)), float(np.exp(log_sf2)), float(np.exp(log_sn2))

        X, Y = self.X_hist, self.Y_hist
        if X is None:
            return 0.0

        y_sim = self._sim(X, theta)
        r = np.asarray(Y).reshape(-1) - y_sim.reshape(-1)     # (n,)
        K = self.rbf_kernel(X, ell=ell, sf2=sf2) + sn2 * np.eye(len(X))

        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            # small jitter for stability
            L = cholesky(K + 1e-8 * np.eye(len(K)))

        v = solve(L, r)
        alpha = solve(L.T, v)
        # use sum(r*alpha) to avoid shape issues with dot
        nll = 0.5 * np.sum(r * alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
        return float(nll)

    # ------------------ API ------------------
    def update(self, X_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        """Append a batch; optionally keep only the last `window_size` samples."""
        Xb = np.asarray(X_batch)
        Yb = np.asarray(Y_batch)
        if self.X_hist is None:
            self.X_hist, self.Y_hist = Xb.copy(), Yb.copy()
        else:
            self.X_hist = np.concatenate([self.X_hist, Xb])
            self.Y_hist = np.concatenate([self.Y_hist, Yb])

        if self.window_size is not None and len(self.X_hist) > self.window_size:
            self.X_hist = self.X_hist[-self.window_size:]
            self.Y_hist = self.Y_hist[-self.window_size:]

        # print(f"Update: X_hist shape: {self.X_hist.shape}, Y_hist shape: {self.Y_hist.shape}")

    def fit(self):
        """Refit by minimizing NLL over the current (windowed) history (batch re-fit)."""
        if self.X_hist is None:
            raise RuntimeError("No data to fit yet. Call update() first.")

        # randomize around last estimates to avoid getting stuck
        x0 = np.array([
            self.theta_ * np.random.uniform(0.9, 1.1),
            np.log(self.l_)  + np.random.uniform(-0.2, 0.2),
            np.log(self.sf2_) + np.random.uniform(-0.1, 0.1),
            np.log(self.sn2_) + np.random.uniform(-0.1, 0.1),
        ], dtype=float)

        bounds = [
            self.theta_bounds,
            (np.log(1e-3), np.log(10.0)),
            (np.log(1e-6), np.log(10.0)),
            (np.log(1e-9), np.log(1e-1)),
        ]

        res = minimize(
            self._neg_log_marginal_likelihood,
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=dict(ftol=1e-10, gtol=1e-10, maxiter=300),
        )

        theta, log_ell, log_sf2, log_sn2 = res.x
        self.theta_ = float(theta)
        self.l_ = float(np.exp(log_ell))
        self.sf2_ = float(np.exp(log_sf2))
        self.sn2_ = float(np.exp(log_sn2))
        return res

    def predict(self, X_new: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict y at new inputs using KOH posterior (mean+var)."""
        if self.X_hist is None:
            raise RuntimeError("Model not fitted yet.")

        X = self.X_hist
        Y = self.Y_hist
        theta, ell, sf2, sn2 = self.theta_, self.l_, self.sf2_, self.sn2_

        y_sim = self._sim(X, theta)
        r = np.asarray(Y).reshape(-1) - y_sim.reshape(-1)     # (n,)

        K = self.rbf_kernel(X, ell=ell, sf2=sf2) + sn2 * np.eye(len(X))
        Ks = self.rbf_kernel(X, X_new, ell=ell, sf2=sf2)
        try:
            L = cholesky(K)
        except np.linalg.LinAlgError:
            L = cholesky(K + 1e-8 * np.eye(len(K)))

        alpha = solve(L.T, solve(L, r))
        mu_delta = Ks.T @ alpha                                 # (m,)

        Kss_diag = np.full(len(np.asarray(X_new).reshape(-1)), sf2)
        v = solve(L, Ks)
        var_delta = Kss_diag - np.sum(v * v, axis=0)
        var_delta = np.maximum(var_delta, 1e-10)

        mu = self._sim(X_new, theta) + mu_delta
        var = var_delta
        return {"mu": mu, "var": var}

    # ------------------ metrics ------------------
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
        """Closed-form CRPS for Gaussian predictive (per-point avg)."""
        y_true = np.asarray(y_true).reshape(-1)
        mu = np.asarray(mu).reshape(-1)
        var = np.asarray(var).reshape(-1)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        z = (y_true - mu) / sigma
        crps = sigma * (z * erf(z / np.sqrt(2.0))
                        + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z ** 2)
                        - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps))
