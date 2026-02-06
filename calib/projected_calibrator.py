# =============================================================
# file: projected_calibrator.py
# Bayesian Projected Calibration with BOCPD + Delta GP
# - YHatExpert: gpytorch Exact GP for y(x)
# - ThetaSolver: sample y-hat trajectories -> project to θ, then fit delta-GP on residuals
# - BOCPDProjectedCalibrator: online BOCPD over run-length hypotheses with top-k pruning
# =============================================================
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Literal
import math
import numpy as np
import torch
import gpytorch

Tensor = torch.Tensor


# =========================
# 1) GP building blocks
# =========================
class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x: Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _train_exact_gp(
    X: Tensor, y: Tensor, iters: int = 200, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Tuple[_ExactGPModel, gpytorch.likelihoods.GaussianLikelihood, float]:
    X = X.to(device, dtype)
    y = y.reshape(-1).to(device, dtype)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device, dtype)
    model = _ExactGPModel(X, y, lik).to(device, dtype)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        opt.step()
    model.eval(); lik.eval()
    with torch.no_grad():
        nll = float((-mll(model(X), y)).item())
    return model, lik, nll


# =========================
# 2) YHatExpert: a single GP y-hat with window/full
# =========================
class YHatExpert:
    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        update_mode: Literal["full", "window"] = "full",
        window_length: int = 1000,
        fit_iters: int = 200,
        start_idx: int = 0,
    ):
        self.device, self.dtype = device, dtype
        self.update_mode = update_mode
        self.window_length = int(window_length)
        self.fit_iters = int(fit_iters)
        self.start_idx = int(start_idx)
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        self.model: Optional[_ExactGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self.nll_: float = float("inf")

    def append(self, X_batch: Tensor, y_batch: Tensor):
        X_batch = self._as2d(X_batch)
        y_batch = y_batch.reshape(-1, 1).to(self.device, self.dtype)
        if self.X is None:
            self.X, self.y = X_batch, y_batch
        else:
            self.X = torch.cat([self.X, X_batch], dim=0)
            self.y = torch.cat([self.y, y_batch], dim=0)
        if self.update_mode == "window":
            n = self.X.shape[0]
            if n > self.window_length:
                self.X = self.X[-self.window_length:]
                self.y = self.y[-self.window_length:]

    def refit(self):
        assert self.X is not None and self.y is not None
        self.model, self.likelihood, self.nll_ = _train_exact_gp(
            self.X, self.y, iters=self.fit_iters, device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def posterior(self, X: Tensor) -> gpytorch.distributions.MultivariateNormal:
        assert self.model is not None and self.likelihood is not None
        self.model.eval(); self.likelihood.eval()
        return self.model(X.to(self.device, self.dtype))

    @torch.no_grad()
    def sample_trajectories(self, X: Tensor, n_samples: int) -> Tensor:
        # returns [S, n]
        post = self.posterior(X)
        return post.rsample(torch.Size([n_samples]))

    def score(self) -> float:
        return -self.nll_

    def _as2d(self, X: Tensor) -> Tensor:
        X = X.detach().to(self.device, self.dtype)
        return X if X.ndim == 2 else X[:, None]


# =========================
# 3) ThetaSolver: project θ + fit delta GP on residuals
# =========================
class ThetaSolver:
    def __init__(
        self,
        simulator: Callable[[Tensor, Tensor], Tensor],  # y_s(X, theta) -> [B,1]
        theta_bounds: Tuple[Tensor, Tensor],            # (lo[p], hi[p])
        n_theta_samples: int = 64,                      # total θ samples to draw
        n_restart: int = 5,                             # LBFGS multi-start
        proj_on: Literal["history", "grid"] = "history",
        x_range: Tuple[float, float] = (0.0, 1.0),
        n_grid: int = 200,
        delta_update_mode: Literal["full", "window"] = "full",
        delta_window_length: int = 1000,
        delta_fit_iters: int = 200,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.simulator = simulator
        self.lo = theta_bounds[0].flatten().to(device, dtype)
        self.hi = theta_bounds[1].flatten().to(device, dtype)
        self.S = int(n_theta_samples)
        self.n_restart = int(n_restart)
        self.proj_on = proj_on
        self.x_range = x_range
        self.n_grid = int(n_grid)
        self.device, self.dtype = device, dtype

        # delta GP (for residuals)
        self.delta_mode = delta_update_mode
        self.delta_window_length = int(delta_window_length)
        self.delta_fit_iters = int(delta_fit_iters)
        self.delta_model: Optional[_ExactGPModel] = None
        self.delta_lik: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self.delta_X: Optional[Tensor] = None
        self.delta_y: Optional[Tensor] = None

        # cache of θ samples
        self.theta_samples_: Optional[Tensor] = None   # [S, p]
        self.theta_weights_: Optional[Tensor] = None   # [S,1]

    def project_and_fit_delta(
        self,
        experts: List[YHatExpert],
        weights: Tensor,                  # [K,] softmax weights of experts
        X_hist: Tensor, y_hist: Tensor,   # for delta fitting
    ):
        """Sample yhat trajectories on projection set; solve θ; fit delta-GP on residuals."""
        Xp = self._projection_set(X_hist)
        # total samples S across experts
        S_each = max(1, self.S // max(1, len(experts)))
        theta_list = []
        w_list = []

        for k, exp in enumerate(experts):
            if exp.X is None or exp.model is None:
                continue
            samples = exp.sample_trajectories(Xp, S_each)  # [S_each, n]
            for s in range(samples.shape[0]):
                theta_s = self._project_one(samples[s, :].reshape(-1), Xp)  # [p]
                theta_list.append(theta_s[None, :])
                w_list.append(float(weights[k].item()))

        if len(theta_list) == 0:
            # fallback: center of bounds
            center = (self.lo + self.hi) / 2.0
            theta = center[None, :].repeat(self.S, 1)
            w = torch.ones(self.S, 1, device=self.device, dtype=self.dtype) / self.S
        else:
            theta = torch.cat(theta_list, dim=0)  # [S_tot, p]
            w = torch.tensor(w_list, device=self.device, dtype=self.dtype).reshape(-1, 1)
            w = w / w.sum()

        self.theta_samples_ = theta
        self.theta_weights_ = w

        # ---- fit delta GP on residuals using θ_mean (weighted)
        theta_mean = (theta * w).sum(dim=0, keepdim=True)  # [1,p]
        y_s_hist = self.simulator(X_hist, theta_mean).reshape(-1, 1)
        resid = (y_hist.reshape(-1, 1).to(self.device, self.dtype) - y_s_hist).reshape(-1, 1)

        self._update_delta_data(X_hist, resid)
        self.delta_model, self.delta_lik, _ = _train_exact_gp(
            self.delta_X, self.delta_y, iters=self.delta_fit_iters, device=self.device, dtype=self.dtype
        )

    @torch.no_grad()
    def predict(self, X_new: Tensor) -> Dict[str, Tensor]:
        """μ = E_θ[y_s(x,θ)] + μ_δ(x), Var ≈ Var_θ[y_s(x,θ)] + Var_δ(x)."""
        X_new = self._as2d(X_new)
        # θ-part
        if self.theta_samples_ is None:
            theta_mean = (self.lo + self.hi)[None, :] / 2.0
            ys = self.simulator(X_new, theta_mean).reshape(-1, 1)
            mu_theta = ys
            var_theta = torch.full_like(mu_theta, 0.5)
        else:
            ts = self.theta_samples_
            w = self.theta_weights_ if self.theta_weights_ is not None else torch.ones(ts.shape[0], 1, device=self.device, dtype=self.dtype)/ts.shape[0]
            preds = []
            for s in range(ts.shape[0]):
                preds.append(self.simulator(X_new, ts[s:s+1, :]))
            YS = torch.cat(preds, dim=1)  # [B, S]
            w = (w / w.sum()).reshape(1, -1)
            mu_theta = (YS * w).sum(dim=1, keepdim=True)
            var_theta = ((YS - mu_theta) ** 2 * w).sum(dim=1, keepdim=True).clamp_min(1e-10)

        # δ-part
        if self.delta_model is None or self.delta_lik is None:
            mu_delta = torch.zeros_like(mu_theta)
            var_delta = torch.full_like(mu_theta, 0.5)
        else:
            self.delta_model.eval(); self.delta_lik.eval()
            with gpytorch.settings.fast_pred_var(), torch.no_grad():
                post = self.delta_model(X_new)
                mu_delta = post.mean.reshape(-1, 1)
                var_delta = post.variance.reshape(-1, 1).clamp_min(1e-10)

        mu = mu_theta + mu_delta
        var = var_theta + var_delta
        return {"mu": mu, "var": var}

    # ---------- helpers ----------
    def _project_one(self, f_vals: Tensor, Xp: Tensor) -> Tensor:
        """θ* = argmin_θ || f_vals - y_s(Xp, θ) ||^2"""
        best = None
        best_val = float("inf")
        for _ in range(self.n_restart):
            theta0 = (self.lo + (self.hi - self.lo) * torch.rand_like(self.lo)).clone().detach().requires_grad_(True)
            opt = torch.optim.LBFGS([theta0], max_iter=60, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(set_to_none=True)
                ys = self.simulator(Xp, theta0[None, :]).reshape(-1)
                loss = torch.mean((f_vals - ys) ** 2)
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                theta0[:] = torch.clamp(theta0, self.lo, self.hi)
                final = torch.mean((f_vals - self.simulator(Xp, theta0[None, :]).reshape(-1)) ** 2).item()
                if final < best_val:
                    best_val = final
                    best = theta0.detach().clone()
        return best  # [p]

    def _projection_set(self, X_hist: Optional[Tensor]) -> Tensor:
        if self.proj_on == "history" and X_hist is not None:
            return X_hist.to(self.device, self.dtype)
        lo, hi = self.x_range
        xs = torch.linspace(lo, hi, self.n_grid, device=self.device, dtype=self.dtype).reshape(-1, 1)
        return xs

    def _update_delta_data(self, X: Tensor, r: Tensor):
        X = X.to(self.device, self.dtype)
        r = r.to(self.device, self.dtype).reshape(-1, 1)
        if self.delta_mode == "full":
            self.delta_X, self.delta_y = X, r
        else:
            if self.delta_X is None:
                self.delta_X, self.delta_y = X, r
            else:
                self.delta_X = torch.cat([self.delta_X, X], dim=0)
                self.delta_y = torch.cat([self.delta_y, r], dim=0)
                n = self.delta_X.shape[0]
                if n > self.delta_window_length:
                    self.delta_X = self.delta_X[-self.delta_window_length:]
                    self.delta_y = self.delta_y[-self.delta_window_length:]

    def _as2d(self, X: Tensor) -> Tensor:
        X = X.detach().to(self.device, self.dtype)
        return X if X.ndim == 2 else X[:, None]


# =========================
# 4) BOCPDProjectedCalibrator
# =========================
class BOCPDProjectedCalibrator:
    """
    Online BOCPD over YHatExpert run-length hypotheses (right-anchored),
    then project θ via ThetaSolver, then fit delta-GP on residuals.
    """
    def __init__(
        self,
        simulator: Callable[[Tensor, Tensor], Tensor],
        theta_bounds: Tuple[Tensor, Tensor],
        # y-hat expert options
        yhat_update_mode: Literal["full", "window"] = "window",
        yhat_window_length: int = 800,
        yhat_fit_iters: int = 200,
        # BOCPD options
        topk: int = 3,
        hazard_lambda: float = 800.0,  # geometric hazard: h=1/lambda
        # Theta solver & delta GP
        theta_solver_kwargs: Optional[Dict] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.simulator = simulator
        self.theta_bounds = theta_bounds
        self.device, self.dtype = device, dtype
        self.topk = int(topk)
        self.hazard_lambda = float(hazard_lambda)
        self.h = 1.0 / max(1.0, self.hazard_lambda)

        self.experts: List[YHatExpert] = []
        self.t_seen: int = 0

        self.yhat_update_mode = yhat_update_mode
        self.yhat_window_length = int(yhat_window_length)
        self.yhat_fit_iters = int(yhat_fit_iters)

        self.theta_solver = ThetaSolver(
            simulator=simulator,
            theta_bounds=theta_bounds,
            device=device,
            dtype=dtype,
            **(theta_solver_kwargs or {})
        )

        # buffers of global history (for projection set & delta GP)
        self.X_hist: Optional[Tensor] = None
        self.y_hist: Optional[Tensor] = None

        # latest weights over experts
        self.weights_: Optional[Tensor] = None  # [K,]

    # @torch.no_grad()
    def update(self, X_batch: Tensor, y_batch: Tensor):
        X_batch = self._as2d(X_batch)
        y_batch = y_batch.reshape(-1, 1).to(self.device, self.dtype)

        # grow existing experts (right-anchored segments)
        for e in self.experts:
            e.append(X_batch, y_batch)
            e.refit()

        # new restart expert (run-length=0)
        new_e = YHatExpert(
            device=self.device, dtype=self.dtype,
            update_mode=self.yhat_update_mode, window_length=self.yhat_window_length,
            fit_iters=self.yhat_fit_iters, start_idx=self.t_seen
        )
        new_e.append(X_batch, y_batch)
        new_e.refit()
        self.experts.append(new_e)

        # score & prune (top-k) with hazard prior
        # log prior ~ r * log(1-h)  for grown experts; for restart expert add log h
        scores = []
        K = len(self.experts)
        for k, e in enumerate(self.experts):
            r = e.X.shape[0] if (e.X is not None) else 0
            log_prior = r * math.log(max(1e-12, 1.0 - self.h))
            if e.start_idx == self.t_seen:  # the new restart expert
                log_prior = math.log(max(1e-12, self.h))
            scores.append(e.score() + log_prior)
        scores = torch.tensor(scores, device=self.device, dtype=self.dtype)

        # keep top-k
        idx = torch.topk(scores, k=min(self.topk, K)).indices.tolist()
        self.experts = [self.experts[i] for i in idx]
        scores = scores[idx]
        # softmax weights
        w = torch.softmax(scores - scores.max(), dim=0)  # [K']
        self.weights_ = w

        # update global history
        if self.X_hist is None:
            self.X_hist, self.y_hist = X_batch, y_batch
        else:
            self.X_hist = torch.cat([self.X_hist, X_batch], dim=0)
            self.y_hist = torch.cat([self.y_hist, y_batch], dim=0)

        # (optional) you can also cap global history if desired; here we keep all

        # project θ and fit delta GP using current experts and weights
        self.theta_solver.project_and_fit_delta(self.experts, w, self.X_hist, self.y_hist)

        self.t_seen += X_batch.shape[0]

    @torch.no_grad()
    def predict(self, X_new: Tensor) -> Dict[str, Tensor]:
        return self.theta_solver.predict(X_new)

    # ---- helpers
    def _as2d(self, X: Tensor) -> Tensor:
        X = X.detach().to(self.device, self.dtype)
        return X if X.ndim == 2 else X[:, None]

if __name__ == "__main__":
    def simulator(X, theta):
        # X: [B,1], theta: [1,p] or [B,p], assume p=1
        return X * theta[:, :1]   # output [B,1]
    def make_fake_expert(device="cpu", dtype=torch.float32):
        X = torch.rand(30, 1, device=device, dtype=dtype)
        y = torch.sin(2 * torch.pi * X) + 0.1 * torch.randn_like(X)

        exp = YHatExpert(
            device=device, dtype=dtype,
            update_mode="full",
            window_length=200,
            fit_iters=50
        )
        exp.append(X, y)
        exp.refit()
        return exp
    experts = [make_fake_expert() for _ in range(2)]
    theta_lo = torch.tensor([0.0])
    theta_hi = torch.tensor([3.0])
    solver = ThetaSolver(
        simulator=simulator,
        theta_bounds=(theta_lo, theta_hi),
        n_theta_samples=16,
        n_restart=3,
        delta_update_mode="full",
        delta_fit_iters=50
    )
    theta_true = torch.tensor([[1.5]])
    X_hist = torch.rand(50, 1)
    y_hist = simulator(X_hist, theta_true) + 0.05 * torch.randn_like(X_hist)
    solver.project_and_fit_delta(
        exp=experts[0],
        X_hist=X_hist,
        y_hist=y_hist
    )
    X_new = torch.linspace(0, 1, 100).reshape(-1,1)
    pred = solver.predict(X_new)

    print("mu shape:", pred["mu"].shape)
    print("var shape:", pred["var"].shape)


