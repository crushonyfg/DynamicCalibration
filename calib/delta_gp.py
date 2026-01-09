# calib/delta_gp.py
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Literal, Optional, Sequence, Union
import math
import torch

from .kernels import Kernel, make_kernel
from .utils import normal_logpdf

# ------------------------- Online exact GP (with rank-1 append) -------------------------
@dataclass
class OnlineGPState:
    """
    Online exact GP state for discrepancy δ(x).

    update_mode:
      - 'exact_full': rebuild full K and Cholesky each append (O(t^3))
      - 'exact_rank1': rank-1 Cholesky append using previous factor (O(t^2) per new point)
      - 'sparse_inducing': delegates to SVGPState (gpytorch) via an internal adapter

    hyperparam_mode:
      - 'fixed': kernel hyperparameters are held constant
      - 'fit': refit hyperparameters via ML-II when calling refit_hyperparams()
               or when append(..., maybe_refit=True)
    """
    X: torch.Tensor  # [t, dx]
    y: torch.Tensor  # [t]
    kernel: Kernel
    # noise: float
    noise: Union[float, torch.Tensor]
    update_mode: Literal['exact_full', 'exact_rank1', 'sparse_inducing'] = 'exact_rank1'
    hyperparam_mode: Literal['fixed', 'fit'] = 'fixed'
    cache: Dict[str, Any] = field(default_factory=dict)
    Z: Optional[torch.Tensor] = None  # for sparse_inducing adapter (inducing inputs)

    # --- internal helpers ---
    def _kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        K = self.kernel.cov(X, X)
        t = X.shape[0]
        if isinstance(self.noise, torch.Tensor):
            jitter = 1e-8
            K = K + torch.diag(self.noise.to(X.device, X.dtype)+jitter)
        else:
            K = K + (self.noise + 1e-8) * torch.eye(t, dtype=X.dtype, device=X.device)
        return K

    def _recompute_cache_full(self) -> None:
        if self.X.numel() == 0:
            self.cache.clear()
            return
        K = self._kernel_matrix(self.X)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.y[:, None], L).squeeze(-1)  # [t]
        self.cache["L"] = L
        self.cache["alpha"] = alpha

    def _append_rank1_single(self, x_new: torch.Tensor, y_new: torch.Tensor) -> None:
        """
        Rank-1 Cholesky append for a single new point.
        """
        t = self.X.shape[0]
        if t == 0:
            self.X = x_new[None, :]
            self.y = y_new[None]
            self._recompute_cache_full()
            return

        X_old = self.X
        self.X = torch.cat([self.X, x_new[None, :]], dim=0)
        self.y = torch.cat([self.y, y_new[None]], dim=0)

        if "L" not in self.cache:
            self._recompute_cache_full()
            return
        L = self.cache["L"]  # [t,t]

        k = self.kernel.cov(x_new[None, :], X_old).squeeze(0)  # [t]
        if isinstance(self.noise, torch.Tensor):
            kxx = self.kernel.cov(x_new[None, :], x_new[None, :]).squeeze() + torch.diag(self.noise.to(x_new.device, x_new.dtype)+1e-8)
        else:
            kxx = self.kernel.cov(x_new[None, :], x_new[None, :]).squeeze() + (self.noise + 1e-8)

        v = torch.linalg.solve_triangular(L, k[:, None], upper=False).squeeze(-1)
        residual_var = kxx - (v @ v)
        residual_var = torch.clamp(residual_var, min=1e-14, max=self.kernel.variance)
        diag_new = torch.sqrt(residual_var)

        L_new = torch.zeros(t + 1, t + 1, dtype=L.dtype, device=L.device)
        L_new[:t, :t] = L
        L_new[:t, t] = v
        L_new[t, t] = diag_new

        y_aug = self.y
        alpha_new = torch.cholesky_solve(y_aug[:, None], L_new).squeeze(-1)

        self.cache["L"] = L_new
        self.cache["alpha"] = alpha_new

    def _recompute_cache(self) -> None:
        if self.update_mode in ('exact_full', 'exact_rank1'):
            # For recompute, both modes just rebuild fully
            self._recompute_cache_full()
        elif self.update_mode == 'sparse_inducing':
            raise RuntimeError("Use SVGPState adapter for sparse_inducing mode.")
        else:
            raise ValueError(f"Unknown update_mode: {self.update_mode}")

    # --- public API ---
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.update_mode == 'sparse_inducing':
            raise RuntimeError("This OnlineGPState is not in SVGP mode. Use SVGPState for predictions.")
        if self.X.numel() == 0:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device), self.kernel.diag(x)
        if "L" not in self.cache:
            self._recompute_cache()
        L: torch.Tensor = self.cache["L"]
        alpha: torch.Tensor = self.cache["alpha"]
        kx = self.kernel.cov(x, self.X)  # [b, t]
        mu = kx @ alpha  # [b]
        v = torch.cholesky_solve(kx.transpose(0, 1), L)  # [t,b]
        var = self.kernel.diag(x) - (kx * v.transpose(0, 1)).sum(-1)
        var = var.clamp_min(1e-12)
        return mu, var

    def log_predictive(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (mu, var, logpdf) for δ at x for observation y.
        """
        x_ = x if x.dim() == 2 else x[None, :]
        mu, var = self.predict(x_)
        lp = normal_logpdf(y if y.dim() > 0 else y[None], mu[:, None], var[:, None]).squeeze(-1)
        return mu, var, lp

    def append(self, x_new: torch.Tensor, y_new: torch.Tensor, maybe_refit: bool = False) -> "OnlineGPState":
        if self.update_mode == 'sparse_inducing':
            raise RuntimeError("Use SVGPState adapter for sparse_inducing mode.")
        if x_new.dim() == 1:
            if self.update_mode == 'exact_rank1':
                self._append_rank1_single(x_new, y_new)
            elif self.update_mode == 'exact_full':
                if self.X.numel() == 0:
                    self.X = x_new[None, :]
                    self.y = y_new[None]
                else:
                    self.X = torch.cat([self.X, x_new[None, :]], dim=0)
                    self.y = torch.cat([self.y, y_new[None]], dim=0)
                self._recompute_cache_full()
        else:
            # Batch append: loop rank-1 for each new point for simplicity
            assert x_new.dim() == 2 and y_new.dim() == 1 and x_new.shape[0] == y_new.shape[0]
            if self.update_mode == 'exact_full':
                self.X = torch.cat([self.X, x_new], dim=0) if self.X.numel() > 0 else x_new.clone()
                self.y = torch.cat([self.y, y_new], dim=0) if self.y.numel() > 0 else y_new.clone()
                self._recompute_cache_full()
            elif self.update_mode == 'exact_rank1':
                for i in range(x_new.shape[0]):
                    self._append_rank1_single(x_new[i], y_new[i])
            else:
                raise ValueError(f"Unknown update_mode: {self.update_mode}")

        if self.hyperparam_mode == 'fit' and maybe_refit:
            self.refit_hyperparams()
        return self

    def append_batch(self, X_new: torch.Tensor, y_new: torch.Tensor, maybe_refit: bool = False) -> None:
        """批量添加新数据点到GP状态"""
        if X_new.shape[0] == 0:
            return
        
        if self.X.numel() == 0:
            self.X = X_new.clone()
            self.y = y_new.clone()
            self._recompute_cache_full()
            return
        
        # 批量添加到现有数据
        self.X = torch.cat([self.X, X_new], dim=0)
        self.y = torch.cat([self.y, y_new], dim=0)
        
        if self.update_mode == "exact_full":
            self._recompute_cache_full()
        elif self.update_mode == "exact_rank1":
            # 对于批量数据，使用full recompute更稳定
            self._recompute_cache_full()
        
        if maybe_refit and self.hyperparam_mode == "fit":
            self.refit_hyperparams()

        # ---------------- Hyperparameter refit (optional) ----------------
    def refit_hyperparams(self, max_iter: int = 100, lr: float = 0.1, fit_noise: bool = True) -> None:
        """
        Refit kernel hyperparameters by maximizing the GP log marginal likelihood (ML-II).
        Supports vector lengthscales.

        Args:
            max_iter: number of optimization steps
            lr: learning rate
            fit_noise: whether to refit noise as well
        """
        if self.X.numel() == 0:
            return

        device, dtype = self.X.device, self.X.dtype
        # Initialize trainable log-parameters
        lengthscale = getattr(self.kernel, 'lengthscale', torch.tensor([1.0], dtype=dtype, device=device))
        variance = getattr(self.kernel, 'variance', 1.0)
        if not isinstance(lengthscale, torch.Tensor):
            lengthscale = torch.tensor([float(lengthscale)], dtype=dtype, device=device)

        log_ls = torch.nn.Parameter(torch.log(lengthscale.to(device, dtype)))
        log_var = torch.nn.Parameter(torch.log(torch.tensor(float(variance), dtype=dtype, device=device)))
        if fit_noise:
            log_noise = torch.nn.Parameter(torch.log(torch.tensor(self.noise, dtype=dtype, device=device)))
            params = [log_ls, log_var, log_noise]
        else:
            log_noise = None
            params = [log_ls, log_var]

        opt = torch.optim.Adam(params, lr=lr)

        for _ in range(max_iter):
            opt.zero_grad()

            # Use current values without detach (to keep graph)
            ls_val = torch.exp(log_ls)            # [dx]
            var_val = torch.exp(log_var)          # scalar
            noise_val = torch.exp(log_noise) if fit_noise else torch.tensor(self.noise, dtype=dtype, device=device)

            # Temporarily plug values into kernel for K computation
            orig_ls, orig_var = self.kernel.lengthscale, self.kernel.variance
            self.kernel.lengthscale = ls_val
            self.kernel.variance = var_val

            K = self._kernel_matrix(self.X)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(self.y[:, None], L)

            # log marginal likelihood
            mll = -0.5 * (self.y[:, None].T @ alpha).squeeze()
            mll += -torch.log(torch.diagonal(L)).sum()
            mll += -0.5 * self.X.shape[0] * torch.log(torch.tensor(2.0 * math.pi, dtype=dtype, device=device))

            # restore original kernel params (do not detach yet)
            self.kernel.lengthscale, self.kernel.variance = orig_ls, orig_var

            loss = -mll
            loss.backward()
            opt.step()

        # After optimization: write back final detached values
        self.kernel.lengthscale = torch.exp(log_ls).detach()
        self.kernel.variance = float(torch.exp(log_var).detach().item())
        if fit_noise and log_noise is not None:
            self.noise = float(torch.exp(log_noise).detach().item())

        # Recompute cache with updated hyperparameters
        self._recompute_cache_full()



# ------------------------- SVGP (gpytorch) adapter -------------------------
class SVGPState:
    """
    A light wrapper around gpytorch SVGP for discrepancy δ(x).
    Provides predict/append interfaces similar to OnlineGPState.

    Note: Requires gpytorch to be installed.
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        inducing_points: torch.Tensor,
        noise: float = 1e-3,
        lengthscale: Union[float, Sequence[float], torch.Tensor] = 1.0,
        variance: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        try:
            import gpytorch  # noqa
        except Exception as e:
            raise ImportError("gpytorch is required for SVGPState. Please install gpytorch.") from e

        self.device = device or X.device
        self.dtype = dtype or X.dtype
        self.X = X.to(self.device, self.dtype)
        self.y = y.to(self.device, self.dtype)
        self.Z = inducing_points.to(self.device, self.dtype)
        self.noise = float(noise)
        self.lengthscale = lengthscale
        self.variance = float(variance)

        self._build_model()

        if self.X.numel() > 0:
            self.train_steps(self.X, self.y, steps=200, lr=0.05)

    def _build_model(self):
        import gpytorch
        from gpytorch.kernels import ScaleKernel, RBFKernel
        from gpytorch.means import ZeroMean
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
        from gpytorch.models import ApproximateGP

        Z = self.Z

        class _SVGPModel(ApproximateGP):
            def __init__(self, Z, init_ls, init_var):
                variational_distribution = CholeskyVariationalDistribution(Z.shape[0])
                variational_strategy = VariationalStrategy(self, Z, variational_distribution, learn_inducing_locations=True)
                super().__init__(variational_strategy)
                self.mean_module = ZeroMean()
                self.base_kernel = RBFKernel(ard_num_dims=Z.shape[1])
                self.covar_module = ScaleKernel(self.base_kernel)
                # init
                with torch.no_grad():
                    if isinstance(init_ls, torch.Tensor):
                        self.base_kernel.lengthscale.copy_(init_ls.view(1, -1))
                    else:
                        self.base_kernel.lengthscale.fill_(float(init_ls))
                    self.covar_module.outputscale.fill_(float(init_var))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        init_ls = torch.as_tensor(self.lengthscale, dtype=self.dtype, device=self.device)
        self.model = _SVGPModel(Z, init_ls, self.variance).to(self.device, self.dtype)
        self.likelihood = GaussianLikelihood(noise=self.noise).to(self.device, self.dtype)

    def train_steps(self, X, y, steps=200, lr=0.05):
        import gpytorch
        self.model.train()
        self.likelihood.train()
        opt = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y.numel())

        for _ in range(steps):
            opt.zero_grad()
            out = self.model(X)
            loss = -mll(out, y)
            loss.backward()
            opt.step()

    def append(self, X_new: torch.Tensor, y_new: torch.Tensor, steps: int = 50, lr: float = 0.05):
        if X_new.dim() == 1:
            X_new = X_new[None, :]
            y_new = y_new[None]
        self.X = torch.cat([self.X, X_new.to(self.device, self.dtype)], dim=0) if self.X.numel() > 0 else X_new.to(self.device, self.dtype)
        self.y = torch.cat([self.y, y_new.to(self.device, self.dtype)], dim=0) if self.y.numel() > 0 else y_new.to(self.device, self.dtype)
        # small number of steps to adapt
        self.train_steps(self.X, self.y, steps=steps, lr=lr)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import gpytorch
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x.to(self.device, self.dtype)))
            mu = pred.mean
            var = pred.variance
        return mu, var


# ------------------------- Adapter factory -------------------------
def make_gp_state(
    mode: Literal['exact_full', 'exact_rank1', 'sparse_inducing'],
    X: torch.Tensor,
    y: torch.Tensor,
    kernel_or_cfg: Union[Kernel, dict],
    noise: float,
    inducing_points: Optional[torch.Tensor] = None,
    hyperparam_mode: Literal['fixed', 'fit'] = 'fixed',
):
    """
    Helper to construct either OnlineGPState or SVGPState based on mode.

    - For exact_* modes: kernel_or_cfg should be a Kernel instance.
    - For sparse_inducing: kernel_or_cfg can be a dict with keys for 'lengthscale', 'variance'.
    """
    if mode in ('exact_full', 'exact_rank1'):
        if isinstance(kernel_or_cfg, dict):
            from .configs import DeltaKernelConfig
            cfg = DeltaKernelConfig(**kernel_or_cfg)
            k = make_kernel(cfg)  # expects DeltaKernelConfig-like dict
        else:
            k = kernel_or_cfg
        return OnlineGPState(
            X=X, y=y, kernel=k, noise=noise,
            update_mode=mode, hyperparam_mode=hyperparam_mode
        )
    elif mode == 'sparse_inducing':
        if inducing_points is None:
            raise ValueError("inducing_points is required for sparse_inducing mode.")
        if not isinstance(kernel_or_cfg, dict):
            raise ValueError("For sparse_inducing, pass kernel params as dict {'name','lengthscale','variance'}")
        lengthscale = kernel_or_cfg.get('lengthscale', 1.0)
        variance = kernel_or_cfg.get('variance', 1.0)
        return SVGPState(
            X=X, y=y, inducing_points=inducing_points, noise=noise,
            lengthscale=lengthscale, variance=variance,
            device=X.device, dtype=X.dtype
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------------- Minimal test in __main__ -------------------------
if __name__ == "__main__":
    # Minimal self-test comparing exact_full vs exact_rank1, and optional SVGP if gpytorch exists.
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float64

    # Synthetic 2D function for delta: delta(x) = sin(2*pi*x0) + 0.3*x1
    def f_delta(X):
        return torch.sin(2.0 * math.pi * X[:, 0]) + 0.3 * X[:, 1]

    # Training set
    n0, dx = 8, 2
    X0 = torch.rand(n0, dx, dtype=dtype, device=device)
    y0 = f_delta(X0) + 0.01 * torch.randn(n0, dtype=dtype, device=device)

    # A query batch
    Xq = torch.rand(5, dx, dtype=dtype, device=device)

    # Kernel with vector lengthscale
    from .kernels import RBFKernel
    k = RBFKernel(lengthscale=torch.tensor([0.4, 0.7], dtype=dtype), variance=1.2)

    print("=== exact_full ===")
    gp_full = OnlineGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                            kernel=k, noise=1e-3, update_mode="exact_full", hyperparam_mode="fixed")
    gp_full.append_batch(X0, y0)
    mu_f, var_f = gp_full.predict(Xq)
    print("mu_f:", mu_f.detach().cpu().numpy())
    print("var_f:", var_f.detach().cpu().numpy())

    print("\n=== exact_rank1 (single appends) ===")
    gp_rank1 = OnlineGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                             kernel=k, noise=1e-3, update_mode="exact_rank1", hyperparam_mode="fixed")
    gp_rank1.append_batch(X0, y0)
    mu_r, var_r = gp_rank1.predict(Xq)
    print("mu_r:", mu_r.detach().cpu().numpy())
    print("var_r:", var_r.detach().cpu().numpy())

    # Sanity: results should be very close
    print("\nmax |mu_f - mu_r| =", (mu_f - mu_r).abs().max().item())
    print("max |var_f - var_r| =", (var_f - var_r).abs().max().item())

    # Test batch append additional points
    X1 = torch.rand(3, dx, dtype=dtype, device=device)
    y1 = f_delta(X1) + 0.01 * torch.randn(3, dtype=dtype, device=device)
    gp_rank1.append_batch(X1, y1)
    mu_r2, var_r2 = gp_rank1.predict(Xq)
    print("\nAfter batch append, mu_r2:", mu_r2.detach().cpu().numpy())

    # Optional: hyperparameter refit (ML-II)
    gp_rank1.hyperparam_mode = "fit"
    gp_rank1.refit_hyperparams(max_iter=50, lr=0.05, fit_noise=True)
    mu_r3, var_r3 = gp_rank1.predict(Xq)
    print("\nAfter refit, mu_r3:", mu_r3.detach().cpu().numpy())

    # Optional: SVGP (requires gpytorch)
    try:
        import gpytorch  # noqa
        print("\n=== SVGP (gpytorch) ===")
        Z = torch.rand(16, dx, dtype=dtype, device=device)  # inducing points
        svgp = SVGPState(X=torch.empty(0, dx, dtype=dtype), y=torch.empty(0, dtype=dtype),
                         inducing_points=Z, noise=1e-3, lengthscale=torch.tensor([0.4, 0.7], dtype=dtype), variance=1.2)
        svgp.append(X0, y0, steps=100, lr=0.05)
        mu_s, var_s = svgp.predict(Xq)
        print("mu_s:", mu_s.detach().cpu().numpy())
    except Exception as e:
        print("\nSVGP not available:", e)
