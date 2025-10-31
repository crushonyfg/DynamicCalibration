# =============================================================
# file: calib/expert_delta.py
# =============================================================
from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple
import math
import torch

from .particles import ParticleSet
from .emulator import Emulator, GPEmulator
from .delta_gp import OnlineGPState
from .kernels import RBFKernel


def compute_mixture_eta_stats(
    emulator: Emulator,
    X_hist: torch.Tensor,               # [M, dx]
    particles: ParticleSet,             # has theta [N,dθ] and logw [N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-x mixture mean and variance of eta under the current particles.
    Returns:
        mu_mix:  [M]
        var_mix: [M]
    """
    if X_hist.dim() == 1:
        X_hist = X_hist[None, :]
    mu_eta, var_eta = emulator.predict(X_hist, particles.theta)  # [M, N]
    w = particles.weights()                                      # [N]
    w = w / (w.sum() + 1e-16)

    mu_mix = (mu_eta * w[None, :]).sum(dim=1)  # [M]
    # Law of total variance: E[var] + var[E]
    var_mix = (var_eta * w[None, :]).sum(dim=1) + ((mu_eta - mu_mix[:, None]) ** 2 * w[None, :]).sum(dim=1)
    return mu_mix, var_mix


def build_delta_targets_for_expert(
    X_hist: torch.Tensor,               # [M, dx]
    y_hist: torch.Tensor,               # [M]
    emulator: Emulator,
    particles: ParticleSet,
    rho: float,
    sigma_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (X_delta, y_delta, noise_vec) for FixedNoise GP fitting of delta.
    """
    mu_mix, var_mix = compute_mixture_eta_stats(emulator, X_hist, particles)  # [M], [M]
    y_delta = y_hist - rho * mu_mix                                           # [M]
    noise_vec = (rho ** 2) * var_mix + (sigma_eps ** 2)                       # [M]
    noise_vec = noise_vec.clamp_min(1e-12)
    return X_hist, y_delta, noise_vec


class _FixedNoiseDeltaExactGP:
    """
    A small wrapper around gpytorch ExactGP with FixedNoiseGaussianLikelihood.
    Construct per call and fit hyperparameters; return fitted lengthscale(s) and variance.
    """
    def __init__(self,
                 X: torch.Tensor, y: torch.Tensor, noise_vec: torch.Tensor,
                 init_lengthscale: torch.Tensor,  # scalar or [dx]
                 init_variance: float,
                 lr: float = 0.05,
                 steps: int = 200):
        try:
            import gpytorch  # noqa
        except Exception as e:
            raise ImportError("gpytorch is required for ExpertDeltaFitter. Install via `pip install gpytorch`.") from e

        self.X = X
        self.y = y
        self.noise_vec = noise_vec
        self.init_lengthscale = init_lengthscale
        self.init_variance = float(init_variance)
        self.lr = lr
        self.steps = steps

    def fit(self) -> Tuple[torch.Tensor, float]:
        import gpytorch
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.means import ZeroMean
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
        from gpytorch.models import ExactGP

        X, y, noise_vec = self.X, self.y, self.noise_vec
        device, dtype = X.device, X.dtype

        class DeltaExactGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood, ard_num_dims: int):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ZeroMean()
                self.base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
                self.covar_module = ScaleKernel(self.base_kernel)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = FixedNoiseGaussianLikelihood(noise=noise_vec).to(device, dtype)
        model = DeltaExactGP(X, y, likelihood, ard_num_dims=X.shape[1]).to(device, dtype)

        with torch.no_grad():
            ls = self.init_lengthscale.to(device, dtype).view(-1)  # [dx] or [1]
            if ls.numel() == 1:
                model.base_kernel.lengthscale.fill_(float(ls.item()))
            else:
                model.base_kernel.lengthscale.copy_(ls.view(1, -1))
            model.covar_module.outputscale.fill_(self.init_variance)

        model.train(); likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.steps):
            opt.zero_grad()
            out = model(X)
            loss = -mll(out, y)
            loss.backward()
            opt.step()

        model.eval(); likelihood.eval()
        ls_fit = model.base_kernel.lengthscale.detach().view(-1)  # [dx]
        var_fit = float(model.covar_module.outputscale.detach().item())
        return ls_fit, var_fit


@dataclass
class Expert:
    """
    Minimal BOCPD expert interface expected by the fitter.

    Required fields:
      - X_hist:      [M, dx]    expert's historical x's (this run-length segment)
      - y_hist:      [M]        expert's historical y's
      - particles:   ParticleSet
      - delta_state: OnlineGPState (used at runtime; we only update its kernel params here)

    Optional fields you may add on your side:
      - rho: float (if per-expert), otherwise pass rho to fitter explicitly.
    """
    X_hist: torch.Tensor
    y_hist: torch.Tensor
    particles: ParticleSet
    delta_state: OnlineGPState
    name: str = "expert"


class ExpertDeltaFitter:
    """
    Fit expert-shared delta GP hyperparameters using Fixed-Noise ExactGP.
    The known per-point noise is built by integrating out eta's particle mixture uncertainty.
    """
    def __init__(self,
                 train_steps: int = 200,
                 lr: float = 0.05):
        self.train_steps = train_steps
        self.lr = lr

    def refit_expert(
        self,
        expert: Expert,
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
        init_lengthscale: Optional[torch.Tensor] = None,
        init_variance: Optional[float] = None,
        update_delta_state: bool = True,
    ) -> Tuple[torch.Tensor, float]:
        """
        Fit expert-level delta kernel hyperparameters, write back to expert.delta_state if requested.
        Returns:
            (lengthscale_fitted [dx], variance_fitted)
        """
        Xd, yd, noise_vec = build_delta_targets_for_expert(
            expert.X_hist, expert.y_hist, emulator, expert.pf.particles, rho, sigma_eps
        )
        # Init from current delta_state if not provided
        k = expert.delta_state.kernel
        ls0 = getattr(k, "lengthscale", torch.tensor([1.0], dtype=Xd.dtype, device=Xd.device))
        var0 = getattr(k, "variance", 1.0)
        if init_lengthscale is None:
            init_lengthscale = torch.as_tensor(ls0, dtype=Xd.dtype, device=Xd.device).view(-1)
        if init_variance is None:
            init_variance = float(var0)

        fitter = _FixedNoiseDeltaExactGP(
            X=Xd, y=yd, noise_vec=noise_vec,
            init_lengthscale=init_lengthscale,
            init_variance=init_variance,
            lr=self.lr, steps=self.train_steps
        )
        ls_fit, var_fit = fitter.fit()

        if update_delta_state:
            # Write back to expert's OnlineGPState kernel (keep its scalar noise as-is).
            expert.delta_state.kernel.lengthscale = ls_fit.detach()
            if hasattr(expert.delta_state.kernel, "variance"):
                expert.delta_state.kernel.variance = float(var_fit)
            # Recompute cache with updated hyperparams (X,y of delta_state are its own residual history)
            expert.delta_state._recompute_cache_full()

        return ls_fit, var_fit

    def refit_topk(
        self,
        experts: Sequence[Expert],
        emulator: Emulator,
        rho: float,
        sigma_eps: float,
        topk_indices: Sequence[int],
        init_from_current: bool = True
    ) -> List[Tuple[int, torch.Tensor, float]]:
        """
        Refit a subset (e.g., BOCPD top-k experts) and return list of (idx, ls, var).
        """
        results = []
        for idx in topk_indices:
            exp = experts[idx]
            if init_from_current:
                init_ls = getattr(exp.delta_state.kernel, "lengthscale", torch.tensor([1.0], dtype=exp.X_hist.dtype, device=exp.X_hist.device))
                init_var = getattr(exp.delta_state.kernel, "variance", 1.0)
            else:
                init_ls = torch.tensor([1.0], dtype=exp.X_hist.dtype, device=exp.X_hist.device)
                init_var = 1.0
            ls_fit, var_fit = self.refit_expert(
                exp, emulator, rho, sigma_eps,
                init_lengthscale=init_ls, init_variance=init_var, update_delta_state=True
            )
            results.append((idx, ls_fit, var_fit))
        return results


# ------------------------- Minimal BOCPD-like integration demo -------------------------
if __name__ == "__main__":
    """
    Demo:
      - Build a toy emulator (GPEmulator) for eta(x,theta)
      - Create a mock Expert with its own particles and delta_state
      - Fit expert-shared delta hyperparameters with FixedNoise ExactGP
    """
    import math
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    # Toy eta: sin(2πx) + 0.5*theta
    def eta_true(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[None, :]
        return torch.sin(2.0 * math.pi * x[:, 0:1]) + 0.5 * theta  # [b,1]

    # Build a GPEmulator over z=[x,theta] using training data with theta=0
    from .emulator import GPEmulator
    n_train = 20
    X_train = torch.linspace(0.0, 1.0, n_train, dtype=dtype, device=device).unsqueeze(1)
    th_train = torch.zeros(n_train, 1, dtype=dtype, device=device)
    y_train = eta_true(X_train, th_train).squeeze(-1) + 0.03 * torch.randn(n_train, dtype=dtype, device=device)
    emulator = GPEmulator(
        X_train, th_train, y_train,
        kernel={"name": "rbf", "lengthscale": 0.5, "variance": 1.0},
        noise=1e-2,
        mode="exact_full",
        hyperparam_mode="fixed",
    )

    # Build a mock Expert
    M = 25
    X_hist = torch.linspace(0.0, 1.0, M, dtype=dtype, device=device).unsqueeze(1)
    # Simulate theta* generating data for y_hist (unknown in practice)
    theta_star = torch.tensor([[0.2]], dtype=dtype, device=device)
    y_hist = eta_true(X_hist, theta_star).squeeze(-1) + 0.05 * torch.randn(M, dtype=dtype, device=device)

    # Particles for this expert
    from .particles import ParticleSet
    Np = 64
    theta_particles = torch.randn(Np, 1, dtype=dtype, device=device) * 0.5  # prior around 0
    logw = torch.full((Np,), -math.log(Np), dtype=dtype, device=device)
    particles = ParticleSet(theta=theta_particles, logw=logw)

    # Expert's delta_state (starts empty; we only update kernel hyperparams here)
    from .delta_gp import OnlineGPState
    delta_kernel = RBFKernel(lengthscale=torch.tensor([0.8], dtype=dtype), variance=0.2)
    delta_state = OnlineGPState(
        X=torch.empty(0, 1, dtype=dtype, device=device),
        y=torch.empty(0, dtype=dtype, device=device),
        kernel=delta_kernel,
        noise=1e-3,
        update_mode="exact_full",
        hyperparam_mode="fixed",
    )

    expert = Expert(
        X_hist=X_hist,
        y_hist=y_hist,
        particles=particles,
        delta_state=delta_state,
        name="expert_0"
    )

    # Fit expert-shared delta hyperparams
    fitter = ExpertDeltaFitter(train_steps=150, lr=0.05)
    rho = 1.0
    sigma_eps = 0.05
    ls_fit, var_fit = fitter.refit_expert(expert, emulator, rho, sigma_eps)

    print("=== Expert-level delta hyperparams fitted ===")
    print("lengthscale (fitted):", ls_fit.detach().cpu().numpy())
    print("variance   (fitted):", var_fit)
    # Check that expert.delta_state kernel has been updated
    print("delta_state.kernel.lengthscale:", expert.delta_state.kernel.lengthscale)
    print("delta_state.kernel.variance   :", getattr(expert.delta_state.kernel, 'variance', None))
