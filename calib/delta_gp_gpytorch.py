import torch
import gpytorch
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Union


# ============================================================
#  Exact GP model
# ============================================================

class DeltaExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel,
    ):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# ============================================================
#  Delta GP State (supports 2 noise modes)
# ============================================================

@dataclass
class GPyTorchDeltaState:
    """
    GPyTorch exact GP for discrepancy δ(x).

    noise_mode:
      - "noise_vec"       : noise_i = λ ρ^2 var_theta_i + σ_base^2
      - "learned_scalar"  : noise = σ^2 (standard GaussianLikelihood)
    """
    X: torch.Tensor                    # [n, d]
    y: torch.Tensor                    # [n]
    kernel: gpytorch.kernels.Kernel

    noise_mode: Literal["noise_vec", "learned_scalar"] = "noise_vec"

    # --- used only in noise_vec mode ---
    var_theta: Optional[torch.Tensor] = None   # [n]
    rho: Optional[float] = None
    noise_min: float = 1e-6
    noise_max: float = 1e1

    # --- internal ---
    model: Optional[DeltaExactGP] = None
    likelihood: Optional[gpytorch.likelihoods.Likelihood] = None
    log_lambda: Optional[torch.nn.Parameter] = None

    def __post_init__(self):
        if self.noise_mode == "noise_vec":
            assert self.var_theta is not None
            assert self.rho is not None
            self.log_lambda = torch.nn.Parameter(
                torch.tensor(0.0, device=self.X.device, dtype=self.X.dtype)
            )
        self._build_model()

    # ------------------------------------------------------------
    #  Build GP
    # ------------------------------------------------------------
    def _build_model(self):
        if self.noise_mode == "noise_vec":
            noise_vec = self._current_noise_vec(detach=True)
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise_vec,
                learn_additional_noise=True,  # ← σ_base²
            )
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = DeltaExactGP(
            self.X, self.y, self.likelihood, self.kernel
        ).to(self.X.device, self.X.dtype)

    # ------------------------------------------------------------
    #  Noise construction
    # ------------------------------------------------------------
    def _current_noise_vec(self, detach: bool) -> torch.Tensor:
        lam = torch.exp(self.log_lambda)
        noise = lam * (self.rho ** 2) * self.var_theta
        noise = noise.clamp(self.noise_min, self.noise_max)
        if detach:
            noise = noise.detach()
        return noise.to(self.X.device, self.X.dtype)

    # ------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------
    def fit(self, max_iter: int = 100, lr: float = 0.1):
        self.model.train()
        self.likelihood.train()

        params = list(self.model.parameters())
        if self.noise_mode == "noise_vec":
            params.append(self.log_lambda)

        opt = torch.optim.Adam(params, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        for _ in range(max_iter):
            opt.zero_grad()

            if self.noise_mode == "noise_vec":
                self.likelihood.noise = self._current_noise_vec(detach=False)

            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            opt.step()

        return self

    # ------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x))
            return pred.mean, pred.variance.clamp_min(1e-12)

'''
state = GPyTorchDeltaState(
    X=X,
    y=y,
    kernel=kernel,
    noise_mode="noise_vec",
    var_theta=eta_var_mix,   # [n]
    rho=rho,
    noise_min=1e-6,
    noise_max=1.0,
)

state.fit(max_iter=80, lr=0.05)

print("learned lambda =", torch.exp(state.log_lambda).item())
print("learned sigma_base^2 =", state.likelihood.noise.item())

state = GPyTorchDeltaState(
    X=X,
    y=y,
    kernel=kernel,
    noise_mode="learned_scalar",
)

state.fit()
'''