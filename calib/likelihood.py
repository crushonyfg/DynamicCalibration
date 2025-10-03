# =============================================================
# file: calib/likelihood.py
# =============================================================
from typing import Dict, Tuple, Optional, Literal
import torch
from .particles import ParticleSet
from .emulator import Emulator
from .delta_gp import OnlineGPState
from .utils import normal_logpdf


def predictive_stats(
    rho: float,
    mu_eta: torch.Tensor,   # [b, N]
    var_eta: torch.Tensor,  # [b, N]
    mu_delta: torch.Tensor, # [b]
    var_delta: torch.Tensor,# [b]
    sigma_eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine emulator eta and discrepancy delta with measurement noise.
    Y ~ N( rho*mu_eta + mu_delta , rho^2*var_eta + var_delta + sigma_eps^2 ).
    Returns:
        mu_tot: [b, N]
        var_tot: [b, N]
    """
    mu = rho * mu_eta + mu_delta[:, None]
    var = (rho**2) * var_eta + var_delta[:, None] + (sigma_eps**2)
    var = var.clamp_min(1e-12)
    return mu, var


def _grad_loglik_theta(
    y: torch.Tensor,            # [b]
    mu_tot: torch.Tensor,       # [b, N]
    var_tot: torch.Tensor,      # [b, N]
    dmu_dth: torch.Tensor,      # [b, N, dθ]  (this is d mu_eta / dθ)
    dvar_dth: Optional[torch.Tensor],  # [b, N, dθ] (this is d var_eta / dθ) or None
    rho: float
) -> torch.Tensor:
    """
    Gradient of log N(y | mu_tot, var_tot) w.r.t theta.
    Using:
      d mu_tot / dθ = rho * d mu_eta / dθ
      d var_tot / dθ = rho^2 * d var_eta / dθ
    Returns:
      grad: [N, dθ]  (sum over batch b)
    """
    if y.dim() == 1:
        y = y[:, None]  # [b,1]
    b, N = mu_tot.shape
    dth = dmu_dth.shape[-1]

    dmu_tot = rho * dmu_dth                                     # [b,N,dθ]
    if dvar_dth is None:
        dvar_tot = torch.zeros_like(dmu_dth)
    else:
        dvar_tot = (rho ** 2) * dvar_dth                         # [b,N,dθ]

    yb   = y.view(b, 1, 1).expand(b, N, dth)
    mu3  = mu_tot.view(b, N, 1).expand_as(dmu_tot)
    var3 = var_tot.view(b, N, 1).expand_as(dmu_tot)

    res = yb - mu3
    inv_var = 1.0 / var3
    # d/dθ log N = res/var * dmu + (-0.5/var + 0.5*res^2/var^2) * dvar
    coeff_var = -0.5 * inv_var + 0.5 * (res * res) * (inv_var * inv_var)
    grad_b = res * inv_var * dmu_tot + coeff_var * dvar_tot      # [b,N,dθ]
    grad = grad_b.sum(dim=0)                                     # [N,dθ]
    return grad


def _hessian_theta(
    mu_tot: torch.Tensor,         # [b,N]
    var_tot: torch.Tensor,        # [b,N]
    dmu_tot: torch.Tensor,        # [b,N,dθ]  (rho * dmu_eta/dθ)
    dvar_tot: torch.Tensor,       # [b,N,dθ]  (rho^2 * dvar_eta/dθ)
    mode: Literal["fisher", "gauss_newton"] = "fisher"
) -> torch.Tensor:
    """
    Return a positive-semi-definite Hessian approximation per particle: [N, dθ, dθ].

    fisher:
      I(θ) = E[-∂² log p / ∂θ∂θ^T]
           = (1/var) (∂mu/∂θ)(∂mu/∂θ)^T  +  (1/(2 var^2)) (∂var/∂θ)(∂var/∂θ)^T
      We implement the empirical version (sum over b).

    gauss_newton:
      Use only the mean term: (1/var) (∂mu/∂θ)(∂mu/∂θ)^T
      (good when variance term is either constant or noisy to estimate).
    """
    b, N, dth = dmu_tot.shape
    inv_var = 1.0 / var_tot.view(b, N, 1)               # [b,N,1]
    term_mu = inv_var * dmu_tot                         # [b,N,dθ]
    # Assemble per-particle outer-products and sum over b
    # H_mu[n] = sum_b (dmu_b[n]^T dmu_b[n] / var_b[n])
    H_mu = torch.zeros(N, dth, dth, dtype=dmu_tot.dtype, device=dmu_tot.device)
    for bb in range(b):
        J = term_mu[bb]                                 # [N,dθ]
        H_mu = H_mu + torch.einsum("ni,nj->nij", J, J)  # accumulate

    if mode == "gauss_newton":
        return H_mu  # [N,dθ,dθ]

    # fisher adds variance term
    inv_var2 = inv_var * inv_var                        # [b,N,1]
    term_var = inv_var2 * 0.5 * dvar_tot                # [b,N,dθ]
    H_var = torch.zeros_like(H_mu)
    for bb in range(b):
        Jv = term_var[bb]                               # [N,dθ]
        H_var = H_var + torch.einsum("ni,nj->nij", Jv, Jv)

    return H_mu + H_var


def loglik_and_grads(
    y: torch.Tensor,
    x: torch.Tensor,
    particles: ParticleSet,
    emulator: Emulator,
    delta_state: OnlineGPState,
    rho: float,
    sigma_eps: float,
    need_grads: bool = False,
    need_hessian: bool = False,
    hessian_mode: Literal["fisher", "gauss_newton"] = "fisher"
) -> Dict[str, torch.Tensor]:
    """
    Compute per-particle log-likelihood and optionally gradients/Hessian w.r.t theta.

    Returns dict with keys:
      - "loglik": [N]
      - "grad":   [N, dθ]           (if need_grads)
      - "hess":   [N, dθ, dθ]       (if need_hessian)
    """
    if x.dim() == 1:
        x = x[None, :]

    # Emulator and discrepancy predictions
    mu_eta, var_eta = emulator.predict(x, particles.theta)  # [b,N]
    mu_delta, var_delta = delta_state.predict(x)            # [b],[b]
    mu_tot, var_tot = predictive_stats(rho, mu_eta, var_eta, mu_delta, var_delta, sigma_eps)

    # Log-likelihood (per batch, per particle) -> assume online with b=1 commonly
    loglik_bn = normal_logpdf(y, mu_tot, var_tot)           # [b,N]
    out: Dict[str, torch.Tensor] = {"loglik": loglik_bn.sum(dim=0)}  # sum over b

    if not (need_grads or need_hessian):
        return out

    # Gradients from emulator
    dmu_dth, dvar_dth = emulator.grad_theta(x, particles.theta)  # [b,N,dθ] and optional [b,N,dθ]
    # Compute gradient w.r.t theta
    grad = _grad_loglik_theta(y, mu_tot, var_tot, dmu_dth, dvar_dth, rho)
    out["grad"] = grad

    if need_hessian:
        # Build dmu_tot and dvar_tot with rho factors
        dmu_tot = rho * dmu_dth
        dvar_tot = torch.zeros_like(dmu_tot) if dvar_dth is None else (rho**2) * dvar_dth
        H = _hessian_theta(mu_tot, var_tot, dmu_tot, dvar_tot, mode=hessian_mode)
        out["hess"] = H

    return out

if __name__ == "__main__":
    # Minimal self-test for likelihood, gradients, and Hessian (finite-diff sanity).
    import math
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    # --- Toy data: 1D x, 1D theta ---
    # True function for eta(x, theta) we want the GP emulator to learn:
    # eta(x, theta) = sin(2*pi*x) + 0.5*theta  (simple, smooth, non-linear in x)
    def eta_true(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[None, :]
        return torch.sin(2.0 * math.pi * x[:, 0:1]) + 0.5 * theta  # [b,1]

    # Training set for GPEmulator
    n_train = 16
    X_train = torch.linspace(0, 1, n_train, dtype=dtype, device=device).unsqueeze(1)   # [n,1]
    th_train = torch.zeros(n_train, 1, dtype=dtype, device=device)                     # fix theta=0 for training
    y_train = eta_true(X_train, th_train).squeeze(-1) + 0.02 * torch.randn(n_train, dtype=dtype, device=device)

    # Build GPEmulator (over z=[x,theta]) with stable settings
    from .emulator import GPEmulator
    gp_emul = GPEmulator(
        X_train, th_train, y_train,
        kernel={"name": "rbf", "lengthscale": 0.5, "variance": 1.0},
        noise=1e-2,
        mode="exact_full",
        hyperparam_mode="fixed",
    )

    # Build a nearly-zero discrepancy GP: delta(x) ~ 0 with tiny variance/noise
    from .kernels import RBFKernel
    from .delta_gp import OnlineGPState
    delta_kernel = RBFKernel(lengthscale=1.0, variance=1e-6)
    delta_state = OnlineGPState(
        X=torch.empty(0, 1, dtype=dtype, device=device),
        y=torch.empty(0, dtype=dtype, device=device),
        kernel=delta_kernel,
        noise=1e-6,
        update_mode="exact_full",
        hyperparam_mode="fixed",
    )

    # Particles (two thetas we will evaluate/compare gradients on)
    from .particles import ParticleSet
    theta_particles = torch.tensor([[-0.2], [0.3]], dtype=dtype, device=device)  # [N=2, dθ=1]
    parts = ParticleSet(theta=theta_particles.clone(), logw=torch.zeros(theta_particles.shape[0], dtype=dtype, device=device))

    # A single online observation (b=1)
    x_t = torch.tensor([0.35], dtype=dtype, device=device)             # [dx=1]
    theta_for_y = torch.tensor([[0.1]], dtype=dtype, device=device)    # just to synthesize y
    y_t = eta_true(x_t, theta_for_y).squeeze() + 0.02 * torch.randn((), dtype=dtype, device=device)

    rho = 1.0
    sigma_eps = 0.02

    # --- Run likelihood and analytic grad/Hessian ---
    info = loglik_and_grads(
        y=y_t.view(1), x=x_t,
        particles=parts,
        emulator=gp_emul,
        delta_state=delta_state,
        rho=rho,
        sigma_eps=sigma_eps,
        need_grads=True,
        need_hessian=True,
        hessian_mode="fisher",
    )
    loglik = info["loglik"]   # [N]
    grad_analytic = info["grad"]  # [N, dθ=1]
    hess = info["hess"]       # [N, dθ, dθ]

    print("== Likelihood & Grad/Hess (analytic) ==")
    print("loglik:", loglik.detach().cpu().numpy())
    print("grad (analytic):", grad_analytic.detach().cpu().numpy())
    print("hess (Fisher) shape:", hess.shape)

    # --- Finite-difference gradient check (central difference) ---
    def fd_grad_single_particle(theta0: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Return central-difference grad wrt a single particle theta0: [dθ].
        """
        dth = theta0.numel()
        g = torch.zeros(dth, dtype=dtype, device=device)
        for j in range(dth):
            e = torch.zeros_like(theta0)
            e[j] = eps
            theta_plus = theta0 + e
            theta_minus = theta0 - e
            parts_p = ParticleSet(theta=theta_plus[None, :], logw=torch.zeros(1, dtype=dtype, device=device))
            parts_m = ParticleSet(theta=theta_minus[None, :], logw=torch.zeros(1, dtype=dtype, device=device))

            info_p = loglik_and_grads(
                y=y_t.view(1), x=x_t,
                particles=parts_p,
                emulator=gp_emul,
                delta_state=delta_state,
                rho=rho,
                sigma_eps=sigma_eps,
                need_grads=False
            )
            info_m = loglik_and_grads(
                y=y_t.view(1), x=x_t,
                particles=parts_m,
                emulator=gp_emul,
                delta_state=delta_state,
                rho=rho,
                sigma_eps=sigma_eps,
                need_grads=False
            )
            g[j] = (info_p["loglik"][0] - info_m["loglik"][0]) / (2.0 * eps)
        return g

    fd_grads = []
    for n in range(theta_particles.shape[0]):
        g_fd = fd_grad_single_particle(theta_particles[n].clone(), eps=1e-5)
        fd_grads.append(g_fd)
    fd_grads = torch.stack(fd_grads, dim=0)  # [N, dθ]

    print("\n== Finite-difference vs Analytic grad ==")
    print("grad (finite-diff):", fd_grads.detach().cpu().numpy())
    print("max |diff| per particle:", (fd_grads - grad_analytic).abs().max(dim=1).values.detach().cpu().numpy())

    # --- Basic Hessian checks ---
    # Symmetry (version-safe): take max-abs over the last two dims per particle
    diff = (hess - hess.transpose(-1, -2)).abs()   # [N, dθ, dθ]
    sym_err = diff.view(diff.shape[0], -1).max(dim=1).values
    print("\nHessian symmetry max-abs per particle:", sym_err.detach().cpu().numpy())

    # PSD check (Fisher should be PSD). Compute minimum eigenvalue.
    min_eigs = []
    for n in range(hess.shape[0]):
        ev = torch.linalg.eigvalsh(hess[n])
        min_eigs.append(ev.min().item())
    print("Hessian min eigenvalue per particle (should be >= -1e-10):", min_eigs)


    # --- Predictive stats sanity (shapes and basic consistency) ---
    mu_eta, var_eta = gp_emul.predict(x_t.view(1, -1), theta_particles)
    mu_delta, var_delta = delta_state.predict(x_t.view(1, -1))
    mu_tot, var_tot = predictive_stats(rho, mu_eta, var_eta, mu_delta, var_delta, sigma_eps)
    assert mu_tot.shape == var_tot.shape == (1, theta_particles.shape[0])
    assert (var_tot > 0).all(), "variance must be positive"
    print("\nPredictive stats OK. Shapes:", mu_tot.shape)
