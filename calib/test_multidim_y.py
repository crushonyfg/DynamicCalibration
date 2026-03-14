# =============================================================
# Toy script to verify multi-dimensional y support across the pipeline.
# Run from repo root: python -m calib.test_multidim_y
# =============================================================
import math
import torch
from .emulator import DeterministicSimulator
from .particles import ParticleSet
from .likelihood import predictive_stats, loglik_and_grads
from .utils import normal_logpdf
from .delta_gp import OnlineGPState
from .kernels import RBFKernel


def test_utils_normal_logpdf():
    """normal_logpdf: 1D vs 2D y."""
    torch.manual_seed(42)
    dtype = torch.float64
    # 1D: y [2], mu/var [2, 3]
    y1 = torch.tensor([0.0, 1.0], dtype=dtype)
    mu1 = torch.randn(2, 3, dtype=dtype)
    var1 = torch.softmax(torch.randn(2, 3), dim=-1) + 0.1
    ll1 = normal_logpdf(y1, mu1, var1)
    assert ll1.shape == (2, 3), ll1.shape
    # 2D: y [2, 2], mu/var [2, 3, 2]
    y2 = torch.tensor([[0.0, 0.5], [1.0, -0.5]], dtype=dtype)
    mu2 = torch.randn(2, 3, 2, dtype=dtype)
    var2 = torch.softmax(torch.randn(2, 3, 2), dim=-1) + 0.1
    ll2 = normal_logpdf(y2, mu2, var2)
    assert ll2.shape == (2, 3), ll2.shape
    print("  [OK] normal_logpdf 1D and 2D y shapes")


def test_predictive_stats():
    """predictive_stats: 2D and 3D mu_eta."""
    torch.manual_seed(42)
    dtype = torch.float64
    rho, sigma_eps = 1.0, 0.01
    # 2D
    mu_eta = torch.randn(4, 5, dtype=dtype)
    var_eta = torch.softmax(torch.randn(4, 5), dim=-1) + 0.1
    mu_d = torch.randn(4, dtype=dtype)
    var_d = torch.softmax(torch.randn(4), dim=-1) + 0.1
    mu_t, var_t = predictive_stats(rho, mu_eta, var_eta, mu_d, var_d, sigma_eps)
    assert mu_t.shape == (4, 5) and var_t.shape == (4, 5), (mu_t.shape, var_t.shape)
    # 3D
    mu_eta3 = torch.randn(4, 5, 2, dtype=dtype)
    var_eta3 = torch.softmax(torch.randn(4, 5, 2), dim=-1) + 0.1
    mu_t3, var_t3 = predictive_stats(rho, mu_eta3, var_eta3, mu_d, var_d, sigma_eps)
    assert mu_t3.shape == (4, 5, 2) and var_t3.shape == (4, 5, 2), (mu_t3.shape, var_t3.shape)
    print("  [OK] predictive_stats 2D and 3D mu_eta")


def test_emulator_multidim():
    """DeterministicSimulator returns [b, N] or [b, N, dy]."""
    torch.manual_seed(42)
    dtype = torch.float64
    # Scalar output
    def f1(x, theta):
        return (x @ theta.T).squeeze(-1)  # [b]
    sim1 = DeterministicSimulator(f1)
    x = torch.randn(3, 2, dtype=dtype)
    theta = torch.randn(4, 2, dtype=dtype)
    mu1, var1 = sim1.predict(x, theta)
    assert mu1.shape == (3, 4) and var1.shape == (3, 4), (mu1.shape, var1.shape)
    # 2D output
    def f2(x, theta):
        # return [b, 2]
        a = (x @ theta.T)  # [b, 1]
        return torch.cat([a, a + 0.5], dim=-1)
    sim2 = DeterministicSimulator(f2)
    mu2, var2 = sim2.predict(x, theta)
    assert mu2.shape == (3, 4, 2) and var2.shape == (3, 4, 2), (mu2.shape, var2.shape)
    print("  [OK] DeterministicSimulator scalar and 2D output")


def test_loglik_and_grads_multidim_y():
    """loglik_and_grads with 2D y and DeterministicSimulator [b,N,dy]."""
    torch.manual_seed(42)
    dtype = torch.float64
    device = "cpu"
    # Model: y = [x*theta, x*theta + 0.5] (2D output)
    def eta(x, theta):
        if x.dim() == 1:
            x = x[None, :]
        p = (x @ theta.T)  # [b, 1]
        return torch.cat([p, p + 0.5], dim=-1)  # [b, 2]
    emulator = DeterministicSimulator(eta, enable_autograd=False)
    N, dth = 5, 2
    theta_particles = torch.randn(N, dth, dtype=dtype, device=device) * 0.1
    parts = ParticleSet(
        theta=theta_particles.clone(),
        logw=torch.zeros(N, dtype=dtype, device=device),
    )
    delta_kernel = RBFKernel(lengthscale=1.0, variance=1e-6)
    delta_state = OnlineGPState(
        X=torch.empty(0, 2, dtype=dtype, device=device),
        y=torch.empty(0, dtype=dtype, device=device),
        kernel=delta_kernel,
        noise=1e-6,
        update_mode="exact_full",
        hyperparam_mode="fixed",
    )
    x = torch.tensor([[0.5, 0.3], [0.2, 0.8]], dtype=dtype, device=device)  # [2, dx]
    y = torch.tensor([[0.1, 0.6], [-0.05, 0.4]], dtype=dtype, device=device)  # [2, 2]
    rho, sigma_eps = 1.0, 0.02
    info = loglik_and_grads(
        y=y,
        x=x,
        particles=parts,
        emulator=emulator,
        delta_state=delta_state,
        rho=rho,
        sigma_eps=sigma_eps,
        need_grads=False,
        use_discrepancy=True,
    )
    loglik = info["loglik"]
    assert loglik.shape == (N,), loglik.shape
    assert torch.isfinite(loglik).all(), loglik
    print("  [OK] loglik_and_grads with 2D y; loglik shape", loglik.shape)
    return info


def test_pf_step_multidim_y():
    """One PF step with 2D y (no grads)."""
    from .pf import ParticleFilter
    from .configs import PFConfig
    torch.manual_seed(42)
    dtype = torch.float64
    device = "cpu"
    def eta(x, theta):
        if x.dim() == 1:
            x = x[None, :]
        p = (x @ theta.T)
        return torch.cat([p, p + 0.5], dim=-1)
    emulator = DeterministicSimulator(eta, enable_autograd=False)
    delta_kernel = RBFKernel(lengthscale=1.0, variance=1e-6)
    delta_state = OnlineGPState(
        X=torch.empty(0, 2, dtype=dtype, device=device),
        y=torch.empty(0, dtype=dtype, device=device),
        kernel=delta_kernel,
        noise=1e-6,
        update_mode="exact_full",
        hyperparam_mode="fixed",
    )
    pf_cfg = PFConfig(
        num_particles=8,
        resample_ess_ratio=0.5,
        move_strategy="random_walk",
        random_walk_scale=0.1,
    )
    def prior(n):
        return torch.randn(n, 2, dtype=dtype, device=device) * 0.2
    pf = ParticleFilter.from_prior(prior, pf_cfg, device=device, dtype=dtype)
    x_t = torch.tensor([0.5, 0.3], dtype=dtype, device=device)
    y_t = torch.tensor([[0.1, 0.6]], dtype=dtype, device=device)  # [1, 2]: one step, 2D obs
    out = pf.step(x_t, y_t, emulator, delta_state, rho=1.0, sigma_eps=0.02, grad_info=False, use_discrepancy=True)
    assert "log_evidence" in out and "ess" in out
    assert torch.isfinite(pf.particles.logw).all()
    print("  [OK] PF step with 2D y")


if __name__ == "__main__":
    print("=== test_multidim_y: multi-dimensional y support ===\n")
    test_utils_normal_logpdf()
    test_predictive_stats()
    test_emulator_multidim()
    test_loglik_and_grads_multidim_y()
    test_pf_step_multidim_y()
    print("\n=== All multi-dim y checks passed. ===")
