# =============================================================
# file: calib/resampling.py
# =============================================================
from typing import Callable, Optional
import torch
# from .likelihood import loglik_and_grads


def resample_indices(weights: torch.Tensor, scheme: str = "systematic") -> torch.Tensor:
    N = weights.numel()
    if scheme == "multinomial":
        return torch.multinomial(weights, N, replacement=True)
    u0 = torch.rand(1, device=weights.device) / N
    if scheme == "stratified":
        u = (torch.arange(N, device=weights.device, dtype=weights.dtype) + torch.rand(N, device=weights.device)) / N
    else:  # systematic
        u = u0 + torch.arange(N, device=weights.device, dtype=weights.dtype) / N
    cdf = torch.cumsum(weights, dim=0)
    idx = torch.searchsorted(cdf, u.clamp(max=1-1e-12))
    return idx


def random_walk_move(theta: torch.Tensor, scale: float) -> torch.Tensor:
    return theta + scale * torch.randn_like(theta)


def liu_west_move(theta: torch.Tensor, weights: torch.Tensor, a: float, h2: Optional[float] = None) -> torch.Tensor:
    if h2 is None:
        h2 = 1.0 - a**2
    mean = (weights[:, None] * theta).sum(dim=0, keepdim=True)  # [1,d]
    cov = (weights[:, None] * (theta - mean)).T @ (theta - mean)  # [d,d]
    N, d = theta.shape
    noise = torch.distributions.MultivariateNormal(torch.zeros(d, device=theta.device, dtype=theta.dtype), h2 * cov + 1e-9 * torch.eye(d, device=theta.device, dtype=theta.dtype))
    shrunk = a * theta + (1 - a) * mean
    return shrunk + noise.sample((N,))


def laplace_proposal(theta: torch.Tensor,
                     grad: torch.Tensor,
                     hess_approx: torch.Tensor,
                     alpha: float,
                     beta: float,
                     eta: float) -> torch.Tensor:
    """
    Local Gaussian proposal:
      theta' ~ N(theta + eta * grad,  Sigma), where Sigma = alpha * H^{-1} + beta * I.
    Supports batched Hessians per particle: hess_approx: [N, d, d].
    """
    N, d = theta.shape
    # Build a batched identity [1,d,d] to broadcast to [N,d,d]
    eye = torch.eye(d, dtype=theta.dtype, device=theta.device).unsqueeze(0)  # [1,d,d]

    # Robust inverse for a batch of Hessians
    try:
        Hinv = torch.linalg.pinv(hess_approx)  # [N,d,d]
    except Exception:
        Hinv = eye.expand(N, d, d).clone()

    Sigma = alpha * Hinv + beta * eye  # [N,d,d]
    # Cholesky per-particle (add small jitter)
    L = torch.linalg.cholesky(Sigma + 1e-9 * eye)  # [N,d,d]

    # Mean shift
    # eta = 1.0
    mean = theta + eta * grad  # [N,d]

    # Sample z ~ N(0, I) and transform with per-particle L
    z = torch.randn_like(theta)  # [N,d]
    # Batched multiplication: for each n, z[n] @ L[n]^T
    step = (z.unsqueeze(1) @ L.transpose(-1, -2)).squeeze(1)  # [N,d]

    return mean + step  # [N,d]



def pmcmc_move(theta: torch.Tensor,
               logpost_fn: Callable[[torch.Tensor], torch.Tensor],
               steps: int = 1,
               proposal_scale: float = 0.05) -> torch.Tensor:
    N, d = theta.shape
    cur = theta.clone()
    cur_lp = logpost_fn(cur)
    for _ in range(steps):
        prop = cur + proposal_scale * torch.randn_like(cur)
        prop_lp = logpost_fn(prop)
        acc = torch.rand(N, device=theta.device) < torch.exp((prop_lp - cur_lp).clamp(max=50))
        cur[acc] = prop[acc]
        cur_lp[acc] = prop_lp[acc]
    return cur

if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    print("=== Test: resample_indices ===")
    N = 1000
    # Skewed weights: make first 5 particles heavy
    raw_w = torch.cat([torch.full((5,), 5.0), torch.ones(N - 5)], dim=0).to(dtype).to(device)
    w = raw_w / raw_w.sum()

    for scheme in ["multinomial", "stratified", "systematic"]:
        idx = resample_indices(w, scheme=scheme)
        counts = torch.bincount(idx, minlength=N).to(dtype)
        top5_frac = counts[:5].sum() / counts.sum()
        print(f"{scheme:>11} | top-5 fraction: {top5_frac.item():.3f}  (expect >> 5/N={5/N:.3f})")

    print("\n=== Test: random_walk_move ===")
    theta = torch.zeros(N, 3, dtype=dtype, device=device)
    scale = 0.1
    moved = random_walk_move(theta, scale)
    emp_std = moved.std(dim=0)
    print("empirical std per-dim (should be ~ scale):", emp_std.cpu().numpy())

    print("\n=== Test: liu_west_move ===")
    # Build a nontrivial theta cloud with non-isotropic spread
    base = torch.randn(N, 3, dtype=dtype, device=device)
    theta = base @ torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=dtype, device=device)) + torch.tensor([2.0, -1.0, 0.5], dtype=dtype, device=device)
    # Make weights moderately skewed
    logits = -0.5 * ((theta - theta.mean(0)) ** 2).sum(-1)
    w = torch.softmax(logits, dim=0)
    a = 0.98
    moved_lw = liu_west_move(theta, w, a=a)  # uses h2 = 1 - a^2
    mean_before = (w[:, None] * theta).sum(0)
    mean_after = moved_lw.mean(0)
    print("weighted mean before:", mean_before.cpu().numpy())
    print("mean after (should be close):", mean_after.cpu().numpy())
    # Compare covariance scales roughly (not exact due to sampling)
    centered = theta - mean_before
    cov_before = (w[:, None] * centered).T @ centered
    centered_after = moved_lw - moved_lw.mean(0)
    cov_after = (centered_after.T @ centered_after) / (N - 1)
    eig_before = torch.linalg.eigvalsh(cov_before)
    eig_after = torch.linalg.eigvalsh(cov_after)
    print("eigvals before:", eig_before.detach().cpu().numpy())
    print("eigvals after :", eig_after.detach().cpu().numpy())

    print("\n=== Test: laplace_proposal ===")
    # Target: Gaussian log posterior: log pi(theta) = -0.5 (theta - mu)^T Prec (theta - mu)
    d = 2
    N_small = 5
    theta = torch.randn(N_small, d, dtype=dtype, device=device)
    mu = torch.tensor([1.0, -2.0], dtype=dtype, device=device)
    Prec = torch.tensor([[4.0, 1.0],
                         [1.0, 3.0]], dtype=dtype, device=device)  # SPD
    # Grad per particle: grad = -Prec @ (theta - mu)
    grad = -(theta - mu) @ Prec.T  # [N,d]
    # Hessian approx per particle: use same Prec for all
    hess = Prec.expand(N_small, d, d).clone()  # [N,d,d]
    alpha, beta = 0.5, 1e-4
    prop = laplace_proposal(theta, grad, hess, alpha=alpha, beta=beta)
    print("theta (first):", theta[0].cpu().numpy())
    print("grad  (first):", grad[0].cpu().numpy())
    print("prop  (first):", prop[0].cpu().numpy())
    # Empirical check: mean shift roughly in grad direction
    delta = (prop - theta).mean(0)
    print("avg step (should align with avg grad):", delta.cpu().numpy())

    print("\n=== Test: pmcmc_move ===")
    def logpost_fn(th: torch.Tensor) -> torch.Tensor:
        # Vectorized Gaussian log density up to a constant
        diff = th - mu
        return -0.5 * torch.einsum("ni,ij,nj->n", diff, Prec, diff)

    theta0 = torch.randn(N_small, d, dtype=dtype, device=device)
    theta1 = pmcmc_move(theta0, logpost_fn, steps=50, proposal_scale=0.1)
    dist0 = (theta0 - mu).norm(dim=1).mean()
    dist1 = (theta1 - mu).norm(dim=1).mean()
    print(f"mean ||theta - mu|| before: {dist0.item():.3f}  after: {dist1.item():.3f}  (should decrease)")

    print("\nAll tests finished.")
