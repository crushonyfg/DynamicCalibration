# =============================================================
# file: calib/emulator.py
# =============================================================
from typing import Callable, Tuple, Optional, Union, Sequence
import torch
from .delta_gp import OnlineGPState, make_gp_state
from .kernels import RBFKernel, Kernel


class Emulator:
    def predict(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
    def grad_theta(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


class DeterministicSimulator(Emulator):
    def __init__(self, func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], enable_autograd: bool = True):
        self.func = func
        self.enable_autograd = enable_autograd

    def predict(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x[None, :]
        b = x.shape[0]
        N = theta.shape[0]
        mu_list = []
        for n in range(N):
            out = self.func(x, theta[n:n+1, :])  # [b], [b,1], or [b, dy]
            if out.dim() == 1:
                out = out[:, None]  # [b, 1]
            mu_list.append(out)
        # Stack: each element [b, dy] -> stack dim=1 -> [b, N, dy]
        mu_eta = torch.stack(mu_list, dim=1)  # [b, N, dy] with dy>=1
        if mu_eta.shape[-1] == 1:
            mu_eta = mu_eta.squeeze(-1)  # [b, N] for backward compat
        var_eta = torch.zeros_like(mu_eta)
        return mu_eta, var_eta

    def grad_theta(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Gradient w.r.t. theta; supports only scalar output (dy=1). For dy>1, returns zeros."""
        if not self.enable_autograd:
            raise RuntimeError("grad_theta called but enable_autograd=False")
        if x.dim() == 1:
            x = x[None, :]
        b, N, dth = x.shape[0], theta.shape[0], theta.shape[1]
        dmu = torch.zeros(b, N, dth, dtype=theta.dtype, device=theta.device)
        for n in range(N):
            th = theta[n:n+1, :].detach().requires_grad_(True)
            y = self.func(x, th)
            out = y.squeeze(-1) if y.dim() > 1 else y
            if out.dim() == 1:
                for k in range(b):
                    grads = torch.autograd.grad(out[k], th, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if grads is None:
                        grads = torch.zeros_like(th)
                    dmu[k, n, :] = grads.squeeze(0)
            # else: multi-dim output, grad_theta not implemented; leave dmu zero
        return dmu, None


class GPEmulator(Emulator):
    """
    GP-based emulator for η(x,θ) over the joint input z = [x, θ].
    Uses exact GP state (OnlineGPState) and computes analytic gradients w.r.t. θ.
    """
    def __init__(
        self,
        X: torch.Tensor,
        theta: torch.Tensor,
        y: torch.Tensor,
        kernel: Optional[Union[RBFKernel, dict, Kernel]] = None,
        noise: float = 1e-3,
        mode: str = "exact_rank1",
        hyperparam_mode: str = "fixed",
    ):
        if X.dim() == 1:
            X = X[None, :]
        self.dx = X.shape[1]
        self.dth = theta.shape[1]
        Xt = torch.cat([X, theta], dim=1)  # [t, dx + dth]

        # Build GP state
        if isinstance(kernel, dict):
            self.gp = make_gp_state(mode, Xt, y, kernel, noise=noise, hyperparam_mode=hyperparam_mode)
        elif kernel is None:
            k = RBFKernel(lengthscale=1.0, variance=1.0)
            self.gp = make_gp_state(mode, Xt, y, k, noise=noise, hyperparam_mode=hyperparam_mode)
        else:
            self.gp = make_gp_state(mode, Xt, y, kernel, noise=noise, hyperparam_mode=hyperparam_mode)

    def predict(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x[None, :]
        b, N = x.shape[0], theta.shape[0]
        mu_eta = torch.zeros(b, N, dtype=x.dtype, device=x.device)
        var_eta = torch.zeros(b, N, dtype=x.dtype, device=x.device)
        for n in range(N):
            z = torch.cat([x, theta[n:n+1, :].repeat(b, 1)], dim=1)  # [b, dx+dth]
            mu, var = self.gp.predict(z)
            mu_eta[:, n] = mu
            var_eta[:, n] = var
        return mu_eta, var_eta

    def grad_theta(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Analytic gradients of mu_eta and var_eta w.r.t. theta using kernel derivatives.

        For mean:  mu(z) = k(z, X) @ alpha     =>  ∂mu/∂z = (∂k/∂z) @ alpha
        For var:   var(z) = k(z,z) - k(z,X) K^{-1} k(X,z)
                   ∂var/∂z = ∂k(z,z)/∂z - 2 * (∂k(z,X)/∂z) @ (K^{-1} k(X,z))
                   For stationary kernels, ∂k(z,z)/∂z = 0.
        We then slice the last dθ components (theta dimensions).
        """
        if x.dim() == 1:
            x = x[None, :]
        b, N, dth = x.shape[0], theta.shape[0], theta.shape[1]

        if not isinstance(self.gp, OnlineGPState):
            # If using SVGP adapter, you can either fall back to autograd or implement gpytorch grads.
            raise RuntimeError("grad_theta analytic path requires OnlineGPState (exact GP).")

        # Ensure cache is ready
        if "L" not in self.gp.cache or "alpha" not in self.gp.cache:
            self.gp._recompute_cache()

        Xtrain = self.gp.X                  # [t, d_tot]
        L: torch.Tensor = self.gp.cache["L"]        # [t, t]
        alpha: torch.Tensor = self.gp.cache["alpha"]  # [t]

        kernel = self.gp.kernel
        d_tot = Xtrain.shape[1]
        assert d_tot == (self.dx + self.dth), "dimension mismatch in GPEmulator"

        dmu = torch.zeros(b, N, dth, dtype=x.dtype, device=x.device)
        dvar = torch.zeros(b, N, dth, dtype=x.dtype, device=x.device)

        for n in range(N):
            # Build joint input z for all b with same theta_n
            z = torch.cat([x, theta[n:n+1, :].repeat(b, 1)], dim=1)  # [b, dx+dth]

            # Mean gradient: (∂k/∂z) @ alpha  -> shape [b, d_tot]
            dk_dz = kernel.cov_grad_z(z, Xtrain)      # [b, t, d_tot]
            gmu_full = torch.einsum("btd,t->bd", dk_dz, alpha)  # [b, d_tot]
            dmu[:, n, :] = gmu_full[:, self.dx:]      # take theta part

            # Var gradient:
            # k_zX = k(z, X) -> [b, t]
            k_zX = kernel.cov(z, Xtrain)              # [b, t]
            # a = K^{-1} k(X,z) = solve(L, solve(L^T, k(X,z))) -> [t, b]
            a = torch.cholesky_solve(k_zX.transpose(0, 1), L)  # [t, b]
            # ∂var/∂z = -2 * (∂k(z,X)/∂z)^T @ a(:, b)
            # For each b: gvar_full[b, :] = -2 * dk_dz[b, :, :].T @ a[:, b]
            gvar_full = -2.0 * torch.einsum("btd,tb->bd", dk_dz, a)  # [b, d_tot]
            dvar[:, n, :] = gvar_full[:, self.dx:]   # take theta part

        return dmu, dvar

    def append(self, X_new: torch.Tensor, theta_new: torch.Tensor, y_new: torch.Tensor, maybe_refit: bool = False):
        Xt_new = torch.cat([X_new, theta_new], dim=1)
        self.gp.append_batch(Xt_new, y_new, maybe_refit=maybe_refit)


# ------------------------- Minimal test -------------------------
if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"

    # Deterministic simulator test
    def f(x, theta):
        return (x @ theta.T)

    sim = DeterministicSimulator(f)
    x = torch.randn(3, 2, dtype=dtype, device=device)
    theta = torch.randn(4, 2, dtype=dtype, device=device)
    mu, var = sim.predict(x, theta)
    dmu, _ = sim.grad_theta(x, theta)
    print("DeterministicSimulator mu:", mu)
    print("DeterministicSimulator grad_theta:", dmu)

    # GPEmulator test on a 1D x, 1D theta toy
    n_train = 12
    X_train = torch.linspace(0, 1, n_train, dtype=dtype, device=device).unsqueeze(1)
    th_train = torch.zeros(n_train, 1, dtype=dtype, device=device)  # fix theta=0 for training
    y_train = torch.sin(2 * math.pi * X_train[:, 0]) + 0.05 * torch.randn(n_train, dtype=dtype, device=device)

    gp_emul = GPEmulator(
        X_train, th_train, y_train,
        kernel={"name": "rbf", "lengthscale": 0.5, "variance": 1.0},
        noise=1e-2,
        mode="exact_full",
        hyperparam_mode="fixed",
    )

    xq = torch.linspace(0, 1, 5, dtype=dtype, device=device).unsqueeze(1)
    thetaq = torch.tensor([[0.0], [0.2]], dtype=dtype, device=device)

    mu_gp, var_gp = gp_emul.predict(xq, thetaq)
    dmu_gp, dvar_gp = gp_emul.grad_theta(xq, thetaq)
    print("GPEmulator mu:", mu_gp)
    print("GPEmulator var:", var_gp)
    print("GPEmulator grad_theta (dmu):", dmu_gp)
    print("GPEmulator grad_theta (dvar):", dvar_gp)

    # Append a couple of points and recompute
    X_new = torch.tensor([[0.25], [0.75]], dtype=dtype, device=device)
    th_new = torch.zeros(2, 1, dtype=dtype, device=device)
    y_new = torch.sin(2 * math.pi * X_new[:, 0])
    gp_emul.append(X_new, th_new, y_new, maybe_refit=False)

    mu_gp2, var_gp2 = gp_emul.predict(xq, thetaq)
    dmu_gp2, dvar_gp2 = gp_emul.grad_theta(xq, thetaq)
    print("GPEmulator mu after append:", mu_gp2)
    print("GPEmulator grad_theta after append (dmu):", dmu_gp2)
    print("GPEmulator grad_theta after append (dvar):", dvar_gp2)
