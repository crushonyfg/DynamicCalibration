# =============================================================
# file: calib/kernels.py
# =============================================================
import math
import torch
from typing import Protocol, Union, Tuple
from .configs import DeltaKernelConfig


class Kernel(Protocol):
    def cov(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor: ...
    def diag(self, X: torch.Tensor) -> torch.Tensor: ...
    def cov_grad_z(self, z: torch.Tensor, Z: torch.Tensor) -> torch.Tensor: ...
    """
    Gradient of k(z, Z_i) w.r.t. the first argument z.
    Args:
        z: [b, d]
        Z: [n, d]
    Returns:
        dk_dz: [b, n, d] where dk_dz[b, i, :] = ∂k(z_b, Z_i)/∂z_b
    """


class RBFKernel:
    def __init__(self, lengthscale: Union[float, torch.Tensor] = 1.0, variance: float = 1.0):
        if isinstance(lengthscale, (float, int)):
            self.lengthscale = torch.tensor([float(lengthscale)])
        else:
            self.lengthscale = torch.as_tensor(lengthscale, dtype=torch.get_default_dtype())
        self.variance = float(variance)

    def _scaled(self, X: torch.Tensor) -> torch.Tensor:
        ls = self.lengthscale.to(X.device, X.dtype)
        return X / ls

    def _d2(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        Xs = self._scaled(X)
        Zs = self._scaled(Z)
        X2 = (Xs**2).sum(-1, keepdim=True)
        Z2 = (Zs**2).sum(-1, keepdim=True).transpose(0, 1)
        return (X2 + Z2 - 2.0 * Xs @ Zs.transpose(0, 1)).clamp_min(0.0)

    def cov(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        d2 = self._d2(X, Z)
        return torch.exp(-0.5 * d2) * self.variance

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full((X.shape[0],), fill_value=self.variance,
                          dtype=X.dtype, device=X.device)

    def cov_grad_z(self, z: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        For RBF: k(z,Z) = s2 * exp(-0.5 * sum((z-Z)^2 / ls^2))
        ∂k/∂z = k(z,Z) * ( (Z - z) / ls^2 )
        """
        s2 = self.variance
        ls = self.lengthscale.to(z.device, z.dtype)  # [d] or [1]
        inv_ls2 = 1.0 / (ls * ls)

        k = self.cov(z, Z)                # [b, n]
        diff = (Z[None, :, :] - z[:, None, :])  # [b, n, d]
        dk_dz = k[:, :, None] * (diff * inv_ls2)  # [b, n, d]
        return dk_dz


class Matern52Kernel:
    def __init__(self, lengthscale: Union[float, torch.Tensor] = 1.0, variance: float = 1.0):
        if isinstance(lengthscale, (float, int)):
            self.lengthscale = torch.tensor([float(lengthscale)])
        else:
            self.lengthscale = torch.as_tensor(lengthscale, dtype=torch.get_default_dtype())
        self.variance = float(variance)

    def _scaled(self, X: torch.Tensor) -> torch.Tensor:
        ls = self.lengthscale.to(X.device, X.dtype)
        return X / ls

    def _r(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        Xs = self._scaled(X)
        Zs = self._scaled(Z)
        X2 = (Xs**2).sum(-1, keepdim=True)
        Z2 = (Zs**2).sum(-1, keepdim=True).transpose(0, 1)
        d2 = (X2 + Z2 - 2.0 * Xs @ Zs.transpose(0, 1)).clamp_min(0.0)
        return torch.sqrt(d2 + 1e-12)

    def cov(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        r = self._r(X, Z)
        s = math.sqrt(5.0) * r
        return self.variance * (1.0 + s + (5.0/3.0) * r**2) * torch.exp(-s)

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full((X.shape[0],), fill_value=self.variance,
                          dtype=X.dtype, device=X.device)

    def cov_grad_z(self, z: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        For Matern-5/2:
            k(r) = s2 * (1 + s + 5/3 r^2) * exp(-s),  s = sqrt(5) r, r = sqrt(sum(((z-Z)/ls)^2))
        dk/dr = -(5/3) * s2 * exp(-s) * r * (1 + sqrt(5) * r)
        ∂r/∂z_d = (z_d - Z_d) / (r * ls_d^2)  (define 0 when r=0)
        => ∂k/∂z = dk/dr * ∂r/∂z
        """
        s2 = self.variance
        ls = self.lengthscale.to(z.device, z.dtype)
        inv_ls2 = 1.0 / (ls * ls)

        # Distances
        diff = (z[:, None, :] - Z[None, :, :])                 # [b, n, d]
        scaled2 = (diff * diff) * inv_ls2                      # [b, n, d]
        q = scaled2.sum(-1)                                    # [b, n]
        r = torch.sqrt(q + 1e-12)                              # [b, n]
        s = math.sqrt(5.0) * r                                 # [b, n]

        # k and dk/dr
        k = s2 * (1.0 + s + (5.0/3.0) * (r * r)) * torch.exp(-s)   # [b, n]
        dk_dr = -(5.0/3.0) * s2 * torch.exp(-s) * r * (1.0 + math.sqrt(5.0) * r)  # [b, n]

        # ∂r/∂z_d
        # Handle r=0 safely: when r≈0, gradient should be zero (kernel peaks at r=0).
        r_safe = r.clone()
        r_safe[r_safe < 1e-12] = 1.0
        dr_dz = diff * inv_ls2[None, None, :] / r_safe[:, :, None]  # [b, n, d]

        dk_dz = dk_dr[:, :, None] * dr_dz  # [b, n, d]
        # Zero out gradients where r==0 (numerical safety)
        mask_zero = (r < 1e-12).unsqueeze(-1)  # [b, n, 1]
        dk_dz = torch.where(mask_zero, torch.zeros_like(dk_dz), dk_dz)
        return dk_dz


def make_kernel(cfg: DeltaKernelConfig) -> Kernel:
    if cfg.name.lower() in ("rbf", "se", "sqexp"):
        return RBFKernel(cfg.lengthscale, cfg.variance)
    elif cfg.name.lower() in ("matern52", "m52"):
        return Matern52Kernel(cfg.lengthscale, cfg.variance)
    else:
        raise ValueError(f"Unknown kernel: {cfg.name}")


if __name__ == "__main__":
    X = torch.randn(4, 3)
    Z = torch.randn(5, 3)

    # single lengthscale
    k1 = RBFKernel(lengthscale=1.0, variance=2.0)
    print("RBF single ls:", k1.cov(X, Z).shape)
    dk_dz = k1.cov_grad_z(X, Z)
    print("RBF single ls grad:", dk_dz.shape)

    # vector lengthscale
    k2 = RBFKernel(lengthscale=torch.tensor([0.5, 1.0, 2.0]), variance=2.0)
    print("RBF vector ls:", k2.cov(X, Z).shape)

    # via make_kernel
    cfg = DeltaKernelConfig(name="matern52", lengthscale=[0.3, 0.7, 1.2], variance=1.5)
    k3 = make_kernel(cfg)
    print("Matern cov shape:", k3.cov(X, Z).shape)
