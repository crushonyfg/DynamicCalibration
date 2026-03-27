from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import torch

from .emulator import Emulator


def _to_lengthscale_tensor(lengthscale, dx: int, *, device, dtype) -> torch.Tensor:
    ls = torch.as_tensor(lengthscale, device=device, dtype=dtype).reshape(-1)
    if ls.numel() == 1:
        ls = ls.repeat(dx)
    if ls.numel() != dx:
        raise ValueError(f"lengthscale dimension mismatch: expected {dx}, got {ls.numel()}")
    return ls.clamp_min(1e-8)


def _rbf_cov(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    variance: float,
) -> torch.Tensor:
    x1s = x1 / lengthscale
    x2s = x2 / lengthscale
    dist2 = (x1s[:, None, :] - x2s[None, :, :]).pow(2).sum(dim=-1)
    return float(variance) * torch.exp(-0.5 * dist2)


@dataclass
class KernelHyperSpec:
    lengthscale: torch.Tensor
    variance: float
    noise: float


class ParticleSpecificGPDeltaState:
    def __init__(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        hyper_specs: Sequence[KernelHyperSpec],
    ):
        self.X_hist = X_hist
        self.Y_hist = Y_hist.reshape(-1)
        self.theta_particles = theta_particles
        self.emulator = emulator
        self.rho = float(rho)
        self.device = X_hist.device
        self.dtype = X_hist.dtype
        self.dx = int(X_hist.shape[1])
        self.hyper_specs = [
            KernelHyperSpec(
                lengthscale=_to_lengthscale_tensor(spec.lengthscale, self.dx, device=self.device, dtype=self.dtype),
                variance=float(spec.variance),
                noise=max(float(spec.noise), 1e-8),
            )
            for spec in hyper_specs
        ]
        self._kernel_caches = [self._build_kernel_cache(spec) for spec in self.hyper_specs]
        self._current_stats = self._build_particle_stats(self.theta_particles)

    def _build_kernel_cache(self, spec: KernelHyperSpec):
        K = _rbf_cov(self.X_hist, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
        K = K + (spec.noise + 1e-6) * torch.eye(K.shape[0], device=self.device, dtype=self.dtype)
        chol = torch.linalg.cholesky(K)
        logdet = 2.0 * torch.log(torch.diag(chol)).sum()
        return {"spec": spec, "chol": chol, "logdet": logdet}

    def _residual_matrix(self, theta_particles: torch.Tensor) -> torch.Tensor:
        mu_eta_hist, _ = self.emulator.predict(self.X_hist, theta_particles)
        if mu_eta_hist.dim() == 3:
            mu_eta_hist = mu_eta_hist.mean(dim=-1)
        return self.Y_hist[:, None] - self.rho * mu_eta_hist

    def _build_particle_stats(self, theta_particles: torch.Tensor):
        R = self._residual_matrix(theta_particles)
        alphas: List[torch.Tensor] = []
        log_evidences: List[torch.Tensor] = []
        n = int(self.X_hist.shape[0])
        const = -0.5 * n * math.log(2.0 * math.pi)
        for cache in self._kernel_caches:
            chol = cache["chol"]
            alpha = torch.cholesky_solve(R, chol)
            quad = (R * alpha).sum(dim=0)
            logev = const - 0.5 * quad - 0.5 * cache["logdet"]
            alphas.append(alpha)
            log_evidences.append(logev)
        if len(alphas) == 1:
            weights = torch.ones(1, theta_particles.shape[0], device=self.device, dtype=self.dtype)
        else:
            weights = torch.softmax(torch.stack(log_evidences, dim=0), dim=0)
        return {"alphas": alphas, "weights": weights}

    def _predict_with_stats(self, x: torch.Tensor, stats) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device, self.dtype)
        mus = []
        vars_ = []
        N = int(stats["weights"].shape[1])
        for alpha, cache in zip(stats["alphas"], self._kernel_caches):
            spec = cache["spec"]
            K_qh = _rbf_cov(x, self.X_hist, lengthscale=spec.lengthscale, variance=spec.variance)
            mu_h = K_qh @ alpha
            solve = torch.cholesky_solve(K_qh.transpose(0, 1), cache["chol"])
            base_var = spec.variance - (K_qh * solve.transpose(0, 1)).sum(dim=1)
            base_var = (base_var + spec.noise).clamp_min(1e-12)
            var_h = base_var[:, None].expand(-1, N)
            mus.append(mu_h)
            vars_.append(var_h)
        if len(mus) == 1:
            return mus[0], vars_[0]
        weights = stats["weights"]
        mix_mu = torch.zeros_like(mus[0])
        mix_second = torch.zeros_like(mus[0])
        for h, (mu_h, var_h) in enumerate(zip(mus, vars_)):
            w = weights[h][None, :]
            mix_mu = mix_mu + w * mu_h
            mix_second = mix_second + w * (var_h + mu_h.pow(2))
        mix_var = (mix_second - mix_mu.pow(2)).clamp_min(1e-12)
        return mix_mu, mix_var

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._predict_with_stats(x, self._current_stats)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rho
        if emulator is not None:
            self.emulator = emulator
        stats = self._build_particle_stats(theta_particles.to(self.device, self.dtype))
        return self._predict_with_stats(x, stats)


class ParticleSpecificBasisDeltaState:
    def __init__(
        self,
        X_hist: torch.Tensor,
        Y_hist: torch.Tensor,
        theta_particles: torch.Tensor,
        emulator: Emulator,
        rho: float,
        *,
        basis_kind: str = "rbf",
        num_features: int = 8,
        lengthscale: float = 0.25,
        ridge: float = 1e-2,
        noise: float = 1e-3,
    ):
        self.X_hist = X_hist
        self.Y_hist = Y_hist.reshape(-1)
        self.theta_particles = theta_particles
        self.emulator = emulator
        self.rho = float(rho)
        self.device = X_hist.device
        self.dtype = X_hist.dtype
        self.basis_kind = str(basis_kind).lower()
        self.num_features = max(1, int(num_features))
        self.lengthscale = max(float(lengthscale), 1e-8)
        self.ridge = max(float(ridge), 1e-8)
        self.noise = max(float(noise), 1e-8)
        self.centers = self._select_centers()
        self.Phi_hist = self._basis(self.X_hist)
        self._chol_A = self._build_precision_chol(self.Phi_hist)
        self._current_beta = self._solve_beta(self.theta_particles)

    def _select_centers(self) -> Optional[torch.Tensor]:
        if self.basis_kind != "rbf":
            return None
        n = int(self.X_hist.shape[0])
        m = min(self.num_features, n)
        idx = torch.linspace(0, n - 1, m, device=self.device).round().long().unique(sorted=True)
        return self.X_hist[idx]

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, self.dtype)
        if self.basis_kind == "linear":
            return torch.cat([torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype), x], dim=1)
        if self.centers is None:
            raise ValueError("rbf basis requires centers")
        dist2 = (x[:, None, :] - self.centers[None, :, :]).pow(2).sum(dim=-1)
        feats = torch.exp(-0.5 * dist2 / (self.lengthscale ** 2))
        return torch.cat([torch.ones(x.shape[0], 1, device=self.device, dtype=self.dtype), feats], dim=1)

    def _build_precision_chol(self, Phi: torch.Tensor) -> torch.Tensor:
        p = int(Phi.shape[1])
        A = (Phi.transpose(0, 1) @ Phi) / self.noise + self.ridge * torch.eye(p, device=self.device, dtype=self.dtype)
        return torch.linalg.cholesky(A)

    def _residual_matrix(self, theta_particles: torch.Tensor) -> torch.Tensor:
        mu_eta_hist, _ = self.emulator.predict(self.X_hist, theta_particles)
        if mu_eta_hist.dim() == 3:
            mu_eta_hist = mu_eta_hist.mean(dim=-1)
        return self.Y_hist[:, None] - self.rho * mu_eta_hist

    def _solve_beta(self, theta_particles: torch.Tensor) -> torch.Tensor:
        R = self._residual_matrix(theta_particles)
        rhs = (self.Phi_hist.transpose(0, 1) @ R) / self.noise
        return torch.cholesky_solve(rhs, self._chol_A)

    def _predict_with_beta(self, x: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Phi_x = self._basis(x)
        mu = Phi_x @ beta
        solve = torch.cholesky_solve(Phi_x.transpose(0, 1), self._chol_A)
        base_var = (Phi_x * solve.transpose(0, 1)).sum(dim=1)
        var = (base_var + self.noise)[:, None].expand(-1, beta.shape[1]).clamp_min(1e-12)
        return mu, var

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._predict_with_beta(x, self._current_beta)

    def predict_for_particles(
        self,
        x: torch.Tensor,
        theta_particles: torch.Tensor,
        *,
        emulator: Optional[Emulator] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rho
        if emulator is not None:
            self.emulator = emulator
        beta = self._solve_beta(theta_particles.to(self.device, self.dtype))
        return self._predict_with_beta(x, beta)
