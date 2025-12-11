# =============================================================
# file: calib/pf.py
# =============================================================
import math
from typing import Any, Callable, Dict
import torch
from .particles import ParticleSet
from .likelihood import *
from .resampling import resample_indices, random_walk_move, liu_west_move, laplace_proposal, pmcmc_move
from .emulator import Emulator
from .delta_gp import OnlineGPState
from .configs import PFConfig
from .kernels import Kernel, make_kernel

from tqdm import tqdm
from typing import Tuple, Literal, List
from dataclasses import dataclass


def predictive_stats_mod(
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
    if mu_delta.dim() == 1:
        mu = rho * mu_eta + mu_delta[:, None]
        var = (rho**2) * var_eta + var_delta[:, None] + (sigma_eps**2)
        var = var.clamp_min(1e-12)
    else:
        mu = rho * mu_eta + mu_delta
        var = (rho**2) * var_eta + var_delta + (sigma_eps**2)
        var = var.clamp_min(1e-12)
    return mu, var

def loglik_and_grads_mod(
    y: torch.Tensor,
    x: torch.Tensor,
    particles: ParticleSet,
    emulator: Emulator,
    delta_states: List[OnlineGPState],
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

    if isinstance(delta_states, dict):
        N = max(max(index_list) for (_, index_list) in delta_states.values())+1
        mu_list, var_list = [None] * N, [None] * N
        for key, (delta_state, index_list) in delta_states.items():
            mu, var = delta_state.predict(x)
            mu, var = mu.squeeze(), var.squeeze()
            for idx in index_list:
                mu_list[idx] = mu
                var_list[idx] = var
        mu_delta, var_delta = torch.stack(mu_list, dim=1), torch.stack(var_list, dim=1)
    else:    
        if len(delta_states) == 1:
            mu_delta, var_delta = delta_states[0].predict(x)
        else:
            mu_list, var_list = [], []
            for delta_state in delta_states:
                mu, var = delta_state.predict(x) # [b],[b]
                mu_list.append(mu.squeeze())
                var_list.append(var.squeeze())
            mu_delta, var_delta = torch.stack(mu_list, dim=1), torch.stack(var_list, dim=1)

    # mu_delta, var_delta = [delta_state.predict(x) for delta_state in delta_states]            # [b],[b]
    mu_tot, var_tot = predictive_stats_mod(rho, mu_eta, var_eta, mu_delta, var_delta, sigma_eps)

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

class ParticleFilter:
    def __init__(self, particles: ParticleSet, config: PFConfig, prior_sampler: Callable[[int], torch.Tensor], device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.particles = particles
        self.config = config
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_prior(cls, prior_sampler: Callable[[int], torch.Tensor], pf_config: PFConfig,
                   device: str = "cpu", dtype: torch.dtype = torch.float64) -> "ParticleFilter":
        theta = prior_sampler(pf_config.num_particles).to(device=device, dtype=dtype)
        logw = torch.full((pf_config.num_particles,), -math.log(pf_config.num_particles), dtype=dtype, device=device)
        return cls(ParticleSet(theta=theta, logw=logw), pf_config, device, dtype)

    def step_batch(self,
                X_batch: torch.Tensor,  # [batch_size, dx]
                Y_batch: torch.Tensor,  # [batch_size]
                emulator: Emulator,
                e,
                model_cfg,
                # delta_states: List[OnlineGPState],
                rho: float,
                sigma_eps: float,
                prior_sampler: Callable[[int], torch.Tensor],
                grad_info: bool = False) -> Dict[str, Any]:
        """批量粒子滤波更新"""
        # 批量计算per-particle log-likelihood
        self.particles.theta = random_walk_move(self.particles.theta, self.config.random_walk_scale)
        X_hist, Y_hist = e.X_hist, e.y_hist
        mu_eta_all, _ = emulator.predict(X_hist, self.particles.theta) # [M, N]
        resid_all = Y_hist[:, None] - rho * mu_eta_all

        delta_states = []
        for i in range(resid_all.shape[1]):
            try:
                old_kernel = e.delta_states[i].kernel
                old_noise = e.delta_states[i].noise
            except:
                old_kernel = make_kernel(model_cfg.delta_kernel)
                old_noise = model_cfg.delta_kernel.noise

            gp = OnlineGPState(
                X=X_hist,
                y=resid_all[:, i],
                kernel=old_kernel,
                noise=old_noise,
                update_mode="exact_full",
                hyperparam_mode="fit",
            )
            gp.refit_hyperparams(max_iter=5, lr=0.05)
            delta_states.append(gp)

        e.delta_states = delta_states

        info = loglik_and_grads_mod(Y_batch, X_batch, self.particles, emulator, delta_states, rho, sigma_eps, need_grads=grad_info)
        loglik_n = info["loglik"]  # [N] - 已经是sum over batch
        
        # Evidence under current particles (mixture)
        logw = self.particles.logw
        logmix = torch.logsumexp(logw + loglik_n, dim=0)
        
        # Weight update
        self.particles.logw = logw + loglik_n
        self.particles.normalize_()
        
        ess = self.particles.ess().item()
        N = self.particles.theta.shape[0]
        resampled = False
        device, dtype = self.particles.theta.device, self.particles.theta.dtype
        
        if ess < self.config.resample_ess_ratio * N:
            # idx = resample_indices(self.particles.weights(), scheme=self.config.resample_scheme)
            # self.particles.theta = self.particles.theta[idx]
            # e.delta_states = [delta_states[i] for i in idx.tolist()]

            self.particles.theta = prior_sampler(N).to(device=device, dtype=dtype)
            self.particles.logw = torch.full_like(self.particles.logw, -math.log(N))
            resampled = True
            # 批量移动步骤
            mu_eta_all, _ = emulator.predict(X_hist, self.particles.theta)
            resid_all = Y_hist[:, None] - rho * mu_eta_all

            new_delta_states = []
            for i in range(N):
                gp = OnlineGPState(
                    X=X_hist,
                    y=resid_all[:, i],
                    kernel=make_kernel(model_cfg.delta_kernel),
                    noise=model_cfg.delta_kernel.noise,
                    update_mode="exact_full",
                    hyperparam_mode="fit",
                )
                gp.refit_hyperparams(max_iter=5, lr=0.05)
                new_delta_states.append(gp)

            e.delta_states = new_delta_states
        
        return e.delta_states, {
            "log_evidence": float(logmix),
            "ess": ess,
            "resampled": resampled,
            "maybe_grad": info.get("grad", None),
        }

    def _maybe_move_batch(self, emulator: Emulator, delta_states: List[OnlineGPState], 
                        X_batch: torch.Tensor, Y_batch: torch.Tensor,
                        rho: float, sigma_eps: float) -> None:
        """批量移动策略"""
        move = self.config.move_strategy
        if move == "none":
            return
        
        w = self.particles.weights()
        if move == "random_walk":
            self.particles.theta = random_walk_move(self.particles.theta, self.config.random_walk_scale)
        elif move == "liu_west":
            self.particles.theta = liu_west_move(self.particles.theta, w, self.config.liu_west_a, self.config.liu_west_h2)
        elif move == "laplace":
            info = loglik_and_grads_mod(Y_batch, X_batch, self.particles, emulator, delta_states, rho, sigma_eps, 
                                need_grads=True, need_hessian=True, hessian_mode="fisher")
            grad = info["grad"]  # [N, dθ]
            hess = info["hess"]  # [N, dθ, dθ] - per-particle Hessian
            self.particles.theta = laplace_proposal(self.particles.theta, grad, hess, 
                                                self.config.laplace_alpha, self.config.laplace_beta, self.config.laplace_eta)
        elif move == "pmcmc":
            def logpost(th: torch.Tensor) -> torch.Tensor:
                ps = ParticleSet(theta=th, logw=torch.zeros(th.shape[0], dtype=self.dtype, device=self.device))
                ll = loglik_and_grads_mod(Y_batch, X_batch, ps, emulator, delta_states, rho, sigma_eps, need_grads=False)["loglik"]
                return ll  # flat prior
            self.particles.theta = pmcmc_move(self.particles.theta, logpost, steps=self.config.pmcmc_steps, proposal_scale=self.config.random_walk_scale)