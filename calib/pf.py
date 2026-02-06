# =============================================================
# file: calib/pf.py
# =============================================================
import math
from typing import Any, Callable, Dict
import torch
from .particles import ParticleSet
from .likelihood import loglik_and_grads
from .resampling import resample_indices, random_walk_move, liu_west_move, laplace_proposal, pmcmc_move
from .emulator import Emulator
from .delta_gp import OnlineGPState
from .configs import PFConfig

class ParticleFilter:
    def __init__(self, particles: ParticleSet, config: PFConfig, device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.particles = particles
        self.config = config
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_prior(cls, prior_sampler: Callable[[int], torch.Tensor], pf_config: PFConfig,
                   device: str = "cpu", dtype: torch.dtype = torch.float64, theta_anchor=None) -> "ParticleFilter":
        try:
            theta = prior_sampler(pf_config.num_particles, theta_anchor=theta_anchor).to(device=device, dtype=dtype)
        except:
            theta = prior_sampler(pf_config.num_particles).to(device=device, dtype=dtype)
        logw = torch.full((pf_config.num_particles,), -math.log(pf_config.num_particles), dtype=dtype, device=device)
        return cls(ParticleSet(theta=theta, logw=logw), pf_config, device, dtype)

    def _maybe_move(self, emulator: Emulator, delta_state: OnlineGPState, x_t: torch.Tensor, y_t: torch.Tensor,
                    rho: float, sigma_eps: float) -> None:
        move = self.config.move_strategy
        if move == "none":
            return
        w = self.particles.weights()
        if move == "random_walk":
            self.particles.theta = random_walk_move(self.particles.theta, self.config.random_walk_scale)
        elif move == "liu_west":
            self.particles.theta = liu_west_move(self.particles.theta, w, self.config.liu_west_a, self.config.liu_west_h2)
        elif move == "laplace":
            info = loglik_and_grads(y_t, x_t, self.particles, emulator, delta_state, rho, sigma_eps, 
                                need_grads=True, need_hessian=True, hessian_mode="fisher")
            grad = info["grad"]  # [N, dθ]
            hess = info["hess"]  # [N, dθ, dθ] - per-particle Hessian
            self.particles.theta = laplace_proposal(self.particles.theta, grad, hess, 
                                                self.config.laplace_alpha, self.config.laplace_beta, self.config.laplace_eta)
        elif move == "pmcmc":
            def logpost(th: torch.Tensor) -> torch.Tensor:
                ps = ParticleSet(theta=th, logw=torch.zeros(th.shape[0], dtype=self.dtype, device=self.device))
                ll = loglik_and_grads(y_t, x_t, ps, emulator, delta_state, rho, sigma_eps, need_grads=False)["loglik"]
                return ll  # flat prior
            self.particles.theta = pmcmc_move(self.particles.theta, logpost, steps=self.config.pmcmc_steps, proposal_scale=self.config.random_walk_scale)

    def step(self,
             x_t: torch.Tensor,
             y_t: torch.Tensor,
             emulator: Emulator,
             delta_state: OnlineGPState,
             rho: float,
             sigma_eps: float,
             grad_info: bool = False,
             use_discrepancy: bool = True) -> Dict[str, Any]:
        # Compute per-particle log-likelihood
        info = loglik_and_grads(y_t, x_t, self.particles, emulator, delta_state, rho, sigma_eps, need_grads=grad_info, use_discrepancy=use_discrepancy)
        loglik_n = info["loglik"]  # [N]
        # Evidence under current particles (mixture)
        logw = self.particles.logw
        logmix = torch.logsumexp(logw + loglik_n, dim=0)
        # Weight update
        self.particles.logw = logw + loglik_n
        self.particles.normalize_()
        ess = self.particles.ess().item()
        N = self.particles.theta.shape[0]
        resampled = False
        if ess < self.config.resample_ess_ratio * N:
            idx = resample_indices(self.particles.weights(), scheme=self.config.resample_scheme)
            self.particles.theta = self.particles.theta[idx]
            self.particles.logw = torch.full_like(self.particles.logw, -math.log(N))
            resampled = True
            # Optional: move step to rejuvenate diversity
            self._maybe_move(emulator, delta_state, x_t, y_t, rho, sigma_eps)
        # Return diagnostics
        return {
            "log_evidence": float(logmix),
            "ess": ess,
            "resampled": resampled,
            "maybe_grad": info.get("grad", None),
        }
    def step_batch(self,
                X_batch: torch.Tensor,  # [batch_size, dx]
                Y_batch: torch.Tensor,  # [batch_size]
                emulator: Emulator,
                delta_state: OnlineGPState,
                rho: float,
                sigma_eps: float,
                grad_info: bool = False,
                use_discrepancy: bool = True) -> Dict[str, Any]:
        """批量粒子滤波更新"""
        # 批量计算per-particle log-likelihood
        info = loglik_and_grads(Y_batch, X_batch, self.particles, emulator, delta_state, rho, sigma_eps, need_grads=grad_info, use_discrepancy=use_discrepancy)
        loglik_n = info["loglik"]  # [N] - 已经是sum over batch
        
        # Evidence under current particles (mixture)
        logw = self.particles.logw
        logmix = torch.logsumexp(logw + loglik_n, dim=0)
        
        # Weight update
        self.particles.logw = logw + loglik_n
        self.particles.normalize_()
        
        ess = self.particles.ess().item()
        gini = self.particles.gini().item()
        N = self.particles.theta.shape[0]
        resampled = False
        
        if ess < self.config.resample_ess_ratio * N:
            idx = resample_indices(self.particles.weights(), scheme=self.config.resample_scheme)
            self.particles.theta = self.particles.theta[idx]
            self.particles.logw = torch.full_like(self.particles.logw, -math.log(N))
            resampled = True
            # 批量移动步骤
            self._maybe_move_batch(emulator, delta_state, X_batch, Y_batch, rho, sigma_eps)
        
        return {
            "log_evidence": float(logmix),
            "ess": ess,
            "gini": gini,
            "resampled": resampled,
            "maybe_grad": info.get("grad", None),
        }

    def _maybe_move_batch(self, emulator: Emulator, delta_state: OnlineGPState, 
                        X_batch: torch.Tensor, Y_batch: torch.Tensor,
                        rho: float, sigma_eps: float) -> None:
        """批量移动策略"""
        move = self.config.move_strategy
        if move == "none":
            return
        
        # 对于批量数据，可以使用最后一个数据点进行移动
        x_t = X_batch[-1]  # 使用最后一个点
        y_t = Y_batch[-1]
        
        # 其他移动策略保持不变...
        w = self.particles.weights()
        if move == "random_walk":
            self.particles.theta = random_walk_move(self.particles.theta, self.config.random_walk_scale)
        elif move == "liu_west":
            self.particles.theta = liu_west_move(self.particles.theta, w, self.config.liu_west_a, self.config.liu_west_h2)
        elif move == "laplace":
            info = loglik_and_grads(Y_batch, X_batch, self.particles, emulator, delta_state, rho, sigma_eps, 
                                need_grads=True, need_hessian=True, hessian_mode="fisher")
            grad = info["grad"]  # [N, dθ]
            hess = info["hess"]  # [N, dθ, dθ] - per-particle Hessian
            self.particles.theta = laplace_proposal(self.particles.theta, grad, hess, 
                                                self.config.laplace_alpha, self.config.laplace_beta, self.config.laplace_eta)
        elif move == "pmcmc":
            def logpost(th: torch.Tensor) -> torch.Tensor:
                ps = ParticleSet(theta=th, logw=torch.zeros(th.shape[0], dtype=self.dtype, device=self.device))
                ll = loglik_and_grads(Y_batch, X_batch, ps, emulator, delta_state, rho, sigma_eps, need_grads=False)["loglik"]
                return ll  # flat prior
            self.particles.theta = pmcmc_move(self.particles.theta, logpost, steps=self.config.pmcmc_steps, proposal_scale=self.config.random_walk_scale)