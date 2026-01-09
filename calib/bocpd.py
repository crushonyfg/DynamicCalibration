# =============================================================
# file: calib/bocpd.py
# =============================================================
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math
import torch

from .pf import ParticleFilter
from .particles import ParticleSet
from .delta_gp import OnlineGPState
from .likelihood import loglik_and_grads, predictive_stats
from .kernels import make_kernel
from .configs import BOCPDConfig, ModelConfig, PFConfig
from .emulator import Emulator
from .expert_delta import ExpertDeltaFitter  # <--- NEW


@dataclass
class Expert:
    run_length: int
    pf: ParticleFilter
    delta_state: OnlineGPState
    log_mass: float  # log posterior mass for this expert
    # NEW: raw history for this expert's run-length segment
    X_hist: torch.Tensor  # [M, dx]
    y_hist: torch.Tensor  # [M]
    # X_buffer: torch.Tensor  # [batch_size, dx]
    # resid_buffer: torch.Tensor  # [batch_size]
    # buffer_size: int = 5


class BOCPD:
    def __init__(
        self,
        config: BOCPDConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        delta_fitter: Optional[ExpertDeltaFitter] = None,  # <--- NEW
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.experts: List[Expert] = []
        self.t: int = 0  # time index
        self.delta_fitter = delta_fitter

        # Optional knobs (read lazily, keep backward compatible)
        self.delta_refit_every: int = int(getattr(config, "delta_refit_every", 0))  # 0 => disabled
        self.delta_refit_topk: int = int(getattr(config, "delta_refit_topk", 1))

    def _haz(self, r: int) -> float:
        t = torch.tensor(float(r), dtype=self.dtype, device=self.device)
        val = self.config.hazard(t)
        return float(val if isinstance(val, torch.Tensor) else val)

    def _init_empty_delta_state(self, dx: int, model_cfg: ModelConfig) -> OnlineGPState:
        kernel = make_kernel(model_cfg.delta_kernel)
        return OnlineGPState(
            X=torch.empty(0, dx, dtype=model_cfg.dtype, device=model_cfg.device),
            y=torch.empty(0, dtype=model_cfg.dtype, device=model_cfg.device),
            kernel=kernel,
            noise=model_cfg.delta_kernel.noise,
            update_mode="exact_full",  # you can change if you prefer exact_full
            hyperparam_mode="fit",
        )

    def _spawn_new_expert(self, pf_cfg: PFConfig, model_cfg: ModelConfig, dx: int,
                          prior_sampler: Callable[[int], torch.Tensor],
                          log_mass: float) -> Expert:
        delta_new = self._init_empty_delta_state(dx, model_cfg)
        pf_new = ParticleFilter.from_prior(prior_sampler, pf_cfg, model_cfg.device, model_cfg.dtype)
        # start with empty history; we will append (x_t,y_t) below
        return Expert(run_length=0, pf=pf_new, delta_state=delta_new,
                      log_mass=log_mass,
                      X_hist=torch.empty(0, dx, dtype=model_cfg.dtype, device=model_cfg.device),
                      y_hist=torch.empty(0, dtype=model_cfg.dtype, device=model_cfg.device))

    def _append_hist(self, e: Expert, x_t: torch.Tensor, y_t: torch.Tensor, max_len: int) -> None:
        """
        Append current (x_t, y_t) to expert's raw history and truncate to last max_len points.
        Also synchronize delta_state if needed.
        """
        x_t2 = x_t.view(1, -1)
        y_t2 = y_t.view(1)
        if e.X_hist.numel() == 0:
            e.X_hist = x_t2
            e.y_hist = y_t2
        else:
            e.X_hist = torch.cat([e.X_hist, x_t2], dim=0)
            e.y_hist = torch.cat([e.y_hist, y_t2], dim=0)
        
        # 截断历史
        if e.X_hist.shape[0] > max_len:
            e.X_hist = e.X_hist[-max_len:, :]
            e.y_hist = e.y_hist[-max_len:]
            
            # 同步delta_state（重建）
            if e.delta_state.X.shape[0] > max_len:
                e.delta_state.X = e.delta_state.X[-max_len:, :]
                e.delta_state.y = e.delta_state.y[-max_len:]
                e.delta_state._recompute_cache_full()

    def ump(self, x_t: torch.Tensor, y_t: torch.Tensor, emulator: Emulator, model_cfg: ModelConfig) -> List[float]:
        # Compute per-expert predictive mixture likelihood (before PF weight update)
        umps: List[float] = []
        for e in self.experts:
            ps = e.pf.particles
            info = loglik_and_grads(y_t, x_t, ps, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, need_grads=False)
            loglik = info["loglik"]
            loglik = torch.clamp(loglik, min=-1e10, max=1e10)
            logmix = torch.logsumexp(ps.logw + loglik, dim=0)
            umps.append(float(logmix))
        return umps


    def update(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        verbose: bool = False,  # ✅ 新增：控制是否输出诊断信息
    ) -> Dict[str, Any]:
        """
        One BOCPD update with a new observation (x_t, y_t):
          - update run-length masses
          - spawn new r=0 expert
          - prune to top-K experts
          - PF step per expert and update delta residuals
          - optionally (every K steps) refit expert-shared delta hyperparams for top-k experts
        """
        x_t = x_t.to(self.device, self.dtype)
        y_t = y_t.to(self.device, self.dtype)
        dx = x_t.numel()

        # Bootstrap first expert if needed
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=0.0)
            self.experts = [e0]

        # 1) UMP per expert (log space)
        log_umps = self.ump(x_t, y_t, emulator, model_cfg)  # list[float]
        log_umps_t = torch.tensor(log_umps, dtype=self.dtype, device=self.device)

        # ✅ 诊断输出 1: UMP值
        if verbose:
            print(f"\n{'─'*70}")
            print(f"Time step t={self.t}")
            print(f"Observation: x_t={x_t.cpu().numpy()}, y_t={y_t.item():.4f}")
            print(f"\n📊 UMP (log predictive likelihood) per expert:")
            for i, (e, log_ump) in enumerate(zip(self.experts, log_umps)):
                print(f"  Expert {i} (r={e.run_length:2d}): log_UMP = {log_ump:10.3f}")

        # 2) BOCPD mass update
        prev_log_mass = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        hazards = torch.tensor([self._haz(e.run_length) for e in self.experts], dtype=self.dtype, device=self.device)
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        # ✅ 诊断输出 2: Hazard函数
        if verbose:
            print(f"\n🎲 Hazard function h(r) per expert:")
            for i, (e, h) in enumerate(zip(self.experts, hazards)):
                print(f"  Expert {i} (r={e.run_length:2d}): h(r) = {h:.6f}, "
                      f"1-h(r) = {(1-h):.6f}")

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        # ✅ 诊断输出 3: Mass更新前
        if verbose:
            print(f"\n📈 Mass update (before normalization):")
            print(f"  {'Expert':<8} {'r':<4} {'prev_mass':<12} {'log_UMP':<12} "
                  f"{'1-h(r)':<10} {'growth_term':<15}")
            for i, e in enumerate(self.experts):
                prev_mass = math.exp(float(prev_log_mass[i]))
                growth_term = float(growth_log_mass[i])
                print(f"  Exp-{i:<3} {e.run_length:<4} {prev_mass:<12.6f} "
                      f"{log_umps[i]:<12.3f} {(1-hazards[i]):<10.6f} "
                      f"{growth_term:<15.3f}")
            print(f"  {'CP term':<8} {'---':<4} {'(sum)':<12} {'---':<12} "
                  f"{'---':<10} {float(cp_log_mass):<15.3f}")

        # 3) Update existing experts (run-length + mass) and spawn the new r=0 expert
        for i, e in enumerate(self.experts):
            e.run_length = min(e.run_length + 1, self.config.max_run_length)
            e.log_mass = float(growth_log_mass[i])
        new_expert = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=float(cp_log_mass))
        self.experts.append(new_expert)

        # 4) Normalize masses and prune top-K
        masses = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        log_Z = torch.logsumexp(masses, dim=0)  # ✅ 保存归一化常数
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # ✅ 诊断输出 4: 归一化后
        if verbose:
            print(f"\n🔄 After normalization (log_Z = {float(log_Z):.3f}):")
            print(f"  {'Expert':<8} {'r':<4} {'log_mass':<15} {'mass':<12} {'mass(%)':<10}")
            for i, e in enumerate(self.experts):
                mass = math.exp(e.log_mass)
                print(f"  Exp-{i:<3} {e.run_length:<4} {e.log_mass:<15.3f} "
                      f"{mass:<12.6f} {mass*100:<10.2f}%")

        # sort & prune
        self.experts.sort(key=lambda e: e.log_mass, reverse=True)
        pruned_experts = self.experts[self.config.max_experts:]
        self.experts = self.experts[: self.config.max_experts]

        if verbose and len(pruned_experts) > 0:
            print(f"\n✂️  Pruned {len(pruned_experts)} experts (keeping top-{self.config.max_experts})")

        # 5) Append (x_t, y_t) into each retained expert's raw history (truncate to max_run_length)
        for e in self.experts:
            self._append_hist(e, x_t, y_t, max_len=self.config.max_run_length)

        # 6) PF step per expert, then update delta residual with mixture eta (using updated weights)
        diags = []
        for e in self.experts:
            diag = e.pf.step(x_t, y_t, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, grad_info=False)
            diags.append(diag)

            # Use updated particle weights to build residual and append to delta_state
            mu_eta, _ = emulator.predict(x_t.view(1, -1), e.pf.particles.theta)  # [1,N]
            w = e.pf.particles.weights().view(1, -1)
            eta_mix = (w * mu_eta).sum(dim=1).squeeze(0)
            resid = y_t - model_cfg.rho * eta_mix
            e.delta_state.append(x_t, resid)

        # 7) Optionally refit delta hyperparameters for top-k experts every `delta_refit_every` steps
        self.t += 1
        did_refit = False
        if self.delta_fitter is not None and self.delta_refit_every > 0 and (self.t % self.delta_refit_every == 0):
            topk = min(self.delta_refit_topk, len(self.experts))
            topk_indices = []
            for i in range(min(topk, len(self.experts))):
                e = self.experts[i]
                # 只重拟合有足够数据的expert
                if e.X_hist.shape[0] >= 10:  
                    topk_indices.append(i)
            
            if topk_indices:
                _ = self.delta_fitter.refit_topk(
                    experts=self.experts,
                    emulator=emulator,
                    rho=model_cfg.rho,
                    sigma_eps=model_cfg.sigma_eps,
                    topk_indices=topk_indices,
                    init_from_current=True,
                )
                did_refit = True

        r0_expert = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = float(math.exp(r0_expert.log_mass)) if r0_expert else 0.0

        # 1. 所有小 run-length (r <= 5) experts 的总 mass
        small_r_mass = sum(math.exp(e.log_mass) for e in self.experts if e.run_length <= 5)
        
        # 2. 最大 mass expert 的 run-length
        max_mass_expert = max(self.experts, key=lambda e: e.log_mass)
        dominant_run_length = max_mass_expert.run_length
        
        # 3. Bayes factor: P(r<=5) / P(r>5)
        large_r_mass = sum(math.exp(e.log_mass) for e in self.experts if e.run_length > 5)
        bayes_factor_cp = small_r_mass / (large_r_mass + 1e-12)
        
        # 4. 熵：run-length 分布的不确定性
        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)
        
        # 5. 最主导 expert 的 UMP（如果突然下降，说明模型失效）
        if hasattr(self, 'prev_max_ump'):
            ump_drop = self.prev_max_ump - max(log_umps) if log_umps else 0.0
        else:
            ump_drop = 0.0
        self.prev_max_ump = max(log_umps) if log_umps else 0.0

        if self.config.use_restart:
            did_restart = False
            if p_cp > self.config.restart_threshold and r0_expert is not None:
                # 删除所有 r>0 的 experts，只保留 r=0
                self.experts = [r0_expert]
                did_restart = True
                if verbose:
                    print(f"\n🔄 RESTART triggered! p_cp={p_cp:.4f} > {self.config.restart_threshold}")
                    print(f"   Removed all experts with r>0, keeping only r=0")
        
        # ✅ 诊断输出 5: 最终状态
        if verbose:
            print(f"\n✅ Final state: P(changepoint) = {p_cp:.4f}")
            print(f"{'─'*70}\n")
        
        return {
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": diags,
            "did_delta_refit": did_refit,
            # ✅ 新增：详细诊断信息
            "log_umps": log_umps,
            "hazards": hazards.cpu().numpy().tolist(),
            "log_Z": float(log_Z) if 'log_Z' in locals() else 0.0,
        }
    
    def update_batch(
        self,
        X_batch: torch.Tensor,  # [batch_size, dx]
        Y_batch: torch.Tensor,  # [batch_size]
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        批量更新BOCPD，一次性处理整个batch的数据点
        """
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0]
        dx = X_batch.shape[1]

        # Bootstrap first expert if needed
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=0.0)
            self.experts = [e0]

        # 1) UMP per expert (log space) - 批量计算
        log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)  # list[float]
        log_umps_t = torch.tensor(log_umps, dtype=self.dtype, device=self.device)

        # 2) BOCPD mass update (批量版本)
        prev_log_mass = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        hazards = torch.tensor([self._haz(e.run_length) for e in self.experts], dtype=self.dtype, device=self.device)
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        # 3) Update existing experts (run-length + mass) and spawn the new r=0 expert
        for i, e in enumerate(self.experts):
            e.run_length = min(e.run_length + batch_size, self.config.max_run_length)
            e.log_mass = float(growth_log_mass[i])
        new_expert = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=float(cp_log_mass))
        self.experts.append(new_expert)

        # 4) Normalize masses and prune top-K
        masses = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # sort & prune
        self.experts.sort(key=lambda e: e.log_mass, reverse=True)
        pruned_experts = self.experts[self.config.max_experts:]
        self.experts = self.experts[: self.config.max_experts]

        # 5) Append batch data to each retained expert's raw history
        for e in self.experts:
            self._append_hist_batch(e, X_batch, Y_batch, max_len=self.config.max_run_length)

        # 6) PF step per expert (批量版本)
        diags = []
        for e in self.experts:
            diag = e.pf.step_batch(X_batch, Y_batch, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, grad_info=False)
            diags.append(diag)

            # Use updated particle weights to build residual and append to delta_state (批量)
            # mu_eta, _ = emulator.predict(X_batch, e.pf.particles.theta)  # [batch_size, N]
            # w = e.pf.particles.weights().view(1, -1)
            # eta_mix = (w * mu_eta).sum(dim=1)  # [batch_size]
            # resid = Y_batch - model_cfg.rho * eta_mix
            # e.delta_state.append_batch(X_batch, resid)
            X_hist = e.X_hist
            Y_hist = e.y_hist

            # 2. 用 PF posterior 更新所有 residual
            mu_eta_all, _ = emulator.predict(X_hist, e.pf.particles.theta)  # shape [M, N]
            weights = e.pf.particles.weights().view(1, -1)  # [1, N]
            eta_mix_all = (weights * mu_eta_all).sum(dim=1)  # [M]
            resid_all = Y_hist - model_cfg.rho * eta_mix_all

            # 3. 完全重写 delta_state
            e.delta_state = OnlineGPState(
                X=X_hist,
                y=resid_all,
                kernel=make_kernel(model_cfg.delta_kernel),
                noise=model_cfg.delta_kernel.noise,
                update_mode="exact_full",
                hyperparam_mode="fit",
            )
            # mu_eta_all, _ = emulator.predict(X_batch, e.pf.particles.theta)  # shape [M, N]
            # weights = e.pf.particles.weights().view(1, -1)  # [1, N]
            # eta_mix_all = (weights * mu_eta_all).sum(dim=1)  # [M]
            # resid_all = Y_batch - model_cfg.rho * eta_mix_all

            # # 3. 完全重写 delta_state
            # e.delta_state = OnlineGPState(
            #     X=X_batch,
            #     y=resid_all,
            #     kernel=make_kernel(model_cfg.delta_kernel),
            #     noise=model_cfg.delta_kernel.noise,
            #     update_mode="exact_full",
            #     hyperparam_mode="fit",
            # )
            
            e.delta_state.refit_hyperparams()

        # 7) 其他逻辑保持不变...
        self.t += batch_size
        did_refit = False
        if self.delta_fitter is not None and self.delta_refit_every > 0 and (self.t % self.delta_refit_every == 0):
            topk = min(self.delta_refit_topk, len(self.experts))
            topk_indices = []
            for i in range(min(topk, len(self.experts))):
                e = self.experts[i]
                # 只重拟合有足够数据的expert
                if e.X_hist.shape[0] >= 10:  
                    topk_indices.append(i)
            
            if topk_indices:
                _ = self.delta_fitter.refit_topk(
                    experts=self.experts,
                    emulator=emulator,
                    rho=model_cfg.rho,
                    sigma_eps=model_cfg.sigma_eps,
                    topk_indices=topk_indices,
                    init_from_current=True,
                )
                did_refit = True
        
        # 返回结果
        r0_expert = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = float(math.exp(r0_expert.log_mass)) if r0_expert else 0.0

        return {
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": diags,
            "log_umps": log_umps,
            "log_Z": float(log_Z),
        }

    def ump_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, emulator: Emulator, model_cfg: ModelConfig) -> List[float]:
        """批量计算per-expert predictive mixture likelihood"""
        umps: List[float] = []
        for e in self.experts:
            ps = e.pf.particles
            info = loglik_and_grads(Y_batch, X_batch, ps, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, need_grads=False)
            loglik = info["loglik"]  # [N] - 已经是sum over batch
            loglik = torch.clamp(loglik, min=-1e10, max=1e10)
            logmix = torch.logsumexp(ps.logw + loglik, dim=0)
            umps.append(float(logmix))
        return umps

    def _append_hist_batch(self, e: Expert, X_batch: torch.Tensor, Y_batch: torch.Tensor, max_len: int) -> None:
        """批量添加历史数据"""
        if e.X_hist.numel() == 0:
            e.X_hist = X_batch.clone()
            e.y_hist = Y_batch.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, X_batch], dim=0)
            e.y_hist = torch.cat([e.y_hist, Y_batch], dim=0)
        
        # 截断历史
        if e.X_hist.shape[0] > max_len:
            e.X_hist = e.X_hist[-max_len:, :]
            e.y_hist = e.y_hist[-max_len:]
            
            # 同步delta_state
            if e.delta_state.X.shape[0] > max_len:
                e.delta_state.X = e.delta_state.X[-max_len:, :]
                e.delta_state.y = e.delta_state.y[-max_len:]
                e.delta_state._recompute_cache_full()