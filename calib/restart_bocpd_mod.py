# =============================================================
# file: calib/restart_bocpd_dubug.py + modification
# R-BOCPD with restart rule + debug exports
# Compatible with OnlineBayesCalibrator.__init__ in online_calibrator.py
# Each particle has a delta GP discrepancy
# =============================================================
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math
import torch

from .configs import BOCPDConfig, ModelConfig, PFConfig
# from .pf import ParticleFilter
from .particles import ParticleSet
from .delta_gp import OnlineGPState
# from .likelihood import loglik_and_grads
from .kernels import make_kernel
from .emulator import Emulator
from .expert_delta import ExpertDeltaFitter

from .pf_mod import loglik_and_grads_mod, ParticleFilter


@dataclass
class Expert:
    run_length: int
    pf: ParticleFilter
    delta_states: List[OnlineGPState]
    log_mass: float          # log posterior mass for this expert
    X_hist: torch.Tensor     # [M, dx]
    y_hist: torch.Tensor     # [M]


class BOCPD:
    """
    Restart-BOCPD implementation used as `RestartBOCPD` in OnlineBayesCalibrator.

    关键点：
    - 支持 Algorithm-2 完全重启 和 backdated restart 两种模式：
        * use_backdated_restart=False → Algorithm-2 (r ← t+1, 全部重新 sample)
        * use_backdated_restart=True  → 从 (r..t] 中选 s* 作为新的 r，保留那个 expert
    - 每次 update / update_batch 返回 `experts_debug`，
      内含每个 expert 的 run_length, start_time, mass, theta_mean, log_ump 等信息。
    """

    def __init__(
        self,
        config: BOCPDConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        delta_fitter: Optional[ExpertDeltaFitter] = None,
        on_restart: Optional[Callable] = None,
        notify_on_restart: bool = True,
        **kwargs,
    ):
        """
        注意：签名要兼容 OnlineBayesCalibrator 里的调用：
            RestartBOCPD(
                config=config.bocpd,
                device=config.model.device,
                dtype=config.model.dtype,
                delta_fitter=None,
                on_restart=on_restart,
                notify_on_restart=notify_on_restart,
            )
        """
        self.config: BOCPDConfig = config
        self.device = device
        self.dtype = dtype

        self.experts: List[Expert] = []
        self.t: int = 0  # processed points count
        self.restart_start_time: int = 0
        self._last_restart_t: int = -10**9
        self.prev_max_ump: float = 0.0

        self.delta_fitter = delta_fitter

        # R-BOCPD hyperparams
        self.restart_margin: float = getattr(config, "restart_margin", 0.05)
        self.restart_cooldown: int = getattr(config, "restart_cooldown", 10)

        self.delta_refit_every: int = getattr(config, "delta_refit_every", 0)
        self.delta_refit_topk: int = getattr(config, "delta_refit_topk", 3)

        self.on_restart = on_restart
        self.notify_on_restart = notify_on_restart

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _init_empty_delta_state(self, dx: int, model_cfg: ModelConfig) -> OnlineGPState:
        kernel = make_kernel(model_cfg.delta_kernel)
        return OnlineGPState(
            X=torch.empty(0, dx, dtype=model_cfg.dtype, device=model_cfg.device),
            y=torch.empty(0, dtype=model_cfg.dtype, device=model_cfg.device),
            kernel=kernel,
            noise=model_cfg.delta_kernel.noise,
            update_mode="exact_full",
            hyperparam_mode="fit",
        )

    def _spawn_new_expert(
        self,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        dx: int,
        log_mass: float,
    ) -> Expert:
        delta_states = [self._init_empty_delta_state(dx, model_cfg) for _ in range(pf_cfg.num_particles)]
        pf = ParticleFilter.from_prior(
            prior_sampler,
            pf_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        return Expert(
            run_length=0,
            pf=pf,
            delta_states=delta_states,
            log_mass=log_mass,
            X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
        )

    def _append_hist_batch(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
    ) -> None:
        if e.X_hist.numel() == 0:
            e.X_hist = X_batch.clone()
            e.y_hist = Y_batch.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, X_batch], dim=0)
            e.y_hist = torch.cat([e.y_hist, Y_batch], dim=0)
        # if e.X_hist.shape[0] > max_len:
        #     e.X_hist = e.X_hist[-max_len:, :]
        #     e.y_hist = e.y_hist[-max_len:]

    def _expert_theta_mean(self, e: Expert) -> torch.Tensor:
        """PF posterior mean of theta for a single expert."""
        ps: ParticleSet = e.pf.particles
        w = ps.weights()       # [N]
        theta = ps.theta       # [N, d]
        return (w.view(-1, 1) * theta).sum(dim=0)  # [d]

    # ------------------------------------------------------------------
    # hazard helper
    # ------------------------------------------------------------------
    def _hazard(self, rl: int) -> float:
        """
        Use either constant hazard_rate (if present), or the `hazard` callable
        in BOCPDConfig (default_hazard).
        """
        if hasattr(self.config, "hazard_rate"):
            h = float(getattr(self.config, "hazard_rate"))
            return h
        # use callable hazard(r)
        r_tensor = torch.tensor([rl], dtype=self.dtype, device=self.device)
        val = self.config.hazard(r_tensor)[0].item()
        # protect against 0 or >1
        val = max(min(val, 1.0 - 1e-12), 1e-12)
        return float(val)

    # ------------------------------------------------------------------
    # per-expert UMP (log predictive under PF mixture)
    # ------------------------------------------------------------------

    def ump_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> List[float]:
        """
        对于 batch，给每个 expert 返回一个“平均 log predictive”。
        """
        out: List[float] = []
        for e in self.experts:
            ps: ParticleSet = e.pf.particles
            info = loglik_and_grads_mod(
                Y_batch,
                X_batch,
                ps,
                emulator,
                e.delta_states,
                model_cfg.rho,
                model_cfg.sigma_eps,
                need_grads=False,
            )
            # info["loglik"]: [batch_size, N]
            loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)
            logmix_per_t = torch.logsumexp(ps.logw.view(1, -1) + loglik, dim=1)  # [batch_size]
            out.append(float(logmix_per_t.mean()))
        return out

    # ------------------------------------------------------------------
    # prune: keep anchor + top mass experts
    # ------------------------------------------------------------------
    def _prune_keep_anchor(self, anchor_run_length: int, max_experts: int) -> None:
        if len(self.experts) <= max_experts:
            return

        anchor_idx: Optional[int] = None
        best_diff = 10**9
        for i, e in enumerate(self.experts):
            diff = abs(e.run_length - anchor_run_length)
            if diff < best_diff:
                best_diff = diff
                anchor_idx = i

        # sort by log_mass desc
        sorted_idx = sorted(
            range(len(self.experts)),
            key=lambda i: self.experts[i].log_mass,
            reverse=True,
        )

        kept: List[int] = []
        for idx in sorted_idx:
            if idx == anchor_idx or len(kept) < max_experts - 1:
                kept.append(idx)
            if len(kept) >= max_experts:
                break
        if anchor_idx is not None and anchor_idx not in kept:
            kept[-1] = anchor_idx

        kept = sorted(set(kept))
        self.experts = [self.experts[i] for i in kept]

    # ------------------------------------------------------------------
    # batch update
    # ------------------------------------------------------------------
    def update_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Any]:

        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0] # [b, dx]
        dx = X_batch.shape[1]

        # 初始化
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(
                model_cfg=model_cfg,
                pf_cfg=pf_cfg,
                prior_sampler=prior_sampler,
                dx=dx,
                log_mass=0.0,
            )
            self.experts.append(e0)
            self.restart_start_time = 0
            self.t = 0
            self._last_restart_t = -10**9

        # 1) per-expert log UMP over batch
        log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
        log_umps_t = torch.tensor(log_umps, device=self.device, dtype=self.dtype)

        # 2) mass update
        prev_log_mass = torch.tensor(
            [e.log_mass for e in self.experts], device=self.device, dtype=self.dtype
        )
        hazards = torch.tensor(
            [self._hazard(e.run_length) for e in self.experts],
            device=self.device,
            dtype=self.dtype,
        )
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        for i, e in enumerate(self.experts):
            e.run_length = min(e.run_length + batch_size, self.config.max_run_length)
            e.log_mass = float(growth_log_mass[i])

        new_e = self._spawn_new_expert(
            model_cfg=model_cfg,
            pf_cfg=pf_cfg,
            prior_sampler=prior_sampler,
            dx=dx,
            log_mass=float(cp_log_mass),
        )
        self.experts.append(new_e)

        masses = torch.tensor(
            [e.log_mass for e in self.experts], device=self.device, dtype=self.dtype
        )
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # 3) restart decision
        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)

        anchor_e = next((e for e in self.experts if e.run_length == anchor_rl), None)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        best_other_mass = 0.0
        s_star: Optional[int] = None
        for e in self.experts:
            rl = e.run_length
            s = t_now - rl
            if s > r_old:
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = int(s)

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

        did_restart = False
        msg_mode = "NO_RESTART"

        if (
            s_star is not None
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        ):
            did_restart = True
            # --------- DEBUG: print top-5 experts BEFORE restart ----------
            print(f"\n[DEBUG BEFORE RESTART] t={t_now}, r_old={r_old}, s_star={s_star}")
            print(f"  p_anchor={p_anchor:.6f}, best_other_mass={best_other_mass:.6f}, p_cp={p_cp:.6f}")
            print("  Top-5 experts BEFORE restart:")
            
            # 取 log_mass 排序
            sorted_experts = sorted(
                self.experts,
                key=lambda e: e.log_mass,
                reverse=True
            )

            for i, e in enumerate(sorted_experts[:5]):
                theta_mean = self._expert_theta_mean(e).detach().cpu().numpy()
                print(f"    Expert[{i}] rl={e.run_length}, mass={math.exp(e.log_mass):.6f}, theta_mean={theta_mean}")
            print("--------------------------------------------------------\n")
            self._last_restart_t = t_now

            if getattr(self.config, "use_backdated_restart", False):
                self.restart_start_time = int(s_star)
                new_anchor_rl = max(t_now - self.restart_start_time, 0)
                keep_e = next(
                    (e for e in self.experts if e.run_length == new_anchor_rl),
                    None,
                )
                if keep_e is None and len(self.experts) > 0:
                    keep_e = min(
                        self.experts,
                        key=lambda e: abs(e.run_length - new_anchor_rl),
                    )
                    keep_e.run_length = new_anchor_rl
                self.experts = [keep_e] if keep_e is not None else self.experts[:1]
                msg_mode = "BACKDATED r←s*"
            else:
                self.restart_start_time = t_now
                r0 = self._spawn_new_expert(
                    model_cfg=model_cfg,
                    pf_cfg=pf_cfg,
                    prior_sampler=prior_sampler,
                    dx=dx,
                    log_mass=0.0,
                )
                self.experts = [r0]
                msg_mode = "ALGO2 r←t+1"

            if self.on_restart is not None and self.notify_on_restart:
                self.on_restart(
                    int(t_now),
                    int(self.restart_start_time),
                    int(s_star) if s_star is not None else None,
                    int(anchor_rl),
                    float(p_anchor),
                    float(best_other_mass),
                )

        # 没有 restart 时进行 prune
        if not did_restart:
            anchor_run_length = max(t_now - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        # 4) 历史 + PF + delta updates
        for e in self.experts:
            self._append_hist_batch(e, X_batch, Y_batch)

        pf_diags = []
        for e in self.experts:
            delta_states, diag = e.pf.step_batch(
                X_batch,
                Y_batch,
                emulator,
                e,
                model_cfg,
                model_cfg.rho,
                model_cfg.sigma_eps,
                prior_sampler,
                grad_info=False,
            )
            pf_diags.append(diag)

            e.delta_states = delta_states

            # X_hist = e.X_hist
            # Y_hist = e.y_hist

            # # 2. 用 PF posterior 更新所有 residual
            # mu_eta_all, _ = emulator.predict(X_hist, e.pf.particles.theta)  # shape [M, N]
            # weights = e.pf.particles.weights().view(1, -1)  # [1, N]
            # eta_mix_all = (weights * mu_eta_all).sum(dim=1)  # [M]
            # resid_all = Y_hist - model_cfg.rho * eta_mix_all

            # # 3. 完全重写 delta_state
            # e.delta_state = OnlineGPState(
            #     X=X_hist,
            #     y=resid_all,
            #     kernel=make_kernel(model_cfg.delta_kernel),
            #     noise=model_cfg.delta_kernel.noise,
            #     update_mode="exact_full",
            #     hyperparam_mode="fit",
            # )

        # 5) delta refit
        self.t += batch_size
        did_refit = False
        if (
            self.delta_fitter is not None
            and self.delta_refit_every > 0
            and (self.t % self.delta_refit_every == 0)
        ):
            topk = min(self.delta_refit_topk, len(self.experts))
            topk_indices: List[int] = []
            for i in range(topk):
                if self.experts[i].X_hist.shape[0] >= 10:
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

        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

        if log_umps:
            self.prev_max_ump = max(log_umps)

        # experts_debug
        experts_debug: List[Dict[str, Any]] = []
        for idx, e in enumerate(self.experts):
            try:
                theta_mean = self._expert_theta_mean(e)
                theta_mean_list = theta_mean.detach().cpu().tolist()
            except Exception:
                theta_mean_list = None
            mass = math.exp(e.log_mass)
            log_ump_val = float(log_umps[idx]) if idx < len(log_umps) else None
            experts_debug.append(
                {
                    "index": idx,
                    "run_length": int(e.run_length),
                    "start_time": int(t_now - e.run_length),
                    "log_mass": float(e.log_mass),
                    "mass": float(mass),
                    "theta_mean": theta_mean_list,
                    "log_ump": log_ump_val,
                }
            )

        if did_restart:

            print(
                f"[R-BOCPD][batch] Restart ending at t={t_now}: "
                f"mode={msg_mode}, r_old={r_old}, r_new={self.restart_start_time}, "
                f"s_star={s_star}, p_anchor={p_anchor:.4g}, p_cp={p_cp:.4g}"
            )
            for info in experts_debug:
                print(
                    f"  expert#{info['index']}: "
                    f"rl={info['run_length']}, start={info['start_time']}, "
                    f"mass={info['mass']:.4g}, log_ump={info['log_ump']}"
                )

        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": pf_diags,
            "did_delta_refit": did_refit,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
            "experts_debug": experts_debug,
        }
