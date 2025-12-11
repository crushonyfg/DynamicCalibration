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
from .likelihood import loglik_and_grads
from .kernels import make_kernel
from .configs import BOCPDConfig, ModelConfig, PFConfig
from .emulator import Emulator
from .expert_delta import ExpertDeltaFitter  # optional fitter for delta GP


@dataclass
class Expert:
    run_length: int
    pf: ParticleFilter
    delta_state: OnlineGPState
    log_mass: float  # log posterior mass for this expert
    # Keep a short raw history to support delta refits / debugging
    X_hist: torch.Tensor  # [M, dx]
    y_hist: torch.Tensor  # [M]


class BOCPD:
    """
    BOCPD with R-BOCPD-style restart rule (two selectable behaviors):

    - Paper/Algorithm-2 mode (default): when Restart triggers, set r <- t+1
      and reset to a fresh r=0 expert (ϑ_{r,r,r}=1, η_{r,r,r}=1).

    - Backdated mode (use_backdated_restart=True):
      set r <- s*  (the best start time in (r..t]) and keep only the
      new anchor expert with run_length = t - r.

    Mapping: an Expert stores run_length rl; start_time s = t - rl.
    Anchor expert corresponds to s = r  <=>  rl = t - r.
    """

    def __init__(
        self,
        config: BOCPDConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        delta_fitter: Optional[ExpertDeltaFitter] = None,
        on_restart=None,                # optional callback
        notify_on_restart: bool = True, # print message on restart
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.experts: List[Expert] = []
        self.t: int = 0  # current time index (0-based)
        self.delta_fitter = delta_fitter

        # Optional knobs (read from config if present; otherwise defaults)
        self.delta_refit_every: int = int(getattr(config, "delta_refit_every", 0))  # 0 => disabled
        self.delta_refit_topk: int = int(getattr(config, "delta_refit_topk", 1))

        # R-BOCPD state: paper's r (segment start time) and cooldown
        self.restart_start_time: int = 0
        self._last_restart_t: int = -10**9

        # Stable restart controls (tune to reduce false restarts)
        self.restart_margin: float = float(getattr(config, "restart_margin", 0.05))
        self.restart_cooldown: int = int(getattr(config, "restart_cooldown", 10))

        # For diagnostics (e.g., UMP drop)
        self.prev_max_ump: float = 0.0

        # Notification hooks
        self.on_restart = on_restart
        self.notify_on_restart = notify_on_restart

    # ------------------------ utilities ------------------------

    def _haz(self, r: int) -> float:
        """Return hazard h(r) and clamp to (0,1) for numerical stability."""
        t = torch.tensor(float(r), dtype=self.dtype, device=self.device)
        val = self.config.hazard(t)
        h = float(val if isinstance(val, torch.Tensor) else val)
        eps = 1e-12
        if not (eps < h < 1 - eps):
            h = min(max(h, eps), 1 - eps)
        return h

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
        pf_cfg: PFConfig,
        model_cfg: ModelConfig,
        dx: int,
        prior_sampler: Callable[[int], torch.Tensor],
        log_mass: float,
    ) -> Expert:
        delta_new = self._init_empty_delta_state(dx, model_cfg)
        pf_new = ParticleFilter.from_prior(prior_sampler, pf_cfg, model_cfg.device, model_cfg.dtype)
        return Expert(
            run_length=0,
            pf=pf_new,
            delta_state=delta_new,
            log_mass=log_mass,
            X_hist=torch.empty(0, dx, dtype=model_cfg.dtype, device=model_cfg.device),
            y_hist=torch.empty(0, dtype=model_cfg.dtype, device=model_cfg.device),
        )

    def _append_hist(self, e: Expert, x_t: torch.Tensor, y_t: torch.Tensor, max_len: int) -> None:
        """Append current (x_t, y_t) to expert's raw history and truncate to last max_len points.
        Also synchronize delta_state if needed."""
        x_t2 = x_t.view(1, -1)
        y_t2 = y_t.view(1)
        if e.X_hist.numel() == 0:
            e.X_hist = x_t2
            e.y_hist = y_t2
        else:
            e.X_hist = torch.cat([e.X_hist, x_t2], dim=0)
            e.y_hist = torch.cat([e.y_hist, y_t2], dim=0)

        if e.X_hist.shape[0] > max_len:
            e.X_hist = e.X_hist[-max_len:, :]
            e.y_hist = e.y_hist[-max_len:]
            if e.delta_state.X.shape[0] > max_len:
                e.delta_state.X = e.delta_state.X[-max_len:, :]
                e.delta_state.y = e.delta_state.y[-max_len:]
                e.delta_state._recompute_cache_full()

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

    def ump(self, x_t: torch.Tensor, y_t: torch.Tensor, emulator: Emulator, model_cfg: ModelConfig) -> List[float]:
        """Per-expert predictive mixture likelihood (log-space) before PF weight update."""
        umps: List[float] = []
        for e in self.experts:
            ps: ParticleSet = e.pf.particles
            info = loglik_and_grads(
                y_t, x_t, ps, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, need_grads=False
            )
            loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)
            logmix = torch.logsumexp(ps.logw + loglik, dim=0)
            umps.append(float(logmix))
        return umps
    
    def ump_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, emulator: Emulator, model_cfg: ModelConfig) -> List[float]:
        """批量计算per-expert predictive mixture likelihood"""
        umps: List[float] = []
        for e in self.experts:
            ps: ParticleSet = e.pf.particles
            info = loglik_and_grads(
                Y_batch, X_batch, ps, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, need_grads=False
            )
            loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)  # [N] - 已经是sum over batch
            logmix = torch.logsumexp(ps.logw + loglik, dim=0)
            umps.append(float(logmix))
        return umps

    def _prune_keep_anchor(self, anchor_run_length: int, max_k: int) -> None:
        """Prune experts by mass but ALWAYS keep the anchor (rl == anchor_run_length)."""
        anchor_list = [e for e in self.experts if e.run_length == anchor_run_length]
        others = [e for e in self.experts if e.run_length != anchor_run_length]

        # If no exact anchor found (rare), keep the closest rl as surrogate
        if not anchor_list and self.experts:
            target_rl = anchor_run_length
            anchor_candidate = min(self.experts, key=lambda e: abs(e.run_length - target_rl))
            anchor_list = [anchor_candidate]
            others = [e for e in self.experts if e is not anchor_candidate]

        anchor_list.sort(key=lambda e: e.log_mass, reverse=True)
        others.sort(key=lambda e: e.log_mass, reverse=True)

        kept: List[Expert] = [anchor_list[0]]
        quota = max(0, max_k - 1)
        kept.extend(others[:quota])
        self.experts = kept

    # ------------------------ main update ------------------------

    def update(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        One BOCPD update with a new observation (x_t, y_t):
          - update run-length masses (growth / cp)
          - spawn new r=0 expert
          - normalize and prune (keep ANCHOR expert where rl = t - r)
          - PF step & delta update
          - Restart test (Algorithm-2 or Backdated) against the anchor
        """
        x_t = x_t.to(self.device, self.dtype)
        y_t = y_t.to(self.device, self.dtype)
        dx = x_t.numel()

        # Bootstrap first expert
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=0.0)
            self.experts = [e0]
            self.restart_start_time = 0  # r = 0 at beginning

        # 1) per-expert predictive (log)
        log_umps = self.ump(x_t, y_t, emulator, model_cfg)
        log_umps_t = torch.tensor(log_umps, dtype=self.dtype, device=self.device)

        # 2) BOCPD mass update (log-space)
        prev_log_mass = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        hazards = torch.tensor([self._haz(e.run_length) for e in self.experts], dtype=self.dtype, device=self.device)
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        # 3) advance run-lengths and set masses; spawn new r=0 expert
        for i, e in enumerate(self.experts):
            e.run_length = min(e.run_length + 1, self.config.max_run_length)
            e.log_mass = float(growth_log_mass[i])
        new_expert = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=float(cp_log_mass))
        self.experts.append(new_expert)

        # 4) normalize and prune (keep anchor rl = t - r)
        masses = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        anchor_run_length = max(self.t - self.restart_start_time, 0)
        self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        # 5) append raw history and PF + delta updates
        for e in self.experts:
            self._append_hist(e, x_t, y_t, max_len=self.config.max_run_length)

        diags = []
        for e in self.experts:
            diag = e.pf.step(
                x_t, y_t, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, grad_info=False
            )
            diags.append(diag)

            # mixture residual for delta GP
            mu_eta, _ = emulator.predict(x_t.view(1, -1), e.pf.particles.theta)  # [B, N]
            w = e.pf.particles.weights().view(1, -1)
            eta_mix = (w * mu_eta).sum(dim=1).squeeze(0)
            resid = y_t - model_cfg.rho * eta_mix
            e.delta_state.append(x_t, resid)

        # 6) optional delta hyperparam refit
        self.t += 1
        did_refit = False
        if self.delta_fitter is not None and self.delta_refit_every > 0 and (self.t % self.delta_refit_every == 0):
            topk = min(self.delta_refit_topk, len(self.experts))
            topk_indices: List[int] = []
            for i in range(min(topk, len(self.experts))):
                e = self.experts[i]
                if e.X_hist.shape[0] >= 10:  # minimal points for a stable refit
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

        # 7) Restart test vs ANCHOR (anchor rl = t - r_old)
        t_now = self.t
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)

        anchor_e = next((e for e in self.experts if e.run_length == anchor_rl), None)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        # best s in (r_old..t_now]  <=>  rl < anchor_rl
        best_other_mass = 0.0
        s_star = None
        for e in self.experts:
            rl = e.run_length
            s = t_now - rl
            if s > r_old:  # strictly in (r..t]
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = s

        # for plotting convenience: p_cp is mass at s=t (rl=0)
        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

        # decide whether to restart
        did_restart = False
        if (
            getattr(self.config, "use_restart", True)
            and s_star is not None
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        ):
            did_restart = True
            self._last_restart_t = t_now

            if getattr(self.config, "use_backdated_restart", False):
                # ------------------ Backdated mode: r <- s* , keep anchor (rl = t - r_new)
                self.restart_start_time = int(s_star)

                new_anchor_rl = max(t_now - self.restart_start_time, 0)
                keep_e: Optional[Expert] = None
                for e in self.experts:
                    if e.run_length == new_anchor_rl:
                        keep_e = e
                        break
                if keep_e is None and len(self.experts) > 0:
                    # choose the closest as surrogate and align its run_length
                    keep_e = min(self.experts, key=lambda e: abs(e.run_length - new_anchor_rl))
                    keep_e.run_length = new_anchor_rl

                # only keep the anchor expert after restart
                self.experts = [keep_e] if keep_e is not None else self.experts[:1]
                msg_mode = "BACKDATED r←s*"
            else:
                # ------------------ Paper Algorithm-2: r <- t+1 , reset to fresh r=0 expert
                self.restart_start_time = self.t  # since self.t == t+1 (0-based next index)
                # fresh r=0 expert
                r0 = self._spawn_new_expert(
                    pf_cfg, model_cfg, dx=dx, prior_sampler=prior_sampler, log_mass=0.0
                )
                self.experts = [r0]
                msg_mode = "ALGO2 r←t+1 (reset)"

            if self.notify_on_restart or verbose:
                msg = (f"[t={t_now}] RESTART ▶ {msg_mode} | "
                       f"p_anchor={p_anchor:.4f} best_other={best_other_mass:.4f} "
                       f"(s*={s_star})")
                print(msg)

            if callable(self.on_restart):
                # callback signature: (t_now, r_new, s_star, mode, p_anchor, best_other)
                self.on_restart(int(t_now), int(self.restart_start_time),
                                int(s_star) if s_star is not None else None,
                                msg_mode, float(p_anchor), float(best_other_mass))

        # diagnostics (optional entropy)
        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

        # update ump drop diagnostic
        if log_umps:
            self.prev_max_ump = max(log_umps)

        return {
            "p_anchor": p_anchor,                  # mass of no-change since r (ϑ_{r,r,t})
            "p_cp": p_cp,                          # mass of change at t (ϑ_{r,t,t})
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": diags,
            "did_delta_refit": did_refit,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
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
        批量更新R-BOCPD，一次性处理整个batch的数据点
        """
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0]
        dx = X_batch.shape[1]

        # Bootstrap first expert
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=0.0)
            self.experts = [e0]
            self.restart_start_time = 0

        # 1) per-expert predictive (log) - 批量版本
        log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
        log_umps_t = torch.tensor(log_umps, dtype=self.dtype, device=self.device)

        # 2) BOCPD mass update (log-space) - 批量版本
        prev_log_mass = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        hazards = torch.tensor([self._haz(e.run_length) for e in self.experts], dtype=self.dtype, device=self.device)
        log_h = torch.log(hazards.clamp_min(1e-12))
        log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

        growth_log_mass = prev_log_mass + log_1mh + log_umps_t
        cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

        # 3) advance run-lengths and set masses; spawn new r=0 expert
        for i, e in enumerate(self.experts):
            e.run_length = min(e.run_length + batch_size, self.config.max_run_length)
            e.log_mass = float(growth_log_mass[i])
        new_expert = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=float(cp_log_mass))
        self.experts.append(new_expert)

        # 4) normalize (但暂不 prune)
        masses = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
        log_Z = torch.logsumexp(masses, dim=0)
        masses = masses - log_Z
        for i, e in enumerate(self.experts):
            e.log_mass = float(masses[i])

        # 【改动点】：在 prune 之前先执行 restart 判断
        # ------------------------------------------------------------------
        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)

        anchor_e = next((e for e in self.experts if e.run_length == anchor_rl), None)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        best_other_mass = 0.0
        s_star = None
        for e in self.experts:
            rl = e.run_length
            s = t_now - rl
            if s > r_old:  # strictly in (r..t]
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = s

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

        did_restart = False
        msg_mode = ""
        if (
            getattr(self.config, "use_restart", True)
            and s_star is not None
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        ):
            did_restart = True
            self._last_restart_t = t_now

            if getattr(self.config, "use_backdated_restart", False):
                # ------------------ Backdated mode: r <- s* , keep anchor (rl = t - r_new)
                print("R-BOCPD保留最好的expert，而不是完全重启")
                self.restart_start_time = int(s_star)

                new_anchor_rl = max(t_now - self.restart_start_time, 0)
                keep_e: Optional[Expert] = None
                for e in self.experts:
                    if e.run_length == new_anchor_rl:
                        keep_e = e
                        break
                if keep_e is None and len(self.experts) > 0:
                    # choose the closest as surrogate and align its run_length
                    keep_e = min(self.experts, key=lambda e: abs(e.run_length - new_anchor_rl))
                    keep_e.run_length = new_anchor_rl

                self.experts = [keep_e] if keep_e is not None else self.experts[:1]
                msg_mode = "BACKDATED r←s*"

            else:
                # ------------------ Paper Algorithm-2: r <- t+1 , reset to fresh r=0 expert
                print("R-BOCPD完全重新sample了所有expert")
                self.restart_start_time = t_now
                r0 = self._spawn_new_expert(
                    pf_cfg, model_cfg, dx=dx, prior_sampler=prior_sampler, log_mass=0.0
                )
                # 可选：立即用当前 batch 拟合 delta
                if self.delta_fitter is not None:
                    r0.X_hist = X_batch.clone()
                    r0.y_hist = Y_batch.clone()
                    self.delta_fitter.refit_expert(
                        r0,
                        emulator,
                        rho=model_cfg.rho,
                        sigma_eps=model_cfg.sigma_eps,
                        update_delta_state=True,
                    )
                self.experts = [r0]
                msg_mode = "ALGO2 r←t+1 (reset)"

            if self.notify_on_restart or verbose:
                msg = (f"[t={t_now}] RESTART ▶ {msg_mode} | "
                    f"p_anchor={p_anchor:.4f} best_other={best_other_mass:.4f} "
                    f"(s*={s_star})")
                print(msg)

            if callable(self.on_restart):
                self.on_restart(int(t_now), int(self.restart_start_time),
                                int(s_star) if s_star is not None else None,
                                msg_mode, float(p_anchor), float(best_other_mass))
        # ------------------------------------------------------------------

        # 如果刚刚没有restart，再进行 prune（保留 anchor）
        if not did_restart:
            anchor_run_length = max(self.t - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        # 5) append raw history and PF + delta updates (批量版本)
        for e in self.experts:
            self._append_hist_batch(e, X_batch, Y_batch, max_len=self.config.max_run_length)

        diags = []
        for e in self.experts:
            diag = e.pf.step_batch(
                X_batch, Y_batch, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, grad_info=False
            )
            diags.append(diag)

            # mixture residual for delta GP (批量版本)
            mu_eta, _ = emulator.predict(X_batch, e.pf.particles.theta)
            w = e.pf.particles.weights().view(1, -1)
            eta_mix = (w * mu_eta).sum(dim=1)
            resid = Y_batch - model_cfg.rho * eta_mix
            e.delta_state.append_batch(X_batch, resid)

        # 6) optional delta hyperparam refit
        self.t += batch_size
        did_refit = False
        if self.delta_fitter is not None and self.delta_refit_every > 0 and (self.t % self.delta_refit_every == 0):
            print(f"{self.t}时刻进行delta hyperparam refit")
            topk = min(self.delta_refit_topk, len(self.experts))
            topk_indices: List[int] = []
            for i in range(min(topk, len(self.experts))):
                e = self.experts[i]
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

        # diagnostics
        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

        if log_umps:
            self.prev_max_ump = max(log_umps)

        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": diags,
            "did_delta_refit": did_refit,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
        }


    # def update_batch(
    #     self,
    #     X_batch: torch.Tensor,  # [batch_size, dx]
    #     Y_batch: torch.Tensor,  # [batch_size]
    #     emulator: Emulator,
    #     model_cfg: ModelConfig,
    #     pf_cfg: PFConfig,
    #     prior_sampler: Callable[[int], torch.Tensor],
    #     verbose: bool = False,
    # ) -> Dict[str, Any]:
    #     """
    #     批量更新R-BOCPD，一次性处理整个batch的数据点
    #     """
    #     X_batch = X_batch.to(self.device, self.dtype)
    #     Y_batch = Y_batch.to(self.device, self.dtype)
    #     batch_size = X_batch.shape[0]
    #     dx = X_batch.shape[1]

    #     # Bootstrap first expert
    #     if len(self.experts) == 0:
    #         e0 = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=0.0)
    #         self.experts = [e0]
    #         self.restart_start_time = 0

    #     # 1) per-expert predictive (log) - 批量版本
    #     log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
    #     log_umps_t = torch.tensor(log_umps, dtype=self.dtype, device=self.device)

    #     # 2) BOCPD mass update (log-space) - 批量版本
    #     prev_log_mass = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
    #     hazards = torch.tensor([self._haz(e.run_length) for e in self.experts], dtype=self.dtype, device=self.device)
    #     log_h = torch.log(hazards.clamp_min(1e-12))
    #     log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

    #     growth_log_mass = prev_log_mass + log_1mh + log_umps_t
    #     cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

    #     # 3) advance run-lengths and set masses; spawn new r=0 expert
    #     for i, e in enumerate(self.experts):
    #         e.run_length = min(e.run_length + batch_size, self.config.max_run_length)
    #         e.log_mass = float(growth_log_mass[i])
    #     new_expert = self._spawn_new_expert(pf_cfg, model_cfg, dx, prior_sampler, log_mass=float(cp_log_mass))
    #     self.experts.append(new_expert)

    #     # 4) normalize and prune (keep anchor rl = t - r)
    #     masses = torch.tensor([e.log_mass for e in self.experts], dtype=self.dtype, device=self.device)
    #     log_Z = torch.logsumexp(masses, dim=0)
    #     masses = masses - log_Z
    #     for i, e in enumerate(self.experts):
    #         e.log_mass = float(masses[i])

    #     anchor_run_length = max(self.t - self.restart_start_time, 0)
    #     self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

    #     # 5) append raw history and PF + delta updates (批量版本)
    #     for e in self.experts:
    #         self._append_hist_batch(e, X_batch, Y_batch, max_len=self.config.max_run_length)

    #     diags = []
    #     for e in self.experts:
    #         diag = e.pf.step_batch(
    #             X_batch, Y_batch, emulator, e.delta_state, model_cfg.rho, model_cfg.sigma_eps, grad_info=False
    #         )
    #         diags.append(diag)

    #         # mixture residual for delta GP (批量版本)
    #         mu_eta, _ = emulator.predict(X_batch, e.pf.particles.theta)  # [batch_size, N]
    #         w = e.pf.particles.weights().view(1, -1)
    #         eta_mix = (w * mu_eta).sum(dim=1)  # [batch_size]
    #         resid = Y_batch - model_cfg.rho * eta_mix
    #         e.delta_state.append_batch(X_batch, resid)

    #     # 6) optional delta hyperparam refit
    #     self.t += batch_size
    #     did_refit = False
    #     if self.delta_fitter is not None and self.delta_refit_every > 0 and (self.t % self.delta_refit_every == 0):
    #         print(f"{self.t}时刻进行delta hyperparam refit")
    #         topk = min(self.delta_refit_topk, len(self.experts))
    #         topk_indices: List[int] = []
    #         for i in range(min(topk, len(self.experts))):
    #             e = self.experts[i]
    #             if e.X_hist.shape[0] >= 10:
    #                 topk_indices.append(i)
    #         if topk_indices:
    #             _ = self.delta_fitter.refit_topk(
    #                 experts=self.experts,
    #                 emulator=emulator,
    #                 rho=model_cfg.rho,
    #                 sigma_eps=model_cfg.sigma_eps,
    #                 topk_indices=topk_indices,
    #                 init_from_current=True,
    #             )
    #             did_refit = True

    #     # 7) Restart test vs ANCHOR (anchor rl = t - r_old)
    #     t_now = self.t
    #     r_old = self.restart_start_time
    #     anchor_rl = max(t_now - r_old, 0)

    #     anchor_e = next((e for e in self.experts if e.run_length == anchor_rl), None)
    #     p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

    #     # best s in (r_old..t_now]  <=>  rl < anchor_rl
    #     best_other_mass = 0.0
    #     s_star = None
    #     for e in self.experts:
    #         rl = e.run_length
    #         s = t_now - rl
    #         if s > r_old:  # strictly in (r..t]
    #             m = math.exp(e.log_mass)
    #             if m > best_other_mass:
    #                 best_other_mass = m
    #                 s_star = s

    #     # for plotting convenience: p_cp is mass at s=t (rl=0)
    #     r0_e = next((e for e in self.experts if e.run_length == 0), None)
    #     p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

    #     # decide whether to restart
    #     did_restart = False
    #     if (
    #         getattr(self.config, "use_restart", True)
    #         and s_star is not None
    #         and best_other_mass > p_anchor * (1.0 + self.restart_margin)
    #         and (t_now - self._last_restart_t) >= self.restart_cooldown
    #     ):
    #         did_restart = True
    #         self._last_restart_t = t_now

    #         if getattr(self.config, "use_backdated_restart", False):
    #             # ------------------ Backdated mode: r <- s* , keep anchor (rl = t - r_new)
    #             print("R-BOCPD保留最好的expert，而不是完全重启")
    #             self.restart_start_time = int(s_star)

    #             new_anchor_rl = max(t_now - self.restart_start_time, 0)
    #             keep_e: Optional[Expert] = None
    #             for e in self.experts:
    #                 if e.run_length == new_anchor_rl:
    #                     keep_e = e
    #                     break
    #             if keep_e is None and len(self.experts) > 0:
    #                 # choose the closest as surrogate and align its run_length
    #                 keep_e = min(self.experts, key=lambda e: abs(e.run_length - new_anchor_rl))
    #                 keep_e.run_length = new_anchor_rl

    #             # only keep the anchor expert after restart
    #             self.experts = [keep_e] if keep_e is not None else self.experts[:1]
    #             msg_mode = "BACKDATED r←s*"
    #         else:
    #             # ------------------ Paper Algorithm-2: r <- t+1 , reset to fresh r=0 expert
    #             print(f"R-BOCPD完全重新sample了所有expert")
    #             self.restart_start_time = self.t  # since self.t == t+1 (0-based next index)
    #             # fresh r=0 expert
    #             r0 = self._spawn_new_expert(
    #                 pf_cfg, model_cfg, dx=dx, prior_sampler=prior_sampler, log_mass=0.0
    #             )
    #             self.experts = [r0]
    #             msg_mode = "ALGO2 r←t+1 (reset)"

    #         if self.notify_on_restart or verbose:
    #             msg = (f"[t={t_now}] RESTART ▶ {msg_mode} | "
    #                 f"p_anchor={p_anchor:.4f} best_other={best_other_mass:.4f} "
    #                 f"(s*={s_star})")
    #             print(msg)

    #         if callable(self.on_restart):
    #             # callback signature: (t_now, r_new, s_star, mode, p_anchor, best_other)
    #             self.on_restart(int(t_now), int(self.restart_start_time),
    #                             int(s_star) if s_star is not None else None,
    #                             msg_mode, float(p_anchor), float(best_other_mass))

    #     # diagnostics (optional entropy)
    #     masses_np = [math.exp(e.log_mass) for e in self.experts]
    #     entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

    #     # update ump drop diagnostic
    #     if log_umps:
    #         self.prev_max_ump = max(log_umps)

    #     return {
    #         "p_anchor": p_anchor,                  # mass of no-change since r (ϑ_{r,r,t})
    #         "p_cp": p_cp,                          # mass of change at t (ϑ_{r,t,t})
    #         "num_experts": len(self.experts),
    #         "experts_log_mass": [float(e.log_mass) for e in self.experts],
    #         "pf_diags": diags,
    #         "did_delta_refit": did_refit,
    #         "did_restart": did_restart,
    #         "restart_start_time": int(self.restart_start_time),
    #         "s_star": int(s_star) if s_star is not None else None,
    #         "log_umps": [float(v) for v in log_umps],
    #         "log_Z": float(log_Z),
    #         "entropy": float(entropy),
    #     }
