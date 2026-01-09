# =============================================================
# file: calib/restart_bocpd_mbr.py
# R-BOCPD with Memory-Based Retrieval (MBR)
# =============================================================
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math
import copy
import torch

from .configs import BOCPDConfig, ModelConfig, PFConfig
from .pf import ParticleFilter
from .particles import ParticleSet
from .delta_gp import OnlineGPState
from .likelihood import loglik_and_grads
from .kernels import make_kernel
from .emulator import Emulator
from .expert_delta import ExpertDeltaFitter


# -------------------------------------------------------------
# Dataclasses
# -------------------------------------------------------------
@dataclass
class Expert:
    run_length: int
    pf: ParticleFilter
    delta_state: OnlineGPState
    log_mass: float          # log posterior mass
    X_hist: torch.Tensor     # [M, dx]
    y_hist: torch.Tensor     # [M]


@dataclass
class FrozenExpert:
    """Snapshot of an expert stored in the memory pool."""
    pf: ParticleFilter
    delta_state: OnlineGPState
    theta_mean: torch.Tensor
    usage_count: int = 1
    last_seen_t: int = 0


# -------------------------------------------------------------
# Main BOCPD + MBR
# -------------------------------------------------------------
class BOCPD:
    """
    Restart-BOCPD with Memory-Based Retrieval (MBR).

    - 与 OnlineBayesCalibrator 接口兼容：
        RestartBOCPD(
            config=config.bocpd,
            device=config.model.device,
            dtype=config.model.dtype,
            delta_fitter=None,
            on_restart=on_restart,
            notify_on_restart=notify_on_restart,
        )

    - 支持 R-BOCPD:
        * use_backdated_restart = False → Algorithm-2 (r ← t+1, full reset)
        * use_backdated_restart = True  → Backdated restart (保留 s* expert)

    - 新增 Memory Pool + CRP Prior:
        * 当判定 restart 时，冻结当前 mass 最大的 expert 到 memory_pool
        * 在 Algorithm-2 restart 时，使用 Chinese Restaurant Process prior
          + 当前 batch 的 predictive log-likelihood，从 pool 或 prior 中选择
          最优的 initial expert 作为新的 r=0 expert。
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
        self.config: BOCPDConfig = config
        self.device = device
        self.dtype = dtype

        self.experts: List[Expert] = []
        self.t: int = 0  # processed points (scalar time index)
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

        # -------- Memory-based retrieval (MBR) ----------
        self.mbr_enable: bool = getattr(config, "mbr_enable", True)
        self.mbr_crp_alpha: float = getattr(config, "mbr_crp_alpha", 1.0)
        self.mbr_max_pool: int = getattr(config, "mbr_max_pool", 20)
        self.memory_pool: List[FrozenExpert] = []

        # -------- Snapshot for clean freeze --------
        self.snapshots: Dict[int, FrozenExpert] = {}


    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _init_empty_delta_state(self, dx: int, model_cfg: ModelConfig) -> OnlineGPState:
        """Create an empty OnlineGPState using project kernel config."""
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
        """Spawn a fresh r=0 expert from the prior."""
        delta_state = self._init_empty_delta_state(dx, model_cfg)
        pf = ParticleFilter.from_prior(
            prior_sampler,
            pf_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        return Expert(
            run_length=0,
            pf=pf,
            delta_state=delta_state,
            log_mass=log_mass,
            X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
        )

    def _clone_pf(self, pf: ParticleFilter) -> ParticleFilter:
        """Clone PF (theta, logw, config)."""
        theta = pf.particles.theta.clone()
        logw = pf.particles.logw.clone()
        ps = ParticleSet(theta=theta, logw=logw)
        return ParticleFilter(ps, pf.config, device=self.device, dtype=self.dtype)

    def _append_hist(self, e: Expert, x_t: torch.Tensor, y_t: torch.Tensor, max_len: int) -> None:
        x_t = x_t.view(1, -1)
        y_t = y_t.view(1)
        if e.X_hist.numel() == 0:
            e.X_hist = x_t.clone()
            e.y_hist = y_t.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, x_t], dim=0)
            e.y_hist = torch.cat([e.y_hist, y_t], dim=0)
        if e.X_hist.shape[0] > max_len:
            e.X_hist = e.X_hist[-max_len:, :]
            e.y_hist = e.y_hist[-max_len:]

    def _append_hist_batch(
        self,
        e: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        max_len: int,
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

    # -------------------------------------------------
    # Hazard helper
    # -------------------------------------------------
    def _hazard(self, rl: int) -> float:
        """
        Use either constant hazard_rate (if present),
        or the `hazard` callable in BOCPDConfig.
        """
        if hasattr(self.config, "hazard_rate"):
            h = float(getattr(self.config, "hazard_rate"))
            return h
        r_tensor = torch.tensor([rl], dtype=self.dtype, device=self.device)
        val = self.config.hazard(r_tensor)[0].item()
        val = max(min(val, 1.0 - 1e-12), 1e-12)
        return float(val)

    # -------------------------------------------------
    # UMP (predictive log-likelihood under PF mixture)
    # -------------------------------------------------
    def ump(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> List[float]:
        out: List[float] = []
        for e in self.experts:
            ps: ParticleSet = e.pf.particles
            info = loglik_and_grads(
                y_t,
                x_t,
                ps,
                emulator,
                e.delta_state,
                model_cfg.rho,
                model_cfg.sigma_eps,
                need_grads=False,
            )
            loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)  # [N]
            logmix = torch.logsumexp(ps.logw + loglik, dim=0)
            out.append(float(logmix))
        return out

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
            info = loglik_and_grads(
                Y_batch,
                X_batch,
                ps,
                emulator,
                e.delta_state,
                model_cfg.rho,
                model_cfg.sigma_eps,
                need_grads=False,
            )
            loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)   # [B, N]
            logmix_per_t = torch.logsumexp(ps.logw.view(1, -1) + loglik, dim=1)  # [B]
            out.append(float(logmix_per_t.mean()))
        return out

    # -------------------------------------------------
    # Candidate predictive log-likelihood (for MBR)
    # -------------------------------------------------
    def _candidate_loglik_single(
        self,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        pf: ParticleFilter,
        delta_state: OnlineGPState,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> float:
        ps: ParticleSet = pf.particles
        info = loglik_and_grads(
            y_t,
            x_t,
            ps,
            emulator,
            delta_state,
            model_cfg.rho,
            model_cfg.sigma_eps,
            need_grads=False,
        )
        loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)  # [N]
        logmix = torch.logsumexp(ps.logw + loglik, dim=0)
        return float(logmix)

    def _candidate_loglik_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        pf: ParticleFilter,
        delta_state: OnlineGPState,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> float:
        ps: ParticleSet = pf.particles
        info = loglik_and_grads(
            Y_batch,
            X_batch,
            ps,
            emulator,
            delta_state,
            model_cfg.rho,
            model_cfg.sigma_eps,
            need_grads=False,
        )
        loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)   # [B, N]
        logmix_per_t = torch.logsumexp(ps.logw.view(1, -1) + loglik, dim=1)  # [B]
        return float(logmix_per_t.mean())

    # -------------------------------------------------
    # Prune: keep anchor + top-mass experts
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Memory operations (freeze + CRP-based init)
    # -------------------------------------------------
    def _freeze_old_regime_expert(self, s_star: int):
        """Freeze the clean old-regime expert using s_star to look up snapshots."""
        if not self.mbr_enable:
            return

        if len(self.snapshots) == 0:
            return

        # find the latest snapshot time <= s_star
        keys = sorted(self.snapshots.keys())
        t_snap = None
        for k in keys:
            if k <= s_star:
                t_snap = k
            else:
                break

        if t_snap is None:
            # fallback: use earliest snapshot
            t_snap = keys[0]

        entry = self.snapshots[t_snap]
        clone_entry = FrozenExpert(
            pf=self._clone_pf(entry.pf),
            delta_state=copy.deepcopy(entry.delta_state),
            theta_mean=entry.theta_mean.clone(),
            usage_count=1,
            last_seen_t=t_snap,
        )
        # clone_entry.start_t = t_snap - (entry.delta_state.X.shape[0])
        # clone_entry.end_t = t_snap


        self.memory_pool.append(clone_entry)

        # keep pool size under limit
        if len(self.memory_pool) > self.mbr_max_pool:
            oldest_idx = min(range(len(self.memory_pool)), key=lambda i: self.memory_pool[i].last_seen_t)
            del self.memory_pool[oldest_idx]

        # print interval
        rl = entry.pf.particles.theta.shape[0]   # approximate run_length? (we lack rl)
        print(f"[MBR FREEZE] Frozen old-regime expert from snapshot @ t={t_snap}")

    def _freeze_best_expert(self, t_now: int) -> None:
        """Freeze current best expert into the memory pool."""
        if (not self.mbr_enable) or (not self.experts):
            return

        # best by log_mass
        best_idx = max(range(len(self.experts)), key=lambda i: self.experts[i].log_mass)
        best_e = self.experts[best_idx]

        theta_mean = self._expert_theta_mean(best_e).detach().clone()
        pf_clone = self._clone_pf(best_e.pf)
        delta_clone = copy.deepcopy(best_e.delta_state)

        entry = FrozenExpert(
            pf=pf_clone,
            delta_state=delta_clone,
            theta_mean=theta_mean,
            usage_count=1,
            last_seen_t=t_now,
        )
        self.memory_pool.append(entry)

        # pool size control
        if len(self.memory_pool) > self.mbr_max_pool:
            # remove the oldest (by last_seen_t)
            oldest_idx = min(range(len(self.memory_pool)), key=lambda i: self.memory_pool[i].last_seen_t)
            del self.memory_pool[oldest_idx]

    # def _spawn_new_expert_mbr_single(
    #     self,
    #     x_t: torch.Tensor,
    #     y_t: torch.Tensor,
    #     model_cfg: ModelConfig,
    #     pf_cfg: PFConfig,
    #     prior_sampler: Callable[[int], torch.Tensor],
    #     dx: int,
    #     emulator: Emulator,
    # ) -> Expert:
    #     """Use CRP prior + current (x_t,y_t) to choose init expert."""
    #     if (not self.mbr_enable) or (len(self.memory_pool) == 0):
    #         # fall back to pure prior
    #         return self._spawn_new_expert(model_cfg, pf_cfg, prior_sampler, dx, log_mass=0.0)

    #     alpha = float(self.mbr_crp_alpha)
    #     total_count = sum(e.usage_count for e in self.memory_pool)
    #     denom = total_count + alpha

    #     best_log_score: Optional[float] = None
    #     best_entry_idx: Optional[int] = None
    #     best_pf = None
    #     best_delta = None

    #     # existing experts
    #     for idx, entry in enumerate(self.memory_pool):
    #         loglik = self._candidate_loglik_single(x_t, y_t, entry.pf, entry.delta_state, emulator, model_cfg)
    #         log_prior = math.log(entry.usage_count) - math.log(denom)
    #         log_score = log_prior + loglik

    #         # 打印 expert 的 training interval
    #         if entry.delta_state.X.shape[0] > 0:
    #             t_start = entry.last_seen_t - entry.delta_state.X.shape[0]
    #             t_end = entry.last_seen_t
    #         else:
    #             t_start, t_end = None, None

    #         print(
    #             f"    Pool expert #{idx}: interval=[{t_start}:{t_end}], "
    #             f"loglik={loglik:.4f}, log_prior={log_prior:.4f}, "
    #             f"log_score={log_score:.4f}, usage={entry.usage_count}"
    #         )

    #         if best_log_score is None or log_score > best_log_score:
    #             best_log_score = log_score
    #             best_entry_idx = idx
    #             best_pf = entry.pf
    #             best_delta = entry.delta_state

    #     # prior candidate
    #     pf_prior = ParticleFilter.from_prior(prior_sampler, pf_cfg, device=self.device, dtype=self.dtype)
    #     delta_prior = self._init_empty_delta_state(dx, model_cfg)
    #     loglik_prior = self._candidate_loglik_single(x_t, y_t, pf_prior, delta_prior, emulator, model_cfg)
    #     log_prior_prior = math.log(alpha) - math.log(denom)
    #     log_score_prior = log_prior_prior + loglik_prior

    #     print(
    #         f"    PRIOR: loglik={loglik_prior:.4f}, "
    #         f"log_prior={log_prior_prior:.4f}, log_score={log_score_prior:.4f}"
    #     )

    #     if best_log_score is None or log_score_prior > best_log_score:
    #         print("→ Winner: PRIOR\n")
    #         # prior wins
    #         chosen_pf = pf_prior
    #         chosen_delta = delta_prior
    #     else:
    #         # pool expert wins
    #         print(f"→ Winner: pool expert #{best_entry_idx}\n")
    #         chosen_pf = self._clone_pf(best_pf)
    #         chosen_delta = copy.deepcopy(best_delta)
    #         if best_entry_idx is not None:
    #             self.memory_pool[best_entry_idx].usage_count += 1
    #             self.memory_pool[best_entry_idx].last_seen_t = self.t

    #     return Expert(
    #         run_length=0,
    #         pf=chosen_pf,
    #         delta_state=chosen_delta,
    #         log_mass=0.0,
    #         X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
    #         y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
    #     )

    def _spawn_new_expert_mbr_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        model_cfg: ModelConfig,
        pf_cfg: PFConfig,
        prior_sampler: Callable[[int], torch.Tensor],
        dx: int,
        emulator: Emulator,
    ) -> Expert:
        """Use CRP prior + current batch to choose init expert."""
        print(f"\n[MBR RETRIEVAL - BATCH] t={self.t}")
        print(f"  Memory pool size = {len(self.memory_pool)}")

        if (not self.mbr_enable) or (len(self.memory_pool) == 0):
            return self._spawn_new_expert(model_cfg, pf_cfg, prior_sampler, dx, log_mass=0.0)

        alpha = float(self.mbr_crp_alpha)
        total_count = sum(e.usage_count for e in self.memory_pool)
        denom = total_count + alpha

        best_log_score: Optional[float] = None
        best_entry_idx: Optional[int] = None
        best_pf = None
        best_delta = None

        # existing experts
        for idx, entry in enumerate(self.memory_pool):
            loglik = self._candidate_loglik_batch(X_batch, Y_batch, entry.pf, entry.delta_state, emulator, model_cfg)
            log_prior = math.log(entry.usage_count) - math.log(denom)
            log_score = log_prior + loglik

            if entry.delta_state.X.shape[0] > 0:
                t_start = entry.last_seen_t - entry.delta_state.X.shape[0]
                t_end = entry.last_seen_t
                # interval from snapshot times
                # t_start = entry.last_seen_t - self.config.max_run_length   # or saved start
                # t_end = entry.last_seen_t

            else:
                t_start, t_end = None, None

            print(
                f"    Pool expert #{idx}: interval=[{t_start}:{t_end}], "
                f"loglik={loglik:.4f}, log_prior={log_prior:.4f}, "
                f"log_score={log_score:.4f}, usage={entry.usage_count}"
            )

            if best_log_score is None or log_score > best_log_score:
                best_log_score = log_score
                best_entry_idx = idx
                best_pf = entry.pf
                best_delta = entry.delta_state

        # prior candidate
        pf_prior = ParticleFilter.from_prior(prior_sampler, pf_cfg, device=self.device, dtype=self.dtype)
        delta_prior = self._init_empty_delta_state(dx, model_cfg)
        loglik_prior = self._candidate_loglik_batch(X_batch, Y_batch, pf_prior, delta_prior, emulator, model_cfg)
        log_prior_prior = math.log(alpha) - math.log(denom)
        log_score_prior = log_prior_prior + loglik_prior

        print(
            f"    PRIOR: loglik={loglik_prior:.4f}, "
            f"log_prior={log_prior_prior:.4f}, log_score={log_score_prior:.4f}"
        )

        if best_log_score is None or log_score_prior > best_log_score:
            print("→ Winner: PRIOR\n")
            chosen_pf = pf_prior
            chosen_delta = delta_prior
        else:
            print(f"→ Winner: pool expert #{best_entry_idx}\n")
            chosen_pf = self._clone_pf(best_pf)
            chosen_delta = copy.deepcopy(best_delta)
            if best_entry_idx is not None:
                self.memory_pool[best_entry_idx].usage_count += 1
                self.memory_pool[best_entry_idx].last_seen_t = self.t

        return Expert(
            run_length=0,
            pf=chosen_pf,
            delta_state=chosen_delta,
            log_mass=0.0,
            X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
        )

    # -------------------------------------------------
    # Main update (single point)
    # -------------------------------------------------
    # def update(
    #     self,
    #     x_t: torch.Tensor,
    #     y_t: torch.Tensor,
    #     emulator: Emulator,
    #     model_cfg: ModelConfig,
    #     pf_cfg: PFConfig,
    #     prior_sampler: Callable[[int], torch.Tensor],
    #     verbose: bool = False,
    # ) -> Dict[str, Any]:

    #     x_t = x_t.to(self.device, self.dtype)
    #     y_t = y_t.to(self.device, self.dtype)
    #     dx = x_t.numel()

    #     # init first expert if needed
    #     if len(self.experts) == 0:
    #         e0 = self._spawn_new_expert(
    #             model_cfg=model_cfg,
    #             pf_cfg=pf_cfg,
    #             prior_sampler=prior_sampler,
    #             dx=dx,
    #             log_mass=0.0,
    #         )
    #         self.experts.append(e0)
    #         self.restart_start_time = 0
    #         self.t = 0
    #         self._last_restart_t = -10**9

    #     # 1) per-expert UMP
    #     log_umps = self.ump(x_t, y_t, emulator, model_cfg)
    #     log_umps_t = torch.tensor(log_umps, device=self.device, dtype=self.dtype)

    #     # 2) Update masses (growth + CP)
    #     prev_log_mass = torch.tensor(
    #         [e.log_mass for e in self.experts],
    #         device=self.device,
    #         dtype=self.dtype,
    #     )

    #     hazards = torch.tensor(
    #         [self._hazard(e.run_length) for e in self.experts],
    #         device=self.device,
    #         dtype=self.dtype,
    #     )
    #     log_h = torch.log(hazards.clamp_min(1e-12))
    #     log_1mh = torch.log((1.0 - hazards).clamp_min(1e-12))

    #     growth_log_mass = prev_log_mass + log_1mh + log_umps_t
    #     cp_log_mass = torch.logsumexp(prev_log_mass + log_h + log_umps_t, dim=0)

    #     for i, e in enumerate(self.experts):
    #         e.run_length = e.run_length + 1
    #         e.log_mass = float(growth_log_mass[i])

    #     new_e = self._spawn_new_expert(
    #         model_cfg=model_cfg,
    #         pf_cfg=pf_cfg,
    #         prior_sampler=prior_sampler,
    #         dx=dx,
    #         log_mass=float(cp_log_mass),
    #     )
    #     self.experts.append(new_e)

    #     # 4) normalize log_mass
    #     masses = torch.tensor(
    #         [e.log_mass for e in self.experts],
    #         device=self.device,
    #         dtype=self.dtype,
    #     )
    #     log_Z = torch.logsumexp(masses, dim=0)
    #     masses = masses - log_Z
    #     for i, e in enumerate(self.experts):
    #         e.log_mass = float(masses[i])

    #     # 5) restart decision
    #     t_now = self.t + 1
    #     r_old = self.restart_start_time
    #     anchor_rl = max(t_now - r_old, 0)

    #     anchor_e = next((e for e in self.experts if e.run_length == anchor_rl), None)
    #     p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

    #     best_other_mass = 0.0
    #     s_star: Optional[int] = None
    #     for e in self.experts:
    #         rl = e.run_length
    #         s = t_now - rl
    #         if s > r_old:
    #             m = math.exp(e.log_mass)
    #             if m > best_other_mass:
    #                 best_other_mass = m
    #                 s_star = int(s)

    #     r0_e = next((e for e in self.experts if e.run_length == 0), None)
    #     p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0

    #     did_restart = False
    #     msg_mode = "NO_RESTART"

    #     if (
    #         getattr(self.config, "use_restart", True)
    #         and s_star is not None
    #         and best_other_mass > p_anchor * (1.0 + self.restart_margin)
    #         and (t_now - self._last_restart_t) >= self.restart_cooldown
    #     ):
    #         did_restart = True
    #         self._last_restart_t = t_now

    #         # 冻结当前 best expert 到 memory pool
    #         # self._freeze_best_expert(t_now)
    #         # use snapshot to freeze clean expert
    #         self._freeze_old_regime_expert(s_star)


    #         if getattr(self.config, "use_backdated_restart", False):
    #             # Backdated restart: r ← s*, keep that expert
    #             self.restart_start_time = int(s_star)
    #             new_anchor_rl = max(t_now - self.restart_start_time, 0)
    #             keep_e = next(
    #                 (e for e in self.experts if e.run_length == new_anchor_rl),
    #                 None,
    #             )
    #             if keep_e is None and len(self.experts) > 0:
    #                 keep_e = min(
    #                     self.experts,
    #                     key=lambda e: abs(e.run_length - new_anchor_rl),
    #                 )
    #                 keep_e.run_length = new_anchor_rl
    #             self.experts = [keep_e] if keep_e is not None else self.experts[:1]
    #             msg_mode = "BACKDATED r←s*"
    #         else:
    #             # Algorithm-2: r ← t+1, reset to fresh r=0 expert (with MBR)
    #             self.restart_start_time = t_now
    #             r0 = self._spawn_new_expert_mbr_single(
    #                 x_t,
    #                 y_t,
    #                 model_cfg=model_cfg,
    #                 pf_cfg=pf_cfg,
    #                 prior_sampler=prior_sampler,
    #                 dx=dx,
    #                 emulator=emulator,
    #             )
    #             self.experts = [r0]
    #             msg_mode = "ALGO2 r←t+1 (MBR)"

    #         if self.on_restart is not None and self.notify_on_restart:
    #             self.on_restart(
    #                 int(t_now),
    #                 int(self.restart_start_time),
    #                 int(s_star) if s_star is not None else None,
    #                 int(anchor_rl),
    #                 float(p_anchor),
    #                 float(best_other_mass),
    #             )

    #     # if no restart → prune
    #     if not did_restart:
    #         anchor_run_length = max(t_now - self.restart_start_time, 0)
    #         self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

    #     # 6) history + PF + delta updates
    #     for e in self.experts:
    #         self._append_hist(e, x_t, y_t, self.config.max_run_length)

    #     pf_diags = []
    #     for e in self.experts:
    #         diag = e.pf.step(
    #             x_t,
    #             y_t,
    #             emulator,
    #             e.delta_state,
    #             model_cfg.rho,
    #             model_cfg.sigma_eps,
    #             grad_info=False,
    #         )
    #         pf_diags.append(diag)

    #         # update delta GP with residual
    #         # mu_eta, _ = emulator.predict(x_t.view(1, -1), e.pf.particles.theta)
    #         # w = e.pf.particles.weights().view(1, -1)
    #         # eta_mix = (w * mu_eta).sum(dim=1)
    #         # resid = y_t - model_cfg.rho * eta_mix
    #         # e.delta_state.append(x_t.view(1, -1), resid.view(1))
    #         # 1. 全部历史取出
    #         X_hist = e.X_hist
    #         Y_hist = e.y_hist

    #         # 2. 用 PF posterior 更新所有 residual
    #         mu_eta_all, _ = emulator.predict(X_hist, e.pf.particles.theta)  # shape [M, N]
    #         weights = e.pf.particles.weights().view(1, -1)  # [1, N]
    #         eta_mix_all = (weights * mu_eta_all).sum(dim=1)  # [M]
    #         resid_all = Y_hist - model_cfg.rho * eta_mix_all

    #         # 3. 完全重写 delta_state
    #         e.delta_state = OnlineGPState(
    #             X=X_hist,
    #             y=resid_all,
    #             kernel=make_kernel(model_cfg.delta_kernel),
    #             noise=model_cfg.delta_kernel.noise,
    #             update_mode="exact_full",
    #             hyperparam_mode="fit",
    #         )

    #     # 7) delta refit (optional)
    #     self.t += 1
    #     did_refit = False
    #     if (
    #         self.delta_fitter is not None
    #         and self.delta_refit_every > 0
    #         and (self.t % self.delta_refit_every == 0)
    #     ):
    #         topk = min(self.delta_refit_topk, len(self.experts))
    #         topk_indices: List[int] = []
    #         for i in range(topk):
    #             if self.experts[i].X_hist.shape[0] >= 10:
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

    #     masses_np = [math.exp(e.log_mass) for e in self.experts]
    #     entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

    #     if log_umps:
    #         self.prev_max_ump = max(log_umps)

    #     # debug info
    #     experts_debug: List[Dict[str, Any]] = []
    #     for idx, e in enumerate(self.experts):
    #         try:
    #             theta_mean = self._expert_theta_mean(e)
    #             theta_mean_list = theta_mean.detach().cpu().tolist()
    #         except Exception:
    #             theta_mean_list = None
    #         mass = math.exp(e.log_mass)
    #         log_ump_val = float(log_umps[idx]) if idx < len(log_umps) else None
    #         experts_debug.append(
    #             {
    #                 "index": idx,
    #                 "run_length": int(e.run_length),
    #                 "start_time": int(t_now - e.run_length),
    #                 "log_mass": float(e.log_mass),
    #                 "mass": float(mass),
    #                 "theta_mean": theta_mean_list,
    #                 "log_ump": log_ump_val,
    #             }
    #         )

    #     if did_restart and verbose:
    #         print(
    #             f"[R-BOCPD+MBR] Restart at t={t_now}: "
    #             f"mode={msg_mode}, r_old={r_old}, r_new={self.restart_start_time}, "
    #             f"s_star={s_star}, p_anchor={p_anchor:.4g}, p_cp={p_cp:.4g}"
    #         )
    #         for info in experts_debug:
    #             print(
    #                 f"  expert#{info['index']}: "
    #                 f"rl={info['run_length']}, start={info['start_time']}, "
    #                 f"mass={info['mass']:.4g}, log_ump={info['log_ump']}, "
    #                 f"theta_mean={info['theta_mean']}"
    #             )

    #     return {
    #         "p_anchor": p_anchor,
    #         "p_cp": p_cp,
    #         "num_experts": len(self.experts),
    #         "experts_log_mass": [float(e.log_mass) for e in self.experts],
    #         "pf_diags": pf_diags,
    #         "did_delta_refit": did_refit,
    #         "did_restart": did_restart,
    #         "restart_start_time": int(self.restart_start_time),
    #         "s_star": int(s_star) if s_star is not None else None,
    #         "log_umps": [float(v) for v in log_umps],
    #         "log_Z": float(log_Z),
    #         "entropy": float(entropy),
    #         "experts_debug": experts_debug,
    #     }


    # -------------------------------------------------
    # Batch update
    # -------------------------------------------------
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
        batch_size = X_batch.shape[0]
        dx = X_batch.shape[1]

        # init
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
            e.run_length = e.run_length + batch_size
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
            getattr(self.config, "use_restart", True)
            and s_star is not None
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
            # self._last_restart_t = t_now
            self._last_restart_t = t_now

            # 冻结当前 best expert
            # self._freeze_best_expert(t_now)
            # self._freeze_best_expert(t_now)
            self._freeze_old_regime_expert(t_now)

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
                r0 = self._spawn_new_expert_mbr_batch(
                    X_batch,
                    Y_batch,
                    model_cfg=model_cfg,
                    pf_cfg=pf_cfg,
                    prior_sampler=prior_sampler,
                    dx=dx,
                    emulator=emulator,
                )
                self.experts = [r0]
                msg_mode = "ALGO2 r←t+1 (MBR)"

            if self.on_restart is not None and self.notify_on_restart:
                self.on_restart(
                    int(t_now),
                    int(self.restart_start_time),
                    int(s_star) if s_star is not None else None,
                    int(anchor_rl),
                    float(p_anchor),
                    float(best_other_mass),
                )

        # no restart → prune
        if not did_restart:
            anchor_run_length = max(t_now - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

        # 4) history + PF + delta updates
        for e in self.experts:
            self._append_hist_batch(e, X_batch, Y_batch, self.config.max_run_length)

        pf_diags = []
        for e in self.experts:
            diag = e.pf.step_batch(
                X_batch,
                Y_batch,
                emulator,
                e.delta_state,
                model_cfg.rho,
                model_cfg.sigma_eps,
                grad_info=False,
            )
            pf_diags.append(diag)

            # mu_eta, _ = emulator.predict(X_batch, e.pf.particles.theta)  # [B,N]
            # w = e.pf.particles.weights().view(1, -1)
            # eta_mix = (w * mu_eta).sum(dim=1)  # [B]
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
            e.delta_state.refit_hyperparams()
        # print(f"check the delta_state.X.shape: {self.experts[0].delta_state.X.shape}, e.X_hist.shape: {self.experts[0].X_hist.shape}")

        # 5) delta refit
        self.t += batch_size
        # did_refit = False
        # if (
        #     self.delta_fitter is not None
        #     and self.delta_refit_every > 0
        #     and (self.t % self.delta_refit_every == 0)
        # ):
        #     topk = min(self.delta_refit_topk, len(self.experts))
        #     topk_indices: List[int] = []
        #     for i in range(topk):
        #         if self.experts[i].X_hist.shape[0] >= 10:
        #             topk_indices.append(i)
        #     if topk_indices:
        #         _ = self.delta_fitter.refit_topk(
        #             experts=self.experts,
        #             emulator=emulator,
        #             rho=model_cfg.rho,
        #             sigma_eps=model_cfg.sigma_eps,
        #             topk_indices=topk_indices,
        #             init_from_current=True,
        #         )
        #         did_refit = True

        masses_np = [math.exp(e.log_mass) for e in self.experts]
        entropy = -sum(m * math.log(m + 1e-12) for m in masses_np)

        if log_umps:
            self.prev_max_ump = max(log_umps)

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

        if did_restart and verbose:
            print(
                f"[R-BOCPD+MBR][batch] Restart ending at t={t_now}: "
                f"mode={msg_mode}, r_old={r_old}, r_new={self.restart_start_time}, "
                f"s_star={s_star}, p_anchor={p_anchor:.4g}, p_cp={p_cp:.4g}"
            )
            for info in experts_debug:
                print(
                    f"  expert#{info['index']}: "
                    f"rl={info['run_length']}, start={info['start_time']}, "
                    f"mass={info['mass']:.4g}, log_ump={info['log_ump']}, "
                    f"theta_mean={info['theta_mean']}"
                )

        self._snapshot_current_best(self.t)
        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": pf_diags,
            # "did_delta_refit": did_refit,
            "did_restart": did_restart,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
            "experts_debug": experts_debug,
        }

    def _snapshot_current_best(self, t_now: int):
        """Save the old-regime best expert at time t_now."""
        if not self.experts:
            return

        # choose the expert with maximum run_length (oldest regime)
        best_idx = max(range(len(self.experts)), key=lambda i: self.experts[i].run_length)
        best_e = self.experts[best_idx]

        pf_clone = self._clone_pf(best_e.pf)
        delta_clone = copy.deepcopy(best_e.delta_state)
        theta_mean = self._expert_theta_mean(best_e).detach().clone()

        entry = FrozenExpert(
            pf=pf_clone,
            delta_state=delta_clone,
            theta_mean=theta_mean,
            usage_count=1,
            last_seen_t=t_now,
        )

        self.snapshots[t_now] = entry

        # remove extremely old snapshots if memory too big
        if len(self.snapshots) > 200:
            oldest_key = min(self.snapshots.keys())
            del self.snapshots[oldest_key]

