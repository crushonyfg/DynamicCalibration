from typing import Any, Callable, Dict, List, Optional
import math
import torch
import gpytorch

from .configs import ModelConfig, PFConfig
from .emulator import Emulator
from .likelihood import loglik_and_grads
from .particles import ParticleSet
from .delta_gp import GPyTorchDeltaState, fit_gpytorch_delta
from .restart_bocpd_debug_260115_gpytorch import BOCPD as BaseBOCPD, Expert, RollingStats


class BOCPD(BaseBOCPD):
    """
    Hybrid restart BOCPD:
    - no restart
    - delta-only restart (keep PF particles)
    - full restart (same behavior as base class)
    """

    def _single_log_ump(
        self,
        e_theta: Expert,
        e_delta: Expert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        model_cfg: ModelConfig,
    ) -> float:
        ps: ParticleSet = e_theta.pf.particles
        info = loglik_and_grads(
            Y_batch,
            X_batch,
            ps,
            emulator,
            e_delta.delta_state,
            model_cfg.rho,
            model_cfg.sigma_eps,
            need_grads=False,
            use_discrepancy=model_cfg.bocpd_use_discrepancy if hasattr(model_cfg, "bocpd_use_discrepancy") else model_cfg.use_discrepancy,
        )
        loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)
        logmix = torch.logsumexp(ps.logw.view(1, -1) + loglik, dim=1)
        return float(logmix.mean().item())

    def _reset_delta_for_expert(self, e: Expert, model_cfg: ModelConfig) -> None:
        if e.X_hist.numel() == 0:
            e.delta_state = None
            return
        X_hist = e.X_hist
        Y_hist = e.y_hist
        mu_eta_all, _ = self._last_emulator.predict(X_hist, e.pf.particles.theta)
        weights = e.pf.particles.weights().view(1, -1)
        if torch.isnan(weights).any() or weights.sum() < 1e-30:
            e.delta_state = None
            return

        if mu_eta_all.dim() == 2:
            eta_mix_all = (weights * mu_eta_all).sum(dim=1)
        else:
            eta_mix_all = (weights.unsqueeze(-1) * mu_eta_all).sum(dim=1)

        resid_all = Y_hist - model_cfg.rho * eta_mix_all
        resid_for_delta = (
            resid_all.mean(dim=-1)
            if resid_all.dim() > 1 and resid_all.shape[-1] > 1
            else resid_all.reshape(-1)
        )
        if torch.isnan(resid_for_delta).any() or torch.isinf(resid_for_delta).any():
            e.delta_state = None
            return

        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        delta_state = GPyTorchDeltaState(
            X=X_hist,
            y=resid_for_delta,
            kernel=kernel,
            noise=model_cfg.delta_kernel.noise,
        )
        try:
            e.delta_state = fit_gpytorch_delta(delta_state, max_iter=120)
        except Exception:
            e.delta_state = None

    def _do_delta_only_restart(
        self,
        t_now: int,
        keep_e: Optional[Expert],
        dx: int,
    ) -> str:
        """
        Delta-only restart policy:
        - keep only candidate expert's PF particles/weights
        - drop old discrepancy state/history entirely
        - let current batch rebuild history and refit delta afterwards
        """
        self.restart_start_time = int(t_now)
        if keep_e is None:
            return "DELTA_ONLY_FALLBACK"

        keep_e.run_length = 0
        keep_e.log_mass = 0.0
        keep_e.delta_state = None
        keep_e.X_hist = torch.empty(0, dx, dtype=self.dtype, device=self.device)
        keep_e.y_hist = torch.empty(0, dtype=self.dtype, device=self.device)
        self.experts = [keep_e]
        return "DELTA_ONLY"

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
        # keep for helper use
        self._last_emulator = emulator

        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0]
        dx = X_batch.shape[1]

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

        log_umps = self.ump_batch(X_batch, Y_batch, emulator, model_cfg)
        log_umps_t = torch.tensor(log_umps, device=self.device, dtype=self.dtype)
        experts_pre = list(self.experts)
        idx_pre = {id(e): i for i, e in enumerate(experts_pre)}

        def get_pre_log_ump(e):
            j = idx_pre.get(id(e), None)
            return None if j is None else float(log_umps[j])

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
            e.run_length += batch_size
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

        t_now = self.t + batch_size
        r_old = self.restart_start_time
        anchor_rl = max(t_now - r_old, 0)
        anchor_e = self._closest_by_run_length(anchor_rl)
        p_anchor = math.exp(anchor_e.log_mass) if anchor_e is not None else 0.0

        best_other_mass = 0.0
        s_star: Optional[int] = None
        cand_e: Optional[Expert] = None
        for e in self.experts:
            rl = e.run_length
            s = t_now - rl
            if s > r_old:
                m = math.exp(e.log_mass)
                if m > best_other_mass:
                    best_other_mass = m
                    s_star = int(s)
                    cand_e = e

        r0_e = next((e for e in self.experts if e.run_length == 0), None)
        p_cp = math.exp(r0_e.log_mass) if r0_e is not None else 0.0
        did_restart = False
        msg_mode = "NO_RESTART"
        theta_pass = False
        theta_stat = None

        if self.restart_criteria == "theta_test" and anchor_e is not None and cand_e is not None:
            method = getattr(self.config, "restart_theta_test", "energy")
            if method == "credible":
                z = float(getattr(self.config, "restart_cred_z", 2.0))
                frac = float(getattr(self.config, "restart_cred_frac", 0.5))
                rate, theta_pass = self._credible_nonoverlap(anchor_e, cand_e, z, frac)
                theta_stat = rate
            elif method == "sw":
                n_proj = int(getattr(self.config, "restart_sw_proj", 32))
                theta_stat = self._sliced_wasserstein(anchor_e, cand_e, n_proj)
                tau = float(getattr(self.config, "restart_theta_tau", 0.1))
                theta_pass = (theta_stat > tau)
            else:
                theta_stat = self._energy_distance(anchor_e, cand_e)
                tau = float(getattr(self.config, "restart_theta_tau", 0.1))
                theta_pass = (theta_stat > tau)
        elif self.restart_criteria != "theta_test":
            theta_pass = True
            theta_stat = 0.0

        # Hybrid decision: full restart vs delta-only restart
        restart_mode = "full"
        L_aa = None
        L_ac = None
        L_ca = None
        L_cc = None
        gain_delta = None
        gain_theta = None

        mass_gate = (
            getattr(self.config, "use_restart", True)
            and s_star is not None
            and best_other_mass > p_anchor * (1.0 + self.restart_margin)
            and (t_now - self._last_restart_t) >= self.restart_cooldown
        )

        if mass_gate:
            did_restart = True
            self._last_restart_t = t_now

            if bool(getattr(self.config, "hybrid_partial_restart", True)) and anchor_e is not None and cand_e is not None:
                L_aa = get_pre_log_ump(anchor_e)
                L_cc = get_pre_log_ump(cand_e)
                if L_aa is None:
                    L_aa = self._single_log_ump(anchor_e, anchor_e, X_batch, Y_batch, emulator, model_cfg)
                if L_cc is None:
                    L_cc = self._single_log_ump(cand_e, cand_e, X_batch, Y_batch, emulator, model_cfg)

                try:
                    L_ac = self._single_log_ump(anchor_e, cand_e, X_batch, Y_batch, emulator, model_cfg)
                    L_ca = self._single_log_ump(cand_e, anchor_e, X_batch, Y_batch, emulator, model_cfg)
                    gain_delta = float(L_ac - L_aa)
                    gain_theta = float(L_ca - L_aa)
                    gain_full = float(L_cc - L_aa)

                    tau_delta = float(getattr(self.config, "hybrid_tau_delta", 0.05))
                    tau_theta = float(getattr(self.config, "hybrid_tau_theta", 0.05))
                    tau_full = float(getattr(self.config, "hybrid_tau_full", 0.05))
                    rho = float(getattr(self.config, "hybrid_delta_share_rho", 0.75))
                    ratio = gain_delta / max(gain_full, 1e-12)

                    # delta-only only when candidate gain is largely explained by delta replacement
                    if (
                        gain_delta > tau_delta
                        and gain_full > tau_full
                        and ratio >= rho
                        and gain_theta < tau_theta
                    ):
                        restart_mode = "delta_only"
                except Exception:
                    restart_mode = "full"

            if restart_mode == "delta_only":
                # Keep anchor PF (theta), reset discrepancy only.
                msg_mode = self._do_delta_only_restart(t_now, anchor_e, dx)
            else:
                # strict policy: once mass_gate passes, branch must restart (full or delta-only)
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

            if did_restart and self.on_restart is not None and self.notify_on_restart:
                self.on_restart(
                    int(t_now),
                    int(self.restart_start_time),
                    int(s_star) if s_star is not None else None,
                    int(anchor_rl),
                    float(p_anchor),
                    float(best_other_mass),
                )

        if not did_restart:
            restart_mode = "none"
            anchor_run_length = max(t_now - self.restart_start_time, 0)
            self._prune_keep_anchor(anchor_run_length, self.config.max_experts)

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
                use_discrepancy=model_cfg.use_discrepancy,
            )
            pf_diags.append(diag)
            if not getattr(model_cfg, "refit_delta_every_batch", True):
                continue
            self._reset_delta_for_expert(e, model_cfg)

        self.t += batch_size

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

        log_ump_anchor = get_pre_log_ump(anchor_e) if anchor_e is not None else None
        log_ump_cand = get_pre_log_ump(cand_e) if cand_e is not None else None
        delta_ll_pair = None
        if (log_ump_anchor is not None) and (log_ump_cand is not None):
            delta_ll_pair = float(log_ump_cand - log_ump_anchor)
        log_odds_mass = None
        if anchor_e is not None and cand_e is not None:
            log_odds_mass = float(cand_e.log_mass - anchor_e.log_mass)
        h_log = float(math.log1p(self.restart_margin))

        gain_full = None if (L_cc is None or L_aa is None) else float(L_cc - L_aa)
        gain_delta_ratio = None
        if gain_delta is not None and gain_full is not None:
            gain_delta_ratio = float(gain_delta / max(gain_full, 1e-12))

        return {
            "p_anchor": p_anchor,
            "p_cp": p_cp,
            "num_experts": len(self.experts),
            "experts_log_mass": [float(e.log_mass) for e in self.experts],
            "pf_diags": pf_diags,
            "did_restart": did_restart,
            "restart_mode": restart_mode if did_restart else "none",
            "restart_message": msg_mode,
            "restart_start_time": int(self.restart_start_time),
            "s_star": int(s_star) if s_star is not None else None,
            "log_umps": [float(v) for v in log_umps],
            "log_Z": float(log_Z),
            "entropy": float(entropy),
            "experts_debug": experts_debug,
            "anchor_rl": int(anchor_e.run_length) if anchor_e is not None else None,
            "cand_rl": int(cand_e.run_length) if cand_e is not None else None,
            "delta_ll_pair": delta_ll_pair,
            "log_odds_mass": log_odds_mass,
            "h_log": h_log,
            "log_ump_anchor": log_ump_anchor,
            "log_ump_cand": log_ump_cand,
            "theta_stat": theta_stat,
            "L_aa": L_aa,
            "L_ac": L_ac,
            "L_ca": L_ca,
            "L_cc": L_cc,
            "gain_delta": gain_delta,
            "gain_theta": gain_theta,
            "gain_full": gain_full,
            "gain_delta_ratio": gain_delta_ratio,
        }

