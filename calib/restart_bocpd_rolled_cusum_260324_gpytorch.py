from typing import Any, Callable, Dict, Optional, Tuple
import math
import torch

from .configs import ModelConfig, PFConfig
from .emulator import Emulator
from .particles import ParticleSet
from .restart_bocpd_hybrid_260319_gpytorch import BOCPD as HybridBOCPD
from .restart_bocpd_debug_260115_gpytorch import Expert


class BOCPD(HybridBOCPD):
    """
    Opt-in hybrid BOCPD variant with discrepancy-memory refresh policies.

    The refresh patch is intentionally isolated from BOCPD/PF logic:
    - `super().update_batch(...)` runs the original hybrid implementation.
    - only when BOCPD did not restart do we optionally refresh the anchor
      expert's discrepancy history.
    - any refresh-policy failure degrades to a no-op so existing runs stay intact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cusum = bool(getattr(self.config, "use_cusum", True))
        self.cusum_threshold = float(getattr(self.config, "cusum_threshold", 10.0))
        self.cusum_recent_obs = max(1, int(getattr(self.config, "cusum_recent_obs", 20)))
        self.cusum_cov_eps = max(float(getattr(self.config, "cusum_cov_eps", 1e-6)), 1e-12)

        mode = str(getattr(self.config, "cusum_mode", "cumulative")).lower()
        if mode in {"gate", "single_step_gate", "standardized_gate", "single_step_standardized_gate"}:
            mode = "standardized_gate"
        else:
            mode = "cumulative"
        self.cusum_mode = mode
        self.standardized_gate_threshold = float(getattr(self.config, "standardized_gate_threshold", 3.0))
        self.standardized_gate_consecutive = max(1, int(getattr(self.config, "standardized_gate_consecutive", 1)))

        self._cusum_stat = 0.0
        self._cusum_tau = 0
        self._gate_hits = 0
        self._cusum_prev_anchor_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _anchor_expert_after_update(self) -> Optional[Expert]:
        if len(self.experts) == 0:
            return None
        anchor_run_length = max(int(self.t) - int(self.restart_start_time), 0)
        return self._closest_by_run_length(anchor_run_length)

    def _posterior_summary(self, e: Expert) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        ps: ParticleSet = e.pf.particles
        theta = ps.theta
        if theta.numel() == 0:
            return None

        w = ps.weights().reshape(-1)
        if w.numel() != theta.shape[0]:
            return None
        if torch.isnan(w).any() or torch.isinf(w).any():
            return None

        w_sum = torch.clamp(w.sum(), min=1e-12)
        w = w / w_sum
        mean = (w.view(-1, 1) * theta).sum(dim=0)
        centered = theta - mean
        cov = centered.transpose(0, 1) @ (w.view(-1, 1) * centered)
        return mean.detach().clone(), cov.detach().clone()

    def _cusum_distance(
        self,
        prev_stats: Tuple[torch.Tensor, torch.Tensor],
        cur_stats: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        prev_mean, prev_cov = prev_stats
        cur_mean, _ = cur_stats
        delta = (cur_mean - prev_mean).reshape(-1, 1)
        dim = int(delta.shape[0])
        eye = torch.eye(dim, dtype=prev_cov.dtype, device=prev_cov.device)
        reg_cov = prev_cov + self.cusum_cov_eps * eye
        try:
            solved = torch.linalg.solve(reg_cov, delta)
        except Exception:
            solved = torch.linalg.pinv(reg_cov) @ delta
        score = float((delta.transpose(0, 1) @ solved).reshape(()).item())
        if not math.isfinite(score):
            raise ValueError("non-finite refresh score")
        return max(score, 0.0)

    def _refresh_anchor_discrepancy(self, anchor_e: Expert, model_cfg: ModelConfig) -> bool:
        if anchor_e.X_hist.numel() == 0 or anchor_e.y_hist.numel() == 0:
            return False

        old_X = anchor_e.X_hist
        old_y = anchor_e.y_hist
        old_delta = anchor_e.delta_state

        keep_n = min(int(anchor_e.X_hist.shape[0]), self.cusum_recent_obs)
        recent_X = anchor_e.X_hist[-keep_n:].clone()
        recent_y = anchor_e.y_hist[-keep_n:].clone()

        try:
            anchor_e.X_hist = recent_X
            anchor_e.y_hist = recent_y
            self._reset_delta_for_expert(anchor_e, model_cfg)
            if keep_n >= 3 and anchor_e.delta_state is None and old_delta is not None:
                raise RuntimeError("delta refit returned None on recent memory")
            return True
        except Exception:
            anchor_e.X_hist = old_X
            anchor_e.y_hist = old_y
            anchor_e.delta_state = old_delta
            return False

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
        out = super().update_batch(
            X_batch=X_batch,
            Y_batch=Y_batch,
            emulator=emulator,
            model_cfg=model_cfg,
            pf_cfg=pf_cfg,
            prior_sampler=prior_sampler,
            verbose=verbose,
        )

        anchor_e = self._anchor_expert_after_update()
        cur_stats = self._posterior_summary(anchor_e) if anchor_e is not None else None

        refresh_score = None
        gate_score = None
        cusum_triggered = False
        gate_triggered = False
        refresh_ok = False
        refresh_error = None

        if bool(out.get("did_restart", False)):
            self._cusum_stat = 0.0
            self._cusum_tau = int(self.t)
            self._gate_hits = 0
            self._cusum_prev_anchor_stats = cur_stats
        elif self.use_cusum and anchor_e is not None and cur_stats is not None:
            prev_stats = self._cusum_prev_anchor_stats
            if prev_stats is not None:
                try:
                    refresh_score = self._cusum_distance(prev_stats, cur_stats)
                    gate_score = math.sqrt(refresh_score)

                    if self.cusum_mode == "standardized_gate":
                        if gate_score > self.standardized_gate_threshold:
                            self._gate_hits += 1
                        else:
                            self._gate_hits = 0
                        if self._gate_hits >= self.standardized_gate_consecutive:
                            gate_triggered = True
                            refresh_ok = self._refresh_anchor_discrepancy(anchor_e, model_cfg)
                            if refresh_ok:
                                self._gate_hits = 0
                                self._cusum_stat = 0.0
                                self._cusum_tau = int(self.t)
                                out["restart_mode"] = "standardized_gate_refresh"
                                out["restart_message"] = "Standardized gate discrepancy refresh"
                    else:
                        self._cusum_stat += refresh_score
                        if self._cusum_stat > self.cusum_threshold:
                            cusum_triggered = True
                            refresh_ok = self._refresh_anchor_discrepancy(anchor_e, model_cfg)
                            if refresh_ok:
                                self._cusum_stat = 0.0
                                self._gate_hits = 0
                                self._cusum_tau = int(self.t)
                                out["restart_mode"] = "cusum_refresh"
                                out["restart_message"] = "CUSUM discrepancy refresh"
                except Exception as exc:
                    refresh_error = str(exc)
            self._cusum_prev_anchor_stats = cur_stats
        else:
            self._cusum_prev_anchor_stats = cur_stats

        out["cusum_enabled"] = bool(self.use_cusum)
        out["cusum_mode"] = self.cusum_mode
        out["cusum_threshold"] = float(self.cusum_threshold)
        out["cusum_recent_obs"] = int(self.cusum_recent_obs)
        out["cusum_score"] = refresh_score
        out["cusum_stat"] = float(self._cusum_stat)
        out["cusum_triggered"] = bool(cusum_triggered)
        out["standardized_gate_threshold"] = float(self.standardized_gate_threshold)
        out["standardized_gate_consecutive"] = int(self.standardized_gate_consecutive)
        out["gate_score"] = gate_score
        out["gate_hits"] = int(self._gate_hits)
        out["gate_triggered"] = bool(gate_triggered)
        out["refresh_triggered"] = bool(cusum_triggered or gate_triggered)
        out["cusum_refresh_ok"] = bool(refresh_ok)
        out["cusum_tau"] = int(self._cusum_tau)
        out["cusum_anchor_rl"] = int(anchor_e.run_length) if anchor_e is not None else None
        out["cusum_error"] = refresh_error
        return out
