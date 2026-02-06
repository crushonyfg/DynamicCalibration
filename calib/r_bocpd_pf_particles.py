# ============================================================
# r_bocpd_pf_particle.py
# True R-BOCPD-PF-P (per-particle discrepancy)
#
# - Inherits existing RestartBOCPD and ParticleFilter
# - Overrides likelihood to use per-particle discrepancy
# - Keeps UMP / run-length / restart intact
# ============================================================

import math
import torch
from dataclasses import dataclass

from .particles import ParticleSet
from .pf import ParticleFilter
from .restart_bocpd_debug_260115_gpytorch import BOCPD as RestartBOCPD
from .online_calibrator import OnlineBayesCalibrator
from .emulator import Emulator
from .configs import CalibrationConfig

@dataclass
class ParticleSetPF_P(ParticleSet):
    """
    Particle set with per-particle discrepancy hyperparameters.
    """
    delta_logsf: torch.Tensor   # [N, 1]
    delta_logell: torch.Tensor  # [N, 1]


class ParticleFilterPF_P(ParticleFilter):

    @classmethod
    def from_prior(
        cls,
        prior_sampler,
        pf_config,
        device="cpu",
        dtype=torch.float64,
        theta_anchor=None,
        model_cfg=None,
    ):
        # --- theta ---
        theta = prior_sampler(pf_config.num_particles).to(device=device, dtype=dtype)
        logw = torch.full((pf_config.num_particles,), -math.log(pf_config.num_particles),
                          device=device, dtype=dtype)

        # --- discrepancy hyperparams ---
        logsf = (
            model_cfg.delta_p_logsf_mean
            + model_cfg.delta_p_logsf_std * torch.randn(pf_config.num_particles, 1, device=device, dtype=dtype)
        )
        logell = (
            model_cfg.delta_p_logell_mean
            + model_cfg.delta_p_logell_std * torch.randn(pf_config.num_particles, 1, device=device, dtype=dtype)
        )

        particles = ParticleSetPF_P(
            theta=theta,
            logw=logw,
            delta_logsf=logsf,
            delta_logell=logell,
        )
        return cls(particles, pf_config, device, dtype)

    # --------------------------------------------------------
    # PF-P likelihood (NO shared discrepancy)
    # --------------------------------------------------------
    def step_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        emulator: Emulator,
        delta_state,   # ignored
        rho: float,
        sigma_eps: float,
        grad_info: bool = False,
        **kwargs,
    ):
        # Emulator prediction
        mu_eta, var_eta = emulator.predict(X_batch, self.particles.theta)  # [B, N]

        # Per-particle discrepancy variance
        sf2 = torch.exp(2.0 * self.particles.delta_logsf.view(1, -1))       # [1, N]

        # Total variance
        var = rho**2 * var_eta + sf2 + sigma_eps**2
        mu = rho * mu_eta

        # Log-likelihood
        res = Y_batch.view(-1, 1) - mu
        ll = -0.5 * (torch.log(2 * math.pi * var) + res**2 / var)
        loglik = ll.sum(dim=0)  # [N]

        # Weight update
        self.particles.logw += loglik
        self.particles.logw -= torch.logsumexp(self.particles.logw, dim=0)

        ess = self.particles.ess()
        N = self.particles.theta.shape[0]

        if ess < self.config.resample_ess_ratio * N:
            idx = torch.multinomial(self.particles.weights(), N, replacement=True)
            self.particles.theta = self.particles.theta[idx]
            self.particles.delta_logsf = self.particles.delta_logsf[idx]
            self.particles.delta_logell = self.particles.delta_logell[idx]
            self.particles.logw.fill_(-math.log(N))

            # Random walk move
            self.particles.theta += self.config.random_walk_scale * torch.randn_like(self.particles.theta)
            self.particles.delta_logsf += model_cfg.delta_p_rw_logsf * torch.randn_like(self.particles.delta_logsf)
            self.particles.delta_logell += model_cfg.delta_p_rw_logell * torch.randn_like(self.particles.delta_logell)

        return {
            "ess": float(ess),
            "log_evidence": float(torch.logsumexp(loglik, dim=0)),
            "resampled": ess < self.config.resample_ess_ratio * N,
        }

class RestartBOCPD_PF_P(RestartBOCPD):

    def _spawn_pf(self, prior_sampler, pf_cfg, model_cfg):
        return ParticleFilterPF_P.from_prior(
            prior_sampler,
            pf_cfg,
            device=self.device,
            dtype=self.dtype,
            model_cfg=model_cfg,
        )

class OnlineBayesCalibrator_PF_P(OnlineBayesCalibrator):
    """
    Drop-in replacement of OnlineBayesCalibrator
    implementing true R-BOCPD-PF-P.
    """

    def __init__(self, calib_cfg, emulator, prior_sampler):
        super().__init__(calib_cfg, emulator, prior_sampler)
        self.bocpd = RestartBOCPD_PF_P(
            config=calib_cfg.bocpd,
            device=calib_cfg.model.device,
            dtype=calib_cfg.model.dtype,
        )
