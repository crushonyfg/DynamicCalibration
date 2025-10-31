# =============================================================
# file: calib/online_calibrator.py
# =============================================================
from typing import Any, Callable, Dict, Optional, Tuple
import math
import torch

# from .restart_bocpd import BOCPD
from .configs import CalibrationConfig
from .emulator import Emulator
from .delta_gp import OnlineGPState
from .likelihood import predictive_stats
from .expert_delta import ExpertDeltaFitter  # <--- NEW

import logging
logging.basicConfig(level=logging.INFO)

def my_restart_hook(t_now, r_new, s_star, anchor_rl, p_anchor, best_other):
    logging.info(f"[HOOK] Restart at t={t_now}: r←{r_new}, s*={s_star}, "
                 f"anchor_rl={anchor_rl}, p_anchor={p_anchor:.4f}, best={best_other:.4f}")

from .bocpd import BOCPD as StandardBOCPD
from .restart_bocpd import BOCPD as RestartBOCPD


class OnlineBayesCalibrator:
    def __init__(
        self,
        calib_cfg: CalibrationConfig,
        emulator: Emulator,
        prior_sampler: Callable[[int], torch.Tensor],
        init_delta_state: Optional[Callable[[], OnlineGPState]] = None,
        delta_fitter: Optional[ExpertDeltaFitter] = None,  # <--- NEW
        on_restart: Callable = None,  # ✅ 可选的restart回调
        notify_on_restart: bool = True,
    ):
        self.cfg = calib_cfg
        self.emulator = emulator
        self.prior_sampler = prior_sampler
        self.init_delta_state = init_delta_state  # not used; BOCPD builds states with kernel cfg
        config = calib_cfg

        # Construct a default fitter if not provided and if refit is enabled via config
        if delta_fitter is None and int(getattr(calib_cfg.bocpd, "delta_refit_every", 0)) > 0:
            # You can also read steps/lr from config if you add them there
            delta_fitter = ExpertDeltaFitter(train_steps=150, lr=0.05)

        bocpd_mode = getattr(calib_cfg.bocpd, "bocpd_mode", "standard").lower()
        if bocpd_mode == "restart":
            # 使用 R-BOCPD 实现（restart_bocpd.py）
            self.bocpd = RestartBOCPD(
                config=config.bocpd,
                device=config.model.device,
                dtype=config.model.dtype,
                delta_fitter=None,  # 可选：如果需要delta refitting
                # delta_fitter=delta_fitter,
                on_restart=on_restart,
                notify_on_restart=notify_on_restart,
            )
            self.bocpd_mode = "restart"
            print(f"✅ Using R-BOCPD mode: {'Backdated' if config.bocpd.use_backdated_restart else 'Algorithm-2'}")
        else:
            # 使用标准 BOCPD 实现（bocpd.py）
            self.bocpd = StandardBOCPD(
                config=config.bocpd,
                device=config.model.device,
                dtype=config.model.dtype,
                delta_fitter=None,
                # delta_fitter=delta_fitter,
            )
            self.bocpd_mode = "standard"
            print(f"✅ Using Standard BOCPD mode (use_restart={config.bocpd.use_restart})")
    

    def step(self, x_t: torch.Tensor, y_t: torch.Tensor, verbose: bool = False) -> Dict[str, Any]:
        out = self.bocpd.update(
            x_t.to(self.cfg.model.device, self.cfg.model.dtype),
            y_t.to(self.cfg.model.device, self.cfg.model.dtype),
            self.emulator,
            self.cfg.model,
            self.cfg.pf,
            self.prior_sampler,
            verbose=verbose,  # ✅ 传递verbose参数
        )
        if self.bocpd_mode == "restart" and "p_cp" not in out:
            # restart mode可能返回 p_anchor，这里统一接口
            out["p_cp"] = out.get("p_cp", 0.0)

        
        return out
    
    def step_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, verbose: bool = False) -> Dict[str, Any]:
        """批量更新校准器"""
        out = self.bocpd.update_batch(
            X_batch.to(self.cfg.model.device, self.cfg.model.dtype),
            Y_batch.to(self.cfg.model.device, self.cfg.model.dtype),
            self.emulator,
            self.cfg.model,
            self.cfg.pf,
            self.prior_sampler,
            verbose=verbose,
        )
        if self.bocpd_mode == "restart" and "p_cp" not in out:
            out["p_cp"] = out.get("p_cp", 0.0)
        return out

    def predict_batch(self, X_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """批量预测"""
        X_batch = X_batch.to(self.cfg.model.device, self.cfg.model.dtype)
        batch_size = X_batch.shape[0]
        
        mix_mu = torch.zeros(batch_size, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        mix_var = torch.zeros(batch_size, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        Z = 0.0
        
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, var_eta = self.emulator.predict(X_batch, e.pf.particles.theta)  # [batch_size, N]
            mu_delta, var_delta = e.delta_state.predict(X_batch)  # [batch_size]
            mu, var = predictive_stats(self.cfg.model.rho, mu_eta, var_eta, mu_delta, var_delta, self.cfg.model.sigma_eps)
            w = e.pf.particles.weights()[None, :]
            mu_mix = (w * mu).sum(dim=1)  # [batch_size]
            var_mix = (w * (var + mu**2)).sum(dim=1) - mu_mix**2
            mix_mu += w_e * mu_mix
            mix_var += w_e * var_mix
            Z += w_e
        
        mix_mu = mix_mu / max(Z, 1e-12)
        mix_var = mix_var / max(Z, 1e-12)
        return {"mu": mix_mu, "var": mix_var}

    def _aggregate_particles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # mixture across experts by their masses
        if len(self.bocpd.experts) == 0:
            return None, None
        d = self.bocpd.experts[0].pf.particles.theta.shape[1]
        mean = torch.zeros(d, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        cov = torch.zeros(d, d, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            w = e.pf.particles.weights()
            th = e.pf.particles.theta
            m = (w[:, None] * th).sum(0)
            C = ((th - m) * w[:, None]).T @ (th - m)
            mean = mean + w_e * m
            cov = cov + w_e * (C + (m - mean)[:, None] @ (m - mean)[None, :])
        return mean, cov

    def predict(self, x_next: torch.Tensor) -> Dict[str, torch.Tensor]:
        # mixture over experts and their particles
        xs = x_next.to(self.cfg.model.device, self.cfg.model.dtype)
        mix_mu = 0.0
        mix_var = 0.0
        Z = 0.0
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, var_eta = self.emulator.predict(xs[None, :], e.pf.particles.theta)  # [1,N]
            mu_delta, var_delta = e.delta_state.predict(xs[None, :])  # [1]
            mu, var = predictive_stats(self.cfg.model.rho, mu_eta, var_eta, mu_delta, var_delta, self.cfg.model.sigma_eps)
            w = e.pf.particles.weights()[None, :]
            mu_mix = (w * mu).sum(dim=1)  # [1]
            var_mix = (w * (var + mu**2)).sum(dim=1) - mu_mix**2
            mix_mu += w_e * mu_mix.squeeze(0)
            mix_var += w_e * var_mix.squeeze(0)
            Z += w_e
        mix_mu = mix_mu / max(Z, 1e-12)
        mix_var = mix_var / max(Z, 1e-12)
        return {"mu": mix_mu, "var": mix_var}
