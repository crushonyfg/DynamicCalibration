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

# from .bocpd import BOCPD as StandardBOCPD
from .bocpd_gpytorch import BOCPD as StandardBOCPD
# from .restart_bocpd import BOCPD as RestartBOCPD

# from .restart_bocpd_mbr import BOCPD as RestartBOCPD
# from .restart_bocpd_mod import BOCPD as RestartBOCPD

# from .restart_bocpd_debug_260108 import BOCPD as RestartBOCPD
# from .restart_bocpd_debug_260114 import BOCPD as RestartBOCPD
# default restart implementation
from .restart_bocpd_debug_260115_gpytorch import BOCPD as RestartBOCPD
# from .restart_bocpd_260123_noisevec import BOCPD as RestartBOCPD

import torch

def crps_weighted(samples, weights, y):
    """
    samples: [N] or [N, D]
    weights: [N], sum to 1
    y: scalar or [D]
    """
    w = weights / weights.sum()

    # term 1: E|X - y|
    term1 = torch.sum(w * torch.norm(samples - y, dim=-1))

    # term 2: E|X - X'|
    diff = samples[:, None, ...] - samples[None, :, ...]
    term2 = torch.sum(
        w[:, None] * w[None, :] *
        torch.norm(diff, dim=-1)
    )

    return term1 - 0.5 * term2

import math
import torch

def normal_pdf(z):
    return torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)

def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def crps_gaussian(mu, var, y, eps=1e-12):
    sigma = torch.sqrt(torch.clamp(var, min=eps))
    z = (y - mu) / sigma
    Phi = normal_cdf(z)
    phi = normal_pdf(z)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))

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
            restart_impl = str(getattr(calib_cfg.bocpd, "restart_impl", "debug_260115")).lower()
            if restart_impl in {"hybrid", "hybrid_260319"}:
                from .restart_bocpd_hybrid_260319_gpytorch import BOCPD as RestartBOCPDImpl
            elif restart_impl in {"cusum", "rolled_cusum", "rolled_cusum_260324", "hybrid_cusum_260324"}:
                from .restart_bocpd_rolled_cusum_260324_gpytorch import BOCPD as RestartBOCPDImpl
            else:
                RestartBOCPDImpl = RestartBOCPD
            # 使用 R-BOCPD 实现（restart_bocpd.py）
            self.bocpd = RestartBOCPDImpl(
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
        """批量预测。支持标量 y（mu/var [batch]）与多维 y（mu/var [batch, dy]）。"""
        X_batch = X_batch.to(self.cfg.model.device, self.cfg.model.dtype)
        batch_size = X_batch.shape[0]
        if len(self.bocpd.experts) == 0:
            return {"mu": torch.zeros(batch_size, device=X_batch.device, dtype=X_batch.dtype),
                    "var": torch.ones(batch_size, device=X_batch.device, dtype=X_batch.dtype),
                    "experts_res": [], "mu_sim": torch.zeros(batch_size, device=X_batch.device, dtype=X_batch.dtype)}
        # Infer output dim from first expert's emulator predict
        mu_eta0, _ = self.emulator.predict(X_batch[:1], self.bocpd.experts[0].pf.particles.theta[:1])
        dy = mu_eta0.shape[-1] if mu_eta0.dim() == 3 else 1
        if dy == 1:
            mix_mu = torch.zeros(batch_size, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
            mix_var = torch.zeros(batch_size, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
            mix_mu_sim = torch.zeros(batch_size, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        else:
            mix_mu = torch.zeros(batch_size, dy, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
            mix_var = torch.zeros(batch_size, dy, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
            mix_mu_sim = torch.zeros(batch_size, dy, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        Z = 0.0
        experts_res = []

        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, var_eta = self.emulator.predict(X_batch, e.pf.particles.theta)  # [B, N] or [B, N, dy]
            try:
                mu_delta, var_delta = e.delta_state.predict(X_batch)  # [B]
            except Exception:
                mu_deltas, var_deltas = [], []
                for delta_state in e.delta_states:
                    md, vd = delta_state.predict(X_batch)
                    mu_deltas.append(md)
                    var_deltas.append(vd)
                mu_delta = torch.stack(mu_deltas, dim=1).mean(dim=1)
                var_delta = torch.stack(var_deltas, dim=1).mean(dim=1)
            if mu_eta.dim() == 3 and mu_delta.dim() == 1:
                mu_delta = mu_delta[:, None].expand(-1, mu_eta.shape[-1])
                var_delta = var_delta[:, None].expand(-1, mu_eta.shape[-1])
            mu, var = predictive_stats(self.cfg.model.rho, mu_eta, var_eta, mu_delta, var_delta, self.cfg.model.sigma_eps)
            w = e.pf.particles.weights()[None, :]
            if mu.dim() == 2:
                mu_mix = (w * mu).sum(dim=1)  # [B]
                var_mix = (w * (var + mu**2)).sum(dim=1) - mu_mix**2
            else:
                mu_mix = (w[:, :, None] * mu).sum(dim=1)  # [B, dy]
                var_mix = (w[:, :, None] * (var + mu**2)).sum(dim=1) - mu_mix**2
            mix_mu += w_e * mu_mix
            mix_var += w_e * var_mix
            experts_res.append({"mu_delta": mu_delta, "var_delta": var_delta, "w": w_e, "mu": mu_mix, "var": var_mix})
            mu_sim = self.cfg.model.rho * mu_eta
            if mu_sim.dim() == 2:
                mu_sim_mix = (w * mu_sim).sum(dim=1)
            else:
                mu_sim_mix = (w[:, :, None] * mu_sim).sum(dim=1)
            mix_mu_sim += w_e * mu_sim_mix
            Z += w_e

        mix_mu = mix_mu / max(Z, 1e-12)
        mix_var = mix_var / max(Z, 1e-12)
        mix_mu_sim = mix_mu_sim / max(Z, 1e-12)
        return {"mu": mix_mu, "var": mix_var, "experts_res": experts_res, "mu_sim": mix_mu_sim}

    def predict_complete(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ):
        X_batch = X_batch.to(self.cfg.model.device, self.cfg.model.dtype)
        y_batch = y_batch.to(self.cfg.model.device, self.cfg.model.dtype)
        batch_size = X_batch.shape[0]
        # Infer dy from first expert
        if len(self.bocpd.experts) == 0:
            return {"mix_mu": None, "mix_var": None, "mu_sim": None, "var_sim": None, "crps_sim": None, "experts_logpred": []}
        mu_eta0, _ = self.emulator.predict(X_batch[:1], self.bocpd.experts[0].pf.particles.theta[:1])
        dy = mu_eta0.shape[-1] if mu_eta0.dim() == 3 else 1
        if dy == 1:
            mix_mu = torch.zeros(batch_size, dtype=X_batch.dtype, device=X_batch.device)
            mix_var = torch.zeros(batch_size, dtype=X_batch.dtype, device=X_batch.device)
            mix_mu_sim = torch.zeros(batch_size, dtype=X_batch.dtype, device=X_batch.device)
        else:
            mix_mu = torch.zeros(batch_size, dy, dtype=X_batch.dtype, device=X_batch.device)
            mix_var = torch.zeros(batch_size, dy, dtype=X_batch.dtype, device=X_batch.device)
            mix_mu_sim = torch.zeros(batch_size, dy, dtype=X_batch.dtype, device=X_batch.device)
        Z = 0.0
        experts_logpred = []

        y_ = y_batch if y_batch.dim() >= 2 else y_batch.unsqueeze(-1)  # [B, dy]
        if dy > 1 and y_.shape[-1] == 1:
            y_ = y_.expand(-1, dy)

        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, var_eta = self.emulator.predict(X_batch, e.pf.particles.theta)
            try:
                mu_delta, var_delta = e.delta_state.predict(X_batch)
            except Exception:
                mu_deltas, var_deltas = [], []
                for ds in e.delta_states:
                    md, vd = ds.predict(X_batch)
                    mu_deltas.append(md)
                    var_deltas.append(vd)
                mu_delta = torch.stack(mu_deltas, dim=1).mean(dim=1)
                var_delta = torch.stack(var_deltas, dim=1).mean(dim=1)
            if mu_eta.dim() == 3 and mu_delta.dim() == 1:
                mu_delta = mu_delta[:, None].expand(-1, mu_eta.shape[-1])
                var_delta = var_delta[:, None].expand(-1, mu_eta.shape[-1])
            mu, var = predictive_stats(
                self.cfg.model.rho, mu_eta, var_eta, mu_delta, var_delta, self.cfg.model.sigma_eps,
            )
            w = e.pf.particles.weights()[None, :]
            if mu.dim() == 2:
                mu_e = (w * mu).sum(dim=1)
                var_e = (w * (var + mu**2)).sum(dim=1) - mu_e**2
            else:
                mu_e = (w[:, :, None] * mu).sum(dim=1)
                var_e = (w[:, :, None] * (var + mu**2)).sum(dim=1) - mu_e**2
            mix_mu += w_e * mu_e
            mix_var += w_e * var_e
            mu_sim = self.cfg.model.rho * mu_eta
            if mu_sim.dim() == 2:
                mu_sim_e = (w * mu_sim).sum(dim=1)
            else:
                mu_sim_e = (w[:, :, None] * mu_sim).sum(dim=1)
            mix_mu_sim += w_e * mu_sim_e
            # log N(y | mu_e, var_e); for dy>1 sum over dims (independent)
            logp_e = -0.5 * (
                torch.log(2.0 * math.pi * torch.clamp(var_e, min=1e-12))
                + (y_ - mu_e)**2 / torch.clamp(var_e, min=1e-12)
            )
            logp_e = logp_e.sum(dim=-1).mean()
            experts_logpred.append({"logp": logp_e.detach(), "weight": w_e, "log_mass": e.log_mass})
            Z += w_e

        Z = max(Z, 1e-12)
        mix_mu = mix_mu / Z
        mix_var = mix_var / Z
        mix_mu_sim = mix_mu_sim / Z

        Ey2_sim = torch.zeros_like(mix_mu_sim)
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, _ = self.emulator.predict(X_batch, e.pf.particles.theta)
            mu_sim = self.cfg.model.rho * mu_eta
            w = e.pf.particles.weights()[None, :]
            if mu_sim.dim() == 2:
                Ey2_sim += w_e * (w * mu_sim**2).sum(dim=1)
            else:
                Ey2_sim += w_e * (w[:, :, None] * mu_sim**2).sum(dim=1)
        Ey2_sim = Ey2_sim / Z
        var_sim = torch.clamp(Ey2_sim - mix_mu_sim**2, min=1e-12)
        crps_sim = crps_gaussian(mix_mu_sim, var_sim, y_ if y_batch.dim() >= 2 else y_batch).mean()

        return {
            "mix_mu": mix_mu,
            "mix_var": mix_var,
            "mu_sim": mix_mu_sim,
            "var_sim": var_sim,
            "crps_sim": crps_sim,
            "experts_logpred": experts_logpred,
        }


    def _aggregate_particles(self, quantile = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # mixture across experts by their masses
        if len(self.bocpd.experts) == 0:
            return None, None
        d = self.bocpd.experts[0].pf.particles.theta.shape[1]
        mean = torch.zeros(d, dtype=self.cfg.model.dtype, device=self.cfg.model.device)
        cov = torch.zeros(d, d, dtype=self.cfg.model.dtype, device=self.cfg.model.device)

        theta_list, weight_list = [], []
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            w = e.pf.particles.weights()
            th = e.pf.particles.theta
            m = (w[:, None] * th).sum(0)
            C = ((th - m) * w[:, None]).T @ (th - m)
            mean = mean + w_e * m
            cov = cov + w_e * (C + (m - mean)[:, None] @ (m - mean)[None, :])
            theta_list.append(th)
            weight_list.append(w_e*w)

        theta_all = torch.cat(theta_list, dim=0)
        weight_all = torch.cat(weight_list, dim=0)
        w_sum = weight_all.sum()

        # Handle degenerate weights (all zero / NaN) — fall back to uniform
        if w_sum < 1e-30 or torch.isnan(w_sum):
            weight_all = torch.ones_like(weight_all) / len(weight_all)
        else:
            weight_all = weight_all / w_sum

        def weighted_quantile_1d(x, w, q):
            idx = torch.argsort(x)
            x = x[idx]
            w = w[idx]
            cw = torch.cumsum(w, dim=0)
            mask = cw >= q
            if mask.any():
                return x[mask][0]
            # Fallback: return last element (highest quantile available)
            return x[-1]

        def particle_ci(theta_all, weight_all, level=0.9):
            alpha = (1.0 - level) / 2.0
            lo_q = alpha
            hi_q = 1.0 - alpha

            d = theta_all.shape[1]
            lo = torch.zeros(d, dtype=theta_all.dtype, device=theta_all.device)
            hi = torch.zeros(d, dtype=theta_all.dtype, device=theta_all.device)
            for j in range(d):
                lo[j] = weighted_quantile_1d(theta_all[:, j], weight_all, lo_q)
                hi[j] = weighted_quantile_1d(theta_all[:, j], weight_all, hi_q)
            return lo, hi
        
        if quantile is None:
            return mean, cov
        else:
            lo, hi = particle_ci(theta_all, weight_all, quantile)
            return mean, cov, lo, hi

    def predict(self, x_next: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单点预测；支持标量/多维 y（返回 mu, var 与 emulator 输出维度一致）。"""
        xs = x_next.to(self.cfg.model.device, self.cfg.model.dtype)
        if xs.dim() == 1:
            xs = xs[None, :]
        if len(self.bocpd.experts) == 0:
            return {"mu": None, "var": None}
        mu_eta0, _ = self.emulator.predict(xs, self.bocpd.experts[0].pf.particles.theta[:1])
        dy = mu_eta0.shape[-1] if mu_eta0.dim() == 3 else 1
        if dy == 1:
            mix_mu = torch.zeros(1, dtype=xs.dtype, device=xs.device)
            mix_var = torch.zeros(1, dtype=xs.dtype, device=xs.device)
        else:
            mix_mu = torch.zeros(dy, dtype=xs.dtype, device=xs.device)
            mix_var = torch.zeros(dy, dtype=xs.dtype, device=xs.device)
        Z = 0.0
        for e in self.bocpd.experts:
            w_e = math.exp(e.log_mass)
            mu_eta, var_eta = self.emulator.predict(xs, e.pf.particles.theta)
            mu_delta, var_delta = e.delta_state.predict(xs)
            if mu_eta.dim() == 3 and mu_delta.dim() == 1:
                mu_delta = mu_delta[:, None].expand(-1, mu_eta.shape[-1])
                var_delta = var_delta[:, None].expand(-1, mu_eta.shape[-1])
            mu, var = predictive_stats(self.cfg.model.rho, mu_eta, var_eta, mu_delta, var_delta, self.cfg.model.sigma_eps)
            w = e.pf.particles.weights()[None, :]
            if mu.dim() == 2:
                mu_mix = (w * mu).sum(dim=1)
                var_mix = (w * (var + mu**2)).sum(dim=1) - mu_mix**2
            else:
                mu_mix = (w[:, :, None] * mu).sum(dim=1).squeeze(0)
                var_mix = (w[:, :, None] * (var + mu**2)).sum(dim=1).squeeze(0) - mu_mix**2
            mix_mu += w_e * mu_mix
            mix_var += w_e * var_mix
            Z += w_e
        mix_mu = mix_mu / max(Z, 1e-12)
        mix_var = mix_var / max(Z, 1e-12)
        if dy == 1:
            mix_mu, mix_var = mix_mu.squeeze(0), mix_var.squeeze(0)
        return {"mu": mix_mu, "var": mix_var}

