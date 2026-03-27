# =============================================================
# run_synthetic_mixed_thetaCmp.py
# Mixed synthetic experiment: gradual drift + sudden changes
# python -m calib.run_synthetic_mixed_thetaCmp --preview_only --out_dir figs/mixed_preview
# python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
# python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir figs/mixed_ablation


# =============================================================

import math
import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from time import time

from .configs import CalibrationConfig, BOCPDConfig, ModelConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *
from .restart_bocpd_debug_260115_gpytorch import RollingStats
from .restart_bocpd_ogp_gpytorch import BOCPD_OGP, OGPPFConfig, OGPParticleFilter, make_fast_batched_grad_func
from .run_synthetic_slope_deltaCmp import (
    build_phi2_from_theta_star,
    computer_model_config2_np,
    computer_model_config2_torch,
    oracle_theta,
)
from .run_synthetic_suddenCmp_tryThm import physical_system, PFWithGPPrediction, KOHSlidingWindow, _aggregate_ogp_particles


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size > 0 else float("nan")


def _theta_var_from_others(others_hist, n: int) -> np.ndarray:
    vals = []
    for item in others_hist[:n]:
        if isinstance(item, dict):
            vals.append(float(item.get("var", float("nan"))))
        else:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def _summarize_mixed_result(data: dict) -> dict:
    theta = np.asarray(data.get("theta", []), dtype=float)
    theta_oracle = np.asarray(data.get("theta_oracle", []), dtype=float)
    theta_var = np.asarray(data.get("theta_var", []), dtype=float)
    rmse = np.asarray(data.get("rmse", []), dtype=float)
    crps_hist = np.asarray(data.get("crps_hist", []), dtype=float)
    n = min(len(theta), len(theta_oracle), len(theta_var))
    if n == 0:
        theta_rmse = float("nan")
        theta_crps = float("nan")
    else:
        theta_rmse = float(np.sqrt(np.mean((theta[:n] - theta_oracle[:n]) ** 2)))
        theta_crps = float(
            crps_gaussian(
                torch.tensor(theta[:n], dtype=torch.float64),
                torch.tensor(np.clip(theta_var[:n], 1e-12, None), dtype=torch.float64),
                torch.tensor(theta_oracle[:n], dtype=torch.float64),
            ).mean().item()
        )
    return dict(
        theta_rmse=theta_rmse,
        theta_crps=theta_crps,
        y_rmse=_finite_mean(rmse),
        y_crps=_finite_mean(crps_hist),
    )


def _summarize_restart_events(data: dict) -> dict:
    rm = list(data.get("restart_mode_hist", []))
    cp_times = list(data.get("cp_times", []))
    batch_size = int(data.get("batch_size", 1))
    cp_batches = [int(cp // batch_size) for cp in cp_times]
    corrective_modes = {"full", "delta_only", "standardized_gate_refresh", "cusum_refresh"}
    full_count = sum(1 for v in rm if v == "full")
    delta_count = sum(1 for v in rm if v == "delta_only")
    gate_count = sum(1 for v in rm if v in ("standardized_gate_refresh", "cusum_refresh"))
    tolerated_batches = set()
    for cpb in cp_batches:
        tolerated_batches.update({cpb, cpb + 1})
    false_full = sum(1 for idx, v in enumerate(rm) if v == "full" and idx not in tolerated_batches)
    delays = []
    for cpb in cp_batches:
        found = None
        for idx, mode in enumerate(rm):
            if idx >= cpb and mode in corrective_modes:
                found = idx
                break
        delays.append(float(found - cpb) if found is not None else float("nan"))
    return dict(
        full_restart_count=float(full_count),
        delta_only_count=float(delta_count),
        gate_refresh_count=float(gate_count),
        false_full_restart_count=float(false_full),
        post_change_correction_delay=_finite_mean(delays),
    )


def _smoothed_theta_noise(rng: np.random.RandomState, num_batches: int, noise_scale: float, rho: float = 0.65) -> np.ndarray:
    eps = np.zeros(num_batches, dtype=float)
    for k in range(1, num_batches):
        eps[k] = rho * eps[k - 1] + rng.randn() * noise_scale
    return eps


def build_mixed_theta_trajectory(total_T: int, batch_size: int, seed: int, theta_range=(1.0, 2.5), drift_scale=0.008, jump_scale=0.35, theta_noise_sd=0.015) -> dict:
    if total_T % batch_size != 0:
        raise ValueError(f"total_T ({total_T}) must be divisible by batch_size ({batch_size})")
    rng = np.random.RandomState(int(seed))
    num_batches = total_T // batch_size
    cpb1 = int(np.clip(round(num_batches * (0.33 + rng.uniform(-0.04, 0.04))), 4, num_batches - 7))
    cpb2 = int(np.clip(round(num_batches * (0.70 + rng.uniform(-0.04, 0.04))), cpb1 + 4, num_batches - 3))
    cp_batches = [cpb1, cpb2]
    theta0 = rng.uniform(1.20, 1.55)
    drift_sign_1 = 1.0 if rng.rand() < 0.6 else -1.0
    drift_sign_2 = -drift_sign_1 if rng.rand() < 0.85 else drift_sign_1
    drift_sign_3 = -drift_sign_2 if rng.rand() < 0.75 else drift_sign_2
    seg_drifts = np.array([
        drift_sign_1 * rng.uniform(0.75, 1.20) * drift_scale,
        drift_sign_2 * rng.uniform(0.55, 1.00) * drift_scale,
        drift_sign_3 * rng.uniform(0.60, 1.10) * drift_scale,
    ], dtype=float)
    jump1_sign = 1.0 if rng.rand() < 0.5 else -1.0
    jump2_sign = -jump1_sign if rng.rand() < 0.80 else jump1_sign
    jumps = np.array([
        jump1_sign * rng.uniform(0.85, 1.15) * jump_scale,
        jump2_sign * rng.uniform(0.70, 1.00) * jump_scale,
    ], dtype=float)
    noise = _smoothed_theta_noise(rng, num_batches, theta_noise_sd)
    theta = np.zeros(num_batches, dtype=float)
    theta[0] = theta0
    jump_map = {cp_batches[0]: jumps[0], cp_batches[1]: jumps[1]}
    for b in range(1, num_batches):
        seg_prev = 0 if (b - 1) < cp_batches[0] else (1 if (b - 1) < cp_batches[1] else 2)
        theta[b] = theta[b - 1] + seg_drifts[seg_prev] + noise[b] + jump_map.get(b, 0.0)
    lo, hi = theta_range
    target_center = rng.uniform(1.55, 1.95)
    raw_center = float(theta.mean())
    raw_halfspan = float(np.max(np.abs(theta - raw_center)))
    allowed_halfspan = 0.48 * (hi - lo)
    scale = 1.0 if raw_halfspan < 1e-8 else min(1.0, allowed_halfspan / raw_halfspan)
    theta = target_center + scale * (theta - raw_center)
    min_jump = max(0.22, 0.65 * jump_scale)
    for idx, cpb in enumerate(cp_batches):
        observed = theta[cpb] - theta[cpb - 1]
        if abs(observed) < min_jump:
            theta[cpb:] += np.sign(jumps[idx]) * (min_jump - abs(observed))
    theta = np.clip(theta, lo + 0.02, hi - 0.02)
    seg_ids = np.zeros(num_batches, dtype=int)
    seg_ids[cp_batches[0]:cp_batches[1]] = 1
    seg_ids[cp_batches[1]:] = 2
    return dict(theta_batches=theta, cp_batches=cp_batches, cp_times=[b * batch_size for b in cp_batches], seg_ids=seg_ids, seg_drifts=seg_drifts, jumps=jumps)


class MixedThetaDataStream:
    def __init__(self, total_T: int, batch_size: int, noise_sd: float, phi2_of_theta, drift_scale: float, jump_scale: float, theta_noise_sd: float, seed: int, phi_base=np.array([5.0, 0.0, 5.0])):
        self.T = int(total_T)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)
        self.phi2_of_theta = phi2_of_theta
        self.phi_base = np.asarray(phi_base, dtype=float).copy()
        self.rng = np.random.RandomState(int(seed))
        self.t = 0
        self.spec = build_mixed_theta_trajectory(total_T=self.T, batch_size=self.bs, seed=int(seed), drift_scale=drift_scale, jump_scale=jump_scale, theta_noise_sd=theta_noise_sd)
        self.theta_batches = self.spec["theta_batches"]
        self.cp_batches = list(self.spec["cp_batches"])
        self.cp_times = list(self.spec["cp_times"])
        self.seg_ids = np.asarray(self.spec["seg_ids"], dtype=int)
        self.theta_star_history = []
        self.phi_history = []
        self.seg_history = []

    def _batch_idx(self, t: int) -> int:
        return min(int(t // self.bs), len(self.theta_batches) - 1)

    def true_theta_star(self, t: int) -> float:
        return float(self.theta_batches[self._batch_idx(t)])

    def true_phi(self, t: int) -> np.ndarray:
        phi = self.phi_base.copy()
        phi[1] = float(self.phi2_of_theta(self.true_theta_star(t)))
        return phi

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.t >= self.T:
            raise StopIteration
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]
        self.rng.shuffle(X)
        phi_t = self.true_phi(self.t)
        theta_star_t = self.true_theta_star(self.t)
        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)
        self.theta_star_history.append(theta_star_t)
        self.phi_history.append(phi_t.copy())
        self.seg_history.append(int(self.seg_ids[self._batch_idx(self.t)]))
        self.t += self.bs
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def preview_mixed_theta_paths(seeds: List[int], total_T: int, batch_size: int, drift_scale: float, jump_scale: float, theta_noise_sd: float, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(11, 5.5))
    rows = []
    for seed in seeds:
        spec = build_mixed_theta_trajectory(total_T=total_T, batch_size=batch_size, seed=int(seed), drift_scale=drift_scale, jump_scale=jump_scale, theta_noise_sd=theta_noise_sd)
        theta = spec["theta_batches"]
        x = np.arange(len(theta))
        plt.plot(x, theta, lw=2, alpha=0.95, label=f"seed={seed}")
        for cpb in spec["cp_batches"]:
            plt.axvline(cpb, color="red", linestyle="--", alpha=0.20)
        for b, th in enumerate(theta):
            rows.append({"seed": int(seed), "batch_idx": int(b), "obs_t": int(b * batch_size), "theta_true": float(th), "is_cp_batch": int(b in spec["cp_batches"])})
    plt.ylim(0.95, 2.55)
    plt.xlabel("batch index")
    plt.ylabel("true theta*")
    plt.title(f"Preview of mixed true theta trajectories (drift_scale={drift_scale:.4f}, jump_scale={jump_scale:.3f})")
    plt.legend(loc="best")
    plt.tight_layout()
    png_path = os.path.join(save_dir, "mixed_theta_preview.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "mixed_theta_preview.csv"), index=False)
    return png_path


def plot_theta_tracking(res: Dict, oracle_hist: np.ndarray, theta_true_hist: np.ndarray, cp_times: List[int], batch_size: int, title: str, save_path: str):
    plt.figure(figsize=(12, 5))
    for name, d in res.items():
        plt.plot(d["theta"], label=name, alpha=0.9)
    plt.plot(np.asarray(oracle_hist), "k--", lw=2, label="oracle theta*")
    plt.plot(np.asarray(theta_true_hist), color="tab:orange", linestyle=":", lw=2, label="true theta*")
    for c in cp_times:
        plt.axvline(x=c // batch_size, color="red", linestyle="--", alpha=0.35)
    plt.title(title)
    plt.xlabel("batch index")
    plt.ylabel("theta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_one_mixed(drift_scale: float, jump_scale: float, methods: Dict, batch_size: int, seed: int, total_T: int, phi2_of_theta, noise_sd: float = 0.2, theta_noise_sd: float = 0.015):
    print(f"\n=== Mixed experiment: drift_scale={drift_scale:.4f}, jump_scale={jump_scale:.3f}, bs={batch_size}, seed={seed} ===")
    theta_grid = np.linspace(0, 3, 600)
    stream_ref = MixedThetaDataStream(total_T=total_T, batch_size=batch_size, noise_sd=noise_sd, phi2_of_theta=phi2_of_theta, drift_scale=drift_scale, jump_scale=jump_scale, theta_noise_sd=theta_noise_sd, seed=seed)
    cp_times = stream_ref.cp_times
    print(f"    cp_times={cp_times}, true-jumps={stream_ref.spec['jumps']}")

    def prior_sampler(N: int):
        return torch.rand(N, 1) * 3.0

    results = {}
    for name, meta in methods.items():
        print(f"  -> {name}")
        t0 = time()
        theta_hist, rmse_hist, others_hist, top0_particles_hist, crps_hist = [], [], [], [], []
        restart_mode_hist = []
        total_obs = 0
        stream = MixedThetaDataStream(total_T=total_T, batch_size=batch_size, noise_sd=noise_sd, phi2_of_theta=phi2_of_theta, drift_scale=drift_scale, jump_scale=jump_scale, theta_noise_sd=theta_noise_sd, seed=seed)

        if name == "R-BOCPD-PF-OGP":
            emulator = DeterministicSimulator(func=computer_model_config2_torch, enable_autograd=True)
            grad_func = make_fast_batched_grad_func(computer_model_config2_torch, device="cuda", dtype=torch.float64)
            ogp_cfg = OGPPFConfig(num_particles=1024, x_domain=(0.0, 1.0), theta_lo=torch.tensor([0.0]), theta_hi=torch.tensor([3.0]), particle_chunk_size=256)
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            bocpd = BOCPD_OGP(config=bocpd_cfg, ogp_pf_cfg=ogp_cfg, batched_grad_func=grad_func, device="cuda")
            while total_obs < total_T:
                Xb, Yb = stream.next()
                dev = bocpd.device
                Xb64 = Xb.to(device=dev, dtype=torch.float64)
                Yb64 = Yb.to(device=dev, dtype=torch.float64)
                if total_obs > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(batch_size, device=dev, dtype=torch.float64)
                    mix_var = torch.zeros(batch_size, device=dev, dtype=torch.float64)
                    Z = 0.0
                    for e in bocpd.experts:
                        w_e = math.exp(e.log_mass)
                        e_X_hist = e.X_hist if e.X_hist.numel() > 0 else None
                        e_y_hist = e.y_hist if e.y_hist.numel() > 0 else None
                        mu_e, var_e = e.pf.predict_batch(Xb64, e_X_hist, e_y_hist, emulator, model_cfg.rho, model_cfg.sigma_eps)
                        mix_mu += w_e * mu_e
                        mix_var += w_e * var_e
                        Z += w_e
                    mix_mu = mix_mu / max(Z, 1e-12)
                    mix_var = mix_var / max(Z, 1e-12)
                    rmse_hist.append(float(torch.sqrt(((mix_mu.cpu() - Yb64.cpu()) ** 2).mean())))
                    crps_hist.append(float(crps_gaussian(mix_mu.cpu(), mix_var.cpu(), Yb64.cpu()).mean().item()))
                rec = bocpd.update_batch(Xb64, Yb64, emulator, model_cfg, None, prior_sampler, verbose=False)
                mean_theta, var_theta, lo_theta, hi_theta = _aggregate_ogp_particles(bocpd, 0.9)
                theta_hist.append(float(mean_theta[0]))
                batch_particles, batch_weights, batch_logmass = [], [], []
                for e in bocpd.experts:
                    batch_particles.append(e.pf.theta.squeeze(-1).detach().cpu())
                    batch_weights.append(e.pf.weights().detach().cpu())
                    batch_logmass.append(float(e.log_mass))
                top0_particles_hist.append(dict(particles=batch_particles, weights=batch_weights, log_mass=torch.tensor(batch_logmass)))
                others_hist.append(dict(did_restart=rec["did_restart"], var=float(var_theta[0]), lo=float(lo_theta[0]), hi=float(hi_theta[0]), seg_id=int(stream.seg_history[-1]), t=int(total_obs), pf_info=rec["pf_diags"]))
                total_obs += batch_size

        elif name == "PF-OGP" or meta.get("type") == "pf_ogp":
            emulator = DeterministicSimulator(func=computer_model_config2_torch, enable_autograd=True)
            pf_grad_func = make_fast_batched_grad_func(computer_model_config2_torch, device="cuda", dtype=torch.float64)
            pf_ogp_cfg = OGPPFConfig(num_particles=1024, x_domain=(0.0, 1.0), theta_lo=torch.tensor([0.0]), theta_hi=torch.tensor([3.0]), theta_move_std=0.02, particle_chunk_size=256)
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            ogp_dev = "cuda"
            pf = OGPParticleFilter(ogp_cfg=pf_ogp_cfg, prior_sampler=prior_sampler, batched_grad_func=pf_grad_func, device=ogp_dev, dtype=torch.float64)
            pf_X_hist = torch.empty(0, 1, dtype=torch.float64, device=ogp_dev)
            pf_y_hist = torch.empty(0, dtype=torch.float64, device=ogp_dev)
            pf_ogp_max_hist = 200
            while total_obs < total_T:
                Xb, Yb = stream.next()
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)
                if total_obs > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(Xb64, pf_Xh, pf_yh, emulator, pf_model_cfg.rho, pf_model_cfg.sigma_eps)
                    rmse_hist.append(float(torch.sqrt(((mu_mix.cpu() - Yb64.cpu()) ** 2).mean())))
                    crps_hist.append(float(crps_gaussian(mu_mix.cpu(), var_mix.cpu(), Yb64.cpu()).mean().item()))
                pf.step_batch(Xb64, Yb64, pf_X_hist if pf_X_hist.numel() > 0 else None, pf_y_hist if pf_y_hist.numel() > 0 else None, emulator, pf_model_cfg.rho, pf_model_cfg.sigma_eps)
                if pf_X_hist.numel() == 0:
                    pf_X_hist = Xb64.clone()
                    pf_y_hist = Yb64.clone()
                else:
                    pf_X_hist = torch.cat([pf_X_hist, Xb64], dim=0)
                    pf_y_hist = torch.cat([pf_y_hist, Yb64], dim=0)
                if pf_X_hist.shape[0] > pf_ogp_max_hist:
                    pf_X_hist = pf_X_hist[-pf_ogp_max_hist:]
                    pf_y_hist = pf_y_hist[-pf_ogp_max_hist:]
                w = pf.weights().view(-1, 1)
                mean_theta = (w * pf.theta).sum(dim=0)
                theta_hist.append(float(mean_theta[0]))
                top0_particles_hist.append(dict(particles=[pf.theta.squeeze(-1).detach().cpu()], weights=[pf.weights().detach().cpu()], log_mass=torch.tensor([0.0])))
                others_hist.append(dict(did_restart=False, var=float((w * (pf.theta - mean_theta).pow(2)).sum(dim=0)[0]), seg_id=int(stream.seg_history[-1]), t=int(total_obs)))
                total_obs += batch_size

        elif meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True
            cfg.bocpd.restart_impl = meta.get("restart_impl", "debug_260115")
            cfg.bocpd.hybrid_partial_restart = bool(meta.get("use_dual_restart", meta.get("hybrid_partial_restart", True)))
            cfg.bocpd.hybrid_tau_delta = float(meta.get("hybrid_tau_delta", 0.05))
            cfg.bocpd.hybrid_tau_theta = float(meta.get("hybrid_tau_theta", 0.05))
            cfg.bocpd.hybrid_tau_full = float(meta.get("hybrid_tau_full", 0.05))
            cfg.bocpd.hybrid_delta_share_rho = float(meta.get("hybrid_delta_share_rho", 0.75))
            cfg.bocpd.hybrid_pf_sigma_mode = str(meta.get("hybrid_pf_sigma_mode", "fixed"))
            cfg.bocpd.hybrid_sigma_delta_alpha = float(meta.get("hybrid_sigma_delta_alpha", 1.0))
            cfg.bocpd.hybrid_sigma_ema_beta = float(meta.get("hybrid_sigma_ema_beta", 0.98))
            cfg.bocpd.hybrid_sigma_min = float(meta.get("hybrid_sigma_min", 1e-4))
            cfg.bocpd.hybrid_sigma_max = float(meta.get("hybrid_sigma_max", 10.0))
            cfg.bocpd.use_cusum = bool(meta.get("use_cusum", False))
            cfg.bocpd.cusum_threshold = float(meta.get("cusum_threshold", 10.0))
            cfg.bocpd.cusum_recent_obs = int(meta.get("cusum_recent_obs", 20))
            cfg.bocpd.cusum_cov_eps = float(meta.get("cusum_cov_eps", 1e-6))
            cfg.bocpd.cusum_mode = str(meta.get("cusum_mode", "cumulative"))
            cfg.bocpd.standardized_gate_threshold = float(meta.get("standardized_gate_threshold", 3.0))
            cfg.bocpd.standardized_gate_consecutive = int(meta.get("standardized_gate_consecutive", 1))
            cfg.bocpd.particle_delta_mode = str(meta.get("particle_delta_mode", "shared_gp"))
            cfg.bocpd.particle_gp_hyper_candidates = meta.get("particle_gp_hyper_candidates", None)
            cfg.bocpd.particle_basis_kind = str(meta.get("particle_basis_kind", "rbf"))
            cfg.bocpd.particle_basis_num_features = int(meta.get("particle_basis_num_features", 8))
            cfg.bocpd.particle_basis_lengthscale = float(meta.get("particle_basis_lengthscale", 0.25))
            cfg.bocpd.particle_basis_ridge = float(meta.get("particle_basis_ridge", 1e-2))
            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]
                cfg.model.bocpd_use_discrepancy = meta.get("bocpd_use_discrepancy", cfg.model.use_discrepancy)
            emulator = DeterministicSimulator(func=computer_model_config2_torch, enable_autograd=True)
            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
            roll = RollingStats(window=50)
            while total_obs < total_T:
                if total_obs % (5 * batch_size) == 0:
                    print(f"     {name}: total_obs={total_obs}")
                Xb, Yb = stream.next()
                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    rmse_hist.append(float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean())))
                    crps_hist.append(float(crps_gaussian(pred["mu"], pred["var"], Yb).mean().item()))
                rec = calib.step_batch(Xb, Yb, verbose=False)
                rm = rec.get("restart_mode", None)
                if rm is None:
                    rm = "full" if bool(rec.get("did_restart", False)) else "none"
                restart_mode_hist.append(rm)
                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)
                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
                theta_hist.append(float(mean_theta[0]))
                batch_particles, batch_weights, batch_logmass = [], [], []
                for e in calib.bocpd.experts:
                    batch_particles.append(e.pf.particles.theta.squeeze(-1).detach().cpu())
                    batch_weights.append(e.pf.particles.weights().squeeze(-1).detach().cpu())
                    batch_logmass.append(float(e.log_mass))
                top0_particles_hist.append(dict(particles=batch_particles, weights=batch_weights, log_mass=torch.tensor(batch_logmass)))
                others_hist.append(dict(did_restart=bool(rec.get("did_restart", False)), var=float(var_theta[0]), lo=float(lo_theta[0]), hi=float(hi_theta[0]), seg_id=int(stream.seg_history[-1]), t=int(total_obs), pf_info=rec["pf_diags"], delta_ll_pair=dll, mu_hat=roll.mean(), sigma_hat=roll.std(), h_log=rec.get("h_log", None), log_odds_mass=rec.get("log_odds_mass", None), anchor_rl=rec.get("anchor_rl", None), cand_rl=rec.get("cand_rl", None)))
                total_obs += batch_size
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            bpc = None
            while total_obs < total_T:
                if total_obs % (5 * batch_size) == 0:
                    print(f"     {name}: total_obs={total_obs}")
                Xb, Yb = stream.next()
                crps_sim = None
                if X_hist is None:
                    X_hist, y_hist = Xb.numpy(), Yb.numpy()
                else:
                    X_hist = np.concatenate([X_hist, Xb.numpy()], axis=0)
                    y_hist = np.concatenate([y_hist, Yb.numpy()], axis=0)
                if X_hist.shape[0] >= W:
                    X_hist = X_hist[-W:]
                    y_hist = y_hist[-W:]
                if total_obs > 0 and bpc is not None:
                    mu_np, var_np = bpc.predict(Xb.detach().cpu().numpy())
                    mu_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device)
                    var_t = torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(float(crps_sim.mean().item()))
                bpc = BayesianProjectedCalibration(theta_lo=np.array([0.0]), theta_hi=np.array([3.0]), noise_var=float(noise_sd ** 2), y_sim=computer_model_config2_np)
                X_grid = np.linspace(0, 1, 400).reshape(-1, 1)
                bpc.fit(X_hist, y_hist, X_grid, n_eta_draws=500, n_restart=10, gp_fit_iters=200)
                theta_hist.append(float(bpc.theta_mean[0]))
                others_hist.append(dict(did_restart=False, var=float(bpc.theta_var[0]) if bpc.theta_var is not None else float("nan"), lo=float("nan"), hi=float("nan"), seg_id=int(stream.seg_history[-1]), t=int(total_obs), entropy=bpc.entropy_theta(), crps_sim=crps_sim))
                theta_samples_bpc = torch.tensor(bpc.theta_samples).squeeze(-1)
                top0_particles_hist.append(dict(particles=[theta_samples_bpc], weights=None, log_mass=torch.tensor([0.0])))
                total_obs += batch_size

        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(theta_lo=np.array([0.0]), theta_hi=np.array([3.0]), noise_var=float(noise_sd ** 2), y_sim=computer_model_config2_np, X_grid=np.linspace(0, 1, 400).reshape(-1, 1), **meta.get("params", {}))
            while total_obs < total_T:
                if total_obs % (5 * batch_size) == 0:
                    print(f"     {name}: total_obs={total_obs}")
                Xb, Yb = stream.next()
                crps_sim = None
                if total_obs > 0:
                    mu, var = calib.predict(Xb)
                    mu_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device)
                    var_t = torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(float(crps_sim.mean().item()))
                info = calib.step_batch(Xb.detach().cpu().numpy(), Yb.detach().cpu().numpy())
                theta_mean, theta_var, theta_lo, theta_hi = calib._aggregate_particles(0.9)
                theta_hist.append(float(theta_mean[0]))
                try:
                    v0 = float(theta_var[0][0]) if np.ndim(theta_var) >= 1 else float(theta_var)
                except Exception:
                    v0 = float(theta_var)
                others_hist.append(dict(did_restart=bool(info.get("did_restart", False)), var=v0, lo=float(theta_lo[0]) if np.ndim(theta_lo) >= 1 else float(theta_lo), hi=float(theta_hi[0]) if np.ndim(theta_hi) >= 1 else float(theta_hi), seg_id=int(stream.seg_history[-1]), t=int(total_obs), crps_sim=crps_sim))
                batch_particles, batch_logmass = [], []
                for e in calib.experts:
                    batch_particles.append(torch.tensor(e.bpc.theta_samples).squeeze(-1))
                    batch_logmass.append(e.logw)
                top0_particles_hist.append(dict(particles=batch_particles, weights=[], log_mass=torch.tensor(batch_logmass)))
                total_obs += batch_size

        elif name == "DA" or meta.get("type") == "da":
            da = PFWithGPPrediction(sim_func_np=computer_model_config2_np, n_particles=1024, theta_lo=0.0, theta_hi=3.0, sigma_obs=noise_sd, resample_ess_ratio=0.5, theta_move_std=0.05, window_size=80, gp_lengthscale=0.3, gp_signal_var=1.0, seed=seed)
            while total_obs < total_T:
                Xb, Yb = stream.next()
                Xb_np, Yb_np = Xb.numpy(), Yb.numpy()
                if total_obs > 0:
                    mu_pred, var_pred = da.predict(Xb_np)
                    mu_t = torch.tensor(mu_pred, dtype=Yb.dtype)
                    var_t = torch.tensor(var_pred, dtype=Yb.dtype)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_hist.append(float(crps_gaussian(mu_t, var_t, Yb).mean().item()))
                da.update_batch(Xb_np, Yb_np)
                theta_hist.append(da.mean_theta())
                top0_particles_hist.append(dict(particles=[torch.tensor(da.theta.copy())], weights=[torch.tensor(np.exp(da.logw.copy()))], log_mass=torch.tensor([0.0])))
                others_hist.append(dict(did_restart=False, var=0.0, seg_id=int(stream.seg_history[-1]), t=int(total_obs)))
                total_obs += batch_size

        elif name == "BC" or meta.get("type") == "bc":
            bc = KOHSlidingWindow(sim_func_np=computer_model_config2_np, theta_grid=np.linspace(0.0, 3.0, 200), window_size=80, sigma_obs=noise_sd, gp_lengthscale=0.3, gp_signal_var=1.0)
            while total_obs < total_T:
                Xb, Yb = stream.next()
                Xb_np, Yb_np = Xb.numpy(), Yb.numpy()
                if total_obs > 0:
                    mu_pred, var_pred = bc.predict(Xb_np)
                    mu_t = torch.tensor(mu_pred, dtype=Yb.dtype)
                    var_t = torch.tensor(var_pred, dtype=Yb.dtype)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_hist.append(float(crps_gaussian(mu_t, var_t, Yb).mean().item()))
                bc.update_batch(Xb_np, Yb_np)
                theta_hist.append(bc.mean_theta())
                top0_particles_hist.append(dict(particles=[], weights=[], log_mass=torch.tensor([0.0])))
                others_hist.append(dict(did_restart=False, var=0.0, seg_id=int(stream.seg_history[-1]), t=int(total_obs)))
                total_obs += batch_size

        else:
            raise ValueError(f"Unsupported method in mixed runner: {name} / {meta}")

        K = len(theta_hist)
        while len(stream_ref.phi_history) < K:
            stream_ref.next()
        phi_hist = stream_ref.phi_history[:K]
        oracle_hist = [oracle_theta(phi, theta_grid) for phi in phi_hist]
        theta_true_hist = np.asarray(stream_ref.theta_star_history[:K], dtype=float)
        results[name] = dict(theta=np.asarray(theta_hist, dtype=float), theta_oracle=np.asarray(oracle_hist, dtype=float), theta_star_true=theta_true_hist, theta_var=_theta_var_from_others(others_hist, len(theta_hist)), others=others_hist, rmse=np.asarray(rmse_hist, dtype=float), cp_times=cp_times, cp_batches=list(stream_ref.cp_batches), drift_scale=float(drift_scale), jump_scale=float(jump_scale), batch_size=int(batch_size), seed=int(seed), top0_particles_hist=top0_particles_hist, crps_hist=np.asarray(crps_hist, dtype=float), restart_mode_hist=restart_mode_hist, traj_spec=stream_ref.spec)
        print(f"     done in {time() - t0:.1f}s")

    first_key = list(results.keys())[0]
    return results, stream_ref.phi_history[: len(results[first_key]["theta"])], results[first_key]["theta_oracle"], results[first_key]["theta_star_true"], cp_times


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/mixed_thetaCmp")
    parser.add_argument("--profile", type=str, default="main", choices=["main", "ablation", "preview"])
    parser.add_argument("--preview_only", action="store_true", default=False)
    parser.add_argument("--preview_seeds", type=int, nargs="*", default=[101, 202, 303, 404])
    parser.add_argument("--preview_total_T", type=int, default=600)
    parser.add_argument("--preview_batch_size", type=int, default=20)
    parser.add_argument("--preview_drift_scale", type=float, default=0.008)
    parser.add_argument("--preview_jump_scale", type=float, default=0.35)
    parser.add_argument("--preview_theta_noise_sd", type=float, default=0.015)
    args = parser.parse_args()
    store_dir = args.out_dir
    os.makedirs(store_dir, exist_ok=True)

    if args.preview_only or args.profile == "preview":
        png_path = preview_mixed_theta_paths(args.preview_seeds, args.preview_total_T, args.preview_batch_size, args.preview_drift_scale, args.preview_jump_scale, args.preview_theta_noise_sd, store_dir)
        print(f"[Saved] {png_path}")
        print("Preview finished. Inspect the theta trajectories before launching full runs.")
        return

    if args.profile == "ablation":
        seeds, batch_sizes, drift_scales, jump_scales, total_T = [101, 202, 303, 404, 505], [20], [0.008], [0.35], 600
    elif args.debug:
        seeds, batch_sizes, drift_scales, jump_scales, total_T = [456], [20], [0.008], [0.35], 300
    else:
        seeds, batch_sizes, drift_scales, jump_scales, total_T = [101, 202, 303, 404, 505], [20], [0.009], [0.28, 0.38], 600

    phi2_of_theta, _ = build_phi2_from_theta_star(phi2_grid=np.linspace(3.0, 12.0, 300), theta_grid=np.linspace(0.0, 3.0, 600))

    if args.profile == "ablation":
        methods = {
            "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
            "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
            "R-BOCPD-PF-halfdiscrepancy-hybrid-rolled": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="hybrid_260319", hybrid_partial_restart=True, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            "RBOCPD_half_STDGate": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=False, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            "RBOCPD_half_STDGate_dual": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=True, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            "R-BOCPD-PF-OGP": dict(type="bocpd", mode="restart"),
        }
    else:
        methods = {
            "PF-OGP": dict(type="pf_ogp"),
            "DA": dict(type="da"),
            "BC": dict(type="bc"),
            "BPC-80": dict(type="bpc"),
            "BOCPD-BPC": dict(type="bpc_bocpd"),
            "BOCPD-PF": dict(type="bocpd", mode="standard"),
            "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
            "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
            # "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
            "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="hybrid_260319", hybrid_partial_restart=False, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            "RBOCPD_half_STDGate": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=False, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            "RBOCPD_half_STDGate_dual": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=True, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, hybrid_tau_full=0.05, hybrid_delta_share_rho=0.75, hybrid_pf_sigma_mode="fixed", hybrid_sigma_delta_alpha=1.0, hybrid_sigma_ema_beta=0.98, hybrid_sigma_min=1e-4, hybrid_sigma_max=10.0),
            # "RBOCPD_half_STDGate_particleGP": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=False, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, particle_delta_mode="particle_gp_shared_hyper"),
            # "RBOCPD_half_STDGate_particleBasis": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True, restart_impl="rolled_cusum_260324", use_dual_restart=False, use_cusum=True, cusum_mode="standardized_gate", standardized_gate_threshold=3.0, standardized_gate_consecutive=1, cusum_recent_obs=20, hybrid_tau_delta=0.05, hybrid_tau_theta=0.05, particle_delta_mode="particle_basis", particle_basis_kind="rbf", particle_basis_num_features=8, particle_basis_lengthscale=0.25, particle_basis_ridge=1e-2),
            "R-BOCPD-PF-OGP": dict(type="bocpd", mode="restart"),
        }

    all_metrics, restart_mode_rows, restart_event_rows = [], [], []
    for drift_scale, jump_scale, batch_size, seed in itertools.product(drift_scales, jump_scales, batch_sizes, seeds):
        res, phi_hist, oracle_hist, theta_true_hist, cp_times = run_one_mixed(drift_scale, jump_scale, methods, batch_size, seed, total_T, phi2_of_theta)
        tag = f"drift{drift_scale:.4f}_jump{jump_scale:.3f}_bs{batch_size}_seed{seed}"
        torch.save(res, os.path.join(store_dir, f"mixed_{tag}_results.pt"))
        torch.save(dict(phi_hist=phi_hist, oracle_hist=oracle_hist, theta_true_hist=theta_true_hist), os.path.join(store_dir, f"mixed_{tag}_phi_oracle.pt"))
        plot_theta_tracking(res, oracle_hist, theta_true_hist, cp_times, batch_size, f"Mixed theta tracking (drift={drift_scale:.4f}, jump={jump_scale:.3f}, bs={batch_size}, seed={seed})", os.path.join(store_dir, f"mixed_{tag}_theta.png"))
        for method_name, data in res.items():
            metrics = _summarize_mixed_result(data)
            all_metrics.append({"method": method_name, "drift_scale": drift_scale, "jump_scale": jump_scale, "batch_size": batch_size, "seed": seed, "theta_rmse": metrics["theta_rmse"], "theta_crps": metrics["theta_crps"], "y_rmse": metrics["y_rmse"], "y_crps": metrics["y_crps"]})
            if "restart_mode_hist" in data and len(data["restart_mode_hist"]) > 0:
                rm = data["restart_mode_hist"]
                n = len(rm)
                n_none = sum(1 for v in rm if v == "none")
                n_delta = sum(1 for v in rm if v == "delta_only")
                n_gate = sum(1 for v in rm if v in ("standardized_gate_refresh", "cusum_refresh"))
                n_full = sum(1 for v in rm if v == "full")
                restart_mode_rows.append({"method": method_name, "drift_scale": drift_scale, "jump_scale": jump_scale, "batch_size": batch_size, "seed": seed, "n_steps": n, "none_ratio": n_none / n, "delta_only_ratio": n_delta / n, "gate_refresh_ratio": n_gate / n, "full_ratio": n_full / n, "n_none": n_none, "n_delta_only": n_delta, "n_gate_refresh": n_gate, "n_full": n_full})
                restart_event_rows.append({"method": method_name, "drift_scale": drift_scale, "jump_scale": jump_scale, "batch_size": batch_size, "seed": seed, **_summarize_restart_events(data)})

    import pandas as pd
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(f"{store_dir}/all_metrics.csv", index=False)
    df_metrics.to_excel(f"{store_dir}/all_metrics.xlsx", index=False)
    if len(restart_mode_rows) > 0:
        pd.DataFrame(restart_mode_rows).to_csv(f"{store_dir}/restart_mode_stats.csv", index=False)
        pd.DataFrame(restart_mode_rows).to_excel(f"{store_dir}/restart_mode_stats.xlsx", index=False)
    df_restart_events = pd.DataFrame(restart_event_rows)
    if len(df_restart_events) > 0:
        df_restart_events.to_csv(f"{store_dir}/restart_event_stats.csv", index=False)
        df_restart_events.to_excel(f"{store_dir}/restart_event_stats.xlsx", index=False)

    print("\n" + "=" * 70)
    print("Average Metrics Across All Combinations (drift_scale x jump_scale x batch_sizes x seeds):")
    print("=" * 70)
    grouped = df_metrics.groupby("method").agg({"theta_rmse": ["mean", "std"], "theta_crps": ["mean", "std"], "y_rmse": ["mean", "std"], "y_crps": ["mean", "std"]})
    for method in df_metrics["method"].unique():
        stats = grouped.loc[method]
        print(f"\n{method}:")
        print(f"  theta_rmse: {stats[('theta_rmse', 'mean')]:.6f} +/- {stats[('theta_rmse', 'std')]:.6f}")
        print(f"  theta_crps: {stats[('theta_crps', 'mean')]:.6f} +/- {stats[('theta_crps', 'std')]:.6f}")
        print(f"  y_rmse:     {stats[('y_rmse', 'mean')]:.6f} +/- {stats[('y_rmse', 'std')]:.6f}")
        print(f"  y_crps:     {stats[('y_crps', 'mean')]:.6f} +/- {stats[('y_crps', 'std')]:.6f}")
    print("\n" + "=" * 70)
    print("All mixed-theta experiments finished.")
    print(f"Results saved to: {store_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
