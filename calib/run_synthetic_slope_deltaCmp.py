# =============================================================
# run_synthetic_slopecomp.py
# Gradual-drift slope comparison experiment
# =============================================================

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *
from .restart_bocpd_debug_260115_gpytorch import RollingStats
# from .restart_bocpd_ogp import (
#     BOCPD_OGP, OGPPFConfig,
#     RollingStats as OGPRollingStats,
#     make_grad_func_from_emulator,
# )
from .restart_bocpd_ogp_gpytorch import (
    BOCPD_OGP, OGPPFConfig, OGPParticleFilter,
    RollingStats as OGPRollingStats, make_fast_batched_grad_func,
)
from .configs import BOCPDConfig, ModelConfig

# -------------------------------------------------------------
# Simulator (Config2)
# -------------------------------------------------------------
def computer_model_config2_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    theta = np.atleast_2d(theta)
    th = theta[:, [0]]
    xx = x[:, [0]]
    return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)


def computer_model_config2_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    return torch.sin(5.0 * theta[:, 0:1] * x[:, 0:1]) + 5.0 * x[:, 0:1]

from scipy.interpolate import interp1d

def build_phi2_of_theta_interp(theta_grid: np.ndarray):
    """
    Build interpolation phi2(theta) by inverting oracle_theta on a phi2 grid.
    This mirrors the logic in your slope synthetic script: ensure theta*(t) corresponds
    to a realizable physical phi2(t).

    Returns: callable phi2_of_theta(theta) -> float
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # choose a phi2 grid (wide enough)
    phi2_grid = np.linspace(2.0, 12.0, 400)
    phi_base = np.array([5.0, 0.0, 5.0], dtype=float)

    # map phi2 -> theta*(phi)
    theta_star_list = []
    for phi2 in phi2_grid:
        phi = phi_base.copy()
        phi[1] = float(phi2)
        th = oracle_theta(phi, theta_grid)
        theta_star_list.append(th)

    theta_star_arr = np.asarray(theta_star_list, dtype=float)

    # theta_star_arr should be monotone-ish; if not, sort by theta for safe inversion
    order = np.argsort(theta_star_arr)
    theta_sorted = theta_star_arr[order]
    phi2_sorted = phi2_grid[order]

    # Invert by interpolation
    f = interp1d(theta_sorted, phi2_sorted, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return lambda th: float(f(float(th)))

def build_phi2_from_theta_star(
    phi2_grid: np.ndarray,
    theta_grid: np.ndarray,
    a1: float = 5.0,
    a3: float = 5.0,
):
    """
    构造 φ2 = f(θ*) 的插值函数
    """

    theta_star_vals = []

    for phi2 in phi2_grid:
        phi = np.array([a1, phi2, a3])
        theta_star = oracle_theta(phi, theta_grid)
        theta_star_vals.append(theta_star)

    theta_star_vals = np.asarray(theta_star_vals)

    phi2_of_theta = interp1d(
        theta_star_vals,
        phi2_grid,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return phi2_of_theta, theta_star_vals
# -------------------------------------------------------------
# True physical system η(x; φ(t))
# -------------------------------------------------------------
def physical_system(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    φ = [a1, a2, a3]
    η(x) = a1 * x * cos(a2 * x) + a3 * x
    """
    x = x.reshape(-1)
    a1, a2, a3 = phi
    return a1 * x * np.cos(a2 * x) + a3 * x


# -------------------------------------------------------------
# Data stream with explicit slope drift
# -------------------------------------------------------------
class SlopeDriftDataStream:
    def __init__(
        self,
        total_T: int = 800,
        batch_size: int = 20,
        noise_sd: float = 0.2,
        slope: float = 0.002,
        phi0 = np.array([5.0, 5.0, 5.0]),
        seed: int = 0,
    ):
        self.T = total_T
        self.bs = batch_size
        self.noise_sd = noise_sd
        self.slope = slope
        self.phi0 = phi0
        self.rng = np.random.RandomState(seed)

        self.t = 0
        self.phi_history = []

    def true_phi(self, t: int) -> np.ndarray:
        phi = self.phi0.copy()
        phi[1] = self.phi0[1] + self.slope * t
        return phi

    def next(self):
        if self.t >= self.T:
            raise StopIteration

        # X = self.rng.rand(self.bs, 1)
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs     # 每个区间一个点
        X = u[:, None]
        # self.rng.shuffle(X)  # 可选

        phi_t = self.true_phi(self.t)
        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.t += self.bs

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

class ThetaDrivenSlopeDataStream:
    """
    Ground truth: θ*(t) 线性变化
    Physical parameter φ2(t) 由数值反推得到
    """

    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        theta0: float,
        theta_slope: float,
        phi2_of_theta,           # 上一步构造的插值函数
        phi_base = np.array([5.0, 0.0, 5.0]),
        seed: int = 0,
    ):
        self.T = total_T
        self.bs = batch_size
        self.noise_sd = noise_sd
        self.theta0 = theta0
        self.theta_slope = theta_slope
        self.phi_base = phi_base.copy()
        self.phi2_of_theta = phi2_of_theta

        self.rng = np.random.RandomState(seed)
        self.t = 0

        self.theta_star_history = []
        self.phi_history = []

    def true_theta_star(self, t: int) -> float:
        return self.theta0 + self.theta_slope * t

    def true_phi(self, t: int) -> np.ndarray:
        theta_star = self.true_theta_star(t)
        phi = self.phi_base.copy()
        phi[1] = float(self.phi2_of_theta(theta_star))
        return phi

    def next(self):
        if self.t >= self.T:
            raise StopIteration

        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]

        phi_t = self.true_phi(self.t)
        theta_star_t = self.true_theta_star(self.t)

        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.theta_star_history.append(theta_star_t)

        self.t += self.bs

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
# -------------------------------------------------------------
# Oracle θ*(φ) via dense grid search
# -------------------------------------------------------------
def oracle_theta(phi: np.ndarray, grid: np.ndarray) -> float:
    """
    θ* = argmin || η(x;φ) - y_s(x,θ) ||^2
    """
    x = np.linspace(0, 1, 400).reshape(-1, 1)
    eta = physical_system(x, phi)

    errs = []
    for th in grid:
        ys = computer_model_config2_np(x, np.array([th]))
        errs.append(np.mean((eta - ys) ** 2))

    return grid[np.argmin(errs)]


# -------------------------------------------------------------
# Aggregate OGP-BOCPD particles across experts (mass-weighted)
# -------------------------------------------------------------
def _aggregate_ogp_particles(bocpd, quantile=0.9):
    experts = bocpd.experts
    if len(experts) == 0:
        return None, None, None, None

    d = experts[0].pf.theta.shape[1]
    device = experts[0].pf.theta.device
    dtype = experts[0].pf.theta.dtype

    mean = torch.zeros(d, dtype=dtype, device=device)
    cov = torch.zeros(d, d, dtype=dtype, device=device)
    theta_list, weight_list = [], []

    for e in experts:
        w_e = math.exp(e.log_mass)
        w = e.pf.weights()
        th = e.pf.theta
        m = (w[:, None] * th).sum(0)
        C = ((th - m) * w[:, None]).T @ (th - m)
        mean = mean + w_e * m
        cov = cov + w_e * (C + (m - mean)[:, None] @ (m - mean)[None, :])
        theta_list.append(th)
        weight_list.append(w_e * w)

    theta_all = torch.cat(theta_list, dim=0)
    weight_all = torch.cat(weight_list, dim=0)
    weight_all = weight_all / weight_all.sum()
    var = torch.diag(cov)

    def weighted_quantile_1d(x, w, q):
        idx = torch.argsort(x)
        x, w = x[idx], w[idx]
        cw = torch.cumsum(w, dim=0)
        return x[cw >= q][0]

    alpha = (1.0 - quantile) / 2.0
    lo = torch.zeros(d, dtype=dtype, device=device)
    hi = torch.zeros(d, dtype=dtype, device=device)
    for j in range(d):
        lo[j] = weighted_quantile_1d(theta_all[:, j], weight_all, alpha)
        hi[j] = weighted_quantile_1d(theta_all[:, j], weight_all, 1.0 - alpha)

    return mean, var, lo, hi


# -------------------------------------------------------------
# Run one slope experiment
# -------------------------------------------------------------
def run_one_slope(
    slope: float,
    methods: Dict,
    total_T: int = 600,
    batch_size: int = 20,
    seed: int = 123,
    phi2_of_theta: callable = None,
    mode: int = 0,#0: slope origin, 1: slope inverse, 2: sudden origin
):
    print(f"\n=== Running slope={slope:.4f} ===")

    # if mode == 0:
    #     stream = SlopeDriftDataStream(
    #         total_T=total_T,
    #         batch_size=batch_size,
    #         slope=slope,
    #         seed=seed,
    #     )
    # elif mode == 1:
    #     stream = ThetaDrivenSlopeDataStream(
    #     total_T=total_T,
    #     batch_size=batch_size,
    #     noise_sd=0.2,
    #     theta0=1.6,                 # 起始 θ*
    #     theta_slope=slope,          # 你想测试的 drift
    #     phi2_of_theta=phi2_of_theta,
    #     seed=seed,
    # )

    # theta prior
    def prior_sampler(N):
        return torch.rand(N, 1) * 3.0
    def prior_sampler1(N, theta_anchor=None, sigma_local=0.2, p_global=0.2):
        """
        theta_anchor: 上一个 anchor / expert 的 posterior mean
        """
        N_global = int(p_global * N)
        N_local  = N - N_global

        samples = []

        # global prior (handle sudden change)
        if N_global > 0:
            samples.append(torch.rand(N_global, 1) * 3.0)

        # local prior (handle gradual drift)
        if theta_anchor is not None:
            local = theta_anchor + sigma_local * torch.randn(N_local, 1)
            local = torch.clamp(local, 0.0, 3.0)
            samples.append(local)
        else:
            samples.append(torch.rand(N_local, 1) * 3.0)

        return torch.cat(samples, dim=0)

    use_sampler1 = False
    # oracle
    theta_grid = np.linspace(0, 3, 400)

    results = {}

    for name, meta in methods.items():
        print(f"  -> {name}")
        t0 = time()

        theta_hist, rmse_hist, crps_hist = [], [], []
        total_obs = 0
        others_hist = []
        report_sub_hist = []
        theta_var_hist = []

        dll_hist = []
        mu_hist = []
        sig_hist = []
        h_hist = []
        odds_hist = []
        anchor_rl_hist = []
        cand_rl_hist = []

        top0_particles_hist = []

        if mode == 0:
            stream2 = SlopeDriftDataStream(
                total_T=total_T,
                batch_size=batch_size,
                slope=slope,
                seed=seed,
            )
        elif mode == 1:
            stream2 = ThetaDrivenSlopeDataStream(
                total_T=total_T,
                batch_size=batch_size,
                noise_sd=0.2,
                theta0=1.5,                 # 起始 θ*
                theta_slope=slope,          # 你想测试的 drift
                phi2_of_theta=phi2_of_theta,
                seed=seed,
            )
        

        # ---------- R-BOCPD-OGP ----------
        if name == "R-BOCPD-PF-OGP":
            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            grad_func = make_fast_batched_grad_func(
                computer_model_config2_torch, device="cuda", dtype=torch.float64,
            )

            ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=(0.0, 1.0),
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([3.0]),
                particle_chunk_size=256,  # 新增：控制 GPU 内存
            )

            
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            roll = OGPRollingStats(window=50)

            # bocpd = BOCPD_OGP(
            #     config=bocpd_cfg,
            #     ogp_pf_cfg=ogp_cfg,
            #     grad_func=grad_func,
            # )
            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,  # 注意：参数名变了
                device="cuda",                # GPU
            )

            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=f"{name}", unit="obs")
            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()

                ogp_dev = bocpd.device
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)

                if total_obs > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(batch_size, device=ogp_dev, dtype=torch.float64)
                    mix_var = torch.zeros(batch_size, device=ogp_dev, dtype=torch.float64)
                    Z = 0.0
                    for e in bocpd.experts:
                        w_e = math.exp(e.log_mass)
                        e_X_hist = e.X_hist if e.X_hist.numel() > 0 else None
                        e_y_hist = e.y_hist if e.y_hist.numel() > 0 else None
                        mu_mix_e, var_mix_e = e.pf.predict_batch(
                            Xb64, e_X_hist, e_y_hist,
                            emulator, model_cfg.rho, model_cfg.sigma_eps,
                        )
                        mix_mu += w_e * mu_mix_e
                        mix_var += w_e * var_mix_e
                        Z += w_e
                    mix_mu = mix_mu / max(Z, 1e-12)
                    mix_var = mix_var / max(Z, 1e-12)

                    mix_mu_cpu = mix_mu.cpu()
                    mix_var_cpu = mix_var.cpu()
                    Yb_cpu = Yb64.cpu()
                    rmse_hist.append(float(torch.sqrt(((mix_mu_cpu - Yb_cpu) ** 2).mean())))
                    crps = crps_gaussian(mix_mu_cpu, mix_var_cpu, Yb_cpu).mean()
                    crps_hist.append(crps.item())

                rec = bocpd.update_batch(
                    Xb64, Yb64, emulator, model_cfg, None, prior_sampler,
                    verbose=False,
                )

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mu_hat = roll.mean()
                sig_hat = roll.std()
                h_log = rec.get("h_log", None)
                log_odds = rec.get("log_odds_mass", None)

                dll_hist.append(dll)
                mu_hist.append(mu_hat)
                sig_hist.append(sig_hat)
                h_hist.append(h_log)
                odds_hist.append(log_odds)

                anchor_rl_hist.append(rec.get("anchor_rl", None))
                cand_rl_hist.append(rec.get("cand_rl", None))

                mean_theta, var_theta, lo_theta, hi_theta = _aggregate_ogp_particles(
                    bocpd, 0.9,
                )
                theta_hist.append(float(mean_theta[0]))

                batch_particles = []
                batch_weights = []
                batch_logmass = []
                for e in bocpd.experts:
                    batch_particles.append(e.pf.theta.squeeze(-1).detach().cpu())
                    batch_weights.append(e.pf.weights().detach().cpu())
                    batch_logmass.append(float(e.log_mass))

                top0_particles_hist.append(dict(
                    particles=batch_particles,
                    weights=batch_weights,
                    log_mass=torch.tensor(batch_logmass),
                ))

                others_hist.append({
                    "did_restart": rec["did_restart"],
                    "var": float(var_theta[0]),
                    "lo": float(lo_theta[0]),
                    "hi": float(hi_theta[0]),
                    "pf_info": rec["pf_diags"],
                })

                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- Standalone PF-OGP (no BOCPD) ----------
        elif name == "PF-OGP":
            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            pf_grad_func = make_fast_batched_grad_func(
                computer_model_config2_torch, device="cuda", dtype=torch.float64,
            )
            pf_ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=(0.0, 1.0),
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([3.0]),
                theta_move_std=0.02,
                particle_chunk_size=256,
            )
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=0.05)
            ogp_dev = "cuda"

            pf = OGPParticleFilter(
                ogp_cfg=pf_ogp_cfg,
                prior_sampler=prior_sampler,
                batched_grad_func=pf_grad_func,
                device=ogp_dev,
                dtype=torch.float64,
            )

            pf_X_hist = torch.empty(
                0, 1, dtype=torch.float64, device=ogp_dev,
            )
            pf_y_hist = torch.empty(0, dtype=torch.float64, device=ogp_dev)
            pf_ogp_max_hist = 200

            from tqdm import tqdm
            pbar = tqdm(total=total_T, desc=name, unit="obs")
            while total_obs < total_T:
                Xb, Yb = stream2.next()
                Xb64 = Xb.to(device=ogp_dev, dtype=torch.float64)
                Yb64 = Yb.to(device=ogp_dev, dtype=torch.float64)

                if total_obs > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(
                        Xb64, pf_Xh, pf_yh,
                        emulator, pf_model_cfg.rho, pf_model_cfg.sigma_eps,
                    )
                    rmse_hist.append(
                        float(torch.sqrt(((mu_mix.cpu() - Yb64.cpu()) ** 2).mean()))
                    )
                    crps = crps_gaussian(mu_mix.cpu(), var_mix.cpu(), Yb64.cpu()).mean()
                    crps_hist.append(crps.item())

                pf.step_batch(
                    Xb64, Yb64,
                    pf_X_hist if pf_X_hist.numel() > 0 else None,
                    pf_y_hist if pf_y_hist.numel() > 0 else None,
                    emulator,
                    pf_model_cfg.rho,
                    pf_model_cfg.sigma_eps,
                )

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

                top0_particles_hist.append(dict(
                    particles=[pf.theta.squeeze(-1).detach().cpu()],
                    weights=[pf.weights().detach().cpu()],
                    log_mass=torch.tensor([0.0]),
                ))

                others_hist.append({
                    "did_restart": False,
                    "var": float(
                        (w * (pf.theta - mean_theta).pow(2)).sum(dim=0)[0]
                    ),
                })

                total_obs += batch_size
                pbar.update(batch_size)
            pbar.close()

        # ---------- BOCPD ----------
        elif meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True
            roll = RollingStats(window=50)

            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]
                cfg.model.bocpd_use_discrepancy = meta["bocpd_use_discrepancy"]

            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )

            if use_sampler1:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler1)
            else:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()

                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    pred_comp = calib.predict_complete(Xb, Yb)
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    # print(name, total_obs, report_hist[-1])
                    rmse_hist.append(
                        float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()))
                    )
                    crps = crps_gaussian(pred["mu"], pred["var"], Yb).mean()
                    # print(crps)
                    crps_hist.append(crps.item())
                    # rmse_hist.append(
                    #     float(torch.sqrt(((pred["mu_sim"] - Yb) ** 2).mean()))
                    # )

                rec = calib.step_batch(Xb, Yb, verbose=False)

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mu_hat = roll.mean()
                sig_hat = roll.std()
                h_log = rec.get("h_log", None)
                log_odds = rec.get("log_odds_mass", None)
                # print("debug: dll, mu_hat, sig_hat, h_log, log_odds",dll, mu_hat, sig_hat, h_log, log_odds)

                dll_hist.append(dll)
                mu_hist.append(mu_hat)
                sig_hist.append(sig_hat)
                h_hist.append(h_log)
                odds_hist.append(log_odds)

                anchor_rl_hist.append(rec.get("anchor_rl", None))
                cand_rl_hist.append(rec.get("cand_rl", None))

                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
                theta_hist.append(float(mean_theta[0]))

                experts = calib.bocpd.experts

                batch_particles = []
                batch_weights = []
                batch_logmass = []

                for e in experts:
                    # particles
                    particles = e.pf.particles.theta          # (N,1)
                    particles_1d = particles.squeeze(-1).detach().cpu()

                    # weights
                    pw = e.pf.particles.weights()             # (N,)
                    pw_1d = pw.squeeze(-1).detach().cpu()

                    # log mass
                    log_mass = float(e.log_mass)

                    batch_particles.append(particles_1d)
                    batch_weights.append(pw_1d)
                    batch_logmass.append(log_mass)

                batch_dict = dict(
                    particles=batch_particles,      # list length E
                    weights=batch_weights,          # list length E
                    log_mass=torch.tensor(batch_logmass)  # (E,)
                )

                top0_particles_hist.append(batch_dict)

                ess_gini_info = []
                for ei, e in enumerate(calib.bocpd.experts):
                    ps = e.pf.particles
                    unique_ratio = float(ps.unique_ratio())
                    entropy_1d_histogram = float(ps.entropy_1d_histogram())
                    # print(ei, unique_ratio, entropy_1d_histogram)
                    ess_gini_info.append({"expert_id": ei, "unique_ratio": unique_ratio, "entropy_1d_histogram": entropy_1d_histogram})
                others_hist.append({"did_restart": rec["did_restart"],"var": float(var_theta[0]), "lo": float(lo_theta[0]), "hi": float(hi_theta[0]), "pf_info": rec["pf_diags"], "report_sub_hist": report_sub_hist, "pf_health_info": ess_gini_info})

                if use_sampler1:
                    calib.theta_anchor = mean_theta[0]

                total_obs += batch_size

        # ---------- BPC ----------
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
                crps_sim = None
                if X_hist is None:
                    X_hist, y_hist = Xb.numpy(), Yb.numpy()
                else:
                    X_hist = np.concatenate([X_hist, Xb.numpy()], axis=0)
                    y_hist = np.concatenate([y_hist, Yb.numpy()], axis=0)
                if X_hist.shape[0] >= W:
                    X_hist = X_hist[-W:]
                    y_hist = y_hist[-W:]
                # X_hist.append(Xb.numpy())
                # y_hist.append(Yb.numpy())
                if total_obs > 0 and bpc is not None:
                    # mu_np, var_np = bpc.predict_sim(Xb.detach().cpu().numpy())
                    mu_np, var_np = bpc.predict(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device), torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device) 
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(crps_sim.item())
                    # print("bpc crps sim:", crps_sim)

                X_all, y_all = X_hist, y_hist

                bpc = BayesianProjectedCalibration(
                    theta_lo=np.array([0.0]),
                    theta_hi=np.array([3.0]),
                    noise_var=0.04,
                    y_sim=computer_model_config2_np,
                )

                X_grid = np.linspace(0, 1, 300).reshape(-1, 1)
                bpc.fit(X_all, y_all, X_grid, n_eta_draws=500, n_restart=10, gp_fit_iters=200)

                theta_hist.append(float(bpc.theta_mean[0]))
                # print("bpc theta var:", bpc.theta_var[0])
                entropy_info = bpc.entropy_theta()
                # print("bpc theta entropy:", entropy_info)
                total_obs += batch_size
                others_hist.append({"var": float(bpc.theta_var[0]), "entropy": entropy_info, "crps_sim": crps_sim})

                theta_samples_bpc = torch.tensor(bpc.theta_samples).squeeze(-1)
                top0_particles_hist.append(theta_samples_bpc)
                batch_dict = dict(
                    particles=[theta_samples_bpc],      # list length E
                    weights=None,          # list length E
                    log_mass=torch.tensor([0.0])  # (E,)
                )

                top0_particles_hist.append(batch_dict)

        # ---------- BPC + BOCPD ----------
        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(
                theta_lo=np.array([0.0]),
                theta_hi=np.array([3.0]),
                noise_var=0.04,
                y_sim=computer_model_config2_np,
                X_grid=np.linspace(0, 1, 300).reshape(-1, 1),
            )

            # stream2 = SlopeDriftDataStream(
            #     total_T=total_T,
            #     batch_size=batch_size,
            #     slope=slope,
            #     seed=seed,
            # )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
                crps_sim = None
                if total_obs > 0:
                    # mu, var = calib.predict_sim(Xb)
                    mu, var = calib.predict(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device), torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)
                    crps_hist.append(crps_sim.item())
                    # print("bocpd-bpc crps sim:", crps_sim)
                info = calib.step_batch(Xb.detach().cpu().numpy(), Yb.detach().cpu().numpy())

                masses = np.asarray(info["masses"])
                thetas = np.asarray(info["theta_means"])

                # if masses.sum() > 0:
                #     w = masses / masses.sum()
                #     theta_hat = float((w * thetas[:, 0]).sum())
                # else:
                #     theta_hat = np.nan

                # theta_hist.append(theta_hat)
                total_obs += batch_size

                theta_mean, theta_var, theta_lo, theta_hi = calib._aggregate_particles(0.9)
                theta_hist.append(float(theta_mean[0]))
                # print("bocpd-bpc theta var:", theta_var[0])
                others_hist.append({"did_restart": info["did_restart"], "var": theta_var[0], "lo": theta_lo, "hi": theta_hi, "crps_sim": crps_sim})

                batch_particles = []
                batch_weights = []
                batch_logmass = []
                for e in calib.experts:
                    particles = torch.tensor(e.bpc.theta_samples).squeeze(-1)
                    batch_logmass.append(e.logw)          # (N,1)
                    batch_particles.append(particles)

                batch_dict = dict(
                    particles=batch_particles,      # list length E
                    weights=batch_weights,          # list length E
                    log_mass=torch.tensor(batch_logmass)  # (E,)
                )

                top0_particles_hist.append(batch_dict)
                # print(theta_mean, theta_var[0], theta_lo, theta_hi)

        # ---------- oracle ----------
        phi_hist = stream2.phi_history[: len(theta_hist)]
        oracle_hist = [
            oracle_theta(phi, theta_grid) for phi in phi_hist
        ]
        if meta["type"] == "bocpd":
            results[name] = dict(
                    theta=np.array(theta_hist),
                    theta_oracle=np.array(oracle_hist),
                    others=others_hist,
                    rmse=np.array(rmse_hist),
                    top0_particles_hist=top0_particles_hist,
                    seed=seed,
                    batch_size=batch_size,
                    slope=slope,
                    mode=mode,
                    oracle_hist=oracle_hist,
                    phi_hist=phi_hist,
                    delta_ll_hist=np.array(dll_hist),
                    mu_hat_hist=np.array(mu_hist),
                    sigma_hat_hist=np.array(sig_hist),
                    h_log_hist=np.array(h_hist),
                    log_odds_hist=np.array(odds_hist),
                    anchor_rl_hist=np.array(anchor_rl_hist),
                    cand_rl_hist=np.array(cand_rl_hist),
                    crps_hist=np.array(crps_hist),
                )
        else:
            results[name] = dict(
                    theta=np.array(theta_hist),
                    theta_oracle=np.array(oracle_hist),
                    others=others_hist,
                    rmse=np.array(rmse_hist),
                    top0_particles_hist=top0_particles_hist,
                    seed=seed,
                    batch_size=batch_size,
                    slope=slope,
                    mode=mode,
                    oracle_hist=oracle_hist,
                    phi_hist=phi_hist,
                    crps_hist=np.array(crps_hist),
                )

        if mode == 1:
            results[name]["theta_star_true"] = np.array(stream2.theta_star_history)

        print(f"     done in {time() - t0:.1f}s")

    K = len(theta_hist)
    phi_hist = [stream2.true_phi(k*batch_size) for k in range(K)]
    oracle_hist = [oracle_theta(phi, theta_grid) for phi in phi_hist]

    return results, phi_hist, oracle_hist


# -------------------------------------------------------------
# Main: multiple slopes
# -------------------------------------------------------------
def main():
    # seeds = [0, 123, 456, 789]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/slope_deltaCmp_v2")
    args = parser.parse_args()

    if args.debug:
        seeds = [456]
        batch_sizes = [10]
        slopes = [0.001]
    else:
        seeds = [101, 202, 303, 404, 505]
        batch_sizes = [20]
        slopes = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
    # batch_sizes = [20]
    # slopes = [0.003]
    store_dir = args.out_dir
    import os
    os.makedirs(store_dir, exist_ok=True)

    mode = 1
    if mode == 1:
        phi2_grid = np.linspace(3.0, 12.0, 300)
        theta_grid = np.linspace(0.0, 3.0, 600)

        phi2_of_theta, _ = build_phi2_from_theta_star(
            phi2_grid=phi2_grid,
            theta_grid=theta_grid,
        )
    else:
        phi2_of_theta = None

    methods = {
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        "R-BOCPD-PF-OGP": dict(type="bocpd", mode="restart"),
        "PF-OGP": dict(type="pf_ogp"),
        "BOCPD-PF": dict(type="bocpd", mode="standard"),
        "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True, bocpd_use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
        # "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
    }

    all_results = {}
    import itertools

    for s,batch_size,seed in itertools.product(slopes, batch_sizes, seeds):
        res, phi_hist, oracle_hist = run_one_slope(s, methods, batch_size=batch_size, seed=seed, phi2_of_theta=phi2_of_theta, mode=mode)
        all_results[(s,batch_size,seed)] = res

        # ---------- plot ----------
        plt.figure(figsize=(10, 5))
        for name, d in res.items():
            plt.plot(d["theta"], label=name)
        plt.plot(d["theta_oracle"], "k--", lw=2, label="oracle θ*")
        plt.title(f"Theta tracking (slope={s})")
        plt.xlabel("batch index")
        plt.ylabel("theta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_theta.png", dpi=300)
        plt.close()

        # np.savez(
        #     f"slope_{s}_results.npz",
        #     **{f"{k}_theta": v["theta"] for k, v in res.items()},
        #     oracle_theta=res["Standard-BOCPD"]["theta_oracle"],
        # )
        torch.save(res, f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_results.pt")
        torch.save(dict(phi_hist=phi_hist, oracle_hist=oracle_hist), f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_phi_oracle_hist.pt")

    # ========== 收集所有组合的 metrics ==========
    all_metrics = []

    for s, batch_size, seed in itertools.product(slopes, batch_sizes, seeds):
        # 从 all_results 获取对应的 res
        res = all_results[(s, batch_size, seed)]  # 注意：当前代码中 all_results[s] 会被覆盖，需要改为 all_results[(s, batch_size, seed)]
        
        for method_name, data in res.items():
            # 计算 theta_rmse: sqrt(mean((theta_pred - theta_oracle)^2))
            theta_rmse = np.sqrt(np.mean((data["theta"] - data["theta_oracle"]) ** 2))
            
            # y_rmse 已经存在于 data["rmse"] 中
            y_rmse_mean = np.mean(data["rmse"])
            
            # y_crps 已经存在于 data["crps_hist"] 中
            y_crps_mean = np.mean(data["crps_hist"])
            
            all_metrics.append({
                "method": method_name,
                "slope": s,
                "batch_size": batch_size,
                "seed": seed,
                "theta_rmse": theta_rmse,
                "y_rmse": y_rmse_mean,
                "y_crps": y_crps_mean,
            })

    # 转换为 DataFrame 并保存
    import pandas as pd
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(f"{store_dir}/all_metrics.csv", index=False)
    df_metrics.to_excel(f"{store_dir}/all_metrics.xlsx", index=False)

    # ========== 打印每个 method 的平均 metrics ==========
    print("\n" + "="*70)
    print("Average Metrics Across All Combinations (slopes × batch_sizes × seeds):")
    print("="*70)

    grouped = df_metrics.groupby("method").agg({
        "theta_rmse": ["mean", "std"],
        "y_rmse": ["mean", "std"],
        "y_crps": ["mean", "std"],
    })

    for method in df_metrics["method"].unique():
        print(f"\n{method}:")
        stats = grouped.loc[method]
        print(f"  theta_rmse: {stats[('theta_rmse', 'mean')]:.6f} ± {stats[('theta_rmse', 'std')]:.6f}")
        print(f"  y_rmse:     {stats[('y_rmse', 'mean')]:.6f} ± {stats[('y_rmse', 'std')]:.6f}")
        print(f"  y_crps:     {stats[('y_crps', 'mean')]:.6f} ± {stats[('y_crps', 'std')]:.6f}")

    print("\n" + "="*70)

    print("All slope experiments finished.")


if __name__ == "__main__":
    main()
