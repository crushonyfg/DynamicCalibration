# =============================================================
# run_synthetic_slopecomp.py
# Gradual-drift slope comparison experiment
# =============================================================

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
from .online_calibrator import OnlineBayesCalibrator
from .bpc import BayesianProjectedCalibration
from .bcp_bocpd import *

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

    # ⚠️ 假设 theta_star_vals 单调（你这个模型下基本成立）
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
):
    print(f"\n=== Running slope={slope:.4f} ===")

    # stream = ThetaDrivenSlopeDataStream(
    #     total_T=total_T,
    #     batch_size=batch_size,
    #     slope=slope,
    #     seed=seed,
    # )

    stream = ThetaDrivenSlopeDataStream(
        total_T=total_T,
        batch_size=batch_size,
        noise_sd=0.2,
        theta0=1.6,                 # 起始 θ*
        theta_slope=slope,          # 你想测试的 drift
        phi2_of_theta=phi2_of_theta,
        seed=seed,
    )


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

        theta_hist, rmse_hist = [], []
        total_obs = 0
        others_hist = []
        

        # ---------- BOCPD ----------
        if meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True

            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )

            if use_sampler1:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler1)
            else:
                calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

            stream2 = SlopeDriftDataStream(
                total_T=total_T,
                batch_size=batch_size,
                slope=slope,
                seed=seed,
            )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()

                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    # rmse_hist.append(
                    #     float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()))
                    # )
                    rmse_hist.append(
                        float(torch.sqrt(((pred["mu_sim"] - Yb) ** 2).mean()))
                    )

                rec = calib.step_batch(Xb, Yb, verbose=False)
                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
                theta_hist.append(float(mean_theta[0]))

                ess_gini_info = []
                for ei, e in enumerate(calib.bocpd.experts):
                    ps = e.pf.particles
                    ess_val = float(ps.ess().detach().cpu())
                    gini_val = float(ps.gini().detach().cpu())
                    ess_gini_info.append({"expert_id": ei, "ess": ess_val, "gini": gini_val})
                others_hist.append({"did_restart": rec["did_restart"],"var": float(var_theta[0]), "lo": float(lo_theta[0]), "hi": float(hi_theta[0]), "ess_gini_info": ess_gini_info})

                if use_sampler1:
                    calib.theta_anchor = mean_theta[0]

                total_obs += batch_size

        # ---------- BPC ----------
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            stream2 = SlopeDriftDataStream(
                total_T=total_T,
                batch_size=batch_size,
                slope=slope,
                seed=seed,
            )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
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
                    mu_np, var_np = bpc.predict_sim(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device), torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device) 
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))

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
                total_obs += batch_size

        # ---------- BPC + BOCPD ----------
        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(
                theta_lo=np.array([0.0]),
                theta_hi=np.array([3.0]),
                noise_var=0.04,
                y_sim=computer_model_config2_np,
                X_grid=np.linspace(0, 1, 300).reshape(-1, 1),
            )

            stream2 = SlopeDriftDataStream(
                total_T=total_T,
                batch_size=batch_size,
                slope=slope,
                seed=seed,
            )

            while total_obs < total_T:
                if total_obs % 100 == 0:
                    print(f"{name}  -> total_obs: {total_obs}")
                Xb, Yb = stream2.next()
                if total_obs > 0:
                    mu, var = calib.predict_sim(Xb)
                    mu_t, var_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device), torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
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
                others_hist.append({"did_restart": info["did_restart"], "var": theta_var[0], "lo": theta_lo, "hi": theta_hi})
                # print(theta_mean, theta_var[0], theta_lo, theta_hi)

        # ---------- oracle ----------
        phi_hist = stream.phi_history[: len(theta_hist)]
        oracle_hist = [
            oracle_theta(phi, theta_grid) for phi in phi_hist
        ]

        results[name] = dict(
            theta=np.array(theta_hist),
            theta_oracle=np.array(oracle_hist),
            others=others_hist,
            rmse=np.array(rmse_hist),
        )

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
    seeds = [456]
    batch_sizes = [10, 20, 40]
    slopes = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]

    phi2_grid = np.linspace(3.0, 12.0, 300)
    theta_grid = np.linspace(0.0, 3.0, 600)

    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=phi2_grid,
        theta_grid=theta_grid,
    )

    # batch_sizes = [40]
    # slopes = [0.005]
    store_dir = "figs/slope_v5_inverse"
    import os
    os.makedirs(store_dir, exist_ok=True)

    methods = {
        "BOCPD-BPC": dict(type="bpc_bocpd"),
        "BOCPD-PF": dict(type="bocpd", mode="standard"),
        "R-BOCPD-PF": dict(type="bocpd", mode="restart"),
        "BPC-80": dict(type="bpc"),
    }

    all_results = {}
    import itertools

    for s,batch_size,seed in itertools.product(slopes, batch_sizes, seeds):
        res, phi_hist, oracle_hist = run_one_slope(s, methods, batch_size=batch_size, seed=seed, phi2_of_theta=phi2_of_theta)
        all_results[s] = res

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

    print("All slope experiments finished.")


if __name__ == "__main__":
    main()
