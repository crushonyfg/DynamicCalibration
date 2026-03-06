# =============================================================
# run_synthetic_suddencomp.py
# Sudden-change (3 changepoints) magnitude/frequency grid experiment
# =============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from time import time
import itertools

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *  # StandardBOCPD_BPC


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


# -------------------------------------------------------------
# True physical system η(x; φ)
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
# Data stream with 3 sudden changepoints
# -------------------------------------------------------------
class SuddenChangeDataStream:
    """
    total_T observations, generated in batches.
    changepoints are in observation-time units (same as t counter).
    """

    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        cp_times: List[int],
        phi_segments: List[np.ndarray],  # length = len(cp_times)+1
        seed: int,
    ):
        assert len(phi_segments) == len(cp_times) + 1
        self.T = int(total_T)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)
        self.cp_times = [int(t) for t in cp_times]
        self.phi_segments = [np.asarray(p, dtype=float) for p in phi_segments]
        self.rng = np.random.RandomState(int(seed))

        self.t = 0
        self.phi_history = []  # per-batch phi (at batch start)
        self.seg_history = []  # per-batch segment id

    def _seg_id(self, t: int) -> int:
        # number of cps with time <= t
        k = 0
        for c in self.cp_times:
            if t >= c:
                k += 1
            else:
                break
        return k

    def true_phi(self, t: int) -> np.ndarray:
        return self.phi_segments[self._seg_id(t)].copy()

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.t >= self.T:
            raise StopIteration

        # X = self.rng.rand(self.bs, 1)
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs     
        X = u[:, None]
        self.rng.shuffle(X)

        phi_t = self.true_phi(self.t)

        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.seg_history.append(self._seg_id(self.t))

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
    θ* = argmin || η(x;φ) - y_s(x,θ) ||^2  (approximated by dense x-grid)
    """
    x = np.linspace(0, 1, 400).reshape(-1, 1)
    eta = physical_system(x, phi)

    errs = []
    for th in grid:
        ys = computer_model_config2_np(x, np.array([th]))
        errs.append(np.mean((eta - ys) ** 2))

    return float(grid[int(np.argmin(errs))])


# -------------------------------------------------------------
# Build 4 segment-phis (3 changepoints), centered around phi2=7.5
# magnitude controls how far we move phi[1] (the "a2" term)
# -------------------------------------------------------------
def build_phi_segments_centered(delta: float, center: float = 7.5):
    """
    4 regimes, 3 changepoints
    phi[1] strictly increasing
    mean(phi[1]) = center
    adjacent jump size = delta
    """
    a1, a3 = 5.0, 5.0

    phi2_vals = np.array([
        center - 1.5 * delta,
        center - 0.5 * delta,
        center + 0.5 * delta,
        center + 1.5 * delta,
    ])

    return [
        np.array([a1, v, a3], dtype=float)
        for v in phi2_vals
    ]



# -------------------------------------------------------------
# Run one (frequency, magnitude) experiment
# frequency here is segment length L (in observation-time units)
# 3 CPs => total_T = 4*L
# -------------------------------------------------------------
def run_one_sudden(
    seg_len_L: int,
    delta_mag: float,
    methods: Dict,
    batch_size: int,
    seed: int,
    noise_sd: float = 0.2,
    phi_center: float = 7.5,
    out_dir: str = ".",
):
    # Ensure CP times and total_T align with batch size to avoid partial batch around CP
    seg_len_L = int(seg_len_L)
    bs = int(batch_size)
    if seg_len_L % bs != 0:
        raise ValueError(f"seg_len_L ({seg_len_L}) must be divisible by batch_size ({bs})")

    total_T = 4 * seg_len_L
    cp_times = [seg_len_L, 2 * seg_len_L, 3 * seg_len_L]
    phi_segments = build_phi_segments_centered(delta=delta_mag, center=phi_center)

    print(f"\n=== Sudden experiment: L={seg_len_L}, delta={delta_mag:.4f}, bs={bs}, seed={seed} ===")
    print(f"    cp_times={cp_times}, total_T={total_T}")
    print(f"    phi[1] values={[p[1] for p in phi_segments]} (center={phi_center})")

    # oracle grid
    theta_grid = np.linspace(0, 3, 600)

    # shared stream for oracle phi_history reference (per-batch)
    stream_ref = SuddenChangeDataStream(
        total_T=total_T,
        batch_size=bs,
        noise_sd=noise_sd,
        cp_times=cp_times,
        phi_segments=phi_segments,
        seed=seed,
    )

    # prior
    def prior_sampler(N: int):
        return torch.rand(N, 1) * 3.0

    results = {}

    for name, meta in methods.items():
        print(f"  -> {name}")
        t0 = time()

        theta_hist: List[float] = []
        rmse_hist: List[float] = []
        others_hist: List[dict] = []
        total_obs = 0
        top0_particles_hist = []


        # fresh stream per method (same data given same seed)
        stream = SuddenChangeDataStream(
            total_T=total_T,
            batch_size=bs,
            noise_sd=noise_sd,
            cp_times=cp_times,
            phi_segments=phi_segments,
            seed=seed,
        )

        # ---------- BOCPD ----------
        if meta["type"] == "bocpd":
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True

            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]

            emulator = DeterministicSimulator(
                func=computer_model_config2_torch,
                enable_autograd=True,
            )
            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
                    print(f"     {name}: total_obs={total_obs}")

                Xb, Yb = stream.next()
                report_sub_hist = None

                if total_obs > 0:
                    pred = calib.predict_batch(Xb)
                    pred_comp = calib.predict_complete(Xb, Yb)
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    # print(name, total_obs, report_hist[-1])
                    # rmse_hist.append(
                    #     float(torch.sqrt(((pred["mu"] - Yb) ** 2).mean()))
                    # )
                    rmse_hist.append(
                        float(torch.sqrt(((pred["mu_sim"] - Yb) ** 2).mean()))
                    )

                rec = calib.step_batch(Xb, Yb, verbose=False)

                # NOTE: assumes your OnlineBayesCalibrator._aggregate_particles(q)
                # returns (mean, var_or_cov, lo, hi) where mean/lo/hi are vectors.
                mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)

                theta_hist.append(float(mean_theta[0]))
                # var_theta may be scalar/vec/cov; keep first dim as scalar for logging
                v0 = float(var_theta[0]) if np.ndim(var_theta) >= 1 else float(var_theta)

                ess_gini_info = []
                for ei, e in enumerate(calib.bocpd.experts):
                    ps = e.pf.particles
                    unique_ratio = float(ps.unique_ratio())
                    entropy_1d_histogram = float(ps.entropy_1d_histogram())
                    # print(ei, unique_ratio, entropy_1d_histogram)
                    ess_gini_info.append({"expert_id": ei, "unique_ratio": unique_ratio, "entropy_1d_histogram": entropy_1d_histogram})

                # experts = calib.bocpd.experts

                # weights = torch.tensor([e.log_mass for e in experts])
                # top_idx = torch.argmax(weights).item()
                # top_expert = experts[top_idx]

                # particles = top_expert.pf.particles.theta  
                # pw = top_expert.pf.particles.weights()

                # particles_1d = particles.squeeze(-1).detach().cpu()
                # pw_1d = pw.squeeze(-1).detach().cpu()

                # top0_particles_hist.append((particles_1d, pw_1d))

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

                others_hist.append(
                    dict(
                        did_restart=bool(rec.get("did_restart", False)),
                        var=v0,
                        lo=float(lo_theta[0]),
                        hi=float(hi_theta[0]),
                        ess_gini_info=ess_gini_info,
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        pf_info=rec["pf_diags"],
                        report_sub_hist=report_sub_hist,
                        pf_health_info=ess_gini_info,
                    )
                )

                total_obs += bs

        # ---------- BPC (batch refit each step) ----------
        elif meta["type"] == "bpc":
            W = 80
            X_hist = None
            y_hist = None
            bpc = None

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
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
                    mu_np, var_np = bpc.predict_sim(Xb.detach().cpu().numpy())
                    mu_t, var_t = torch.tensor(mu_np, dtype=Yb.dtype, device=Yb.device), torch.tensor(var_np, dtype=Yb.dtype, device=Yb.device) 
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)

                X_all, y_all = X_hist, y_hist

                bpc = BayesianProjectedCalibration(
                    theta_lo=np.array([0.0]),
                    theta_hi=np.array([3.0]),
                    noise_var=float(noise_sd ** 2),
                    y_sim=computer_model_config2_np,
                )
                X_grid = np.linspace(0, 1, 400).reshape(-1, 1)
                bpc.fit(X_all, y_all, X_grid, n_eta_draws=500, n_restart=10, gp_fit_iters=200)

                theta_hist.append(float(bpc.theta_mean[0]))
                entropy_info = bpc.entropy_theta()
                others_hist.append(
                    dict(
                        did_restart=False,
                        var=float(bpc.theta_var[0]) if bpc.theta_var is not None else float("nan"),
                        lo=float("nan"),
                        hi=float("nan"),
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        entropy=entropy_info,
                        crps_sim=crps_sim,
                    )
                )

                theta_samples_bpc = torch.tensor(bpc.theta_samples).squeeze(-1)
                top0_particles_hist.append(theta_samples_bpc)
                batch_dict = dict(
                    particles=[theta_samples_bpc],      # list length E
                    weights=None,          # list length E
                    log_mass=torch.tensor([0.0])  # (E,)
                )

                top0_particles_hist.append(batch_dict)

                total_obs += bs

        # ---------- BPC + BOCPD ----------
        elif meta["type"] == "bpc_bocpd":
            calib = StandardBOCPD_BPC(
                theta_lo=np.array([0.0]),
                theta_hi=np.array([3.0]),
                noise_var=float(noise_sd ** 2),
                y_sim=computer_model_config2_np,
                X_grid=np.linspace(0, 1, 400).reshape(-1, 1),
                # if your class supports: hazard_h/topk/etc, put them in meta["params"]
                **meta.get("params", {}),
            )

            while total_obs < total_T:
                if total_obs % (5 * bs) == 0:
                    print(f"     {name}: total_obs={total_obs}")

                Xb, Yb = stream.next()
                crps_sim = None

                if total_obs > 0:
                    mu, var = calib.predict_sim(Xb)
                    mu_t, var_t = torch.tensor(mu, dtype=Yb.dtype, device=Yb.device), torch.tensor(var, dtype=Yb.dtype, device=Yb.device)
                    rmse_hist.append(float(torch.sqrt(((mu_t - Yb) ** 2).mean())))
                    crps_sim = crps_gaussian(mu_t, var_t, Yb)

                info = calib.step_batch(Xb.detach().cpu().numpy(), Yb.detach().cpu().numpy())

                # assumes your StandardBOCPD_BPC._aggregate_particles(q) exists and returns (mean, cov/var, lo, hi)
                theta_mean, theta_var, theta_lo, theta_hi = calib._aggregate_particles(0.9)

                theta_hist.append(float(theta_mean[0]))
                # theta_var could be vector or cov; keep first dim scalar
                try:
                    v0 = float(theta_var[0][0]) if np.ndim(theta_var) >= 1 else float(theta_var)
                except:
                    v0 = theta_var

                others_hist.append(
                    dict(
                        did_restart=bool(info.get("did_restart", False)),
                        var=v0,
                        lo=float(theta_lo[0]) if np.ndim(theta_lo) >= 1 else float(theta_lo),
                        hi=float(theta_hi[0]) if np.ndim(theta_hi) >= 1 else float(theta_hi),
                        seg_id=int(stream.seg_history[-1]),
                        t=int(total_obs),
                        crps_sim=crps_sim,
                    )
                )

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

                total_obs += bs

        else:
            raise ValueError(f"Unknown method type: {meta['type']}")

        # ---------- oracle (aligned by batch index) ----------
        # Use the *reference* stream to define "true phi per batch" for oracle
        # but both streams are deterministic under same seed anyway.
        K = len(theta_hist)
        # Make sure stream_ref advanced to K batches:
        while len(stream_ref.phi_history) < K:
            stream_ref.next()

        phi_hist = stream_ref.phi_history[:K]
        oracle_hist = [oracle_theta(phi, theta_grid) for phi in phi_hist]

        results[name] = dict(
            theta=np.asarray(theta_hist, dtype=float),
            theta_oracle=np.asarray(oracle_hist, dtype=float),
            others=others_hist,
            rmse=np.asarray(rmse_hist, dtype=float),
            cp_times=cp_times,
            seg_len_L=seg_len_L,
            delta_mag=float(delta_mag),
            batch_size=bs,
            seed=int(seed),
            top0_particles_hist=top0_particles_hist,
        )

        print(f"     done in {time() - t0:.1f}s")

    # Also return the phi/oracle series for external plotting
    phi_hist = stream_ref.phi_history[: len(results[list(results.keys())[0]]["theta"])]
    oracle_hist = results[list(results.keys())[0]]["theta_oracle"]
    return results, phi_hist, oracle_hist


# -------------------------------------------------------------
# Plotting helper
# -------------------------------------------------------------
def plot_theta_tracking(
    res: Dict,
    oracle_hist: np.ndarray,
    cp_times: List[int],
    batch_size: int,
    title: str,
    save_path: str,
):
    plt.figure(figsize=(12, 5))
    for name, d in res.items():
        plt.plot(d["theta"], label=name, alpha=0.9)

    plt.plot(np.asarray(oracle_hist), "k--", lw=2, label="oracle θ*")

    # CP vertical lines in batch-index coordinates
    for c in cp_times:
        x = c // batch_size
        plt.axvline(x=x, color="red", linestyle="--", alpha=0.35)

    plt.title(title)
    plt.xlabel("batch index")
    plt.ylabel("theta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------------------------------------------
# Main: traverse (frequency, magnitude) + (seed, batch_size)
# 3 changepoints per run, phi centered around 7.5
# -------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="./figs/sudden_grid_outputs/v1")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- experimental grid ---
    if args.debug:
        seeds = [456]               # you can add more
        batch_sizes = [20]      # you can add more
        seg_lens = [120]
        magnitudes = [2.0] 
    else:
        magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0]
        seeds = [456]               # you can add more
        batch_sizes = [20, 40]      # you can add more

        # frequency: segment length L in observation-time units
        # NOTE: must be divisible by batch_size (enforced in run_one_sudden)
        seg_lens = [80, 120, 200]  # frequency: smaller => more frequent CPs

        # magnitude: delta applied to phi[1] around center=7.5
        magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0]

    # methods
    methods = {
        "BOCPD-PF": dict(type="bocpd", mode="standard"),
        "BPC-80": dict(type="bpc"),
        "BOCPD-BPC": dict(type="bpc_bocpd"),
        "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False),
        # "BPC-80": dict(type="bpc"),
    }

    # run grid
    for seg_len_L, delta_mag, batch_size, seed in itertools.product(seg_lens, magnitudes, batch_sizes, seeds):
        # skip invalid combos early (L must be divisible by batch_size)
        if seg_len_L % batch_size != 0:
            continue

        res, phi_hist, oracle_hist = run_one_sudden(
            seg_len_L=seg_len_L,
            delta_mag=delta_mag,
            methods=methods,
            batch_size=batch_size,
            seed=seed,
            noise_sd=0.2,
            phi_center=7.5,
            out_dir=out_dir,
        )

        tag = f"L{seg_len_L}_delta{delta_mag:g}_bs{batch_size}_seed{seed}"
        save_pt = os.path.join(out_dir, f"sudden_{tag}_results.pt")
        torch.save(res, save_pt)

        save_meta_pt = os.path.join(out_dir, f"sudden_{tag}_phi_oracle.pt")
        # store phi per batch + oracle theta*
        torch.save(dict(phi_hist=phi_hist, oracle_hist=oracle_hist), save_meta_pt)

        # plot
        cp_times = res[list(res.keys())[0]]["cp_times"]
        save_png = os.path.join(out_dir, f"sudden_{tag}_theta.png")
        plot_theta_tracking(
            res=res,
            oracle_hist=oracle_hist,
            cp_times=cp_times,
            batch_size=batch_size,
            title=f"Sudden-change theta tracking (L={seg_len_L}, Δphi2={delta_mag}, bs={batch_size}, seed={seed})",
            save_path=save_png,
        )

        print(f"[Saved] {save_pt}")
        print(f"[Saved] {save_meta_pt}")
        print(f"[Saved] {save_png}")

    print("All sudden-change grid experiments finished.")


if __name__ == "__main__":
    main()
