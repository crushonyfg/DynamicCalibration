
# calib/run_thm_four_experiments_with_plots_v3.py
# ------------------------------------------------------------
# Run 4 experiments + generate figures/tables to ./figs/thm
#   E1: Stationary (delta=0, segmented but identical phi) => false alarm
#   E2: Sudden jumps (phi piecewise const)                => detection delay
#   E3: Gradual drift (theta* drifts)                     => stability
#   E4: Drift + jumps (drift plus occasional jumps)       => combined regime
#
# Interface-aligned with:
#   - run_synthetic_suddenCmp_tryThm.py
#   - run_synthetic_slope_deltaCmp.py
# ------------------------------------------------------------

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator

from .run_synthetic_suddenCmp_tryThm import (
    SuddenChangeDataStream,
    build_phi_segments_centered,
    physical_system,
    oracle_theta,
    computer_model_config2_torch,
)

from .run_synthetic_slope_deltaCmp import (
    ThetaDrivenSlopeDataStream,
    build_phi2_from_theta_star,
)

# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

def first_restart_time(diags: List[Dict[str, Any]]) -> Optional[int]:
    for d in diags:
        if int(d.get("did_restart", 0)) == 1:
            return int(d.get("t_end", -1))
    return None

def restart_count(diags: List[Dict[str, Any]]) -> int:
    return int(sum(int(d.get("did_restart", 0)) for d in diags))

def mean_restart_gap(diags: List[Dict[str, Any]]) -> Optional[float]:
    ts = [int(d.get("t_end")) for d in diags if int(d.get("did_restart", 0)) == 1]
    if len(ts) <= 1:
        return None
    gaps = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    return float(np.mean(gaps))

def detection_delay(diags: List[Dict[str, Any]], cp_time: int) -> Optional[int]:
    for d in diags:
        if int(d.get("t_end", -1)) >= int(cp_time) and int(d.get("did_restart", 0)) == 1:
            return int(d["t_end"]) - int(cp_time)
    return None

def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None

def plot_theta_with_restarts(
    out_png: str,
    title: str,
    diags: List[Dict[str, Any]],
    theta_hat: np.ndarray,
    theta_star: Optional[np.ndarray] = None,
    cp_times: Optional[List[int]] = None,
):
    ensure_dir(os.path.dirname(out_png))
    t = np.array([d["t_end"] for d in diags], dtype=int)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, theta_hat, label="theta_hat", alpha=0.9)
    if theta_star is not None and np.isfinite(theta_star).any():
        ax.plot(t, theta_star, label="theta_star", linestyle="--", alpha=0.9)

    if cp_times is not None:
        for c in cp_times:
            ax.axvline(c, linestyle=":", linewidth=1, alpha=0.6)

    rs_t = [d["t_end"] for d in diags if int(d.get("did_restart", 0)) == 1]
    if len(rs_t) > 0:
        y0 = float(np.nanmedian(theta_hat)) if np.isfinite(theta_hat).any() else 0.0
        ax.scatter(rs_t, [y0] * len(rs_t), marker="x", label="restart")

    ax.set_title(title)
    ax.set_xlabel("t (obs)")
    ax.set_ylabel("theta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_llr_panels(
    out_png: str,
    title: str,
    diags: List[Dict[str, Any]],
):
    ensure_dir(os.path.dirname(out_png))
    t = np.array([d["t_end"] for d in diags], dtype=int)

    dll = np.array([_safe_float(d.get("delta_ll_pair")) for d in diags], dtype=float)
    log_odds = np.array([_safe_float(d.get("log_odds_mass")) for d in diags], dtype=float)
    h_log = np.array([_safe_float(d.get("h_log")) for d in diags], dtype=float)
    restart = np.array([int(d.get("did_restart", 0)) for d in diags], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, dll)
    axes[0].set_ylabel("dll")

    axes[1].plot(t, log_odds, label="log_odds")
    axes[1].plot(t, h_log, label="h_log")
    axes[1].legend()
    axes[1].set_ylabel("log-odds / h")

    axes[2].step(t, restart, where="post")
    axes[2].set_ylabel("restart")
    axes[2].set_xlabel("t (obs)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_dll_hist(
    out_png: str,
    title: str,
    diags: List[Dict[str, Any]],
    max_abs: float = 50.0,
    drop_extreme: bool = True,
):
    ensure_dir(os.path.dirname(out_png))
    vals = []
    for d in diags:
        v = _safe_float(d.get("delta_ll_pair"))
        if v is None or not np.isfinite(v):
            continue
        if drop_extreme and abs(v) > max_abs:
            continue
        vals.append(v)
    vals = np.asarray(vals, dtype=float)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    if vals.size > 0:
        ax.hist(vals, bins=40)
        ax.set_title(f"{title}\nmean={vals.mean():.3f}, std={vals.std():.3f}, n={vals.size}")
    else:
        ax.set_title(f"{title}\n(empty)")
    ax.set_xlabel("dll")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_fa_raster(
    out_png: str,
    title: str,
    runs: List[Dict[str, Any]],
    horizon_T: int,
):
    ensure_dir(os.path.dirname(out_png))
    fig = plt.figure(figsize=(10, 3 + 0.25 * len(runs)))
    ax = fig.add_subplot(111)

    for i, r in enumerate(runs):
        ts = [int(d.get("t_end")) for d in r["diags"] if int(d.get("did_restart", 0)) == 1]
        y = np.full(len(ts), i)
        ax.scatter(ts, y)

    ax.set_xlim(0, horizon_T)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([str(r["seed"]) for r in runs])
    ax.set_title(title)
    ax.set_xlabel("t (obs)")
    ax.set_ylabel("seed")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

# ----------------------------
# Experiment runner
# ----------------------------

@dataclass
class MethodSpec:
    name: str
    use_discrepancy: bool

def make_calibrator(use_discrepancy: bool, restart_margin: float = 1.0) -> OnlineBayesCalibrator:
    cfg = CalibrationConfig()
    cfg.bocpd.bocpd_mode = "restart"
    cfg.bocpd.use_restart = True
    cfg.bocpd.restart_margin = float(restart_margin)
    cfg.model.use_discrepancy = bool(use_discrepancy)

    emulator = DeterministicSimulator(func=computer_model_config2_torch, enable_autograd=True)

    def prior_sampler(N: int):
        return torch.rand(N, 1) * 3.0

    return OnlineBayesCalibrator(cfg, emulator, prior_sampler)

def _theta_hat_from_calib(calib: OnlineBayesCalibrator) -> float:
    mean_theta, _, _, _ = calib._aggregate_particles(0.9)
    try:
        return float(mean_theta[0])
    except Exception:
        return float(mean_theta)

def run_one_method_on_stream(
    method: MethodSpec,
    stream,
    total_T: int,
    batch_size: int,
    seed: int,
    tag: str,
    out_dir_pt: str,
    restart_margin: float = 1.0,
) -> Dict[str, Any]:
    calib = make_calibrator(method.use_discrepancy, restart_margin=restart_margin)

    theta_hat: List[float] = []
    theta_star: List[float] = []
    diags: List[Dict[str, Any]] = []

    total_obs = 0
    while total_obs < total_T:
        Xb, Yb = stream.next()
        rec = calib.step_batch(Xb, Yb, verbose=False)
        if not isinstance(rec, dict):
            rec = {}

        theta_hat.append(_theta_hat_from_calib(calib))

        if hasattr(stream, "phi_history") and len(getattr(stream, "phi_history")) > 0:
            phi_t = stream.phi_history[-1]
            ths = oracle_theta(phi_t, np.linspace(0, 3, 600))
            theta_star.append(float(ths))
        elif hasattr(stream, "theta_star_history") and len(getattr(stream, "theta_star_history")) > 0:
            theta_star.append(float(stream.theta_star_history[-1]))
        else:
            theta_star.append(float("nan"))

        total_obs += batch_size

        d = dict(
            t_end=int(total_obs),
            did_restart=int(bool(rec.get("did_restart", False))),
            restart_start_time=rec.get("restart_start_time", None),
            s_star=rec.get("s_star", None),
            delta_ll_pair=rec.get("delta_ll_pair", None),
            delta_ll_max=rec.get("delta_ll_max", None),
            log_odds_mass=rec.get("log_odds_mass", None),
            h_log=rec.get("h_log", None),
            entropy=rec.get("entropy", None),
            anchor_idx=rec.get("anchor_idx", None),
            cand_idx=rec.get("cand_idx", None),
        )
        diags.append(d)

    out = dict(
        tag=tag,
        method=method.name,
        use_discrepancy=method.use_discrepancy,
        seed=int(seed),
        theta_hat=np.asarray(theta_hat, dtype=float),
        theta_star=np.asarray(theta_star, dtype=float),
        diags=diags,
    )

    ensure_dir(out_dir_pt)
    torch.save(out, os.path.join(out_dir_pt, f"{tag}__{method.name}__seed{seed}.pt"))
    return out

# ----------------------------
# Mixed drift + jumps stream
# ----------------------------

class DriftJumpThetaStream:
    # theta_star(t) = theta0 + slope*t + sum_{j: t>=tj} jump_j
    # phi(t) = [5, phi2_of_theta(theta_star(t)), 5]
    # Generates y = physical_system(x, phi(t)) + noise
    def __init__(
        self,
        total_T: int,
        batch_size: int,
        noise_sd: float,
        theta0: float,
        theta_slope: float,
        phi2_of_theta,
        jump_times: List[int],
        jump_sizes: List[float],
        phi_base: np.ndarray = np.array([5.0, 0.0, 5.0]),
        seed: int = 0,
    ):
        self.T = int(total_T)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)
        self.theta0 = float(theta0)
        self.theta_slope = float(theta_slope)
        self.phi2_of_theta = phi2_of_theta
        self.jump_times = list(map(int, jump_times))
        self.jump_sizes = list(map(float, jump_sizes))
        assert len(self.jump_times) == len(self.jump_sizes)

        self.phi_base = np.asarray(phi_base, dtype=float).copy()
        self.rng = np.random.RandomState(int(seed))
        self.t = 0
        self.phi_history: List[np.ndarray] = []
        self.theta_star_history: List[float] = []

    def _jump_offset(self, t: int) -> float:
        off = 0.0
        for tj, dj in zip(self.jump_times, self.jump_sizes):
            if t >= tj:
                off += float(dj)
        return off

    def true_theta_star(self, t: int) -> float:
        return self.theta0 + self.theta_slope * t + self._jump_offset(t)

    def true_phi(self, t: int) -> np.ndarray:
        th = self.true_theta_star(t)
        phi = self.phi_base.copy()
        phi[1] = float(self.phi2_of_theta(th))
        return phi

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.t >= self.T:
            raise StopIteration

        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]
        self.rng.shuffle(X)

        phi_t = self.true_phi(self.t)
        th_t = self.true_theta_star(self.t)

        y = physical_system(X, phi_t) + self.noise_sd * self.rng.randn(self.bs)

        self.phi_history.append(phi_t.copy())
        self.theta_star_history.append(float(th_t))

        self.t += self.bs

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

# ----------------------------
# Main
# ----------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./figs/thm")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--bs", type=int, default=20)
    parser.add_argument("--noise_sd", type=float, default=0.2)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--restart_margin", type=float, default=1.0)
    parser.add_argument("--seeds_E1", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    out_plots = os.path.join(out_dir, "plots")
    out_tables = os.path.join(out_dir, "tables")
    out_pt = os.path.join(out_dir, "pt")
    ensure_dir(out_plots)
    ensure_dir(out_tables)
    ensure_dir(out_pt)

    bs = int(args.bs)
    noise_sd = float(args.noise_sd)
    total_T = 800

    cp_times = [200, 400, 600]
    L = int(args.L)
    delta_mag = float(args.delta)

    theta_grid = np.linspace(0, 3, 400)
    phi2_grid = np.linspace(2.0, 12.0, 400)
    phi2_of_theta, _ = build_phi2_from_theta_star(
        phi2_grid=phi2_grid, theta_grid=theta_grid, a1=5.0, a3=5.0
    )

    methods = [
        MethodSpec(name="R-BOCPD-PF-nodiscrepancy", use_discrepancy=False),
        MethodSpec(name="R-BOCPD-PF-shared-discrepancy", use_discrepancy=True),
    ]

    # E1: Stationary
    tag = f"E1_stationary_T{total_T}_bs{bs}"
    print(f"\n===== {tag} =====")
    seed_list = [int(s) for s in args.seeds_E1.split(",") if s.strip() != ""]

    phi_segments0 = build_phi_segments_centered(delta=0.0, center=7.5)
    cp_times0 = cp_times

    e1_rows = []
    e1_runs_for_raster = {m.name: [] for m in methods}

    for m in methods:
        for sd in seed_list:
            stream = SuddenChangeDataStream(
                total_T=total_T, batch_size=bs, noise_sd=noise_sd,
                cp_times=cp_times0, phi_segments=phi_segments0, seed=int(sd)
            )
            out = run_one_method_on_stream(
                m, stream, total_T, bs, int(sd), tag, out_pt, restart_margin=float(args.restart_margin)
            )

            plot_theta_with_restarts(
                os.path.join(out_plots, f"{tag}__{m.name}__seed{sd}__theta.png"),
                title=f"{tag} | {m.name} | seed={sd}",
                diags=out["diags"],
                theta_hat=out["theta_hat"],
                theta_star=out["theta_star"],
                cp_times=None,
            )
            plot_llr_panels(
                os.path.join(out_plots, f"{tag}__{m.name}__seed{sd}__llr.png"),
                title=f"{tag} | {m.name} | seed={sd}",
                diags=out["diags"],
            )
            plot_dll_hist(
                os.path.join(out_plots, f"{tag}__{m.name}__seed{sd}__dll_hist.png"),
                title=f"{tag} | {m.name} | seed={sd}",
                diags=out["diags"],
                max_abs=50.0,
            )

            e1_runs_for_raster[m.name].append(dict(seed=int(sd), diags=out["diags"]))
            e1_rows.append(dict(
                experiment="E1",
                method=m.name,
                seed=int(sd),
                total_T=int(total_T),
                restarts=restart_count(out["diags"]),
                first_restart=first_restart_time(out["diags"]),
                mean_restart_gap=mean_restart_gap(out["diags"]),
            ))

    save_csv(
        os.path.join(out_tables, f"{tag}__summary.csv"),
        e1_rows,
        fieldnames=["experiment", "method", "seed", "total_T", "restarts", "first_restart", "mean_restart_gap"],
    )

    for m in methods:
        plot_fa_raster(
            os.path.join(out_plots, f"{tag}__{m.name}__FA_raster.png"),
            title=f"{tag} | {m.name} | false-alarm raster",
            runs=e1_runs_for_raster[m.name],
            horizon_T=total_T,
        )

    # E2: Sudden jumps
    tag = f"E2_sudden_L{L}_delta{delta_mag}_bs{bs}_seed{args.seed}"
    print(f"\n===== {tag} =====")
    phi_segments = build_phi_segments_centered(delta=delta_mag, center=7.5)

    e2_rows = []
    for m in methods:
        stream = SuddenChangeDataStream(
            total_T=total_T, batch_size=bs, noise_sd=noise_sd,
            cp_times=cp_times, phi_segments=phi_segments, seed=int(args.seed)
        )
        out = run_one_method_on_stream(
            m, stream, total_T, bs, int(args.seed), tag, out_pt, restart_margin=float(args.restart_margin)
        )

        plot_theta_with_restarts(
            os.path.join(out_plots, f"{tag}__{m.name}__theta.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            theta_hat=out["theta_hat"],
            theta_star=out["theta_star"],
            cp_times=cp_times,
        )
        plot_llr_panels(
            os.path.join(out_plots, f"{tag}__{m.name}__llr.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
        )
        plot_dll_hist(
            os.path.join(out_plots, f"{tag}__{m.name}__dll_hist.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            max_abs=50.0,
        )

        for c in cp_times:
            e2_rows.append(dict(
                experiment="E2",
                method=m.name,
                seed=int(args.seed),
                cp_time=int(c),
                delay=detection_delay(out["diags"], c),
                restarts=restart_count(out["diags"]),
            ))

    save_csv(
        os.path.join(out_tables, f"{tag}__delays.csv"),
        e2_rows,
        fieldnames=["experiment", "method", "seed", "cp_time", "delay", "restarts"],
    )

    # E3: Drift only
    tag = f"E3_drift_T{total_T}_bs{bs}_seed{args.seed}"
    print(f"\n===== {tag} =====")

    e3_rows = []
    for m in methods:
        stream = ThetaDrivenSlopeDataStream(
            total_T=total_T,
            batch_size=bs,
            noise_sd=noise_sd,
            theta0=1.5,
            theta_slope=1e-3,
            phi2_of_theta=phi2_of_theta,
            seed=int(args.seed),
        )
        out = run_one_method_on_stream(
            m, stream, total_T, bs, int(args.seed), tag, out_pt, restart_margin=float(args.restart_margin)
        )

        plot_theta_with_restarts(
            os.path.join(out_plots, f"{tag}__{m.name}__theta.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            theta_hat=out["theta_hat"],
            theta_star=out["theta_star"],
            cp_times=None,
        )
        plot_llr_panels(
            os.path.join(out_plots, f"{tag}__{m.name}__llr.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
        )
        plot_dll_hist(
            os.path.join(out_plots, f"{tag}__{m.name}__dll_hist.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            max_abs=20.0,
        )

        e3_rows.append(dict(
            experiment="E3",
            method=m.name,
            seed=int(args.seed),
            restarts=restart_count(out["diags"]),
            first_restart=first_restart_time(out["diags"]),
            mean_restart_gap=mean_restart_gap(out["diags"]),
        ))

    save_csv(
        os.path.join(out_tables, f"{tag}__summary.csv"),
        e3_rows,
        fieldnames=["experiment", "method", "seed", "restarts", "first_restart", "mean_restart_gap"],
    )

    # E4: Drift + jumps
    tag = f"E4_driftjump_T{total_T}_bs{bs}_seed{args.seed}"
    print(f"\n===== {tag} =====")

    jump_times = [200, 400, 600]
    jump_sizes = [0.8, 0.8, 0.8]

    e4_rows = []
    for m in methods:
        stream = DriftJumpThetaStream(
            total_T=total_T,
            batch_size=bs,
            noise_sd=noise_sd,
            theta0=1.5,
            theta_slope=1e-3,
            phi2_of_theta=phi2_of_theta,
            jump_times=jump_times,
            jump_sizes=jump_sizes,
            seed=int(args.seed),
        )
        out = run_one_method_on_stream(
            m, stream, total_T, bs, int(args.seed), tag, out_pt, restart_margin=float(args.restart_margin)
        )

        plot_theta_with_restarts(
            os.path.join(out_plots, f"{tag}__{m.name}__theta.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            theta_hat=out["theta_hat"],
            theta_star=out["theta_star"],
            cp_times=jump_times,
        )
        plot_llr_panels(
            os.path.join(out_plots, f"{tag}__{m.name}__llr.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
        )
        plot_dll_hist(
            os.path.join(out_plots, f"{tag}__{m.name}__dll_hist.png"),
            title=f"{tag} | {m.name}",
            diags=out["diags"],
            max_abs=50.0,
        )

        for c in jump_times:
            e4_rows.append(dict(
                experiment="E4",
                method=m.name,
                seed=int(args.seed),
                cp_time=int(c),
                delay=detection_delay(out["diags"], c),
                restarts=restart_count(out["diags"]),
            ))

    save_csv(
        os.path.join(out_tables, f"{tag}__delays.csv"),
        e4_rows,
        fieldnames=["experiment", "method", "seed", "cp_time", "delay", "restarts"],
    )

    print(f"\n[Done] outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
