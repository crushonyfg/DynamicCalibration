# =============================================================
# run_synthetic_orthog_highdim_sweeps.py
#
# NeurIPS-ready synthetic sweeps (3 seeds) focusing on:
#   (A) 1D orthogonal discrepancy amplitude sweep (A)
#   (B) 1D drift slope sweep (v scale) with clear change points
#   (C) High-dim subspace-orthogonal discrepancy sweep over (d,k,A)
#
# For each (setting, seed):
#   - run multiple methods (usediscrepancy / nodiscrepancy / halfdiscrepancy)
#   - plot theta tracking overlay (all methods on ONE figure)
#   - compute metrics:
#        theta_track_rmse
#        pred_rmse_sim, pred_crps_sim
#        pred_rmse_full, pred_crps_full
#        avg_delay, n_false_alarms
#
# Output:
#   out_dir/
#     plots/...png
#     summary_raw.csv          (one row per method x seed x setting)
#     summary_agg.csv          (mean/std over seeds per method x setting)
#
# Notes:
# - Simulator functions are ELEMENTWISE to match your DeterministicSimulator
#   convention (particle-batch flattened to same leading dimension).
# - BOCPD change points are made obvious via large theta jumps + delta regime changes.
# =============================================================

import os
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- repo imports (same style as your existing scripts) ----
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian


# =============================================================
# Reproducibility
# =============================================================
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass


# =============================================================
# Metrics helpers
# =============================================================
def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)))

def l2_rmse_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def greedy_match_delays(
    gt_cps: List[int],
    det_cps: List[int],
    max_delay: Optional[int] = 15,
) -> Tuple[float, Dict]:
    det_sorted = sorted([int(x) for x in det_cps])
    used = [False] * len(det_sorted)

    delays = []
    matched = []
    for cp in gt_cps:
        cp = int(cp)
        for idx, d in enumerate(det_sorted):
            if used[idx]:
                continue
            if d >= cp:
                delay = d - cp
                if max_delay is not None and delay > max_delay:
                    break
                used[idx] = True
                delays.append(delay)
                matched.append((cp, d, delay))
                break

    avg_delay = float(np.mean(delays)) if len(delays) else float("nan")
    false_alarms = [d for idx, d in enumerate(det_sorted) if not used[idx]]
    dbg = dict(gt_cps=gt_cps, det_cps=det_sorted, matched=matched, delays=delays, false_alarms=false_alarms)
    return avg_delay, dbg


# =============================================================
# Prior samplers (same spirit as your slope script)
# =============================================================
def make_prior_sampler_uniform(theta_dim: int, lo: float, hi: float) -> Callable[[int], torch.Tensor]:
    lo, hi = float(lo), float(hi)
    width = hi - lo
    def sampler(N: int) -> torch.Tensor:
        return lo + width * torch.rand(N, theta_dim)
    return sampler

def make_prior_sampler_mixture_local_global(
    theta_dim: int,
    lo: float,
    hi: float,
    sigma_local: float = 0.20,
    p_global: float = 0.25,
) -> Callable[[int, Optional[torch.Tensor]], torch.Tensor]:
    lo, hi = float(lo), float(hi)
    def sampler(N: int, theta_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        N_global = int(p_global * N)
        N_local = N - N_global
        out = []
        if N_global > 0:
            out.append(lo + (hi - lo) * torch.rand(N_global, theta_dim))
        if theta_anchor is not None:
            anch = theta_anchor.reshape(1, theta_dim).repeat(N_local, 1)
            loc = anch + sigma_local * torch.randn(N_local, theta_dim)
            loc = torch.clamp(loc, lo, hi)
            out.append(loc)
        else:
            out.append(lo + (hi - lo) * torch.rand(N_local, theta_dim))
        return torch.cat(out, dim=0)
    return sampler


# =============================================================
# Elementwise simulators (IMPORTANT)
# =============================================================
def simulator_orthog_1d_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Elementwise 1D simulator:
      x:     (M,1) or (M,)
      theta: (M,2)
      out:   (M,1)

    y_s(x,theta)=theta1*sin(2πx)+theta2*sin(4πx)
    """
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    xx  = x[:, 0:1]
    th1 = theta[:, 0:1]
    th2 = theta[:, 1:2]
    ys = th1 * torch.sin(2 * torch.pi * xx) + th2 * torch.sin(4 * torch.pi * xx)
    return ys

def simulator_subspace_hd_torch(x: torch.Tensor, theta: torch.Tensor, k: int) -> torch.Tensor:
    """
    Elementwise high-dim simulator:
      x:     (M,d)
      theta: (M,k)
      out:   (M,1)

    y_s = sin(theta^T x_{1:k}) + 0.15*(theta^T x_{1:k})
    """
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    xk = x[:, :k]
    z = torch.sum(theta[:, :k] * xk, dim=1, keepdim=True)
    ys = torch.sin(z) + 0.15 * z
    return ys


# =============================================================
# 1D synthetic stream (orthogonal discrepancy)
# =============================================================
def make_delta_1d(delta_kind: str, amp: float) -> Callable[[np.ndarray], np.ndarray]:
    if delta_kind == "cos2":
        return lambda x: amp * np.cos(2 * np.pi * x.reshape(-1))
    if delta_kind == "cos8":
        return lambda x: amp * np.cos(8 * np.pi * x.reshape(-1))
    raise ValueError(f"Unknown delta_kind: {delta_kind}")

@dataclass
class SegmentSpec1D:
    start_batch: int
    delta_kind: str
    amp: float

class Orthog1DDataStream:
    """
    Batches on x in [0,1] with stratified jittered design.
    theta_t = theta0 + t*v + jumps
    delta_t piecewise (cos2 -> cos8) with amplitude A
    """
    def __init__(
        self,
        total_batches: int,
        batch_size: int,
        noise_sd: float,
        theta0: Tuple[float, float],
        theta_drift: Tuple[float, float],
        jump_batches: List[int],
        jump_sizes: List[Tuple[float, float]],
        delta_segments: List[SegmentSpec1D],
        seed: int,
    ):
        self.total_batches = int(total_batches)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)

        self.theta0 = np.array(theta0, dtype=float).reshape(2)
        self.theta_drift = np.array(theta_drift, dtype=float).reshape(2)

        self.jump_batches = list(jump_batches)
        self.jump_sizes = [np.array(js, dtype=float).reshape(2) for js in jump_sizes]
        assert len(self.jump_batches) == len(self.jump_sizes)

        self.delta_segments = sorted(delta_segments, key=lambda s: s.start_batch)
        self.rng = np.random.RandomState(seed)
        self.batch_idx = 0

        self.theta_true_hist: List[np.ndarray] = []
        self.cp_batches: List[int] = sorted(set(self.jump_batches + [s.start_batch for s in self.delta_segments if s.start_batch > 0]))

    def true_theta(self, t: int) -> np.ndarray:
        th = self.theta0 + self.theta_drift * t
        for jb, js in zip(self.jump_batches, self.jump_sizes):
            if t >= jb:
                th = th + js
        return th

    def _delta_fn(self, t: int) -> Callable[[np.ndarray], np.ndarray]:
        seg = None
        for s in self.delta_segments:
            if s.start_batch <= t:
                seg = s
            else:
                break
        assert seg is not None
        return make_delta_1d(seg.delta_kind, seg.amp)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx >= self.total_batches:
            raise StopIteration

        # stratified jitter x
        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]  # (B,1)

        th = self.true_theta(self.batch_idx)
        delta = self._delta_fn(self.batch_idx)(u)

        ys = th[0] * np.sin(2 * np.pi * u) + th[1] * np.sin(4 * np.pi * u)
        y = ys + delta + self.noise_sd * self.rng.randn(self.bs)

        self.theta_true_hist.append(th.copy())
        self.batch_idx += 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =============================================================
# High-dim synthetic stream (subspace orthogonality + covariate shift)
# =============================================================
class SubspaceHighDimDataStream:
    """
    x = (x1, x2):
      x1 in R^k ~ Uniform(0,1)
      x2 in R^(d-k) ~ N(mu_t*1, I) (mu_t shifts at change points)

    simulator depends only on x1: y_s = sin(theta^T x1) + 0.15*(theta^T x1)
    discrepancy depends only on x2: delta = A_t * cos(omega_t * u^T x2)
    """
    def __init__(
        self,
        total_batches: int,
        batch_size: int,
        d: int,
        k: int,
        noise_sd: float,
        theta0: np.ndarray,
        theta_drift: np.ndarray,
        jump_batches: List[int],
        jump_sizes: List[np.ndarray],
        A: float,
        mu_shifts: Tuple[float, float, float],
        omega_shifts: Tuple[float, float],
        seed: int,
    ):
        self.total_batches = int(total_batches)
        self.bs = int(batch_size)
        self.d = int(d)
        self.k = int(k)
        self.noise_sd = float(noise_sd)

        self.theta0 = np.asarray(theta0, dtype=float).reshape(self.k)
        self.theta_drift = np.asarray(theta_drift, dtype=float).reshape(self.k)

        self.jump_batches = list(jump_batches)
        self.jump_sizes = [np.asarray(js, dtype=float).reshape(self.k) for js in jump_sizes]
        assert len(self.jump_batches) == len(self.jump_sizes)

        self.A = float(A)
        self.mu0, self.mu1, self.mu2 = [float(x) for x in mu_shifts]   # segments [0,cp1), [cp1,cp2), [cp2, ...]
        self.w0, self.w1 = [float(x) for x in omega_shifts]            # segments [0,cp2), [cp2,...]

        self.rng = np.random.RandomState(seed)
        u = self.rng.normal(size=(self.d - self.k,))
        self.u = u / (np.linalg.norm(u) + 1e-12)

        self.batch_idx = 0
        self.theta_true_hist: List[np.ndarray] = []
        # change points: cp1 = jump_batches[0], cp2 = jump_batches[1] (assumed)
        self.cp_batches: List[int] = sorted(set(self.jump_batches))

    def true_theta(self, t: int) -> np.ndarray:
        th = self.theta0 + self.theta_drift * t
        for jb, js in zip(self.jump_batches, self.jump_sizes):
            if t >= jb:
                th = th + js
        return th

    def _mu_t(self, t: int) -> float:
        cp1, cp2 = self.cp_batches[0], self.cp_batches[1]
        if t < cp1:
            return self.mu0
        if t < cp2:
            return self.mu1
        return self.mu2

    def _omega_t(self, t: int) -> float:
        cp2 = self.cp_batches[1]
        return self.w0 if t < cp2 else self.w1

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx >= self.total_batches:
            raise StopIteration

        x1 = self.rng.rand(self.bs, self.k)  # uniform
        mu = self._mu_t(self.batch_idx)
        x2 = self.rng.randn(self.bs, self.d - self.k) + mu

        X = np.concatenate([x1, x2], axis=1)

        th = self.true_theta(self.batch_idx)
        z = (X[:, :self.k] * th.reshape(1, -1)).sum(axis=1)
        ys = np.sin(z) + 0.15 * z

        proj = X[:, self.k:] @ self.u
        omega = self._omega_t(self.batch_idx)
        # discrepancy turns on after first CP (make CP obvious)
        cp1 = self.cp_batches[0]
        A_t = 0.0 if self.batch_idx < cp1 else self.A
        delta = A_t * np.cos(omega * proj)

        y = ys + delta + self.noise_sd * self.rng.randn(self.bs)

        self.theta_true_hist.append(th.copy())
        self.batch_idx += 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =============================================================
# Method runner
# =============================================================
def _get_pred_fields(pred: Dict, pred_comp: Dict) -> Dict[str, torch.Tensor]:
    """
    Try to extract sim-only and full predictive mean/var with fallback.

    You may need to adjust keys if your OnlineBayesCalibrator uses different names.
    """
    mu_sim = pred.get("mu_sim", pred.get("mu", None))
    var_sim = pred_comp.get("var_sim", pred.get("var_sim", pred.get("var", None)))

    mu_full = pred.get("mu_full", pred.get("mu", mu_sim))
    var_full = pred_comp.get("var_full", pred_comp.get("var", var_sim))

    if not torch.is_tensor(mu_sim):
        mu_sim = torch.tensor(mu_sim)
    if not torch.is_tensor(mu_full):
        mu_full = torch.tensor(mu_full)

    if isinstance(var_sim, (float, int, np.floating, np.integer)):
        var_sim = torch.full_like(mu_sim, float(var_sim))
    elif not torch.is_tensor(var_sim):
        var_sim = torch.tensor(var_sim)

    if isinstance(var_full, (float, int, np.floating, np.integer)):
        var_full = torch.full_like(mu_full, float(var_full))
    elif not torch.is_tensor(var_full):
        var_full = torch.tensor(var_full)

    return dict(mu_sim=mu_sim, var_sim=var_sim, mu_full=mu_full, var_full=var_full)

def run_method_on_stream(
    stream_factory: Callable[[], object],
    simulator_func: Callable,
    theta_dim: int,
    method_meta: Dict,
    prior_lo: float,
    prior_hi: float,
) -> Dict:
    """
    Runs one method on a fresh stream (created by stream_factory).
    Returns history + summary metrics.
    """
    stream = stream_factory()

    cfg = CalibrationConfig()
    cfg.bocpd.bocpd_mode = method_meta.get("mode", "restart")
    cfg.bocpd.use_restart = True
    cfg.model.use_discrepancy = bool(method_meta.get("use_discrepancy", False))
    cfg.model.bocpd_use_discrepancy = bool(method_meta.get("bocpd_use_discrepancy", False))

    emulator = DeterministicSimulator(func=simulator_func, enable_autograd=True)

    prior_mix = make_prior_sampler_mixture_local_global(theta_dim, prior_lo, prior_hi)
    theta_anchor_holder = {"value": None}

    def prior_wrapper(N: int) -> torch.Tensor:
        return prior_mix(N, theta_anchor=theta_anchor_holder["value"])

    calib = OnlineBayesCalibrator(cfg, emulator, prior_wrapper)

    theta_est_hist = []
    theta_true_hist = []
    restart_hist = []

    rmse_sim_hist = []
    rmse_full_hist = []
    crps_sim_hist = []
    crps_full_hist = []

    for t in range(stream.total_batches):
        Xb, Yb = stream.next()

        # one-step-ahead evaluation (use t-1 state)
        if t > 0:
            pred = calib.predict_batch(Xb)
            pred_comp = calib.predict_complete(Xb, Yb)

            fields = _get_pred_fields(pred, pred_comp)
            mu_sim = fields["mu_sim"].reshape(-1)
            mu_full = pred["mu"].reshape(-1)

            rmse_sim_hist.append(rmse_torch(mu_sim, Yb))
            rmse_full_hist.append(rmse_torch(mu_full, Yb))

            try:
                var_sim = fields["var_sim"].reshape(-1)
                crps_sim_hist.append(float(crps_gaussian(mu_sim, var_sim, Yb)))
            except Exception:
                crps_sim_hist.append(float("nan"))

            try:
                var_full = pred["var"].reshape(-1)
                crps_full_hist.append(crps_gaussian(mu_full, var_full, Yb).mean().item())
            except Exception:
                crps_full_hist.append(float("nan"))

        rec = calib.step_batch(Xb, Yb, verbose=False)
        did_restart = bool(rec.get("did_restart", False))
        restart_hist.append(did_restart)

        # posterior mean of theta
        mean_theta, _, _, _ = calib._aggregate_particles(0.9)
        theta_anchor_holder["value"] = mean_theta.detach()
        theta_est_hist.append(mean_theta.detach().cpu().numpy().reshape(-1))

        th_true = np.asarray(stream.theta_true_hist[-1]).reshape(-1)
        theta_true_hist.append(th_true)

    # ---- summaries ----
    theta_errs = [l2_rmse_vec(te, tt) for te, tt in zip(theta_est_hist, theta_true_hist)]
    theta_track_rmse = float(np.mean(theta_errs))

    pred_rmse_sim = float(np.mean(rmse_sim_hist)) if rmse_sim_hist else float("nan")
    pred_rmse_full = float(np.mean(rmse_full_hist)) if rmse_full_hist else float("nan")
    pred_crps_sim = float(np.mean(crps_sim_hist)) if crps_sim_hist else float("nan")
    pred_crps_full = float(np.mean(crps_full_hist)) if crps_full_hist else float("nan")

    gt_cps = [int(c) for c in getattr(stream, "cp_batches", [])]
    det_cps = [i for i, f in enumerate(restart_hist) if f]
    avg_delay, dbg = greedy_match_delays(gt_cps, det_cps, max_delay=15)

    out = dict(
        theta_est=np.asarray(theta_est_hist),
        theta_true=np.asarray(theta_true_hist),
        restart=np.asarray(restart_hist, dtype=bool),
        summary=dict(
            theta_track_rmse=theta_track_rmse,
            pred_rmse_sim=pred_rmse_sim,
            pred_rmse_full=pred_rmse_full,
            pred_crps_sim=pred_crps_sim,
            pred_crps_full=pred_crps_full,
            avg_delay=avg_delay,
            n_false_alarms=len(dbg.get("false_alarms", [])),
            n_det_cp=len(det_cps),
            n_gt_cp=len(gt_cps),
        ),
    )
    return out


# =============================================================
# Plot: overlay all methods (one seed + one setting)
# =============================================================
def plot_theta_overlay(
    out_path: str,
    exp_title: str,
    theta_true: np.ndarray,
    theta_est_by_method: Dict[str, np.ndarray],
    gt_cps: List[int],
):
    theta_true = np.asarray(theta_true)
    theta_dim = theta_true.shape[1]
    T = theta_true.shape[0]
    xaxis = np.arange(T)

    fig, axes = plt.subplots(theta_dim, 1, figsize=(11, 3.0 * theta_dim), sharex=True)
    if theta_dim == 1:
        axes = [axes]

    for j in range(theta_dim):
        axes[j].plot(xaxis, theta_true[:, j], "k--", lw=2.2, label="true")

    for mname, theta_est in theta_est_by_method.items():
        theta_est = np.asarray(theta_est)
        for j in range(theta_dim):
            axes[j].plot(xaxis, theta_est[:, j], lw=1.8, label=mname if j == 0 else None)

    for j in range(theta_dim):
        for cp in gt_cps:
            axes[j].axvline(int(cp), color="k", alpha=0.18, lw=1)
        axes[j].set_ylabel(f"theta[{j}]")

    axes[-1].set_xlabel("batch index")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(exp_title, y=0.99)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =============================================================
# CSV writers (no overwrite, aggregated)
# =============================================================
def write_rows_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def aggregate_over_seeds(raw_rows: List[Dict], group_keys: List[str], metric_keys: List[str]) -> List[Dict]:
    # group values
    groups: Dict[Tuple, List[Dict]] = {}
    for r in raw_rows:
        k = tuple(r[g] for g in group_keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, rows in groups.items():
        out = {g: v for g, v in zip(group_keys, k)}
        out["n_seeds"] = len(set([rr["seed"] for rr in rows]))
        # for each metric, compute mean/std over seeds
        for mk in metric_keys:
            vals = [float(rr[mk]) for rr in rows if rr[mk] is not None and str(rr[mk]) != "nan"]
            if len(vals) == 0:
                out[mk + "_mean"] = float("nan")
                out[mk + "_std"] = float("nan")
            else:
                out[mk + "_mean"] = float(np.mean(vals))
                out[mk + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out_rows.append(out)
    return out_rows


# =============================================================
# Experiment definitions (sweeps)
# =============================================================
def exp_1d_amp_sweep(args) -> Tuple[List[Dict], List[Dict]]:
    """
    Sweep A for 1D orthogonal discrepancy.
    """
    raw_rows = []

    # clear, obvious change points
    total_batches = args.total_batches
    B = args.batch_size_1d
    noise_sd = args.noise_sd

    # big jumps to make CP obvious
    jump_batches = [20, 50]
    jump_sizes  = [(1.2, -0.8), (-1.0, 0.9)]  # larger than demo

    theta0 = (0.8, -0.3)
    base_v = np.array([0.004, -0.002], dtype=float)

    methods = args.methods

    for A in args.A_list:
        delta_segments = [
            SegmentSpec1D(0,  "cos2", 0.0),
            SegmentSpec1D(jump_batches[0], "cos2", float(A)),
            SegmentSpec1D(jump_batches[1], "cos8", float(A)),
        ]

        for seed in args.seeds:
            set_all_seeds(seed)

            def stream_factory():
                return Orthog1DDataStream(
                    total_batches=total_batches,
                    batch_size=B,
                    noise_sd=noise_sd,
                    theta0=theta0,
                    theta_drift=tuple(base_v),
                    jump_batches=jump_batches,
                    jump_sizes=jump_sizes,
                    delta_segments=delta_segments,
                    seed=seed,
                )

            # run all methods on identical data (same seed/params)
            theta_est_by_method = {}
            theta_true_ref = None
            gt_cps = None

            for mname, meta in methods.items():
                out = run_method_on_stream(
                    stream_factory=stream_factory,
                    simulator_func=simulator_orthog_1d_torch,
                    theta_dim=2,
                    method_meta=meta,
                    prior_lo=args.prior_lo,
                    prior_hi=args.prior_hi,
                )
                theta_est_by_method[mname] = out["theta_est"]
                if theta_true_ref is None:
                    theta_true_ref = out["theta_true"]
                    gt_cps = stream_factory().cp_batches

                s = out["summary"]
                raw_rows.append(dict(
                    experiment="1D_amp_sweep",
                    setting=f"A={A}",
                    A=A,
                    d="",
                    k="",
                    v_scale="",
                    seed=seed,
                    method=mname,
                    theta_track_rmse=s["theta_track_rmse"],
                    pred_rmse_sim=s["pred_rmse_sim"],
                    pred_crps_sim=s["pred_crps_sim"],
                    pred_rmse_full=s["pred_rmse_full"],
                    pred_crps_full=s["pred_crps_full"],
                    avg_delay=s["avg_delay"],
                    n_false_alarms=s["n_false_alarms"],
                ))

            # overlay plot
            plot_path = os.path.join(args.out_dir, "plots", f"1D_amp__A{A}__seed{seed}__theta_compare.png")
            plot_theta_overlay(
                out_path=plot_path,
                exp_title=f"1D orthog amp sweep | A={A} | seed={seed}",
                theta_true=theta_true_ref,
                theta_est_by_method=theta_est_by_method,
                gt_cps=gt_cps,
            )

    metric_keys = ["theta_track_rmse", "pred_rmse_full", "pred_crps_full", "avg_delay", "n_false_alarms", "pred_rmse_sim", "pred_crps_sim"]
    agg_rows = aggregate_over_seeds(raw_rows, group_keys=["experiment", "setting", "method"], metric_keys=metric_keys)
    return raw_rows, agg_rows


def exp_1d_slope_sweep(args) -> Tuple[List[Dict], List[Dict]]:
    """
    Sweep drift slope scale (v_scale) in 1D orthogonal discrepancy.
    """
    raw_rows = []

    total_batches = args.total_batches
    B = args.batch_size_1d
    noise_sd = args.noise_sd

    jump_batches = [20, 50]
    jump_sizes  = [(1.2, -0.8), (-1.0, 0.9)]  # keep CP obvious

    theta0 = (0.8, -0.3)
    base_v = np.array([0.004, -0.002], dtype=float)

    # fix A moderate-large
    A = float(args.slope_sweep_A)
    delta_segments = [
        SegmentSpec1D(0,  "cos2", 0.0),
        SegmentSpec1D(jump_batches[0], "cos2", A),
        SegmentSpec1D(jump_batches[1], "cos8", A),
    ]

    methods = args.methods

    for v_scale in args.v_scales:
        v = base_v * float(v_scale)

        for seed in args.seeds:
            set_all_seeds(seed)

            def stream_factory():
                return Orthog1DDataStream(
                    total_batches=total_batches,
                    batch_size=B,
                    noise_sd=noise_sd,
                    theta0=theta0,
                    theta_drift=tuple(v),
                    jump_batches=jump_batches,
                    jump_sizes=jump_sizes,
                    delta_segments=delta_segments,
                    seed=seed,
                )

            theta_est_by_method = {}
            theta_true_ref = None
            gt_cps = None

            for mname, meta in methods.items():
                out = run_method_on_stream(
                    stream_factory=stream_factory,
                    simulator_func=simulator_orthog_1d_torch,
                    theta_dim=2,
                    method_meta=meta,
                    prior_lo=args.prior_lo,
                    prior_hi=args.prior_hi,
                )
                theta_est_by_method[mname] = out["theta_est"]
                if theta_true_ref is None:
                    theta_true_ref = out["theta_true"]
                    gt_cps = stream_factory().cp_batches

                s = out["summary"]
                raw_rows.append(dict(
                    experiment="1D_slope_sweep",
                    setting=f"v_scale={v_scale}",
                    A=A,
                    d="",
                    k="",
                    v_scale=v_scale,
                    seed=seed,
                    method=mname,
                    theta_track_rmse=s["theta_track_rmse"],
                    pred_rmse_sim=s["pred_rmse_sim"],
                    pred_crps_sim=s["pred_crps_sim"],
                    pred_rmse_full=s["pred_rmse_full"],
                    pred_crps_full=s["pred_crps_full"],
                    avg_delay=s["avg_delay"],
                    n_false_alarms=s["n_false_alarms"],
                ))

            plot_path = os.path.join(args.out_dir, "plots", f"1D_slope__v{v_scale}__seed{seed}__theta_compare.png")
            plot_theta_overlay(
                out_path=plot_path,
                exp_title=f"1D orthog slope sweep | v_scale={v_scale} | A={A} | seed={seed}",
                theta_true=theta_true_ref,
                theta_est_by_method=theta_est_by_method,
                gt_cps=gt_cps,
            )

    metric_keys = ["theta_track_rmse", "pred_rmse_full", "pred_crps_full", "avg_delay", "n_false_alarms", "pred_rmse_sim", "pred_crps_sim"]
    agg_rows = aggregate_over_seeds(raw_rows, group_keys=["experiment", "setting", "method"], metric_keys=metric_keys)
    return raw_rows, agg_rows


def exp_hd_sweep(args) -> Tuple[List[Dict], List[Dict]]:
    """
    Sweep (d,k,A) for high-dim subspace orthogonality + covariate shift.
    """
    raw_rows = []

    total_batches = args.total_batches
    B = args.batch_size_hd
    noise_sd = args.noise_sd

    # obvious CP via big theta jumps + mu shifts + omega change
    jump_batches = [20, 50]

    methods = args.methods

    for (d, k) in args.dk_list:
        d = int(d); k = int(k)
        # theta configs in R^k (make jumps obvious)
        theta0 = np.linspace(0.6, -0.2, k)  # deterministic
        theta_drift = np.linspace(0.012, -0.004, k)  # mild drift
        jump_sizes = [
            np.linspace(1.0, -0.6, k),
            np.linspace(-0.9, 0.7, k),
        ]

        for A in args.A_list_hd:
            for seed in args.seeds:
                set_all_seeds(seed)

                def stream_factory():
                    return SubspaceHighDimDataStream(
                        total_batches=total_batches,
                        batch_size=B,
                        d=d,
                        k=k,
                        noise_sd=noise_sd,
                        theta0=theta0,
                        theta_drift=theta_drift,
                        jump_batches=jump_batches,
                        jump_sizes=jump_sizes,
                        A=float(A),
                        mu_shifts=(0.0, 1.5, -1.5),     # covariate shift more obvious
                        omega_shifts=(1.0, 2.0),        # omega changes at cp2
                        seed=seed,
                    )

                def sim_hd(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
                    return simulator_subspace_hd_torch(x, theta, k=k)

                theta_est_by_method = {}
                theta_true_ref = None
                gt_cps = None

                for mname, meta in methods.items():
                    out = run_method_on_stream(
                        stream_factory=stream_factory,
                        simulator_func=sim_hd,
                        theta_dim=k,
                        method_meta=meta,
                        prior_lo=args.prior_lo,
                        prior_hi=args.prior_hi,
                    )
                    theta_est_by_method[mname] = out["theta_est"]
                    if theta_true_ref is None:
                        theta_true_ref = out["theta_true"]
                        gt_cps = stream_factory().cp_batches

                    s = out["summary"]
                    raw_rows.append(dict(
                        experiment="HD_dkA_sweep",
                        setting=f"d={d},k={k},A={A}",
                        A=A,
                        d=d,
                        k=k,
                        v_scale="",
                        seed=seed,
                        method=mname,
                        theta_track_rmse=s["theta_track_rmse"],
                        pred_rmse_sim=s["pred_rmse_sim"],
                        pred_crps_sim=s["pred_crps_sim"],
                        pred_rmse_full=s["pred_rmse_full"],
                        pred_crps_full=s["pred_crps_full"],
                        avg_delay=s["avg_delay"],
                        n_false_alarms=s["n_false_alarms"],
                    ))

                plot_path = os.path.join(args.out_dir, "plots", f"HD__d{d}_k{k}_A{A}__seed{seed}__theta_compare.png")
                plot_theta_overlay(
                    out_path=plot_path,
                    exp_title=f"HD subspace sweep | d={d}, k={k}, A={A} | seed={seed}",
                    theta_true=theta_true_ref,
                    theta_est_by_method=theta_est_by_method,
                    gt_cps=gt_cps,
                )

    metric_keys = ["theta_track_rmse", "pred_rmse_full", "pred_crps_full", "avg_delay", "n_false_alarms", "pred_rmse_sim", "pred_crps_sim"]
    agg_rows = aggregate_over_seeds(raw_rows, group_keys=["experiment", "setting", "method"], metric_keys=metric_keys)
    return raw_rows, agg_rows


# =============================================================
# Main
# =============================================================
def build_methods() -> Dict[str, Dict]:
    return {
        "R-BOCPD-PF-usediscrepancy": dict(mode="restart", use_discrepancy=True,  bocpd_use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy":  dict(mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
        "R-BOCPD-PF-halfdiscrepancy":dict(mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
    }

def parse_dk_list(s: str) -> List[Tuple[int, int]]:
    # e.g. "20:3,50:5" -> [(20,3),(50,5)]
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        d_str, k_str = part.split(":")
        out.append((int(d_str), int(k_str)))
    return out

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/sweeps_orthog_hd")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--total_batches", type=int, default=80)

    parser.add_argument("--batch_size_1d", type=int, default=20)
    parser.add_argument("--batch_size_hd", type=int, default=32)
    parser.add_argument("--noise_sd", type=float, default=0.10)

    parser.add_argument("--prior_lo", type=float, default=-2.0)
    parser.add_argument("--prior_hi", type=float, default=2.0)

    # sweep controls
    parser.add_argument("--run_1d_amp", action="store_true", default=True)
    parser.add_argument("--run_1d_slope", action="store_true", default=True)
    parser.add_argument("--run_hd", action="store_true", default=True)

    parser.add_argument("--A_list", type=str, default="0,0.5,1,2,3,4")
    parser.add_argument("--v_scales", type=str, default="0,0.5,1,2")
    parser.add_argument("--slope_sweep_A", type=float, default=2.5)

    parser.add_argument("--dk_list", type=str, default="20:3,50:5,100:10")
    parser.add_argument("--A_list_hd", type=str, default="0,1,2,3")

    args = parser.parse_args()

    # parse lists
    args.seeds = parse_int_list(args.seeds)
    args.A_list = parse_float_list(args.A_list)
    args.v_scales = parse_float_list(args.v_scales)
    args.dk_list = parse_dk_list(args.dk_list)
    args.A_list_hd = parse_float_list(args.A_list_hd)

    args.methods = build_methods()

    os.makedirs(args.out_dir, exist_ok=True)

    # write into a run-stamped file name to avoid overwriting across reruns
    import datetime
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv = os.path.join(args.out_dir, f"summary_raw__{run_stamp}.csv")
    agg_csv = os.path.join(args.out_dir, f"summary_agg__{run_stamp}.csv")

    all_raw = []
    all_agg = []

    if args.run_1d_amp:
        raw, agg = exp_1d_amp_sweep(args)
        all_raw.extend(raw)
        all_agg.extend(agg)

    if args.run_1d_slope:
        raw, agg = exp_1d_slope_sweep(args)
        all_raw.extend(raw)
        all_agg.extend(agg)

    if args.run_hd:
        raw, agg = exp_hd_sweep(args)
        all_raw.extend(raw)
        all_agg.extend(agg)

    write_rows_csv(raw_csv, all_raw)
    write_rows_csv(agg_csv, all_agg)

    print("\nFinished sweeps.")
    print(f"Raw rows: {len(all_raw)} saved to {raw_csv}")
    print(f"Agg rows: {len(all_agg)} saved to {agg_csv}")
    print(f"Plots in: {os.path.join(args.out_dir, 'plots')}")

if __name__ == "__main__":
    main()
