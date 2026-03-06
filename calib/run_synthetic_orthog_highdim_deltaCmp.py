# =============================================================
# run_synthetic_orthog_highdim_deltaCmp.py
#
# Synthetic experiments for:
# (1) 1D function-space orthogonality vs collinearity:
#     y = y_s(x, theta_t) + delta_t(x) + eps
#     y_s(x,theta)=theta1*sin(2πx)+theta2*sin(4πx)
#     delta orthogonal: A*cos(2πx) or A*cos(8πx)
#     delta collinear:  A*sin(2πx)
#     theta_t: gradual drift + occasional jump
#     delta_t: changes at change points (amplitude/basis)
#
# (2) High-dim subspace orthogonality + covariate shift:
#     x in R^d, simulator depends only on first k dims,
#     discrepancy depends only on remaining dims,
#     and x_{k+1:d} distribution shifts at change points.
#
# Metrics:
#   - theta tracking RMSE (vs ground truth)
#   - predictive RMSE + CRPS (sim-only + full if available)
#   - average detection delay (restart-based)
#
# Plots:
#   - theta estimate vs truth with change-point markers
#
# This script mirrors your BOCPD-PF API usage pattern (cfg.bocpd.bocpd_mode,
# cfg.model.use_discrepancy, cfg.model.bocpd_use_discrepancy) as in
# run_synthetic_slope_deltaCmp.py.
# =============================================================

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Your existing modules (same style as your slope script)
# -------------------------------------------------------------
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .restart_bocpd_debug_260115_gpytorch import RollingStats


# =============================================================
# Utilities
# =============================================================
def _as_2d_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x

def _as_2d_torch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x[:, None]
    return x

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)))

def l2_rmse_vec(a: np.ndarray, b: np.ndarray) -> float:
    # per-step L2 error; return RMS over dims
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def greedy_match_delays(
    gt_cps: List[int],
    det_cps: List[int],
    max_delay: Optional[int] = None,
) -> Tuple[float, Dict]:
    """
    Greedy match: for each ground-truth change point (in batch index),
    match the first detection at/after it.

    Optionally cap delay by max_delay (in batches); detections beyond are treated as missing.

    Returns: (avg_delay, debug_dict)
    """
    det_sorted = sorted(det_cps)
    used = [False] * len(det_sorted)

    delays = []
    matched_pairs = []

    for cp in gt_cps:
        for idx, d in enumerate(det_sorted):
            if used[idx]:
                continue
            if d >= cp:
                delay = d - cp
                if (max_delay is not None) and (delay > max_delay):
                    break
                used[idx] = True
                delays.append(delay)
                matched_pairs.append((cp, d, delay))
                break

    avg_delay = float(np.mean(delays)) if len(delays) > 0 else float("nan")
    false_alarms = [d for idx, d in enumerate(det_sorted) if not used[idx]]

    dbg = dict(
        gt_cps=gt_cps,
        det_cps=det_sorted,
        matched=matched_pairs,
        delays=delays,
        false_alarms=false_alarms,
    )
    return avg_delay, dbg


# =============================================================
# 1D Orthogonality / Collinearity synthetic
# =============================================================
def simulator_orthog_1d_torch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    y_s(x,theta) = theta1*sin(2πx) + theta2*sin(4πx)

    IMPORTANT (matches your DeterministicSimulator convention):
    Your emulator flattens particle-batch pairs so that x and theta share the
    same leading dimension M = (batch_size * num_particles), and the simulator
    must operate pointwise (no outer product).

      x:     (M,1) or (M,)
      theta: (M,2)
      out:   (M,1)
    """
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    xx = x[:, 0:1]                 # (M,1)
    th1 = theta[:, 0:1]            # (M,1)
    th2 = theta[:, 1:2]            # (M,1)
    ys = th1 * torch.sin(2 * torch.pi * xx) + th2 * torch.sin(4 * torch.pi * xx)
    return ys

def make_delta_1d(delta_kind: str, amp: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    delta_kind:
      - "cos2": A*cos(2πx)
      - "cos8": A*cos(8πx)
      - "sin2": A*sin(2πx)  (collinear with theta1 basis)
    """
    if delta_kind == "cos2":
        return lambda x: amp * np.cos(2 * np.pi * x.reshape(-1))
    if delta_kind == "cos8":
        return lambda x: amp * np.cos(8 * np.pi * x.reshape(-1))
    if delta_kind == "sin2":
        return lambda x: amp * np.sin(2 * np.pi * x.reshape(-1))
    raise ValueError(f"Unknown delta_kind: {delta_kind}")

@dataclass
class SegmentSpec1D:
    start_batch: int
    delta_kind: str
    amp: float

class Orthog1DDataStream:
    """
    Streaming batches on x in [0,1] with stratified design.

    theta_t: gradual drift + jumps at specified change batches (batch indices).
    delta_t: piecewise (segment specs), changes at segment boundaries.
    """

    def __init__(
        self,
        total_batches: int = 80,
        batch_size: int = 20,
        noise_sd: float = 0.10,
        theta0: Tuple[float, float] = (0.8, -0.3),
        theta_drift: Tuple[float, float] = (0.004, -0.002),
        jump_batches: Optional[List[int]] = None,
        jump_sizes: Optional[List[Tuple[float, float]]] = None,
        delta_segments: Optional[List[SegmentSpec1D]] = None,
        seed: int = 0,
    ):
        self.total_batches = int(total_batches)
        self.bs = int(batch_size)
        self.noise_sd = float(noise_sd)

        self.theta0 = np.array(theta0, dtype=float).reshape(2)
        self.theta_drift = np.array(theta_drift, dtype=float).reshape(2)

        self.jump_batches = list(jump_batches or [])
        self.jump_sizes = list(jump_sizes or [])
        assert len(self.jump_batches) == len(self.jump_sizes), "jump_batches and jump_sizes length mismatch"

        if delta_segments is None:
            delta_segments = [
                SegmentSpec1D(0,  "cos2", 0.0),
                SegmentSpec1D(20, "cos2", 2.0),
                SegmentSpec1D(50, "cos8", 2.0),
            ]
        self.delta_segments = sorted(delta_segments, key=lambda s: s.start_batch)

        self.rng = np.random.RandomState(seed)
        self.batch_idx = 0

        # histories
        self.theta_true_hist: List[np.ndarray] = []
        self.delta_kind_hist: List[str] = []
        self.delta_amp_hist: List[float] = []
        self.cp_batches: List[int] = sorted(set(
            self.jump_batches + [s.start_batch for s in self.delta_segments if s.start_batch > 0]
        ))

    def true_theta(self, b: int) -> np.ndarray:
        th = self.theta0 + self.theta_drift * b
        for jb, js in zip(self.jump_batches, self.jump_sizes):
            if b >= jb:
                th = th + np.array(js, dtype=float)
        return th

    def _current_delta(self, b: int) -> Tuple[str, float, Callable[[np.ndarray], np.ndarray]]:
        seg = None
        for s in self.delta_segments:
            if s.start_batch <= b:
                seg = s
            else:
                break
        assert seg is not None
        return seg.delta_kind, seg.amp, make_delta_1d(seg.delta_kind, seg.amp)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx >= self.total_batches:
            raise StopIteration

        u = (np.arange(self.bs) + self.rng.rand(self.bs)) / self.bs
        X = u[:, None]  # (B,1)

        th = self.true_theta(self.batch_idx)
        delta_kind, amp, delta_fn = self._current_delta(self.batch_idx)

        ys = th[0] * np.sin(2 * np.pi * u) + th[1] * np.sin(4 * np.pi * u)
        y = ys + delta_fn(u) + self.noise_sd * self.rng.randn(self.bs)

        self.theta_true_hist.append(th.copy())
        self.delta_kind_hist.append(delta_kind)
        self.delta_amp_hist.append(float(amp))

        self.batch_idx += 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =============================================================
# High-dim subspace orthogonality synthetic
# =============================================================
def simulator_subspace_hd_torch(x: torch.Tensor, theta: torch.Tensor, k: int) -> torch.Tensor:
    """
    Elementwise high-dim simulator (matches your DeterministicSimulator convention):

      x:     (M,d)
      theta: (M,k)
      out:   (M,1)

    y_s(x,theta)=sin(theta^T x_{1:k}) + 0.15*(theta^T x_{1:k})
    """
    if x.dim() == 1:
        x = x[:, None]
    if theta.dim() == 1:
        theta = theta[None, :]
    xk = x[:, :k]                                        # (M,k)
    z = torch.sum(theta[:, :k] * xk, dim=1, keepdim=True) # (M,1)
    ys = torch.sin(z) + 0.15 * z
    return ys

class SubspaceHighDimDataStream:
    """
    x in R^d:
      - x[:k] ~ Uniform(0,1)
      - x[k:] ~ N(mu_seg, I)  (covariate shift at change points)

    theta_t in R^k: drift + jumps
    delta_t depends only on x[k:] via a fixed random direction u.
    """

    def __init__(
        self,
        total_batches: int = 80,
        batch_size: int = 32,
        d: int = 20,
        k: int = 3,
        noise_sd: float = 0.10,
        theta0: Optional[np.ndarray] = None,
        theta_drift: Optional[np.ndarray] = None,
        jump_batches: Optional[List[int]] = None,
        jump_sizes: Optional[List[np.ndarray]] = None,
        # discrepancy spec
        amp_segments: Optional[List[Tuple[int, float]]] = None,    # (start_batch, amplitude)
        omega_segments: Optional[List[Tuple[int, float]]] = None,  # (start_batch, omega)
        # covariate shift on x[k:]
        mu_segments: Optional[List[Tuple[int, float]]] = None,     # (start_batch, mean_shift_scalar)
        seed: int = 0,
    ):
        self.total_batches = int(total_batches)
        self.bs = int(batch_size)
        self.d = int(d)
        self.k = int(k)
        assert 1 <= self.k < self.d
        self.noise_sd = float(noise_sd)

        rng = np.random.RandomState(seed)
        self.rng = rng

        self.theta0 = (rng.uniform(-1.0, 1.0, size=(k,)) if theta0 is None else np.asarray(theta0, dtype=float).reshape(k))
        self.theta_drift = (rng.normal(scale=0.01, size=(k,)) if theta_drift is None else np.asarray(theta_drift, dtype=float).reshape(k))

        self.jump_batches = list(jump_batches or [])
        self.jump_sizes = [np.asarray(js, dtype=float).reshape(k) for js in (jump_sizes or [])]
        assert len(self.jump_batches) == len(self.jump_sizes)

        self.amp_segments = sorted(amp_segments or [(0, 0.0), (20, 2.0), (50, 2.0)], key=lambda t: t[0])
        self.omega_segments = sorted(omega_segments or [(0, 1.0), (50, 2.0)], key=lambda t: t[0])
        self.mu_segments = sorted(mu_segments or [(0, 0.0), (20, 1.0), (50, -1.0)], key=lambda t: t[0])

        u = rng.normal(size=(self.d - self.k,))
        u = u / (np.linalg.norm(u) + 1e-12)
        self.u = u

        self.batch_idx = 0
        self.theta_true_hist: List[np.ndarray] = []
        self.cp_batches: List[int] = sorted(set(
            self.jump_batches
            + [s for s, _ in self.amp_segments if s > 0]
            + [s for s, _ in self.mu_segments if s > 0]
        ))

    def true_theta(self, b: int) -> np.ndarray:
        th = self.theta0 + self.theta_drift * b
        for jb, js in zip(self.jump_batches, self.jump_sizes):
            if b >= jb:
                th = th + js
        return th

    def _piecewise_value(self, segments: List[Tuple[int, float]], b: int) -> float:
        val = segments[0][1]
        for s, v in segments:
            if s <= b:
                val = v
            else:
                break
        return float(val)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_idx >= self.total_batches:
            raise StopIteration

        xk = self.rng.rand(self.bs, self.k)
        mu_shift = self._piecewise_value(self.mu_segments, self.batch_idx)
        xrest = self.rng.randn(self.bs, self.d - self.k) + mu_shift
        X = np.concatenate([xk, xrest], axis=1)

        th = self.true_theta(self.batch_idx)
        amp = self._piecewise_value(self.amp_segments, self.batch_idx)
        omega = self._piecewise_value(self.omega_segments, self.batch_idx)

        z = (X[:, :self.k] * th.reshape(1, -1)).sum(axis=1)
        ys = np.sin(z) + 0.15 * z

        proj = (X[:, self.k:] @ self.u.reshape(-1, 1)).reshape(-1)
        delta = amp * np.cos(omega * proj)

        y = ys + delta + self.noise_sd * self.rng.randn(self.bs)

        self.theta_true_hist.append(th.copy())
        self.batch_idx += 1
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =============================================================
# BOCPD-PF runner (metrics + plots + summary table)
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
    sigma_local: float = 0.15,
    p_global: float = 0.2,
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

def _get_pred_fields(pred: Dict, pred_comp: Dict) -> Dict[str, torch.Tensor]:
    """
    Normalize prediction outputs across your variants.

    We try to extract:
      mu_sim, var_sim, mu_full, var_full
    """
    mu_sim = pred.get("mu_sim", pred.get("mu", None))
    var_sim = pred_comp.get("var_sim", pred.get("var_sim", pred.get("var", None)))

    mu_full = pred.get("mu", pred.get("mu_full", mu_sim))
    var_full = pred_comp.get("var", pred_comp.get("var_full", var_sim))

    # tensors
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

def run_one_stream(
    stream,
    simulator_torch_func: Callable,
    theta_dim: int,
    methods: Dict[str, Dict],
    out_dir: str,
    exp_tag: str,
    seed: int = 0,
    prior_lo: float = -2.0,
    prior_hi: float = 2.0,
    use_local_mixture_prior: bool = True,
    max_delay_batches: Optional[int] = 10,
):
    os.makedirs(out_dir, exist_ok=True)

    prior_sampler_global = make_prior_sampler_uniform(theta_dim, prior_lo, prior_hi)
    prior_sampler_mix = make_prior_sampler_mixture_local_global(theta_dim, prior_lo, prior_hi)

    results = {}
    gt_cps = [int(c) for c in getattr(stream, "cp_batches", [])]

    for name, meta in methods.items():
        print(f"[{exp_tag}] -> {name}")

        cfg = CalibrationConfig()
        cfg.bocpd.bocpd_mode = meta.get("mode", "restart")
        cfg.bocpd.use_restart = True

        if cfg.bocpd.bocpd_mode == "restart":
            cfg.model.use_discrepancy = bool(meta.get("use_discrepancy", False))
            cfg.model.bocpd_use_discrepancy = bool(meta.get("bocpd_use_discrepancy", False))

        emulator = DeterministicSimulator(func=simulator_torch_func, enable_autograd=True)

        if use_local_mixture_prior:
            theta_anchor_holder = {"value": None}
            def prior_wrapper(N: int) -> torch.Tensor:
                return prior_sampler_mix(N, theta_anchor=theta_anchor_holder["value"])
            calib = OnlineBayesCalibrator(cfg, emulator, prior_wrapper)
        else:
            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler_global)

        theta_est_hist: List[np.ndarray] = []
        theta_true_hist: List[np.ndarray] = []
        restart_hist: List[bool] = []

        rmse_sim_hist: List[float] = []
        rmse_full_hist: List[float] = []
        crps_sim_hist: List[float] = []
        crps_full_hist: List[float] = []

        roll = RollingStats(window=50)
        delta_ll_hist: List[float] = []

        for b in range(stream.total_batches):
            Xb, Yb = stream.next()

            if b > 0:
                pred = calib.predict_batch(Xb)
                pred_comp = calib.predict_complete(Xb, Yb)

                fields = _get_pred_fields(pred, pred_comp)
                mu_sim = fields["mu_sim"].reshape(-1)
                mu_full = pred["mu"].reshape(-1)

                rmse_sim_hist.append(rmse(mu_sim, Yb))
                rmse_full_hist.append(rmse(mu_full, Yb))

                # CRPS
                try:
                    var_sim = fields["var_sim"].reshape(-1)
                    crps_sim = crps_gaussian(mu_sim, var_sim, Yb)
                    crps_sim_hist.append(float(crps_sim))
                except Exception:
                    cs = pred_comp.get("crps_sim", None)
                    crps_sim_hist.append(float(cs) if cs is not None else float("nan"))
                # print("crps_sim", crps_sim_hist[-1])

                try:
                    var_full = pred["var"].reshape(-1)
                    crps_full = crps_gaussian(mu_full, var_full, Yb)
                    # print(crps_full.mean().item())
                    crps_full_hist.append(float(crps_full.mean().item()))
                except Exception:
                    cf = pred_comp.get("crps", pred_comp.get("crps_full", None))
                    crps_full_hist.append(float(cf) if cf is not None else float("nan"))

            rec = calib.step_batch(Xb, Yb, verbose=False)
            did_restart = bool(rec.get("did_restart", False))
            restart_hist.append(did_restart)

            dll = rec.get("delta_ll_pair", None)
            if dll is not None and np.isfinite(dll):
                roll.update(dll)
                delta_ll_hist.append(float(dll))
            else:
                delta_ll_hist.append(float("nan"))

            mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
            mean_theta_np = mean_theta.detach().cpu().numpy().reshape(-1)
            theta_est_hist.append(mean_theta_np)

            if use_local_mixture_prior:
                theta_anchor_holder["value"] = mean_theta.detach()

            th_true = np.asarray(stream.theta_true_hist[-1]).reshape(-1)
            theta_true_hist.append(th_true)

        theta_errs = [l2_rmse_vec(te, tt) for te, tt in zip(theta_est_hist, theta_true_hist)]
        theta_track_rmse = float(np.mean(theta_errs))

        pred_rmse_sim = float(np.mean(rmse_sim_hist)) if rmse_sim_hist else float("nan")
        pred_rmse_full = float(np.mean(rmse_full_hist)) if rmse_full_hist else float("nan")
        pred_crps_sim = float(np.mean(crps_sim_hist)) if crps_sim_hist else float("nan")
        pred_crps_full = float(np.mean(crps_full_hist)) if crps_full_hist else float("nan")

        det_cps = [i for i, f in enumerate(restart_hist) if f]
        avg_delay, delay_dbg = greedy_match_delays(gt_cps=gt_cps, det_cps=det_cps, max_delay=max_delay_batches)

        results[name] = dict(
            theta_est=np.asarray(theta_est_hist),
            theta_true=np.asarray(theta_true_hist),
            restart=np.asarray(restart_hist, dtype=bool),
            rmse_sim=np.asarray(rmse_sim_hist),
            rmse_full=np.asarray(rmse_full_hist),
            crps_sim=np.asarray(crps_sim_hist),
            crps_full=np.asarray(crps_full_hist),
            delta_ll=np.asarray(delta_ll_hist),
            summary=dict(
                theta_track_rmse=theta_track_rmse,
                pred_rmse_sim=pred_rmse_sim,
                pred_crps_sim=pred_crps_sim,
                pred_rmse_full=pred_rmse_full,
                pred_crps_full=pred_crps_full,
                avg_delay=avg_delay,
                delay_debug=delay_dbg,
                gt_cps=gt_cps,
                det_cps=det_cps,
            ),
            meta=meta,
        )

        # plots
        te = results[name]["theta_est"]
        tt = results[name]["theta_true"]
        B = te.shape[0]
        xaxis = np.arange(B)

        fig, axes = plt.subplots(theta_dim, 1, figsize=(10, 2.8 * theta_dim), sharex=True)
        if theta_dim == 1:
            axes = [axes]
        for j in range(theta_dim):
            axes[j].plot(xaxis, tt[:, j], "k--", lw=2, label="true")
            axes[j].plot(xaxis, te[:, j], lw=1.8, label="estimate")
            for cp in gt_cps:
                axes[j].axvline(cp, color="k", alpha=0.2, lw=1)
            axes[j].set_ylabel(f"theta[{j}]")
            axes[j].legend(loc="best")
        axes[-1].set_xlabel("batch index")
        fig.suptitle(f"{exp_tag} | {name} | theta tracking", y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{exp_tag}__{name}__theta.png"), dpi=200)
        plt.close(fig)

    # summary csv
    rows = []
    for mname, d in results.items():
        s = d["summary"]
        rows.append(dict(
            method=mname,
            theta_track_rmse=s["theta_track_rmse"],
            pred_rmse_sim=s["pred_rmse_sim"],
            pred_crps_sim=s["pred_crps_sim"],
            pred_rmse_full=s["pred_rmse_full"],
            pred_crps_full=s["pred_crps_full"],
            avg_delay=s["avg_delay"],
            n_gt_cp=len(s["gt_cps"]),
            n_det_cp=len(s["det_cps"]),
            n_false_alarms=len(s["delay_debug"]["false_alarms"]) if isinstance(s.get("delay_debug"), dict) else 0,
        ))

    csv_path = os.path.join(out_dir, f"{exp_tag}__summary.csv")
    # Append mode to avoid overwriting when you run methods one-by-one.
    write_header = (not os.path.exists(csv_path))
    with open(csv_path, "a", encoding="utf-8") as f:
        header = list(rows[0].keys()) if rows else []
        if write_header and header:
            f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    # Merge-save results across repeated calls (so you keep all methods)
    res_path = os.path.join(out_dir, f"{exp_tag}__results.pt")
    if os.path.exists(res_path):
        try:
            prev = torch.load(res_path, weights_only=False)
            if isinstance(prev, dict):
                prev.update(results)
                torch.save(prev, res_path)
            else:
                torch.save(results, res_path)
        except Exception:
            torch.save(results, res_path)
    else:
        torch.save(results, res_path)

    return results, csv_path




# =============================================================
# Overlay plot utility (compare theta estimates across methods)
# =============================================================
def plot_theta_overlay(out_dir: str, exp_tag: str, theta_dim: int):
    """
    After running all methods for a given exp_tag, call this to create an overlay plot:
      - dashed black: ground-truth theta
      - colored lines: each method's posterior mean estimate
    Saved to: {out_dir}/{exp_tag}__theta_compare.png
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    res_path = os.path.join(out_dir, f"{exp_tag}__results.pt")
    if not os.path.exists(res_path):
        print(f"[plot_theta_overlay] Missing results: {res_path}")
        return
    results = torch.load(res_path, weights_only=False)

    if not isinstance(results, dict) or len(results) == 0:
        print(f"[plot_theta_overlay] Empty results: {res_path}")
        return

    method_names = sorted(list(results.keys()))
    # Use first method to get truth + CPs
    first = results[method_names[0]]
    theta_true = np.asarray(first.get("theta_true", None))
    if theta_true is None or theta_true.size == 0:
        print("[plot_theta_overlay] Missing theta_true in results.")
        return

    # ground-truth CPs (batch indices)
    gt_cps = []
    try:
        gt_cps = list(first.get("summary", {}).get("gt_cps", []))
    except Exception:
        gt_cps = []

    B = theta_true.shape[0]
    xaxis = np.arange(B)

    fig, axes = plt.subplots(theta_dim, 1, figsize=(11, 2.9 * theta_dim), sharex=True)
    if theta_dim == 1:
        axes = [axes]

    # plot truth once per dim
    for j in range(theta_dim):
        axes[j].plot(xaxis, theta_true[:, j], "k--", lw=2.2, label="true")

    # plot each method estimate (matplotlib will auto-cycle colors)
    for m in method_names:
        theta_est = np.asarray(results[m].get("theta_est", None))
        if theta_est is None or theta_est.size == 0:
            continue
        for j in range(theta_dim):
            axes[j].plot(xaxis, theta_est[:, j], lw=1.8, label=m if j == 0 else None)

    # CP markers
    for j in range(theta_dim):
        for cp in gt_cps:
            axes[j].axvline(int(cp), color="k", alpha=0.18, lw=1)
        axes[j].set_ylabel(f"theta[{j}]")

    axes[-1].set_xlabel("batch index")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"{exp_tag} | theta comparison (all methods)", y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{exp_tag}__theta_compare.png"), dpi=220)
    plt.close(fig)

# =============================================================
# Main
# =============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="figs/orthog_highdim_deltaCmpv2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = args.out_dir
    seed = args.seed
    os.makedirs(out_dir, exist_ok=True)

    # Methods (same knobs as your existing scripts)
    methods = {
        "R-BOCPD-PF-usediscrepancy": dict(mode="restart", use_discrepancy=True,  bocpd_use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy":  dict(mode="restart", use_discrepancy=False, bocpd_use_discrepancy=False),
        "R-BOCPD-PF-halfdiscrepancy":dict(mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
    }

    total_batches = 80 if not args.debug else 30
    batch_size = 20 if not args.debug else 15

    jump_batches = [20, 50]
    jump_sizes = [(0.6, -0.3), (-0.5, 0.4)]

    # --------------------------
    # A: 1D orthogonal delta
    # --------------------------
    delta_segments_orthog = [
        SegmentSpec1D(0,  "cos2", 0.0),
        SegmentSpec1D(20, "cos2", 2.5),
        SegmentSpec1D(50, "cos8", 2.5),
    ]
    exp_tag = "1D_orthog_cos2_to_cos8"
    # reset per-experiment summary/results (otherwise later methods overwrite earlier ones)
    for _fn in [f"{exp_tag}__summary.csv", f"{exp_tag}__results.pt"]:
        _fp = os.path.join(out_dir, _fn)
        if os.path.exists(_fp):
            os.remove(_fp)
    for name in list(methods.keys()):
        stream = Orthog1DDataStream(
            total_batches=total_batches,
            batch_size=batch_size,
            noise_sd=0.10,
            theta0=(0.8, -0.3),
            theta_drift=(0.004, -0.002),
            jump_batches=jump_batches,
            jump_sizes=jump_sizes,
            delta_segments=delta_segments_orthog,
            seed=seed,
        )
        run_one_stream(
            stream=stream,
            simulator_torch_func=simulator_orthog_1d_torch,
            theta_dim=2,
            methods={name: methods[name]},
            out_dir=out_dir,
            exp_tag=exp_tag,
            seed=seed,
            prior_lo=-2.0,
            prior_hi=2.0,
        )

    plot_theta_overlay(out_dir=out_dir, exp_tag=exp_tag, theta_dim=2)

    # # --------------------------
    # # B: 1D collinear delta
    # # --------------------------
    # delta_segments_collinear = [
    #     SegmentSpec1D(0,  "sin2", 0.0),
    #     SegmentSpec1D(20, "sin2", 2.5),
    #     SegmentSpec1D(50, "sin2", 2.5),
    # ]
    # exp_tag = "1D_collinear_sin2"
    # # reset per-experiment summary/results (otherwise later methods overwrite earlier ones)
    # for _fn in [f"{exp_tag}__summary.csv", f"{exp_tag}__results.pt"]:
    #     _fp = os.path.join(out_dir, _fn)
    #     if os.path.exists(_fp):
    #         os.remove(_fp)
    # for name in list(methods.keys()):
    #     stream = Orthog1DDataStream(
    #         total_batches=total_batches,
    #         batch_size=batch_size,
    #         noise_sd=0.10,
    #         theta0=(0.8, -0.3),
    #         theta_drift=(0.004, -0.002),
    #         jump_batches=jump_batches,
    #         jump_sizes=jump_sizes,
    #         delta_segments=delta_segments_collinear,
    #         seed=seed,
    #     )
    #     run_one_stream(
    #         stream=stream,
    #         simulator_torch_func=simulator_orthog_1d_torch,
    #         theta_dim=2,
    #         methods={name: methods[name]},
    #         out_dir=out_dir,
    #         exp_tag=exp_tag,
    #         seed=seed,
    #         prior_lo=-2.0,
    #         prior_hi=2.0,
    #     )

    # plot_theta_overlay(out_dir=out_dir, exp_tag=exp_tag, theta_dim=2)

    # --------------------------
    # C: High-dim subspace orthog
    # --------------------------
    d = 10
    k = 4
    def sim_hd(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return simulator_subspace_hd_torch(x, theta, k=k)

    exp_tag = f"HD_subspace_orthog_d{d}_k{k}"
    # reset per-experiment summary/results (otherwise later methods overwrite earlier ones)
    for _fn in [f"{exp_tag}__summary.csv", f"{exp_tag}__results.pt"]:
        _fp = os.path.join(out_dir, _fn)
        if os.path.exists(_fp):
            os.remove(_fp)
    jump_batches_hd = [20, 50]
    jump_sizes_hd = [
        np.array([0.6, -0.2, 0.3]),
        np.array([-0.4, 0.5, -0.2]),
    ]
    for name in list(methods.keys()):
        stream = SubspaceHighDimDataStream(
            total_batches=total_batches,
            batch_size=32 if not args.debug else 24,
            d=d,
            k=k,
            noise_sd=0.10,
            theta0=np.array([0.6, -0.4, 0.2]),
            theta_drift=np.array([0.01, -0.005, 0.008]),
            jump_batches=jump_batches_hd,
            jump_sizes=jump_sizes_hd,
            amp_segments=[(0, 0.0), (20, 2.0), (50, 2.0)],
            omega_segments=[(0, 1.0), (50, 2.0)],
            mu_segments=[(0, 0.0), (20, 1.0), (50, -1.0)],
            seed=seed,
        )
        run_one_stream(
            stream=stream,
            simulator_torch_func=sim_hd,
            theta_dim=k,
            methods={name: methods[name]},
            out_dir=out_dir,
            exp_tag=exp_tag,
            seed=seed,
            prior_lo=-2.0,
            prior_hi=2.0,
        )

    plot_theta_overlay(out_dir=out_dir, exp_tag=exp_tag, theta_dim=k)

    print("\nAll experiments finished.")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
