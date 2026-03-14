"""
run_plantSim_comparison.py  —  DA / BC / Ours 三方法对比 (PlantSim 数据)

三种方法都在标准化空间 (x_base_s, theta_s, y_s) 下工作，与 run_plantSim_v3_std.py 一致。

Methods
-------
  DA   : PF-NoDiscrepancy  (简单粒子滤波，无 GP discrepancy)
  BC   : KOH Sliding Window  (GP 边际似然校准)
  Ours : R-BOCPD-PF-NoDiscrepancy  (重启 BOCPD + 粒子滤波，无 discrepancy)

Data modes
----------
  mode 0 : ordered by t   (gradual θ drift)
  mode 1 : mode 1 stream
  mode 2 : mode-0 + JumpPlan   (sudden θ jumps)

Output
------
  1) 每个 mode 的 theta 估计日志 (csv)
  2) 一张图: 每个 mode 一个 subplot, DA / BC / Ours vs Ground Truth

Usage
-----
  python -m calib.run_plantSim_comparison --csv physical_data.csv --modes 0 1 2
  python -m calib.run_plantSim_comparison --data_dir "path/to/PhysicalData_v3" --modes 0 2
"""

import os, math, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from time import time as timer
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

import warnings
warnings.filterwarnings("ignore")

# ---- calib sub-package imports ----
from .configs import CalibrationConfig
from .online_calibrator import OnlineBayesCalibrator
from .v3_utils import StreamClass, JumpPlan

# ---- from run_plantSim_v3_std (module-level code 已封装为函数, 导入安全) ----
from .run_plantSim_v3_std import (
    GlobalTransformSep,
    NNModelTorchStd,
    PlantEmulatorNNStd,
    batch_X_base_to_s,
    batch_y_to_s,
    init_pipeline,
    prior_sampler,
)


# =====================================================================
# DA : PF-NoDiscrepancy  (标准化空间, NN emulator)
# =====================================================================
class PFNoDiscrepancyNN:
    """
    简单粒子滤波 (无 GP discrepancy).
    在标准化空间运行: θ_s 粒子, 使用 PlantEmulatorNNStd 作为 simulator.
    似然:  p(y_s | x_s, θ_s) = N(y_s | NN(x_s, θ_s), σ²_s)
    """

    def __init__(
        self,
        emulator: PlantEmulatorNNStd,
        n_particles: int = 1024,
        theta_lo_s: float = -2.0,
        theta_hi_s: float = 2.0,
        sigma_obs_s: float = 1.0,
        resample_ess_ratio: float = 0.5,
        theta_move_std_s: float = 0.05,
        seed: int = 42,
    ):
        self.emu = emulator
        self.N = n_particles
        self.lo_s = theta_lo_s
        self.hi_s = theta_hi_s
        self.sigma2 = sigma_obs_s ** 2
        self.ess_ratio = resample_ess_ratio
        self.move_std = theta_move_std_s
        self.rng = np.random.default_rng(seed)

        # 初始化粒子 (标准化空间)
        self.theta_s = self.rng.uniform(self.lo_s, self.hi_s, size=self.N)
        self.logw = np.zeros(self.N) - np.log(self.N)

    def _normalize_logw(self):
        self.logw -= logsumexp(self.logw)

    def _ess(self) -> float:
        w = np.exp(self.logw)
        return 1.0 / np.sum(w ** 2)

    def _systematic_resample(self):
        w = np.exp(self.logw)
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        cumsum = np.cumsum(w)
        idx = np.searchsorted(cumsum, positions, side="left")
        idx = np.clip(idx, 0, self.N - 1)
        self.theta_s = self.theta_s[idx]
        self.logw[:] = -np.log(self.N)

    def _rejuvenate(self):
        self.theta_s += self.rng.normal(0.0, self.move_std, size=self.N)
        self.theta_s = np.clip(self.theta_s, self.lo_s, self.hi_s)

    def update_batch(self, Xb_s: torch.Tensor, Yb_s: torch.Tensor):
        """
        Xb_s: (B, 5) torch.Tensor — 标准化 x_base
        Yb_s: (B,)   torch.Tensor — 标准化 y
        """
        # 用 PlantEmulatorNNStd.predict 一次性计算所有粒子的预测
        theta_t = torch.tensor(self.theta_s, dtype=torch.float64).reshape(-1, 1)  # (N,1)
        mu_s, _ = self.emu.predict(Xb_s, theta_t)    # mu_s: (B, N)

        # 计算 log likelihood: sum over batch
        # mu_s[b, n] = NN(x_s[b], theta_s[n])
        # loglik[n] = sum_b -0.5 * [(y_s[b] - mu_s[b,n])^2 / sigma^2 + log(2pi*sigma^2)]
        mu_np = mu_s.detach().cpu().numpy()           # (B, N)
        y_np = Yb_s.detach().cpu().numpy()            # (B,)
        resid = y_np[:, None] - mu_np                 # (B, N)
        loglik = np.sum(
            -0.5 * (resid ** 2 / self.sigma2 + np.log(2 * np.pi * self.sigma2)),
            axis=0,
        )                                             # (N,)

        self.logw += loglik
        self._normalize_logw()

        if self._ess() < self.ess_ratio * self.N:
            self._systematic_resample()
            self._rejuvenate()

    def mean_theta_s(self) -> float:
        w = np.exp(self.logw)
        return float(np.sum(w * self.theta_s))


# =====================================================================
# BC : KOH Sliding Window  (NN emulator, 标准化空间)
# =====================================================================
class KOHSlidingWindowNN:
    """
    KOH-style batch calibration (profile marginal log-likelihood).
    GP kernel 在 5-d 标准化 x_base 空间上,  θ grid 在标准化空间.
    """

    def __init__(
        self,
        nn_emulator: NNModelTorchStd,
        theta_grid_s: np.ndarray,
        window_batches: int = 20,
        batch_size: int = 4,
        sigma_obs_s: float = 1.0,
        gp_lengthscale: float = 1.5,
        gp_signal_var: float = 1.0,
    ):
        self.nn = nn_emulator
        self.theta_grid_s = theta_grid_s
        self.W = window_batches * batch_size
        self.sigma2 = sigma_obs_s ** 2
        self.ls = gp_lengthscale
        self.sv = gp_signal_var
        self.X_buf: List[np.ndarray] = []
        self.Y_buf: List[np.ndarray] = []
        self.current_theta_s = float(np.median(theta_grid_s))

    def _nn_predict(self, X_s: np.ndarray, theta_s: float) -> np.ndarray:
        n = X_s.shape[0]
        th_col = np.full((n, 1), theta_s, dtype=np.float32)
        X_full = np.concatenate([X_s.astype(np.float32), th_col], axis=1)
        return self.nn.predict_y_s_from_Xfull_s(X_full)

    def update_batch(self, Xb_s: np.ndarray, Yb_s: np.ndarray):
        self.X_buf.append(Xb_s.copy())
        self.Y_buf.append(Yb_s.copy())

        X_all = np.concatenate(self.X_buf, axis=0)
        Y_all = np.concatenate(self.Y_buf, axis=0)

        if len(X_all) > self.W:
            X_all = X_all[-self.W:]
            Y_all = Y_all[-self.W:]
            self.X_buf = [X_all]
            self.Y_buf = [Y_all]

        n = len(X_all)
        if n < 5:
            return

        dist_sq = cdist(X_all, X_all, metric="sqeuclidean")
        K = self.sv * np.exp(-0.5 * dist_sq / self.ls ** 2)
        K += self.sigma2 * np.eye(n) + 1e-6 * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return

        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        const = n * np.log(2.0 * np.pi)

        log_ml = np.empty(len(self.theta_grid_s))
        for i, th_s in enumerate(self.theta_grid_s):
            ys = self._nn_predict(X_all, th_s)
            r = Y_all - ys
            alpha = np.linalg.solve(L, r)
            log_ml[i] = -0.5 * (np.dot(alpha, alpha) + logdet + const)

        w = np.exp(log_ml - logsumexp(log_ml))
        self.current_theta_s = float(np.sum(w * self.theta_grid_s))

    def mean_theta_s(self) -> float:
        return self.current_theta_s


# =====================================================================
# Plotting helpers
# =====================================================================
COLORS  = {"BC": "#e74c3c", "DA": "#2980b9", "Ours": "#27ae60"}
MARKERS = {"BC": "o", "DA": "s", "Ours": "^"}


def _plot_scenario(ax, batch_indices, gt_theta, method_results, title):
    ax.plot(batch_indices, gt_theta, "k--", lw=2.0, label="Ground Truth", zorder=5)
    for label in ["BC", "DA", "Ours"]:
        if label not in method_results:
            continue
        arr = method_results[label]
        ax.plot(
            batch_indices[: len(arr)], arr,
            color=COLORS.get(label, "gray"),
            marker=MARKERS.get(label, "."),
            markersize=2, linewidth=1.2, alpha=0.85,
            label=label,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Batch Index", fontsize=11)
    ax.set_ylabel(r"$\theta$ (minutes)", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)


# =====================================================================
# Stream helpers
# =====================================================================
def _iter_batches(stream: StreamClass, batch_size: int):
    while True:
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break


def _make_stream(mode: int, data_dir, csv_path):
    """与 run_plantSim_v3_std 保持一致的 stream 创建方式."""
    if mode == 2:
        jp = JumpPlan(
            max_jumps=5, min_gap_theta=500.0,
            min_interval=180, max_interval=320,
            min_jump_span=40, seed=7,
        )
        return StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
    return StreamClass(mode, folder=data_dir, csv_path=csv_path)


# =====================================================================
# Main
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PlantSim 三方法对比: DA / BC / Ours (标准化空间, 无OGP)",
    )
    parser.add_argument("--out_dir", type=str, default="figs/plantSim_comparison")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="PhysicalData_v3 directory (Excel files)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Aggregated physical-data CSV")
    parser.add_argument("--npz", type=str, default=None,
                        help="Computer-data NPZ (factory_aggregated.npz)")
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2],
                        help="Data modes (0=gradual, 1=mixed, 2=jumps)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--koh_window", type=int, default=20,
                        help="KOH sliding window (number of batches)")
    parser.add_argument("--n_particles", type=int, default=1024)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    batch_size = args.batch_size

    # ---- 初始化 NN pipeline (加载数据 + 训练/读取 NN 模型) ----
    gt_tf, nn_model, emu, a_s, b_s = init_pipeline(npz_path=args.npz)

    # sigma_obs 在标准化 y 空间
    sigma_obs_s = float(gt_tf.y_scaler.scale_[0])
    print(f"sigma_obs_s = {sigma_obs_s:.4f}")
    print(f"theta_s range: [{a_s:.3f}, {b_s:.3f}]")
    print(f"theta_raw range: [3.0, 21.0] minutes\n")

    # KOH theta grid (标准化空间)
    theta_grid_s = np.linspace(a_s, b_s, 200)

    MODE_NAMES = {
        0: "Gradual (mode 0)",
        1: "Mixed (mode 1)",
        2: "Sudden Jump (mode 2)",
    }

    all_results: Dict[int, dict] = {}

    for mode in args.modes:
        mode_label = MODE_NAMES.get(mode, f"mode {mode}")
        print(f"\n{'=' * 60}")
        print(f"  Mode {mode} : {mode_label}")
        print(f"{'=' * 60}")

        # ============================================================
        # (1) DA — PF-NoDiscrepancy (简单粒子滤波)
        # ============================================================
        print("\n--- DA (PF-NoDiscrepancy) ---")
        t0 = timer()

        pf = PFNoDiscrepancyNN(
            emulator=emu,
            n_particles=args.n_particles,
            theta_lo_s=a_s,
            theta_hi_s=b_s,
            sigma_obs_s=sigma_obs_s,
            resample_ess_ratio=0.5,
            theta_move_std_s=0.1 / gt_tf.theta_sd,
            seed=42,
        )

        stream_da = _make_stream(mode, args.data_dir, args.csv)
        da_theta, da_gt = [], []
        for Xb, yb, thb in tqdm(
            _iter_batches(stream_da, batch_size), desc="  DA  "
        ):
            newX = batch_X_base_to_s(gt_tf, Xb)   # (B, 5) torch
            newY = batch_y_to_s(gt_tf, yb)         # (B,)   torch

            pf.update_batch(newX, newY)
            mean_raw = gt_tf.theta_s_to_raw(pf.mean_theta_s())
            da_theta.append(mean_raw)
            da_gt.append(float(np.mean(thb)))
        print(f"  DA   done in {timer() - t0:.1f}s  ({len(da_theta)} batches)")

        # ============================================================
        # (2) BC — KOH Sliding Window
        # ============================================================
        print("\n--- BC (KOH Sliding Window) ---")
        t0 = timer()

        koh = KOHSlidingWindowNN(
            nn_emulator=nn_model,
            theta_grid_s=theta_grid_s,
            window_batches=args.koh_window,
            batch_size=batch_size,
            sigma_obs_s=sigma_obs_s,
            gp_lengthscale=1.5,
            gp_signal_var=1.0,
        )

        stream_bc = _make_stream(mode, args.data_dir, args.csv)
        bc_theta, bc_gt = [], []
        for Xb, yb, thb in tqdm(
            _iter_batches(stream_bc, batch_size), desc="  BC  "
        ):
            Xb_s = gt_tf.X_base_to_s(Xb)   # (B, 5) numpy
            Yb_s = gt_tf.y_raw_to_s(yb)     # (B,)   numpy
            koh.update_batch(Xb_s, Yb_s)

            mean_raw = gt_tf.theta_s_to_raw(koh.mean_theta_s())
            bc_theta.append(mean_raw)
            bc_gt.append(float(np.mean(thb)))
        print(f"  BC   done in {timer() - t0:.1f}s  ({len(bc_theta)} batches)")

        # ============================================================
        # (3) Ours — R-BOCPD-PF-NoDiscrepancy
        # ============================================================
        print("\n--- Ours (R-BOCPD-PF-NoDiscrepancy) ---")
        t0 = timer()

        cfg = CalibrationConfig()
        cfg.bocpd.bocpd_mode = "restart"
        cfg.bocpd.use_restart = True
        cfg.model.use_discrepancy = False
        cfg.model.refit_delta_every_batch = False
        cfg.model.bocpd_use_discrepancy = False
        cfg.model.sigma_eps = sigma_obs_s

        calib = OnlineBayesCalibrator(cfg, emu, prior_sampler)

        stream_ours = _make_stream(mode, args.data_dir, args.csv)
        ours_theta, ours_gt = [], []
        for Xb, yb, thb in tqdm(
            _iter_batches(stream_ours, batch_size), desc="  Ours"
        ):
            newX = batch_X_base_to_s(gt_tf, Xb)
            newY = batch_y_to_s(gt_tf, yb)

            calib.step_batch(newX, newY, verbose=False)

            mean_theta_s, var_theta_s, lo_s, hi_s = calib._aggregate_particles(0.9)
            mean_raw = gt_tf.theta_s_to_raw(float(mean_theta_s[0]))
            ours_theta.append(mean_raw)
            ours_gt.append(float(np.mean(thb)))
        print(f"  Ours done in {timer() - t0:.1f}s  ({len(ours_theta)} batches)")

        # ---- 汇总该 mode 的结果 ----
        n = min(len(da_theta), len(bc_theta), len(ours_theta))
        gt_arr = np.array(da_gt[:n])

        all_results[mode] = {
            "gt":   gt_arr,
            "DA":   np.array(da_theta[:n]),
            "BC":   np.array(bc_theta[:n]),
            "Ours": np.array(ours_theta[:n]),
        }

        # θ-RMSE
        for lbl in ["DA", "BC", "Ours"]:
            rmse = np.sqrt(np.mean((all_results[mode][lbl] - gt_arr) ** 2))
            print(f"    {lbl:>5s}  θ-RMSE = {rmse:.4f}")

        # ---- 保存 theta 估计日志 (CSV) ----
        log_path = os.path.join(out_dir, f"theta_log_mode{mode}.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_idx", "gt_theta", "DA_theta", "BC_theta", "Ours_theta"])
            for j in range(n):
                writer.writerow([
                    j,
                    f"{gt_arr[j]:.6f}",
                    f"{all_results[mode]['DA'][j]:.6f}",
                    f"{all_results[mode]['BC'][j]:.6f}",
                    f"{all_results[mode]['Ours'][j]:.6f}",
                ])
        print(f"  [Saved] θ log → {log_path}")

    # ==================================================================
    # 绘图 : 每个 mode 一个 subplot, DA / BC / Ours vs Ground Truth
    # ==================================================================
    n_modes = len(args.modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), squeeze=False)
    axes = axes.ravel()

    for i, mode in enumerate(args.modes):
        res = all_results[mode]
        bidx = np.arange(len(res["gt"]))
        _plot_scenario(
            axes[i], bidx, res["gt"],
            {"DA": res["DA"], "BC": res["BC"], "Ours": res["Ours"]},
            MODE_NAMES.get(mode, f"Mode {mode}"),
        )

    fig.suptitle("DA / BC / Ours  on PlantSim Data", fontsize=15, y=1.02)
    plt.tight_layout()
    fig_pdf = os.path.join(out_dir, "comparison_DA_BC_Ours.pdf")
    fig_png = os.path.join(out_dir, "comparison_DA_BC_Ours.png")
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] Figure → {fig_png}")

    # 保存原始结果
    results_path = os.path.join(out_dir, "comparison_results.pt")
    torch.save(all_results, results_path)
    print(f"[Saved] Results → {results_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
