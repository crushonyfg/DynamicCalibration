"""
BOCPD-PF on RCAM (PSim-RCAM) wind-bias dataset, using R-BOCPD-PF-nodiscrepancy.

用法（在 DynamicCalibration 根目录下）：

1) 先用 PSim-RCAM 生成基础飞行日志（test_data.csv）
   - 确保已在 `PSim-RCAM-main/PSim-RCAM-main` 目录下运行过 main.py，
     得到包含 u,v,w,phi,theta,psi,lat,lon,h,V_N,V_E,V_D 等列的 test_data.csv。

2) 在 DynamicCalibration 根目录下，运行本脚本：
   python -m calib.run_rcam_bocpd_pf --rcam_root "C:/Users/xxx/.../PSim-RCAM-main/PSim-RCAM-main" \\
                                     --data_csv "test_data.csv" \\
                                     --out_csv "rcam_stream_windjump.csv"

   这一步会：
   - 从 RCAM 的 test_data.csv 中读取时间序列；
   - 计算 Va_NED（由 u,v,w 和欧拉角得到的空速 NED）；
   - 构造 piecewise-constant + 小扰动 的风偏 theta_t（N,E,D 三维）；
   - 生成对应的观测 y_t = Va_NED + theta_t，并保存到 out_csv。

3) 在同一脚本中继续运行 R-BOCPD-PF-nodiscrepancy：
   - s_t / theta_t 统一放在 PF 粒子里的 theta 向量中（这里只用 3 维风偏 [bN,bE,bD]）；
   - emulator 是确定性的：y = Va_NED + theta；
   - 使用 OnlineBayesCalibrator + RestartBOCPD，配置为
     "R-BOCPD-PF-nodiscrepancy"（use_discrepancy=False, bocpd_use_discrepancy=False）。

说明：
- 不修改任何已有文件，只新建本脚本并复用已有模块（configs, emulator, online_calibrator, pf 等）。
- 如需多维 y（例如三维 V_N/V_E/V_D），依赖我们之前在 calib/ 中完成的多维 y 适配。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .configs import CalibrationConfig, ModelConfig, PFConfig, BOCPDConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator


# =============================================================
# 1) RCAM 工具：从 test_data.csv 构造 Va_NED, y, theta_true（风偏）
# =============================================================

R_EARTH = 6_378_137.0


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Body-to-NED rotation matrix.
    复制自 PSim-RCAM 中 run_ekf.py 的实现，避免改动原文件。
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cthe, sthe = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R = np.zeros((3, 3))
    R[0, 0] = cthe * cpsi
    R[0, 1] = sphi * sthe * cpsi - cphi * spsi
    R[0, 2] = cphi * sthe * cpsi + sphi * spsi
    R[1, 0] = cthe * spsi
    R[1, 1] = sphi * sthe * spsi + cphi * cpsi
    R[1, 2] = cphi * sthe * spsi - sphi * cpsi
    R[2, 0] = -sthe
    R[2, 1] = sphi * cthe
    R[2, 2] = cphi * cthe
    return R


@dataclass
class RCAMWindJumpConfig:
    """
    控制 piecewise-constant 风偏 schedule 的简单配置。
    """
    dt: float = 0.1                 # 采样间隔（秒），与 run_ekf 中一致
    segment_length_s: float = 60.0  # 每个风段的持续时间（秒）
    num_segments: int = 5           # 段数
    base_bias_scale: float = 5.0    # 每段风偏均值的尺度（m/s）
    noise_std: float = 0.3          # 段内小扰动的标准差（m/s）


def build_wind_jump_schedule(
    T: int,
    cfg: RCAMWindJumpConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成长度为 T 的风偏时间序列 theta_true[t,3]（N,E,D）。
    - 每个 segment 内风偏近似常数；
    - segment 之间会 jump；
    - 段内叠加小的高斯扰动。
    """
    seg_len_steps = max(int(cfg.segment_length_s / cfg.dt), 1)
    num_segments = min(cfg.num_segments, max((T + seg_len_steps - 1) // seg_len_steps, 1))

    # 为每个 segment 采样一个三维均值（单位 m/s）
    base_thetas = rng.normal(
        loc=0.0,
        scale=cfg.base_bias_scale,
        size=(num_segments, 3),
    )

    theta_true = np.zeros((T, 3), dtype=float)
    for seg_idx in range(num_segments):
        start = seg_idx * seg_len_steps
        end = min((seg_idx + 1) * seg_len_steps, T)
        if start >= end:
            break
        base = base_thetas[seg_idx]    # [3]
        noise = rng.normal(loc=0.0, scale=cfg.noise_std, size=(end - start, 3))
        theta_true[start:end, :] = base[None, :] + noise

    # 截断到长度 T
    return theta_true[:T, :]


def build_rcam_windjump_stream(
    csv_path: str,
    cfg_wind: RCAMWindJumpConfig,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 PSim-RCAM 的 test_data.csv 构造校准用的 stream：
      - X_t: Va_NED(t) ∈ R^3
      - y_t: 观测 V_NED_meas(t) = Va_NED(t) + theta_true(t) ∈ R^3
      - theta_true(t): piecewise-constant + 扰动的真实风偏 ∈ R^3
      - t_seq: 对应的时间戳（如果存在 PSim_Time 列，否则用步数×dt）
    """
    df = pd.read_csv(csv_path)
    if "PSim_Time" in df.columns:
        t_seq = df["PSim_Time"].to_numpy()
        # 覆盖 dt，以数据为准（假设近似等间隔）
        if len(t_seq) >= 2:
            cfg_wind.dt = float(t_seq[1] - t_seq[0])
    else:
        t_seq = np.arange(len(df), dtype=float) * cfg_wind.dt

    # 计算 Va_NED：与 run_ekf 中一致
    Va_seq = []
    for _, row in df.iterrows():
        V_body = np.array([row["u"], row["v"], row["w"]], dtype=float)
        R_b2n = euler_to_dcm(float(row["phi"]), float(row["theta"]), float(row["psi"]))
        Va_seq.append(R_b2n @ V_body)
    Va_seq = np.stack(Va_seq, axis=0)  # [T,3]

    T = Va_seq.shape[0]
    rng = np.random.default_rng(seed)
    theta_true = build_wind_jump_schedule(T, cfg_wind, rng)  # [T,3]

    # 观测 y = Va + theta_true
    y_seq = Va_seq + theta_true

    return Va_seq, y_seq, theta_true, t_seq


def save_rcam_stream_csv(
    out_csv: str,
    Va_seq: np.ndarray,
    y_seq: np.ndarray,
    theta_true: np.ndarray,
    t_seq: np.ndarray,
) -> None:
    """
    将构造好的 stream 写到 csv，便于离线调试。
    列包括：
      - t
      - Va_N, Va_E, Va_D
      - y_VN, y_VE, y_VD
      - thetaN_true, thetaE_true, thetaD_true
    """
    T = Va_seq.shape[0]
    assert y_seq.shape == (T, 3)
    assert theta_true.shape == (T, 3)
    assert t_seq.shape[0] == T

    df_out = pd.DataFrame(
        {
            "t": t_seq,
            "Va_N": Va_seq[:, 0],
            "Va_E": Va_seq[:, 1],
            "Va_D": Va_seq[:, 2],
            "y_VN": y_seq[:, 0],
            "y_VE": y_seq[:, 1],
            "y_VD": y_seq[:, 2],
            "thetaN_true": theta_true[:, 0],
            "thetaE_true": theta_true[:, 1],
            "thetaD_true": theta_true[:, 2],
        }
    )
    df_out.to_csv(out_csv, index=False)


# =============================================================
# 2) Emulator: y = Va_NED + theta  （多维 y, 无 discrepancy）
# =============================================================


def make_rcam_emulator() -> DeterministicSimulator:
    """
    构造一个 DeterministicSimulator：
      输入:
        x: [b, 3]  -- Va_NED
        theta: [N, 3] -- 粒子中的风偏 (bN,bE,bD)
      输出:
        y: [b, 3]  -- V_NED = Va_NED + theta
    """

    def f_eta(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # x: [b,3] 或 [3]; theta: [1,3]
        if x.dim() == 1:
            x_ = x[None, :]
        else:
            x_ = x
        # theta[0] : [3]
        b = theta[0, :]
        # 广播：x_[b,3] + b[3] -> [b,3]
        y = x_ + b[None, :]
        return y  # [b,3]

    return DeterministicSimulator(f_eta, enable_autograd=False)


def make_prior_sampler(theta_scale: float = 10.0, d_theta: int = 3) -> Callable[[int], torch.Tensor]:
    """
    简单各向同性高斯先验：theta ~ N(0, theta_scale^2 I_d)。
    """

    def prior_sampler(num_particles: int) -> torch.Tensor:
        return theta_scale * torch.randn(num_particles, d_theta, dtype=torch.float64)

    return prior_sampler


# =============================================================
# 3) 构造 CalibrationConfig（R-BOCPD-PF-nodiscrepancy）
# =============================================================


def make_rcam_calib_config(
    use_cuda: bool = False,
    num_particles: int = 512,
) -> CalibrationConfig:
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    model_cfg = ModelConfig()
    model_cfg.rho = 1.0
    model_cfg.sigma_eps = 0.5  # 观测噪声的标度（可根据 RCAM EKF 的设定微调）
    model_cfg.emulator_type = "deterministic"
    model_cfg.device = device
    model_cfg.dtype = dtype
    # 关键：禁用 discrepancy
    model_cfg.use_discrepancy = False
    # R-BOCPD 在 ump_batch 里用到 bocpd_use_discrepancy 字段
    # 这里通过 setattr 动态添加，沿用 run_synthetic_slope_deltaCmp.py 的用法
    setattr(model_cfg, "bocpd_use_discrepancy", False)

    pf_cfg = PFConfig()
    pf_cfg.num_particles = num_particles
    pf_cfg.resample_ess_ratio = 0.5
    pf_cfg.move_strategy = "random_walk"
    pf_cfg.random_walk_scale = 0.5

    bocpd_cfg = BOCPDConfig()
    bocpd_cfg.hazard_lambda = 600.0  # 期望 run-length，大概 600 步左右
    bocpd_cfg.hazard_type = "geometric"
    bocpd_cfg.bocpd_mode = "restart"
    bocpd_cfg.max_experts = 5
    bocpd_cfg.use_restart = True
    bocpd_cfg.restart_threshold = 0.85
    bocpd_cfg.restart_margin = 1.0
    bocpd_cfg.restart_cooldown = 20
    bocpd_cfg.restart_criteria = "rank_change"

    cfg = CalibrationConfig(model=model_cfg, pf=pf_cfg, bocpd=bocpd_cfg)
    return cfg


# =============================================================
# 4) 主函数：生成 RCAM wind-jump 数据 + 运行 R-BOCPD-PF-nodiscrepancy
# =============================================================


def run_rcam_bocpd_pf(
    rcam_root: str,
    data_csv: str,
    out_stream_csv: str,
    seed: int = 0,
    use_cuda: bool = False,
    num_particles: int = 512,
    batch_size: int = 20,
    plot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    rcam_root: PSim-RCAM-main/PSim-RCAM-main 的路径（仅用于文档说明；本函数不直接调用 main.py）
    data_csv:  已存在的 RCAM 日志 csv（例如 rcam_root/test_data.csv）
    out_stream_csv: 输出我们构造的 wind-jump stream
    batch_size: 每批步数，按批更新并记录 theta_est / y_pred 用于画图
    plot_dir: 若给定，将 theta 追踪图与 y_pred 对比图保存到此目录

    返回：
      - results: dict，包含 run-length, theta 轨迹、y_pred 轨迹及绘图路径等。
    """
    # 1) 构造 stream
    wind_cfg = RCAMWindJumpConfig()
    Va_seq, y_seq, theta_true, t_seq = build_rcam_windjump_stream(data_csv, wind_cfg, seed=seed)
    save_rcam_stream_csv(out_stream_csv, Va_seq, y_seq, theta_true, t_seq)

    T = Va_seq.shape[0]
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # 2) 构造 emulator + prior + calibrator
    emulator = make_rcam_emulator()
    prior_sampler = make_prior_sampler(theta_scale=10.0, d_theta=3)
    cfg = make_rcam_calib_config(use_cuda=use_cuda, num_particles=num_particles)

    calibrator = OnlineBayesCalibrator(
        calib_cfg=cfg,
        emulator=emulator,
        prior_sampler=prior_sampler,
        init_delta_state=None,
        delta_fitter=None,
        on_restart=None,
        notify_on_restart=True,
    )

    X_all = torch.from_numpy(Va_seq).to(device=device, dtype=dtype)   # [T,3]
    Y_all = torch.from_numpy(y_seq).to(device=device, dtype=dtype)   # [T,3]

    # 3) 按 batch 逐批更新，记录 theta_est、theta_true、y_pred、y_obs
    theta_est_list: List[np.ndarray] = []   # 每批一个 [3]
    theta_true_list: List[np.ndarray] = []  # 每批一个 [3]（取该批末尾或均值）
    y_pred_list: List[np.ndarray] = []      # 每批 [B,3]
    y_obs_list: List[np.ndarray] = []       # 每批 [B,3]
    t_batch_list: List[np.ndarray] = []     # 每批时间戳 [B]

    k = 0
    batch_idx = 0
    while k < T:
        B = min(batch_size, T - k)
        Xb = X_all[k : k + B]
        Yb = Y_all[k : k + B]
        tb = t_seq[k : k + B]

        # 有 expert 时用当前 posterior 做预测（本批的 y_pred）
        if len(calibrator.bocpd.experts) > 0:
            pred = calibrator.predict_batch(Xb)
            mu = pred["mu"]  # [B] or [B,3]
            if isinstance(mu, torch.Tensor):
                mu_np = mu.detach().cpu().numpy()
            else:
                mu_np = np.asarray(mu)
            if mu_np.ndim == 1:
                mu_np = mu_np[:, None]
            y_pred_list.append(mu_np)
        else:
            y_pred_list.append(np.full((B, 3), np.nan))

        y_obs_list.append(Yb.detach().cpu().numpy())
        t_batch_list.append(tb)

        # 更新
        calibrator.step_batch(Xb, Yb, verbose=False)

        # 混合 theta（所有 expert 的加权均值的加权和）
        mean_theta, _ = calibrator._aggregate_particles(quantile=None)
        if mean_theta is not None:
            theta_est_list.append(mean_theta.detach().cpu().numpy())
        else:
            theta_est_list.append(np.full(3, np.nan))

        # 本批对应的真实 theta：取批末尾时刻
        theta_true_list.append(theta_true[k + B - 1, :].copy())

        k += B
        batch_idx += 1

    theta_est_arr = np.stack(theta_est_list, axis=0)   # [n_batch, 3]
    theta_true_arr = np.stack(theta_true_list, axis=0) # [n_batch, 3]
    # 时间轴：每批一个点，用该批末尾时间
    t_plot = np.array([t_batch_list[i][-1] for i in range(len(t_batch_list))])

    # 4) 绘图：theta 估计 vs 真实
    fig_theta_path: Optional[str] = None
    fig_y_path: Optional[str] = None
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

        fig1, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["bN (N)", "bE (E)", "bD (D)"]
        for j in range(3):
            axes[j].plot(t_plot, theta_true_arr[:, j], "k-", alpha=0.8, label="theta_true")
            axes[j].plot(t_plot, theta_est_arr[:, j], "b-", alpha=0.7, label="theta_est")
            axes[j].set_ylabel(labels[j])
            axes[j].legend(loc="upper right")
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig1.suptitle("RCAM wind bias: theta estimate vs true (R-BOCPD-PF-nodiscrepancy)")
        fig1.tight_layout()
        fig_theta_path = os.path.join(plot_dir, "rcam_theta_tracking.png")
        fig1.savefig(fig_theta_path, dpi=150)
        plt.close(fig1)

        # 5) 绘图：y_pred vs y_obs（按时间展平）
        y_pred_flat = np.concatenate(y_pred_list, axis=0)  # [T, 3]
        y_obs_flat = np.concatenate(y_obs_list, axis=0)    # [T, 3]
        valid = ~np.isnan(y_pred_flat[:, 0])
        if np.any(valid):
            t_flat = np.concatenate(t_batch_list, axis=0)
            fig2, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            y_labels = ["V_N", "V_E", "V_D"]
            for j in range(3):
                axes[j].plot(t_flat, y_obs_flat[:, j], "k-", alpha=0.6, label="y_obs")
                axes[j].plot(t_flat, y_pred_flat[:, j], "b-", alpha=0.7, label="y_pred")
                axes[j].set_ylabel(y_labels[j])
                axes[j].legend(loc="upper right")
                axes[j].grid(True, alpha=0.3)
            axes[-1].set_xlabel("Time (s)")
            fig2.suptitle("RCAM: y_pred vs y_obs (V_NED)")
            fig2.tight_layout()
            fig_y_path = os.path.join(plot_dir, "rcam_y_pred_vs_obs.png")
            fig2.savefig(fig_y_path, dpi=150)
            plt.close(fig2)

    # 6) 最终 expert 信息（与原来一致）
    theta_means = []
    run_lengths = []
    masses = []
    for e in calibrator.bocpd.experts:
        ps = e.pf.particles
        w = ps.weights()
        theta = ps.theta
        th_mean = (w.view(-1, 1) * theta).sum(dim=0).detach().cpu().numpy()
        theta_means.append(th_mean)
        run_lengths.append(e.run_length)
        masses.append(e.log_mass)

    # 7) 简单指标：theta / y 的 RMSE 与“CRPS-like”度量（对确定性预测退化为 L2/L1 误差）
    # --- theta metrics ---
    theta_rmse_overall = float(
        np.sqrt(np.mean(np.sum((theta_est_arr - theta_true_arr) ** 2, axis=1)))
    )
    theta_rmse_per_dim = np.sqrt(np.mean((theta_est_arr - theta_true_arr) ** 2, axis=0))
    # 对确定性预测，CRPS ~ L1 误差；这里给出 L1 范数的平均值
    theta_crps_overall = float(
        np.mean(np.linalg.norm(theta_est_arr - theta_true_arr, ord=1, axis=1))
    )
    theta_crps_per_dim = np.mean(np.abs(theta_est_arr - theta_true_arr), axis=0)

    # --- y metrics ---
    y_pred_flat = np.concatenate(y_pred_list, axis=0)  # [T, 3]
    y_obs_flat = np.concatenate(y_obs_list, axis=0)    # [T, 3]
    valid_mask = ~np.isnan(y_pred_flat[:, 0])
    if np.any(valid_mask):
        y_pred_valid = y_pred_flat[valid_mask]
        y_obs_valid = y_obs_flat[valid_mask]
        y_rmse_overall = float(
            np.sqrt(np.mean(np.sum((y_pred_valid - y_obs_valid) ** 2, axis=1)))
        )
        y_rmse_per_dim = np.sqrt(np.mean((y_pred_valid - y_obs_valid) ** 2, axis=0))
        y_crps_overall = float(
            np.mean(np.linalg.norm(y_pred_valid - y_obs_valid, ord=1, axis=1))
        )
        y_crps_per_dim = np.mean(np.abs(y_pred_valid - y_obs_valid), axis=0)
    else:
        y_rmse_overall = float("nan")
        y_rmse_per_dim = np.full(3, np.nan)
        y_crps_overall = float("nan")
        y_crps_per_dim = np.full(3, np.nan)

    results = {
        "theta_means": np.stack(theta_means, axis=0) if theta_means else None,
        "run_lengths": np.array(run_lengths, dtype=int),
        "log_masses": np.array(masses, dtype=float),
        "theta_true_last": theta_true[-1, :],
        "stream_length": T,
        "out_stream_csv": out_stream_csv,
        "theta_est_trajectory": theta_est_arr,
        "theta_true_trajectory": theta_true_arr,
        "t_batch": t_plot,
        "y_pred_list": y_pred_list,
        "y_obs_list": y_obs_list,
        "fig_theta_path": fig_theta_path,
        "fig_y_path": fig_y_path,
        "theta_rmse_overall": theta_rmse_overall,
        "theta_rmse_per_dim": theta_rmse_per_dim,
        "theta_crps_overall": theta_crps_overall,
        "theta_crps_per_dim": theta_crps_per_dim,
        "y_rmse_overall": y_rmse_overall,
        "y_rmse_per_dim": y_rmse_per_dim,
        "y_crps_overall": y_crps_overall,
        "y_crps_per_dim": y_crps_per_dim,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="RCAM BOCPD-PF (R-BOCPD-PF-nodiscrepancy) on wind-jump dataset.")
    parser.add_argument(
        "--rcam_root",
        type=str,
        default="C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main",
        help="PSim-RCAM-main/PSim-RCAM-main 根目录（仅用于说明，实际数据路径由 --data_csv 控制）。",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data.csv",
        help="RCAM 日志 csv 路径（包含 u,v,w,phi,theta,psi,lat,lon,h,V_N,V_E,V_D）。",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="rcam_stream_windjump.csv",
        help="在 DynamicCalibration 根目录下保存 wind-jump stream 的 csv 文件名或路径。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=5120)
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for online updates and trajectory recording.")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save theta tracking and y_pred vs y_obs figures.")
    parser.add_argument("--use_cuda", action="store_true", default=False)

    args = parser.parse_args()

    # 归一化 out_csv 路径：若给的是相对路径，则放在当前工作目录
    out_csv = os.path.abspath(args.out_csv)
    plot_dir = os.path.abspath(args.plot_dir) if args.plot_dir else None

    print(f"[RCAM-BOCPD] Using data_csv = {args.data_csv}")
    print(f"[RCAM-BOCPD] Output stream csv = {out_csv}")
    if plot_dir:
        print(f"[RCAM-BOCPD] Plot directory = {plot_dir}")

    results = run_rcam_bocpd_pf(
        rcam_root=args.rcam_root,
        data_csv=args.data_csv,
        out_stream_csv=out_csv,
        seed=args.seed,
        use_cuda=args.use_cuda,
        num_particles=args.num_particles,
        batch_size=args.batch_size,
        plot_dir=plot_dir,
    )

    print("\n=== R-BOCPD-PF-nodiscrepancy on RCAM wind-jump stream ===")
    print(f"Stream length       : {results['stream_length']}")
    print(f"Output stream csv   : {results['out_stream_csv']}")
    print(f"Num experts (final) : {len(results['run_lengths'])}")
    if results["theta_means"] is not None:
        print("Final experts theta means (per expert, [bN,bE,bD]):")
        for i, (th, rl, lm) in enumerate(
            zip(results["theta_means"], results["run_lengths"], results["log_masses"])
        ):
            print(f"  Expert {i}: rl={rl:4d}, log_mass={lm:8.3f}, theta_mean={th}")
        print(f"Last true theta (from schedule): {results['theta_true_last']}")
    if results.get("fig_theta_path"):
        print(f"Theta tracking figure : {results['fig_theta_path']}")
    if results.get("fig_y_path"):
        print(f"Y pred vs obs figure  : {results['fig_y_path']}")
    # 打印 RMSE / CRPS 指标
    print("\nTheta metrics:")
    print(f"  RMSE overall (L2 over dims) : {results['theta_rmse_overall']:.4f}")
    print(f"  RMSE per dim [bN,bE,bD]     : {results['theta_rmse_per_dim']}")
    print(f"  CRPS-like overall (L1)      : {results['theta_crps_overall']:.4f}")
    print(f"  CRPS-like per dim           : {results['theta_crps_per_dim']}")

    print("\nY metrics (only where y_pred is defined):")
    print(f"  RMSE overall (L2 over dims) : {results['y_rmse_overall']:.4f}")
    print(f"  RMSE per dim [V_N,V_E,V_D]  : {results['y_rmse_per_dim']}")
    print(f"  CRPS-like overall (L1)      : {results['y_crps_overall']:.4f}")
    print(f"  CRPS-like per dim           : {results['y_crps_per_dim']}")


if __name__ == "__main__":
    main()

'''
python -m calib.run_rcam_bocpd_pf --rcam_root "C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main" --data_csv "C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data.csv" --out_csv "rcam_stream_windjump.csv" --num_particles 512 --seed 0

python -m calib.run_rcam_bocpd_pf --data_csv "C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data.csv" --out_csv "rcam_stream_windjump.csv" --batch_size 20 --plot_dir "figs/rcam_bocpd_windjump_v2"
'''

