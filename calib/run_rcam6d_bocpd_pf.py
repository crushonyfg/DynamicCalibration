"""
BOCPD-PF on RCAM (PSim-RCAM) with full 6D y = [lat, lon, h, V_N, V_E, V_D],
including a navigation process model in the PF state.

工作流概述：
1. 在 PSim-RCAM-main/PSim-RCAM-main 下运行 main_windjump.py 生成 wind-jump 版本的 test_data_windjump.csv。
2. 本脚本读取该 csv，计算 Va_NED(t)，并构造 6D 观测 y_t。
3. PF 粒子状态为 s_t = [lat, lon, h, bN, bE, bD]，我们在每个时间步显式用导航过程模型
   f_nav_disc(s_t, Va_NED_t) 推进所有 experts 的粒子状态，然后调用 R-BOCPD-PF-nodiscrepancy
   来更新权重。
4. 输出：
   - theta（bN,bE,bD）追踪图与 RMSE / CRPS-like 指标；
   - 6D y 预测 vs 观测的对比图与 RMSE / CRPS-like 指标。

目前只跑 R-BOCPD-PF-nodiscrepancy；接口预留了将来支持 discrepancy 的可能。
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
# 1) RCAM nav model: f_nav_cont, h_meas, euler_to_dcm
# =============================================================

R_EARTH = 6_378_137.0  # meters


def euler_to_dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body-to-NED rotation matrix. 复制自 run_ekf.py，避免修改原文件。"""
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


def f_nav_cont(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Continuous-time navigation dynamics:
      x = [lat, lon, h, bN, bE, bD]
      u = [Va_N, Va_E, Va_D] (air-relative NED)
    与 run_ekf.py 中的版本一致。
    """
    lat, lon, h, bN, bE, bD = x
    VN, VE, VD = u[0] + bN, u[1] + bE, u[2] + bD

    dlat = VN / R_EARTH
    dlon = VE / (R_EARTH * np.cos(lat)) if abs(np.cos(lat)) > 1e-9 else 0.0
    dh = -VD  # VD positive down
    return np.array([dlat, dlon, dh, 0.0, 0.0, 0.0], dtype=float)


def f_nav_disc(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    简单的 RK4 离散化：x_{k+1} = x_k + dt * f_nav_cont(x,u) (with RK4).
    """
    k1 = f_nav_cont(x, u)
    k2 = f_nav_cont(x + 0.5 * dt * k1, u)
    k3 = f_nav_cont(x + 0.5 * dt * k2, u)
    k4 = f_nav_cont(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def h_meas(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Measurement model:
      y = [lat, lon, h, V_N, V_E, V_D]
      x = [lat, lon, h, bN, bE, bD]
      u = [Va_N, Va_E, Va_D]
    """
    lat, lon, h, bN, bE, bD = x
    VN = u[0] + bN
    VE = u[1] + bE
    VD = u[2] + bD
    return np.array([lat, lon, h, VN, VE, VD], dtype=float)


# =============================================================
# 2) 从 RCAM wind-jump 数据生成 6D y stream + 标准化支持
# =============================================================


@dataclass
class YScaler:
    """y 的标准化器 (6D)"""
    mean: np.ndarray  # [6]
    std: np.ndarray   # [6]
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """标准化: (y - mean) / std"""
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y_norm: np.ndarray) -> np.ndarray:
        """反标准化: y_norm * std + mean"""
        return y_norm * self.std + self.mean
    
    def transform_torch(self, y: torch.Tensor) -> torch.Tensor:
        mean_t = torch.from_numpy(self.mean).to(y.device, y.dtype)
        std_t = torch.from_numpy(self.std).to(y.device, y.dtype)
        return (y - mean_t) / std_t
    
    def inverse_transform_torch(self, y_norm: torch.Tensor) -> torch.Tensor:
        mean_t = torch.from_numpy(self.mean).to(y_norm.device, y_norm.dtype)
        std_t = torch.from_numpy(self.std).to(y_norm.device, y_norm.dtype)
        return y_norm * std_t + mean_t


def build_rcam6d_stream(
    csv_path: str,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[YScaler]]:
    """
    从 main_windjump.py 生成的 csv 中读取：
      - Va_NED(t) (通过 uvw + euler 计算)
      - y_t = [lat, lon, h, V_N, V_E, V_D] (可能已标准化)
      - y_t_physical = [lat, lon, h, V_N, V_E, V_D] (物理空间，用于计算 theta_true)
      - 初始状态 x0 = [lat0, lon0, h0, 0, 0, 0]
      - t_seq (PSim_Time)
      - scaler (如果 normalize=True)
    """
    df = pd.read_csv(csv_path)
    if "PSim_Time" in df.columns:
        t_seq = df["PSim_Time"].to_numpy()
    else:
        raise ValueError("CSV must contain 'PSim_Time' column.")

    required_cols = ["lat", "lon", "h", "V_N", "V_E", "V_D", "u", "v", "w", "phi", "theta", "psi"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # 计算 Va_NED
    Va_seq = []
    for _, row in df.iterrows():
        V_body = np.array([row["u"], row["v"], row["w"]], dtype=float)
        R_b2n = euler_to_dcm(float(row["phi"]), float(row["theta"]), float(row["psi"]))
        Va_seq.append(R_b2n @ V_body)
    Va_seq = np.stack(Va_seq, axis=0)  # [T,3]

    # 6D 观测 (物理空间)
    y_seq_physical = df[["lat", "lon", "h", "V_N", "V_E", "V_D"]].to_numpy(dtype=float)  # [T,6]

    # 初始状态：使用第一行的 lat,lon,h + 零风偏
    row0 = df.iloc[0]
    x0 = np.array([row0["lat"], row0["lon"], row0["h"], 0.0, 0.0, 0.0], dtype=float)

    scaler: Optional[YScaler] = None
    if normalize:
        y_mean = y_seq_physical.mean(axis=0)
        y_std = y_seq_physical.std(axis=0)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)  # 避免除零
        scaler = YScaler(mean=y_mean, std=y_std)
        y_seq = scaler.transform(y_seq_physical)
        print(f"[build_rcam6d_stream] Normalization enabled:")
        print(f"  y_mean = {y_mean}")
        print(f"  y_std  = {y_std}")
    else:
        y_seq = y_seq_physical

    return Va_seq, y_seq, y_seq_physical, x0, t_seq, scaler


# =============================================================
# 3) Emulator: y_hat = h_meas(x_state, Va_NED)  with PF state in theta
# =============================================================


def make_rcam6d_emulator(scaler: Optional[YScaler] = None) -> DeterministicSimulator:
    """
    构造一个 DeterministicSimulator，使得：
      粒子中的 theta 实际上是当前状态向量 x_state = [lat,lon,h,bN,bE,bD]
      x (传入的) 则是当前时间步的 Va_NED(t)。

    对于每个粒子 n：
      输入:  x: [b,3]  -- Va_NED
             theta[n]: [6] -- state (物理空间)
      输出:  y_hat: [b,6] = h_meas(theta[n], x)，如果 scaler 存在则输出标准化后的 y

    注意：PF state 始终在物理空间，只有输出 y 在标准化空间（如果启用）。
    """

    def f_eta(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # x: [b,3] or [3] (Va_NED), theta: [1,6] (state in physical space)
        if x.dim() == 1:
            x_ = x[None, :]
        else:
            x_ = x
        state = theta[0, :]  # [6]
        lat, lon, h, bN, bE, bD = [float(v) for v in state]
        u_np = x_.detach().cpu().numpy()
        # 对每个 batch 样本应用 h_meas
        ys = []
        for b in range(u_np.shape[0]):
            y_b = h_meas(np.array([lat, lon, h, bN, bE, bD], dtype=float), u_np[b])
            ys.append(y_b)
        y = torch.from_numpy(np.stack(ys, axis=0)).to(x_.dtype).to(x_.device)  # [b,6]
        
        # 如果有 scaler，将输出标准化
        if scaler is not None:
            y = scaler.transform_torch(y)
        return y

    return DeterministicSimulator(f_eta, enable_autograd=False)


def make_prior_sampler_state(
    x0: np.ndarray,
    pos_noise: float = 1e-5,
    bias_scale: float = 5.0,
) -> Callable[[int], torch.Tensor]:
    """
    先验：围绕 x0 的小扰动 + 偏差较大的风偏分量。
      x0 = [lat0, lon0, h0, 0, 0, 0]
    """

    def prior_sampler(num_particles: int) -> torch.Tensor:
        lat0, lon0, h0, _, _, _ = x0
        pos_pert = np.stack(
            [
                np.random.normal(lat0, pos_noise, size=num_particles),
                np.random.normal(lon0, pos_noise, size=num_particles),
                np.random.normal(h0, 10.0, size=num_particles),
            ],
            axis=1,
        )  # [N,3]
        bias_pert = np.random.normal(0.0, bias_scale, size=(num_particles, 3))  # [N,3]
        theta0 = np.concatenate([pos_pert, bias_pert], axis=1)
        return torch.from_numpy(theta0).to(torch.float64)

    return prior_sampler


# =============================================================
# 4) 运行 R-BOCPD-PF-nodiscrepancy，包含 nav 过程的显式 PF state 更新
# =============================================================


# 三种方法的配置
METHODS = {
    "R-BOCPD-PF-usediscrepancy": dict(use_discrepancy=True, bocpd_use_discrepancy=True),
    "R-BOCPD-PF-nodiscrepancy": dict(use_discrepancy=False, bocpd_use_discrepancy=False),
    "R-BOCPD-PF-halfdiscrepancy": dict(use_discrepancy=False, bocpd_use_discrepancy=True),
}


def run_rcam6d_bocpd_pf(
    data_csv: str,
    seed: int = 0,
    use_cuda: bool = False,
    num_particles: int = 256,
    dt: float = 0.1,
    plot_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    normalize: bool = False,
    method: str = "R-BOCPD-PF-nodiscrepancy",
    bias_scale: float = 3.0,
    bias_rw_std: float = 0.1,
) -> Dict[str, Any]:
    """
    data_csv: main_windjump.py 生成的 csv 路径（包含 lat,lon,h,V_N,V_E,V_D,u,v,w,phi,theta,psi,PSim_Time）。
    normalize: 是否对 y 进行标准化（推荐开启，因为各维度尺度差异大）。
    method: 方法选择，可选：
        - "R-BOCPD-PF-usediscrepancy": PF + BOCPD 都使用 discrepancy
        - "R-BOCPD-PF-nodiscrepancy": 都不使用 discrepancy
        - "R-BOCPD-PF-halfdiscrepancy": PF 不用，BOCPD 用 discrepancy
    bias_scale: prior sampler 的 bias 分布标准差 (m/s)，应与真实 theta 变化范围匹配
    bias_rw_std: bias 随机游走每步的标准差 (m/s)，控制粒子探索能力
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")
    
    method_cfg = METHODS[method]
    print(f"[RCAM6D-BOCPD] Method = {method}")
    print(f"[RCAM6D-BOCPD] Config = {method_cfg}")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    Va_seq, y_seq, y_seq_physical, x0, t_seq, scaler = build_rcam6d_stream(data_csv, normalize=normalize)
    T = Va_seq.shape[0]
    if max_steps is not None and max_steps < T:
        T = max_steps
        print(f"[RCAM6D-BOCPD] Limiting to {T} steps (max_steps={max_steps})")

    # 1) 构造模型与 BOCPD 配置
    model_cfg = ModelConfig()
    model_cfg.rho = 1.0
    # 标准化后 sigma_eps 应该更小（因为数据已是 ~N(0,1) 尺度）
    model_cfg.sigma_eps = 0.5 if not normalize else 0.2
    model_cfg.emulator_type = "deterministic"
    model_cfg.device = device
    model_cfg.dtype = dtype
    model_cfg.use_discrepancy = method_cfg["use_discrepancy"]
    setattr(model_cfg, "bocpd_use_discrepancy", method_cfg["bocpd_use_discrepancy"])

    pf_cfg = PFConfig()
    pf_cfg.num_particles = num_particles
    pf_cfg.resample_ess_ratio = 0.5
    pf_cfg.move_strategy = "none"  # 过程模型由我们自己显式推进，不用内部 move

    bocpd_cfg = BOCPDConfig()
    bocpd_cfg.hazard_lambda = 600.0
    bocpd_cfg.hazard_type = "geometric"
    bocpd_cfg.bocpd_mode = "restart"
    bocpd_cfg.max_experts = 5
    bocpd_cfg.use_restart = True
    bocpd_cfg.restart_threshold = 0.85
    bocpd_cfg.restart_margin = 1.0
    bocpd_cfg.restart_cooldown = 20
    bocpd_cfg.restart_criteria = "rank_change"

    cfg = CalibrationConfig(model=model_cfg, pf=pf_cfg, bocpd=bocpd_cfg)

    emulator = make_rcam6d_emulator(scaler=scaler)
    prior_sampler = make_prior_sampler_state(x0, bias_scale=bias_scale)

    # 记录 restart 事件的时间点
    restart_times: List[float] = []
    current_step = [0]  # 用 list 包装以便在闭包中修改
    
    def on_restart_callback(t_now, r_new, s_star, anchor_rl, p_anchor, best_other):
        # t_now 是 BOCPD 内部的计数，我们需要转换为实际时间
        restart_times.append(t_seq[current_step[0]] if current_step[0] < len(t_seq) else t_seq[-1])
        print(f"  [Restart] at step {current_step[0]}, t={restart_times[-1]:.2f}s")

    calibrator = OnlineBayesCalibrator(
        calib_cfg=cfg,
        emulator=emulator,
        prior_sampler=prior_sampler,
        init_delta_state=None,
        delta_fitter=None,
        on_restart=on_restart_callback,
        notify_on_restart=True,
    )

    X_all = torch.from_numpy(Va_seq).to(device=device, dtype=dtype)  # [T,3]
    Y_all = torch.from_numpy(y_seq).to(device=device, dtype=dtype)   # [T,6] (可能已标准化)

    # 2) 轨迹记录
    theta_est_list: List[np.ndarray] = []   # [bN,bE,bD] per time (物理空间)
    theta_true_list: List[np.ndarray] = []  # 物理风偏等价项 (V_N - Va_N, etc.)
    y_pred_list: List[np.ndarray] = []      # 标准化空间 (如果启用)
    y_obs_list: List[np.ndarray] = []       # 标准化空间 (如果启用)

    # "真实 theta" 近似：始终用物理空间的 y 计算 (V_N - Va_N, V_E - Va_E, V_D - Va_D)
    theta_true_seq = y_seq_physical[:, 3:6] - Va_seq  # [T,3]

    from tqdm import tqdm
    for k in tqdm(range(T)):
        current_step[0] = k  # 更新当前步数供 restart 回调使用
        xk = X_all[k : k + 1]  # [1,3]
        yk = Y_all[k : k + 1]  # [1,6]

        # (1) 在调用 BOCPD 更新前，显式推进所有 experts 的粒子状态：s_{t+1} = f_nav_disc(s_t, Va_k, dt)
        # 同时给 bias 分量添加 random walk process noise，让粒子能够探索 theta 空间
        for e in calibrator.bocpd.experts:
            ps = e.pf.particles
            theta_state = ps.theta.detach().cpu().numpy()  # [N,6]
            N_particles = theta_state.shape[0]
            theta_next = []
            for n in range(N_particles):
                s = theta_state[n]
                s_next = f_nav_disc(s, Va_seq[k], dt)
                theta_next.append(s_next)
            theta_next = np.stack(theta_next, axis=0)  # [N, 6]
            
            # 添加 bias process noise (random walk on bN, bE, bD)
            # bias_rw_std 控制每步的随机游走幅度，应该与 dt 和预期 theta 变化率匹配
            bias_rw_noise = np.random.normal(0.0, bias_rw_std, size=(N_particles, 3))
            theta_next[:, 3:6] += bias_rw_noise
            
            ps.theta = torch.from_numpy(theta_next).to(device=device, dtype=dtype)

        # (2) 预测 y_hat
        if len(calibrator.bocpd.experts) > 0:
            pred = calibrator.predict_batch(xk)
            mu = pred["mu"]  # [1,6]
            if isinstance(mu, torch.Tensor):
                mu_np = mu.detach().cpu().numpy()
            else:
                mu_np = np.asarray(mu)
            if mu_np.ndim == 1:
                mu_np = mu_np[None, :]
            y_pred_list.append(mu_np)
        else:
            y_pred_list.append(np.full((1, 6), np.nan))

        y_obs_list.append(yk.detach().cpu().numpy())

        # (3) 调用 BOCPD 更新 (使用 step_batch，因为 BOCPD 类只有 update_batch 方法)
        calibrator.step_batch(xk, yk, verbose=False)

        # (4) 汇总粒子混合 theta（取 bias 部分 [3:6]）
        mean_theta, _ = calibrator._aggregate_particles(quantile=None)
        if mean_theta is not None:
            mt = mean_theta.detach().cpu().numpy()
            theta_est_list.append(mt[3:6].copy())
        else:
            theta_est_list.append(np.full(3, np.nan))

        theta_true_list.append(theta_true_seq[k, :].copy())

    theta_est_arr = np.stack(theta_est_list, axis=0)   # [T,3]
    theta_true_arr = np.stack(theta_true_list, axis=0) # [T,3]
    y_pred_flat = np.concatenate(y_pred_list, axis=0)  # [T,6] (标准化空间 if normalize)
    y_obs_flat = np.concatenate(y_obs_list, axis=0)    # [T,6] (标准化空间 if normalize)
    t_flat = t_seq[:T]

    # 如果启用了标准化，将 y_pred 和 y_obs 反标准化回物理空间用于画图和指标计算
    if scaler is not None:
        y_pred_flat_phys = scaler.inverse_transform(y_pred_flat)
        y_obs_flat_phys = scaler.inverse_transform(y_obs_flat)
    else:
        y_pred_flat_phys = y_pred_flat
        y_obs_flat_phys = y_obs_flat

    # 3) 指标：theta / y 的 RMSE 与 CRPS-like (在物理空间计算)
    # 计算每个维度的标准差用于归一化（NRMSE = RMSE / std）
    theta_std_per_dim = np.std(theta_true_arr, axis=0)
    theta_std_per_dim = np.where(theta_std_per_dim < 1e-8, 1.0, theta_std_per_dim)
    
    # 绝对误差
    theta_rmse_per_dim = np.sqrt(np.mean((theta_est_arr - theta_true_arr) ** 2, axis=0))
    theta_crps_per_dim = np.mean(np.abs(theta_est_arr - theta_true_arr), axis=0)
    
    # 相对误差 (NRMSE = RMSE / std)
    theta_nrmse_per_dim = theta_rmse_per_dim / theta_std_per_dim
    theta_ncrps_per_dim = theta_crps_per_dim / theta_std_per_dim
    
    # 总体误差：使用相对误差的平均值（各维度等权重）
    theta_nrmse_overall = float(np.mean(theta_nrmse_per_dim))
    theta_ncrps_overall = float(np.mean(theta_ncrps_per_dim))
    # 也保留绝对误差（L2 over dims）用于参考
    theta_rmse_overall = float(np.sqrt(np.mean(np.sum((theta_est_arr - theta_true_arr) ** 2, axis=1))))
    theta_crps_overall = float(np.mean(np.linalg.norm(theta_est_arr - theta_true_arr, ord=1, axis=1)))

    valid_mask = ~np.isnan(y_pred_flat_phys[:, 0])
    if np.any(valid_mask):
        y_pred_valid = y_pred_flat_phys[valid_mask]
        y_obs_valid = y_obs_flat_phys[valid_mask]
        
        # 每个维度的标准差
        y_std_per_dim = np.std(y_obs_valid, axis=0)
        y_std_per_dim = np.where(y_std_per_dim < 1e-8, 1.0, y_std_per_dim)
        
        # 绝对误差
        y_rmse_per_dim = np.sqrt(np.mean((y_pred_valid - y_obs_valid) ** 2, axis=0))
        y_crps_per_dim = np.mean(np.abs(y_pred_valid - y_obs_valid), axis=0)
        
        # 相对误差
        y_nrmse_per_dim = y_rmse_per_dim / y_std_per_dim
        y_ncrps_per_dim = y_crps_per_dim / y_std_per_dim
        
        # 总体相对误差
        y_nrmse_overall = float(np.mean(y_nrmse_per_dim))
        y_ncrps_overall = float(np.mean(y_ncrps_per_dim))
        # 绝对误差
        y_rmse_overall = float(np.sqrt(np.mean(np.sum((y_pred_valid - y_obs_valid) ** 2, axis=1))))
        y_crps_overall = float(np.mean(np.linalg.norm(y_pred_valid - y_obs_valid, ord=1, axis=1)))
    else:
        y_rmse_overall = float("nan")
        y_rmse_per_dim = np.full(6, np.nan)
        y_crps_overall = float("nan")
        y_crps_per_dim = np.full(6, np.nan)
        y_nrmse_overall = float("nan")
        y_nrmse_per_dim = np.full(6, np.nan)
        y_ncrps_overall = float("nan")
        y_ncrps_per_dim = np.full(6, np.nan)
        y_std_per_dim = np.full(6, np.nan)

    # 4) 作图 (使用物理空间的值)
    fig_theta_path: Optional[str] = None
    fig_y_path: Optional[str] = None
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

        # theta 追踪
        fig1, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        labels = ["bN (N)", "bE (E)", "bD (D)"]
        for j in range(3):
            axes[j].plot(t_flat, theta_true_arr[:, j], "k-", alpha=0.8, label="theta_true")
            axes[j].plot(t_flat, theta_est_arr[:, j], "b-", alpha=0.7, label="theta_est")
            # 标记 restart 点
            for rt in restart_times:
                idx = np.searchsorted(t_flat, rt)
                if idx < len(t_flat):
                    axes[j].axvline(x=rt, color='r', linestyle='--', alpha=0.5, lw=0.8)
                    axes[j].plot(rt, theta_est_arr[min(idx, len(theta_est_arr)-1), j], 
                               'rx', markersize=8, markeredgewidth=2)
            axes[j].set_ylabel(labels[j])
            axes[j].legend(loc="upper right")
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        # 添加 restart 图例
        if restart_times:
            axes[0].plot([], [], 'rx', markersize=8, markeredgewidth=2, label=f"restart ({len(restart_times)})")
            axes[0].legend(loc="upper right")
        fig1.suptitle(f"RCAM 6D: wind bias tracking ({method})")
        fig1.tight_layout()
        fig_theta_path = os.path.join(plot_dir, "rcam6d_theta_tracking.png")
        fig1.savefig(fig_theta_path, dpi=150)
        plt.close(fig1)

        # y 预测 vs 观测（6 维，物理空间）
        if np.any(valid_mask):
            fig2, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
            y_labels = ["lat", "lon", "h", "V_N", "V_E", "V_D"]
            for j in range(6):
                axes[j].plot(t_flat, y_obs_flat_phys[:, j], "k-", alpha=0.6, label="y_obs")
                axes[j].plot(t_flat, y_pred_flat_phys[:, j], "b-", alpha=0.7, label="y_pred")
                # 标记 restart 点
                for rt in restart_times:
                    idx = np.searchsorted(t_flat, rt)
                    if idx < len(t_flat):
                        axes[j].axvline(x=rt, color='r', linestyle='--', alpha=0.5, lw=0.8)
                        axes[j].plot(rt, y_pred_flat_phys[min(idx, len(y_pred_flat_phys)-1), j], 
                                   'rx', markersize=6, markeredgewidth=1.5)
                axes[j].set_ylabel(y_labels[j])
                axes[j].legend(loc="upper right")
                axes[j].grid(True, alpha=0.3)
            axes[-1].set_xlabel("Time (s)")
            # 添加 restart 图例
            if restart_times:
                axes[0].plot([], [], 'rx', markersize=6, markeredgewidth=1.5, label=f"restart ({len(restart_times)})")
                axes[0].legend(loc="upper right")
            fig2.suptitle(f"RCAM 6D: y_pred vs y_obs ({method})")
            fig2.tight_layout()
            fig_y_path = os.path.join(plot_dir, "rcam6d_y_pred_vs_obs.png")
            fig2.savefig(fig_y_path, dpi=150)
            plt.close(fig2)

    results: Dict[str, Any] = {
        "theta_est": theta_est_arr,
        "theta_true": theta_true_arr,
        # 绝对误差
        "theta_rmse_overall": theta_rmse_overall,
        "theta_rmse_per_dim": theta_rmse_per_dim,
        "theta_crps_overall": theta_crps_overall,
        "theta_crps_per_dim": theta_crps_per_dim,
        # 相对误差 (NRMSE = RMSE / std)
        "theta_nrmse_overall": theta_nrmse_overall,
        "theta_nrmse_per_dim": theta_nrmse_per_dim,
        "theta_ncrps_overall": theta_ncrps_overall,
        "theta_ncrps_per_dim": theta_ncrps_per_dim,
        "theta_std_per_dim": theta_std_per_dim,
        # y 绝对误差
        "y_rmse_overall": y_rmse_overall,
        "y_rmse_per_dim": y_rmse_per_dim,
        "y_crps_overall": y_crps_overall,
        "y_crps_per_dim": y_crps_per_dim,
        # y 相对误差
        "y_nrmse_overall": y_nrmse_overall,
        "y_nrmse_per_dim": y_nrmse_per_dim,
        "y_ncrps_overall": y_ncrps_overall,
        "y_ncrps_per_dim": y_ncrps_per_dim,
        "y_std_per_dim": y_std_per_dim,
        # 其他
        "t": t_flat,
        "y_pred": y_pred_flat_phys,
        "y_obs": y_obs_flat_phys,
        "fig_theta_path": fig_theta_path,
        "fig_y_path": fig_y_path,
        "scaler": scaler,
        "restart_times": restart_times,
        "num_restarts": len(restart_times),
        "method": method,
    }
    return results


def run_all_methods(
    data_csv: str,
    seed: int = 0,
    use_cuda: bool = False,
    num_particles: int = 256,
    dt: float = 0.1,
    plot_dir: Optional[str] = None,
    max_steps: Optional[int] = None,
    normalize: bool = False,
    bias_scale: float = 3.0,
    bias_rw_std: float = 0.1,
) -> Dict[str, Dict[str, Any]]:
    """运行所有三种方法并比较结果"""
    all_results = {}
    
    for method_name in METHODS.keys():
        print(f"\n{'='*60}")
        print(f"Running {method_name}")
        print(f"{'='*60}")
        
        method_plot_dir = None
        if plot_dir:
            method_plot_dir = os.path.join(plot_dir, method_name.replace("-", "_").lower())
        
        results = run_rcam6d_bocpd_pf(
            data_csv=data_csv,
            seed=seed,
            use_cuda=use_cuda,
            num_particles=num_particles,
            dt=dt,
            plot_dir=method_plot_dir,
            max_steps=max_steps,
            normalize=normalize,
            method=method_name,
            bias_scale=bias_scale,
            bias_rw_std=bias_rw_std,
        )
        all_results[method_name] = results
    
    # 打印比较表格（使用相对误差 NRMSE/NCRPS）
    print("\n" + "="*100)
    print("COMPARISON TABLE (Relative Errors: NRMSE = RMSE/std, NCRPS = MAE/std)")
    print("="*100)
    print(f"{'Method':<35} | {'θ NRMSE':>10} | {'θ NCRPS':>10} | {'y NRMSE':>10} | {'y NCRPS':>10} | {'#Restarts':>10}")
    print("-"*100)
    for method_name, res in all_results.items():
        print(f"{method_name:<35} | {res['theta_nrmse_overall']:>10.4f} | {res['theta_ncrps_overall']:>10.4f} | {res['y_nrmse_overall']:>10.4f} | {res['y_ncrps_overall']:>10.4f} | {res['num_restarts']:>10}")
    print("="*100)
    
    # 也打印绝对误差供参考
    print("\n(Absolute errors for reference)")
    print(f"{'Method':<35} | {'θ RMSE':>12} | {'θ CRPS':>12} | {'y RMSE':>12} | {'y CRPS':>12}")
    print("-"*90)
    for method_name, res in all_results.items():
        print(f"{method_name:<35} | {res['theta_rmse_overall']:>12.4f} | {res['theta_crps_overall']:>12.4f} | {res['y_rmse_overall']:>12.4f} | {res['y_crps_overall']:>12.4f}")
    
    # 绘制比较图
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        
        # 比较 theta tracking
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ["bN (N)", "bE (E)", "bD (D)"]
        colors = {"R-BOCPD-PF-usediscrepancy": "r", 
                  "R-BOCPD-PF-nodiscrepancy": "b", 
                  "R-BOCPD-PF-halfdiscrepancy": "g"}
        
        # 取第一个结果的 theta_true 和 t 作为参考
        first_res = list(all_results.values())[0]
        t = first_res["t"]
        theta_true = first_res["theta_true"]
        
        for j in range(3):
            axes[j].plot(t, theta_true[:, j], "k-", lw=2, alpha=0.8, label="theta_true")
            for method_name, res in all_results.items():
                axes[j].plot(t, res["theta_est"][:, j], 
                           color=colors.get(method_name, "gray"), 
                           alpha=0.7, label=method_name)
            axes[j].set_ylabel(labels[j])
            if j == 0:
                axes[j].legend(loc="upper right", fontsize=8)
            axes[j].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("RCAM 6D: Method Comparison - Wind Bias Tracking")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "comparison_theta_tracking.png"), dpi=150)
        plt.close(fig)
        
        # 比较柱状图（使用相对误差）
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        method_names = list(all_results.keys())
        x = np.arange(len(method_names))
        
        # 打印调试信息确认顺序
        print("\n[DEBUG] Bar chart data order:")
        for i, m in enumerate(method_names):
            print(f"  x={i}: {m} -> theta_nrmse={all_results[m]['theta_nrmse_overall']:.4f}, y_nrmse={all_results[m]['y_nrmse_overall']:.4f}")
        
        # Theta metrics (相对误差)
        theta_nrmse = [all_results[m]["theta_nrmse_overall"] for m in method_names]
        theta_ncrps = [all_results[m]["theta_ncrps_overall"] for m in method_names]
        width = 0.35
        bars1 = axes[0].bar(x - width/2, theta_nrmse, width, label='NRMSE', color='steelblue')
        bars2 = axes[0].bar(x + width/2, theta_ncrps, width, label='NCRPS', color='coral')
        # 在柱子上方添加数值
        for bar, val in zip(bars1, theta_nrmse):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, theta_ncrps):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0].set_ylabel('Relative Error (RMSE/std)')
        axes[0].set_title('Theta Estimation Error (Normalized)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace("R-BOCPD-PF-", "") for m in method_names], rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Y metrics (相对误差)
        y_nrmse = [all_results[m]["y_nrmse_overall"] for m in method_names]
        y_ncrps = [all_results[m]["y_ncrps_overall"] for m in method_names]
        bars3 = axes[1].bar(x - width/2, y_nrmse, width, label='NRMSE', color='steelblue')
        bars4 = axes[1].bar(x + width/2, y_ncrps, width, label='NCRPS', color='coral')
        # 在柱子上方添加数值
        for bar, val in zip(bars3, y_nrmse):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars4, y_ncrps):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1].set_ylabel('Relative Error (RMSE/std)')
        axes[1].set_title('Y Prediction Error (Normalized)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m.replace("R-BOCPD-PF-", "") for m in method_names], rotation=15, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "comparison_metrics.png"), dpi=150)
        plt.close(fig)
        
        print(f"\nComparison plots saved to {plot_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="RCAM 6D y BOCPD-PF with nav process in PF.")
    parser.add_argument(
        "--data_csv",
        type=str,
        default="C:/Users/yxu59/files/winter2026/park/simulation/PSim-RCAM-main/PSim-RCAM-main/test_data_windjump.csv",
        help="main_windjump.py 生成的 RCAM wind-jump 日志 csv 路径。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_particles", type=int, default=256)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--dt", type=float, default=0.1, help="Navigation integration time step (s).")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of time steps to process (for quick testing).")
    parser.add_argument("--normalize", action="store_true", default=False, help="Enable y normalization (recommended for better PF performance).")
    parser.add_argument(
        "--method",
        type=str,
        default="R-BOCPD-PF-nodiscrepancy",
        choices=list(METHODS.keys()),
        help="Method to run. Use --run_all to run all methods.",
    )
    parser.add_argument("--run_all", action="store_true", default=False, help="Run all three methods and compare.")
    parser.add_argument("--bias_scale", type=float, default=3.0, help="Prior bias std (m/s). Should match expected theta range.")
    parser.add_argument("--bias_rw_std", type=float, default=0.1, help="Bias random walk std per step (m/s). Controls particle exploration.")

    args = parser.parse_args()
    plot_dir = os.path.abspath(args.plot_dir) if args.plot_dir else None

    print(f"[RCAM6D-BOCPD] Using data_csv = {args.data_csv}")
    print(f"[RCAM6D-BOCPD] Normalize = {args.normalize}")
    if plot_dir:
        print(f"[RCAM6D-BOCPD] Plot directory = {plot_dir}")

    print(f"[RCAM6D-BOCPD] bias_scale = {args.bias_scale}, bias_rw_std = {args.bias_rw_std}")

    if args.run_all:
        # 运行所有方法并比较
        all_results = run_all_methods(
            data_csv=args.data_csv,
            seed=args.seed,
            use_cuda=args.use_cuda,
            num_particles=args.num_particles,
            dt=args.dt,
            plot_dir=plot_dir,
            max_steps=args.max_steps,
            normalize=args.normalize,
            bias_scale=args.bias_scale,
            bias_rw_std=args.bias_rw_std,
        )
    else:
        # 运行单个方法
        results = run_rcam6d_bocpd_pf(
            data_csv=args.data_csv,
            seed=args.seed,
            use_cuda=args.use_cuda,
            num_particles=args.num_particles,
            dt=args.dt,
            plot_dir=plot_dir,
            max_steps=args.max_steps,
            normalize=args.normalize,
            method=args.method,
            bias_scale=args.bias_scale,
            bias_rw_std=args.bias_rw_std,
        )

        print(f"\n=== RCAM 6D {args.method} results ===")
        print(f"Number of restarts: {results['num_restarts']}")
        
        print(f"\n--- Theta (wind bias) ---")
        print(f"  Relative errors (NRMSE = RMSE/std, NCRPS = MAE/std):")
        print(f"    NRMSE overall (avg over dims): {results['theta_nrmse_overall']:.4f}")
        print(f"    NRMSE per dim [bN,bE,bD]     : {results['theta_nrmse_per_dim']}")
        print(f"    NCRPS overall                : {results['theta_ncrps_overall']:.4f}")
        print(f"    NCRPS per dim                : {results['theta_ncrps_per_dim']}")
        print(f"  Absolute errors:")
        print(f"    RMSE overall (L2)            : {results['theta_rmse_overall']:.4f}")
        print(f"    RMSE per dim                 : {results['theta_rmse_per_dim']}")
        print(f"  Std per dim (for reference)    : {results['theta_std_per_dim']}")

        print(f"\n--- Y (6D observation) ---")
        print(f"  Relative errors:")
        print(f"    NRMSE overall (avg over dims): {results['y_nrmse_overall']:.4f}")
        print(f"    NRMSE per dim [lat,lon,h,VN,VE,VD]: {results['y_nrmse_per_dim']}")
        print(f"    NCRPS overall                : {results['y_ncrps_overall']:.4f}")
        print(f"    NCRPS per dim                : {results['y_ncrps_per_dim']}")
        print(f"  Absolute errors:")
        print(f"    RMSE overall (L2)            : {results['y_rmse_overall']:.4f}")
        print(f"    RMSE per dim                 : {results['y_rmse_per_dim']}")
        print(f"  Std per dim (for reference)    : {results['y_std_per_dim']}")

        if results.get("fig_theta_path"):
            print(f"\nTheta tracking figure : {results['fig_theta_path']}")
        if results.get("fig_y_path"):
            print(f"Y pred vs obs figure  : {results['fig_y_path']}")


if __name__ == "__main__":
    main()

'''
# 运行单个方法（快速测试，200步）
python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d" --num_particles 128 --max_steps 200 --normalize --method R-BOCPD-PF-nodiscrepancy

# 运行所有三种方法并比较（快速测试，200步）
python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d_compare" --num_particles 128 --max_steps 200 --normalize --run_all

# 完整运行所有方法
python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d_compare" --num_particles 256 --normalize --run_all

# 可选方法:
#   R-BOCPD-PF-usediscrepancy   - PF + BOCPD 都使用 discrepancy
#   R-BOCPD-PF-nodiscrepancy    - 都不使用 discrepancy
#   R-BOCPD-PF-halfdiscrepancy  - PF 不用，BOCPD 用 discrepancy

python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d_compare" --num_particles 1024 --max_steps 1000 --normalize --run_all


# 运行所有三种方法并比较（快速测试）
conda run -n jumpGP python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d_compare" --num_particles 1024 --max_steps 100 --normalize --run_all

# 运行单个方法
conda run -n jumpGP python -m calib.run_rcam6d_bocpd_pf --plot_dir "figs/rcam6d" --num_particles 128 --max_steps 200 --normalize --method R-BOCPD-PF-nodiscrepancy
'''
