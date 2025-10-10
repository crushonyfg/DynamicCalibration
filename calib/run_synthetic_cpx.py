# =============================================================
# file: calib/run_synthetic.py
# =============================================================
import torch
import math
import numpy as np

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator
from .data import SyntheticDataStream, SyntheticGeneratorConfig, ChangepointConfig

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def calculate_crps(y_true: torch.Tensor, mu_pred: torch.Tensor, var_pred: torch.Tensor) -> torch.Tensor:
    """计算CRPS（正态近似下的闭式表达）"""
    sigma = torch.sqrt(var_pred.clamp_min(1e-12))
    z = (y_true - mu_pred) / sigma
    c1 = z * torch.erf(z / torch.sqrt(torch.tensor(2.0)))
    c2 = torch.sqrt(torch.tensor(2.0 / np.pi)) * torch.exp(-0.5 * z**2)
    c3 = torch.sqrt(torch.tensor(1.0 / np.pi))
    crps = sigma * (c1 + c2 - c3)
    return crps


def eta_func_complex(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    更复杂的 η(x, θ)：非线性 + 二次 + 交互 + 正弦
    约定：x 维度=3，θ 维度=4
    y = θ0*x0 + θ1*sin(θ2*x1) + θ3*x0*x1 + 0.5*x2^2 + 0.3*tanh(x0-0.5)
    """
    if theta.dim() == 1:
        theta = theta[0:][None, :]
    if x.dim() == 1:
        x = x[None, :]
    th0, th1, th2, th3 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    x0, x1, x2 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    outs = []
    for n in range(theta.shape[0]):
        y = (
            th0[n] * x0
            + th1[n] * torch.sin(th2[n] * x1)
            + th3[n] * (x0 * x1)
            + 0.5 * (x2 ** 2)
            + 0.3 * torch.tanh(x0 - 0.5)
        )  # [b,1]
        outs.append(y)
    return torch.cat(outs, dim=1)  # [b,N]


def run_comparison_experiment(prefix: str = "exp"):
    """
    同时运行 Standard / Restart 两种 BOCPD，对比 RMSE/CRPS、重启检测、expert的run_length。
    prefix: 保存图片与结果文件的前缀，避免覆盖以往实验。
    """
    # 实验规模与复杂度设置
    target_observations = 2000          # 总时长
    batch_size_range = (20, 50)         # 批量范围
    x_dim = 3
    theta_dim = 4

    # 配置
    calib_cfg = CalibrationConfig()
    device, dtype = calib_cfg.model.device, calib_cfg.model.dtype

    # 先用复杂η构建 emulator
    emulator = DeterministicSimulator(func=eta_func_complex, enable_autograd=True)

    # 先占位的x分布（构建stream时使用），稍后改为随时间漂移的分布
    def x_dist_uniform(b: int) -> torch.Tensor:
        return torch.rand(b, x_dim, dtype=dtype, device=device)

    # prior：θ∈[-1,1]^theta_dim
    def prior_sampler(N: int) -> torch.Tensor:
        lo = torch.full((theta_dim,), -1.0, dtype=dtype, device=device)
        hi = torch.full((theta_dim,), +1.0, dtype=dtype, device=device)
        u = torch.rand(N, theta_dim, dtype=dtype, device=device)
        return lo + (hi - lo) * u

    # 构建changepoints：混合“θ跳变/δ偏移/重置δGP/组合”
    import random
    random.seed(0)

    def rand_theta():
        return torch.empty(theta_dim, dtype=dtype, device=device).uniform_(-1.0, 1.0)

    num_cps = 10
    times = torch.linspace(80, target_observations - 100, num_cps, dtype=torch.long).tolist()
    changepoints = []
    for i, t_cp in enumerate(times):
        mode = random.choice(["theta", "delta_shift", "delta_new", "combo"])
        if mode == "theta":
            changepoints.append(ChangepointConfig(time=int(t_cp), theta_new=rand_theta(), new_delta_gp=False))
        elif mode == "delta_shift":
            changepoints.append(ChangepointConfig(time=int(t_cp),
                                                 delta_shift=float(random.uniform(-0.5, 0.5)),
                                                 new_delta_gp=False))
        elif mode == "delta_new":
            changepoints.append(ChangepointConfig(time=int(t_cp), new_delta_gp=True))
        else:  # combo
            changepoints.append(ChangepointConfig(time=int(t_cp),
                                                 theta_new=rand_theta(),
                                                 delta_shift=float(random.uniform(-0.5, 0.5)),
                                                 new_delta_gp=True))

    # 创建数据流（先用均匀分布x_dist占位）
    # stream = SyntheticDataStream(
    #     cfg=SyntheticGeneratorConfig(
    #         theta_true=torch.tensor([0.2, -0.5, 0.8, -0.3], dtype=dtype, device=device),
    #         rho=calib_cfg.model.rho,
    #         sigma_eps=calib_cfg.model.sigma_eps,
    #         delta_kernel=calib_cfg.model.delta_kernel,
    #         x_dist=x_dist_uniform,
    #         batch_size_range=batch_size_range,
    #         changepoints=changepoints
    #     ),
    #     eta_func=lambda X, th: eta_func_complex(X, th)
    # )
    def build_stream(seed: int) -> SyntheticDataStream:
        # 构建与当前实验一致的 stream，但用固定 seed
        stream_local = SyntheticDataStream(
            cfg=SyntheticGeneratorConfig(
                theta_true=torch.tensor([0.2, -0.5, 0.8, -0.3], dtype=dtype, device=device),
                rho=calib_cfg.model.rho,
                sigma_eps=calib_cfg.model.sigma_eps,
                delta_kernel=calib_cfg.model.delta_kernel,
                x_dist=lambda b: torch.rand(b, x_dim, dtype=dtype, device=device),  # 先占位
                batch_size_range=batch_size_range,
                changepoints=changepoints
            ),
            eta_func=lambda X, th: eta_func_complex(X, th),
            seed=seed,  # 关键：固定随机种子
        )
        # 设置随时间漂移的 x 分布（使用 stream_local.t）
        def drifting_x_dist(b: int) -> torch.Tensor:
            alpha = min(max(stream_local.t / float(target_observations), 0.0), 1.0)
            base = torch.rand(b, x_dim, dtype=dtype, device=device)
            edge_bias = torch.bernoulli(0.5 * torch.ones(b, x_dim, dtype=dtype, device=device)) * 0.5
            x = (1 - alpha) * base + alpha * edge_bias + 0.05 * torch.randn(b, x_dim, dtype=dtype, device=device)
            return torch.clamp(x, 0.0, 1.0)
        stream_local.cfg.x_dist = drifting_x_dist
        return stream_local

    # 将 x_dist 改为随时间漂移：前期均匀，后期偏向边缘，且带噪
    def drifting_x_dist(b: int) -> torch.Tensor:
        # 读取 stream.t（数据流内部的全局时间）
        alpha = min(max(stream.t / float(target_observations), 0.0), 1.0)
        base = torch.rand(b, x_dim, dtype=dtype, device=device)
        edge_bias = torch.bernoulli(0.5 * torch.ones(b, x_dim, dtype=dtype, device=device)) * 0.5
        x = (1 - alpha) * base + alpha * edge_bias + 0.05 * torch.randn(b, x_dim, dtype=dtype, device=device)
        return torch.clamp(x, 0.0, 1.0)

    # 替换数据流的 x_dist
    # stream.cfg.x_dist = drifting_x_dist

    # 运行两种方法
    results = {}

    seed_fixed = 12345
    for method_name, bocpd_mode in [("Standard", "standard"), ("Restart", "restart")]:
        print(f"\n{'='*60}")
        print(f"Running {method_name} BOCPD")
        print(f"{'='*60}")
        stream = build_stream(seed_fixed)

        # 重置数据流（时间/参数/δGP）
        stream.t = 0
        stream.theta_current = torch.tensor([0.2, -0.5, 0.8, -0.3], dtype=dtype, device=device)
        stream.delta_shift = 0.0
        stream.processed_changepoints = set()
        stream._init_delta_gp()

        # 配置 calibrator 模式
        calib_cfg.bocpd.bocpd_mode = bocpd_mode
        if bocpd_mode == "restart":
            calib_cfg.bocpd.use_backdated_restart = False
            calib_cfg.bocpd.restart_margin = 0.08
            calib_cfg.bocpd.restart_cooldown = 25
        else:
            calib_cfg.bocpd.use_restart = True
            calib_cfg.bocpd.restart_threshold = 0.85

        calibrator = OnlineBayesCalibrator(calib_cfg, emulator, prior_sampler)

        # 记录
        prediction_errors = []
        rmse_history = []
        crps_history = []
        restart_detections = []        # t（观测起始索引）列表
        expert_run_lengths = []        # 每batch记录一次
        batch_times_all = []           # 每batch记录一次（该batch的起始时间点，用 total_observations 表示）

        total_observations = 0
        iteration = 0

        while total_observations < target_observations:
            X_batch, Y_batch = stream.next()
            batch_size = X_batch.shape[0]
            batch_times_all.append(total_observations)

            # 注入异常与异方差（更复杂）
            if torch.rand(()) < 0.3:
                k = torch.randint(1, max(2, batch_size // 10), ())
                idx = torch.randperm(batch_size)[:k]
                Y_batch[idx] = Y_batch[idx] + torch.randn_like(Y_batch[idx]) * 3.0  # 异常点

            mask = (X_batch[:, 0] > 0.8)
            Y_batch = Y_batch + mask.float() * torch.randn_like(Y_batch) * 0.3     # 异方差

            # 预测与评估（先预测再更新）
            if total_observations > 0:
                pred_result = calibrator.predict_batch(X_batch)
                mu_pred = pred_result["mu"]
                var_pred = pred_result["var"]

                pred_err = (Y_batch - mu_pred)
                prediction_errors.extend(pred_err.detach().cpu().numpy().tolist())

                rmse = float(np.sqrt(np.mean(np.array(prediction_errors, dtype=np.float64) ** 2)))
                rmse_history.append(rmse)

                crps_values = calculate_crps(Y_batch, mu_pred, var_pred)
                crps_history.extend(crps_values.detach().cpu().numpy().tolist())

            # 更新
            out = calibrator.step_batch(X_batch, Y_batch, verbose=False)

            # 记录重启检测（restart 模式下）
            if out.get("did_restart", False):
                restart_detections.append(total_observations)

            # 记录expert run_lengths（仅取top2便于画图）
            rls = {}
            for i, e in enumerate(calibrator.bocpd.experts[:2]):
                rls[f"expert_{i}"] = int(e.run_length)
            expert_run_lengths.append(rls)

            total_observations += batch_size
            iteration += 1

        # times_rmse 与 rmse_history 一一对应（从第二个batch开始）
        times_rmse = batch_times_all[1:1 + len(rmse_history)]

        results[method_name] = {
            "rmse_history": rmse_history,
            "crps_history": crps_history,
            "restart_detections": restart_detections,
            "expert_run_lengths": expert_run_lengths,
            "batch_times_all": batch_times_all,   # 每个batch的起始t
            "times_rmse": times_rmse,            # 与rmse_history对齐的时间轴
            "changepoint_times": [cp.time for cp in changepoints],
        }

    # 绘图
    plot_comparison_results(results, prefix=prefix)
    # plot_detailed_analysis(results, prefix=prefix)

    # 保存简要结果（npz）
    np.savez_compressed(
        f"{prefix}_results_summary.npz",
        standard_rmse=np.array(results["Standard"]["rmse_history"]),
        restart_rmse=np.array(results["Restart"]["rmse_history"]),
        standard_times_rmse=np.array(results["Standard"]["times_rmse"]),
        restart_times_rmse=np.array(results["Restart"]["times_rmse"]),
        standard_cps=np.array(results["Standard"]["changepoint_times"]),
        restart_cps=np.array(results["Restart"]["changepoint_times"]),
        standard_restart_detect=np.array(results["Standard"]["restart_detections"]),
        restart_restart_detect=np.array(results["Restart"]["restart_detections"]),
    )

    # 打印简要统计
    print("\n" + "=" * 60)
    print("Comparison Summary:")
    print("=" * 60)
    for method_name, data in results.items():
        final_rmse = data["rmse_history"][-1] if data["rmse_history"] else float("nan")
        avg_crps_last = float(np.mean(data["crps_history"][-50:])) if len(data["crps_history"]) >= 50 else float("nan")
        print(f"\n{method_name} BOCPD:")
        print(f"  Final RMSE: {final_rmse:.4f}")
        print(f"  Avg CRPS (last 50 pts): {avg_crps_last:.4f}")
        print(f"  Restarts detected: {len(data['restart_detections'])} at {data['restart_detections']}")
        print(f"  True changepoints: {data['changepoint_times']}")

    return results


def plot_comparison_results(results, prefix: str = "exp"):
    """绘制 RMSE/CRPS 对比、重启检测标注、Standard 的前两个expert run_length 的曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. RMSE对比
    ax1 = axes[0, 0]
    for method_name, data in results.items():
        xs = data["times_rmse"]
        ys = data["rmse_history"]
        m = min(len(xs), len(ys))
        if m > 0:
            ax1.plot(xs[:m], ys[:m], label=f"{method_name} BOCPD", linewidth=2)

    # 标注真实跳变点
    changepoint_times = results["Standard"]["changepoint_times"]
    for cp_time in changepoint_times:
        ax1.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7,
                    label='True Changepoint' if cp_time == changepoint_times[0] else "")

    # 标注restart检测点
    for method_name, data in results.items():
        if data["restart_detections"]:
            ax1.vlines(data["restart_detections"], ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1],
                       color='green', linestyle=':', alpha=0.5,
                       label=f'{method_name} Restart' if method_name == "Standard" else "")

    ax1.set_xlabel('Observation Time (t)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CRPS对比（按rmse对齐的batch估计一个CRPS）
    ax2 = axes[0, 1]
    for method_name, data in results.items():
        xs_rmse = data["times_rmse"]
        crps_by_batch = []
        # 近似按 batch（rmse 时间轴长度）去均匀取 CRPS（每个batch取5个点均值）
        current = 0
        crps_all = data["crps_history"]
        for _ in range(len(xs_rmse)):
            if current < len(crps_all):
                window = crps_all[current:current + 5]
                if len(window) == 0:
                    break
                crps_by_batch.append(float(np.mean(window)))
                current += 5
            else:
                break
        m = min(len(xs_rmse), len(crps_by_batch))
        if m > 0:
            ax2.plot(xs_rmse[:m], crps_by_batch[:m], label=f"{method_name} BOCPD", linewidth=2)

    for cp_time in changepoint_times:
        ax2.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Observation Time (t)')
    ax2.set_ylabel('CRPS (per-batch avg)')
    ax2.set_title('CRPS Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Expert Run Length (Standard方法，前两个expert)
    ax3 = axes[1, 0]
    std = results["Standard"]
    xs_all = std["batch_times_all"]
    for expert_key in ["expert_0", "expert_1"]:
        rls = [rl.get(expert_key, 0) for rl in std["expert_run_lengths"]]
        m = min(len(xs_all), len(rls))
        if m > 0:
            ax3.plot(xs_all[:m], rls[:m], label=expert_key.replace("_", " ").title(), linewidth=2)

    for cp_time in changepoint_times:
        ax3.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7,
                    label='True Changepoint' if cp_time == changepoint_times[0] else "")

    ax3.set_xlabel('Observation Time (t)')
    ax3.set_ylabel('Run Length')
    ax3.set_title('Expert Run Lengths (Standard BOCPD)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Restart检测对比
    ax4 = axes[1, 1]
    for method_name, data in results.items():
        if data["restart_detections"]:
            ax4.scatter(data["restart_detections"], [1] * len(data["restart_detections"]),
                        label=f'{method_name} Restarts', s=100, alpha=0.7)

    for cp_time in changepoint_times:
        ax4.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7,
                    label='True Changepoint' if cp_time == changepoint_times[0] else "")

    ax4.set_xlabel('Observation Time (t)')
    ax4.set_yticks([1])
    ax4.set_yticklabels(['Detected'])
    ax4.set_title('Restart Detection vs True Changepoints')
    ax4.set_ylim(0.5, 1.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{prefix}_bocpd_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_detailed_analysis(results, prefix: str = "exp"):
    """更详细的分析图表（如需可单独启用）。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 1. 双轴图：RMSE + Run Length
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    for method_name, data in results.items():
        xs = data["times_rmse"]
        ys = data["rmse_history"]
        m = min(len(xs), len(ys))
        if m > 0:
            ax1.plot(xs[:m], ys[:m], label=f"{method_name} RMSE", linewidth=2)

    std = results["Standard"]
    xs_std = std["batch_times_all"]
    expert_0_rls = [rl.get("expert_0", 0) for rl in std["expert_run_lengths"]]
    m0 = min(len(xs_std), len(expert_0_rls))
    if m0 > 0:
        ax1_twin.plot(xs_std[:m0], expert_0_rls[:m0], 'g--', label='Expert 0 Run Length', linewidth=2)

    for cp_time in std["changepoint_times"]:
        ax1.axvline(x=cp_time, color='red', linestyle=':', alpha=0.7)
        ax1_twin.axvline(x=cp_time, color='red', linestyle=':', alpha=0.7)

    ax1.set_xlabel('Observation Time (t)')
    ax1.set_ylabel('RMSE', color='blue')
    ax1_twin.set_ylabel('Run Length', color='green')
    ax1.set_title('RMSE vs Run Length Evolution')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Expert质量分析
    ax2 = axes[0, 1]
    for expert_key in ["expert_0", "expert_1"]:
        rls = [rl.get(expert_key, 0) for rl in std["expert_run_lengths"]]
        m = min(len(xs_std), len(rls))
        if m > 0:
            ax2.plot(xs_std[:m], rls[:m], label=expert_key.replace("_", " ").title(), linewidth=2)

    for cp_time in std["changepoint_times"]:
        ax2.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Observation Time (t)')
    ax2.set_ylabel('Run Length')
    ax2.set_title('Expert Run Length Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 预测误差分布（滑窗）
    ax3 = axes[1, 0]
    for method_name, data in results.items():
        window_size = 10
        error_windows = []
        time_windows = []
        xs_rmse = data["times_rmse"]
        for i in range(0, len(data["rmse_history"]), window_size):
            if i + window_size <= len(data["rmse_history"]):
                window_errors = data["rmse_history"][i:i + window_size]
                error_windows.append(float(np.mean(window_errors)))
                time_windows.append(xs_rmse[min(i + window_size // 2, len(xs_rmse) - 1)])
        if len(time_windows) > 0:
            ax3.plot(time_windows, error_windows, label=f"{method_name} (windowed)", linewidth=2)

    ax3.set_xlabel('Observation Time (t)')
    ax3.set_ylabel('Windowed RMSE')
    ax3.set_title('Error Evolution (Smoothed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Restart检测准确性（平均延迟）
    ax4 = axes[1, 1]
    true_cps = std["changepoint_times"]
    methods = []
    deltas = []
    for method_name, data in results.items():
        detected = data["restart_detections"]
        delays = []
        for true_cp in true_cps:
            if detected:
                closest = min(detected, key=lambda x: abs(x - true_cp))
                delays.append(closest - true_cp)
        if delays:
            methods.append(method_name)
            deltas.append(float(np.mean(delays)))
    if methods:
        ax4.bar(methods, deltas, label='Avg detection delay')
    ax4.set_ylabel('Average Detection Delay')
    ax4.set_title('Changepoint Detection Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 模型置信度（留空位）
    ax5 = axes[2, 0]
    ax5.text(0.5, 0.5, 'Prediction Variance Analysis\n(需要从calibrator获取更多信息)',
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Model Confidence Analysis')

    # 6. 综合性能指标
    ax6 = axes[2, 1]
    methods = list(results.keys())
    final_rmse = [results[m]["rmse_history"][-1] if results[m]["rmse_history"] else float("nan")
                  for m in methods]
    final_crps = [float(np.mean(results[m]["crps_history"][-50:])) if len(results[m]["crps_history"]) >= 50 else float("nan")
                  for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax6.bar(x - width / 2, final_rmse, width, label='Final RMSE', alpha=0.8)
    ax6.bar(x + width / 2, final_crps, width, label='Final CRPS (last 50 avg)', alpha=0.8)
    ax6.set_xlabel('Method')
    ax6.set_ylabel('Performance Metric')
    ax6.set_title('Final Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{prefix}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 可选的单跑入口（默认不执行），推荐使用 run_comparison_experiment
    print("=" * 60)
    print("Starting Online Bayesian Calibration with BOCPD (Single-run not configured in this script).")
    print("=" * 60)


if __name__ == "__main__":
    # 主要入口：对比实验，支持自定义前缀
    results = run_comparison_experiment(prefix="exp")
    # 如需更多图：打开下面注释
    # plot_detailed_analysis(results, prefix="exp")