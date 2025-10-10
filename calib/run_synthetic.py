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

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']  # 按系统可用字体列几个
rcParams['axes.unicode_minus'] = False

def calculate_crps(y_true: torch.Tensor, mu_pred: torch.Tensor, var_pred: torch.Tensor) -> torch.Tensor:
    """计算CRPS"""
    sigma = torch.sqrt(var_pred)
    z = (y_true - mu_pred) / sigma
    crps = sigma * (z * torch.erf(z / torch.sqrt(torch.tensor(2.0))) + 
                    torch.sqrt(torch.tensor(2.0 / np.pi)) * torch.exp(-0.5 * z**2) - 
                    torch.sqrt(torch.tensor(1.0 / np.pi)))
    return crps

def main():
    # 1) Build configs
    calib_cfg = CalibrationConfig()
    # calib_cfg.bocpd.bocpd_mode = "standard"  # 使用标准 BOCPD
    calib_cfg.bocpd.bocpd_mode = "standard"

    if calib_cfg.bocpd.bocpd_mode == "restart":
        calib_cfg.bocpd.use_backdated_restart = False  # False=Algorithm-2, True=Backdated
        calib_cfg.bocpd.restart_margin = 0.05
        calib_cfg.bocpd.restart_cooldown = 10
    else:
        # Standard BOCPD 配置
        calib_cfg.bocpd.use_restart = True
        calib_cfg.bocpd.restart_threshold = 0.8
    device, dtype = calib_cfg.model.device, calib_cfg.model.dtype

    # 2) Define prior sampler over θ (toy: 2-D uniform box)
    def prior_sampler(N: int) -> torch.Tensor:
        lo = torch.tensor([-1.0, -1.5], dtype=dtype, device=device)
        hi = torch.tensor([+1.0, +0.5], dtype=dtype, device=device)
        u = torch.rand(N, 2, dtype=dtype, device=device)
        return lo + (hi - lo) * u

    # 3) Define toy η(x, θ): η = θ₀ * x₀ + sin(θ₁ * x₁)
    def eta_func(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta[None, :]
        if x.dim() == 1:
            x = x[None, :]
        outs = []
        for n in range(theta.shape[0]):
            th = theta[n]
            val = th[0] * x[:, 0] + torch.sin(th[1] * x[:, 1])
            outs.append(val[:, None])
        return torch.cat(outs, dim=1)  # [b,N]

    emulator = DeterministicSimulator(func=eta_func, enable_autograd=True)

    # 4) Calibrator orchestrator
    def on_restart_hook(t, r_new, s_star, mode, p_anchor, best_other):
        """Optional callback when restart happens (only for R-BOCPD)"""
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.info(
            f"[HOOK] Restart at t={t}: r←{r_new}, s*={s_star}, "
            f"anchor_rl={mode}, p_anchor={p_anchor:.4f}, best={best_other:.4f}"
        )
    
    calibrator = OnlineBayesCalibrator(
        calib_cfg, 
        emulator, 
        prior_sampler,
        on_restart=on_restart_hook if calib_cfg.bocpd.bocpd_mode == "restart" else None,
        notify_on_restart=True,
    )

    # 5) ✅ Synthetic stream with changepoints
    changepoints = [
        ChangepointConfig(
            time=30,  # 第30个数据点时发生跳变
            theta_new=torch.tensor([0.5, -0.3], dtype=dtype, device=device),  # 切换到新的 theta
            new_delta_gp=True  # 生成全新的 delta GP
        ),
        ChangepointConfig(
            time=60,  # 第60个数据点时再次跳变
            theta_new=torch.tensor([-0.3, -1.0], dtype=dtype, device=device),
            delta_shift=0.2,  # 添加一个整体偏移
            new_delta_gp=True  # 保持相同的 delta GP，只加偏移
        ),
    ]
    
    stream = SyntheticDataStream(
        cfg=SyntheticGeneratorConfig(
            theta_true=torch.tensor([0.3, -0.7], dtype=dtype, device=device),
            rho=calib_cfg.model.rho,
            sigma_eps=calib_cfg.model.sigma_eps,
            delta_kernel=calib_cfg.model.delta_kernel,
            x_dist=lambda b: torch.rand(b, 2, dtype=dtype, device=device),
            batch_size_range=(5, 10),  # ✅ 每次生成5-10个数据点
            changepoints=changepoints  # ✅ 跳变点配置
        ),
        eta_func=lambda X, th: eta_func(X, th)
    )

    # 6) Run online
    print("="*60)
    print("Starting Online Bayesian Calibration with BOCPD")
    print("="*60)
    print(f"Initial θ_true: {stream.theta_current.cpu().numpy()}")
    print(f"Changepoints at t={[cp.time for cp in changepoints]}")
    print(f"Batch size: {stream.cfg.batch_size_range}")
    print("="*60)
    
    verbose_steps = [0, 1, 2, 10, 20, 30, 40, 60, 70]  # 关键时间步

    use_batch = True
    if use_batch:
        prediction_errors = []
        prediction_variances = []
        rmse_history = []
        crps_history = []
        
        total_observations = 0
        iteration = 0
        
        while total_observations < 100:
            X_batch, Y_batch = stream.next()
            batch_size = X_batch.shape[0]
            
            # ✅ 关键修改：先预测当前batch，计算误差，再更新calibrator
            if total_observations > 0:  # 不是第一批数据
                # 1. 用当前calibrator状态预测X_batch
                pred_result = calibrator.predict_batch(X_batch)
                mu_pred = pred_result["mu"]  # [batch_size]
                var_pred = pred_result["var"]  # [batch_size]
                
                # 2. 计算预测误差和评估指标
                pred_errors = Y_batch - mu_pred  # [batch_size]
                prediction_errors.extend(pred_errors.cpu().numpy().tolist())
                prediction_variances.extend(var_pred.cpu().numpy().tolist())
                
                # 3. 计算累积RMSE
                rmse = np.sqrt(np.mean(np.array(prediction_errors)**2))
                rmse_history.append(rmse)
                
                # 4. 计算CRPS
                crps_values = calculate_crps(Y_batch, mu_pred, var_pred)
                crps_history.extend(crps_values.cpu().numpy().tolist())
                
                # 输出评估结果
                avg_crps = np.mean(crps_values.cpu().numpy())
                print(f"Batch {iteration}: RMSE={rmse:.4f}, Avg CRPS={avg_crps:.4f}")
                print(f"  Predictions: μ={mu_pred.cpu().numpy()}")
                print(f"  True values: {Y_batch.cpu().numpy()}")
                print(f"  Errors: {pred_errors.cpu().numpy()}")
            
            # 3. 用真实数据更新calibrator（批量更新）
            verbose = any(total_observations + i in verbose_steps for i in range(batch_size))
            out = calibrator.step_batch(X_batch, Y_batch, verbose=verbose)
            
            total_observations += batch_size
            iteration += 1
        
        # 输出最终评估结果
        print(f"\n{'='*60}")
        print("Final Prediction Performance:")
        print(f"{'='*60}")
        print(f"Final RMSE: {rmse_history[-1]:.4f}")
        print(f"Final Avg CRPS: {np.mean(crps_history):.4f}")
        print(f"Average prediction variance: {np.mean(prediction_variances):.4f}")
    
    else:
        total_observations = 0
        iteration = 0
        
        # ✅ 运行直到观测到至少100个数据点
        while total_observations < 100:
            X_batch, Y_batch = stream.next()  # 生成一批数据
            batch_size = X_batch.shape[0]
            
            # ✅ 逐个处理批次中的每个数据点
            for i in range(batch_size):
                X_t = X_batch[i:i+1, :]  # [1, dx]
                Y_t = Y_batch[i:i+1]     # [1]
                
                verbose = total_observations in verbose_steps
                out = calibrator.step(X_t.squeeze(0), Y_t.squeeze(0), verbose=verbose)
                
                ess_str = f"{out['pf_diags'][0]['ess']:.1f}" if out['pf_diags'] else "N/A"
                
                if not verbose:
                    print(f"obs={total_observations:3d} (iter={iteration:2d}, i={i:1d}) | "
                        f"p_cp={out['p_cp']:.3f} | "
                        f"experts={out['num_experts']} | "
                        f"ESS(e0)={ess_str} | "
                        f"resampled={out['pf_diags'][0]['resampled'] if out['pf_diags'] else False}")
                
                total_observations += 1
                
                if total_observations >= 100:
                    break
            
            iteration += 1
        
        print("="*60)
        print("Calibration completed successfully!")
        print("="*60)
        
        # Final statistics
        if len(calibrator.bocpd.experts) > 0:
            print(f"\n{'='*60}")
            print("Run-length Distribution:")
            print(f"{'='*60}")
            
            experts_sorted = sorted(calibrator.bocpd.experts, key=lambda e: e.run_length)
            
            for e in experts_sorted:
                prob = math.exp(e.log_mass)
                theta_mean = (e.pf.particles.weights()[:, None] * 
                            e.pf.particles.theta).sum(0)
                ess = e.pf.particles.ess().item()
                
                print(f"  r={e.run_length:3d} | "
                    f"P(r_t={e.run_length}|data)={prob:6.4f} | "
                    f"log_mass={e.log_mass:8.3f} | "
                    f"θ̂=[{theta_mean[0]:+.3f}, {theta_mean[1]:+.3f}] | "
                    f"ESS={ess:5.1f}")
            
            print(f"{'='*60}")
            
            best_expert = max(experts_sorted, key=lambda e: e.log_mass)
            print(f"\nMost probable run-length: r={best_expert.run_length} "
                f"(P={math.exp(best_expert.log_mass):.4f})")

            # Longest-lived expert
            longest_expert = max(calibrator.bocpd.experts, key=lambda e: e.run_length)
            theta_mean = (longest_expert.pf.particles.weights()[:, None] * 
                        longest_expert.pf.particles.theta).sum(0)
            theta_std = torch.sqrt(((longest_expert.pf.particles.theta - theta_mean)**2 * 
                                longest_expert.pf.particles.weights()[:, None]).sum(0))
            
            print(f"\nLongest-lived expert (run_length={longest_expert.run_length}):")
            print(f"  Estimated θ: {theta_mean.cpu().numpy()} ± {theta_std.cpu().numpy()}")
            print(f"  Current true θ: {stream.theta_current.cpu().numpy()}")
            print(f"  Log mass: {longest_expert.log_mass:.3f}")
            print(f"  Error: {torch.norm(theta_mean - stream.theta_current).item():.3f}")

            # Delta GP diagnostics
            print(f"\n{'='*60}")
            print("Delta GP Diagnostics:")
            print(f"{'='*60}")
            
            for i, e in enumerate(calibrator.bocpd.experts[:3]):
                if e.delta_state.X.shape[0] > 0:
                    mu_delta, var_delta = e.delta_state.predict(e.delta_state.X)
                    residuals = e.delta_state.y - mu_delta
                    rmse = torch.sqrt((residuals**2).mean()).item()
                    
                    print(f"\n  Expert {i} (r={e.run_length}, {e.delta_state.X.shape[0]} points):")
                    print(f"    Delta RMSE: {rmse:.4f}")
                    print(f"    Delta mean: {e.delta_state.y.mean():.4f} ± {e.delta_state.y.std():.4f}")


def run_comparison_experiment():
    """同时运行standard和restart两种BOCPD方法进行对比"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 配置
    calib_cfg = CalibrationConfig()
    device, dtype = calib_cfg.model.device, calib_cfg.model.dtype
    
    # 定义相同的prior和emulator
    def prior_sampler(N: int) -> torch.Tensor:
        lo = torch.tensor([-1.0, -1.5], dtype=dtype, device=device)
        hi = torch.tensor([+1.0, +0.5], dtype=dtype, device=device)
        u = torch.rand(N, 2, dtype=dtype, device=device)
        return lo + (hi - lo) * u

    def eta_func(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta[None, :]
        if x.dim() == 1:
            x = x[None, :]
        outs = []
        for n in range(theta.shape[0]):
            th = theta[n]
            val = th[0] * x[:, 0] + torch.sin(th[1] * x[:, 1])
            outs.append(val[:, None])
        return torch.cat(outs, dim=1)

    emulator = DeterministicSimulator(func=eta_func, enable_autograd=True)
    
    # 定义changepoints
    changepoints = [
        ChangepointConfig(time=30, theta_new=torch.tensor([0.5, -0.3], dtype=dtype, device=device), new_delta_gp=True),
        ChangepointConfig(time=60, theta_new=torch.tensor([-0.3, -1.0], dtype=dtype, device=device), delta_shift=0.2, new_delta_gp=True),
    ]
    
    # 创建数据流
    # stream = SyntheticDataStream(
    #     cfg=SyntheticGeneratorConfig(
    #         theta_true=torch.tensor([0.3, -0.7], dtype=dtype, device=device),
    #         rho=calib_cfg.model.rho,
    #         sigma_eps=calib_cfg.model.sigma_eps,
    #         delta_kernel=calib_cfg.model.delta_kernel,
    #         x_dist=lambda b: torch.rand(b, 2, dtype=dtype, device=device),
    #         batch_size_range=(5, 10),
    #         changepoints=changepoints
    #     ),
    #     eta_func=lambda X, th: eta_func(X, th)
    # )
    def build_stream(seed: int) -> SyntheticDataStream:
        # 构建与当前实验一致的 stream，但用固定 seed
        stream_local = SyntheticDataStream(
        cfg=SyntheticGeneratorConfig(
            theta_true=torch.tensor([0.3, -0.7], dtype=dtype, device=device),
            rho=calib_cfg.model.rho,
            sigma_eps=calib_cfg.model.sigma_eps,
            delta_kernel=calib_cfg.model.delta_kernel,
            x_dist=lambda b: torch.rand(b, 2, dtype=dtype, device=device),
            batch_size_range=(5, 10),
            changepoints=changepoints
        ),
        eta_func=lambda X, th: eta_func(X, th),
        seed=seed,
    )
        # 设置随时间漂移的 x 分布（使用 stream_local.t）
        x_dim = 2
        target_observations = 100
        def drifting_x_dist(b: int) -> torch.Tensor:
            alpha = min(max(stream_local.t / float(target_observations), 0.0), 1.0)
            base = torch.rand(b, x_dim, dtype=dtype, device=device)
            edge_bias = torch.bernoulli(0.5 * torch.ones(b, x_dim, dtype=dtype, device=device)) * 0.5
            x = (1 - alpha) * base + alpha * edge_bias + 0.05 * torch.randn(b, x_dim, dtype=dtype, device=device)
            return torch.clamp(x, 0.0, 1.0)
        stream_local.cfg.x_dist = drifting_x_dist
        return stream_local
    
    # 运行两种方法
    results = {}
    seed_fixed = 12345
    
    for method_name, bocpd_mode in [("Standard", "standard"), ("Restart", "restart")]:
        print(f"\n{'='*60}")
        print(f"Running {method_name} BOCPD")
        print(f"{'='*60}")
        
        # 重置数据流
        stream = build_stream(seed_fixed) 
        stream.t = 0
        stream.theta_current = torch.tensor([0.3, -0.7], dtype=dtype, device=device)
        stream.processed_changepoints = set()
        stream._init_delta_gp()
        
        # 配置calibrator
        calib_cfg.bocpd.bocpd_mode = bocpd_mode
        if bocpd_mode == "restart":
            calib_cfg.bocpd.use_backdated_restart = False
            calib_cfg.bocpd.restart_margin = 0.05
            calib_cfg.bocpd.restart_cooldown = 10
        else:
            calib_cfg.bocpd.use_restart = True
            calib_cfg.bocpd.restart_threshold = 0.8
        
        calibrator = OnlineBayesCalibrator(calib_cfg, emulator, prior_sampler)
        
        # 运行实验
        prediction_errors = []
        rmse_history = []
        crps_history = []
        restart_detections = []  # 记录restart检测到的时间点
        expert_run_lengths = []  # 记录expert的run_length变化
        batch_times = []  # 记录每个batch对应的时间点
        
        total_observations = 0
        iteration = 0
        
        while total_observations < 100:
            X_batch, Y_batch = stream.next()
            batch_size = X_batch.shape[0]
            batch_start_time = total_observations
            
            # 预测和评估
            if total_observations > 0:
                pred_result = calibrator.predict_batch(X_batch)
                mu_pred = pred_result["mu"]
                var_pred = pred_result["var"]
                
                pred_errors = Y_batch - mu_pred
                prediction_errors.extend(pred_errors.cpu().numpy().tolist())
                
                rmse = np.sqrt(np.mean(np.array(prediction_errors)**2))
                rmse_history.append(rmse)
                
                crps_values = calculate_crps(Y_batch, mu_pred, var_pred)
                crps_history.extend(crps_values.cpu().numpy().tolist())
            
            # 更新calibrator
            out = calibrator.step_batch(X_batch, Y_batch, verbose=False)
            
            # 记录restart检测
            if out.get("did_restart", False):
                restart_detections.append(total_observations)
            
            # 记录expert run_lengths
            expert_rls = {}
            for i, e in enumerate(calibrator.bocpd.experts[:2]):  # 只记录前两个expert
                expert_rls[f"expert_{i}"] = e.run_length
            expert_run_lengths.append(expert_rls)

            batch_times.append(total_observations)
            
            total_observations += batch_size
            iteration += 1
        
        # 保存结果
        results[method_name] = {
            "rmse_history": rmse_history,
            "crps_history": crps_history,
            "restart_detections": restart_detections,
            "expert_run_lengths": expert_run_lengths,
            "batch_times": batch_times[1:1+len(rmse_history)],
            "changepoint_times": [cp.time for cp in changepoints]
        }
    
    # 绘制对比图
    plot_comparison_results(results)
    
    return results

def plot_comparison_results(results):
    """绘制对比结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. RMSE对比
    ax1 = axes[0, 0]
    # for method_name, data in results.items():
    #     ax1.plot(data["batch_times"], data["rmse_history"], label=f"{method_name} BOCPD", linewidth=2)
    for method_name, data in results.items():
        xs = data.get("times_rmse", data["batch_times"][1:1+len(data["rmse_history"])])
        ys = data["rmse_history"]
        m = min(len(xs), len(ys))
        ax1.plot(xs[:m], ys[:m], label=f"{method_name} BOCPD", linewidth=2)
    
    # 标注真实跳变点
    changepoint_times = results["Standard"]["changepoint_times"]
    for cp_time in changepoint_times:
        ax1.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7, label='True Changepoint' if cp_time == changepoint_times[0] else "")
    
    # 标注restart检测点
    for method_name, data in results.items():
        for restart_time in data["restart_detections"]:
            ax1.axvline(x=restart_time, color='green', linestyle=':', alpha=0.7, 
                       label=f'{method_name} Restart' if restart_time == data["restart_detections"][0] else "")
    
    ax1.set_xlabel('Observation Time')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CRPS对比
    ax2 = axes[0, 1]
    for method_name, data in results.items():
        xs_rmse = data.get("times_rmse", data["batch_times"][1:1+len(data["rmse_history"])])
        # 将逐点CRPS粗略聚合到每个batch；聚合数量取与rmse_history一致
        crps_by_batch = []
        current = 0
        for _ in range(len(xs_rmse)):
            if current < len(data["crps_history"]):
                window = data["crps_history"][current:current+5]  # 近似每batch≈5点
                if len(window) == 0: break
                crps_by_batch.append(float(np.mean(window)))
                current += 5
            else:
                break
        m = min(len(xs_rmse), len(crps_by_batch))
        ax2.plot(xs_rmse[:m], crps_by_batch[:m], label=f"{method_name} BOCPD", linewidth=2)
    
    for cp_time in changepoint_times:
        ax2.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Observation Time')
    ax2.set_ylabel('CRPS')
    ax2.set_title('CRPS Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Expert Run Length (Standard方法)
    ax3 = axes[1, 0]
    standard_data = results["Standard"]
    xs = standard_data["batch_times"]
    for expert_key in ["expert_0", "expert_1"]:
        rls = [rl.get(expert_key, 0) for rl in standard_data["expert_run_lengths"]]
        m = min(len(xs), len(rls))
        ax3.plot(xs[:m], rls[:m], label=expert_key.replace("_", " ").title(), linewidth=2)
    
    for cp_time in changepoint_times:
        ax3.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7, 
                   label='True Changepoint' if cp_time == changepoint_times[0] else "")
    
    ax3.set_xlabel('Observation Time')
    ax3.set_ylabel('Run Length')
    ax3.set_title('Expert Run Lengths (Standard BOCPD)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Restart检测对比
    ax4 = axes[1, 1]
    for method_name, data in results.items():
        if data["restart_detections"]:
            ax4.scatter(data["restart_detections"], [1]*len(data["restart_detections"]), 
                       label=f'{method_name} Restarts', s=100, alpha=0.7)
    
    for cp_time in changepoint_times:
        ax4.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7, 
                   label='True Changepoint' if cp_time == changepoint_times[0] else "")
    
    ax4.set_xlabel('Observation Time')
    ax4.set_ylabel('Restart Detection')
    ax4.set_title('Restart Detection vs True Changepoints')
    ax4.set_ylim(0.5, 1.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bocpd_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_analysis(results):
    """更详细的分析图表"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. 双轴图：RMSE + Run Length
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    # RMSE
    for method_name, data in results.items():
        xs = data.get("times_rmse", data["batch_times"][1:1+len(data["rmse_history"])])
        ys = data["rmse_history"]
        m = min(len(xs), len(ys))
        ax1.plot(xs[:m], ys[:m], label=f"{method_name} RMSE", linewidth=2)
    
    # Run Length (Standard方法)
    standard_data = results["Standard"]
    xs_std = standard_data["batch_times"]
    expert_0_rls = [rl.get("expert_0", 0) for rl in standard_data["expert_run_lengths"]]
    m0 = min(len(xs_std), len(expert_0_rls))
    ax1_twin.plot(xs_std[:m0], expert_0_rls[:m0], 'g--', label='Expert 0 Run Length', linewidth=2)
    
    # 标注跳变点
    for cp_time in standard_data["changepoint_times"]:
        ax1.axvline(x=cp_time, color='red', linestyle=':', alpha=0.7)
        ax1_twin.axvline(x=cp_time, color='red', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Observation Time')
    ax1.set_ylabel('RMSE', color='blue')
    ax1_twin.set_ylabel('Run Length', color='green')
    ax1.set_title('RMSE vs Run Length Evolution')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Expert质量分析
    ax2 = axes[0, 1]
    for expert_key in ["expert_0", "expert_1"]:
        rls = [rl.get(expert_key, 0) for rl in standard_data["expert_run_lengths"]]
        m = min(len(xs_std), len(rls))
        ax2.plot(xs_std[:m], rls[:m], label=expert_key.replace("_", " ").title(), linewidth=2)
    
    for cp_time in standard_data["changepoint_times"]:
        ax2.axvline(x=cp_time, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Observation Time')
    ax2.set_ylabel('Run Length')
    ax2.set_title('Expert Run Length Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测误差分布
    ax3 = axes[1, 0]
    for method_name, data in results.items():
        # 计算每个时间窗口的误差统计
        window_size = 10
        error_windows = []
        time_windows = []
        
        for i in range(0, len(data["rmse_history"]), window_size):
            if i + window_size <= len(data["rmse_history"]):
                window_errors = data["rmse_history"][i:i+window_size]
                error_windows.append(np.mean(window_errors))
                time_windows.append(data["batch_times"][i + window_size//2])
        
        ax3.plot(time_windows, error_windows, label=f"{method_name} (windowed)", linewidth=2)
    
    ax3.set_xlabel('Observation Time')
    ax3.set_ylabel('Windowed RMSE')
    ax3.set_title('Error Evolution (Smoothed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Restart检测准确性
    ax4 = axes[1, 1]
    true_cps = standard_data["changepoint_times"]
    
    for method_name, data in results.items():
        detected_cps = data["restart_detections"]
        
        # 计算检测延迟
        delays = []
        for true_cp in true_cps:
            closest_detection = min(detected_cps, key=lambda x: abs(x - true_cp)) if detected_cps else None
            if closest_detection:
                delay = closest_detection - true_cp
                delays.append(delay)
        
        if delays:
            ax4.bar([method_name], [np.mean(delays)], label=f'{method_name} (avg delay: {np.mean(delays):.1f})')
    
    ax4.set_ylabel('Average Detection Delay')
    ax4.set_title('Changepoint Detection Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 模型置信度分析
    ax5 = axes[2, 0]
    # 这里可以添加预测方差的分析
    ax5.text(0.5, 0.5, 'Prediction Variance Analysis\n(需要从calibrator获取更多信息)', 
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Model Confidence Analysis')
    
    # 6. 综合性能指标
    ax6 = axes[2, 1]
    methods = list(results.keys())
    final_rmse = [results[method]["rmse_history"][-1] for method in methods]
    final_crps = [np.mean(results[method]["crps_history"][-10:]) for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax6.bar(x - width/2, final_rmse, width, label='Final RMSE', alpha=0.8)
    ax6.bar(x + width/2, final_crps, width, label='Final CRPS', alpha=0.8)
    
    ax6.set_xlabel('Method')
    ax6.set_ylabel('Performance Metric')
    ax6.set_title('Final Performance Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # main()
    results = run_comparison_experiment()
    # plot_detailed_analysis(results)
