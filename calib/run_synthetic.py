# =============================================================
# file: calib/run_synthetic.py
# =============================================================
import torch
import math

from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator
from .data import SyntheticDataStream, SyntheticGeneratorConfig, ChangepointConfig


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


if __name__ == "__main__":
    main()