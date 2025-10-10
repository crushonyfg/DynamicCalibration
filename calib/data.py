# =============================================================
# file: calib/data.py
# =============================================================
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict
import torch
from .configs import DeltaKernelConfig
from .kernels import make_kernel


@dataclass
class ChangepointConfig:
    """跳变点配置"""
    time: int  # 发生跳变的时间步
    theta_new: Optional[torch.Tensor] = None  # 新的 theta (如果为None则保持不变)
    delta_shift: Optional[float] = None  # delta 的整体偏移 (如果为None则不偏移)
    new_delta_gp: bool = False  # 是否生成全新的 delta GP 函数


@dataclass
class SyntheticGeneratorConfig:
    theta_true: torch.Tensor  # [dθ] - 初始真实参数
    rho: float
    sigma_eps: float
    delta_kernel: DeltaKernelConfig
    x_dist: Callable[[int], torch.Tensor]  # sampler for X: returns [b, dx]
    batch_size_range: Tuple[int, int] = (5, 10)  # 每次生成的数据点数量范围
    changepoints: List[ChangepointConfig] = field(default_factory=list)  # 跳变点列表


class SyntheticDataStream:
    def __init__(self,
                 cfg: SyntheticGeneratorConfig,
                 eta_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 seed: Optional[int] = None):
        self.cfg = cfg
        try:
            x_probe = cfg.x_dist(1)
            # 保证 dtype/device 一致
            x_probe = x_probe.to(cfg.theta_true.device, cfg.theta_true.dtype)
            assert x_probe.dim() == 2 and x_probe.shape[0] == 1, "x_dist(1) 必须返回 [1, dx]"
            self.dx = int(x_probe.shape[1])
        except Exception as e:
            # 回退：默认2维（与旧代码兼容），但建议修复 x_dist 以可探测
            self.dx = 2

        self.eta_func = eta_func
        self.rng = torch.Generator(device=cfg.theta_true.device).manual_seed(seed or 0)
        self.kernel = make_kernel(cfg.delta_kernel)
        
        # 当前状态
        self.t = 0  # 当前时间步（单个数据点计数）
        self.theta_current = cfg.theta_true.clone()  # 当前使用的 theta
        self.delta_shift = 0.0  # delta 的整体偏移
        
        # 初始化第一个 delta GP 函数
        self._init_delta_gp()
        
        # 跟踪已处理的 changepoints
        self.processed_changepoints = set()

    def _init_delta_gp(self):
        """初始化或重新生成 delta GP 函数"""
        self.n_inducing = 100
        self.X_inducing = torch.rand(
            self.n_inducing, self.dx, 
            dtype=self.cfg.theta_true.dtype, 
            device=self.cfg.theta_true.device
        )
        K_uu = self.kernel.cov(self.X_inducing, self.X_inducing)
        K_uu += 1e-6 * torch.eye(self.n_inducing, dtype=K_uu.dtype, device=K_uu.device)
        L_uu = torch.linalg.cholesky(K_uu)
        z = torch.randn(
            self.n_inducing, 
            dtype=self.cfg.theta_true.dtype, 
            device=self.cfg.theta_true.device, 
            generator=self.rng
        )
        self.delta_inducing = L_uu @ z  # [n_inducing]

    def _check_changepoints(self, t_current: int):
        """检查并应用 changepoints"""
        for cp in self.cfg.changepoints:
            if cp.time == t_current and cp.time not in self.processed_changepoints:
                print(f"\n🔄 [t={t_current}] Changepoint occurred!")
                
                # 更新 theta
                if cp.theta_new is not None:
                    old_theta = self.theta_current.clone()
                    self.theta_current = cp.theta_new.to(
                        device=self.cfg.theta_true.device,
                        dtype=self.cfg.theta_true.dtype
                    )
                    print(f"   θ: {old_theta.cpu().numpy()} → {self.theta_current.cpu().numpy()}")
                
                # 更新 delta shift
                if cp.delta_shift is not None:
                    old_shift = self.delta_shift
                    self.delta_shift = cp.delta_shift
                    print(f"   δ_shift: {old_shift:.3f} → {self.delta_shift:.3f}")
                
                # 生成新的 delta GP
                if cp.new_delta_gp:
                    self._init_delta_gp()
                    print(f"   Generated new δ GP function")
                
                self.processed_changepoints.add(cp.time)
                print()

    def _delta(self, X: torch.Tensor) -> torch.Tensor:
        """✅ 从固定的 GP 函数条件采样，加上可能的偏移"""
        if X.shape[0] == 0:
            return torch.zeros(0, dtype=self.cfg.theta_true.dtype, device=self.cfg.theta_true.device)
        
        # 使用 inducing points 进行条件预测
        K_xu = self.kernel.cov(X, self.X_inducing)  # [b, n_inducing]
        K_uu = self.kernel.cov(self.X_inducing, self.X_inducing)
        K_uu += 1e-6 * torch.eye(self.n_inducing, dtype=K_xu.dtype, device=K_xu.device)
        
        # 条件均值：mu = K_xu @ K_uu^{-1} @ delta_inducing + shift
        alpha = torch.linalg.solve(K_uu, self.delta_inducing)
        mu = K_xu @ alpha + self.delta_shift
        
        K_xx = self.kernel.cov(X, X)
        cond_var = K_xx - K_xu @ torch.linalg.solve(K_uu, K_xu.T)
        cond_var += 1e-6 * torch.eye(X.shape[0], dtype=K_xx.dtype, device=K_xx.device)
        
        try:
            L_cond = torch.linalg.cholesky(cond_var)
            z = torch.randn(X.shape[0], dtype=X.dtype, device=X.device, generator=self.rng)
            return mu + L_cond @ z
        except:
            return mu


    def next(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成下一批数据
        """
        # 确定批量大小
        if batch_size is None:
            b_min, b_max = self.cfg.batch_size_range
            batch_size = torch.randint(b_min, b_max + 1, (1,), generator=self.rng).item()
        
        # ✅ 修复：检查batch范围内的所有changepoints
        batch_start = self.t
        batch_end = self.t + batch_size
        
        # 找到batch范围内的所有changepoints
        changepoints_in_batch = []
        for cp in self.cfg.changepoints:
            if batch_start <= cp.time < batch_end and cp.time not in self.processed_changepoints:
                changepoints_in_batch.append(cp)
        
        # 如果有changepoints，分段生成数据
        if changepoints_in_batch:
            # 按时间排序
            changepoints_in_batch.sort(key=lambda cp: cp.time)
            
            X_list = []
            Y_list = []
            current_t = self.t
            
            for cp in changepoints_in_batch:
                # 生成changepoint之前的数据
                if cp.time > current_t:
                    n_before = cp.time - current_t
                    X_before = self.cfg.x_dist(n_before).to(self.cfg.theta_true.device, self.cfg.theta_true.dtype)
                    mu_eta = self.eta_func(X_before, self.theta_current[None, :]).squeeze(-1)
                    delta = self._delta(X_before)
                    eps = self.cfg.sigma_eps * torch.randn(n_before, dtype=X_before.dtype, device=X_before.device, generator=self.rng)
                    Y_before = self.cfg.rho * mu_eta + delta + eps
                    X_list.append(X_before)
                    Y_list.append(Y_before)
                    current_t = cp.time
                
                # 应用changepoint
                self._check_changepoints(cp.time)
                current_t = cp.time
            
            # 生成changepoint之后的数据
            if current_t < batch_end:
                n_after = batch_end - current_t
                X_after = self.cfg.x_dist(n_after).to(self.cfg.theta_true.device, self.cfg.theta_true.dtype)
                mu_eta = self.eta_func(X_after, self.theta_current[None, :]).squeeze(-1)
                delta = self._delta(X_after)
                eps = self.cfg.sigma_eps * torch.randn(n_after, dtype=X_after.dtype, device=X_after.device, generator=self.rng)
                Y_after = self.cfg.rho * mu_eta + delta + eps
                X_list.append(X_after)
                Y_list.append(Y_after)
            
            X = torch.cat(X_list, dim=0)
            Y = torch.cat(Y_list, dim=0)
            
        else:
            # 没有changepoints，正常生成
            self._check_changepoints(self.t)
            X = self.cfg.x_dist(batch_size).to(self.cfg.theta_true.device, self.cfg.theta_true.dtype)
            mu_eta = self.eta_func(X, self.theta_current[None, :]).squeeze(-1)
            delta = self._delta(X)
            eps = self.cfg.sigma_eps * torch.randn(batch_size, dtype=X.dtype, device=X.device, generator=self.rng)
            Y = self.cfg.rho * mu_eta + delta + eps
        
        # 更新时间步
        self.t += batch_size
        
        return X, Y