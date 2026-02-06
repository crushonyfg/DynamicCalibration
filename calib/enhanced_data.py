# =============================================================
# file: calib/enhanced_data.py
# =============================================================
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict, Union
import torch
import math
from .configs import DeltaKernelConfig
from .kernels import make_kernel


@dataclass
class PhysicalSystemConfig:
    """物理系统配置"""
    name: str  # "config1", "config2", "config3"
    
    # 计算机模型参数
    computer_model: str  # 模型类型标识
    theta_space: Tuple[float, float]  # 参数空间范围 (min, max)
    theta_dim: int  # 参数维度
    
    # 物理系统参数
    physical_system: str  # 物理系统类型标识
    theta_optimal: torch.Tensor  # 最优参数 θ₀
    
    # 设计空间
    design_space: Tuple[float, float] = (0.0, 1.0)  # 默认 [0,1]
    
    # 噪声参数
    noise_variance: float = 0.04  # 默认 0.2²
    
    # 采样参数
    n_observations: int = 50  # 观测数量
    sampling_strategy: str = "lhs"  # "uniform", "equidistant", "fixed", "lhs"
    fixed_x_values: Optional[List[float]] = None  # 固定x值（用于config3）

    phys_param_dim: int = 0
    phys_param_init: Optional[torch.Tensor] = None


@dataclass
class EnhancedChangepointConfig:
    """增强的跳变点配置"""
    time: int  # 发生跳变的时间步
    theta_new: Optional[torch.Tensor] = None  # 新的 theta
    physical_system_new: Optional[str] = None  # 新的物理系统类型
    noise_variance_new: Optional[float] = None  # 新的噪声方差
    phys_param_new: Optional[torch.Tensor] = None


@dataclass
class EnhancedSyntheticConfig:
    """增强的合成数据配置"""
    physical_config: PhysicalSystemConfig
    delta_kernel: DeltaKernelConfig
    batch_size_range: Tuple[int, int] = (5, 10)
    changepoints: List[EnhancedChangepointConfig] = field(default_factory=list)
    
    # 额外的噪声注入选项
    inject_outliers: bool = False
    outlier_prob: float = 0.3
    outlier_magnitude: float = 3.0
    inject_heteroscedastic: bool = False
    heteroscedastic_condition: str = "x0 > 0.8"
    heteroscedastic_noise: float = 0.3


class EnhancedSyntheticDataStream:
    """支持三个配置的增强合成数据流"""
    
    def __init__(self, cfg: EnhancedSyntheticConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.physical_cfg = cfg.physical_config
        
        # 设置随机数生成器
        self.rng = torch.Generator().manual_seed(seed or 0)
        
        # 当前状态
        self.t = 0
        self.theta_current = self.physical_cfg.theta_optimal.clone()
        self.current_physical_system = self.physical_cfg.physical_system
        self.current_noise_variance = self.physical_cfg.noise_variance

        if self.physical_cfg.phys_param_dim > 0:
            assert self.physical_cfg.phys_param_init is not None
            self.phys_param_current = self.physical_cfg.phys_param_init.clone()
        else:
            self.phys_param_current = None
        
        # 跟踪已处理的 changepoints
        self.processed_changepoints = set()
        
        # 预计算固定x值（用于config3）
        if self.physical_cfg.sampling_strategy == "fixed" and self.physical_cfg.fixed_x_values:
            self.fixed_x_values = torch.tensor(
                self.physical_cfg.fixed_x_values, 
                dtype=torch.float64
            )
        else:
            self.fixed_x_values = None
    
    def _computer_model(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """计算机模型 y*(x, θ)"""
        if self.physical_cfg.computer_model == "config1":
            # y*(x, θ) = 7[sin(2πθ₁ - π)]² + 2[(2πθ₂ - π)² sin(2πx - π)]
            theta1, theta2 = theta[0], theta[1]
            term1 = 7 * torch.sin(2 * math.pi * theta1 - math.pi) ** 2
            term2 = 2 * (2 * math.pi * theta2 - math.pi) ** 2 * torch.sin(2 * math.pi * x - math.pi)
            return term1 + term2
            
        elif self.physical_cfg.computer_model == "config2":
            # y*(x, θ) = sin(5θx) + 5x
            theta_val = theta[0]
            return torch.sin(5 * theta_val * x) + 5 * x
            
        elif self.physical_cfg.computer_model == "config3":
            # y*(x, θ) = θx
            theta_val = theta[0]
            return theta_val * x
            
        else:
            raise ValueError(f"Unknown computer model: {self.physical_cfg.computer_model}")
    
    def _physical_system(self, x: torch.Tensor) -> torch.Tensor:
        """
        物理系统 η(x; φ)。三种配置都参数化，默认 φ 取论文中对应常数。
        - config1: η(x; φ)= y*(x, φ)，φ∈R^2（直接等于 computer model）
        - config2: η(x; φ)= φ1*x*cos(φ2*x) + φ3*x（默认 φ=[5, 15/2, 5]）
        - config3: η(x; φ)= φ1*x + x*sin(φ2*x)（默认 φ=[4, 5]）
        """
        if self.current_physical_system == "config1":
            # φ 直接作为 computer model 的参数（2维）
            assert self.phys_param_current is not None and self.phys_param_current.numel() == 2
            return self._computer_model(x, self.phys_param_current)

        elif self.current_physical_system == "config2":
            # 频率与幅值可跳变
            assert self.phys_param_current is not None and self.phys_param_current.numel() == 3
            a1, a2, a3 = self.phys_param_current
            return a1 * x * torch.cos(a2 * x) + a3 * x

        elif self.current_physical_system == "config3":
            # 线性项与正弦频率可跳变
            assert self.phys_param_current is not None and self.phys_param_current.numel() == 2
            a, b = self.phys_param_current
            return a * x + x * torch.sin(b * x)

        else:
            raise ValueError(f"Unknown physical system: {self.current_physical_system}")
    
    def _sample_x(self, n: int) -> torch.Tensor:
        """根据配置采样x值"""
        if self.physical_cfg.sampling_strategy == "uniform":
            # 均匀采样
            return torch.rand(n, 1, dtype=torch.float64, generator=self.rng) * \
                   (self.physical_cfg.design_space[1] - self.physical_cfg.design_space[0]) + \
                   self.physical_cfg.design_space[0]
                   
        elif self.physical_cfg.sampling_strategy == "equidistant":
            # 等距采样
            x_values = torch.linspace(
                self.physical_cfg.design_space[0],
                self.physical_cfg.design_space[1],
                n,
                dtype=torch.float64
            )
            return x_values.unsqueeze(-1)

        elif self.physical_cfg.sampling_strategy == "lhs":
            return self._sample_x_lhs(n)
            
        elif self.physical_cfg.sampling_strategy == "fixed":
            # 固定值采样（用于config3）
            if self.fixed_x_values is not None:
                # 从固定值中随机选择n个
                indices = torch.randint(0, len(self.fixed_x_values), (n,), generator=self.rng)
                return self.fixed_x_values[indices].unsqueeze(-1)
            else:
                raise ValueError("Fixed x values not provided for fixed sampling strategy")
        else:
            raise ValueError(f"Unknown sampling strategy: {self.physical_cfg.sampling_strategy}")
    
    def _check_changepoints(self, t_current: int):
        """检查并应用 changepoints"""
        for cp in self.cfg.changepoints:
            if cp.time == t_current and cp.time not in self.processed_changepoints:
                print(f"\n🔄 [t={t_current}] Enhanced Changepoint occurred!")
                
                # 更新 theta
                if cp.theta_new is not None:
                    old_theta = self.theta_current.clone()
                    self.theta_current = cp.theta_new
                    print(f"   θ: {old_theta.cpu().numpy()} → {self.theta_current.cpu().numpy()}")
                
                # 更新物理系统
                if cp.physical_system_new is not None:
                    old_system = self.current_physical_system
                    self.current_physical_system = cp.physical_system_new
                    print(f"   Physical System: {old_system} → {self.current_physical_system}")
                
                # 更新噪声方差
                if cp.noise_variance_new is not None:
                    old_var = self.current_noise_variance
                    self.current_noise_variance = cp.noise_variance_new
                    print(f"   Noise Variance: {old_var:.4f} → {self.current_noise_variance:.4f}")

                if cp.phys_param_new is not None:
                    old_phi = None if self.phys_param_current is None else self.phys_param_current.clone()
                    self.phys_param_current = cp.phys_param_new
                    if old_phi is None:
                        print(f"   φ initialized → {self.phys_param_current.cpu().numpy()}")
                    else:
                        print(f"   φ: {old_phi.cpu().numpy()} → {self.phys_param_current.cpu().numpy()}")

                self.processed_changepoints.add(cp.time)
                print()

    def _sample_x_lhs(self, n: int) -> torch.Tensor:
        """
        Latin Hypercube / stratified sampling in 1D.
        - 每个 batch 内均匀覆盖 [low, high]
        - batch 间随机
        """
        low, high = self.physical_cfg.design_space

        # n 个区间
        edges = torch.linspace(
            low, high, n + 1, dtype=torch.float64
        )

        # 每个区间内采一个点
        u = torch.empty(n, dtype=torch.float64)
        for i in range(n):
            u[i] = edges[i] + (edges[i + 1] - edges[i]) * torch.rand(
                (), generator=self.rng, dtype=torch.float64
            )

        # 打乱顺序（LHS 关键）
        perm = torch.randperm(n, generator=self.rng)
        u = u[perm]

        return u.unsqueeze(-1)

    
    def _inject_additional_noise(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """注入额外的噪声（异常值和异方差）"""
        Y_modified = Y.clone()
        
        # 注入异常值
        if self.cfg.inject_outliers:
            if torch.rand(()) < self.cfg.outlier_prob:
                batch_size = X.shape[0]
                k = torch.randint(1, max(2, batch_size // 10), ())
                idx = torch.randperm(batch_size)[:k]
                Y_modified[idx] = Y_modified[idx] + torch.randn_like(Y_modified[idx]) * self.cfg.outlier_magnitude
        
        # 注入异方差
        if self.cfg.inject_heteroscedastic:
            if self.cfg.heteroscedastic_condition == "x0 > 0.8":
                mask = (X[:, 0] > 0.8)
                Y_modified = Y_modified + mask.float() * torch.randn_like(Y_modified) * self.cfg.heteroscedastic_noise
        
        return Y_modified
    
    def next(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成下一批数据"""
        # 确定批量大小
        if batch_size is None:
            b_min, b_max = self.cfg.batch_size_range
            batch_size = torch.randint(b_min, b_max + 1, (1,), generator=self.rng).item()
        
        # 检查batch范围内的所有changepoints
        batch_start = self.t
        batch_end = self.t + batch_size
        
        changepoints_in_batch = []
        for cp in self.cfg.changepoints:
            if batch_start <= cp.time < batch_end and cp.time not in self.processed_changepoints:
                changepoints_in_batch.append(cp)
        
        # 如果有changepoints，分段生成数据
        if changepoints_in_batch:
            changepoints_in_batch.sort(key=lambda cp: cp.time)
            
            X_list = []
            Y_list = []
            current_t = self.t
            
            for cp in changepoints_in_batch:
                # 生成changepoint之前的数据
                if cp.time > current_t:
                    n_before = cp.time - current_t
                    X_before = self._sample_x(n_before)
                    eta_before = self._physical_system(X_before).squeeze(-1)
                    eps_before = torch.sqrt(torch.tensor(self.current_noise_variance)) * \
                                torch.randn(n_before, dtype=X_before.dtype, generator=self.rng)
                    Y_before = eta_before + eps_before
                    X_list.append(X_before)
                    Y_list.append(Y_before)
                    current_t = cp.time
                
                # 应用changepoint
                self._check_changepoints(cp.time)
                current_t = cp.time
            
            # 生成changepoint之后的数据
            if current_t < batch_end:
                n_after = batch_end - current_t
                X_after = self._sample_x(n_after)
                eta_after = self._physical_system(X_after).squeeze(-1)
                eps_after = torch.sqrt(torch.tensor(self.current_noise_variance)) * \
                           torch.randn(n_after, dtype=X_after.dtype, generator=self.rng)
                Y_after = eta_after + eps_after
                X_list.append(X_after)
                Y_list.append(Y_after)
            
            X = torch.cat(X_list, dim=0)
            Y = torch.cat(Y_list, dim=0)
            
        else:
            # 没有changepoints，正常生成
            self._check_changepoints(self.t)
            X = self._sample_x(batch_size)
            eta = self._physical_system(X).squeeze(-1)
            eps = torch.sqrt(torch.tensor(self.current_noise_variance)) * \
                  torch.randn(batch_size, dtype=X.dtype, generator=self.rng)
            Y = eta + eps
        
        # 注入额外的噪声
        Y = self._inject_additional_noise(X, Y)
        
        # 更新时间步
        self.t += batch_size
        
        return X, Y

class EnhancedSyntheticDataStream1:
    """
    Enhanced synthetic data stream supporting:
    - abrupt changepoints
    - continuous parameter drift within regimes
    """

    def __init__(self, cfg: EnhancedSyntheticConfig, seed: Optional[int] = None, output_a2: bool = False):
        self.cfg = cfg
        self.physical_cfg = cfg.physical_config

        self.output_a2 = output_a2

        self.rng = torch.Generator().manual_seed(seed or 0)

        # ---- time & state ----
        self.t = 0
        self.theta_current = self.physical_cfg.theta_optimal.clone()
        self.current_physical_system = self.physical_cfg.physical_system
        self.current_noise_variance = self.physical_cfg.noise_variance

        if self.physical_cfg.phys_param_dim > 0:
            self.phys_param_current = self.physical_cfg.phys_param_init.clone()
        else:
            self.phys_param_current = None

        # 当前 regime（由 changepoint 决定）
        self.current_regime = 0
        self.processed_changepoints = set()

        # fixed-x support
        if self.physical_cfg.sampling_strategy == "fixed" and self.physical_cfg.fixed_x_values:
            self.fixed_x_values = torch.tensor(
                self.physical_cfg.fixed_x_values, dtype=torch.float64
            )
        else:
            self.fixed_x_values = None

    # =========================================================
    #  NEW: continuous drift model for phys_param
    # =========================================================
    def _update_physical_param_drift(self, t: int, t1: int, t2: int):
        """
        Continuous drift of physical parameters within each regime.
        Hard-coded example for config2:
            regime 0: a2 from 7.5 -> 12
            regime 1: fixed at 5
            regime 2: a2 from 5 -> 7.5
        """
        if self.current_physical_system != "config2":
            return

        # phys_param_current = [a1, a2, a3]
        a1, a2, a3 = self.phys_param_current

        if self.current_regime == 0:
            # 7.5 -> 12 over t in [0, 1600)
            a2_new = 7.5 + (12.0 - 7.5) * min(t / t1, 1.0)

        elif self.current_regime == 1:
            # flat at 5
            a2_new = 5.0

        elif self.current_regime == 2:
            # 5 -> 7.5 over t in [2000, 4000)
            a2_new = 5.0 + (7.5 - 5.0) * min((t - t2) / t2, 1.0)

        else:
            a2_new = a2

        self.phys_param_current = torch.tensor(
            [a1, a2_new, a3],
            dtype=self.phys_param_current.dtype,
            device=self.phys_param_current.device,
        )
        return a2_new

    # =========================================================
    # computer / physical systems (unchanged)
    # =========================================================
    def _computer_model(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if self.physical_cfg.computer_model == "config2":
            return torch.sin(5 * theta[0] * x) + 5 * x
        raise ValueError

    def _physical_system(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_physical_system == "config2":
            a1, a2, a3 = self.phys_param_current
            return a1 * x * torch.cos(a2 * x) + a3 * x
        raise ValueError

    def _sample_x(self, n: int) -> torch.Tensor:
        return torch.rand(n, 1, dtype=torch.float64, generator=self.rng)

    # =========================================================
    #  Changepoint now switches REGIME, not just value
    # =========================================================
    def _check_changepoints(self, t_current: int):
        for cp in self.cfg.changepoints:
            if cp.time == t_current and cp.time not in self.processed_changepoints:
                print(f"\n🔄 [t={t_current}] Changepoint!")

                # 切换 regime
                self.current_regime += 1
                print(f"   Regime switched → {self.current_regime}")

                # 可选：hard reset φ
                if cp.phys_param_new is not None:
                    self.phys_param_current = cp.phys_param_new.clone()
                    print(f"   φ hard reset → {self.phys_param_current.cpu().numpy()}")

                self.processed_changepoints.add(cp.time)

    # =========================================================
    #  Main generator
    # =========================================================
    def next(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list, Y_list = [], []

        for _ in range(batch_size):
            # check cp
            self._check_changepoints(self.t)

            # update continuous drift
            drift_start, drift_end = self.cfg.changepoints[0].time, self.cfg.changepoints[1].time
            a2_new = self._update_physical_param_drift(self.t, drift_start, drift_end)

            # sample
            x = self._sample_x(1)
            eta = self._physical_system(x).squeeze()
            eps = torch.sqrt(torch.tensor(self.current_noise_variance)) * \
                  torch.randn((), generator=self.rng)

            X_list.append(x)
            Y_list.append(eta + eps)

            self.t += 1

        X = torch.cat(X_list, dim=0)
        Y = torch.stack(Y_list)

        if self.output_a2:
            return X, Y, a2_new
        else:
            return X, Y


# 预定义的配置工厂函数
def create_config1_config(
    theta_optimal: torch.Tensor = torch.tensor([0.2, 0.3], dtype=torch.float64),
    noise_variance: float = 0.04,
    n_observations: int = 50,
    batch_size_range: Tuple[int, int] = (5, 10),
    **kwargs
) -> EnhancedSyntheticConfig:
    physical_cfg = PhysicalSystemConfig(
        name="config1",
        computer_model="config1",
        theta_space=(0.0, 0.25),
        theta_dim=2,
        physical_system="config1",
        theta_optimal=theta_optimal,
        noise_variance=noise_variance,
        n_observations=n_observations,
        sampling_strategy="uniform",
        # φ = θ0（与 computer model 一致）
        phys_param_dim=2,
        phys_param_init=theta_optimal.clone(),
    )
    delta_kernel = DeltaKernelConfig(name="rbf", lengthscale=1.0, variance=0.01)
    cfg = EnhancedSyntheticConfig(
        physical_config=physical_cfg,
        delta_kernel=delta_kernel,
        **kwargs
    )
    cfg.batch_size_range = batch_size_range
    return cfg


def create_config2_config(
    theta_optimal: torch.Tensor = torch.tensor([1.8771], dtype=torch.float64),
    noise_variance: float = 0.04,
    n_observations: int = 30,
    batch_size_range: Tuple[int, int] = (5, 10),
    **kwargs
) -> EnhancedSyntheticConfig:
    physical_cfg = PhysicalSystemConfig(
        name="config2",
        computer_model="config2",
        theta_space=(0.0, 3.0),
        theta_dim=1,
        physical_system="config2",
        theta_optimal=theta_optimal,
        noise_variance=noise_variance,
        n_observations=n_observations,
        # sampling_strategy="equidistant",
        sampling_strategy="lhs",
        # sampling_strategy="uniform",
        # φ = [a1, a2, a3]，默认 [5, 15/2, 5]
        phys_param_dim=3,
        phys_param_init=torch.tensor([5.0, 7.5, 5.0], dtype=torch.float64),
    )
    delta_kernel = DeltaKernelConfig(name="rbf", lengthscale=1.0, variance=0.01)
    cfg = EnhancedSyntheticConfig(
        physical_config=physical_cfg,
        delta_kernel=delta_kernel,
        **kwargs
    )
    cfg.batch_size_range = batch_size_range
    return cfg


def create_config3_config(
    theta_optimal: torch.Tensor = torch.tensor([3.5609], dtype=torch.float64),
    noise_variance: float = 0.0004,
    fixed_x_values: Optional[List[float]] = None,
    batch_size_range: Tuple[int, int] = (5, 10),
    **kwargs
) -> EnhancedSyntheticConfig:
    if fixed_x_values is None:
        fixed_x_values = [i * 0.05 for i in range(17)]  # 0,0.05,...,0.8
    physical_cfg = PhysicalSystemConfig(
        name="config3",
        computer_model="config3",
        theta_space=(2.0, 4.0),
        theta_dim=1,
        physical_system="config3",
        theta_optimal=theta_optimal,
        noise_variance=noise_variance,
        n_observations=len(fixed_x_values),
        sampling_strategy="fixed",
        fixed_x_values=fixed_x_values,
        # φ = [a, b]，默认 [4, 5]
        phys_param_dim=2,
        phys_param_init=torch.tensor([4.0, 5.0], dtype=torch.float64),
    )
    delta_kernel = DeltaKernelConfig(name="rbf", lengthscale=1.0, variance=0.01)
    cfg = EnhancedSyntheticConfig(
        physical_config=physical_cfg,
        delta_kernel=delta_kernel,
        **kwargs
    )
    cfg.batch_size_range = batch_size_range
    return cfg


# 使用示例
def example_usage():
    """使用示例"""
    # 创建配置1的数据流
    cfg = create_config2_config()
    cfg.changepoints = [
        # t=120 时改变物理系统参数（把频率从 7.5 调到 6.0，线性项放大）
        EnhancedChangepointConfig(
            time=120,
            phys_param_new=torch.tensor([6.0, 6.0, 6.0], dtype=torch.float64)
        ),
        # t=240 时再改变频率
        EnhancedChangepointConfig(
            time=240,
            phys_param_new=torch.tensor([5.0, 9.0, 5.0], dtype=torch.float64)
        ),
    ]
    stream = EnhancedSyntheticDataStream(cfg, seed=123)
    
    # 生成数据
    for i in range(5):
        X, Y = stream.next()
        print(f"Batch {i}: X shape {X.shape}, Y shape {Y.shape}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
        print()