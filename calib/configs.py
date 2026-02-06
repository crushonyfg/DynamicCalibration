# =============================================================
# file: calib/configs.py
# =============================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Tuple, List, Protocol, Union, Sequence
import math
import torch


# ------------------- Configs -------------------
@dataclass
class DeltaKernelConfig:
    name: str = "rbf"  # or "matern52"
    lengthscale: Union[float, Sequence[float], torch.Tensor] = 1.0
    variance: float = 0.1
    noise: float = 1e-6  # nugget for numerical stability

'''
cfg1 = DeltaKernelConfig(name="rbf", lengthscale=1.0)  
# single scalar lengthscale

cfg2 = DeltaKernelConfig(name="rbf", lengthscale=[0.5, 1.0, 2.0])  
# per-dimension lengthscales

cfg3 = DeltaKernelConfig(name="matern52", lengthscale=torch.tensor([0.3, 0.7]))  
# tensor lengthscales
'''


@dataclass
class PFConfig:
    num_particles: int = 1024
    resample_ess_ratio: float = 0.5
    resample_scheme: str = "systematic"  # or "stratified", "multinomial"
    # move_strategy: str = "pmcmc"  # or "random_walk","liu_west", "laplace", "pmcmc", "none"
    move_strategy: str = "random_walk"
    # Liu–West hyperparams
    liu_west_a: float = 0.90
    liu_west_h2: Optional[float] = None  # if None, derive from a via h^2 = 1-a^2
    # Random-walk and Laplace proposal scales
    random_walk_scale: float = 0.1
    laplace_alpha: float = 0.05
    laplace_beta: float = 1e-3
    laplace_eta: float = 0.01
    pmcmc_steps: int = 2


# @dataclass
# class BOCPDConfig:
#     def default_hazard(r: torch.Tensor) -> torch.Tensor:
#         """
#         Geometric hazard: h(r) = 1 / (λ + r)
#         期望 run-length = λ
#         """
#         lam = 100.0  # 期望 100 步发生一次变点
#         return 1.0 / (lam + r)
#     # hazard: Callable[[torch.Tensor], torch.Tensor] = lambda r: torch.full_like(r, 0.01, dtype=torch.float64)  # h(r)
#     hazard: Callable[[torch.Tensor], torch.Tensor] = default_hazard
#     max_experts: int = 5  # keep top-k experts
#     max_run_length: int = 512  # truncation for run-length posterior (advisory)
#     restart_threshold: float = 0.8  # if P(CP) > threshold, trigger reset policy
#     log_space: bool = True
#     delta_refit_every: int = 0  # 0 means never
#     delta_refit_topk: int = 1  # 0 means no refit
#     use_restart: bool = True

# @dataclass
# class BOCPDConfig:
#     def default_hazard(r: torch.Tensor) -> torch.Tensor:
#         """
#         Geometric hazard: h(r) = 1 / (λ + r)
#         期望 run-length = λ
#         """
#         lam = 100.0  # 期望 100 步发生一次变点
#         return 1.0 / (lam + r)
    
#     # ✅ 新增：选择BOCPD模式
#     bocpd_mode: str = "standard"  # "standard" 或 "restart"
    
#     hazard: Callable[[torch.Tensor], torch.Tensor] = default_hazard
#     max_experts: int = 10  # keep top-k experts
#     max_run_length: int = 512  # truncation for run-length posterior (advisory)
    
#     # ✅ Standard BOCPD 相关配置
#     use_restart: bool = False  # 仅用于 standard mode
#     restart_threshold: float = 0.8  # 仅用于 standard mode
#     restart_small_r: int = 5  # 仅用于 standard mode
    
#     # ✅ R-BOCPD (restart_bocpd.py) 相关配置
#     use_backdated_restart: bool = False  # False=Algorithm-2, True=Backdated
#     restart_margin: float = 0.05  # 稳定性margin，防止频繁restart
#     restart_cooldown: int = 10  # restart后的冷却期（步数）
    
#     log_space: bool = True
#     delta_refit_every: int = 1  # 0 means never
#     delta_refit_topk: int = 11  # 0 means no refit

@dataclass
class BOCPDConfig:
    """
    Configuration for Bayesian Online Change Point Detection (BOCPD)
    """
    # ==== 核心参数 ====
    hazard_lambda: float = 200   # <--- 新增：可调 λ 值
    hazard_type: str = "geometric"  # <--- 可选类型：constant, linear, weibull 等

    def hazard(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute hazard function value h(r)
        """
        lam = self.hazard_lambda
        if self.hazard_type == "constant":
            return torch.full_like(r, 1.0 / lam)
        elif self.hazard_type == "linear":
            return torch.clamp(r / lam, max=1.0)
        elif self.hazard_type == "weibull":
            k = 1.5  # shape parameter
            return (k / lam) * ((r / lam) ** (k - 1))
        elif self.hazard_type == "geometric":
            return 1.0 / (lam + r)
        else:
            raise ValueError(f"Unknown hazard_type: {self.hazard_type}")

    # ==== 其他参数 ====
    bocpd_mode: str = "standard"
    max_experts: int = 5
    max_run_length: int = 1000000
    use_restart: bool = False
    restart_threshold: float = 0.85
    restart_small_r: int = 5
    use_backdated_restart: bool = False
    restart_margin: float = 1
    restart_cooldown: int = 10
    log_space: bool = True
    delta_refit_every: int = 1
    delta_refit_topk: int = 11
    use_restart: bool = True

    restart_criteria: str = "rank_change" # "theta_test" or "rank_change"
    restart_theta_test: str = "energy" # or "credible" or "sw" or "energy"
    restart_cred_z: float = 2.0
    restart_cred_frac: float = 0.5
    restart_sw_proj: int = 32
    restart_theta_tau: float = 0.5





@dataclass
class ModelConfig:
    rho: float = 1.0
    sigma_eps: float = 0.05
    delta_kernel: DeltaKernelConfig = field(default_factory=lambda: DeltaKernelConfig(
        name="rbf",
        lengthscale=1.0,
        variance=0.01,  # ✅ 设置为0，禁用delta
        # noise=1e-6,
        noise=1e-6
    ))
    emulator_type: str = "deterministic"  # or "gp"
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    use_discrepancy: bool = True


@dataclass
class CalibrationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pf: PFConfig = field(default_factory=PFConfig)
    bocpd: BOCPDConfig = field(default_factory=BOCPDConfig)