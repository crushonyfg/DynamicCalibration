import math
import torch
from dataclasses import dataclass

# # ===============================
# # 1. Gu & Wang (2017) models
# # ===============================

# def computer_model(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
#     """
#     y^s(x, theta) = sin(5 theta x) + 5x
#     x:     [B]
#     theta: [N]
#     return: [N,B]
#     """
#     return torch.sin(5.0 * theta[:, None] * x[None, :]) + 5.0 * x[None, :]


# def physical_model(x: torch.Tensor) -> torch.Tensor:
#     """
#     eta_0(x) = 5x cos(15x/2) + 5x
#     """
#     return 5.0 * x * torch.cos(15.0 * x / 2.0) + 5.0 * x


# # ===============================
# # 2. Utilities
# # ===============================

# def normalize_logw(logw):
#     m = logw.max()
#     w = torch.exp(logw - m)
#     return w / w.sum()


# def ess(w):
#     return float(1.0 / (w * w).sum())


# def gini(w):
#     w = w / w.sum()
#     ws = torch.sort(w).values
#     n = w.numel()
#     idx = torch.arange(1, n + 1, device=w.device)
#     return float((2 * (idx * ws).sum()) / n - (n + 1)) / n


# def systematic_resample(w, gen):
#     N = w.numel()
#     u0 = torch.rand((), generator=gen) / N
#     u = u0 + torch.arange(N) / N
#     cdf = torch.cumsum(w, 0)
#     return torch.searchsorted(cdf, u)


# # ===============================
# # 3. RBF kernel (exact GP, small n)
# # ===============================

# def rbf_kernel(x1, x2, ell, sigma_f):
#     x1 = x1[:, None]
#     x2 = x2[None, :]
#     return sigma_f**2 * torch.exp(-0.5 * (x1 - x2) ** 2 / ell**2)


# # ===============================
# # 4. PF Config
# # ===============================

# @dataclass
# class PFConfig:
#     N: int = 1024
#     obs_sigma: float = 0.2
#     resample_frac: float = 0.5
#     theta_rw: float = 0.02
#     logell_rw: float = 0.05
#     logsig_rw: float = 0.05
#     device: str = "cpu"


# # ===============================
# # 5. PF-θ (δ marginalized, fixed hyper)
# # ===============================

# class PFTheta:
#     def __init__(self, cfg: PFConfig, ell=0.2, sigma_f=1.0):
#         self.cfg = cfg
#         self.theta = torch.rand(cfg.N) * 3.0
#         self.logw = torch.zeros(cfg.N)
#         self.w = torch.ones(cfg.N) / cfg.N
#         self.ell = ell
#         self.sigma_f = sigma_f
#         self.gen = torch.Generator().manual_seed(0)

#     def step(self, x, y):
#         self.theta += self.cfg.theta_rw * torch.randn_like(self.theta)

#         y_s = computer_model(x, self.theta)
#         r = y[None, :] - y_s

#         K = rbf_kernel(x, x, self.ell, self.sigma_f)
#         Ky = K + (self.cfg.obs_sigma**2) * torch.eye(len(x))
#         L = torch.linalg.cholesky(Ky)
#         alpha = torch.cholesky_solve(r.T, L)
#         ll = -0.5 * (r * alpha.T).sum(1)

#         self.logw += ll
#         self.w = normalize_logw(self.logw)

#         if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
#             idx = systematic_resample(self.w, self.gen)
#             self.theta = self.theta[idx]
#             self.logw.zero_()
#             self.w.fill_(1 / self.cfg.N)

#         return ess(self.w), gini(self.w)

#     def mean_theta(self):
#         return (self.w * self.theta).sum()


# # ===============================
# # 6. PF-θ + shared GP
# # ===============================

# # class PFThetaSharedGP(PFTheta):
# #     def __init__(self, cfg: PFConfig, ell=0.2, sigma_f=1.0):
# #         super().__init__(cfg, ell, sigma_f)
# #         self.delta_mean = torch.zeros(0)
# #         self.Kinv = None

# #     def step(self, x, y):
# #         self.theta += self.cfg.theta_rw * torch.randn_like(self.theta)
# #         y_s = computer_model(x, self.theta)
# #         r = y[None, :] - y_s
# #         r_bar = (self.w[:, None] * r).sum(0)

# #         K = rbf_kernel(x, x, self.ell, self.sigma_f)
# #         Ky = K + self.cfg.obs_sigma**2 * torch.eye(len(x))
# #         self.Kinv = torch.inverse(Ky)
# #         self.delta_mean = self.Kinv @ r_bar

# #         resid = r - self.delta_mean[None, :]
# #         ll = -0.5 * (resid @ self.Kinv * resid).sum(1)

# #         self.logw += ll
# #         self.w = normalize_logw(self.logw)

# #         if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
# #             idx = systematic_resample(self.w, self.gen)
# #             self.theta = self.theta[idx]
# #             self.logw.zero_()
# #             self.w.fill_(1 / self.cfg.N)

# #         return ess(self.w), gini(self.w)
# class PFThetaSharedGP(PFTheta):
#     def __init__(self, cfg: PFConfig, ell=0.2, sigma_f=1.0):
#         super().__init__(cfg, ell, sigma_f)

#         # shared GP posterior state
#         self.delta_mean = None   # [B] after first update
#         self.Kinv = None

#     def step(self, x, y):
#         """
#         Correct ordering:
#           1) propagate theta
#           2) compute likelihood using *old* shared delta
#           3) update weights (+ resample)
#           4) update shared delta using NEW weights
#         """

#         # -----------------------------------
#         # 1. propagate theta (random walk)
#         # -----------------------------------
#         self.theta += self.cfg.theta_rw * torch.randn_like(self.theta)

#         # -----------------------------------
#         # 2. likelihood with CURRENT delta
#         # -----------------------------------
#         y_s = computer_model(x, self.theta)        # [N,B]
#         r = y[None, :] - y_s                       # [N,B]

#         # build kernel once per batch
#         K = rbf_kernel(x, x, self.ell, self.sigma_f)
#         Ky = K + self.cfg.obs_sigma**2 * torch.eye(len(x))
#         self.Kinv = torch.inverse(Ky)

#         if self.delta_mean is None:
#             # before first update: delta ≡ 0
#             resid = r
#         else:
#             resid = r - self.delta_mean[None, :]  # [N,B]

#         # Gaussian log-likelihood per particle
#         # r_i ~ N(delta_mean, Ky)
#         ll = -0.5 * (resid @ self.Kinv * resid).sum(dim=1)

#         # weight update
#         self.logw += ll
#         self.w = normalize_logw(self.logw)

#         # -----------------------------------
#         # 3. resample if needed
#         # -----------------------------------
#         if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
#             idx = systematic_resample(self.w, self.gen)
#             self.theta = self.theta[idx]
#             self.logw.zero_()
#             self.w.fill_(1.0 / self.cfg.N)

#         # -----------------------------------
#         # 4. update shared discrepancy δ
#         #    using NEW weights
#         # -----------------------------------
#         # particle-weighted residual mean
#         r_bar = (self.w[:, None] * r).sum(dim=0)   # [B]

#         # GP posterior mean: δ = K (K+σ²I)^{-1} r̄
#         self.delta_mean = self.Kinv @ r_bar        # [B]

#         return ess(self.w), gini(self.w)



# # ===============================
# # 7. PF-(θ, GP hypers)
# # ===============================

# class PFThetaGPHyper:
#     def __init__(self, cfg: PFConfig):
#         self.cfg = cfg
#         self.theta = torch.rand(cfg.N) * 3.0
#         self.logell = torch.randn(cfg.N) - 1.5
#         self.logsig = torch.zeros(cfg.N)
#         self.logw = torch.zeros(cfg.N)
#         self.w = torch.ones(cfg.N) / cfg.N
#         self.gen = torch.Generator().manual_seed(1)

#     def step(self, x, y):
#         self.theta += self.cfg.theta_rw * torch.randn_like(self.theta)
#         self.logell += self.cfg.logell_rw * torch.randn_like(self.logell)
#         self.logsig += self.cfg.logsig_rw * torch.randn_like(self.logsig)

#         ll = torch.zeros_like(self.theta)
#         y_s = computer_model(x, self.theta)
#         r = y[None, :] - y_s

#         for i in range(self.cfg.N):
#             ell = torch.exp(self.logell[i])
#             sf = torch.exp(self.logsig[i])
#             K = rbf_kernel(x, x, ell, sf)
#             Ky = K + self.cfg.obs_sigma**2 * torch.eye(len(x))
#             L = torch.linalg.cholesky(Ky)
#             alpha = torch.cholesky_solve(r[i][:, None], L)
#             ll[i] = -0.5 * (r[i] * alpha[:, 0]).sum()

#         self.logw += ll
#         self.w = normalize_logw(self.logw)

#         if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
#             idx = systematic_resample(self.w, self.gen)
#             self.theta = self.theta[idx]
#             self.logell = self.logell[idx]
#             self.logsig = self.logsig[idx]
#             self.logw.zero_()
#             self.w.fill_(1 / self.cfg.N)

#         return ess(self.w), gini(self.w)

#     def mean_theta(self):
#         return (self.w * self.theta).sum()

# class PFThetaNoDiscrepancy:
#     """
#     PF over theta ONLY.
#     No discrepancy, no GP.
#     Likelihood: y ~ N(y^s(x,theta), sigma^2 I)
#     """

#     def __init__(self, cfg: PFConfig):
#         self.cfg = cfg
#         self.theta = torch.rand(cfg.N) * 3.0
#         self.logw = torch.zeros(cfg.N)
#         self.w = torch.ones(cfg.N) / cfg.N
#         self.gen = torch.Generator().manual_seed(0)

#     def step(self, x, y):
#         # -------------------------
#         # 1. propagate theta
#         # -------------------------
#         self.theta += self.cfg.theta_rw * torch.randn_like(self.theta)

#         # -------------------------
#         # 2. model prediction
#         # -------------------------
#         y_s = computer_model(x, self.theta)      # [N,B]
#         r = y[None, :] - y_s                     # [N,B]

#         # -------------------------
#         # 3. Gaussian likelihood
#         # -------------------------
#         sigma2 = self.cfg.obs_sigma ** 2
#         ll = -0.5 * (r * r).sum(dim=1) / sigma2
#         # constant terms omitted (same for all particles)

#         # -------------------------
#         # 4. weight update
#         # -------------------------
#         self.logw += ll
#         self.w = normalize_logw(self.logw)

#         # -------------------------
#         # 5. resample if needed
#         # -------------------------
#         if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
#             idx = systematic_resample(self.w, self.gen)
#             self.theta = self.theta[idx]
#             self.logw.zero_()
#             self.w.fill_(1.0 / self.cfg.N)

#         return ess(self.w), gini(self.w)

#     def mean_theta(self):
#         return (self.w * self.theta).sum()



# # ===============================
# # 8. Run experiment
# # ===============================

# # def main():
# #     torch.set_default_dtype(torch.float64)
# #     cfg = PFConfig(N=512)

# #     x = torch.linspace(0, 1, 30)
# #     y = physical_model(x) + cfg.obs_sigma * torch.randn_like(x)

# #     pf1 = PFThetaNoDiscrepancy(cfg)
# #     pf2 = PFTheta(cfg)
# #     # pf2 = PFThetaSharedGP(cfg)
# #     pf3 = PFThetaGPHyper(cfg)

# #     for t in range(30):
# #         e1, g1 = pf1.step(x, y)
# #         e2, g2 = pf2.step(x, y)
# #         e3, g3 = pf3.step(x, y)

# #         print(
# #             f"[iter {t:02d}] "
# #             f"θ̂: {pf1.mean_theta():.3f} | {pf2.mean_theta():.3f} | {pf3.mean_theta():.3f}   "
# #             f"ESS: {e1:.1f},{e2:.1f},{e3:.1f}"
# #         )

# import matplotlib.pyplot as plt
# import torch

# def run_and_record(pf1, pf2, pf3, x, y, n_iter):
#     theta_hist_1 = []
#     theta_hist_2 = []
#     theta_hist_3 = []

#     ess_hist_1 = []
#     ess_hist_2 = []
#     ess_hist_3 = []

#     for t in range(n_iter):
#         ess1, _ = pf1.step(x, y)
#         ess2, _ = pf2.step(x, y)
#         ess3, _ = pf3.step(x, y)

#         theta_hist_1.append(pf1.mean_theta().item())
#         theta_hist_2.append(pf2.mean_theta().item())
#         theta_hist_3.append(pf3.mean_theta().item())

#         ess_hist_1.append(ess1)
#         ess_hist_2.append(ess2)
#         ess_hist_3.append(ess3)

#         print(
#             f"[iter {t:02d}] "
#             f"theta: {theta_hist_1[-1]:.3f}, "
#             f"{theta_hist_2[-1]:.3f}, "
#             f"{theta_hist_3[-1]:.3f} | "
#             f"ESS: {ess1:.1f}, {ess2:.1f}, {ess3:.1f}"
#         )

#     return (
#         torch.tensor(theta_hist_1),
#         torch.tensor(theta_hist_2),
#         torch.tensor(theta_hist_3),
#         torch.tensor(ess_hist_1),
#         torch.tensor(ess_hist_2),
#         torch.tensor(ess_hist_3),
#     )
# def plot_theta_traj(theta1, theta2, theta3, theta_star=None, save_path="theta_traj.png"):
#     plt.figure(figsize=(7, 4))

#     plt.plot(theta1, label="PF-θ (no discrepancy)")
#     plt.plot(theta2, label="PF-θ + shared GP")
#     plt.plot(theta3, label="PF-(θ, GP hypers)")

#     if theta_star is not None:
#         plt.axhline(theta_star, linestyle="--", color="k", label=r"$\theta_0^\star$")

#     plt.xlabel("Iteration")
#     plt.ylabel(r"$\hat{\theta}$")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200)
#     plt.close()

# def plot_ess_traj(ess1, ess2, ess3, N, save_path="ess_traj.png"):
#     plt.figure(figsize=(7, 4))

#     plt.plot(ess1, label="PF-θ (no discrepancy)")
#     plt.plot(ess2, label="PF-θ + shared GP")
#     plt.plot(ess3, label="PF-(θ, GP hypers)")

#     plt.axhline(0.5 * N, linestyle="--", color="gray", label="0.5 N")

#     plt.xlabel("Iteration")
#     plt.ylabel("ESS")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=200)
#     plt.close()
# def main():
#     torch.set_default_dtype(torch.float64)

#     cfg = PFConfig(N=512)
#     x = torch.linspace(0, 1, 30)
#     y = physical_model(x) + cfg.obs_sigma * torch.randn_like(x)

#     pf1 = PFThetaNoDiscrepancy(cfg)
#     pf2 = PFTheta(cfg)
#     pf3 = PFThetaGPHyper(cfg)

#     (
#         th1, th2, th3,
#         ess1, ess2, ess3
#     ) = run_and_record(pf1, pf2, pf3, x, y, n_iter=40)

#     plot_theta_traj(
#         th1, th2, th3,
#         theta_star=1.8771,
#         save_path="theta_traj.png"
#     )

#     plot_ess_traj(
#         ess1, ess2, ess3,
#         N=cfg.N,
#         save_path="ess_traj.png"
#     )


# if __name__ == "__main__":
#     main()
import math
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt

# ===============================
# 1. Gu & Wang (2017) models (Config 2)
# ===============================

def computer_model(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    y^s(x, theta) = sin(5 theta x) + 5x
    x:     [B] or [T]
    theta: [N]
    return: [N,B]
    """
    return torch.sin(5.0 * theta[:, None] * x[None, :]) + 5.0 * x[None, :]

def physical_model(x: torch.Tensor) -> torch.Tensor:
    """
    eta_0(x) = 5x cos(15x/2) + 5x
    x: [T]
    """
    return 5.0 * x * torch.cos(15.0 * x / 2.0) + 5.0 * x


# ===============================
# 2. Utilities
# ===============================

def normalize_logw(logw: torch.Tensor) -> torch.Tensor:
    m = logw.max()
    w = torch.exp(logw - m)
    return w / (w.sum() + 1e-30)

def ess(w: torch.Tensor) -> float:
    return float(1.0 / (w * w).sum().clamp_min(1e-30))

def gini(w: torch.Tensor) -> float:
    w = (w / (w.sum() + 1e-30)).clamp_min(0)
    ws = torch.sort(w).values
    n = w.numel()
    idx = torch.arange(1, n + 1, device=w.device, dtype=w.dtype)
    # G = (2*sum(i*wi))/n - (n+1) all divided by n (since sum(w)=1)
    g = ((2 * (idx * ws).sum()) / n - (n + 1)) / n
    return float(g.clamp(0, 1))

def systematic_resample(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    N = w.numel()
    u0 = torch.rand((), generator=gen, device=w.device) / N
    u = u0 + torch.arange(N, device=w.device, dtype=torch.float64) / N
    cdf = torch.cumsum(w, 0)
    return torch.searchsorted(cdf, u).clamp(max=N - 1)


# ===============================
# 3. GP kernel + GP helpers
# ===============================

def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, ell: torch.Tensor, sigma_f: torch.Tensor) -> torch.Tensor:
    """
    x1: [n1], x2: [n2]
    returns [n1,n2]
    """
    x1 = x1[:, None]
    x2 = x2[None, :]
    return sigma_f**2 * torch.exp(-0.5 * (x1 - x2) ** 2 / (ell**2 + 1e-30))

def gp_posterior_from_data(X: torch.Tensor, y: torch.Tensor, ell: float, sigma_f: float, obs_sigma: float):
    """
    Build GP posterior for f~GP(0,K), y = f + eps, eps~N(0,obs_sigma^2).
    Returns:
      Kinv = (K + obs^2 I)^(-1)
      alpha = Kinv @ y
    """
    device = X.device
    ell_t = torch.tensor(float(ell), device=device, dtype=X.dtype)
    sf_t = torch.tensor(float(sigma_f), device=device, dtype=X.dtype)

    K = rbf_kernel(X, X, ell_t, sf_t)
    Ky = K + (obs_sigma**2) * torch.eye(X.numel(), device=device, dtype=X.dtype)
    Kinv = torch.inverse(Ky)
    alpha = Kinv @ y
    return Kinv, alpha, ell_t, sf_t, Ky

def gp_predictive(X_train: torch.Tensor, Kinv: torch.Tensor, y_train: torch.Tensor,
                  X_star: torch.Tensor, ell: float, sigma_f: float, obs_sigma: float):
    """
    Predictive for noisy observation r* at X_star:
      r* | data ~ N(mean, cov)
    where cov includes observation noise obs_sigma^2 I.
    """
    device = X_train.device
    dtype = X_train.dtype
    ell_t = torch.tensor(float(ell), device=device, dtype=dtype)
    sf_t = torch.tensor(float(sigma_f), device=device, dtype=dtype)

    Kxs = rbf_kernel(X_star, X_train, ell_t, sf_t)          # [m,n]
    Kss = rbf_kernel(X_star, X_star, ell_t, sf_t)           # [m,m]
    mean = Kxs @ (Kinv @ y_train)                           # [m]
    cov_f = Kss - Kxs @ Kinv @ Kxs.T                        # [m,m]
    cov_y = cov_f + (obs_sigma**2) * torch.eye(X_star.numel(), device=device, dtype=dtype)
    return mean, cov_y

def mvn_logpdf_zero_mean(resid: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """
    resid: [m]
    cov: [m,m] SPD
    returns scalar log N(resid | 0, cov)
    """
    L = torch.linalg.cholesky(cov)
    alpha = torch.cholesky_solve(resid[:, None], L)[:, 0]
    quad = (resid * alpha).sum()
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    m = resid.numel()
    return -0.5 * (quad + logdet + m * math.log(2 * math.pi))

def mvn_logpdf(resid: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    return mvn_logpdf_zero_mean(resid - mean, cov)


# ===============================
# 4. PF Config
# ===============================

@dataclass
class PFConfig:
    N: int = 512
    obs_sigma: float = 0.2
    resample_frac: float = 0.5
    theta_rw: float = 0.02
    logell_rw: float = 0.05
    logsig_rw: float = 0.05
    device: str = "cpu"


# ===============================
# 5. PF-θ (no discrepancy)
# ===============================

class PFThetaNoDiscrepancy:
    """
    Particles = theta only.
    No discrepancy.
    Likelihood increment for batch:
      y | theta ~ N(y^s(x,theta), obs_sigma^2 I)
    """
    def __init__(self, cfg: PFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.theta = torch.rand(cfg.N, device=self.device, dtype=torch.float64) * 3.0
        self.logw = torch.zeros(cfg.N, device=self.device, dtype=torch.float64)
        self.w = torch.ones(cfg.N, device=self.device, dtype=torch.float64) / cfg.N
        self.gen = torch.Generator(device=self.device).manual_seed(0)

    def step(self, x: torch.Tensor, y: torch.Tensor):
        # 1) propagate
        self.theta = self.theta + self.cfg.theta_rw * torch.randn_like(self.theta)

        # 2) likelihood increment on new batch
        y_s = computer_model(x, self.theta)          # [N,B]
        r = y[None, :] - y_s                         # [N,B]
        sigma2 = self.cfg.obs_sigma ** 2
        ll = -0.5 * (r * r).sum(dim=1) / sigma2      # constants omitted (same across particles)

        # 3) weight update
        self.logw = self.logw + ll
        self.w = normalize_logw(self.logw)

        # 4) resample
        if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
            idx = systematic_resample(self.w, self.gen)
            self.theta = self.theta[idx]
            self.logw.zero_()
            self.w.fill_(1.0 / self.cfg.N)

        return ess(self.w), gini(self.w)

    def mean_theta(self):
        return (self.w * self.theta).sum()


# ===============================
# 6. PF-θ + shared GP discrepancy (accumulate history)
# ===============================

class PFThetaSharedGP:
    """
    Particles = theta.
    Discrepancy delta is ONE shared GP across all particles.
    We maintain a GP posterior over delta using historical weighted residual means.
    SMC ordering per step:
      - propagate theta
      - weight update using predictive distribution of delta at current x given past
      - resample
      - update shared GP posterior by appending weighted residual mean of current batch
    """
    def __init__(self, cfg: PFConfig, ell=0.2, sigma_f=1.0):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.theta = torch.rand(cfg.N, device=self.device, dtype=torch.float64) * 3.0
        self.logw = torch.zeros(cfg.N, device=self.device, dtype=torch.float64)
        self.w = torch.ones(cfg.N, device=self.device, dtype=torch.float64) / cfg.N
        self.gen = torch.Generator(device=self.device).manual_seed(777)

        self.ell = float(ell)
        self.sigma_f = float(sigma_f)

        # history for shared GP fit: X_hist, rbar_hist
        self.X_hist = torch.empty((0,), device=self.device, dtype=torch.float64)
        self.rbar_hist = torch.empty((0,), device=self.device, dtype=torch.float64)

        # cached posterior inverse on history
        self.Kinv = None

    def _build_posterior(self):
        if self.X_hist.numel() == 0:
            self.Kinv = None
            return
        self.Kinv, _, _, _, _ = gp_posterior_from_data(
            self.X_hist, self.rbar_hist, self.ell, self.sigma_f, self.cfg.obs_sigma
        )

    def step(self, x: torch.Tensor, y: torch.Tensor):
        # 1) propagate theta
        self.theta = self.theta + self.cfg.theta_rw * torch.randn_like(self.theta)

        # 2) compute residuals for new batch
        y_s = computer_model(x, self.theta)      # [N,B]
        r = y[None, :] - y_s                     # [N,B]

        # 3) weight update using predictive distribution of delta(x) given HISTORY
        # If no history, delta(x) ~ N(0, K(x,x)) and observation adds obs_sigma^2 I.
        B = x.numel()
        ll = torch.empty((self.cfg.N,), device=self.device, dtype=torch.float64)

        if self.X_hist.numel() == 0:
            # prior predictive: delta(x) ~ N(0, Kss), r = delta + eps => cov = Kss + obs^2 I
            Kss = rbf_kernel(x, x, torch.tensor(self.ell, device=self.device, dtype=torch.float64),
                             torch.tensor(self.sigma_f, device=self.device, dtype=torch.float64))
            cov = Kss + (self.cfg.obs_sigma**2) * torch.eye(B, device=self.device, dtype=torch.float64)
            # mean = 0
            # per particle: r_i ~ N(0,cov)
            for i in range(self.cfg.N):
                ll[i] = mvn_logpdf_zero_mean(r[i], cov)
        else:
            # posterior predictive at x
            mean_d, cov_y = gp_predictive(self.X_hist, self.Kinv, self.rbar_hist, x,
                                          self.ell, self.sigma_f, self.cfg.obs_sigma)
            for i in range(self.cfg.N):
                ll[i] = mvn_logpdf(r[i], mean_d, cov_y)

        self.logw = self.logw + ll
        self.w = normalize_logw(self.logw)

        # 4) resample
        if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
            idx = systematic_resample(self.w, self.gen)
            self.theta = self.theta[idx]
            self.logw.zero_()
            self.w.fill_(1.0 / self.cfg.N)

        # 5) update shared GP posterior using NEW weights (append weighted residual mean)
        r_bar = (self.w[:, None] * r).sum(dim=0)     # [B]
        self.X_hist = torch.cat([self.X_hist, x])
        self.rbar_hist = torch.cat([self.rbar_hist, r_bar])
        self._build_posterior()

        return ess(self.w), gini(self.w)

    def mean_theta(self):
        return (self.w * self.theta).sum()


# ===============================
# 7. PF-(θ, GP hypers) with history accumulation (no double count)
# ===============================

class PFThetaGPHyper:
    """
    Particles = (theta, logell, logsig).
    Discrepancy is GP with those hypers, integrated out.
    MUST use history. To avoid double counting, we compute:
      delta_ll = full_marginal_ll(t) - full_marginal_ll(t-1)
    and add delta_ll to logw.
    """
    def __init__(self, cfg: PFConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.theta = torch.rand(cfg.N, device=self.device, dtype=torch.float64) * 3.0
        self.logell = (torch.randn(cfg.N, device=self.device, dtype=torch.float64) - 1.5)
        self.logsig = torch.zeros(cfg.N, device=self.device, dtype=torch.float64)

        self.logw = torch.zeros(cfg.N, device=self.device, dtype=torch.float64)
        self.w = torch.ones(cfg.N, device=self.device, dtype=torch.float64) / cfg.N

        self.gen = torch.Generator(device=self.device).manual_seed(1)

        # global history of observations (x,y)
        self.X_hist = torch.empty((0,), device=self.device, dtype=torch.float64)
        self.y_hist = torch.empty((0,), device=self.device, dtype=torch.float64)

        # store previous full marginal ll per particle to do differencing
        self.prev_full_ll = torch.zeros(cfg.N, device=self.device, dtype=torch.float64)

    def _full_marginal_ll(self, X: torch.Tensor, y: torch.Tensor, theta_i: float, ell: float, sigma_f: float, obs_sigma: float):
        """
        full marginal log p(y | theta_i, ell, sigma_f) after integrating delta:
          r = y - y_s(theta)
          r ~ N(0, K + obs^2 I)
        """
        y_s = computer_model(X, torch.tensor([theta_i], device=X.device, dtype=X.dtype))[0]
        r = y - y_s
        K = rbf_kernel(X, X,
                       torch.tensor(ell, device=X.device, dtype=X.dtype),
                       torch.tensor(sigma_f, device=X.device, dtype=X.dtype))
        Ky = K + (obs_sigma**2) * torch.eye(X.numel(), device=X.device, dtype=X.dtype)
        return mvn_logpdf_zero_mean(r, Ky)

    def step(self, x: torch.Tensor, y: torch.Tensor):
        # 1) propagate states
        self.theta = self.theta + self.cfg.theta_rw * torch.randn_like(self.theta)
        self.logell = self.logell + self.cfg.logell_rw * torch.randn_like(self.logell)
        self.logsig = self.logsig + self.cfg.logsig_rw * torch.randn_like(self.logsig)

        # 2) append new data to history
        self.X_hist = torch.cat([self.X_hist, x])
        self.y_hist = torch.cat([self.y_hist, y])

        # 3) compute full marginal ll on all history, then take difference
        full_ll = torch.empty_like(self.theta)
        for i in range(self.cfg.N):
            ell = float(torch.exp(self.logell[i]).clamp(1e-3, 1e3).item())
            sf = float(torch.exp(self.logsig[i]).clamp(1e-3, 1e3).item())
            full_ll[i] = self._full_marginal_ll(self.X_hist, self.y_hist,
                                                float(self.theta[i].item()), ell, sf, self.cfg.obs_sigma)

        delta_ll = full_ll - self.prev_full_ll
        self.prev_full_ll = full_ll.detach()

        # 4) weight update
        self.logw = self.logw + delta_ll
        self.w = normalize_logw(self.logw)

        # 5) resample (need to resample prev_full_ll consistently)
        if ess(self.w) < self.cfg.resample_frac * self.cfg.N:
            idx = systematic_resample(self.w, self.gen)
            self.theta = self.theta[idx]
            self.logell = self.logell[idx]
            self.logsig = self.logsig[idx]
            self.prev_full_ll = self.prev_full_ll[idx]
            self.logw.zero_()
            self.w.fill_(1.0 / self.cfg.N)

        return ess(self.w), gini(self.w)

    def mean_theta(self):
        return (self.w * self.theta).sum()


# ===============================
# 8. Plotting + runner
# ===============================

def run_and_record_stream(pf1, pf2, pf3, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 1):
    """
    Stream data sequentially to ensure 'history accumulation' is meaningful.
    X: [T]
    Y: [T]
    """
    T = X.numel()
    assert T % batch_size == 0, "For simplicity, choose batch_size dividing T."

    theta_hist_1, theta_hist_2, theta_hist_3 = [], [], []
    ess_hist_1, ess_hist_2, ess_hist_3 = [], [], []

    K = T // batch_size
    for k in range(K):
        x_b = X[k * batch_size:(k + 1) * batch_size]
        y_b = Y[k * batch_size:(k + 1) * batch_size]

        e1, _ = pf1.step(x_b, y_b)
        e2, _ = pf2.step(x_b, y_b)
        e3, _ = pf3.step(x_b, y_b)

        theta_hist_1.append(pf1.mean_theta().item())
        theta_hist_2.append(pf2.mean_theta().item())
        theta_hist_3.append(pf3.mean_theta().item())

        ess_hist_1.append(e1)
        ess_hist_2.append(e2)
        ess_hist_3.append(e3)

        print(f"[t {k+1:02d}/{K}] "
              f"theta: {theta_hist_1[-1]:.3f}, {theta_hist_2[-1]:.3f}, {theta_hist_3[-1]:.3f} | "
              f"ESS: {e1:.1f}, {e2:.1f}, {e3:.1f}")

    return (torch.tensor(theta_hist_1),
            torch.tensor(theta_hist_2),
            torch.tensor(theta_hist_3),
            torch.tensor(ess_hist_1),
            torch.tensor(ess_hist_2),
            torch.tensor(ess_hist_3))

def plot_theta_traj(theta1, theta2, theta3, theta_star=None, save_path="theta_traj.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(theta1, label="PF-θ (no discrepancy)")
    plt.plot(theta2, label="PF-θ + shared GP")
    plt.plot(theta3, label="PF-(θ, GP hypers)")
    if theta_star is not None:
        plt.axhline(theta_star, linestyle="--", color="k", label=r"$\theta_0^\star$")
    plt.xlabel("Step")
    plt.ylabel(r"$\hat{\theta}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_ess_traj(ess1, ess2, ess3, N, save_path="ess_traj.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(ess1, label="PF-θ (no discrepancy)")
    plt.plot(ess2, label="PF-θ + shared GP")
    plt.plot(ess3, label="PF-(θ, GP hypers)")
    plt.axhline(0.5 * N, linestyle="--", color="gray", label="0.5 N")
    plt.xlabel("Step")
    plt.ylabel("ESS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    torch.set_default_dtype(torch.float64)
    cfg = PFConfig(N=256, obs_sigma=0.2, theta_rw=0.05, device="cpu")

    # Generate one dataset (n=30) and stream it point-by-point
    X = torch.linspace(0, 1, 30, dtype=torch.float64)
    Y = physical_model(X) + cfg.obs_sigma * torch.randn_like(X)

    pf1 = PFThetaNoDiscrepancy(cfg)
    pf2 = PFThetaSharedGP(cfg, ell=0.2, sigma_f=1.0)
    pf3 = PFThetaGPHyper(cfg)

    th1, th2, th3, e1, e2, e3 = run_and_record_stream(pf1, pf2, pf3, X, Y, batch_size=1)

    plot_theta_traj(th1, th2, th3, theta_star=1.8771, save_path="theta_traj.png")
    plot_ess_traj(e1, e2, e3, N=cfg.N, save_path="ess_traj.png")

    print("Saved: theta_traj.png, ess_traj.png")

if __name__ == "__main__":
    main()
