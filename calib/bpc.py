import numpy as np
import torch
import gpytorch
from scipy.optimize import minimize
from tqdm import tqdm

Tensor = torch.Tensor

def _gaussian_logpdf_sum(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Sum_i log N(y_i | mu_i, var_i)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    var = np.clip(var, 1e-12, np.inf)
    return float(np.sum(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y - mu) ** 2 / var))
# =========================
# 0) User simulator y_s(x, theta)
# =========================
def y_sim(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    X: [N, d]
    theta: [p]
    return y_s(X,theta): [N]
    Replace this with your simulator/emulator mean.
    """
    # example: 1D linear
    return (X[:, 0] * theta[0]).reshape(-1)


# =========================
# 1) GP for physical system eta(x)
# =========================
class PhysicalGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-2)
            )
        )

    def forward(self, x: Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )

def fit_physical_gp(X: np.ndarray, y: np.ndarray, noise_var: float, iters: int = 200, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    # fix observation noise if you want (paper has yi = eta(xi)+eps)
    likelihood.noise = torch.tensor(noise_var, dtype=torch.float32, device=device)
    # likelihood.noise_covar.raw_noise.requires_grad_(False)

    model = PhysicalGP(X_t, y_t, likelihood).to(device)
    model.train(); likelihood.train()

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        out = model(X_t)
        loss = -mll(out, y_t)
        loss.backward()
        opt.step()

    model.eval(); likelihood.eval()
    return model, likelihood


# =========================
# 2) L2 projection theta*_eta
#    theta* = argmin \int (eta(x)-y_s(x,theta))^2 dx
#    Approximate integral via grid sum
# =========================
def project_theta_L2(
    X_grid: np.ndarray,          # [M,d] integration grid
    eta_grid: np.ndarray,        # [M] one GP function draw at grid
    theta_lo: np.ndarray,        # [p]
    theta_hi: np.ndarray,        # [p]
    n_restart: int = 5,
    y_sim1: callable = y_sim,
) -> np.ndarray:
    p = theta_lo.shape[0]

    def obj(theta):
        ys = y_sim1(X_grid, theta)               # [M]
        return float(np.mean((eta_grid - ys) ** 2))

    best_theta, best_val = None, np.inf
    bounds = [(float(theta_lo[j]), float(theta_hi[j])) for j in range(p)]

    for _ in range(n_restart):
        theta0 = theta_lo + (theta_hi - theta_lo) * np.random.rand(p)
        res = minimize(obj, theta0, bounds=bounds, method="L-BFGS-B")
        if res.fun < best_val:
            best_val = float(res.fun)
            best_theta = res.x.copy()

    return best_theta


# =========================
# 3) Delta GP for residuals r(x)=y - y_s(x, theta_bar)
# =========================
class DeltaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )

def fit_delta_gp(X: np.ndarray, r: np.ndarray, iters: int = 200, device="cpu"):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    r_t = torch.tensor(r, dtype=torch.float32, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DeltaGP(X_t, r_t, likelihood).to(device)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        out = model(X_t)
        loss = -mll(out, r_t)
        loss.backward()
        opt.step()

    model.eval(); likelihood.eval()
    return model, likelihood


def gini_theta(samples):
    x = np.sort(np.asarray(samples).reshape(-1))
    n = len(x)
    idx = np.arange(1, n+1)
    return 1 - 2 * np.sum((n - idx + 0.5) * x) / (n * np.sum(x))

def entropy_theta(samples, bins=30):
    hist, _ = np.histogram(samples, bins=bins, density=True)
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# =========================
# 4) Bayesian Projected Calibration (paper-style)
# =========================
class BayesianProjectedCalibration:
    """
    Paper-style BPC:
    1) GP posterior for eta
    2) draw eta functions on grid, project to theta via L2
    => theta posterior samples induced by eta posterior
    Then:
    - theta_mean/var
    - delta GP on residual using theta_mean (as requested)
    - predict: combine Var_theta[y_s] + Var_delta + noise
    """

    def __init__(
        self,
        theta_lo: np.ndarray,
        theta_hi: np.ndarray,
        noise_var: float = 1e-2,
        y_sim: callable = y_sim,
        device: str = "cpu",
    ):
        self.theta_lo = np.asarray(theta_lo, dtype=float)
        self.theta_hi = np.asarray(theta_hi, dtype=float)
        self.noise_var = float(noise_var)
        self.device = device

        self.phys_gp = None
        self.phys_lik = None
        self.delta_gp = None
        self.delta_lik = None

        self.theta_samples = None
        self.theta_mean = None
        self.theta_var = None

        self.X_train = None
        self.y_train = None

        self.y_sim = y_sim

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_grid: np.ndarray,
        n_eta_draws: int = 200,
        n_restart: int = 5,
        gp_fit_iters: int = 200,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        X_grid = np.asarray(X_grid, dtype=float)

        self.X_train, self.y_train = X, y

        # (1) fit GP posterior for physical eta
        self.phys_gp, self.phys_lik = fit_physical_gp(
            X, y, noise_var=self.noise_var, iters=gp_fit_iters, device=self.device
        )

        # (2) draw eta functions on grid from GP posterior
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Xg_t = torch.tensor(X_grid, dtype=torch.float32, device=self.device)
            post_g = self.phys_gp(Xg_t)                 # posterior of eta on grid
            eta_draws = post_g.rsample(torch.Size([n_eta_draws]))  # [S, M]

        eta_draws_np = eta_draws.detach().cpu().numpy()

        # (3) L2 projection for each eta draw => theta samples
        thetas = []
        for s in tqdm(range(n_eta_draws), desc="L2 projection"):
            theta_s = project_theta_L2(
                X_grid=X_grid,
                eta_grid=eta_draws_np[s],
                theta_lo=self.theta_lo,
                theta_hi=self.theta_hi,
                n_restart=n_restart,
                y_sim1=self.y_sim,
            )
            thetas.append(theta_s[None, :])
        thetas = np.concatenate(thetas, axis=0)  # [S,p]

        self.theta_samples = thetas
        self.theta_mean = thetas.mean(axis=0)
        self.theta_var = thetas.var(axis=0)

        # (4) fit delta GP on residuals using theta_mean (your requested workflow)
        # resid = y - self.y_sim(X, self.theta_mean)
        resid = y - self.y_sim(X, self.theta_mean).reshape(-1)
        # print("BPC debug: ", resid.shape, y.shape, self.y_sim(X, self.theta_mean).shape)
        self.delta_gp, self.delta_lik = fit_delta_gp(X, resid, iters=gp_fit_iters, device=self.device)

    def predict(self, Xt: np.ndarray):
        Xt = np.asarray(Xt, dtype=float)
        # theta-induced uncertainty: Var_theta[y_s]
        Ys = np.array([self.y_sim(Xt, th) for th in self.theta_samples])  # [S, T]
        mu_theta = Ys.mean(axis=0)
        var_theta = Ys.var(axis=0)

        # delta GP
        Xt_t = torch.tensor(Xt, dtype=torch.float32, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post_d = self.delta_gp(Xt_t)
            mu_delta = post_d.mean.detach().cpu().numpy()
            var_delta = post_d.variance.detach().cpu().numpy()

        mu = mu_theta + mu_delta
        var = var_theta + var_delta + self.noise_var
        return mu, var

    def predict_sim(self, Xt: np.ndarray):
        Xt = np.asarray(Xt, dtype=float)
        # theta-induced uncertainty: Var_theta[y_s]
        Ys = np.array([self.y_sim(Xt, th) for th in self.theta_samples])  # [S, T]
        mu_theta = Ys.mean(axis=0)
        var_theta = Ys.var(axis=0)

        return mu_theta, var_theta

    def entropy_theta(self):
        return entropy_theta(self.theta_samples[:,0])

    def prior_log_predictive(
        self,
        Xb: np.ndarray,
        yb: np.ndarray,
        X_grid: np.ndarray,
        n_eta_draws: int = 200,
        n_restart: int = 5,
        seed: int | None = None,
    ) -> float:
        """
        log p(yb | Xb) under the *BPC prior* (no data).

        This constructs:
        eta ~ GP prior on X_grid  (using the SAME PhysicalGP architecture/kernel you coded)
        theta_s = L2Projection(eta_s) for s=1..S
        predictive approx:
            p(y|x) ≈ N( mu_theta(x), var_theta(x) + noise_var )
        where mu_theta/var_theta come from Monte Carlo over theta_s.

        Requirements:
        - self.theta_lo, self.theta_hi
        - self.noise_var
        - self.device
        - self.y_sim
        - project_theta_L2(...) must exist
        - PhysicalGP(...) must exist and match your fit_physical_gp model class

        Notes:
        - This is a *prior predictive*. It should NOT fit anything.
        - It uses your GP prior hyperparameters as currently set in PhysicalGP.
            If you want the paper’s exact hyperparameters, you must fix them in the kernel.
        """
        if seed is not None:
            rs = np.random.RandomState(seed)
            # also seed torch if you want strict reproducibility:
            import torch
            torch.manual_seed(int(seed))
        else:
            rs = None

        Xb = np.asarray(Xb, dtype=float)
        yb = np.asarray(yb, dtype=float).reshape(-1)
        X_grid = np.asarray(X_grid, dtype=float)

        # --- Build a "prior GP" that can be evaluated on X_grid without training ---
        # In gpytorch ExactGP, you still need some dummy train inputs/targets for constructor.
        import torch
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        dtype = torch.float32  # keep consistent with your implementation

        X_dummy = torch.zeros(1, X_grid.shape[1], dtype=dtype, device=device)
        y_dummy = torch.zeros(1, dtype=dtype, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        # For eta prior, the observation noise is irrelevant; but we keep it consistent.
        likelihood.noise = torch.tensor(self.noise_var, dtype=dtype, device=device)

        # Use your PhysicalGP class (must be in scope / imported in your file)
        prior_gp = PhysicalGP(X_dummy, y_dummy, likelihood).to(device)
        prior_gp.eval()
        likelihood.eval()

        # --- Draw eta on grid from the *prior* (not posterior) ---
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Xg_t = torch.tensor(X_grid, dtype=dtype, device=device)
            prior_dist = prior_gp(Xg_t)  # prior over f(X_grid)
            eta_draws = prior_dist.rsample(torch.Size([int(n_eta_draws)]))  # [S, M]
        eta_draws_np = eta_draws.detach().cpu().numpy()

        # --- Project each eta draw to theta via your L2 projection solver ---
        thetas = []
        for s in range(int(n_eta_draws)):
            theta_s = project_theta_L2(
                X_grid=X_grid,
                eta_grid=eta_draws_np[s],
                theta_lo=self.theta_lo,
                theta_hi=self.theta_hi,
                n_restart=int(n_restart),
                y_sim1=self.y_sim,
            )
            thetas.append(theta_s[None, :])
        thetas = np.concatenate(thetas, axis=0)  # [S, p]

        # --- Prior predictive induced by theta samples (delta ignored under CP prior) ---
        # y(x) ≈ y_s(x,theta) + eps ; eps ~ N(0, noise_var)
        # MC over theta:
        Ys = np.asarray([self.y_sim(Xb, th) for th in thetas], dtype=float)  # [S, B]
        mu = Ys.mean(axis=0)
        var = Ys.var(axis=0) + float(self.noise_var)

        return _gaussian_logpdf_sum(yb, mu, var)


# =========================
# 5) Example usage
# =========================
if __name__ == "__main__":
    np.random.seed(0)

    # -----------------------------
    # Configuration 2 (paper exact)
    # -----------------------------
    def computer_model_config2_np(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Config2 computer model (NumPy version):
        y*(x, θ) = sin(5 θ x) + 5x
        - x: shape [n] or [n, 1]
        - theta: shape [p] or [1, p]
        Returns: shape [n, 1]
        """
        x = np.atleast_2d(x)
        theta = np.atleast_2d(theta)
        th = theta[:, [0]]
        xx = x[:, [0]]
        return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)

    # 1) equidistant design points on [0,1]
    n = 30
    X = np.linspace(0, 1, n).reshape(-1, 1)

    # 2) true physical system eta_0(x)
    def eta0(x):
        x = x.reshape(-1)
        return 5.0 * x * np.cos(15.0 * x / 2.0) + 5.0 * x

    # 3) noisy observations
    noise_sd = 0.2
    y = eta0(X) + noise_sd * np.random.randn(n)

    # 4) integration grid for L2 projection
    X_grid = np.linspace(0, 1, 500).reshape(-1, 1)

    # 5) Bayesian Projected Calibration
    bpc = BayesianProjectedCalibration(
        theta_lo=np.array([0.0]),
        theta_hi=np.array([3.0]),
        noise_var=noise_sd**2,
        y_sim=computer_model_config2_np,          # y_s(x,theta) = sin(5θx)+5x
        device="cpu",
    )

    bpc.fit(
        X, y, X_grid,
        n_eta_draws=500,      # 论文里 S=1000，你可先 500
        n_restart=10,
        gp_fit_iters=200,
    )

    print("theta_mean:", bpc.theta_mean)
    print("theta_var :", bpc.theta_var)

    # 6) prediction (optional)
    Xt = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, var = bpc.predict(Xt)
    print("pred mu shape:", mu.shape, "var shape:", var.shape)

# if __name__ == "__main__":
#     np.random.seed(0)

#     # training data
#     X = np.random.rand(60, 1)
#     theta_true = np.array([1.5])
#     y = y_sim(X, theta_true) + 0.15 * np.random.randn(60)

#     # integration grid for L2 projection (Omega discretization)
#     X_grid = np.linspace(0, 1, 200).reshape(-1, 1)

#     bpc = BayesianProjectedCalibration(
#         theta_lo=np.array([0.0]),
#         theta_hi=np.array([3.0]),
#         noise_var=0.15**2,
#         y_sim=y_sim,
#         device="cpu",
#     )
#     bpc.fit(
#         X, y, X_grid,
#         n_eta_draws=200,
#         n_restart=8,
#         gp_fit_iters=150,
#     )

#     print("theta_mean:", bpc.theta_mean)
#     print("theta_var:", bpc.theta_var)

#     Xt = np.linspace(0, 1, 100).reshape(-1, 1)
#     mu, var = bpc.predict(Xt)
#     print("pred mu shape:", mu.shape, "var shape:", var.shape)
