import numpy as np
import math
from .bpc import BayesianProjectedCalibration

def _gaussian_logpdf_sum(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """Sum_i log N(y_i | mu_i, var_i)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    var = np.clip(var, 1e-12, np.inf)
    return float(np.sum(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y - mu) ** 2 / var))

class BPCExpert:
    """
    One BOCPD expert = one run-length hypothesis,
    modeled by a batch Bayesian Projected Calibration.
    """

    def __init__(
        self,
        theta_lo,
        theta_hi,
        noise_var,
        y_sim,
        X_grid,
        device="cpu",
    ):
        self.theta_lo = theta_lo
        self.theta_hi = theta_hi
        self.noise_var = noise_var
        self.y_sim = y_sim
        self.device = device
        self.X_grid = X_grid

        self.X_hist = None
        self.y_hist = None
        self.run_length = 0

        self.bpc = None   # BayesianProjectedCalibration instance

    # ---------- model fitting ----------
    def fit(self, X, y, **bpc_fit_kwargs):

        self.X_hist = np.asarray(X, float)
        self.y_hist = np.asarray(y, float).reshape(-1)
        self.run_length = self.X_hist.shape[0]

        self.bpc = BayesianProjectedCalibration(
            theta_lo=self.theta_lo,
            theta_hi=self.theta_hi,
            noise_var=self.noise_var,
            y_sim=self.y_sim,
            device=self.device,
        )

        self.bpc.fit(
            self.X_hist,
            self.y_hist,
            self.X_grid,
            **bpc_fit_kwargs,
        )

    # ---------- data assimilation ----------
    def append(self, Xb, yb):
        Xb = np.asarray(Xb, float)
        yb = np.asarray(yb, float).reshape(-1)

        if self.X_hist is None:
            self.X_hist = Xb
            self.y_hist = yb
        else:
            self.X_hist = np.vstack([self.X_hist, Xb])
            self.y_hist = np.concatenate([self.y_hist, yb])

        self.run_length = self.X_hist.shape[0]

    # ---------- predictive likelihood ----------
    def log_predictive_likelihood(self, Xb, yb):
        """
        log p(yb | Xb, current expert model)
        """
        mu, var = self.bpc.predict(Xb)
        mu = mu.reshape(-1)
        var = np.maximum(var.reshape(-1), 1e-10)
        yb = np.asarray(yb, float).reshape(-1)

        return float(
            np.sum(
                -0.5 * np.log(2 * np.pi * var)
                -0.5 * (yb - mu) ** 2 / var
            )
        )

    # ---------- predictive distribution ----------
    def predict(self, Xnew):
        return self.bpc.predict(Xnew)


class StandardBOCPD_BPC:
    """
    Standard BOCPD recursion, where each run-length hypothesis is represented
    by a *BPC expert* fitted on that segment's data.

    Key correctness:
      - At time t, predictive for each hypothesis uses its model fitted on y_{1:t-1}.
      - Then weights are updated using p(y_t | r_{t-1}=r-1) (growth) and
        p(y_t | r_t=0) (CP prior predictive).
      - Only AFTER updating weights, we append (X_t,y_t) to the retained experts and re-fit BPC.

    This is expensive (re-fitting many GPs + many projections). Use small topk.

    Assumptions:
      - Observations arrive in batches (X_batch, y_batch).
      - Hazard is constant h (geometric).
      - CP prior predictive uses BPC prior induced theta (no delta).
    """

    class _Expert:
        def __init__(self, run_length: int, logw: float, bpc):
            self.run_length = int(run_length)
            self.logw = float(logw)   # unnormalized log mass
            self.bpc = bpc            # fitted on data up to previous step
            self.X = None
            self.y = None

        def has_data(self) -> bool:
            return self.X is not None and self.X.shape[0] > 0

    def __init__(
        self,
        *,
        theta_lo: np.ndarray,
        theta_hi: np.ndarray,
        noise_var: float,
        y_sim: callable,
        X_grid: np.ndarray,
        hazard_h: float = 1.0 / 800.0,
        topk: int = 5,
        # BPC fitting knobs
        n_eta_draws_fit: int = 500,
        n_restart_fit: int = 10,
        gp_fit_iters: int = 200,
        # CP prior predictive knobs
        n_eta_draws_prior: int = 200,
        n_restart_prior: int = 5,
        device: str = "cpu",
        seed: int | None = 123,
    ):
        self.theta_lo = np.asarray(theta_lo, dtype=float).reshape(-1)
        self.theta_hi = np.asarray(theta_hi, dtype=float).reshape(-1)
        self.noise_var = float(noise_var)
        self.y_sim = y_sim
        self.X_grid = np.asarray(X_grid, dtype=float)

        self.h = float(hazard_h)
        self.h = min(max(self.h, 1e-12), 1.0 - 1e-12)
        self.topk = int(topk)

        self.n_eta_draws_fit = int(n_eta_draws_fit)
        self.n_restart_fit = int(n_restart_fit)
        self.gp_fit_iters = int(gp_fit_iters)

        self.n_eta_draws_prior = int(n_eta_draws_prior)
        self.n_restart_prior = int(n_restart_prior)

        self.device = device
        self.seed = seed
        self.t = 0  # number of points processed

        # Start with a single "empty" expert representing r=0 at t=0 (no data yet).
        e0 = self._new_bpc_expert(run_length=0, logw=0.0)
        self.experts = [e0]

        self.prev_dominant_rl = None

    def _new_bpc(self):
        # Uses your BayesianProjectedCalibration class (must be in scope).
        bpc = BayesianProjectedCalibration(
            theta_lo=self.theta_lo,
            theta_hi=self.theta_hi,
            noise_var=self.noise_var,
            y_sim=self.y_sim,
            device=self.device,
        )
        # attach the method if you didn’t literally paste it inside the class
        if not hasattr(bpc, "prior_log_predictive"):
            bpc.prior_log_predictive = prior_log_predictive.__get__(bpc, type(bpc))
        return bpc

    def _new_bpc_expert(self, run_length: int, logw: float):
        bpc = self._new_bpc()
        return self._Expert(run_length=run_length, logw=logw, bpc=bpc)

    def _append_data(self, e: _Expert, Xb: np.ndarray, yb: np.ndarray):
        Xb = np.asarray(Xb, dtype=float)
        yb = np.asarray(yb, dtype=float).reshape(-1)
        if e.X is None:
            e.X = Xb.copy()
            e.y = yb.copy()
        else:
            e.X = np.concatenate([e.X, Xb], axis=0)
            e.y = np.concatenate([e.y, yb], axis=0)

    def _refit_expert(self, e: _Expert):
        # Fit BPC posterior induced theta + delta on this expert's segment data.
        # Uses the same X_grid for L2 projection integral discretization.
        assert e.X is not None and e.y is not None
        e.bpc.fit(
            e.X,
            e.y,
            self.X_grid,
            n_eta_draws=self.n_eta_draws_fit,
            n_restart=self.n_restart_fit,
            gp_fit_iters=self.gp_fit_iters,
        )

    def _expert_log_predictive(self, e: _Expert, Xb: np.ndarray, yb: np.ndarray) -> float:
        # Predict with the expert model (fitted on past segment) and score yb.
        # If expert has no data yet (shouldn't happen except at init), fall back to CP prior predictive.
        if not e.has_data() or e.bpc.theta_samples is None:
            return e.bpc.prior_log_predictive(
                Xb, yb, self.X_grid,
                n_eta_draws=self.n_eta_draws_prior,
                n_restart=self.n_restart_prior,
                seed=self.seed,
            )
        mu, var = e.bpc.predict(Xb)
        return _gaussian_logpdf_sum(yb, mu, var)

    def _normalize_and_prune(self):
        logw = np.array([e.logw for e in self.experts], dtype=float)
        m = np.max(logw)
        w = np.exp(logw - m)
        w /= np.sum(w)
        logZ = m + np.log(np.sum(np.exp(logw - m)))
        # write back normalized logw
        for i, e in enumerate(self.experts):
            e.logw = float(np.log(w[i] + 1e-300))

        # prune top-k by mass
        if len(self.experts) > self.topk:
            idx = np.argsort(-w)[: self.topk].tolist()
            self.experts = [self.experts[i] for i in idx]
            # renormalize again
            logw2 = np.array([e.logw for e in self.experts], dtype=float)
            m2 = np.max(logw2)
            w2 = np.exp(logw2 - m2)
            w2 /= np.sum(w2)
            for i, e in enumerate(self.experts):
                e.logw = float(np.log(w2[i] + 1e-300))

        return float(logZ)

    def step_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> dict:
        """
        One BOCPD update with a batch (treated as a single time step for BOCPD),
        but expert run_length increases by batch_size.

        Returns diagnostics.
        """
        Xb = np.asarray(X_batch, dtype=float)
        yb = np.asarray(y_batch, dtype=float).reshape(-1)
        B = Xb.shape[0]

        # 1) Compute predictive likelihoods using models fitted on *past data*
        logpreds = []
        for e in self.experts:
            logpreds.append(self._expert_log_predictive(e, Xb, yb))

        # CP prior predictive (does not use any expert data)
        bpc_prior = self._new_bpc()
        logpred_cp = bpc_prior.prior_log_predictive(
            Xb, yb, self.X_grid,
            n_eta_draws=self.n_eta_draws_prior,
            n_restart=self.n_restart_prior,
            seed=self.seed,
        )

        # 2) Standard BOCPD weight recursion (growth + CP)
        log_h = np.log(self.h)
        log_1mh = np.log(1.0 - self.h)

        new_experts = []

        # growth: each existing expert advances run_length by +B
        for e, lp in zip(self.experts, logpreds):
            eg = self._Expert(run_length=e.run_length + B, logw=e.logw + log_1mh + lp, bpc=e.bpc)
            eg.X = e.X
            eg.y = e.y
            new_experts.append(eg)

        # cp: new expert with run_length = 0 (before seeing data), weight uses prior predictive
        e_cp = self._new_bpc_expert(run_length=0, logw=log_h + logpred_cp)
        new_experts.append(e_cp)

        self.experts = new_experts

        # 3) Normalize + prune (still BEFORE incorporating current batch into models)
        logZ = self._normalize_and_prune()

        # 4) Now incorporate current batch into each retained expert and re-fit BPC
        #    (this makes them ready for next step's predictive)
        for e in self.experts:
            self._append_data(e, Xb, yb)
            self._refit_expert(e)

        self.t += B

        dominant_expert = max(self.experts, key=lambda e: e.logw)
        dominant_rl = dominant_expert.run_length

        if self.prev_dominant_rl is None:
            did_restart = False
            expected_rl = None
        else:
            expected_rl = self.prev_dominant_rl + B
            # tol = int(getattr(self.config, "implicit_rl_tol", 1))
            tol = 0
            did_restart = abs(expected_rl - dominant_rl) > tol*B

        self.prev_dominant_rl = dominant_rl

        # diagnostics
        masses = np.exp([e.logw for e in self.experts])
        masses = (masses / masses.sum()).tolist()
        run_lengths = [e.run_length for e in self.experts]
        theta_means = []
        for e in self.experts:
            if getattr(e.bpc, "theta_mean", None) is None:
                theta_means.append(None)
            else:
                theta_means.append(np.asarray(e.bpc.theta_mean, dtype=float).reshape(-1).tolist())

        return {
            "t": int(self.t),
            "batch_size": int(B),
            "logZ": float(logZ),
            "num_experts": int(len(self.experts)),
            "run_lengths": run_lengths,
            "masses": masses,
            "theta_means": theta_means,
            "logpred_cp": float(logpred_cp),
            "logpreds_existing": [float(v) for v in logpreds],
            "did_restart": did_restart,
            "dominant_rl": int(dominant_rl),
            "expected_rl": None if expected_rl is None else int(expected_rl),
        }

    def predict(self, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mixture predictive across experts:
          mu = sum_k w_k mu_k
          var = sum_k w_k (var_k + mu_k^2) - mu^2
        """
        Xt = np.asarray(Xt, dtype=float)
        logw = np.array([e.logw for e in self.experts], dtype=float)
        m = np.max(logw)
        w = np.exp(logw - m)
        w /= np.sum(w)

        mus = []
        vars_ = []
        for e in self.experts:
            mu_k, var_k = e.bpc.predict(Xt)
            mus.append(np.asarray(mu_k, dtype=float).reshape(-1))
            vars_.append(np.asarray(var_k, dtype=float).reshape(-1))

        mus = np.stack(mus, axis=0)   # [K, T]
        vars_ = np.stack(vars_, axis=0)  # [K, T]

        w = w.reshape(-1, 1)          # [K,1]
        mu = np.sum(w * mus, axis=0)
        second = np.sum(w * (vars_ + mus ** 2), axis=0)
        var = np.maximum(second - mu ** 2, 1e-12)
        return mu, var

    def predict_sim(self, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Mixture predictive across experts:
          mu = sum_k w_k mu_k
          var = sum_k w_k (var_k + mu_k^2) - mu^2
        """
        Xt = np.asarray(Xt, dtype=float)
        logw = np.array([e.logw for e in self.experts], dtype=float)
        m = np.max(logw)
        w = np.exp(logw - m)
        w /= np.sum(w)

        mus = []
        vars_ = []
        for e in self.experts:
            mu_k, var_k = e.bpc.predict_sim(Xt)
            mus.append(np.asarray(mu_k, dtype=float).reshape(-1))
            vars_.append(np.asarray(var_k, dtype=float).reshape(-1))

        mus = np.stack(mus, axis=0)   # [K, T]
        vars_ = np.stack(vars_, axis=0)  # [K, T]

        w = w.reshape(-1, 1)          # [K,1]
        mu = np.sum(w * mus, axis=0)
        second = np.sum(w * (vars_ + mus ** 2), axis=0)
        var = np.maximum(second - mu ** 2, 1e-12)
        return mu, var

    def _aggregate_particles(self, quantile = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta_list, weight_list = [], []
        for e in self.experts:
            if e.bpc.theta_samples is None:
                continue
            w_e = math.exp(e.logw)
            th = e.bpc.theta_samples
            S = th.shape[0]
            theta_list.append(th)
            weight_list.append(np.full(S, w_e/S))

        theta_all = np.concatenate(theta_list, axis=0)
        weight_all = np.concatenate(weight_list, axis=0)

        weight_all = weight_all / weight_all.sum()

        def weighted_mean_and_cov(theta, w):
            """
            theta: [M, p]
            w:     [M], sum to 1
            """
            mean = np.sum(theta * w[:, None], axis=0)            # [p]
            diff = theta - mean
            cov = diff.T @ (diff * w[:, None])                   # [p, p]
            return mean, cov

        def weighted_quantile_1d(x, w, q):
            """
            x: [M]
            w: [M] sum to 1
            q: quantile in (0,1)
            """
            idx = np.argsort(x)
            x = x[idx]
            w = w[idx]
            cw = np.cumsum(w)
            return x[cw >= q][0]

        def weighted_credible_interval(theta, w, level=0.9):
            """
            theta: [M, p]
            w:     [M]
            """
            alpha = (1.0 - level) / 2.0
            lo_q = alpha
            hi_q = 1.0 - alpha

            p = theta.shape[1]
            lo = np.zeros(p)
            hi = np.zeros(p)

            for j in range(p):
                lo[j] = weighted_quantile_1d(theta[:, j], w, lo_q)
                hi[j] = weighted_quantile_1d(theta[:, j], w, hi_q)

            return lo, hi

        mean, cov = weighted_mean_and_cov(theta_all, weight_all)
        if quantile is None:
            return mean, cov
        else:
            lo, hi = weighted_credible_interval(theta_all, weight_all, quantile)
            return mean, cov, lo, hi





# =========================
# Minimal usage sketch
# =========================
if __name__ == "__main__":
    np.random.seed(0)

    # ---- Config2 simulator ----
    def y_sim_cfg2(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        theta = np.atleast_2d(theta)
        th = theta[:, [0]]
        xx = x[:, [0]]
        return (np.sin(5.0 * th * xx) + 5.0 * xx).reshape(-1)

    # true physical system eta0
    def eta0(x):
        x = x.reshape(-1)
        return 5.0 * x * np.cos(15.0 * x / 2.0) + 5.0 * x

    n = 30
    X = np.linspace(0, 1, n).reshape(-1, 1)
    noise_sd = 0.2
    y = eta0(X) + noise_sd * np.random.randn(n)

    # integration grid for projection
    X_grid = np.linspace(0, 1, 500).reshape(-1, 1)

    # make batches (e.g., size 5)
    bs = 5
    batches = [(X[i:i+bs], y[i:i+bs]) for i in range(0, n, bs)]

    bocpd = StandardBOCPD_BPC(
        theta_lo=np.array([0.0]),
        theta_hi=np.array([3.0]),
        noise_var=noise_sd**2,
        y_sim=y_sim_cfg2,
        X_grid=X_grid,
        hazard_h=1.0/800.0,
        topk=3,
        n_eta_draws_fit=300,
        n_restart_fit=8,
        gp_fit_iters=150,
        n_eta_draws_prior=200,
        n_restart_prior=6,
        device="cpu",
        seed=0,
    )

    for Xb, yb in batches:
        info = bocpd.step_batch(Xb, yb)
        print(info)

    Xt = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, var = bocpd.predict(Xt)
    print("pred:", mu.shape, var.shape)