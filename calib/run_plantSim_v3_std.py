"""
run_plantSim_v3_std.py - Plant Simulation 校准实验 (标准化版本)

用法:
    # 方式1: 从目录读取Excel文件 (默认)
    python -m calib.run_plantSim_v3_std --data_dir "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"
    
    # 方式2: 从CSV文件读取
    python -m calib.run_plantSim_v3_std --csv "C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/physical_data.csv"
    
    # 其他参数
    python -m calib.run_plantSim_v3_std --csv physical_data.csv --out_dir figs/plantSim/v3_std --modes 0 1 2
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time
from tqdm import tqdm

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig, BOCPDConfig, ModelConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *
from .restart_bocpd_ogp_gpytorch import (
    BOCPD_OGP, OGPPFConfig, OGPParticleFilter,
    RollingStats as OGPRollingStats,
    make_fast_batched_grad_func,
)

from .Deal_data import *
from tqdm import tqdm

import warnings
from gpytorch.utils.warnings import GPInputWarning

warnings.filterwarnings("ignore")

from calib.v3_utils import *

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# =========================
# 1) y transform: signed log1p
# =========================
@dataclasses.dataclass
class SignedLog1pTransformer:
    c: float = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        abs_y = np.abs(y)
        c = np.median(abs_y[abs_y > 0]) if np.any(abs_y > 0) else 1.0
        self.c = float(c)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted")
        y = np.asarray(y).reshape(-1)
        return np.sign(y) * np.log1p(np.abs(y) / self.c)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted")
        z = np.asarray(z).reshape(-1)
        return np.sign(z) * self.c * np.expm1(np.abs(z))


# =========================
# 2) GlobalTransformSep
#    - X_base (5) and theta (1) are standardized SEPARATELY
#    - y_raw -> y_t(signedlog) -> y_s(zscore)
# =========================
@dataclasses.dataclass
class GlobalTransformSep:
    x_base_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)   # 5-d
    theta_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)   # 1-d
    y_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)       # 1-d (on y_t)
    y_transform: SignedLog1pTransformer = dataclasses.field(default_factory=SignedLog1pTransformer)
    fitted: bool = False

    def fit(self, X_base: np.ndarray, theta_raw: np.ndarray, y_raw: np.ndarray):
        X_base = np.asarray(X_base)
        theta_raw = np.asarray(theta_raw).reshape(-1, 1)  # minutes
        y_raw = np.asarray(y_raw).reshape(-1)

        if X_base.shape[0] != theta_raw.shape[0] or X_base.shape[0] != y_raw.shape[0]:
            raise ValueError("fit: length mismatch")

        # fit x scalers
        self.x_base_scaler.fit(X_base)
        self.theta_scaler.fit(theta_raw)

        # fit y transform + y scaler
        self.y_transform.fit(y_raw)
        y_t = self.y_transform.transform(y_raw)
        self.y_scaler.fit(y_t.reshape(-1, 1))

        self.fitted = True
        return self

    # ---- X_base ----
    def X_base_to_s(self, X_base: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        return self.x_base_scaler.transform(np.asarray(X_base)).astype(np.float32)

    # ---- theta ----
    def theta_raw_to_s(self, theta_raw: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        th = np.asarray(theta_raw).reshape(-1, 1)
        return self.theta_scaler.transform(th).ravel().astype(np.float32)

    def theta_s_to_raw(self, theta_s: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        ths = np.asarray(theta_s).reshape(-1, 1)
        return self.theta_scaler.inverse_transform(ths).ravel()

    @property
    def theta_mu(self) -> float:
        return float(self.theta_scaler.mean_[0])

    @property
    def theta_sd(self) -> float:
        return float(self.theta_scaler.scale_[0])

    # ---- y ----
    def y_raw_to_s(self, y_raw: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        y_raw = np.asarray(y_raw).reshape(-1)
        y_t = self.y_transform.transform(y_raw)
        y_s = self.y_scaler.transform(y_t.reshape(-1, 1)).ravel()
        return y_s.astype(np.float32)

    def y_s_to_raw(self, y_s: np.ndarray) -> np.ndarray:
        if not self.fitted: raise ValueError("GlobalTransformSep not fitted")
        y_s = np.asarray(y_s).reshape(-1)
        y_t = self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()
        y_raw = self.y_transform.inverse_transform(y_t)
        return y_raw


# =========================
# 3) MLP (input_dim=6: [x_base_s(5), theta_s(1)])
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128, 64), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =========================
# 4) NNModelTorchStd: train/predict in standardized space only
#    - inputs: X_full_s (B,6)
#    - outputs: y_s (B,)
# =========================
@dataclasses.dataclass
class NNModelTorchStd:
    input_dim: int = 6
    device: str = None
    model: nn.Module = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_full_s: np.ndarray,    # (N,6) standardized
        y_s: np.ndarray,         # (N,) standardized
        val_frac: float = 0.10,
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 200,
        hidden=(128, 64, 32),
        dropout: float = 0.0,
        weight_decay: float = 1e-6,
        seed: int = 0,
        verbose_every: int = 20,
    ):
        dev = self._get_device()

        X_full_s = np.asarray(X_full_s).astype(np.float32)
        y_s = np.asarray(y_s).astype(np.float32).reshape(-1)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full_s, y_s, test_size=val_frac, random_state=seed, shuffle=True
        )

        X_tr_t = torch.from_numpy(X_tr).to(dev)
        y_tr_t = torch.from_numpy(y_tr).to(dev)
        X_va_t = torch.from_numpy(X_va).to(dev)
        y_va_t = torch.from_numpy(y_va).to(dev)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        self.model = MLP(self.input_dim, hidden=hidden, dropout=dropout).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 30
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_va_t)
                val_loss = loss_fn(val_pred, y_va_t).item()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if verbose_every and ep % verbose_every == 0:
                print(f"epoch {ep:4d} | val_mse(y_s)={val_loss:.6f} | best={best_val:.6f}")

            if bad >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_y_s_from_Xfull_s(self, X_full_s: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("NN model not fitted/loaded.")
        dev = self._get_device()
        Xs = np.asarray(X_full_s).astype(np.float32)
        Xt = torch.from_numpy(Xs).to(dev)
        self.model.eval()
        with torch.no_grad():
            y_s = self.model(Xt).detach().cpu().numpy()
        return y_s

    def save(self, path: str):
        if self.model is None:
            raise ValueError("Nothing to save.")
        bundle = {"state_dict": self.model.state_dict()}
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = None, input_dim: int = 6, hidden=(128,128,64)):
        bundle = joblib.load(path)
        obj = cls(input_dim=input_dim, device=device)
        obj.model = MLP(input_dim, hidden=tuple(hidden), dropout=0.0).to(obj._get_device())
        obj.model.load_state_dict(bundle["state_dict"])
        obj.model.eval()
        return obj


# =========================
# 5) Emulator in standardized space
#    predict(x_base_s (B,5), theta_s (N,1)) -> mu_s (N,B), var_s (N,B)
# =========================
class PlantEmulatorNNStd:
    def __init__(self, nn_std: NNModelTorchStd):
        self.nn = nn_std

    def predict(self, x, theta):
        """
        x: torch.Tensor
           - either (B,5) standardized X_base_s
        theta: torch.Tensor
           - either (N,1) theta_s particles
           - or (B,1) theta_s per-sample
        Returns:
           mu_s, var_s  (both torch.float64)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        x = x.to(torch.float64)
        theta = theta.to(torch.float64)

        # Case 1: theta is particle set (N,1), x is batch (B,5) -> output (N,B)
        if theta.ndim == 2 and theta.shape[1] == 1 and x.ndim == 2 and x.shape[1] == 5:
            N = theta.shape[0]
            B = x.shape[0]

            # Build X_full_s for all (particle, batch)
            x_rep = x.unsqueeze(0).repeat(N, 1, 1)              # (N,B,5)
            th_rep = theta.unsqueeze(1).repeat(1, B, 1)         # (N,B,1)
            X_full = torch.cat([x_rep, th_rep], dim=-1)         # (N,B,6)

            X_full_np = X_full.reshape(N*B, 6).cpu().numpy()
            mu_np = self.nn.predict_y_s_from_Xfull_s(X_full_np).reshape(N, B)
            mu = torch.tensor(mu_np, dtype=torch.float64, device=x.device).T
            var = torch.zeros_like(mu)
            return mu, var

        # Case 2: theta is per-sample (B,1) -> output (B,)
        if theta.ndim == 2 and theta.shape[1] == 1 and x.ndim == 2 and x.shape[1] == 5 and theta.shape[0] == x.shape[0]:
            X_full = torch.cat([x, theta], dim=1)               # (B,6)
            mu_np = self.nn.predict_y_s_from_Xfull_s(X_full.cpu().numpy())
            mu = torch.tensor(mu_np, dtype=torch.float64, device=x.device)
            var = torch.zeros_like(mu)
            return mu, var

        raise ValueError(f"Unsupported shapes: x={tuple(x.shape)}, theta={tuple(theta.shape)}")


def batch_X_base_to_s(gt: GlobalTransformSep, Xb: np.ndarray) -> torch.Tensor:
    Xs = gt.X_base_to_s(Xb).astype(np.float64)      # (B,5)
    return torch.tensor(Xs, dtype=torch.float64)

def batch_y_to_s(gt: GlobalTransformSep, yb: np.ndarray) -> torch.Tensor:
    ys = gt.y_raw_to_s(yb).astype(np.float64)       # (B,)
    return torch.tensor(ys, dtype=torch.float64)


# =========================
# PlantEmulatorNNStdTorch: Pure-torch differentiable for OGP
# =========================
from calib.emulator import Emulator


class PlantEmulatorNNStdTorch(Emulator):
    """
    Pure-torch differentiable wrapper for standardized NN.
    Works in standardized space: x_base_s [5], theta_s [1] -> y_s.
    """
    _NN_CHUNK = 8192

    def __init__(self, nn_std: NNModelTorchStd, gt: GlobalTransformSep,
                 device: str = "cuda", dtype= torch.float64):
        self.device = device
        self.dtype = dtype
        self.gt = gt
        self.nn = nn_std.model.to(device)
        self.nn.eval()
        for p in self.nn.parameters():
            p.requires_grad_(False)

    def _forward_y_s(self, x_full_s: torch.Tensor) -> torch.Tensor:
        """x_full_s [M, 6] standardized -> y_s [M]. Differentiable."""
        return self.nn(x_full_s.float()).to(self.dtype)

    def predict(self, x: torch.Tensor, theta: torch.Tensor):
        """x [B, 5] x_base_s, theta [N, 1] theta_s -> (mu [B,N], var [B,N])"""
        B, N = x.shape[0], theta.shape[0]
        x_dev = x.to(device=self.device, dtype=self.dtype)
        th_dev = theta.to(device=self.device, dtype=self.dtype)
        x_rep = x_dev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        th_rep = th_dev.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        x_full = torch.cat([x_rep, th_rep], dim=1)
        total = B * N
        with torch.no_grad():
            if total <= self._NN_CHUNK:
                y = self._forward_y_s(x_full)
            else:
                y = torch.empty(total, device=self.device, dtype=self.dtype)
                for i in range(0, total, self._NN_CHUNK):
                    j = min(i + self._NN_CHUNK, total)
                    y[i:j] = self._forward_y_s(x_full[i:j])
        mu = y.reshape(B, N)
        return mu, torch.zeros_like(mu)

    def sim_func(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """One-to-one: x [M,5], theta [M,1] -> y [M]. Differentiable."""
        x_full = torch.cat([x.to(self.dtype), theta.to(self.dtype)], dim=1)
        return self._forward_y_s(x_full)

    def x_domain_from_scaler(self, dx: int = 5):
        """Auto-compute x_domain in STANDARDIZED space: mean ± 3*std."""
        x_mean = self.gt.x_base_scaler.mean_
        x_std = self.gt.x_base_scaler.scale_
        lo = x_mean[:dx] - 3 * x_std[:dx]
        hi = x_mean[:dx] + 3 * x_std[:dx]
        return [(float(lo[i]), float(hi[i])) for i in range(dx)]


def _aggregate_ogp_particles(bocpd, ci=0.9):
    all_theta, all_w = [], []
    for e in bocpd.experts:
        w_e = math.exp(e.log_mass)
        all_theta.append(e.pf.theta)
        all_w.append(e.pf.weights() * w_e)
    theta_cat = torch.cat(all_theta, dim=0)
    w_cat = torch.cat(all_w, dim=0)
    w_cat = w_cat / w_cat.sum()
    mean_th = (w_cat.unsqueeze(1) * theta_cat).sum(dim=0)
    var_th = (w_cat.unsqueeze(1) * (theta_cat - mean_th).pow(2)).sum(dim=0)
    alpha = (1 - ci) / 2
    sorted_th, sorted_idx = torch.sort(theta_cat, dim=0)
    sorted_w = w_cat[sorted_idx.squeeze()]
    cum_w = torch.cumsum(sorted_w, dim=0)
    lo_idx = (cum_w >= alpha).nonzero(as_tuple=True)[0][0]
    hi_idx = (cum_w >= 1 - alpha).nonzero(as_tuple=True)[0][0]
    return mean_th, var_th, sorted_th[lo_idx], sorted_th[hi_idx]


# =========================
# 6) Helper: build standardized batches
# =========================
def batch_X_base_to_s(gt: GlobalTransformSep, Xb: np.ndarray) -> torch.Tensor:
    Xs = gt.X_base_to_s(Xb).astype(np.float64)      # (B,5)
    return torch.tensor(Xs, dtype=torch.float64)

def batch_y_to_s(gt: GlobalTransformSep, yb: np.ndarray) -> torch.Tensor:
    ys = gt.y_raw_to_s(yb).astype(np.float64)       # (B,)
    return torch.tensor(ys, dtype=torch.float64)


# =========================
# 7) Pipeline initialisation & helpers
# =========================
# _DEFAULT_NPZ = r"C:/Users/yxu59/files/winter2026/park/simulation/ComputerData_v3/factory_aggregated.npz"
_DEFAULT_NPZ = r"factory_aggregated.npz"

# Module-level globals — populated by init_pipeline()
gt  = None   # type: GlobalTransformSep
nn_std = None   # type: NNModelTorchStd
emu = None   # type: PlantEmulatorNNStd
a_s = 0.0
b_s = 0.0


def init_pipeline(
    npz_path: str = None,
    model_save_path: str = "nn_std.bundle.joblib",
    epochs: int = 200,
    force_retrain: bool = False,
):
    """Load computer-sim data, fit transforms, train / load NN emulator.

    Sets module-level globals (gt, nn_std, emu, a_s, b_s).
    Returns (gt, nn_std, emu, a_s, b_s) for convenience.
    """
    global gt, nn_std, emu, a_s, b_s
    if gt is not None and not force_retrain:
        return gt, nn_std, emu, a_s, b_s

    if npz_path is None:
        npz_path = _DEFAULT_NPZ

    print(f"[init] Loading computer data from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_base = data["X"]       # (N,5)
    y_raw  = data["y"]       # (N,)
    theta_raw = data["theta"]  # (N,) minutes

    X_tr, X_te, y_tr, y_te, th_tr, th_te = train_test_split(
        X_base, y_raw, theta_raw, test_size=0.2, random_state=0, shuffle=True
    )

    gt = GlobalTransformSep().fit(X_tr, th_tr, y_tr)

    if os.path.exists(model_save_path) and not force_retrain:
        print(f"[init] Loading pre-trained NN from {model_save_path}")
        nn_std = NNModelTorchStd.load(model_save_path, hidden=(128, 64, 32))
    else:
        print(f"[init] Training NN emulator ({epochs} epochs) ...")
        X_tr_s = gt.X_base_to_s(X_tr)
        th_tr_s = gt.theta_raw_to_s(th_tr).reshape(-1, 1)
        X_full_tr_s = np.concatenate([X_tr_s, th_tr_s], axis=1)
        y_tr_s = gt.y_raw_to_s(y_tr)
        nn_std = NNModelTorchStd(input_dim=6).fit(X_full_tr_s, y_tr_s, epochs=epochs)
        nn_std.save(model_save_path)

    emu = PlantEmulatorNNStd(nn_std)

    a_raw, b_raw = 3.0, 21.0
    a_s = (a_raw - gt.theta_mu) / gt.theta_sd
    b_s = (b_raw - gt.theta_mu) / gt.theta_sd

    print("[init] Pipeline ready.\n")
    return gt, nn_std, emu, a_s, b_s


def prior_sampler(N):
    return torch.rand(N, 1, dtype=torch.float64) * (b_s - a_s) + a_s   # theta_s

def batches(stream: StreamClass, batch_size: int):
    while True: 
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break

def summarize_metrics(result: dict):
    """Compute theta/y metrics from one method result dict."""
    theta = np.asarray(result.get("theta_hist", []), dtype=float)
    theta_var = np.asarray(result.get("theta_var_hist", []), dtype=float)
    gt_theta = np.asarray(result.get("gt_theta_hist", []), dtype=float)
    y_rmse_hist = np.asarray(result.get("rmse_hist", []), dtype=float)
    y_crps_hist = np.asarray(result.get("y_crps_hist", []), dtype=float)

    n_theta = min(len(theta), len(gt_theta), len(theta_var))
    if n_theta == 0:
        theta_rmse = float("nan")
        theta_crps = float("nan")
    else:
        theta_rmse = float(np.sqrt(np.mean((theta[:n_theta] - gt_theta[:n_theta]) ** 2)))
        theta_var_clip = np.clip(theta_var[:n_theta], 1e-12, None)
        theta_crps = float(
            crps_gaussian(
                torch.tensor(theta[:n_theta], dtype=torch.float64),
                torch.tensor(theta_var_clip, dtype=torch.float64),
                torch.tensor(gt_theta[:n_theta], dtype=torch.float64),
            ).mean().item()
        )

    # rmse_hist has an initial placeholder 0 in this script
    y_rmse = float(np.mean(y_rmse_hist[1:])) if len(y_rmse_hist) > 1 else float("nan")
    y_crps = float(np.mean(y_crps_hist)) if len(y_crps_hist) > 0 else float("nan")

    return dict(
        theta_rmse=theta_rmse,
        theta_crps=theta_crps,
        y_rmse=y_rmse,
        y_crps=y_crps,
    )

def run_plantSim(mode, methods, batch_size, data_dir=None, csv_path=None):
    if data_dir is None and csv_path is None:
        data_dir = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"

    # emulator = PlantEmulatorNN()
    emulator = emu

    results = {}

    for name, meta in methods.items():
        theta_hist, theta_var_hist, gt_theta_hist = [], [], []
        rmse_hist, comp_rmse_hist = [0], [0]
        y_crps_hist = []
        restart_hist = []
        idx = 0
        batch_size = batch_size

        if mode == 2:
            jp = JumpPlan(
                max_jumps=5,           # ~4-5 jumps for ~1200 pts
                min_gap_theta=500.0,   # seconds; tune
                min_interval=180,
                max_interval=320,
                min_jump_span=40,
                seed=7
            )
            stream = StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
        else:
            stream = StreamClass(mode, folder=data_dir, csv_path=csv_path)

        # ---------- R-BOCPD-PF-OGP ----------
        if name == "R-BOCPD-PF-OGP":
            ogp_dev = "cuda"
            emulator_nn_std = PlantEmulatorNNStdTorch(nn_std, gt, device=ogp_dev)
            grad_func = make_fast_batched_grad_func(
                emulator_nn_std.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn_std.x_domain_from_scaler(dx=5)
            a_s = (3.0 - gt.theta_mu) / gt.theta_sd
            b_s = (21.0 - gt.theta_mu) / gt.theta_sd
            ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([a_s]),
                theta_hi=torch.tensor([b_s]),
                theta_move_std=0.5 / gt.theta_sd,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=gt.y_scaler.scale_[0])
            roll = OGPRollingStats(window=50)

            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,
                device=ogp_dev,
            )

            for Xb, yb, thb in tqdm(batches(stream, batch_size), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb).to(device=ogp_dev)
                newY = batch_y_to_s(gt, yb).to(device=ogp_dev)
                gt_theta = torch.tensor(thb)

                if idx > 0 and len(bocpd.experts) > 0:
                    mix_mu = torch.zeros(newX.shape[0], device=ogp_dev, dtype=torch.float64)
                    mix_var = torch.zeros(newX.shape[0], device=ogp_dev, dtype=torch.float64)
                    Z = 0.0
                    for e in bocpd.experts:
                        w_e = math.exp(e.log_mass)
                        e_Xh = e.X_hist if e.X_hist.numel() > 0 else None
                        e_yh = e.y_hist if e.y_hist.numel() > 0 else None
                        mu_e, var_e = e.pf.predict_batch(
                            newX, e_Xh, e_yh,
                            emulator_nn_std, model_cfg.rho, model_cfg.sigma_eps,
                        )
                        mix_mu += w_e * mu_e
                        mix_var += w_e * var_e
                        Z += w_e
                    mix_mu /= max(Z, 1e-12)
                    mix_var /= max(Z, 1e-12)
                    mu_raw = gt.y_s_to_raw(mix_mu.cpu().numpy())
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps = crps_gaussian(mix_mu.detach().cpu(), mix_var.detach().cpu(), newY.detach().cpu()).mean()
                    y_crps_hist.append(float(y_crps.item()))
                idx += 1

                rec = bocpd.update_batch(
                    newX, newY, emulator_nn_std, model_cfg, None, prior_sampler,
                    verbose=False,
                )

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mean_theta_s, var_theta_s, lo_s, hi_s = _aggregate_ogp_particles(
                    bocpd, 0.9,
                )
                mean_theta_raw = gt.theta_s_to_raw(float(mean_theta_s[0]))
                var_theta_raw = float(var_theta_s[0]) * (gt.theta_sd ** 2)
                gt_theta_hist.append(float(np.mean(thb)))
                # print(mean_theta_raw, var_theta_raw)
                theta_hist.append(mean_theta_raw)
                theta_var_hist.append(var_theta_raw)
                restart_hist.append(rec["did_restart"])

        # ---------- Standalone PF-OGP (no BOCPD) ----------
        elif name == "PF-OGP":
            ogp_dev = "cuda"
            emulator_nn_std = PlantEmulatorNNStdTorch(nn_std, gt, device=ogp_dev)
            pf_grad_func = make_fast_batched_grad_func(
                emulator_nn_std.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn_std.x_domain_from_scaler(dx=5)
            a_s = (3.0 - gt.theta_mu) / gt.theta_sd
            b_s = (21.0 - gt.theta_mu) / gt.theta_sd
            pf_ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([a_s]),
                theta_hi=torch.tensor([b_s]),
                theta_move_std=0.5 / gt.theta_sd,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=gt.y_scaler.scale_[0])

            pf = OGPParticleFilter(
                ogp_cfg=pf_ogp_cfg,
                prior_sampler=prior_sampler,
                batched_grad_func=pf_grad_func,
                device=ogp_dev,
                dtype=torch.float64,
            )

            pf_X_hist = torch.empty(0, 5, dtype=torch.float64, device=ogp_dev)
            pf_y_hist = torch.empty(0, dtype=torch.float64, device=ogp_dev)
            pf_ogp_max_hist = 200

            for Xb, yb, thb in tqdm(batches(stream, batch_size), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb).to(device=ogp_dev)
                newY = batch_y_to_s(gt, yb).to(device=ogp_dev)
                gt_theta = torch.tensor(thb)

                if idx > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(
                        newX, pf_Xh, pf_yh,
                        emulator_nn_std, pf_model_cfg.rho, pf_model_cfg.sigma_eps,
                    )
                    mu_raw = gt.y_s_to_raw(mu_mix.cpu().numpy())
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps = crps_gaussian(mu_mix.detach().cpu(), var_mix.detach().cpu(), newY.detach().cpu()).mean()
                    y_crps_hist.append(float(y_crps.item()))
                idx += 1

                pf.step_batch(
                    newX, newY,
                    pf_X_hist if pf_X_hist.numel() > 0 else None,
                    pf_y_hist if pf_y_hist.numel() > 0 else None,
                    emulator_nn_std,
                    pf_model_cfg.rho,
                    pf_model_cfg.sigma_eps,
                )

                if pf_X_hist.numel() == 0:
                    pf_X_hist = newX.clone()
                    pf_y_hist = newY.clone()
                else:
                    pf_X_hist = torch.cat([pf_X_hist, newX], dim=0)
                    pf_y_hist = torch.cat([pf_y_hist, newY], dim=0)
                if pf_X_hist.shape[0] > pf_ogp_max_hist:
                    pf_X_hist = pf_X_hist[-pf_ogp_max_hist:]
                    pf_y_hist = pf_y_hist[-pf_ogp_max_hist:]

                w = pf.weights().view(-1, 1)
                mean_theta_s = (w * pf.theta).sum(dim=0)
                mean_theta_raw = gt.theta_s_to_raw(float(mean_theta_s[0]))
                var_theta_raw = float(
                    (w * (pf.theta - mean_theta_s).pow(2)).sum(dim=0)[0]
                ) * (gt.theta_sd ** 2)
                gt_theta_hist.append(float(np.mean(thb)))
                theta_hist.append(mean_theta_raw)
                theta_var_hist.append(var_theta_raw)
                restart_hist.append(False)

        # ---------- Existing BOCPD ----------
        else:
            cfg = CalibrationConfig()
            cfg.bocpd.bocpd_mode = meta["mode"]
            cfg.bocpd.use_restart = True

            if meta["mode"] == "restart":
                cfg.model.use_discrepancy = meta["use_discrepancy"]

            calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
            for Xb, yb, thb in tqdm(batches(stream, batch_size), desc=f"Running {name}"):
                newX = batch_X_base_to_s(gt, Xb)    # (B,5) standardized; DO NOT include thb
                newY = batch_y_to_s(gt, yb)         # (B,) standardized

                if idx > 0:
                    pred = calib.predict_batch(newX)           # pred["mu"] is y_s
                    pred_comp = calib.predict_complete(newX, newY)
                    mu_s = pred["mu"].detach().cpu().numpy()
                    mu_raw = gt.y_s_to_raw(mu_s)
                    rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                    rmse_hist.append(rmse)
                    y_crps = crps_gaussian(pred["mu"].detach().cpu(), pred["var"].detach().cpu(), newY.detach().cpu()).mean()
                    y_crps_hist.append(float(y_crps.item()))
                    report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                    comp_rmse_hist.append(report_sub_hist)

                idx += 1

                rec = calib.step_batch(newX, newY, verbose=False)

                mean_theta_s, var_theta_s, lo_s, hi_s = calib._aggregate_particles(0.9)
                mean_theta_raw = gt.theta_s_to_raw(mean_theta_s.item())
                var_theta_raw = var_theta_s * (gt.theta_sd ** 2)

                gt_theta_hist.append(float(np.mean(thb)))      # raw minutes
                theta_hist.append(mean_theta_raw.item())       # raw minutes
                theta_var_hist.append(float(var_theta_raw))    # raw^2
                restart_hist.append(rec["did_restart"])

            
            # newX = torch.tensor(Xb)
            # theta = torch.tensor(thb)
            # newY = torch.tensor(yb)
            # # X_torch, Y_torch = X_torch[:batch_size,:], Y_torch[:batch_size]

            # if idx > 0: 
            #     pred = calib.predict_batch(newX)
            #     rmse_hist.append(torch.sqrt(((pred["mu"] - newY)**2).mean()))
            #     pred_comp = calib.predict_complete(newX, newY)
            #     report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
            #     comp_rmse_hist.append(report_sub_hist)
            # idx += 1
            
            # rec = calib.step_batch(newX, newY, verbose=False)
            # mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
            # gt_theta_hist.append(theta.mean().item())
            # theta_hist.append(mean_theta.item())
            # theta_var_hist.append(var_theta)

            # restart_hist.append(rec["did_restart"])

        results[name] = dict(
            theta_hist=theta_hist,
            theta_var_hist=theta_var_hist,
            gt_theta_hist=gt_theta_hist,
            rmse_hist=rmse_hist,
            y_crps_hist=y_crps_hist,
            comp_rmse_hist=comp_rmse_hist,
            restart_hist=restart_hist,
        )
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plant Simulation Calibration Experiment (Standardized)")
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v3_std", help="Output directory for figures")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to PhysicalData directory (Excel files)")
    parser.add_argument("--csv", type=str, default=None, help="Path to aggregated CSV file")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 0], help="Modes to run (e.g., --modes 0 1 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Initialise NN pipeline (load data + train/load model)
    init_pipeline()
    
    methods = {
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        "R-BOCPD-PF-OGP": dict(type="ogp_bocpd"),
        # "PF-OGP": dict(type="pf_ogp"),
        # "BOCPD-PF": dict(type="bocpd", mode="standard"),
        # "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
        # "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False),
        # "R-BOCPD-PF-halfdiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False, bocpd_use_discrepancy=True),
        # "BPC-80": dict(type="bpc"),
    }
    all_results = {}
    for mode in args.modes:
        for bs in [args.batch_size]:
            results = run_plantSim(
                mode=mode, 
                methods=methods, 
                batch_size=bs,
                data_dir=args.data_dir,
                csv_path=args.csv,
            )
            all_results[f"mode{mode}_bs{bs}"] = results

            print("\n" + "=" * 70)
            print(f"Mode={mode}, batch_size={bs} metrics")
            print("=" * 70)
            for name, result in results.items():
                metrics = summarize_metrics(result)
                print(
                    f"{name}: "
                    f"theta_rmse={metrics['theta_rmse']:.6f}, "
                    f"theta_crps={metrics['theta_crps']:.6f}, "
                    f"y_rmse={metrics['y_rmse']:.6f}, "
                    f"y_crps={metrics['y_crps']:.6f}"
                )

            plt.figure(figsize=(10, 5))
            for name, result in results.items():
                plt.plot(result["theta_hist"], label=name)
            plt.plot(result["gt_theta_hist"], "k--", lw=2, label="oracle θ*")
            plt.title(f"Theta tracking (mode={mode}, batch size={bs})")
            plt.xlabel("batch index")
            plt.ylabel("theta")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/mode{mode}_bs{bs}_theta.png", dpi=300)
            plt.close()

        torch.save(all_results, f"{out_dir}/plantSim_results_mode{mode}.pt")
