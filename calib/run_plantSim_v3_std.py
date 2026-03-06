import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time
from tqdm import tqdm

# -------------------------------------------------------------
# Your existing modules (keep same as before)
# -------------------------------------------------------------
from .configs import CalibrationConfig
from .emulator import DeterministicSimulator
from .online_calibrator import OnlineBayesCalibrator, crps_gaussian
from .bpc import BayesianProjectedCalibration
from .bpc_bocpd import *

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
# 7) MAIN: train + run PF loop
# =========================
# 7.1 load aggregated computer data
data = np.load(r"C:/Users/yxu59/files/winter2026/park/simulation/ComputerData_v3/factory_aggregated.npz", allow_pickle=True)
X_base = data["X"]       # (N,5)
y_raw  = data["y"]       # (N,)
theta_raw = data["theta"]  # (N,) minutes

# train/test split
X_tr, X_te, y_tr, y_te, th_tr, th_te = train_test_split(
    X_base, y_raw, theta_raw, test_size=0.2, random_state=0, shuffle=True
)

# fit transforms ONLY on train
gt = GlobalTransformSep().fit(X_tr, th_tr, y_tr)

# build standardized training set for NN: X_full_s = [X_base_s, theta_s]
X_tr_s = gt.X_base_to_s(X_tr)                           # (N,5)
th_tr_s = gt.theta_raw_to_s(th_tr).reshape(-1, 1)       # (N,1)
X_full_tr_s = np.concatenate([X_tr_s, th_tr_s], axis=1) # (N,6)
y_tr_s = gt.y_raw_to_s(y_tr)                            # (N,)

# train NN in standardized space
nn_std = NNModelTorchStd(input_dim=6).fit(X_full_tr_s, y_tr_s, epochs=200)
nn_std.save("nn_std.bundle.joblib")

# build standardized-space emulator
emu = PlantEmulatorNNStd(nn_std)

a_raw, b_raw = 3.0, 21.0
a_s = (a_raw - gt.theta_mu) / gt.theta_sd
b_s = (b_raw - gt.theta_mu) / gt.theta_sd

def prior_sampler(N):
    return torch.rand(N, 1, dtype=torch.float64) * (b_s - a_s) + a_s   # theta_s

def batches(stream: StreamClass, batch_size: int):
    while True: 
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break

def run_plantSim(mode, methods, batch_size):
    folder = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"

    # emulator = PlantEmulatorNN()
    emulator = emu

    results = {}

    for name, meta in methods.items():
        theta_hist, theta_var_hist, gt_theta_hist = [], [], []
        rmse_hist, comp_rmse_hist = [0], [0]
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
            stream = StreamClass(0, folder, jump_plan=jp)
        else:
            stream = StreamClass(mode, folder)

        cfg = CalibrationConfig()
        cfg.bocpd.bocpd_mode = meta["mode"]
        cfg.bocpd.use_restart = True

        if meta["mode"] == "restart":
            cfg.model.use_discrepancy = meta["use_discrepancy"]

        calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
        for Xb, yb, thb in tqdm(batches(stream, batch_size), desc=f"Running {name}"):
            # standardized inputs for PF/BOCPD
            newX = batch_X_base_to_s(gt, Xb)    # (B,5) standardized; DO NOT include thb
            newY = batch_y_to_s(gt, yb)         # (B,) standardized

            # prediction RMSE in raw revenue space
            if idx > 0:
                pred = calib.predict_batch(newX)           # pred["mu"] is y_s
                pred_comp = calib.predict_complete(newX, newY)
                mu_s = pred["mu"].detach().cpu().numpy()
                mu_raw = gt.y_s_to_raw(mu_s)
                rmse = float(np.sqrt(np.mean((mu_raw - np.asarray(yb))**2)))
                rmse_hist.append(rmse)
                report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                comp_rmse_hist.append(report_sub_hist)

            idx += 1

            # update PF
            rec = calib.step_batch(newX, newY, verbose=False)

            # theta posterior (theta_s) -> raw minutes for logging
            mean_theta_s, var_theta_s, lo_s, hi_s = calib._aggregate_particles(0.9)
            mean_theta_raw = gt.theta_s_to_raw(mean_theta_s.item())
            var_theta_raw = var_theta_s * (gt.theta_sd ** 2)

            gt_theta_hist.append(float(np.mean(thb)))      # raw minutes
            theta_hist.append(mean_theta_raw.item())       # raw minutes
            # print(np.mean(thb), mean_theta_raw)
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

        results[name] = dict(theta_hist=theta_hist, theta_var_hist=theta_var_hist, gt_theta_hist=gt_theta_hist, rmse_hist=rmse_hist, comp_rmse_hist=comp_rmse_hist, restart_hist=restart_hist)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v3_std")
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    methods = {
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        "BOCPD-PF": dict(type="bocpd", mode="standard"),
        "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
        "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False),
        # "BPC-80": dict(type="bpc"),
    }
    all_results = {}
    # for mode in [0, 1]:
    for mode in [1,2,0]:
        for bs in [4]:
            results = run_plantSim(mode=mode, methods=methods, batch_size=bs)
            all_results[f"mode{mode}_bs{bs}"] = results

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
