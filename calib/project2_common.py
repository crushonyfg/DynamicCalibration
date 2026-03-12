from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
from typing import Callable, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from calib.configs import CalibrationConfig
    from calib.online_calibrator import OnlineBayesCalibrator
    from calib.emulator import Emulator
else:
    from .configs import CalibrationConfig
    from .online_calibrator import OnlineBayesCalibrator
    from .emulator import Emulator


@dataclasses.dataclass
class SignedLog1pTransformer:
    c: Optional[float] = None

    def fit(self, y: np.ndarray) -> "SignedLog1pTransformer":
        y = np.asarray(y).reshape(-1)
        abs_y = np.abs(y)
        c = np.median(abs_y[abs_y > 0]) if np.any(abs_y > 0) else 1.0
        self.c = float(c)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer is not fitted.")
        y = np.asarray(y).reshape(-1)
        return np.sign(y) * np.log1p(np.abs(y) / self.c)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer is not fitted.")
        z = np.asarray(z).reshape(-1)
        return np.sign(z) * self.c * np.expm1(np.abs(z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer is not fitted.")
        z = np.asarray(z).reshape(-1)
        return self.c * np.exp(np.abs(z))


@dataclasses.dataclass
class GlobalTransformSep:
    x_base_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)
    theta_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)
    y_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)
    y_transform: SignedLog1pTransformer = dataclasses.field(default_factory=SignedLog1pTransformer)
    fitted: bool = False

    def fit(self, X_base: np.ndarray, theta_raw: np.ndarray, y_raw: np.ndarray) -> "GlobalTransformSep":
        X_base = np.asarray(X_base)
        theta_raw = np.asarray(theta_raw).reshape(-1, 1)
        y_raw = np.asarray(y_raw).reshape(-1)

        if not (len(X_base) == len(theta_raw) == len(y_raw)):
            raise ValueError("Length mismatch in transform fitting.")

        self.x_base_scaler.fit(X_base)
        self.theta_scaler.fit(theta_raw)
        self.y_transform.fit(y_raw)

        y_t = self.y_transform.transform(y_raw)
        self.y_scaler.fit(y_t.reshape(-1, 1))
        self.fitted = True
        return self

    def X_base_to_s(self, X_base: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        return self.x_base_scaler.transform(np.asarray(X_base)).astype(np.float32)

    def theta_raw_to_s(self, theta_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        theta_raw = np.asarray(theta_raw).reshape(-1, 1)
        return self.theta_scaler.transform(theta_raw).ravel().astype(np.float32)

    def theta_s_to_raw(self, theta_s: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        theta_s = np.asarray(theta_s).reshape(-1, 1)
        return self.theta_scaler.inverse_transform(theta_s).ravel()

    def y_raw_to_s(self, y_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        y_raw = np.asarray(y_raw).reshape(-1)
        y_t = self.y_transform.transform(y_raw)
        return self.y_scaler.transform(y_t.reshape(-1, 1)).ravel().astype(np.float32)

    def y_s_to_raw(self, y_s: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        y_s = np.asarray(y_s).reshape(-1)
        y_t = self.y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()
        return self.y_transform.inverse_transform(y_t)

    def y_s_stats_to_raw(
        self,
        mu_s: np.ndarray,
        var_s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("GlobalTransformSep is not fitted.")
        mu_s = np.asarray(mu_s).reshape(-1)
        var_s = np.maximum(np.asarray(var_s).reshape(-1), 0.0)

        mu_t = self.y_scaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        std_t = np.sqrt(var_s) * float(self.y_scaler.scale_[0])
        mu_raw = self.y_transform.inverse_transform(mu_t)
        dy_dt = self.y_transform.derivative(mu_t)
        std_raw = std_t * dy_dt
        return mu_raw, std_raw

    @property
    def theta_mu(self) -> float:
        return float(self.theta_scaler.mean_[0])

    @property
    def theta_sd(self) -> float:
        return float(self.theta_scaler.scale_[0])


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64, 32), dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclasses.dataclass
class NNModelTorchStd:
    input_dim: int = 6
    device: Optional[str] = None
    model: Optional[nn.Module] = None
    hidden: Tuple[int, ...] = (128, 64, 32)

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_full_s: np.ndarray,
        y_s: np.ndarray,
        val_frac: float = 0.10,
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 200,
        hidden: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
        weight_decay: float = 1e-6,
        seed: int = 0,
        verbose_every: int = 25,
    ) -> "NNModelTorchStd":
        self.hidden = tuple(hidden)
        dev = self._get_device()

        X_full_s = np.asarray(X_full_s).astype(np.float32)
        y_s = np.asarray(y_s).astype(np.float32).reshape(-1)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full_s,
            y_s,
            test_size=val_frac,
            random_state=seed,
            shuffle=True,
        )

        X_tr_t = torch.from_numpy(X_tr).to(dev, dtype=torch.float32)
        y_tr_t = torch.from_numpy(y_tr).to(dev, dtype=torch.float32)
        X_va_t = torch.from_numpy(X_va).to(dev, dtype=torch.float32)
        y_va_t = torch.from_numpy(y_va).to(dev, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model = MLP(self.input_dim, hidden=self.hidden, dropout=dropout).to(dev).float()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = 30
        bad = 0

        for ep in range(1, epochs + 1):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(X_va_t), y_va_t).item()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if verbose_every and ep % verbose_every == 0:
                print(f"epoch {ep:4d} | val_mse={val_loss:.6f} | best={best_val:.6f}")

            if bad >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_y_s_from_Xfull_s(self, X_full_s: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("NN model is not fitted.")
        dev = self._get_device()
        Xs = np.asarray(X_full_s).astype(np.float32)
        Xt = torch.from_numpy(Xs).to(dev, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(Xt).detach().cpu().numpy()

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Nothing to save.")
        bundle = {
            "state_dict": self.model.state_dict(),
            "hidden": list(self.hidden),
            "input_dim": self.input_dim,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "NNModelTorchStd":
        bundle = joblib.load(path)
        input_dim = int(bundle.get("input_dim", 6))
        hidden = tuple(bundle.get("hidden", [128, 64, 32]))
        obj = cls(input_dim=input_dim, device=device, hidden=hidden)
        obj.model = MLP(input_dim, hidden=hidden, dropout=0.0).to(obj._get_device()).float()
        obj.model.load_state_dict(bundle["state_dict"])
        obj.model.eval()
        return obj


class PlantEmulatorNNStd(Emulator):
    def __init__(self, nn_std: NNModelTorchStd):
        self.nn = nn_std

    def predict(self, x, theta):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)

        x = x.to(torch.float64)
        theta = theta.to(torch.float64)

        if theta.ndim == 2 and theta.shape[1] == 1 and x.ndim == 2 and x.shape[1] == 5:
            if theta.shape[0] == x.shape[0]:
                X_full = torch.cat([x, theta], dim=1)
                mu_np = self.nn.predict_y_s_from_Xfull_s(X_full.cpu().numpy())
                mu = torch.tensor(mu_np, dtype=torch.float64, device=x.device)
                return mu, torch.zeros_like(mu)

            n_particles = theta.shape[0]
            batch_size = x.shape[0]
            x_rep = x.unsqueeze(0).repeat(n_particles, 1, 1)
            th_rep = theta.unsqueeze(1).repeat(1, batch_size, 1)
            X_full = torch.cat([x_rep, th_rep], dim=-1)
            mu_np = self.nn.predict_y_s_from_Xfull_s(X_full.reshape(-1, 6).cpu().numpy())
            mu = torch.tensor(mu_np.reshape(n_particles, batch_size), dtype=torch.float64, device=x.device).T
            return mu, torch.zeros_like(mu)

        raise ValueError(f"Unsupported shapes: x={tuple(x.shape)}, theta={tuple(theta.shape)}")


def batch_X_base_to_s(gt: GlobalTransformSep, Xb: np.ndarray) -> torch.Tensor:
    return torch.tensor(gt.X_base_to_s(Xb).astype(np.float64), dtype=torch.float64)


def batch_y_to_s(gt: GlobalTransformSep, yb: np.ndarray) -> torch.Tensor:
    return torch.tensor(gt.y_raw_to_s(yb).astype(np.float64), dtype=torch.float64)


def theta_prior_sampler_factory(
    gt: GlobalTransformSep,
    theta_bounds_raw: Tuple[float, float] = (3.0, 21.0),
) -> Callable[[int], torch.Tensor]:
    lo_raw, hi_raw = theta_bounds_raw
    lo_s = float(gt.theta_raw_to_s(np.array([lo_raw]))[0])
    hi_s = float(gt.theta_raw_to_s(np.array([hi_raw]))[0])

    def prior_sampler(num_particles: int) -> torch.Tensor:
        return torch.rand(num_particles, 1, dtype=torch.float64) * (hi_s - lo_s) + lo_s

    return prior_sampler


def build_restart_bocpd_calibrator(
    emulator: Emulator,
    prior_sampler: Callable[[int], torch.Tensor],
    gt: GlobalTransformSep,
    device: str = "cpu",
    use_discrepancy: bool = False,
    num_particles: int = 512,
) -> OnlineBayesCalibrator:
    cfg = CalibrationConfig()
    cfg.model.device = device
    cfg.model.dtype = torch.float64
    cfg.model.use_discrepancy = use_discrepancy
    cfg.model.sigma_eps = float(gt.y_scaler.scale_[0])
    cfg.pf.num_particles = int(num_particles)
    cfg.bocpd.bocpd_mode = "restart"
    cfg.bocpd.use_restart = True
    return OnlineBayesCalibrator(cfg, emulator, prior_sampler)


def train_or_load_standardized_emulator(
    data_path: str,
    bundle_dir: str,
    device: Optional[str] = None,
    seed: int = 0,
    epochs: int = 200,
    hidden: Tuple[int, ...] = (128, 64, 32),
) -> Tuple[GlobalTransformSep, NNModelTorchStd, PlantEmulatorNNStd]:
    data_path_obj = Path(data_path)
    bundle_dir_obj = Path(bundle_dir)
    bundle_dir_obj.mkdir(parents=True, exist_ok=True)

    gt_path = bundle_dir_obj / "project2_gt.joblib"
    nn_path = bundle_dir_obj / "project2_nn_std.joblib"

    if gt_path.exists() and nn_path.exists():
        gt = joblib.load(gt_path)
        nn_std = NNModelTorchStd.load(str(nn_path), device=device)
        return gt, nn_std, PlantEmulatorNNStd(nn_std)

    data = np.load(data_path_obj, allow_pickle=True)
    X_base = data["X"]
    y_raw = data["y"]
    theta_raw = data["theta"]

    X_tr, _, y_tr, _, th_tr, _ = train_test_split(
        X_base,
        y_raw,
        theta_raw,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    gt = GlobalTransformSep().fit(X_tr, th_tr, y_tr)
    X_tr_s = gt.X_base_to_s(X_tr)
    th_tr_s = gt.theta_raw_to_s(th_tr).reshape(-1, 1)
    X_full_tr_s = np.concatenate([X_tr_s, th_tr_s], axis=1)
    y_tr_s = gt.y_raw_to_s(y_tr)

    nn_std = NNModelTorchStd(input_dim=6, device=device).fit(
        X_full_tr_s,
        y_tr_s,
        epochs=epochs,
        hidden=hidden,
        seed=seed,
    )

    joblib.dump(gt, gt_path)
    nn_std.save(str(nn_path))
    return gt, nn_std, PlantEmulatorNNStd(nn_std)
