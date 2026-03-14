# stream_factory_physical.py
# Stream batches from:
#   C:\...\PhysicalData_v3\factory_Mode{mode}t{t}.xlsx
#
# Requirements you gave:
# - mode=0: strictly increasing t order (no jump)
# - mode=1: mixed with multiple jumps (4-5 jumps over ~1200 points),
#           jump timing not fixed, can jump forward/backward,
#           and jumps should make CustomerLbd difference "large"
#   CustomerLbd generation function (by t index):
#       CustomerLbd(t) = (11.5 + 8.5*sin(2*pi*t/400))*60
# - CustomerLbd in xlsx is "min:sec(.fraction)" (NO hour)
# - API:
#     stream = StreamClass(mode, folder)
#     X, y, theta = stream.next(batch_size)
#   where X=(B,5) in order [W,R,M1,M2,Q], y=NetRevenue, theta=CustomerLbd

import os
import re
import glob
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


def parse_mmss_to_seconds(val) -> float:
    """
    Parse CustomerLbd formatted as "min:sec(.fraction)" into SECONDS (float).
    Examples:
      "11:38.0107318" -> 11*60 + 38.0107318 = 698.0107318 seconds
      698.01 -> 698.01 (already numeric)
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val).strip()
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan

    m_str, sec_str = s.split(":", 1)
    try:
        m = int(float(m_str))
        sec = float(sec_str)
        return float(m * 60.0 + sec)
    except Exception:
        return np.nan


def lbd_from_t(t: int) -> float:
    """
    Your generation function:
      CustomerLbd := (11.5 + 8.5*sin(2*pi*t/400))*60
    Returns SECONDS.
    """
    return (11.5 + 8.5 * math.sin(2.0 * math.pi * t / 400.0)) * 60.0


@dataclass
class JumpPlan:
    max_jumps: int = 5
    min_gap_theta: float = 400.0   # seconds; "large diff" threshold, tune if needed
    min_interval: int = 180        # min points between jumps
    max_interval: int = 320        # max points between jumps
    min_jump_span: int = 40        # avoid micro-jumps (t index distance)
    max_tries: int = 200
    seed: int = 42


class StreamClass:
    """
    stream = StreamClass(mode, folder)          # 从目录读取多个Excel文件
    stream = StreamClass(mode, csv_path=path)   # 从单个CSV文件读取
    X, y, theta = stream.next(batch_size)

    mode=0: ordered by t
    mode=1: ordered stream with multiple jumps (forward/back), not too frequent,
            with theta-gap constraint based on lbd_from_t(t).
    
    CSV文件格式要求列: t, mode, W, R, M1, M2, Q, NetRevenue, CustomerLbd_min
    """

    def __init__(self, mode: int, folder: str = None, csv_path: str = None, jump_plan: Optional[JumpPlan] = None):
        self.mode = int(mode)
        self.folder = folder
        self.csv_path = csv_path
        self.jump_plan = jump_plan if jump_plan is not None else JumpPlan()
        self.rng = np.random.default_rng(self.jump_plan.seed) if self.jump_plan is not None else np.random.default_rng()
        self._use_jump = jump_plan is not None
        
        self._use_csv = csv_path is not None
        self._csv_data = None

        self._index = self._build_index()  # list of (t, filepath_or_row_idx) sorted by t
        self._n = len(self._index)

        self._pos = 0            # current index position into _index
        self._emitted = 0        # how many samples have been emitted
        self._jumps_done = 0
        self._next_jump_at = self._draw_next_jump_at() if self._use_jump else None

        # Track last emitted t for theta gap calculation
        self._last_t: Optional[int] = None

    def _build_index(self) -> List[Tuple[int, any]]:
        if self._use_csv:
            return self._build_index_from_csv()
        else:
            return self._build_index_from_folder()
    
    def _build_index_from_csv(self) -> List[Tuple[int, int]]:
        """从CSV文件构建索引，返回 (t, row_idx) 列表"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        required = ["t", "mode", "W", "R", "M1", "M2", "Q", "NetRevenue", "CustomerLbd_min"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"CSV missing columns: {miss}. Found: {list(df.columns)}")
        
        df_mode = df[df["mode"] == self.mode].reset_index(drop=True)
        if len(df_mode) == 0:
            raise ValueError(f"No data found for mode={self.mode} in CSV")
        
        self._csv_data = df_mode
        
        items: List[Tuple[int, int]] = []
        for idx, row in df_mode.iterrows():
            t = int(row["t"])
            items.append((t, idx))
        
        items.sort(key=lambda x: x[0])
        return items
    
    def _build_index_from_folder(self) -> List[Tuple[int, str]]:
        """从目录构建索引，返回 (t, filepath) 列表"""
        if self.folder is None:
            raise ValueError("Either folder or csv_path must be provided")
        
        pattern = os.path.join(self.folder, f"factory_Mode{self.mode}t*.xlsx")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No files matched: {pattern}")

        items: List[Tuple[int, str]] = []
        rx = re.compile(rf"factory_Mode{self.mode}t(\d+)\.xlsx$", re.IGNORECASE)
        for fp in files:
            m = rx.search(os.path.basename(fp))
            if m:
                t = int(m.group(1))
                items.append((t, fp))

        if not items:
            raise FileNotFoundError(f"Found files but none matched naming format: factory_Mode{self.mode}t{{t}}.xlsx")

        items.sort(key=lambda x: x[0])
        return items

    def _draw_next_jump_at(self) -> int:
        # schedule next jump in terms of emitted sample count
        interval = int(self.rng.integers(self.jump_plan.min_interval, self.jump_plan.max_interval + 1))
        return self._emitted + interval

    def _maybe_jump(self):
        """
        Jump logic for mode=1:
        - Only consider jumping when emitted >= next_jump_at
        - Limit total jumps to max_jumps (about 4-5 for ~1200 points)
        - Jump can go forward or backward (data can "go back")
        - Prefer large theta diff based on lbd_from_t(t)
        """
        if not self._use_jump:
            return
        if self._jumps_done >= self.jump_plan.max_jumps:
            return
        if self._emitted < (self._next_jump_at or 10**18):
            return
        if self._last_t is None:
            return
        if self._n <= 1:
            return

        cur_pos = self._pos
        cur_t = self._last_t
        cur_theta = lbd_from_t(cur_t)

        best = None
        best_gap = -1.0

        for _ in range(self.jump_plan.max_tries):
            # allow back & forth; sample a candidate position anywhere
            cand_pos = int(self.rng.integers(0, self._n))
            if cand_pos == cur_pos:
                continue

            cand_t = self._index[cand_pos][0]
            # avoid trivial tiny jumps
            if abs(cand_t - cur_t) < self.jump_plan.min_jump_span:
                continue

            cand_theta = lbd_from_t(cand_t)
            gap = abs(cand_theta - cur_theta)

            # keep best fallback
            if gap > best_gap:
                best_gap = gap
                best = cand_pos

            # accept if theta gap large enough
            if gap >= self.jump_plan.min_gap_theta:
                self._pos = cand_pos
                self._jumps_done += 1
                self._next_jump_at = self._draw_next_jump_at()
                return

        # fallback: if no candidate meets threshold, still jump to the best we found
        if best is not None:
            self._pos = best
            self._jumps_done += 1
            self._next_jump_at = self._draw_next_jump_at()

    def _read_one_sample_csv(self, row_idx: int):
        """从CSV读取单个样本"""
        row = self._csv_data.iloc[row_idx]
        W = int(row["W"])
        R = int(row["R"])
        M1 = int(row["M1"])
        M2 = int(row["M2"])
        Q = int(row["Q"])
        y = float(row["NetRevenue"])
        theta_min = float(row["CustomerLbd_min"])
        theta_sec = theta_min * 60.0
        return W, R, M1, M2, Q, y, theta_sec
    
    def _read_one_sample_excel(self, fp: str):
        """从Excel读取单个样本"""
        df = pd.read_excel(fp)
        if df.shape[0] < 1:
            return None

        df.columns = [str(c).strip() for c in df.columns]
        required = ["CustomerLbd", "NetRevenue", "W", "R", "M1", "M2", "Q"]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Missing columns {miss} in {fp}. Found={list(df.columns)}")

        r0 = df.iloc[0]
        theta = parse_mmss_to_seconds(r0["CustomerLbd"])
        y = float(pd.to_numeric(r0["NetRevenue"], errors="coerce"))

        W  = int(pd.to_numeric(r0["W"],  errors="coerce"))
        R  = int(pd.to_numeric(r0["R"],  errors="coerce"))
        M1 = int(pd.to_numeric(r0["M1"], errors="coerce"))
        M2 = int(pd.to_numeric(r0["M2"], errors="coerce"))
        Q  = int(pd.to_numeric(r0["Q"],  errors="coerce"))

        if np.isnan(theta) or np.isnan(y):
            return None
        
        return W, R, M1, M2, Q, y, theta

    def next(self, batch_size: int):
        """
        Returns:
          X: (B,5) int64 [W,R,M1,M2,Q]
          y: (B,) float64 NetRevenue
          theta: (B,) float64 CustomerLbd (minutes)
        Raises StopIteration if no more data can be read.
        """
        B = int(batch_size)
        if B <= 0:
            raise ValueError("batch_size must be positive")

        if self._pos >= self._n:
            raise StopIteration

        rows = []
        while len(rows) < B:
            if self._pos >= self._n:
                break

            self._maybe_jump()

            t, ref = self._index[self._pos]
            self._pos += 1

            if self._use_csv:
                sample = self._read_one_sample_csv(ref)
            else:
                sample = self._read_one_sample_excel(ref)
            
            if sample is None:
                continue

            rows.append(sample)
            self._emitted += 1
            self._last_t = t

        if not rows:
            raise StopIteration

        arr = np.array(rows, dtype=np.float64)  # (B,7)
        X = arr[:, 0:5].astype(np.int64)
        y = arr[:, 5].astype(np.float64)
        theta = arr[:, 6].astype(np.float64)    # seconds
        return X, y, theta/60

    def reset(self, start_t: Optional[int] = None):
        """
        Reset stream. Optionally start at first t >= start_t (still mode-dependent).
        """
        self._pos = 0
        self._emitted = 0
        self._jumps_done = 0
        self._last_t = None
        self._next_jump_at = self._draw_next_jump_at() if self._use_jump else None

        if start_t is not None:
            ts = [t for t, _ in self._index]
            # lower_bound
            lo, hi = 0, len(ts)
            while lo < hi:
                mid = (lo + hi) // 2
                if ts[mid] < start_t:
                    lo = mid + 1
                else:
                    hi = mid
            self._pos = lo


import dataclasses
import numpy as np
import torch
import gpytorch
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1) y transform: signed log1p
# -----------------------------
@dataclasses.dataclass
class SignedLog1pTransformer:
    c: float = None  # scale

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1)
        abs_y = np.abs(y)
        c = np.median(abs_y[abs_y > 0]) if np.any(abs_y > 0) else 1.0
        self.c = float(c)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted: c is None")
        y = np.asarray(y).reshape(-1)
        return np.sign(y) * np.log1p(np.abs(y) / self.c)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.c is None:
            raise ValueError("SignedLog1pTransformer not fitted: c is None")
        z = np.asarray(z).reshape(-1)
        return np.sign(z) * self.c * np.expm1(np.abs(z))


# -----------------------------
# 2) Scaling: X scaler + y scaler (on transformed y)
# -----------------------------
@dataclasses.dataclass
class XYScaler:
    x_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)
    y_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)

    def fit(self, X: np.ndarray, y_t: np.ndarray):
        X = np.asarray(X)
        y_t = np.asarray(y_t).reshape(-1)
        self.x_scaler.fit(X)
        self.y_scaler.fit(y_t.reshape(-1, 1))
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.x_scaler.transform(np.asarray(X)).astype(np.float32)

    def transform_y(self, y_t: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(np.asarray(y_t).reshape(-1, 1)).ravel().astype(np.float32)

    def inverse_transform_y(self, y_s: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(np.asarray(y_s).reshape(-1, 1)).ravel()

    @property
    def y_scale(self) -> float:
        return float(self.y_scaler.scale_[0])


# -----------------------------
# 3) GP: gpytorch model wrapper
# -----------------------------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim: int):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )


@dataclasses.dataclass
class GPModelTorch:
    input_dim: int = 6
    device: str = None

    # components
    y_transform: SignedLog1pTransformer = dataclasses.field(default_factory=SignedLog1pTransformer)
    scaler: XYScaler = dataclasses.field(default_factory=XYScaler)
    likelihood: gpytorch.likelihoods.GaussianLikelihood = None
    model: ExactGPModel = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_full: np.ndarray, y_raw: np.ndarray, noise_init=1e-4, lr=0.05, training_iter=300):
        """
        X_full: (N, input_dim) already includes theta column(s)
        y_raw:  (N,) raw y (original revenue, can be negative)
        """
        dev = self._get_device()

        # 1) y transform
        self.y_transform.fit(y_raw)
        y_t = self.y_transform.transform(y_raw)

        # 2) scalers
        self.scaler.fit(X_full, y_t)

        X_s = self.scaler.transform_X(X_full)
        y_s = self.scaler.transform_y(y_t)

        train_x = torch.from_numpy(X_s).to(dev)
        train_y = torch.from_numpy(y_s).to(dev)

        # 3) GP + likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
        ).to(dev)
        self.likelihood.initialize(noise=noise_init)

        self.model = ExactGPModel(train_x, train_y, self.likelihood, self.input_dim).to(dev)

        # train
        self.model.train()
        self.likelihood.train()

        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        # de-dup
        seen, uniq = set(), []
        for p in params:
            if id(p) not in seen:
                uniq.append(p)
                seen.add(id(p))

        optimizer = torch.optim.Adam(uniq, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iter):
            optimizer.zero_grad()
            out = self.model(train_x)
            loss = -mll(out, train_y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()
        return self

    def predict_single(self, X_base: np.ndarray, theta: np.ndarray, return_std=False, include_noise=False):
        """
        One-to-one prediction.
        X_base: (N,5) or (5,)
        theta : scalar or (N,) or (N,1)
        Returns mean/std in ORIGINAL y space, shape (N,).
        """
        if self.model is None or self.likelihood is None:
            raise ValueError("Model not fitted/loaded.")

        X_base = np.atleast_2d(X_base)
        N = X_base.shape[0]

        theta = np.asarray(theta)
        if theta.ndim == 2 and theta.shape[1] == 1:
            theta = theta.reshape(-1)
        theta = np.asarray(theta).reshape(-1)

        if theta.size == 1:
            theta = np.repeat(theta.item(), N)
        if theta.size != N:
            raise ValueError(f"predict_single expects theta size 1 or N. Got theta={theta.size}, N={N}")

        X_full = np.column_stack([X_base, theta])  # (N,6)
        Xs = self.scaler.transform_X(X_full)

        dev = next(self.model.parameters()).device
        test_x = torch.from_numpy(Xs).to(dev)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(test_x)) if include_noise else self.model(test_x)
            mean_s = dist.mean.detach().cpu().numpy()
            std_s  = dist.variance.sqrt().detach().cpu().numpy()

        mean_t = self.scaler.inverse_transform_y(mean_s)
        std_t  = std_s * self.scaler.y_scale

        mean_y = self.y_transform.inverse_transform(mean_t)
        dy_dt = self.y_transform.c * np.exp(np.abs(mean_t))
        std_y = std_t * dy_dt

        if return_std:
            return mean_y, std_y
        return mean_y


    def predict_grid(
        self,
        X_base: np.ndarray,
        theta: np.ndarray,
        return_std: bool = True,
        include_noise: bool = False,
        return_torch: bool = False,
        transpose: bool = False,
    ):
        """
        Grid prediction for batches x particles.

        Inputs:
        X_base: (b,5)
        theta : (M,) or (M,1) or scalar

        Outputs:
        mean: (b,M) by default
        std : (b,M) if return_std=True
        If transpose=True: (M,b)
        """
        if self.model is None or self.likelihood is None:
            raise ValueError("Model not fitted/loaded.")

        X_base = np.atleast_2d(X_base)
        b = X_base.shape[0]

        theta = np.asarray(theta)
        if theta.ndim == 2 and theta.shape[1] == 1:
            theta = theta.reshape(-1)
        theta = np.asarray(theta).reshape(-1)

        if theta.size == 1:
            theta = np.repeat(theta.item(), 1)
        M = theta.shape[0]

        X_rep = np.repeat(X_base, M, axis=0)              # (b*M,5)
        theta_tile = np.tile(theta, reps=b).reshape(-1)   # (b*M,)

        X_full = np.column_stack([X_rep, theta_tile])     # (b*M,6)
        Xs = self.scaler.transform_X(X_full)

        dev = next(self.model.parameters()).device
        test_x = torch.from_numpy(Xs).to(dev)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = self.likelihood(self.model(test_x)) if include_noise else self.model(test_x)
            mean_s = dist.mean.detach().cpu().numpy()
            std_s  = dist.variance.sqrt().detach().cpu().numpy()

        mean_t = self.scaler.inverse_transform_y(mean_s)
        std_t  = std_s * self.scaler.y_scale

        mean_y = self.y_transform.inverse_transform(mean_t)
        dy_dt = self.y_transform.c * np.exp(np.abs(mean_t))
        std_y = std_t * dy_dt

        mean_y = mean_y.reshape(b, M)
        std_y  = std_y.reshape(b, M)

        if transpose:
            mean_y = mean_y.T
            std_y  = std_y.T

        if return_torch:
            mean_y = torch.from_numpy(mean_y).float()
            std_y  = torch.from_numpy(std_y).float()

        if return_std:
            return mean_y, std_y
        return mean_y


    def predict(
        self,
        X_base: np.ndarray,
        theta: np.ndarray,
        return_std: bool = True,
        include_noise: bool = False,
        return_torch: bool = False,
        transpose: bool = False,
        force_grid: bool = False,
    ):
        """
        Auto-dispatch between predict_single and predict_grid based on shapes.

        Rules:
        - If force_grid=True: always use grid.
        - Let b = number of rows in X_base after atleast_2d.
        - theta scalar or size==1:
            -> single (returns (b,)) unless force_grid=True.
        - theta size == b:
            -> single (one-to-one) unless force_grid=True.
        - theta size != b:
            -> grid (returns (b,M)).
        """
        Xb = np.atleast_2d(X_base)
        b = Xb.shape[0]

        th = np.asarray(theta)
        if th.ndim == 2 and th.shape[1] == 1:
            th = th.reshape(-1)
        th = th.reshape(-1)

        if force_grid:
            return self.predict_grid(
                Xb, th, return_std=return_std, include_noise=include_noise,
                return_torch=return_torch, transpose=transpose
            )

        if th.size <= 1:
            # default to single: output (b,)
            return self.predict_single(Xb, th, return_std=return_std, include_noise=include_noise)

        if th.size == b:
            return self.predict_single(Xb, th, return_std=return_std, include_noise=include_noise)

        # otherwise treat as particles grid
        return self.predict_grid(
            Xb, th, return_std=return_std, include_noise=include_noise,
            return_torch=return_torch, transpose=transpose
        )

    def save(self, path: str):
        if self.model is None or self.likelihood is None:
            raise ValueError("Nothing to save (model not fitted).")
        bundle = {
            "state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
            "x_scaler": self.scaler.x_scaler,
            "y_scaler": self.scaler.y_scaler,
            "c": self.y_transform.c,
            "meta": {"input_dim": self.input_dim, "kernel": "ScaleKernel(Matern nu=2.5, ARD)"},
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        bundle = joblib.load(path)
        input_dim = int(bundle.get("meta", {}).get("input_dim", 6))
        obj = cls(input_dim=input_dim, device=device)

        # restore scalers + y transform
        obj.scaler.x_scaler = bundle["x_scaler"]
        obj.scaler.y_scaler = bundle["y_scaler"]
        obj.y_transform.c = float(bundle["c"])

        dev = obj._get_device()
        # dummy train data for ExactGP init
        dummy_x = torch.zeros(1, input_dim, dtype=torch.float32).to(dev)
        dummy_y = torch.zeros(1, dtype=torch.float32).to(dev)

        obj.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
        ).to(dev)

        obj.model = ExactGPModel(dummy_x, dummy_y, obj.likelihood, input_dim).to(dev)

        obj.model.load_state_dict(bundle["state_dict"])
        obj.likelihood.load_state_dict(bundle["likelihood_state_dict"])

        obj.model.eval()
        obj.likelihood.eval()
        return obj

import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1) y transform: signed log1p
# -----------------------------
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


# -----------------------------
# 2) Scaling: X scaler + y scaler (on transformed y)
# -----------------------------
@dataclasses.dataclass
class XYScaler:
    x_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)
    y_scaler: StandardScaler = dataclasses.field(default_factory=StandardScaler)

    def fit(self, X: np.ndarray, y_t: np.ndarray):
        X = np.asarray(X)
        y_t = np.asarray(y_t).reshape(-1)
        self.x_scaler.fit(X)
        self.y_scaler.fit(y_t.reshape(-1, 1))
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return self.x_scaler.transform(np.asarray(X)).astype(np.float32)

    def transform_y(self, y_t: np.ndarray) -> np.ndarray:
        return self.y_scaler.transform(np.asarray(y_t).reshape(-1, 1)).ravel().astype(np.float32)

    def inverse_transform_y(self, y_s: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(np.asarray(y_s).reshape(-1, 1)).ravel()

    @property
    def y_scale(self) -> float:
        return float(self.y_scaler.scale_[0])


# -----------------------------
# 3) MLP model
# -----------------------------
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


@dataclasses.dataclass
class NNModelTorch:
    input_dim: int = 6
    device: str = None

    y_transform: SignedLog1pTransformer = dataclasses.field(default_factory=SignedLog1pTransformer)
    scaler: XYScaler = dataclasses.field(default_factory=XYScaler)

    model: nn.Module = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_full: np.ndarray,
        y_raw: np.ndarray,
        val_frac: float = 0.10,
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 200,
        hidden=(128, 128, 64),
        dropout: float = 0.0,
        weight_decay: float = 1e-6,
        seed: int = 0,
        verbose_every: int = 20,
    ):
        rng = np.random.default_rng(seed)
        dev = self._get_device()

        X_full = np.asarray(X_full)
        y_raw = np.asarray(y_raw).reshape(-1)

        # 1) y transform
        self.y_transform.fit(y_raw)
        y_t = self.y_transform.transform(y_raw)

        # 2) split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full, y_t, test_size=val_frac, random_state=seed, shuffle=True
        )

        # 3) scalers (fit on train only)
        self.scaler.fit(X_tr, y_tr)

        X_tr_s = self.scaler.transform_X(X_tr)
        X_va_s = self.scaler.transform_X(X_va)

        y_tr_s = self.scaler.transform_y(y_tr)
        y_va_s = self.scaler.transform_y(y_va)

        # torch data
        X_tr_t = torch.from_numpy(X_tr_s).to(dev)
        y_tr_t = torch.from_numpy(y_tr_s).to(dev)
        X_va_t = torch.from_numpy(X_va_s).to(dev)
        y_va_t = torch.from_numpy(y_va_s).to(dev)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        # 4) model
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

            # val
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

        # final metrics in BOTH spaces (y_t and raw y)
        self.model.eval()
        with torch.no_grad():
            y_va_pred_s = self.model(X_va_t).detach().cpu().numpy()

        y_va_pred_t = self.scaler.inverse_transform_y(y_va_pred_s)  # back to y_t
        y_va_pred_raw = self.y_transform.inverse_transform(y_va_pred_t)
        y_va_raw = self.y_transform.inverse_transform(y_va)

        err_raw = y_va_pred_raw - y_va_raw
        rmse_raw = float(np.sqrt(np.mean(err_raw ** 2)))
        mae_raw = float(np.mean(np.abs(err_raw)))

        eps = 1e-12
        smape = float(np.mean(2*np.abs(err_raw) / (np.abs(y_va_raw) + np.abs(y_va_pred_raw) + eps)))
        rel_mae = float(np.mean(np.abs(err_raw) / (np.maximum(np.abs(y_va_raw), eps))))

        print("\nNN validation (RAW y space)")
        print("RMSE:", rmse_raw)
        print("MAE :", mae_raw)
        print("sMAPE:", smape, f"({100*smape:.2f}%)")
        print("Rel-MAE:", rel_mae, f"({100*rel_mae:.2f}%)")

        return self

    def _predict_y_t_from_Xfull(self, X_full: np.ndarray) -> np.ndarray:
        """Return prediction in y_t space (after inverse y_scaler, before inverse signed-log)."""
        if self.model is None:
            raise ValueError("NN model not fitted/loaded.")
        dev = self._get_device()

        Xs = self.scaler.transform_X(X_full)
        Xt = torch.from_numpy(Xs).to(dev)

        self.model.eval()
        with torch.no_grad():
            y_s = self.model(Xt).detach().cpu().numpy()

        y_t = self.scaler.inverse_transform_y(y_s)
        return y_t

    def predict_single(self, X_base: np.ndarray, theta: np.ndarray):
        """
        X_base: (N,5)
        theta : scalar or (N,) or (N,1)
        Returns mean_y in RAW y space, shape (N,).
        """
        X_base = np.atleast_2d(X_base)
        N = X_base.shape[0]

        th = np.asarray(theta)
        if th.ndim == 2 and th.shape[1] == 1:
            th = th.reshape(-1)
        th = th.reshape(-1)

        if th.size == 1:
            th = np.repeat(th.item(), N)
        if th.size != N:
            raise ValueError(f"predict_single expects theta size 1 or N. Got {th.size} vs N={N}")

        X_full = np.column_stack([X_base, th])
        y_t = self._predict_y_t_from_Xfull(X_full)
        y_raw = self.y_transform.inverse_transform(y_t)
        return y_raw

    def predict_grid(self, X_base: np.ndarray, theta: np.ndarray, transpose: bool = False):
        """
        X_base: (b,5)
        theta : (M,) or (M,1) or scalar
        Returns mean_y in RAW y space, shape (b,M) by default.
        """
        X_base = np.atleast_2d(X_base)
        b = X_base.shape[0]

        th = np.asarray(theta)
        if th.ndim == 2 and th.shape[1] == 1:
            th = th.reshape(-1)
        th = th.reshape(-1)
        if th.size == 1:
            th = np.repeat(th.item(), 1)
        M = th.size

        X_rep = np.repeat(X_base, M, axis=0)              # (b*M,5)
        th_tile = np.tile(th, reps=b).reshape(-1)         # (b*M,)
        X_full = np.column_stack([X_rep, th_tile])        # (b*M,6)

        y_t = self._predict_y_t_from_Xfull(X_full)        # (b*M,)
        y_raw = self.y_transform.inverse_transform(y_t)   # (b*M,)

        y_raw = y_raw.reshape(b, M)
        if transpose:
            y_raw = y_raw.T
        return y_raw

    def predict(self, X_base: np.ndarray, theta: np.ndarray, force_grid: bool = False, transpose: bool = False):
        Xb = np.atleast_2d(X_base)
        b = Xb.shape[0]

        th = np.asarray(theta)
        if th.ndim == 2 and th.shape[1] == 1:
            th = th.reshape(-1)
        th = th.reshape(-1)

        if force_grid:
            return self.predict_grid(Xb, th, transpose=transpose)

        if th.size <= 1:
            return self.predict_single(Xb, th)
        if th.size == b:
            return self.predict_single(Xb, th)
        return self.predict_grid(Xb, th, transpose=transpose)

    def save(self, path: str):
        if self.model is None:
            raise ValueError("Nothing to save.")
        bundle = {
            "state_dict": self.model.state_dict(),
            "x_scaler": self.scaler.x_scaler,
            "y_scaler": self.scaler.y_scaler,
            "c": self.y_transform.c,
            "meta": {
                "input_dim": self.input_dim,
                "hidden": [m.out_features for m in self.model.net if isinstance(m, nn.Linear)][:-1],
            },
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        bundle = joblib.load(path)
        input_dim = int(bundle["meta"].get("input_dim", 6))
        obj = cls(input_dim=input_dim, device=device)

        obj.scaler.x_scaler = bundle["x_scaler"]
        obj.scaler.y_scaler = bundle["y_scaler"]
        obj.y_transform.c = float(bundle["c"])

        # reconstruct model
        hidden = bundle["meta"].get("hidden", [128, 128, 64])
        obj.model = MLP(input_dim, hidden=tuple(hidden), dropout=0.0).to(obj._get_device())
        obj.model.load_state_dict(bundle["state_dict"])
        obj.model.eval()
        return obj

# stream_factory_physical.py
# Stream batches from:
#   C:\...\PhysicalData_v3\factory_Mode{mode}t{t}.xlsx
#
# Requirements you gave:
# - mode=0: strictly increasing t order (no jump)
# - mode=1: mixed with multiple jumps (4-5 jumps over ~1200 points),
#           jump timing not fixed, can jump forward/backward,
#           and jumps should make CustomerLbd difference "large"
#   CustomerLbd generation function (by t index):
#       CustomerLbd(t) = (11.5 + 8.5*sin(2*pi*t/400))*60
# - CustomerLbd in xlsx is "min:sec(.fraction)" (NO hour)
# - API:
#     stream = StreamClass(mode, folder)
#     X, y, theta = stream.next(batch_size)
#   where X=(B,5) in order [W,R,M1,M2,Q], y=NetRevenue, theta=CustomerLbd

import os
import re
import glob
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional


def parse_mmss_to_seconds(val) -> float:
    """
    Parse CustomerLbd formatted as "min:sec(.fraction)" into SECONDS (float).
    Examples:
      "11:38.0107318" -> 11*60 + 38.0107318 = 698.0107318 seconds
      698.01 -> 698.01 (already numeric)
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val).strip()
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan

    m_str, sec_str = s.split(":", 1)
    try:
        m = int(float(m_str))
        sec = float(sec_str)
        return float(m * 60.0 + sec)
    except Exception:
        return np.nan


def lbd_from_t(t: int) -> float:
    """
    Your generation function:
      CustomerLbd := (11.5 + 8.5*sin(2*pi*t/400))*60
    Returns SECONDS.
    """
    return (11.5 + 8.5 * math.sin(2.0 * math.pi * t / 400.0)) * 60.0


# @dataclass
# class JumpPlan:
#     max_jumps: int = 5
#     min_gap_theta: float = 400.0   # seconds; "large diff" threshold, tune if needed
#     min_interval: int = 180        # min points between jumps
#     max_interval: int = 320        # max points between jumps
#     min_jump_span: int = 40        # avoid micro-jumps (t index distance)
#     max_tries: int = 200
#     seed: int = 42


# class StreamClass:
#     """
#     stream = StreamClass(mode, folder)
#     X, y, theta = stream.next(batch_size)

#     mode=0: ordered by t
#     mode=1: ordered stream with multiple jumps (forward/back), not too frequent,
#             with theta-gap constraint based on lbd_from_t(t).
#     """

#     def __init__(self, mode: int, folder: str, jump_plan: Optional[JumpPlan] = None):
#         self.mode = int(mode)
#         self.folder = folder
#         self.jump_plan = jump_plan 
#         self._use_jump = jump_plan is not None
#         self.rng = np.random.default_rng(self.jump_plan.seed)

#         self._index = self._build_index()  # list of (t, filepath) sorted by t
#         self._n = len(self._index)

#         self._pos = 0            # current index position into _index
#         self._emitted = 0        # how many samples have been emitted
#         self._jumps_done = 0
#         self._next_jump_at = self._draw_next_jump_at() if self._use_jump else None

#         # Track last emitted t for theta gap calculation
#         self._last_t: Optional[int] = None

#     def _build_index(self) -> List[Tuple[int, str]]:
#         pattern = os.path.join(self.folder, f"factory_Mode{self.mode}t*.xlsx")
#         files = glob.glob(pattern)
#         if not files:
#             raise FileNotFoundError(f"No files matched: {pattern}")

#         items: List[Tuple[int, str]] = []
#         rx = re.compile(rf"factory_Mode{self.mode}t(\d+)\.xlsx$", re.IGNORECASE)
#         for fp in files:
#             m = rx.search(os.path.basename(fp))
#             if m:
#                 t = int(m.group(1))
#                 items.append((t, fp))

#         if not items:
#             raise FileNotFoundError(f"Found files but none matched naming format: factory_Mode{self.mode}t{{t}}.xlsx")

#         items.sort(key=lambda x: x[0])
#         return items

#     def _draw_next_jump_at(self) -> int:
#         # schedule next jump in terms of emitted sample count
#         if not self._use_jump:
#             raise RuntimeError("Jump plan is not used")
#         interval = int(self.rng.integers(self.jump_plan.min_interval, self.jump_plan.max_interval + 1))
#         return self._emitted + interval

#     def _maybe_jump(self):
#         """
#         Jump logic for mode=1:
#         - Only consider jumping when emitted >= next_jump_at
#         - Limit total jumps to max_jumps (about 4-5 for ~1200 points)
#         - Jump can go forward or backward (data can "go back")
#         - Prefer large theta diff based on lbd_from_t(t)
#         """
#         if not self._use_jump:
#             return
#         if self._jumps_done >= self.jump_plan.max_jumps:
#             return
#         if self._emitted < (self._next_jump_at or 10**18):
#             return
#         if self._last_t is None:
#             return
#         if self._n <= 1:
#             return

#         cur_pos = self._pos
#         cur_t = self._last_t
#         cur_theta = lbd_from_t(cur_t)

#         best = None
#         best_gap = -1.0

#         for _ in range(self.jump_plan.max_tries):
#             # allow back & forth; sample a candidate position anywhere
#             cand_pos = int(self.rng.integers(0, self._n))
#             if cand_pos == cur_pos:
#                 continue

#             cand_t = self._index[cand_pos][0]
#             # avoid trivial tiny jumps
#             if abs(cand_t - cur_t) < self.jump_plan.min_jump_span:
#                 continue

#             cand_theta = lbd_from_t(cand_t)
#             gap = abs(cand_theta - cur_theta)

#             # keep best fallback
#             if gap > best_gap:
#                 best_gap = gap
#                 best = cand_pos

#             # accept if theta gap large enough
#             if gap >= self.jump_plan.min_gap_theta:
#                 self._pos = cand_pos
#                 self._jumps_done += 1
#                 self._next_jump_at = self._draw_next_jump_at()
#                 return

#         # fallback: if no candidate meets threshold, still jump to the best we found
#         if best is not None:
#             self._pos = best
#             self._jumps_done += 1
#             self._next_jump_at = self._draw_next_jump_at()

#     def next(self, batch_size: int):
#         """
#         Returns:
#           X: (B,5) int64 [W,R,M1,M2,Q]
#           y: (B,) float64 NetRevenue
#           theta: (B,) float64 CustomerLbd (SECONDS)
#         Raises StopIteration if no more data can be read.
#         """
#         B = int(batch_size)
#         if B <= 0:
#             raise ValueError("batch_size must be positive")

#         if self._pos >= self._n:
#             raise StopIteration

#         rows = []
#         while len(rows) < B:
#             if self._pos >= self._n:
#                 break

#             # jump is checked BETWEEN draws, so it can cut segments like:
#             # [0:200] [250:450] [180:320] ...
#             self._maybe_jump()

#             t, fp = self._index[self._pos]
#             self._pos += 1

#             df = pd.read_excel(fp)
#             if df.shape[0] < 1:
#                 continue

#             df.columns = [str(c).strip() for c in df.columns]
#             required = ["CustomerLbd", "NetRevenue", "W", "R", "M1", "M2", "Q"]
#             miss = [c for c in required if c not in df.columns]
#             if miss:
#                 raise ValueError(f"Missing columns {miss} in {fp}. Found={list(df.columns)}")

#             r0 = df.iloc[0]
#             theta = parse_mmss_to_seconds(r0["CustomerLbd"])
#             y = float(pd.to_numeric(r0["NetRevenue"], errors="coerce"))

#             W  = int(pd.to_numeric(r0["W"],  errors="coerce"))
#             R  = int(pd.to_numeric(r0["R"],  errors="coerce"))
#             M1 = int(pd.to_numeric(r0["M1"], errors="coerce"))
#             M2 = int(pd.to_numeric(r0["M2"], errors="coerce"))
#             Q  = int(pd.to_numeric(r0["Q"],  errors="coerce"))

#             if np.isnan(theta) or np.isnan(y):
#                 continue

#             rows.append((W, R, M1, M2, Q, y, theta))
#             self._emitted += 1
#             self._last_t = t

#         if not rows:
#             raise StopIteration

#         arr = np.array(rows, dtype=np.float64)  # (B,7)
#         X = arr[:, 0:5].astype(np.int64)
#         y = arr[:, 5].astype(np.float64)
#         theta = arr[:, 6].astype(np.float64)    # seconds
#         return X, y, theta/60

#     def reset(self, start_t: Optional[int] = None):
#         """
#         Reset stream. Optionally start at first t >= start_t (still mode-dependent).
#         """
#         self._pos = 0
#         self._emitted = 0
#         self._jumps_done = 0
#         self._last_t = None
#         self._next_jump_at = self._draw_next_jump_at() if self._use_jump else None

#         if start_t is not None:
#             ts = [t for t, _ in self._index]
#             # lower_bound
#             lo, hi = 0, len(ts)
#             while lo < hi:
#                 mid = (lo + hi) // 2
#                 if ts[mid] < start_t:
#                     lo = mid + 1
#                 else:
#                     hi = mid
#             self._pos = lo


if __name__ == "__main__":
    # aggregate_factory_data.py
    # Collect factory_t{t}.xlsx files into:
    #   X     = (N,5) with columns [W, R, M1, M2, Q]
    #   y     = (N,)  NetRevenue
    #   theta = (N,)  CustomerLbd
    #
    # Usage:
    #   python aggregate_factory_data.py
    #
    # Output:
    #   - factory_aggregated.npz  (contains X, y, theta, file_ids)
    #   - factory_aggregated.csv  (optional inspection)

    import os
    import re
    import glob
    import numpy as np
    import pandas as pd


    DATA_DIR = r"C:\Users\yxu59\files\winter2026\park\simulation\ComputerData_v3"
    folder = "C:\\Users\\yxu59\\files\\winter2026\\park\\simulation\\ComputerData_v3"
    PATTERN = "factory_t*.xlsx"


    def parse_customer_lbd(val):
        """
        Robustly parse CustomerLbd from Excel.
        Handles:
        - numeric (int/float)
        - strings like "6:00" or "06:00:00"
        - pandas Timestamp/Timedelta/time-like objects
        Returns float.
        Default convention for "H:MM(:SS)" is minutes = H*60 + MM + SS/60.
        If you want hours instead, change the return to hours.
        """
        if pd.isna(val):
            return np.nan

        # Already numeric
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        # pandas timedelta
        if isinstance(val, pd.Timedelta):
            return val.total_seconds() / 60.0  # minutes

        # datetime/time-like
        if hasattr(val, "hour") and hasattr(val, "minute"):
            sec = getattr(val, "second", 0)
            return float(val.hour * 60 + val.minute + sec / 60.0)  # minutes

        s = str(val).strip()

        # Try parse "H:MM" or "H:MM:SS"
        if ":" in s:
            parts = s.split(":")
            try:
                h = 0
                m = int(parts[0])
                sec = int(parts[1]) if len(parts) >= 2 else 0
                # sec = int(parts[2]) if len(parts) >= 3 else 0
                return float(h * 60 + m + sec / 60.0)  # minutes
            except Exception:
                pass

        # Fallback: try float
        try:
            return float(s)
        except Exception:
            return np.nan



    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(DATA_DIR, PATTERN)}")

    rows = []
    file_ids = []

    # Accept some common header variants
    col_map = {
        "W": "W",
        "R": "R",
        "M1": "M1",
        "M2": "M2",
        "Q": "Q",
        "NetRevenue": "NetRevenue",
        "CustomerLbd": "CustomerLbd",
    }

    for fp in files:
        df = pd.read_excel(fp)

        # If there are multiple rows, keep them all; if exactly one row (your case), it’s fine.
        # Normalize column names (strip spaces)
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in col_map if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in file: {fp}. Columns found: {list(df.columns)}")

        # Extract numeric columns
        sub = df[list(col_map.keys())].copy()

        # CustomerLbd parsing
        sub["CustomerLbd"] = sub["CustomerLbd"].apply(parse_customer_lbd)

        # Force numeric types for X,y
        for c in ["W", "R", "M1", "M2", "Q", "NetRevenue"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        # Attach file id (t) if available from filename
        m = re.search(r"factory_t(\d+)\.xlsx$", os.path.basename(fp), flags=re.IGNORECASE)
        fid = int(m.group(1)) if m else None

        # Append rows
        rows.append(sub)
        file_ids.extend([fid] * len(sub))

    all_df = pd.concat(rows, ignore_index=True)

    # Drop any rows with missing essentials
    all_df = all_df.dropna(subset=["W", "R", "M1", "M2", "Q", "NetRevenue", "CustomerLbd"]).reset_index(drop=True)

    X = all_df[["W", "R", "M1", "M2", "Q"]].to_numpy(dtype=np.float64)   # (N,5)
    y = all_df["NetRevenue"].to_numpy(dtype=np.float64)                  # (N,)
    theta = all_df["CustomerLbd"].to_numpy(dtype=np.float64)             # (N,)

    # Save outputs
    out_npz = os.path.join(DATA_DIR, "factory_aggregated.npz")
    np.savez(out_npz, X=X, y=y, theta=theta, file_ids=np.array(file_ids[: len(all_df)], dtype=object))

    out_csv = os.path.join(DATA_DIR, "factory_aggregated.csv")
    all_df.assign(file_id=file_ids[: len(all_df)]).to_csv(out_csv, index=False)

    print("Done.")
    print("Files read:", len(files))
    print("Rows kept:", len(all_df))
    print("X shape:", X.shape, "y shape:", y.shape, "theta shape:", theta.shape)
    print("Saved:", out_npz)
    print("Saved:", out_csv)
    print("CustomerLbd parse note: '6:00' -> 360.0 (minutes). Adjust parse_customer_lbd if you want a different unit.")

    import numpy as np
    import torch
    import gpytorch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    # -----------------------------
    # Assumptions:
    #   X: (N,5) numpy array
    #   theta: (N,) or (N,1) numpy array
    #   y: (N,) numpy array
    #   folder: string path to save (e.g., "./" or "C:/.../")
    # -----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # ============ 1) Build full input ============
    theta_1d = np.asarray(theta).reshape(-1)               # force (N,)
    X_full = np.column_stack([np.asarray(X), theta_1d])    # (N,6)
    y_vec = np.asarray(y).reshape(-1)                      # (N,)
    abs_y = np.abs(y_vec)
    c = np.median(abs_y[abs_y > 0]) if np.any(abs_y > 0) else 1.0  # 稳健尺度
    c = float(c)

    y_t = np.sign(y_vec) * np.log1p(abs_y / c)

    # 之后对 y_t 做 StandardScaler -> y_train_s
    # 预测完 inverse scaler 得到 y_t_pred 后再 inverse signed log：
    def signed_log1p_inverse(z, c):
        return np.sign(z) * c * np.expm1(np.abs(z))

    # ============ 2) Train/Val split (10% val) ============
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_t, test_size=0.10, random_state=0, shuffle=True
    )

    # ============ 3) Standardize using TRAIN only ============
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = x_scaler.transform(X_val).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    y_scale = float(y_scaler.scale_[0])

    # torch tensors
    train_x = torch.from_numpy(X_train_s).to(device=device, dtype=dtype)
    train_y = torch.from_numpy(y_train_s).to(device=device, dtype=dtype)
    val_x   = torch.from_numpy(X_val_s).to(device=device, dtype=dtype)

    # ============ 4) GP model ============
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_x.shape[1]  # 6
                )
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Likelihood (homoskedastic noise, learnable)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    ).to(device)

    # Optional: initialize noise (in standardized y-space)
    likelihood.initialize(noise=1e-4)

    model = ExactGPModel(train_x, train_y, likelihood).to(device)

    # ============ 5) Train ============
    model.train()
    likelihood.train()

    # IMPORTANT: include likelihood params so noise updates
    params = list(model.parameters()) + list(likelihood.parameters())
    # (safety) de-duplicate just in case
    seen = set()
    unique_params = []
    for p in params:
        if id(p) not in seen:
            unique_params.append(p)
            seen.add(id(p))

    optimizer = torch.optim.Adam(unique_params, lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 300
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            os = float(model.covar_module.outputscale.detach().cpu().item())
            nz = float(likelihood.noise.detach().cpu().item())
            print(f"iter {i+1:3d} | loss={loss.item():.4f} | noise={nz:.3e} | outputscale={os:.3e} | lengthscale={ls}")

    # ============ 6) Validate ============
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_dist = likelihood(model(val_x))          # predictive y (含噪声)
        y_mean_s = y_dist.mean.detach().cpu().numpy()
        y_std_s  = y_dist.variance.sqrt().detach().cpu().numpy()

    # inverse transform to original y units
    y_mean = y_scaler.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
    y_std  = y_std_s * y_scale

    err = y_mean - y_val

    # 绝对指标
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))

    # -------- 相对指标（逐点除以 |y|，避免 y≈0）--------
    eps = 1e-12
    den = np.maximum(np.abs(y_val), eps)

    rel_mae  = float(np.mean(np.abs(err) / den))                 # mean absolute percentage-like (not *100)
    rel_rmse = float(np.sqrt(np.mean((err / den) ** 2)))         # rmse in relative space

    # 若你想百分比显示：
    rel_mae_pct  = 100.0 * rel_mae
    rel_rmse_pct = 100.0 * rel_rmse

    # -------- 归一化指标（整体尺度归一，便于对比）--------
    mean_abs_y = float(np.mean(np.abs(y_val)) + eps)
    nrmse_mean = float(rmse / mean_abs_y)
    nmae_mean  = float(mae  / mean_abs_y)

    # 区间覆盖率（95%）
    lower = y_mean - 1.96 * y_std
    upper = y_mean + 1.96 * y_std
    coverage95 = float(np.mean((y_val >= lower) & (y_val <= upper)))

    eps = 1e-12
    smape = np.mean(2*np.abs(y_mean - y_val)/(np.abs(y_val)+np.abs(y_mean)+eps))
    print("sMAPE:", smape, f"({100*smape:.2f}%)")

    mask = np.abs(y_val) > 0.1 * np.median(np.abs(y_val)+eps)  # 也可直接用固定阈值
    rel_mae_masked = np.mean(np.abs(y_mean[mask]-y_val[mask]) / (np.abs(y_val[mask])+eps))
    print("Masked Rel-MAE:", rel_mae_masked, f"({100*rel_mae_masked:.2f}%)", "kept:", mask.mean())

    print("\nVAL metrics (absolute)")
    print("RMSE:", rmse)
    print("MAE :", mae)

    print("\nVAL metrics (relative, pointwise / |y| )")
    print("Rel-RMSE:", rel_rmse, f"({rel_rmse_pct:.2f}%)")
    print("Rel-MAE :", rel_mae,  f"({rel_mae_pct:.2f}%)")

    print("\nVAL metrics (normalized by mean(|y|))")
    print("NRMSE(mean|y|):", nrmse_mean)
    print("NMAE (mean|y|):", nmae_mean)

    print("\nUncertainty check")
    print("95% coverage:", coverage95)

    # ============ 7) Save ============
    bundle = {
        "state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "val_metrics": {"rmse": rmse, "mae": mae, "coverage95": coverage95},
        "meta": {
            "input_dim": int(X_full.shape[1]),
            "kernel": "ScaleKernel(Matern nu=2.5, ARD)",
            "standardized_y_noise": True,
            "device": str(device),
        },
        "c": c,
    }
    save_path = f"{folder}/gpytorch_gp_model_revenue.pkl"
    joblib.dump(bundle, save_path)
    print("\nSaved to:", save_path)

    ## NN
    X_full = np.column_stack([X, theta.reshape(-1)])  # (N,6)

    nnwrap = NNModelTorch(input_dim=6).fit(
        X_full, y,
        val_frac=0.10,
        batch_size=128,
        lr=1e-3,
        epochs=400,
        hidden=(256, 256, 128),
        dropout=0.0,
        weight_decay=1e-6,
        seed=0,
        verbose_every=20,
    )
    nnwrap.save("nn_model_revenue.pkl")