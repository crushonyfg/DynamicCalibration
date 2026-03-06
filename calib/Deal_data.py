import joblib
import numpy as np
import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import dataclasses

@dataclasses.dataclass
class GPModel:
    gp: GaussianProcessRegressor=None
    x_scaler: StandardScaler=None
    y_scaler: StandardScaler=None
    y_scale: float=None

    def __post_init__(self, path="C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v1/computerdata/gp_model_revenue.pkl"):
        bundle = joblib.load(path)

        self.gp = bundle["gp"]
        self.x_scaler = bundle["x_scaler"]
        self.y_scaler = bundle["y_scaler"]

        self.y_scale = self.y_scaler.scale_[0]

    def __refit_init__(self,path="C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v1/computerdata/gp_training_data_revenue.csv"):
        data = pd.read_csv(path)
        X = data[["Q","R","W","M1","M2","Lbd"]].values
        y = data["y"].values
        var = data["var"].values

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_s = x_scaler.fit_transform(X)
        y_s = y_scaler.fit_transform(y.reshape(-1,1)).ravel()

        y_scale = y_scaler.scale_[0]
        alpha = var / (y_scale**2)

        kernel = ConstantKernel(1.0, (1e-2,1e3)) * Matern(length_scale=[1]*6, nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=10,
            normalize_y=False
        )

        gp.fit(X_s, y_s)

        self.gp = gp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.y_scale = y_scale

    def gp_predict(self, X_base, theta, return_std=False):
        '''
        X_base: (N,5)
        theta: (N,)
        return: (N,)
        '''
        X_base = np.atleast_2d(X_base)
        theta = np.atleast_1d(theta)

        if len(theta)==1:
            theta = np.repeat(theta, len(X_base))

        X_full = np.column_stack([X_base, theta])
        Xs = self.x_scaler.transform(X_full)

        y_pred_s, y_std_s = self.gp.predict(Xs, return_std=True)

        y_pred = self.y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
        y_std  = y_std_s * self.y_scale

        if return_std:
            return y_pred, y_std
        return y_pred

import os
import re
import numpy as np
import pandas as pd

# ==========================================================
# 1️⃣ CustomerLbd 时间字符串 → 分钟float
# ==========================================================
def time_to_min(t):
    """
    '12:30' -> 12.5
    '12:02.025' -> 12.034
    """
    if isinstance(t, (int,float)):
        return float(t)

    s = str(t)
    if ":" not in s:
        return float(s)

    m, sec = s.split(":")
    return float(m) + float(sec)/60.0


# ==========================================================
# 2️⃣ 文件名解析
# ==========================================================
# pattern = re.compile(
#     r"Mode(?P<Mode>\d+)"
#     r"Q(?P<Q>\d+)_R(?P<R>\d+)_W(?P<W>\d+)"
#     r"_M1(?P<M1>\d+)_M2(?P<M2>\d+)"
# )
pattern = re.compile(
    r"Mode(?P<Mode>\d+)"
    r"seed(?P<seed>\d+)"
)

# ==========================================================
# 3️⃣ 读取physical目录
# ==========================================================
# def load_physical_stream(folder, mode=0):
#     """
#     返回：
#         X_list[t] : (N,5)
#         Y_list[t] : (N,)
#         theta_list[t] : (N,)
#     """

#     files = []

#     for f in os.listdir(folder):
#         if not f.endswith(".xlsx"):
#             continue
#         if not f.startswith("factory_"):
#             continue

#         m = pattern.search(f)
#         if not m:
#             continue

#         if int(m.group("Mode")) != mode:
#             continue

#         params = m.groupdict()
#         files.append((f, params))

#     if len(files)==0:
#         raise ValueError("No files for mode")

#     streams = []

#     for fname, params in files:
#         df = pd.read_excel(os.path.join(folder, fname))

#         # Δthroughput
#         # delta = df["throughput"].diff().values
#         delta = df["NetRevenue"].diff().values
#         delta[0] = np.nan

#         # θ*
#         theta = df["CustomerLbd"].apply(time_to_min).values

#         # X
#         X = np.array([
#             float(params["Q"]),
#             float(params["R"]),
#             float(params["W"]),
#             float(params["M1"]),
#             float(params["M2"])
#         ])

#         streams.append({
#             "file": fname,
#             "X": X,
#             "Y": delta,
#             "theta": theta
#         })

#     # ======================================================
#     # 对齐时间长度
#     # ======================================================
#     min_len = min(len(s["Y"]) for s in streams)

#     for s in streams:
#         s["Y"] = s["Y"][:min_len]
#         s["theta"] = s["theta"][:min_len]

#     # ======================================================
#     # 构造batch stream
#     # ======================================================
#     X_list = []
#     Y_list = []
#     theta_list = []

#     for t in range(min_len):
#         X_batch = []
#         Y_batch = []
#         theta_batch = []

#         for s in streams:
#             if np.isnan(s["Y"][t]):
#                 continue

#             X_batch.append(s["X"])
#             Y_batch.append(s["Y"][t])
#             theta_batch.append(s["theta"][t])

#         X_list.append(np.array(X_batch))
#         Y_list.append(np.array(Y_batch))
#         theta_list.append(np.array(theta_batch))

#     return X_list[1:], Y_list[1:], theta_list[1:]
def load_physical_stream(folder, mode=0):
    """
    返回：
        X_list[t] : (N,5)
        Y_list[t] : (N,)
        theta_list[t] : (N,)
    """

    files = []

    for f in os.listdir(folder):
        if not f.endswith(".xlsx"):
            continue
        if not f.startswith("factory_"):
            continue

        m = pattern.search(f)
        if not m:
            continue

        if int(m.group("Mode")) != mode:
            continue

        files.append(f)

    if len(files) == 0:
        raise ValueError("No files for mode")

    streams = []

    # ======================================================
    # 读取每个 seed 文件
    # ======================================================
    for fname in files:
        df = pd.read_excel(os.path.join(folder, fname))

        # ===== y =====
        delta = df["NetRevenue"].diff().values
        delta[0] = np.nan

        # ===== theta =====
        theta = df["CustomerLbd"].apply(time_to_min).values

        # ===== X 逐行 =====
        X = df[["Q", "R", "W", "M1", "M2"]].values.astype(float)

        streams.append({
            "file": fname,
            "X": X,
            "Y": delta,
            "theta": theta
        })

    # ======================================================
    # 对齐时间长度
    # ======================================================
    min_len = min(len(s["Y"]) for s in streams)

    for s in streams:
        s["Y"] = s["Y"][:min_len]
        s["theta"] = s["theta"][:min_len]
        s["X"] = s["X"][:min_len]

    # ======================================================
    # 构造batch stream
    # ======================================================
    X_list = []
    Y_list = []
    theta_list = []

    for t in range(min_len):
        X_batch = []
        Y_batch = []
        theta_batch = []

        for s in streams:
            if np.isnan(s["Y"][t]):
                continue

            X_batch.append(s["X"][t])
            Y_batch.append(s["Y"][t])
            theta_batch.append(s["theta"][t])

        X_list.append(np.array(X_batch))
        Y_list.append(np.array(Y_batch))
        theta_list.append(np.array(theta_batch))

    return X_list[1:], Y_list[1:], theta_list[1:]


# ==========================================================
# 4️⃣ Stream 类
# ==========================================================
class PhysicalStream:
    def __init__(self, folder, mode=0):
        self.X_list, self.Y_list, self.theta_list = load_physical_stream(folder, mode)
        self.T = len(self.X_list)
        self.idx = 0

    def __len__(self):
        return self.T

    def reset(self):
        self.idx = 0

    def next(self):
        if self.idx >= self.T:
            raise StopIteration

        X = self.X_list[self.idx]
        Y = self.Y_list[self.idx]
        theta = self.theta_list[self.idx]

        self.idx += 1
        theta_gt = np.unique(theta)[0]
        return X, Y, theta_gt

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()


# ==========================================================
# 5️⃣ 使用示例
# ==========================================================
if __name__ == "__main__":

    computer_model = GPModel()
    computer_model.__post_init__()

    folder = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v1"

    stream = PhysicalStream(folder, mode=0)

    for X, Y, theta in stream:
        print("batch size:", len(X))
        print("X:", X.shape) # (N,5)
        print("Y:", Y.shape) # (N,)
        print("theta:", theta.shape) # (N,)
        theta_gt = np.unique(theta)[0]
        print("theta:", theta_gt) # (1,)
        break
    