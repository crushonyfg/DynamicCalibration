import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from time import time

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

def prior_sampler(N):
    return torch.rand(N, 1)*30

from calib.emulator import Emulator
class PlantEmulator(Emulator):
    def __init__(self):
        computer_model = GPModel()
        computer_model.__post_init__("C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v2/computerdata/gp_model_revenue.pkl")
        self.computer_model = computer_model
        self.return_std = True

    def predict(self, x, theta):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()

        b = x.shape[0]
        N = theta.shape[0]

        # expand
        X_rep = np.repeat(x, N, axis=0)        # [b*N, d]
        theta_tile = np.tile(theta, (b,1))     # [b*N, p]

        X_full = np.column_stack([X_rep, theta_tile])  # [b*N, d+p]

        y_pred, y_std = self.computer_model.gp_predict(X_rep, theta_tile, return_std=True)

        # reshape back to [b,N]
        y_pred = y_pred.reshape(b, N)
        y_var  = (y_std**2).reshape(b, N)

        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_var, np.ndarray):
            y_var = torch.from_numpy(y_var)

        if self.return_std:
            return y_pred, y_var
        else:
            return y_pred

def run_plantSim(mode, methods, batch_size):
    folder = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v2"

    emulator = PlantEmulator()

    results = {}

    for name, meta in methods.items():
        theta_hist, theta_var_hist, gt_theta_hist = [], [], []
        rmse_hist, comp_rmse_hist = [0], [0]
        restart_hist = []
        idx = 0

        stream = PhysicalStream(folder, mode=mode)

        cfg = CalibrationConfig()
        cfg.bocpd.bocpd_mode = meta["mode"]
        cfg.bocpd.use_restart = True

        if meta["mode"] == "restart":
            cfg.model.use_discrepancy = meta["use_discrepancy"]

        calib = OnlineBayesCalibrator(cfg, emulator, prior_sampler)
        for X,Y,theta in tqdm(stream, desc=f"Running {name}"):
            X_torch, Y_torch = torch.from_numpy(X), torch.from_numpy(Y)
            # X_torch, Y_torch = X_torch[:batch_size,:], Y_torch[:batch_size]

            if idx > 0: 
                pred = calib.predict_batch(X_torch)
                rmse_hist.append(torch.sqrt(((pred["mu"] - Y_torch)**2).mean()))
                pred_comp = calib.predict_complete(X_torch, Y_torch)
                comp_rmse_hist.append(torch.sqrt(((pred_comp["mu_sim"] - Y_torch)**2).mean()))
            idx += 1
            
            rec = calib.step_batch(X_torch, Y_torch, verbose=False)
            mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
            gt_theta_hist.append(theta)
            theta_hist.append(mean_theta)
            theta_var_hist.append(var_theta)

            restart_hist.append(rec["did_restart"])

        results[name] = dict(theta_hist=theta_hist, theta_var_hist=theta_var_hist, gt_theta_hist=gt_theta_hist, rmse_hist=rmse_hist, comp_rmse_hist=comp_rmse_hist, restart_hist=restart_hist)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v2")
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
    for mode in [0, 1]:
        for bs in [2]:
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
    

    torch.save(all_results, f"{out_dir}/plantSim_results.pt")
