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

def prior_sampler(N):
    return torch.rand(N, 1)*30

from calib.emulator import Emulator
class PlantEmulatorNN(Emulator):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nnwrap = NNModelTorch(input_dim=6).load("C:/Users/yxu59/files/autumn2025/park/codes/plant simulation/nn_model_revenue.pkl", device=device)
        self.computer_model = nnwrap
        # self.return_std = True

    def predict(self, x, theta):
        mu_eta = self.computer_model.predict(x, theta)
        mu_eta = torch.from_numpy(mu_eta)
        var_eta = torch.zeros_like(mu_eta)
        return mu_eta, var_eta

def batches(stream: StreamClass, batch_size: int):
    while True: 
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break

def run_plantSim(mode, methods, batch_size):
    folder = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"

    emulator = PlantEmulatorNN()

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
            newX = torch.tensor(Xb)
            theta = torch.tensor(thb)
            newY = torch.tensor(yb)
            # X_torch, Y_torch = X_torch[:batch_size,:], Y_torch[:batch_size]

            if idx > 0: 
                pred = calib.predict_batch(newX)
                rmse_hist.append(torch.sqrt(((pred["mu"] - newY)**2).mean()))
                pred_comp = calib.predict_complete(newX, newY)
                report_sub_hist = (pred_comp["crps_sim"].item(),pred_comp["experts_logpred"],pred_comp["var_sim"])
                comp_rmse_hist.append(report_sub_hist)
            idx += 1
            
            rec = calib.step_batch(newX, newY, verbose=False)
            mean_theta, var_theta, lo_theta, hi_theta = calib._aggregate_particles(0.9)
            gt_theta_hist.append(theta.mean().item())
            theta_hist.append(mean_theta.item())
            theta_var_hist.append(var_theta)

            restart_hist.append(rec["did_restart"])

        results[name] = dict(theta_hist=theta_hist, theta_var_hist=theta_var_hist, gt_theta_hist=gt_theta_hist, rmse_hist=rmse_hist, comp_rmse_hist=comp_rmse_hist, restart_hist=restart_hist)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v3")
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
    for mode in [2]:
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
