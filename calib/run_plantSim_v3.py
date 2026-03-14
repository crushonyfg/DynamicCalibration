"""
run_plantSim_v3.py - Plant Simulation 校准实验

用法:
    # 方式1: 从目录读取Excel文件 (默认)
    python -m calib.run_plantSim_v3 --data_dir "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"
    
    # 方式2: 从CSV文件读取
    python -m calib.run_plantSim_v3 --csv "C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/physical_data.csv"
    
    # 其他参数
    python -m calib.run_plantSim_v3 --csv physical_data.csv --out_dir figs/plantSim/v3 --modes 0 1 2
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

NN_MODEL_PATH = "C:/Users/yxu59/files/autumn2025/park/codes/plant simulation/nn_model_revenue.pkl"

from calib.emulator import Emulator


class PlantEmulatorNNTorch(Emulator):
    """
    Pure-torch differentiable wrapper around a trained NNModelTorch.
    All operations stay on GPU as torch tensors, enabling autograd
    for OGP gradient computation via make_fast_batched_grad_func.
    """

    def __init__(self, nn_path: str = NN_MODEL_PATH,
                 device: str = "cuda", dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        nn_wrap = NNModelTorch(input_dim=6).load(nn_path, device=device)

        self.x_mean = torch.tensor(
            nn_wrap.scaler.x_scaler.mean_, device=device, dtype=dtype,
        )
        self.x_std = torch.tensor(
            nn_wrap.scaler.x_scaler.scale_, device=device, dtype=dtype,
        )
        self.y_mean_s = torch.tensor(
            float(nn_wrap.scaler.y_scaler.mean_[0]), device=device, dtype=dtype,
        )
        self.y_std_s = torch.tensor(
            float(nn_wrap.scaler.y_scaler.scale_[0]), device=device, dtype=dtype,
        )
        self.c = float(nn_wrap.y_transform.c)

        self.nn = nn_wrap.model.to(device)
        self.nn.eval()
        for p in self.nn.parameters():
            p.requires_grad_(False)

    def _forward_raw(self, x_full: torch.Tensor) -> torch.Tensor:
        """x_full [M, 6] -> y_raw [M].  Differentiable w.r.t. input."""
        x_scaled = (x_full.to(self.dtype) - self.x_mean) / self.x_std
        y_s = self.nn(x_scaled.float())
        y_t = y_s.to(self.dtype) * self.y_std_s + self.y_mean_s
        return torch.sign(y_t) * self.c * torch.expm1(torch.abs(y_t))

    _NN_CHUNK = 8192

    def predict(self, x: torch.Tensor, theta: torch.Tensor):
        """x [B, 5], theta [N, 1] -> (mu [B,N], var [B,N])"""
        B, N = x.shape[0], theta.shape[0]
        x_dev = x.to(device=self.device, dtype=self.dtype)
        th_dev = theta.to(device=self.device, dtype=self.dtype)
        x_rep = x_dev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        th_rep = th_dev.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1)
        x_full = torch.cat([x_rep, th_rep], dim=1)
        total = B * N
        with torch.no_grad():
            if total <= self._NN_CHUNK:
                y = self._forward_raw(x_full)
            else:
                y = torch.empty(total, device=self.device, dtype=self.dtype)
                for i in range(0, total, self._NN_CHUNK):
                    j = min(i + self._NN_CHUNK, total)
                    y[i:j] = self._forward_raw(x_full[i:j])
        mu = y.reshape(B, N)
        return mu, torch.zeros_like(mu)

    def sim_func(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """One-to-one: x [M,5], theta [M,1] -> y [M].  Differentiable."""
        return self._forward_raw(
            torch.cat([x.to(self.dtype), theta.to(self.dtype)], dim=1)
        )

    def x_domain_from_scaler(self, dx: int = 5):
        """Auto-compute x_domain from scaler: mean +/- 3*std, clamped >= 0."""
        lo = torch.clamp(self.x_mean[:dx] - 3 * self.x_std[:dx], min=0)
        hi = self.x_mean[:dx] + 3 * self.x_std[:dx]
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


def batches(stream: StreamClass, batch_size: int):
    while True: 
        try:
            yield stream.next(batch_size)
        except StopIteration:
            break

def run_plantSim(mode, methods, batch_size, data_dir=None, csv_path=None):
    if data_dir is None and csv_path is None:
        data_dir = "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3"

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
            stream = StreamClass(0, folder=data_dir, csv_path=csv_path, jump_plan=jp)
        else:
            stream = StreamClass(mode, folder=data_dir, csv_path=csv_path)

        # ---------- R-BOCPD-PF-OGP ----------
        if name == "R-BOCPD-PF-OGP":
            ogp_dev = "cuda"
            emulator_nn = PlantEmulatorNNTorch(NN_MODEL_PATH, device=ogp_dev)
            grad_func = make_fast_batched_grad_func(
                emulator_nn.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn.x_domain_from_scaler(dx=5)
            ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([30.0]),
                theta_move_std=0.5,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            bocpd_cfg = BOCPDConfig()
            bocpd_cfg.use_restart = True
            model_cfg = ModelConfig(rho=1.0, sigma_eps=50.0)
            roll = OGPRollingStats(window=50)

            bocpd = BOCPD_OGP(
                config=bocpd_cfg,
                ogp_pf_cfg=ogp_cfg,
                batched_grad_func=grad_func,
                device=ogp_dev,
            )

            for Xb, yb, thb in tqdm(batches(stream, batch_size), desc=f"Running {name}"):
                newX = torch.tensor(Xb, device=ogp_dev, dtype=torch.float64)
                gt_theta = torch.tensor(thb)
                newY = torch.tensor(yb, device=ogp_dev, dtype=torch.float64)

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
                            emulator_nn, model_cfg.rho, model_cfg.sigma_eps,
                        )
                        mix_mu += w_e * mu_e
                        mix_var += w_e * var_e
                        Z += w_e
                    mix_mu /= max(Z, 1e-12)
                    mix_var /= max(Z, 1e-12)
                    rmse_hist.append(
                        float(torch.sqrt(((mix_mu.cpu() - newY.cpu()) ** 2).mean()))
                    )
                idx += 1

                rec = bocpd.update_batch(
                    newX, newY, emulator_nn, model_cfg, None, prior_sampler,
                    verbose=False,
                )

                dll = rec.get("delta_ll_pair", None)
                if dll is not None and np.isfinite(dll):
                    roll.update(dll)

                mean_theta, var_theta, lo_theta, hi_theta = _aggregate_ogp_particles(
                    bocpd, 0.9,
                )
                gt_theta_hist.append(gt_theta.mean().item())
                theta_hist.append(float(mean_theta[0]))
                theta_var_hist.append(float(var_theta[0]))
                restart_hist.append(rec["did_restart"])

        # ---------- Standalone PF-OGP (no BOCPD) ----------
        elif name == "PF-OGP":
            ogp_dev = "cuda"
            emulator_nn = PlantEmulatorNNTorch(NN_MODEL_PATH, device=ogp_dev)
            pf_grad_func = make_fast_batched_grad_func(
                emulator_nn.sim_func, device=ogp_dev, dtype=torch.float64,
            )
            x_domain = emulator_nn.x_domain_from_scaler(dx=5)
            pf_ogp_cfg = OGPPFConfig(
                num_particles=1024,
                x_domain=x_domain,
                theta_lo=torch.tensor([0.0]),
                theta_hi=torch.tensor([30.0]),
                theta_move_std=0.5,
                ogp_quad_n=3,
                particle_chunk_size=64,
                max_hist=200,
            )
            pf_model_cfg = ModelConfig(rho=1.0, sigma_eps=50.0)

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
                newX = torch.tensor(Xb, device=ogp_dev, dtype=torch.float64)
                gt_theta = torch.tensor(thb)
                newY = torch.tensor(yb, device=ogp_dev, dtype=torch.float64)

                if idx > 0:
                    pf_Xh = pf_X_hist if pf_X_hist.numel() > 0 else None
                    pf_yh = pf_y_hist if pf_y_hist.numel() > 0 else None
                    mu_mix, var_mix = pf.predict_batch(
                        newX, pf_Xh, pf_yh,
                        emulator_nn, pf_model_cfg.rho, pf_model_cfg.sigma_eps,
                    )
                    rmse_hist.append(
                        float(torch.sqrt(((mu_mix.cpu() - newY.cpu()) ** 2).mean()))
                    )
                idx += 1

                pf.step_batch(
                    newX, newY,
                    pf_X_hist if pf_X_hist.numel() > 0 else None,
                    pf_y_hist if pf_y_hist.numel() > 0 else None,
                    emulator_nn,
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
                mean_theta = (w * pf.theta).sum(dim=0)
                gt_theta_hist.append(gt_theta.mean().item())
                theta_hist.append(float(mean_theta[0]))
                theta_var_hist.append(
                    float((w * (pf.theta - mean_theta).pow(2)).sum(dim=0)[0])
                )
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
                newX = torch.tensor(Xb)
                theta = torch.tensor(thb)
                newY = torch.tensor(yb)

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

        results[name] = dict(
            theta_hist=theta_hist,
            theta_var_hist=theta_var_hist,
            gt_theta_hist=gt_theta_hist,
            rmse_hist=rmse_hist,
            comp_rmse_hist=comp_rmse_hist,
            restart_hist=restart_hist,
        )
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plant Simulation Calibration Experiment")
    parser.add_argument("--out_dir", type=str, default="figs/plantSim/v3", help="Output directory for figures")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to PhysicalData directory (Excel files)")
    parser.add_argument("--csv", type=str, default=None, help="Path to aggregated CSV file")
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2], help="Modes to run (e.g., --modes 0 1 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    methods = {
        # "BPC-80": dict(type="bpc"),
        # "BOCPD-BPC": dict(type="bpc_bocpd"),
        "R-BOCPD-PF-OGP": dict(type="ogp_bocpd"),
        "PF-OGP": dict(type="pf_ogp"),
        # "BOCPD-PF": dict(type="bocpd", mode="standard"),
        # "R-BOCPD-PF-usediscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=True),
        # "R-BOCPD-PF-nodiscrepancy": dict(type="bocpd", mode="restart", use_discrepancy=False),
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
