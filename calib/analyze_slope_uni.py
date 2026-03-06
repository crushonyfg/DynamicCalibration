import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_slope_uni_ori(s, seed, batch_size, prefix):
    store_dir = f"figs/slope_{prefix}/{s}_seed{seed}_batch{batch_size}"
    os.makedirs(store_dir, exist_ok=True)
    data = torch.load(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/slope_{prefix}/slope_{s}_seed{seed}_batch{batch_size}_results.pt", weights_only=False)
    theta_stars = torch.load(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/slope_{prefix}/slope_{s}_seed{seed}_batch{batch_size}_phi_oracle_hist.pt", weights_only=False)

    # print(theta_stars)
    phi_hist, oracle_hist = theta_stars["phi_hist"], theta_stars["oracle_hist"]

    rmse_list = [data[name]["rmse"] for name in data.keys()]
    names = [name for name in data.keys()]

    plt.figure(figsize=(8, 5))

    for i, rmse in enumerate(rmse_list):
        plt.plot(rmse, alpha=0.7, label=names[i])

    plt.xlabel("time step")
    plt.ylabel("RMSE")
    plt.title("RMSE trajectories")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_rmse.png")

    rmse_mat = np.stack(rmse_list, axis=0)  # shape = (N_runs, T)
    rmse_mean = rmse_mat.mean(axis=1)   # shape (T,)
    rmse_var  = rmse_mat.var(axis=1)    # shape (T,)
    rmse_std  = rmse_mat.std(axis=1)
    print(names)
    print(rmse_mean, rmse_var, rmse_std)

    theta_rmse_all = {}

    for n in names:
        theta_n = np.asarray(data[n]["theta"])
        theta_rmse_all[n] = np.sqrt(np.mean((theta_n - oracle_hist) ** 2))

    for k, v in theta_rmse_all.items():
        print(f"{k}: theta RMSE = {v:.4f}")

    coverage_all, lo_all, hi_all = {}, {}, {}
    for n in names:
        try:
            others = data[n]["others"]

            lo = np.array([o["lo"] for o in others])
            hi = np.array([o["hi"] for o in others])
            oracle = np.asarray(oracle_hist)

            coverage = np.mean((oracle >= lo) & (oracle <= hi))
            print(f"{n}'s 90% coverage:", coverage)
            coverage_all[n] = coverage
            lo_all[n] = lo
            hi_all[n] = hi
        except:
            continue

    plt.figure(figsize=(9, 5))

    # -------- point estimates --------
    for i, n in enumerate(names):
        plt.plot(
            data[n]["theta"],
            alpha=0.7,
            label=n
        )

    # -------- oracle --------
    plt.plot(
        oracle_hist,
        color="black",
        linestyle="dashed",
        lw=2,
        label="oracle"
    )

    # -------- 90% credible interval (Restart-BOCPD) --------
    # others = data["R-BOCPD-PF"]["others"]
    others = data["R-BOCPD-PF-usediscrepancy"]["others"]
    lo = np.array([o["lo"] for o in others])
    hi = np.array([o["hi"] for o in others])

    t = np.arange(len(lo))
    plt.fill_between(
        t,
        lo,
        hi,
        color="red",
        alpha=0.2,
        label="Restart-BOCPD 90% CI"
    )

    plt.xlabel("time step")
    plt.ylabel("Estimated theta")
    plt.title("Theta tracking with 90% credible interval")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_theta.png")

    # lst = [x["did_restart"] for x in data["R-BOCPD-PF"]["others"]]
    if "R-BOCPD-PF-nodiscrepancy" in data:
        lst = [x["did_restart"] for x in data["R-BOCPD-PF-nodiscrepancy"]["others"]]
        count_true = sum(lst)
        avg_interval = len(lst)/count_true if count_true > 0 else np.nan
        print("R-BOCPD-PF-nodiscrepancy", count_true, avg_interval)
    else:
        count_true = np.nan
        avg_interval = np.nan
    if "R-BOCPD-PF-usediscrepancy" in data:
        lst1 = [x["did_restart"] for x in data["R-BOCPD-PF-usediscrepancy"]["others"]]
        count_true1 = sum(lst1)
        avg_interval1 = len(lst1)/count_true1 if count_true1 > 0 else np.nan
        print("R-BOCPD-PF-usediscrepancy", count_true1, avg_interval1)
    if count_true is np.nan and count_true1 is np.nan: 
        lst = [x["did_restart"] for x in data["R-BOCPD-PF"]["others"]]
        count_true = sum(lst)
        avg_interval = len(lst)/count_true if count_true > 0 else np.nan
        print("R-BOCPD-PF", count_true, avg_interval)
        count_true1, avg_interval1 = np.nan, np.nan
    return {"rmse":[names, rmse_mean, rmse_var, rmse_std], "theta":[names, theta_rmse_all, coverage_all, lo_all, hi_all], "restart":[count_true, avg_interval], "restart1":[count_true1, avg_interval1]}

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_slope_uni(s, seed, batch_size, prefix1, prefix2=None, prefix3=None):

    # -------- prefix3 fallback --------
    if prefix3 is None:
        prefix3 = prefix1

    store_dir = f"figs/slope_{prefix3}/{s}_seed{seed}_batch{batch_size}"
    os.makedirs(store_dir, exist_ok=True)

    # -------- load main data (prefix1) --------
    data = torch.load(
        f"figs/slope_{prefix1}/slope_{s}_seed{seed}_batch{batch_size}_results.pt",
        weights_only=False
    )

    theta_stars = torch.load(
        f"figs/slope_{prefix1}/slope_{s}_seed{seed}_batch{batch_size}_phi_oracle_hist.pt",
        weights_only=False
    )

    try:
        phi_hist, oracle_hist = theta_stars["phi_hist"], np.asarray(theta_stars["oracle_hist"], dtype=float)
    except:
        phi_hist, oracle_hist,_ = theta_stars

    # -------- optional merge from prefix2 --------
    if prefix2 is not None:
        data2 = torch.load(
            f"figs/slope_{prefix2}/slope_{s}_seed{seed}_batch{batch_size}_results.pt",
            weights_only=False
        )

        # 1) add R-BOCPD-PF-nodiscrepancy
        if "R-BOCPD-PF-nodiscrepancy" in data2:
            data["R-BOCPD-PF-nodiscrepancy"] = data2["R-BOCPD-PF-nodiscrepancy"]

        # 2) replace Restart-BOCPD or R-BOCPD-PF with usediscrepancy
        if "R-BOCPD-PF-usediscrepancy" in data2:
            if "Restart-BOCPD" in data:
                data.pop("Restart-BOCPD")
            if "R-BOCPD-PF" in data:
                data.pop("R-BOCPD-PF")

            data["R-BOCPD-PF-usediscrepancy"] = data2["R-BOCPD-PF-usediscrepancy"]

    # -------- RMSE trajectories --------
    rmse_list = [data[name]["rmse"] for name in data.keys()]
    names = list(data.keys())

    plt.figure(figsize=(8, 5))
    for i, rmse in enumerate(rmse_list):
        plt.plot(rmse, alpha=0.7, label=names[i])

    plt.xlabel("time step")
    plt.ylabel("RMSE")
    plt.title("RMSE trajectories")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_rmse.png")
    plt.close()

    rmse_mat = np.stack(rmse_list, axis=0)
    rmse_mean = rmse_mat.mean(axis=1)
    rmse_var = rmse_mat.var(axis=1)
    rmse_std = rmse_mat.std(axis=1)

    # -------- theta RMSE --------
    theta_rmse_all = {}
    for n in names:
        theta_n = np.asarray(data[n]["theta"], dtype=float)
        theta_rmse_all[n] = np.sqrt(np.mean((theta_n - oracle_hist) ** 2))

    # -------- coverage --------
    coverage_all, lo_all, hi_all = {}, {}, {}
    for n in names:
        if "others" not in data[n]:
            continue
        try:
            others = data[n]["others"]
            lo = np.array([o["lo"] for o in others])
            hi = np.array([o["hi"] for o in others])
            coverage = np.mean((oracle_hist >= lo) & (oracle_hist <= hi))
            coverage_all[n] = coverage
            lo_all[n] = lo
            hi_all[n] = hi
        except Exception:
            pass

    # -------- theta plot --------
    plt.figure(figsize=(9, 5))

    for n in names:
        plt.plot(data[n]["theta"], alpha=0.7, label=n)

    plt.plot(
        oracle_hist,
        color="black",
        linestyle="dashed",
        lw=2,
        label="oracle"
    )

    if "R-BOCPD-PF-usediscrepancy" in data:
        others = data["R-BOCPD-PF-usediscrepancy"]["others"]
        lo = np.array([o["lo"] for o in others])
        hi = np.array([o["hi"] for o in others])
        t = np.arange(len(lo))
        plt.fill_between(t, lo, hi, color="red", alpha=0.2, label="Restart-BOCPD 90% CI")

    plt.xlabel("time step")
    plt.ylabel("Estimated theta")
    plt.title("Theta tracking with 90% credible interval")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{store_dir}/slope_{s}_seed{seed}_batch{batch_size}_theta.png")
    plt.close()

    # -------- restart stats --------
    def restart_stats(key):
        if key not in data:
            return np.nan, np.nan
        lst = [x["did_restart"] for x in data[key]["others"]]
        c = sum(lst)
        avg = len(lst) / c if c > 0 else np.nan
        return c, avg

    count_true, avg_interval = restart_stats("R-BOCPD-PF-nodiscrepancy")
    count_true1, avg_interval1 = restart_stats("R-BOCPD-PF-usediscrepancy")

    y_crps_all = {}

    for name in data.keys():
        crps = []
        if "BOCPD-PF" not in name:
            for i in range(1,len(data[name]["others"])):
                crps.append(data[name]["others"][i]["crps_sim"].mean().item())
            y_crps_all[name] = np.mean(crps)
        else:
            for i in range(1, len(data[name]["others"])):
                crps.append(data[name]["others"][i]["report_sub_hist"][0])
            y_crps_all[name] = np.mean(crps)

    from scipy.stats import norm

    def gaussian_crps_1d(mu, var, y):
        sigma = np.sqrt(max(var, 1e-12))
        z = (y - mu) / sigma
        return sigma * (
            z * (2 * norm.cdf(z) - 1)
            + 2 * norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
            
    
    theta_crps_dict = {}
    for name in data.keys():
        theta_vars = []
        pf_infos = []
        for i in range(len(data[name]["others"])):
            var = data[name]["others"][i]["var"]
            try:
                gini, ess, unique, entropy = data[name]["others"][i]["pf_info"][0]["gini"], \
                    data[name]["others"][i]["pf_info"][0]["ess"], data[name]["others"][i]["pf_health_info"][0]["unique_ratio"], \
                        data[name]["others"][i]["pf_health_info"][0]["entropy_1d_histogram"]
            except:
                # entropy = data[name]["others"][i]["entropy"]
                entropy = None
                gini, ess, unique = None, None, None
            theta_vars.append(var)
            pf_infos.append((gini, ess, unique, entropy))
        theta_n = np.asarray(data[n]["theta"], dtype=float)
        theta_vars = np.asarray(theta_vars, dtype=float)
        theta_crps_list = [
        gaussian_crps_1d(theta_n[t], theta_vars[t], oracle_hist[t])
        for t in range(len(oracle_hist))
            ]
        theta_crps = np.mean(theta_crps_list)
        theta_crps_dict[name] = theta_crps
        # print(name, theta_crps)
        


    return {
        "rmse": [names, rmse_mean, rmse_var, rmse_std],
        "theta": [names, theta_rmse_all, coverage_all, lo_all, hi_all],
        "restart": [count_true, avg_interval],
        "restart1": [count_true1, avg_interval1],
        "y-crps": y_crps_all,
        "theta_crps": theta_crps_dict
    }


def main():
    import itertools

    # prefix1 = "v5"
    # prefix2 = "v6"        # or e.g. "v6_alt"
    # prefix3 = "v6_merge"  # output folder
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix1", type=str, default="deltaCmp_v1")
    parser.add_argument("--prefix2", type=str, default=None)
    parser.add_argument("--prefix3", type=str, default=None)
    parser.add_argument("--slopes", type=str, default=[0.0005, 0.001, 0.0015, 0.002, 0.0025])
    parser.add_argument("--bcs", type=str, default=[10, 20, 40])
    parser.add_argument("--seeds", type=str, default=[456])
    args = parser.parse_args()
    prefix1 = args.prefix1
    prefix2, prefix3 = args.prefix2, args.prefix3

    slopes = [float(s) for s in args.slopes]
    bcs = [int(bc) for bc in args.bcs]
    seeds = [int(seed) for seed in args.seeds]

    results = {}
    for slope, bc, seed in itertools.product(slopes, bcs, seeds):
        results[(slope, bc, seed)] = analyze_slope_uni(
            slope, seed, bc, prefix1, prefix2, prefix3
        )

    if prefix3 is not None:
        torch.save(results, f"figs/slope_{prefix3}/slope_{prefix3}_results.pt")
    else:
        torch.save(results, f"figs/slope_{prefix1}/slope_{prefix1}_results.pt")
    return results


if __name__ == "__main__":
    results = main()
    print(results)


# def main():
#     import itertools
#     # slopes = [0.001, 0.002, 0.005, 0.01]
#     prefix = "v6"
#     slopes = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
#     bcs = [10,20,40]
#     seeds = [456]
#     results = {}
#     for slope, bc, seed in itertools.product(slopes, bcs, seeds):
#         results[(slope, bc, seed)] = analyze_slope_uni(slope, seed, bc, prefix)
#     torch.save(results, f"figs/slope_{prefix}/slope_{prefix}_results.pt")
#     return results

# if __name__ == "__main__":
#     results = main()
#     print(results)
