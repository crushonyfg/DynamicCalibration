import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import seaborn as sns

import pandas as pd

color_map = {
    # ---- PF / BOCPD-PF variants ----
    "BOCPD-PF": "#1f77b4",                    # blue
    "R-BOCPD-PF": "#2ca02c",                  # green
    "R-BOCPD-PF-nodiscrepancy": "#ff7f0e",    # orange
    "R-BOCPD-PF-usediscrepancy": "#9467bd",   # purple

    # ---- BPC family ----
    "BPC-80": "#d62728",                      # red
    "BOCPD-BPC": "#8c564b",                   # brown
}

def compute_rank_magnitude(df, metric):
    ranks = []
    for delta_mag in df["delta_mag"].unique():
        d = df[df["delta_mag"]==delta_mag]
        order = d.groupby("method")[metric].mean().rank(method="average")
        for m, r in order.items():
            ranks.append(dict(
                delta_mag=delta_mag,
                method=m,
                metric=metric,
                rank=r
            ))
    return pd.DataFrame(ranks)

def plot_coverage_magnitude(df):
    plt.figure(figsize=(7,5))
    for method in df["method"].unique():
        # method = "Restart-BOCPD"
        d = df[df["method"]==method]
        if d["coverage"].isna().all():
            continue
        m = d.groupby("delta_mag")["coverage"].mean()
        plt.plot(m.index, m.values, marker="o", label=method, color=color_map[method])
    # method = "Restart-BOCPD"
    # d = df[df["method"]==method]
    # # if d["coverage"].isna().all():
    # #     continue
    # m = d.groupby("slope")["coverage"].mean()
    # plt.plot(m.index, m.values, marker="o", label=method)

    plt.axhline(0.9, color="k", linestyle="--", alpha=0.6)
    plt.xlabel("magnitude")
    plt.ylabel("Coverage")
    plt.title("90% interval coverage vs magnitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_restart(df, metric="magnitude", type="sudden"):
    metric_dict = {"magnitude": "delta_mag", "slope": "slope"}
    if metric not in metric_dict:
        metric_dict[metric] = metric
    d = df[df["method"]=="R-BOCPD-PF-nodiscrepancy"]
    g = d.groupby(metric_dict[metric])["restart_count"].mean()
    d1 = df[df["method"]=="R-BOCPD-PF-usediscrepancy"]
    g1 = d1.groupby(metric_dict[metric])["restart_count"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(g.index, g.values, marker="o", label="R-BOCPD-PF-nodiscrepancy", color=color_map["R-BOCPD-PF-nodiscrepancy"])
    plt.plot(g1.index, g1.values, marker="o", label="R-BOCPD-PF-usediscrepancy", color=color_map["R-BOCPD-PF-usediscrepancy"])
    plt.xlabel(metric)
    plt.ylabel("Restart count")
    plt.title(f"Restart count vs {metric}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/{type}_{prefix}/{type}_{prefix}_restart_vs_{metric}.png")
    plt.show()

def flatten_results(results):
    rows = []
    for (seg_len_L, delta_mag, batch_size, seed), res in results.items():
    # for t, res in results.items():
        # print(t)
        names, rmse_mean, rmse_var, rmse_std = res["rmse"]
        _, theta_rmse_all, coverage_all, lo_all, hi_all = res["theta"]
        count_true, avg_interval = res["restart"]
        count_true1, avg_interval1 = res["restart1"]
        count_true1, avg_interval1 = res["restart1"]
        y_crps, theta_crps = res["y-crps"], res["theta_crps"]
        for i, name in enumerate(names):
            # print(coverage_all[name])
            rows.append(dict(
                seg_len_L=seg_len_L,
                delta_mag=delta_mag,
                batch_size=batch_size,
                seed=seed,
                method=name,
                rmse_mean=float(rmse_mean[i]),
                rmse_std=float(rmse_std[i]),
                theta_rmse=float(theta_rmse_all[name]),
                coverage=float(coverage_all[name]) if (name in coverage_all and coverage_all[name] is not None) else np.nan,
                restart_count=count_true if name=="R-BOCPD-PF-nodiscrepancy" else (count_true1 if name=="R-BOCPD-PF-usediscrepancy" else np.nan),
                avg_restart_interval=avg_interval if name=="R-BOCPD-PF-nodiscrepancy" else (avg_interval1 if name=="R-BOCPD-PF-usediscrepancy" else np.nan),
                y_crps=y_crps[name],
                theta_crps=theta_crps[name],
            ))
    return pd.DataFrame(rows)

def summarize_rank(df, func_rank):
    rank_rmse       = func_rank(df, "rmse_mean")
    rank_theta      = func_rank(df, "theta_rmse")
    rank_ycrps      = func_rank(df, "y_crps")
    rank_thetacrps  = func_rank(df, "theta_crps")

    # 合并
    rank_all = pd.concat([
        rank_rmse,
        rank_theta,
        rank_ycrps,
        rank_thetacrps
    ])

    # 计算每个 method 每个 metric 的平均 rank
    rank_summary = (
        rank_all
        .groupby(["method", "metric"])["rank"]
        .mean()
        .unstack("metric")
        .reset_index()
    )

    summary_df = (
        df
        .groupby("method")[["theta_crps", "y_crps", "theta_rmse", "rmse_mean"]]
        .mean()
        .reset_index()
    )

    final_df = summary_df.merge(rank_summary, on="method")
    return final_df

def plot_theta_rmse_heatmap_multi(df, methods, cmap="viridis", prefix="sudden"):
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        d = df[df["method"] == method]
        mat = d.pivot_table(
            index="seg_len_L",
            columns="batch_size",
            values="theta_rmse",
            aggfunc="mean"
        )

        sns.heatmap(
            mat,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            ax=ax,
            cbar=ax is axes[-1],  # 只在最后一个画 colorbar
            cbar_kws={"label": "Theta RMSE"},
        )
        ax.set_title(method)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Segment length L")

    plt.tight_layout()
    plt.savefig(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/sudden_{prefix}/sudden_{prefix}_theta_rmse_heatmap_multi.png")
    plt.show()

def plot_restart_vs_magnitude(df, prefix="sudden"):
    d = df[df["method"]=="R-BOCPD-PF-nodiscrepancy"]
    g = d.groupby("delta_mag")["restart_count"].mean()
    d1 = df[df["method"]=="R-BOCPD-PF-usediscrepancy"]
    g1 = d1.groupby("delta_mag")["restart_count"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(g.index, g.values, marker="o", label="R-BOCPD-PF-nodiscrepancy", color=color_map["R-BOCPD-PF-nodiscrepancy"])
    plt.plot(g1.index, g1.values, marker="o", label="R-BOCPD-PF-usediscrepancy", color=color_map["R-BOCPD-PF-usediscrepancy"])
    plt.xlabel("magnitude")
    plt.ylabel("Restart count")
    plt.title("Restart count vs magnitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/sudden_{prefix}/sudden_{prefix}_restart_vs_magnitude.png")
    plt.show()

def plot_metric_vs_variable(df, metric, ylabel, variable="magnitude", prefix="sudden"):
    plt.figure(figsize=(7,5))
    for method in df["method"].unique():
        d = df[df["method"]==method]
        m = d.groupby(variable)[metric].mean()
        plt.plot(m.index, m.values, marker="o", label=method, color=color_map[method])

    plt.xlabel(variable)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {variable}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/figs/sudden_{prefix}/sudden_{prefix}_{metric}_vs_{variable}.png")
    plt.show()


def plot_vertical_density_oracle_centered_multilayer(
    method_dict,
    oracle,
    grid_points=200,
    width=0.35,
    window=0.3,
    threshold=10
):

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    any_particles = list(method_dict.values())[0][0]
    B = any_particles.shape[0]

    oracle = np.asarray(oracle).reshape(-1)
    if len(oracle) != B:
        raise ValueError("oracle length must equal number of batches")

    # ===== 自动计算层数 =====
    n_rows = math.ceil(B / threshold)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(9, 5 * n_rows),
        sharey=True
    )

    if n_rows == 1:
        axes = [axes]

    labeled_methods = set()

    for row in range(n_rows):

        ax = axes[row]

        start = row * threshold
        end   = min((row + 1) * threshold, B)
        batch_subset = range(start, end)

        y_grid = np.linspace(-window, window, grid_points)

        for m_idx, (name, (particles, weights)) in enumerate(method_dict.items()):

            particles_np = particles.detach().cpu().numpy()
            weights_np   = weights.detach().cpu().numpy()

            for b in batch_subset:

                centered_particles = particles_np[b] - oracle[b]

                kde = gaussian_kde(
                    centered_particles,
                    weights=weights_np[b]
                )

                density = kde(y_grid)

                # if density.max() > 0:
                #     density = density / density.max() * width
                # else:
                #     continue
                dx = y_grid[1] - y_grid[0]
                area = np.sum(density) * dx

                density = density * width/3

                label = None
                if name not in labeled_methods:
                    label = name
                    labeled_methods.add(name)

                ax.plot(
                    b + density,
                    y_grid,
                    color=colors[m_idx % len(colors)],
                    alpha=0.8,
                    label=label
                )

        ax.axhline(0.0, color="k", linewidth=1.0)
        ax.set_xlim(start - 0.3, end - 1 + 0.3)
        ax.set_ylim(-window, window)
        ax.set_ylabel(r"$\theta - \theta^{*}$")

    axes[-1].set_xlabel("Batch")
    axes[0].set_title("Particle density evolution (oracle-centered)")

    # 统一 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()



def plot_expert_density_ranked_centered_multilayer(
    batch_dict,
    grid_points=200,
    width=0.4,
    window=0.3,
    topk=None,
    threshold=10,
):

    sorted_batches = sorted(batch_dict.keys())
    B = len(sorted_batches)

    # ===== 自动计算层数 =====
    n_rows = math.ceil(B / threshold)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(9, 5 * n_rows),
        sharey=True
    )

    if n_rows == 1:
        axes = [axes]

    labeled_ranks = set()

    for row in range(n_rows):

        ax = axes[row]

        start = row * threshold
        end   = min((row + 1) * threshold, B)

        batch_subset = sorted_batches[start:end]

        for b in batch_subset:

            experts = batch_dict[b]

            experts_sorted = sorted(
                experts,
                key=lambda x: float(x[2]),
                reverse=True
            )

            if topk is not None:
                experts_sorted = experts_sorted[:topk]

            # ===== rank-0 mean =====
            p0, w0, _ = experts_sorted[0]
            p0 = p0.detach().cpu().numpy()
            w0 = w0.detach().cpu().numpy()
            theta_star = np.sum(p0 * w0)

            y_grid = np.linspace(-window, window, grid_points)

            for rank, (particles, weights, logmass) in enumerate(experts_sorted):

                p = particles.detach().cpu().numpy()
                w = weights.detach().cpu().numpy()

                centered_p = p - theta_star

                kde = gaussian_kde(centered_p, weights=w)
                density = kde(y_grid)

                if density.max() > 0:
                    density = density / density.max() * width
                else:
                    continue

                lw = 2.0 if rank == 0 else 1.0

                label = None
                if rank not in labeled_ranks:
                    label = f"Rank {rank}"
                    labeled_ranks.add(rank)

                ax.plot(
                    b + density,
                    y_grid,
                    color=f"C{rank % 10}",
                    linewidth=lw,
                    alpha=0.9,
                    label=label
                )

        ax.axhline(0.0, color="k", linewidth=2.0)
        ax.set_xlim(min(batch_subset) - 0.5, max(batch_subset) + 0.5)
        ax.set_ylim(-window, window)
        ax.set_ylabel(r"$\theta - \theta^*_{\mathrm{rank0}}$")

    axes[-1].set_xlabel("Batch")
    axes[0].set_title("Expert density evolution (rank-colored, centered)")

    # 统一 legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_expert_density_ranked_centered_split(
    batch_dict,
    grid_points=200,
    width=0.4,
    window=0.3,
    topk=None,
    split_threshold=10,
):

    B = len(batch_dict)

    # ===== 判断是否分割 =====
    if B <= split_threshold:
        fig, axes = plt.subplots(1, 1, figsize=(9, 5))
        axes = [axes]
        batch_splits = [sorted(batch_dict.keys())]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharey=True)

        sorted_batches = sorted(batch_dict.keys())
        mid = B // 2
        batch_splits = [
            sorted_batches[:mid],
            sorted_batches[mid:]
        ]

    labeled_ranks = set()

    for ax, batch_subset in zip(axes, batch_splits):

        for b in batch_subset:

            experts = batch_dict[b]

            experts_sorted = sorted(
                experts,
                key=lambda x: float(x[2]),
                reverse=True
            )

            if topk is not None:
                experts_sorted = experts_sorted[:topk]

            # ===== rank-0 mean =====
            p0, w0, _ = experts_sorted[0]
            p0 = p0.detach().cpu().numpy()
            w0 = w0.detach().cpu().numpy()
            theta_star = np.sum(p0 * w0)

            y_grid = np.linspace(-window, window, grid_points)

            for rank, (particles, weights, logmass) in enumerate(experts_sorted):

                p = particles.detach().cpu().numpy()
                w = weights.detach().cpu().numpy()

                centered_p = p - theta_star

                kde = gaussian_kde(centered_p, weights=w)
                density = kde(y_grid)

                if density.max() > 0:
                    density = density / density.max() * width
                else:
                    continue

                lw = 2.0 if rank == 0 else 1.0

                label = None
                if rank not in labeled_ranks:
                    label = f"Rank {rank}"
                    labeled_ranks.add(rank)

                ax.plot(
                    b + density,
                    y_grid,
                    color=f"C{rank % 10}",
                    linewidth=lw,
                    alpha=0.9,
                    label=label
                )

        ax.axhline(0.0, color="k", linewidth=2.0)
        ax.set_xlim(min(batch_subset) - 0.5, max(batch_subset) + 0.5)
        ax.set_ylim(-window, window)
        ax.set_ylabel(r"$\theta - \theta^*_{\mathrm{rank0}}$")

    axes[-1].set_xlabel("Batch")
    axes[0].set_title("Expert density evolution (rank-colored, centered)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_vertical_density(method_dict, grid_points=200, width=0.35, oracle=None, ymin=0, ymax=3):
    """
    method_dict:
        {
            "methodA": (particlesA, weightsA),
            "methodB": (particlesB, weightsB),
        }
    """

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # 假设所有方法 B 相同
    any_particles = list(method_dict.values())[0][0]
    B, N = any_particles.shape

    # 全局 y-grid
    # all_particles = torch.cat(
    #     [v[0] for v in method_dict.values()],
    #     dim=0
    # ).detach().cpu().numpy()

    y_min, y_max = ymin, ymax
    y_grid = np.linspace(y_min, y_max, grid_points)

    plt.figure(figsize=(9,5))

    for m_idx, (name, (particles, weights)) in enumerate(method_dict.items()):

        particles = particles.detach().cpu().numpy()
        weights   = weights.detach().cpu().numpy()

        for b in range(B):

            kde = gaussian_kde(
                particles[b],
                weights=weights[b]
            )

            density = kde(y_grid)

            density = density / density.max() * width

            # x_shift = b + (m_idx - len(method_dict)/2) * width
            x_shift = b

            if b == 0:
                plt.plot(
                    x_shift + density,
                    y_grid,
                    color=colors[m_idx],
                    alpha=0.8,
                    label=name
                )
            else:
                plt.plot(
                    x_shift + density,
                    y_grid,
                    color=colors[m_idx],
                    alpha=0.8
                )

    if oracle is not None:
        oracle = np.asarray(oracle).reshape(-1)
        if len(oracle) != B:
            raise ValueError(f"oracle length {len(oracle)} must equal B={B}")

        x = np.arange(B)
        plt.step(x, oracle, where="post", color="k", linewidth=2.0, label="oracle")

    plt.xlabel("Batch")
    plt.ylabel("Particle value")
    plt.title("Particle density evolution (multi-method)")
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_vertical_density_oracle_centered(
    method_dict,
    oracle,
    grid_points=200,
    width=0.35,
    window=0.3,
    split_threshold=10
):

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    any_particles = list(method_dict.values())[0][0]
    B = any_particles.shape[0]

    oracle = np.asarray(oracle).reshape(-1)
    if len(oracle) != B:
        raise ValueError("oracle length must equal number of batches")

    # ===== 判断是否分图 =====
    if B <= split_threshold:
        fig, axes = plt.subplots(1, 1, figsize=(9, 5))
        axes = [axes]
        batch_splits = [range(B)]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharey=True)

        mid = B // 2
        batch_splits = [
            range(0, mid),
            range(mid, B)
        ]

    labeled_methods = set()

    for ax, batch_subset in zip(axes, batch_splits):

        y_grid = np.linspace(-window, window, grid_points)

        for m_idx, (name, (particles, weights)) in enumerate(method_dict.items()):

            particles_np = particles.detach().cpu().numpy()
            weights_np   = weights.detach().cpu().numpy()

            for b in batch_subset:

                centered_particles = particles_np[b] - oracle[b]

                kde = gaussian_kde(
                    centered_particles,
                    weights=weights_np[b]
                )

                density = kde(y_grid)

                if density.max() > 0:
                    density = density / density.max() * width
                else:
                    continue

                label = None
                if name not in labeled_methods:
                    label = name
                    labeled_methods.add(name)

                ax.plot(
                    b + density,
                    y_grid,
                    color=colors[m_idx % len(colors)],
                    alpha=0.8,
                    label=label
                )

        ax.axhline(0.0, color="k", linewidth=1.0)
        ax.set_xlim(min(batch_subset) - 0.3, max(batch_subset) + 0.3)
        ax.set_ylim(-window, window)
        ax.set_ylabel(r"$\theta - \theta^{*}$")

    axes[-1].set_xlabel("Batch")
    axes[0].set_title("Particle density evolution (oracle-centered)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_particle_scatter_weighted(particles, weights):
    particles = particles.detach().cpu().numpy()
    weights   = weights.detach().cpu().numpy()

    B, N = particles.shape

    plt.figure(figsize=(8,5))

    xs = []
    ys = []
    cs = []

    for b in range(B):
        xs.append(np.full(N, b))
        ys.append(particles[b])
        cs.append(weights[b])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    cs = np.concatenate(cs)

    plt.scatter(xs, ys, c=cs, s=6, alpha=0.6)
    plt.colorbar(label="particle weight")

    plt.xlabel("Batch index")
    plt.ylabel("Particle value")
    plt.title("Weighted particle evolution")
    plt.tight_layout()
    plt.show()

def plot_ridgeline(particles, weights, grid_points=200):
    particles = particles.detach().cpu().numpy()
    weights   = weights.detach().cpu().numpy()

    B, N = particles.shape

    x_min = particles.min()
    x_max = particles.max()
    x_grid = np.linspace(x_min, x_max, grid_points)

    plt.figure(figsize=(8,6))

    offset = 0.0
    gap = 0.6

    for b in range(B):
        kde = gaussian_kde(
            particles[b],
            weights=weights[b]
        )
        density = kde(x_grid)

        density = density / density.max() * 0.5  # 归一化控制高度

        plt.fill_between(
            x_grid,
            offset,
            offset + density,
            alpha=0.8
        )

        offset += gap

    plt.xlabel("Particle value")
    plt.ylabel("Batch progression")
    plt.title("Particle density evolution")
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def weighted_quantile(values, weights, quantiles):
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, values)

def plot_weighted_boxplot(particles, weights):
    particles = particles.detach().cpu().numpy()
    weights   = weights.detach().cpu().numpy()

    B, N = particles.shape

    q_low  = []
    q25    = []
    q50    = []
    q75    = []
    q_high = []

    for b in range(B):
        qs = weighted_quantile(
            particles[b],
            weights[b],
            [0.05, 0.25, 0.5, 0.75, 0.95]
        )
        q_low.append(qs[0])
        q25.append(qs[1])
        q50.append(qs[2])
        q75.append(qs[3])
        q_high.append(qs[4])

    y = np.arange(B)

    plt.figure(figsize=(7,5))

    for i in range(B):
        plt.plot([q25[i], q75[i]], [y[i], y[i]], linewidth=6)
        plt.plot([q_low[i], q_high[i]], [y[i], y[i]], linewidth=2)
        plt.scatter(q50[i], y[i], s=15)

    plt.xlabel("Particle value")
    plt.ylabel("Batch index")
    plt.title("Weighted box evolution")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_particle_evolution(particles, weights, bins=60):
    """
    particles: (B, N)
    weights:   (B, N)
    """

    particles = particles.detach().cpu().numpy()
    weights   = weights.detach().cpu().numpy()

    B, N = particles.shape

    # 全局 bin（保证时间可比较）
    x_min = particles.min()
    x_max = particles.max()
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    density_matrix = np.zeros((B, bins))

    for b in range(B):
        hist, _ = np.histogram(
            particles[b],
            bins=bin_edges,
            weights=weights[b],
            density=True
        )
        density_matrix[b] = hist

    plt.figure(figsize=(8, 5))

    plt.imshow(
        density_matrix,
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, 0, B],
    )

    plt.colorbar(label="weighted density")
    plt.xlabel("Particle value")
    plt.ylabel("Batch index")
    plt.title("Particle evolution over batches")

    plt.tight_layout()
    plt.show()

def plot_weighted_histograms(particles, weights, bins=30):
    """
    particles: (B, N)
    weights:   (B, N)
    """

    B, N = particles.shape

    # 转 numpy（避免 GPU / grad 问题）
    particles = particles.detach().cpu().numpy()
    weights   = weights.detach().cpu().numpy()

    fig, axes = plt.subplots(
        nrows=B,
        ncols=1,
        figsize=(6, 2.2 * B),
        sharex=True
    )

    # 如果 B=1，axes 不是数组
    if B == 1:
        axes = [axes]

    for b in range(B):

        p = particles[b]
        w = weights[b]

        # 可选：排序（不影响 histogram 结果，仅为了可读性）
        idx = np.argsort(p)
        p_sorted = p[idx]
        w_sorted = w[idx]

        axes[b].hist(
            p_sorted,
            bins=bins,
            weights=w_sorted,
            density=True,
            alpha=0.8
        )

        axes[b].set_ylabel(f"Batch {b}")
        axes[b].grid(alpha=0.2)

    axes[-1].set_xlabel("Particle value")

    plt.tight_layout()
    plt.show()
