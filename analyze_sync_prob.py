import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) 定义两个模型
# ------------------------------------------------------------
# simulator:
#   y_sim(x, theta) = sin(5 * theta * x) + 5x
#
# physical system:
#   eta(x; phi) = a1 * x * cos(a2 * x) + a3 * x
#
# 这里按照你之前的设定，固定 a1 = a3 = 5，只让 phi2 = a2 变化。
# ============================================================

def computer_model_np(x, theta):
    """
    x: shape (n, 1) or (n,)
    theta: float or array-like with one element
    return: shape (n,)
    """
    x = np.asarray(x).reshape(-1)
    theta = float(np.asarray(theta).reshape(-1)[0])
    return np.sin(5.0 * theta * x) + 5.0 * x


def physical_system(x, phi):
    """
    x: shape (n, 1) or (n,)
    phi: [a1, a2, a3]
    return: shape (n,)
    """
    x = np.asarray(x).reshape(-1)
    a1, a2, a3 = map(float, phi)
    return a1 * x * np.cos(a2 * x) + a3 * x


# ============================================================
# 2) 给定 phi，求 oracle theta*
# ------------------------------------------------------------
# 用 grid search:
#   theta*(phi) = argmin_theta MSE( eta(x;phi), y_sim(x,theta) )
# ============================================================

def oracle_theta(phi, theta_grid, x_grid):
    """
    给定 phi，在 theta_grid 上搜索最优 theta
    """
    eta = physical_system(x_grid, phi)
    losses = []

    for th in theta_grid:
        y_sim = computer_model_np(x_grid, th)
        mse = np.mean((eta - y_sim) ** 2)
        losses.append(mse)

    losses = np.asarray(losses)
    idx = np.argmin(losses)
    return float(theta_grid[idx]), losses


# ============================================================
# 3) 找到 theta*=target_theta 对应的 phi=[5,phi2,5]
# ------------------------------------------------------------
# 方法：
#   扫一遍 phi2_grid
#   对每个 phi2 算 theta*(phi)
#   找到最接近 target_theta 的 phi2
# ============================================================

def find_phi_for_target_theta(
    target_theta=1.25,
    phi2_grid=None,
    theta_grid=None,
    x_grid=None,
    a1=5.0,
    a3=5.0,
):
    if phi2_grid is None:
        phi2_grid = np.linspace(3.0, 12.0, 400)
    if theta_grid is None:
        theta_grid = np.linspace(0.0, 3.0, 800)
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.0, 500)

    theta_star_vals = []

    for phi2 in phi2_grid:
        phi = np.array([a1, phi2, a3], dtype=float)
        theta_star, _ = oracle_theta(phi, theta_grid, x_grid)
        theta_star_vals.append(theta_star)

    theta_star_vals = np.asarray(theta_star_vals)

    idx_best = np.argmin(np.abs(theta_star_vals - target_theta))
    phi2_best = float(phi2_grid[idx_best])
    theta_star_best = float(theta_star_vals[idx_best])

    phi_best = np.array([a1, phi2_best, a3], dtype=float)

    return phi_best, phi2_grid, theta_star_vals, theta_star_best


# ============================================================
# 4) 固定找到的 phi，计算所有 theta 的 L2(theta)
# ------------------------------------------------------------
# L2(theta) = sqrt( integral_0^1 [eta(x;phi)-y_sim(x,theta)]^2 dx )
# 用 trapz 做数值积分
# ============================================================

def compute_l2_curve(phi, theta_grid, x_grid):
    eta = physical_system(x_grid, phi)
    l2_vals = []
    mse_vals = []

    for th in theta_grid:
        y_sim = computer_model_np(x_grid, th)
        diff = eta - y_sim

        mse = np.mean(diff ** 2)
        l2 = np.sqrt(np.trapezoid(diff ** 2, x_grid))

        mse_vals.append(mse)
        l2_vals.append(l2)

    return np.asarray(l2_vals), np.asarray(mse_vals)


# ============================================================
# 5) 主程序
# ============================================================

def main():
    # ---------- 配置 ----------
    target_theta = 1.25

    x_grid = np.linspace(0.0, 1.0, 500)
    theta_grid_oracle = np.linspace(0.0, 3.0, 800)
    theta_grid_eval = np.linspace(0.0, 3.0, 800)
    phi2_grid = np.linspace(3.0, 12.0, 400)

    out_dir = "analysis_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Step 1: 找到 theta*=1.25 左右对应的 phi ----------
    phi_target, phi2_grid_used, theta_star_vals, theta_star_check = find_phi_for_target_theta(
        target_theta=target_theta,
        phi2_grid=phi2_grid,
        theta_grid=theta_grid_oracle,
        x_grid=x_grid,
        a1=5.0,
        a3=5.0,
    )

    print("=" * 60)
    print("Step 1: find phi for target theta*")
    print(f"target theta*         = {target_theta:.6f}")
    print(f"found phi             = {phi_target}")
    print(f"oracle theta*(phi)    = {theta_star_check:.6f}")
    print("=" * 60)

    # ---------- Step 2: 固定 phi，计算所有 theta 的 L2 ----------
    l2_vals, mse_vals = compute_l2_curve(phi_target, theta_grid_eval, x_grid)

    idx_l2_min = int(np.argmin(l2_vals))
    theta_l2_min = float(theta_grid_eval[idx_l2_min])

    idx_mse_min = int(np.argmin(mse_vals))
    theta_mse_min = float(theta_grid_eval[idx_mse_min])

    print("Step 2: evaluate L2(theta) with fixed phi")
    print(f"argmin_theta L2       = {theta_l2_min:.6f}")
    print(f"argmin_theta MSE      = {theta_mse_min:.6f}")
    print("=" * 60)

    # ---------- Step 3A: 画 phi2 -> theta*(phi) ----------
    plt.figure(figsize=(8, 5))
    plt.plot(phi2_grid_used, theta_star_vals, lw=2)
    plt.axhline(target_theta, color="red", linestyle="--", label=f"target theta* = {target_theta:.3f}")
    plt.axvline(phi_target[1], color="green", linestyle=":", label=f"phi2 = {phi_target[1]:.4f}")
    plt.xlabel(r"$\phi_2$")
    plt.ylabel(r"$\theta^*(\phi)$")
    plt.title(r"Mapping from $\phi_2$ to $\theta^*(\phi)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "phi2_to_theta_star.png"), dpi=300)
    plt.close()

    # ---------- Step 3B: 画 L2(theta) ----------
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid_eval, l2_vals, lw=2, label=r"$L_2(\theta)$")
    plt.axvline(target_theta, color="red", linestyle="--", label=fr"target $\theta^*={target_theta:.3f}$")
    plt.axvline(theta_l2_min, color="green", linestyle=":", label=fr"$\arg\min L_2={theta_l2_min:.3f}$")
    plt.scatter([theta_l2_min], [l2_vals[idx_l2_min]], s=40)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$L_2(\theta)$")
    plt.title(fr"$L_2(\theta)$ with fixed $\phi=[5,{phi_target[1]:.4f},5]$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "l2_vs_theta.png"), dpi=300)
    plt.close()

    # ---------- Step 3C: 画 MSE(theta) ----------
    plt.figure(figsize=(8, 5))
    plt.plot(theta_grid_eval, mse_vals, lw=2, label="MSE(theta)")
    plt.axvline(target_theta, color="red", linestyle="--", label=fr"target $\theta^*={target_theta:.3f}$")
    plt.axvline(theta_mse_min, color="green", linestyle=":", label=fr"$\arg\min MSE={theta_mse_min:.3f}$")
    plt.scatter([theta_mse_min], [mse_vals[idx_mse_min]], s=40)
    plt.xlabel(r"$\theta$")
    plt.ylabel("MSE")
    plt.title(fr"MSE$(\theta)$ with fixed $\phi=[5,{phi_target[1]:.4f},5]$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_vs_theta.png"), dpi=300)
    plt.close()

    # ---------- Step 3D: 画 physical vs best simulator ----------
    eta = physical_system(x_grid, phi_target)
    y_best = computer_model_np(x_grid, theta_l2_min)

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, eta, lw=2, label=fr"physical $\eta(x;\phi)$, $\phi=[5,{phi_target[1]:.4f},5]$")
    plt.plot(x_grid, y_best, lw=2, linestyle="--", label=fr"simulator $y_{{sim}}(x,\theta)$, $\theta={theta_l2_min:.4f}$")
    plt.xlabel("x")
    plt.ylabel("output")
    plt.title("Physical system vs best matching simulator")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "physical_vs_best_simulator.png"), dpi=300)
    plt.close()

    print("Saved figures:")
    print("  - analysis_outputs/phi2_to_theta_star.png")
    print("  - analysis_outputs/l2_vs_theta.png")
    print("  - analysis_outputs/mse_vs_theta.png")
    print("  - analysis_outputs/physical_vs_best_simulator.png")


if __name__ == "__main__":
    main()