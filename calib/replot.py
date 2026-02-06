import numpy as np
import matplotlib.pyplot as plt

# ---------- 计算机模型：安全版 ----------
def y_sim_np(x, theta):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    th = float(theta)
    return np.sin(5.0 * th * x) + 5.0 * x

# ---------- 真实系统：只改 cos 的频率 a2 ----------
def y_true_np(x, a2):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return 5.0 * x * np.cos(a2 * x) + 5.0 * x

# ---------- 在每一段上用 L2 误差扫 θ ----------
def best_theta_on_segment(a2, theta_lo=0.05, theta_hi=3.0, n_grid=2001):
    x = np.linspace(0.0, 1.0, 1000)
    yt = y_true_np(x, a2)
    thetas = np.linspace(theta_lo, theta_hi, n_grid)
    errs = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        ys = y_sim_np(x, th)
        errs[i] = np.mean((yt - ys)**2)
    th_star = thetas[np.argmin(errs)]
    return th_star, thetas, errs

# ---------- 五个分段频率 ----------
# prefix = "cfg2_t4000_debug_26010802_40_batch_restart"  # 改成你的实验前缀
# prefix = "cfg2_t4000_mod"
# prefix = "cfg2_t4000_debug_26010802_40_batch_compareBatchonly"
# prefix = "cfg2_t4000_debug_26011401_20_thetatest0.1"
# prefix = "cfg2_t4000_debug_26011401_40_bpc200"
# prefix = "cfg2_t4000_debug_26011401_40_bpcbocpd"
# prefix = "cfg2_t4000_debug_26011501_20_pfcompare_thetatest0.5_lhs"
# prefix = "cfg2_t4000_debug_26011501_40_pfcompare_thetatest0.5_lhs"
# prefix = "cfg2_26011501_40_bocpdpf_diag"
prefix = "cfg2_260121_10_1000total"
a2_list = [7.5, 5.0, 12.0, 7.0, 11.0]
theta_stars = []
all_curves = []
for a2 in a2_list:
    th_star, thetas, errs = best_theta_on_segment(a2)
    theta_stars.append(th_star)
    all_curves.append((thetas, errs))

print("✅ 每段最优 θ：", [round(float(t), 4) for t in theta_stars])

# ============================================================
# 1️⃣ 绘制误差 L2 vs θ，查看每段最优点
# ============================================================
# fig, axes = plt.subplots(len(a2_list), 1, figsize=(7, 10), sharex=True)
# for i, (a2, ax) in enumerate(zip(a2_list, axes)):
#     thetas, errs = all_curves[i]
#     ax.plot(thetas, errs, color='C0')
#     ax.axvline(theta_stars[i], color='r', ls='--', lw=1.5)
#     ax.set_title(f"Segment {i+1}: a2 = {a2}, θ* = {theta_stars[i]:.3f}")
#     ax.set_ylabel("L2 error")
#     ax.grid(True, alpha=0.3)
# axes[-1].set_xlabel("θ")
# plt.tight_layout()
# plt.savefig(f"{prefix}_theta_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()

# ============================================================
# 2️⃣ 绘制真实系统 vs 模型（含多 θ 曲线）
# ============================================================
x_plot = np.linspace(0, 1, 400)
theta_samples = [0.5, 1.0, 1.5, 2.0, 2.5]  # 额外的示例 θ

# fig, axes = plt.subplots(len(a2_list), 1, figsize=(8, 12), sharex=True)
# for i, (a2, ax) in enumerate(zip(a2_list, axes)):
#     # 真实系统
#     yt = y_true_np(x_plot, a2)
#     ax.plot(x_plot, yt, 'b', lw=2, label="True η(x)")
    
#     # 最优模型
#     ys_star = y_sim_np(x_plot, theta_stars[i])
#     ax.plot(x_plot, ys_star, 'r--', lw=2.5, label=f"Model θ*={theta_stars[i]:.3f}")
    
#     # 若干不同 θ 的模型曲线
#     for th in theta_samples:
#         ax.plot(x_plot, y_sim_np(x_plot, th), color="gray", alpha=0.4, lw=1)
    
#     ax.grid(alpha=0.3)
#     ax.set_ylabel(f"a₂={a2}")
#     ax.set_title(f"Segment {i+1} | a₂={a2:.1f} | θ*={theta_stars[i]:.3f}")
#     ax.legend(loc="best", fontsize=8)

# axes[-1].set_xlabel("x")
# plt.tight_layout()
# plt.savefig(f"{prefix}_theta_examples.png", dpi=300, bbox_inches="tight")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 读取保存的结果
# ============================================================
data = np.load(f"{prefix}_results_summary.npz", allow_pickle=True)

plt.figure(figsize=(10, 6))
method_names = sorted(
    key[:-6]                      
    for key in data.keys()
    if key.endswith("_theta")
)

for method_name in method_names:
    theta_history = data[f"{method_name}_theta"]
    times_history = data[f"{method_name}_times"]
    times_std = np.concatenate(([0], times_history))
    times_std = times_std + 40
    plt.plot(times_std, theta_history, label=method_name, lw=2)
# plt.legend()
# plt.show()

# theta_std = data["Standard_theta"]
# theta_rst = data["Restart_theta"]
# # theta_koh = data["KOH_theta"]

# # 时间轴（每 batch 40 个点）
# times_std = data["Standard_times"]
# times_rst = data["Restart_times"]
# # times_koh = data["KOH_times"]
# times_std = np.concatenate(([0], times_std))
# times_std = times_std + 40
# theta_std = np.concatenate(([theta_std[0]], theta_std))
# theta_rst = np.concatenate(([theta_rst[0]], theta_rst))


# ============================================================
# changepoint 及真实最优 θ*
# ============================================================
# changepoints 与 run_synthetic.py 中一致
# cp_times = [800, 1600, 2400, 3200]
cp_times = [200, 400, 600, 800]
# 每个阶段真实系统的 a2（参见你的 EnhancedChangepointConfig 设置）

# ============================================================
# 绘制 θ_t 估计曲线 + changepoints + θ*
# ============================================================

# plt.plot(times_std, theta_std, label="Standard BOCPD", lw=2)
# plt.plot(times_std, theta_rst, label="Restart BOCPD", lw=2)
# plt.plot(times_std, theta_koh, label="KOH Calibration", lw=2)

# changepoints（竖线）
for cp in cp_times:
    plt.axvline(cp, color='red', linestyle='--', alpha=0.6)

# 每段真实 θ*（水平线）
for i, th_star in enumerate(theta_stars):
    x0 = 0 if i == 0 else cp_times[i-1]
    # x0 = cp_times[i-1]
    x1 = cp_times[i] if i < len(cp_times) else max(times_std)
    plt.hlines(th_star, x0, x1, colors='#555555', linestyles='dashed', lw=3, alpha=0.9)

plt.xlabel("Observation Time (t)")
plt.ylabel(r"Estimated $\theta$")
plt.title("Online Estimated θ Trajectories (Config2)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{prefix}_theta_history_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
