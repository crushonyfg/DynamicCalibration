# CUSUM Discrepancy Memory Patch（设计说明）

## 1. 目标（必须读）

本补丁不是一个新的 change-point detector，也不参与 BOCPD 决策。

它的唯一作用是：

在 BOCPD 未触发 restart 的情况下，检测当前 discrepancy training data 是否已经“过时”，并进行 memory refresh。也就是说BOCPD是比较不同expert之间，而CUSUM比较的是anchor expert的现在的粒子对比历史的分布来说是否已经迁移的太远。

---

## 2. 核心思想

当前系统有两个问题来源：

(1) BOCPD（即使 halfdiscrepancy）的问题  
- small jump / gradual drift 时  
- PF tracking 正常  
- predictive likelihood 不明显变化  
- 不触发 restart  

(2) discrepancy 的问题  
- residual target r*(·;θ_t) 在 drift  
- 但仍在用旧数据训练  
- discrepancy 学坏  

结论：  
需要一个机制，在“没有 restart”的情况下，仍然触发 discrepancy 数据更新。

---

## 3. 系统结构（非常关键）

本补丁引入一个第二层控制变量：

- BOCPD：决定是否 restart（结构层）
- CUSUM：决定是否 refresh discrepancy（数据层）

两者完全解耦：
- CUSUM 不影响 BOCPD
- 不参与 likelihood
- 不改变 expert
- 不改变 PF

---

## 4. 算法定义

在每个时刻 t：

### Step 1（不变）

运行原有：
- PF update
- BOCPD update（可选 nodiscrepancy / halfdiscrepancy）

### Step 2（BOCPD restart）

若 BOCPD 触发 restart：
- 执行原有 dual-restart
- 设置 tau_delta = t
- 重置 G_delta = 0

### Step 3（CUSUM统计）

若 BOCPD 未触发 restart：

d_t = d(Π_t^anc, Π_{t-1}^anc)  
G_t = G_{t-1} + d_t

### Step 4（CUSUM触发）

若 G_t > h_delta：

执行 discrepancy-only memory refresh：
- 不做 restart
- 不动 PF
- 不动 BOCPD
- 丢弃旧 residual 数据
- 只保留最近 L 个 observations
- 重新训练 discrepancy

并设置：
tau_delta = t  
G_t = 0

---

## 5. d_t 定义

使用 PF posterior summary：

m_t = weighted_mean(theta_particles)  
Sigma_t = weighted_cov(theta_particles)

d_t = (m_t - m_{t-1})^T (Sigma_{t-1} + eps I)^(-1) (m_t - m_{t-1})

---

## 6. L 定义

使用固定 recent memory：

L_obs = 20

规则：
- 保留最近 >=20 个 observations
- 若当前 batch 不够 → 往前补
- 不使用更早数据

---

## 7. h_delta

h_delta = 常数（建议 5~20）

---

## 8. 不允许做的事情

CUSUM 不允许：
- 触发 BOCPD restart
- 改变 expert weights
- 改变 likelihood
- 创建新 expert
- 参与 run-length recursion

---

## 9. 可配置开关

use_cusum = True / False  
use_dual_restart = True / False  
bocpd_mode = ['nodiscrepancy', 'halfdiscrepancy']

---

## 10. 方法组合（用于 ablation）

需要支持：

- R-BOCPD-PF（无 discrepancy）
- halfdiscrepancy
- +CUSUM
- +dual restart
- half + CUSUM（主方法）

---

## 11. 应用到现有脚本

需要在以下文件中：

- run_synthetic_suddenCmp_tryThm.py  
- run_synthetic_slope_deltaCmp.py  
- run_plantSim_v3_std.py  

新增 method：

method = 'RBOCPD_half_CUSUM'

并保证：
- RMSE 计算一致
- CRPS 计算一致
- 接口一致

---

## 12. 总结

这是一个只作用于 discrepancy 数据窗口的 CUSUM-based memory truncation 机制，
不参与 BOCPD，不改变 PF，只在 BOCPD 未触发时工作。

## 13. 补充说明

保留 ablation 开关
实现时必须支持下面几类组合，以便后续做 ablation：
`dual-restart` 开 / 关；
`CUSUM patch` 开 / 关；
`CUSUM patch` 开，但 `BOCPD likelihood` 使用 `nodiscrepancy`；
`CUSUM patch` 开，`BOCPD likelihood` 使用 `halfdiscrepancy`。
不要把 `CUSUM` 写死到某一种 likelihood 模式里。

基于：
`restart_bocpd_hybrid_260319_gpytorch.py`
新建一个新的实现文件，例如：
`restart_bocpd_rolledCusum_260324_gpytorch.py`
文件名可由 Codex 视现有命名风格微调，但建议直接体现 `cusum`。
要求：
新文件尽量复用原脚本主体结构；
不要破坏原脚本已有方法；
新逻辑通过配置项和 `method` 分支注入；
便于后续把同一逻辑同步移植到其他脚本。

推荐的 ablation 实验最小集合
建议至少支持下面几组：
`R-BOCPD-PF-halfdiscrepancy-dualrestart`
`R-BOCPD-PF-halfdiscrepancy-dualrestart-cusum`
`R-BOCPD-PF-halfdiscrepancy-cusum-noDR`
`R-BOCPD-PF-nodiscrepancy-cusum`
用于回答以下问题：
`CUSUM` 是否对 `halfdiscrepancy + dual-restart` 有增益；
`CUSUM` 是否能在没有 dual-restart 时单独补救；
`CUSUM + nodiscrepancy` 是否能替代 `halfdiscrepancy`；
`CUSUM` 和 `halfdiscrepancy` 是替代关系还是互补关系。
预期上，最重要的是比较：
`halfdiscrepancy-dualrestart`
`halfdiscrepancy-dualrestart-cusum`
