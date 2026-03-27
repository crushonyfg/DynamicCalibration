# Rolled CUSUM Modeling Workdoc

## 1. Scope

This document is the maintained reference for the `rolled_cusum_260324` line.
It explains:

- what mathematical model the current code is implementing,
- which switches exist,
- how the discrepancy layer is parameterized,
- which options affect PF, BOCPD, restart, or discrepancy refresh,
- and which runner method names map to which modeling choice.

This document should be updated whenever the rolled-CUSUM path gains a new option or changes its semantics.

## 2. Layered view of the model

The code is easiest to understand as four layers.

### 2.1 Observation model

For scalar output, the working model is

$$
y_t(x) = \rho \, \eta(x, \theta_t) + \delta_t(x) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal N(0, \sigma_\varepsilon^2).
$$

Here:

- $\eta(x,\theta)$ is the emulator / simulator term,
- $\theta_t$ is the calibration latent state tracked by PF,
- $\delta_t(x)$ is the discrepancy term,
- $\rho$ is the simulator scale factor,
- $\sigma_\varepsilon^2$ is observation noise.

### 2.2 PF layer

The PF layer tracks $\theta_t$.
The intended design remains discrepancy-free PF weighting:

$$
w_t^{(i)} \propto w_{t-1}^{(i)}
\; p\!\left(Y_t \mid X_t, \theta_t^{(i)}, \text{simulator only}\right).
$$

Interpretation:

- PF is the latent calibration tracker.
- Discrepancy is not supposed to dominate particle identifiability.
- BOCPD-side discrepancy enrichment happens after or around PF, not inside PF weights.

### 2.3 BOCPD / expert layer

Each BOCPD expert maintains:

- a particle cloud for $\theta$,
- expert history,
- and a discrepancy state used for predictive scoring and/or refresh.

The BOCPD predictive object can be written as

$$
q_{e,t}(Y_t \mid X_t)
=
\int p(Y_t \mid X_t, \theta_t, \phi_t)
\, p(\phi_t \mid \theta_t, \mathcal D_{e,t-1})
\, p(\theta_t \mid \mathcal D_{e,t-1})
\, d\phi_t \, d\theta_t,
$$

where $\phi_t$ denotes the discrepancy-side latent object.

### 2.4 Refresh / restart layer

The code currently separates two decisions:

- BOCPD restart: structural decision about experts / restart behavior.
- Rolled-CUSUM refresh: discrepancy-memory maintenance decision.

This is important:

- the standardized gate or cumulative statistic is not meant to replace BOCPD,
- it is used to decide when discrepancy memory should be refreshed,
- and it should not change PF weights by itself.

## 3. Core discrepancy parameterizations

The current rolled-CUSUM path supports several discrepancy models through `particle_delta_mode`.

### 3.1 `shared_gp`

This is the original expert-shared discrepancy design.
For expert $e$, one residual target is formed using a PF-weighted simulator mean:

$$
r_e(x) = y(x) - \rho \sum_i w_i \, \eta(x, \theta_i).
$$

Then a single GP is fit to that expert-level residual history:

$$
\delta_e(\cdot) \sim \mathcal{GP}(0, k_\psi(\cdot,\cdot)).
$$

This gives one shared posterior for the whole expert.

### 3.2 `particle_gp_shared_hyper`

This is particle-specific discrepancy with one shared GP hyperparameter setting.

For each particle $\theta_i$, define particle-specific residuals

$$
r_e^{(i)}(x) = y(x) - \rho \, \eta(x, \theta_i).
$$

Conditioned on each particle, discrepancy is a GP posterior using the same kernel hyperparameters $\psi$:

$$
\delta_e^{(i)}(\cdot) \mid \theta_i, \mathcal D_e
\sim
\mathcal{GP\ posterior}(r_e^{(i)}, \psi).
$$

Implementation idea:

- fit one shared GP on the expert-shared residual to get a stable hyperparameter setting,
- reuse that hyperparameter set for all particle-specific residual posteriors,
- reuse kernel-factorization work across particles for efficiency.

### 3.3 `particle_gp_hyper_pool`

This keeps particle-specific residuals, but replaces a single shared hyperparameter setting by a small candidate pool.

For a small set of hyperparameters $\{\psi_h\}_{h=1}^H$,

$$
p(\delta \mid \theta_i, \mathcal D_e)
\approx
\sum_{h=1}^H \omega_{ih}
\, p(\delta \mid \theta_i, \mathcal D_e, \psi_h).
$$

Implementation intent:

- hyperparameter candidates are shared across experts / particles at configuration time,
- kernel matrices are reusable for each candidate,
- only the particle-specific residual vectors differ across particles.

This is meant to approximate a lightweight hyper-mixture without exploding cost.

### 3.4 `particle_basis`

This is the particle-specific basis-form discrepancy.

For each particle,

$$
\delta_e^{(i)}(x) = \phi(x)^\top \beta_e^{(i)}.
$$

A Bayesian linear / ridge-style posterior is fit using the particle-specific residuals:

$$
r_e^{(i)} = \Phi_e \beta_e^{(i)} + \xi,
\qquad
\xi \sim \mathcal N(0, \sigma_\delta^2 I).
$$

Current basis options:

- `particle_basis_kind="linear"`
- `particle_basis_kind="rbf"`

This branch is useful for fast ablations and for checking whether a lower-rank discrepancy parameterization behaves differently from GP discrepancy.

## 4. Prediction semantics

For a fixed expert and particle, the predictive law is approximately

$$
Y \mid X, \theta_i, \mathcal D_e
\sim
\mathcal N
\left(
\rho \, \mu_\eta(X,\theta_i) + \mu_{\delta,i}(X),
\rho^2 \sigma_\eta^2(X,\theta_i) + \sigma_{\delta,i}^2(X) + \sigma_\varepsilon^2
\right).
$$

Then the particle mixture is taken using the PF weights.

Important code-level note:

- shared discrepancy returns `mu_delta, var_delta` with batch shape,
- particle-specific discrepancy returns `mu_delta, var_delta` with batch-by-particle shape,
- the prediction path must preserve that distinction.

## 5. Restart and refresh options

### 5.1 `use_dual_restart`

Controls whether the BOCPD hybrid restart logic allows dual / partial restart behavior.
This is a BOCPD-side structural switch.

### 5.2 `use_cusum`

Master switch for the discrepancy refresh patch.
If `False`, the extra rolled-CUSUM refresh logic is inactive.

### 5.3 `cusum_mode`

Current supported values:

- `"cumulative"`
- `"standardized_gate"`

#### `cumulative`

This is the earlier cumulative drift statistic:

$$
d_t = (m_t - m_{t-1})^\top (\Sigma_{t-1} + \epsilon I)^{-1} (m_t - m_{t-1}),
$$

$$
G_t = G_{t-1} + d_t.
$$

If $G_t > h$, discrepancy memory is refreshed.

This is best interpreted as a cumulative standardized drift budget, not textbook centered CUSUM.

#### `standardized_gate`

This is the current preferred lightweight gate.
Define

$$
z_t = \sqrt{d_t},
$$

with the same Mahalanobis-type increment score $d_t$ above.
Then discrepancy memory is refreshed if the standardized move exceeds a threshold, for example:

$$
z_t > \tau_{gate}
$$

for one or more consecutive batches.

Current config controls:

- `standardized_gate_threshold`
- `standardized_gate_consecutive`

Interpretation:

- half-discrepancy handles weak sustained drift at the predictive level,
- the standardized gate is a safeguard for local latent moves that PF can absorb without BOCPD restart,
- gate-triggered action is discrepancy refresh, not full restart.

### 5.4 `cusum_recent_obs`

Controls how much recent discrepancy history is retained during a refresh.
The refresh truncates discrepancy training memory to the most recent observations instead of resetting the full expert or PF state.

## 6. Discrepancy-use switches

These are easy to confuse, so they should always be documented together.

### 6.1 `use_discrepancy`

This controls whether the main predictive side uses discrepancy.

### 6.2 `bocpd_use_discrepancy`

This controls whether BOCPD-side scoring / restart attribution uses discrepancy-aware predictive laws.

A useful interpretation is:

- `use_discrepancy=False, bocpd_use_discrepancy=False`
  means nodiscrepancy on both the predictive and BOCPD-side logic selected by that runner.
- `use_discrepancy=False, bocpd_use_discrepancy=True`
  is the half-discrepancy style split.
- `use_discrepancy=True, bocpd_use_discrepancy=True`
  is the fully discrepancy-aware variant.

When reading code or experiments, always check both switches together.

## 7. Current configuration summary

### 7.1 Restart / refresh controls

- `restart_impl`
  - use `rolled_cusum_260324` for this line.
- `use_dual_restart`
- `use_cusum`
- `cusum_mode`
- `cusum_threshold`
- `cusum_recent_obs`
- `cusum_cov_eps`
- `standardized_gate_threshold`
- `standardized_gate_consecutive`

### 7.2 Hybrid BOCPD controls

- `hybrid_tau_delta`
- `hybrid_tau_theta`
- `hybrid_tau_full`
- `hybrid_delta_share_rho`
- `hybrid_pf_sigma_mode`
- `hybrid_sigma_delta_alpha`
- `hybrid_sigma_ema_beta`
- `hybrid_sigma_min`
- `hybrid_sigma_max`

### 7.3 Discrepancy controls

- `use_discrepancy`
- `bocpd_use_discrepancy`
- `particle_delta_mode`
  - `shared_gp`
  - `particle_gp_shared_hyper`
  - `particle_gp_hyper_pool`
  - `particle_basis`
- `particle_gp_hyper_candidates`
- `particle_basis_kind`
- `particle_basis_num_features`
- `particle_basis_lengthscale`
- `particle_basis_ridge`
- `particle_basis_noise`

## 8. Current interface summary

### 8.1 BOCPD implementation file

Primary implementation:

- `calib/restart_bocpd_rolled_cusum_260324_gpytorch.py`

Key responsibilities:

- preserve the hybrid restart implementation,
- add refresh logic that is no-op-safe,
- construct discrepancy states according to `particle_delta_mode`,
- fall back to shared GP if richer discrepancy construction fails.

### 8.2 Discrepancy state file

Primary discrepancy extensions:

- `calib/particle_specific_discrepancy.py`

Current state classes:

- `ParticleSpecificGPDeltaState`
- `ParticleSpecificBasisDeltaState`

### 8.3 Likelihood hook

`calib/likelihood.py` must remain aware that discrepancy prediction can be either:

- shared across particles, or
- particle-specific via `predict_for_particles(...)`.

This is a backward-compatible interface hook, not a change of the likelihood formula itself.

## 9. Runner-level method naming

Current recommended naming style is ablation-oriented.
Examples:

- `RBOCPD_half_STDGate`
- `RBOCPD_half_STDGate_particleGP`
- `RBOCPD_half_STDGate_particleBasis`

Recommendation:

- add new variants rather than mutating these names,
- keep the method name descriptive enough that a results table is readable without digging into config.

## 10. Efficiency notes

The efficiency rationale for particle-specific discrepancy is:

- residual vectors differ across particles because $\theta_i$ changes,
- but kernel matrices only depend on $X$ and hyperparameters,
- so kernel factorizations can be shared whenever hyperparameters are shared,
- and even in a small hyper-pool, the expensive matrix work is reused candidate-by-candidate.

This is why particle-specific discrepancy is practical here despite many particles.

## 11. Recommended maintenance checklist

Whenever rolled-CUSUM code changes, update this document with:

1. the exact new config switch or method name,
2. whether it changes PF, BOCPD scoring, restart, or refresh only,
3. the mathematical object being approximated,
4. the fallback / backward-compatible behavior,
5. and at least one recommended smoke-test command.

## 12. Example smoke-test style

Use the `jumpGP` environment.
Prefer a short synthetic run that exercises the modified branch, for example a small `run_one_slope(...)` call with one method entry.

The goal of the smoke test is not to validate final metrics.
It is to verify that:

- the new interface wires correctly,
- prediction shape logic is correct,
- and the modified branch can run through at least one prediction-update cycle.


## 13. Runner Invocation Reference

These are the current recommended command-line entrypoints for the synthetic rolled-CUSUM experiments.
Use the `jumpGP` environment.

### 13.1 Gradual drift main/ablation/appendix

Runner:

- `calib.run_synthetic_slope_deltaCmp`

Supported profiles:

- `--profile main`
  - current default behavior for the slope script
- `--profile ablation`
  - fixed gradual-drift ablation setting
  - `mode=1`, `slope=0.0025`, `batch_size=20`, `seeds=[101,202,303,404,505]`
  - writes `ablation_gradual_metrics.csv/.xlsx`
- `--profile appendix`
  - fixed gradual appendix comparison for shared vs particle-specific discrepancy
  - writes `appendix_extension_gradual_metrics.csv/.xlsx`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile main --out_dir figs/slope_main
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile ablation --out_dir figs/slope_ablation
conda run -n jumpGP python -m calib.run_synthetic_slope_deltaCmp --profile appendix --out_dir figs/slope_appendix
```

### 13.2 Sudden change main/ablation

Runner:

- `calib.run_synthetic_suddenCmp_tryThm`

Supported profiles:

- `--profile main`
  - current default sudden-change grid behavior
- `--profile ablation`
  - fixed sudden-change ablation setting
  - `seg_len_L=80`, `delta_mag=2.0`, `batch_size=20`, `seeds=[101,202,303]`
  - writes `ablation_sudden_metrics.csv/.xlsx`
  - writes `ablation_sudden_restart_stats.csv/.xlsx`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile main --out_dir figs/sudden_main
conda run -n jumpGP python -m calib.run_synthetic_suddenCmp_tryThm --profile ablation --out_dir figs/sudden_ablation
```

### 13.3 Mixed gradual-plus-sudden runner

Runner:

- `calib.run_synthetic_mixed_thetaCmp`

Supported profiles and preview entrypoints:

- `--profile main`
  - mixed scenario grid over `drift_scale x jump_scale x seed`
  - current defaults: `drift_scale in {0.006, 0.009}`, `jump_scale in {0.28, 0.38}`, `batch_size=20`
- `--profile ablation`
  - fixed mixed ablation setting
  - `drift_scale=0.008`, `jump_scale=0.35`, `batch_size=20`, `seeds=[101,202,303,404,505]`
- `--profile preview`
  - only generate true theta trajectories, no method run
- `--preview_only`
  - backdoor mode for visually checking the latent theta paths before launching the experiment
  - writes `mixed_theta_preview.png` and `mixed_theta_preview.csv`

Example commands:

```bash
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --preview_only --out_dir figs/mixed_preview
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile main --out_dir figs/mixed_main
conda run -n jumpGP python -m calib.run_synthetic_mixed_thetaCmp --profile ablation --out_dir figs/mixed_ablation
```

### 13.4 Output conventions

For these runners, `--out_dir` is the main experiment artifact directory.
Typical outputs include:

- per-run `.pt` files,
- aggregate `all_metrics.csv/.xlsx`,
- `restart_mode_stats.csv/.xlsx` when restart histories are available,
- profile-specific ablation tables,
- plots saved under the chosen output directory.

When a runner gains a new profile or a new export file, update this section in the same code change.
