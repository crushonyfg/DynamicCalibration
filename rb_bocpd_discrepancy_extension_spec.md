# RB-BOCPD Discrepancy Extension — Working Specification

## 0. Purpose

This document specifies the current recommended design discussed in our thread, for later implementation in code.

The goal is not to change the PF weighting rule.
The goal is to do a more theoretical pure version. The idea is based on "our current approach of estimating theta and delta is very similar to the Rao-Blackwellized Particle Filter, because delta posterior is analytically available conditioned on theta. We could do particle filter on theta, conditioned on which we could get the conditional posterior of delta analytically. Change point detections could work on the changes in theta and delta both."

while keeping the PF calibration pipeline identifiable and stable.

This document is written to avoid the main source of confusion:

**PF weights stay discrepancy-free.**
Discrepancy is introduced after PF weighting, and mainly enters the BOCPD expert likelihood / attribution / restart side.

---

## 1. High-level design summary

We separate the system into two layers.

### 1.1 PF layer

PF only tracks the calibration latent state `theta_t`.

Particle weights are always updated using the no-discrepancy likelihood:

$$
w_t^{(i)} \propto w_{t-1}^{(i)} \; p\!\left(Y_t \mid X_t,\theta_t^{(i)}, \text{simulator only}\right).
$$

Equivalently, if the simulator predictive is Gaussian,

$$
Y_t \mid X_t,\theta_t^{(i)} \sim \mathcal N\!\big(\mu_s(X_t,\theta_t^{(i)}), \Sigma_s(X_t,\theta_t^{(i)}) + \sigma^2 I\big),
$$

then

$$
w_t^{(i)} \propto w_{t-1}^{(i)}
\; \mathcal N\!\Big(
Y_t;
\mu_s(X_t,\theta_t^{(i)}),
\Sigma_s(X_t,\theta_t^{(i)}) + \sigma^2 I
\Big).
$$

Important:
PF weights do not use discrepancy.
This is deliberate, to preserve theta identifiability.

---

### 1.2 BOCPD / restart layer

BOCPD is allowed to use richer predictive laws that include discrepancy.

We introduce a joint latent view for BOCPD:

$$
x_t = (\theta_t, \phi_t),
$$

where:

- $\theta_t$: calibration state tracked by PF,
- $\phi_t$: discrepancy-related state.

The discrepancy-related state $\phi_t$ can be instantiated in several ways:

- GP discrepancy posterior / GP latent values,
- basis coefficients $\beta_t$,
- inducing-point values,
- or another low-dimensional discrepancy representation.

BOCPD expert scoring then uses a joint predictive law based on $(\theta_t, \phi_t)$, but PF itself remains discrepancy-free.

---

## 2. Core modeling objects

We observe batches $B_t = (X_t, Y_t)$, where:

- $X_t$: batch inputs / design points at time $t$,
- $Y_t$: batch outputs at time $t$.

We use the observation model

$$
Y_t = y^s(X_t, \theta_t) + \delta_t(X_t) + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal N(0,\sigma^2 I).
$$

Here:

- $y^s(X_t,\theta_t)$: simulator / surrogate prediction,
- $\delta_t(\cdot)$: discrepancy function,
- $\varepsilon_t$: observation noise.

We define the joint latent state as

$$
x_t = (\theta_t,\phi_t),
$$

with $\phi_t$ representing discrepancy in finite- or infinite-dimensional form.

---

## 3. What is fixed, and what is changed

### 3.1 Fixed

PF weighting remains exactly as before:

- only `theta_t`,
- only no-discrepancy likelihood,
- discrepancy never enters particle weights.

This means:

- no RBPF on PF weights,
- no particle-specific discrepancy in the PF update,
- no change to the identifiability-preserving PF channel.

### 3.2 Changed

BOCPD expert likelihood and restart attribution are enriched using discrepancy-aware predictive laws.

So the extension is:

- not "replace the whole method by full joint PF",
- not "change PF into RBPF for weights",

but rather:

keep PF as nodiscrepancy,
and make BOCPD scoring / attribution more joint-latent and discrepancy-aware.

---

## 4. Expert representation for BOCPD

For an expert $e$, let its segment start be $s_e$.
This expert represents the hypothesis:

No change point occurred on $[s_e,t]$.

For BOCPD, define the expert-specific history:

$$
\mathcal D_{e,t-1} = \{B_{s_e}, B_{s_e+1}, \dots, B_{t-1}\}.
$$

The BOCPD side uses an expert-specific predictive law based on this history.

The general joint-latent predictive law is

$$
q^x_{e,t}(B_t)
=
p(B_t \mid \mathcal D_{e,t-1})
=
\int p(B_t \mid x_t)\, p(x_t \mid \mathcal D_{e,t-1})\, dx_t.
$$

This is the correct BOCPD likelihood object.

Important clarification:
The BOCPD likelihood is not just $p(Y_t \mid \theta_t,\phi_t)$.
It is the predictive distribution after integrating over the expert’s latent uncertainty.

---

## 5. Semi-joint factorization used here

Because PF only tracks $\theta_t$, and discrepancy is learned afterward, we use the following factorization on the BOCPD side:

$$
q^x_{e,t}(B_t)
=
\int
p(Y_t \mid X_t,\theta_t,\phi_t)\,
p(\phi_t \mid \theta_t, \mathcal D_{e,t-1})\,
p(\theta_t \mid \mathcal D_{e,t-1})\,
d\phi_t\, d\theta_t.
$$

This is the key formula for implementation.

Interpretation:

- $p(\theta_t \mid \mathcal D_{e,t-1})$: comes from PF particles / expert-specific particle cloud,
- $p(\phi_t \mid \theta_t, \mathcal D_{e,t-1})$: discrepancy posterior conditioned on a given $\theta_t$,
- $p(Y_t \mid X_t,\theta_t,\phi_t)$: observation model.

Then BOCPD integrates both sources of uncertainty.

---

## 6. Particle approximation of the BOCPD predictive law

Assume the PF side gives, for expert $e$,

$$
p(\theta_t \mid \mathcal D_{e,t-1})
\approx
\sum_{i=1}^N w_{e,t-1}^{(i)} \, \delta_{\theta_{e,t}^{(i)}}.
$$

Then

$$
q^x_{e,t}(B_t)
\approx
\sum_{i=1}^N
w_{e,t-1}^{(i)}
\int
p(Y_t \mid X_t,\theta_{e,t}^{(i)},\phi_t)\,
p(\phi_t \mid \theta_{e,t}^{(i)}, \mathcal D_{e,t-1})
\, d\phi_t.
$$

This is the main implementation-level BOCPD likelihood.

Again: this does not feed back into PF weights.

---

## 7. Option A: delta as GP discrepancy

### 7.1 Model

For a fixed particle $\theta^{(i)}$, define residuals on the expert history:

$$
r^{(i)}_{e,t-1}
=
Y_{s_e:t-1}
-
y^s(X_{s_e:t-1}, \theta^{(i)}).
$$

Assume

$$
\delta(\cdot) \sim \mathcal{GP}(0, k_\psi(\cdot,\cdot)).
$$

Then conditioned on $\theta^{(i)}$, discrepancy learning becomes standard GP regression on the residuals.

### 7.2 GP posterior conditioned on a particle

Let

$$
K_e = K(X_{s_e:t-1}, X_{s_e:t-1}),
\quad
k_t = K(X_t, X_{s_e:t-1}).
$$

Then for fixed $\theta^{(i)}$,

$$
\delta_t(\cdot)\mid \theta^{(i)}, \mathcal D_{e,t-1}
\sim
\mathcal{GP}\big(
m^\delta_{e,t-1}(\cdot;\theta^{(i)}),
C^\delta_{e,t-1}(\cdot,\cdot;\theta^{(i)})
\big),
$$

with

$$
m^\delta_{e,t-1}(X_t;\theta^{(i)})
=
k_t (K_e + \sigma^2 I)^{-1} r^{(i)}_{e,t-1},
$$

and

$$
C^\delta_{e,t-1}(X_t,X_t;\theta^{(i)})
=
K(X_t,X_t)
-
k_t (K_e + \sigma^2 I)^{-1} k_t^\top.
$$

### 7.3 GP-integrated likelihood for one particle

For fixed $\theta^{(i)}$, integrating out the GP discrepancy gives

$$
\int
p(Y_t \mid X_t,\theta^{(i)},\delta_t)\,
p(\delta_t \mid \theta^{(i)}, \mathcal D_{e,t-1})
\, d\delta_t
=
\mathcal N\!\Big(
Y_t;
\mu_s(X_t,\theta^{(i)}) + m^\delta_{e,t-1}(X_t;\theta^{(i)}),
\Sigma_s(X_t,\theta^{(i)}) + C^\delta_{e,t-1}(X_t,X_t;\theta^{(i)}) + \sigma^2 I
\Big).
$$

Therefore the BOCPD expert likelihood becomes

$$
q^{\text{GP}}_{e,t}(B_t)
\approx
\sum_{i=1}^N
w_{e,t-1}^{(i)}
\,
\mathcal N\!\Big(
Y_t;
\mu_s(X_t,\theta_{e,t}^{(i)}) + m^\delta_{e,t-1}(X_t;\theta_{e,t}^{(i)}),
\Sigma_s(X_t,\theta_{e,t}^{(i)}) + C^\delta_{e,t-1}(X_t,X_t;\theta_{e,t}^{(i)}) + \sigma^2 I
\Big).
$$

This is the most direct GP-based semi-joint BOCPD likelihood.

---

## 8. Option B: delta as basis expansion / beta-state

A more implementation-friendly alternative is to use a finite-dimensional discrepancy model:

$$
\delta_t(x) = \Phi(x)^\top \beta_t,
$$

where:

- $\Phi(x)$: basis vector,
- $\beta_t$: discrepancy coefficient state.

Then the joint latent becomes

$$
x_t = (\theta_t,\beta_t).
$$

If $\beta_t$ has Gaussian dynamics, for example

$$
\beta_t = A \beta_{t-1} + \eta_t,
\qquad
\eta_t \sim \mathcal N(0,Q_\beta),
$$

and the observation model is

$$
Y_t = y^s(X_t,\theta_t) + \Phi(X_t)\beta_t + \varepsilon_t,
$$

then for fixed $\theta^{(i)}$, the discrepancy side is linear-Gaussian.

This yields a standard Rao–Blackwellized structure on the BOCPD side:

- PF over $\theta_t$,
- Kalman-style update over $\beta_t$.

### 8.1 One-particle predictive with beta-state

Suppose for fixed $\theta^{(i)}$,

$$
\beta_t \mid \theta^{(i)}, \mathcal D_{e,t-1}
\sim
\mathcal N(m^\beta_{e,t-1}(\theta^{(i)}), P^\beta_{e,t-1}(\theta^{(i)})).
$$

Then integrating out $\beta_t$,

$$
p(Y_t \mid X_t,\theta^{(i)},\mathcal D_{e,t-1})
=
\mathcal N\!\Big(
Y_t;
\mu_s(X_t,\theta^{(i)}) + \Phi(X_t)m^\beta_{e,t-1}(\theta^{(i)}),
\Sigma_s(X_t,\theta^{(i)}) + \Phi(X_t)P^\beta_{e,t-1}(\theta^{(i)})\Phi(X_t)^\top + \sigma^2 I
\Big).
$$

So the BOCPD expert likelihood becomes

$$
q^{\beta}_{e,t}(B_t)
\approx
\sum_{i=1}^N
w_{e,t-1}^{(i)}
\,
\mathcal N\!\Big(
Y_t;
\mu_s(X_t,\theta_{e,t}^{(i)}) + \Phi(X_t)m^\beta_{e,t-1}(\theta_{e,t}^{(i)}),
\Sigma_s(X_t,\theta_{e,t}^{(i)}) + \Phi(X_t)P^\beta_{e,t-1}(\theta_{e,t}^{(i)})\Phi(X_t)^\top + \sigma^2 I
\Big).
$$

This is lower-dimensional and easier to implement than a full GP.

---

## 10. BOCPD recursion

Let $r_t$ be the run length or expert index state.

BOCPD uses the predictive likelihood

$$
q_{e,t}(B_t)
$$

for each expert $e$. Then the recursion is the usual one:

$$
p(r_t = 0 \mid B_{1:t})
\propto
\sum_{e}
H_e \, q_{e,t}(B_t)\, p(r_{t-1}=e \mid B_{1:t-1}),
$$

$$
p(r_t = e+1 \mid B_{1:t})
\propto
(1-H_e)\, q_{e,t}(B_t)\, p(r_{t-1}=e \mid B_{1:t-1}),
$$

where $H_e$ is the hazard or restart probability.

For implementation, the exact BOCPD bookkeeping can stay as in the current codebase.
The main replacement is the expert predictive likelihood $q_{e,t}$.

---

## 11. Attribution: theta-change vs delta-change vs both-change

This is the main reason to introduce the semi-joint discrepancy-aware BOCPD extension.

Let:

- $a$: current anchor expert,
- $c$: candidate expert.

We define three predictive comparisons.

### 11.1 Anchor predictive

$$
q_a(B_t)
$$

This is the predictive law under the current anchor expert.

### 11.2 Full candidate predictive

$$
q_c^{\text{full}}(B_t)
$$

This uses candidate theta-history and candidate discrepancy-history.

This captures the total gain from switching anchor to candidate.

Define

$$
\Delta_{\text{full}}
=
\log q_c^{\text{full}}(B_t) - \log q_a(B_t).
$$

### 11.3 Theta-only candidate predictive

Construct a hybrid predictive law:

- theta from candidate,
- discrepancy from anchor.

Call it

$$
q_{c\leftarrow a}^{\theta}(B_t).
$$

Then define

$$
\Delta_\theta
=
\log q_{c\leftarrow a}^{\theta}(B_t) - \log q_a(B_t).
$$

Interpretation: gain explained by changing theta while holding discrepancy fixed.

### 11.4 Delta-only candidate predictive

Construct a second hybrid predictive law:

- theta from anchor,
- discrepancy from candidate.

Call it

$$
q_{c\leftarrow a}^{\delta}(B_t).
$$

Then define

$$
\Delta_\delta
=
\log q_{c\leftarrow a}^{\delta}(B_t) - \log q_a(B_t).
$$

Interpretation: gain explained by changing discrepancy while holding theta fixed.

### 11.5 Decision logic

These attribution scores can be used to diagnose:

- theta-change only:
  $$
  \Delta_\theta \text{ large}, \quad \Delta_\delta \text{ small}
  $$

- delta-change only:
  $$
  \Delta_\delta \text{ large}, \quad \Delta_\theta \text{ small}
  $$

- both changed:
  $$
  \Delta_\theta \text{ and } \Delta_\delta \text{ both large}
  $$

This is the intended BOCPD-side interpretation of the joint latent extension.

---

## 12. Restart / action logic

This extension supports several downstream actions.

### 12.1 Full restart
Use when theta and discrepancy both change.

### 12.2 Theta-priority restart
Use when theta-change dominates.

### 12.3 Delta-only refresh
Use when discrepancy-change dominates.

This fits naturally with the existing dual-restart philosophy.

---

## 13. Single-step gate remains compatible

The previously discussed single-step standardized gate can still be used as a separate auxiliary mechanism.

For 1D theta, define

$$
z_t
=
\frac{|m_t - m_{t-1}|}{\sqrt{\operatorname{Var}_{t-1} + \varepsilon}},
\qquad
d_t = z_t^2.
$$

Then examples of possible discrepancy-refresh triggers are:

- $z_t > 2.5$ for two consecutive steps,
- or $z_t > 3.0$ once.

Important:
this gate does not replace BOCPD.
It remains an auxiliary discrepancy-memory trigger.

---

## 14. Clear implementation boundaries

These points must remain explicit to avoid confusion.

### 14.1 PF weights
PF weights are still updated using nodiscrepancy likelihood only.

No discrepancy term enters the PF weight update.

### 14.2 BOCPD likelihood
BOCPD expert scoring uses the discrepancy-aware predictive law $q_{e,t}$.

This is where the new semi-joint latent logic enters.

### 14.3 Prediction
Final prediction can still combine simulator prediction and discrepancy correction.

### 14.4 Attribution / restart
Attribution scores $\Delta_\theta, \Delta_\delta, \Delta_{\text{full}}$ should be computed from discrepancy-aware BOCPD predictive laws, not from PF weights.
