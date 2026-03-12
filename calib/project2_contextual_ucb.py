from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from calib.project2_common import GlobalTransformSep, NNModelTorchStd
else:
    from .project2_common import GlobalTransformSep, NNModelTorchStd


@dataclasses.dataclass(frozen=True)
class DecisionSpace:
    q_range: Tuple[int, int] = (50, 100)
    r_range: Tuple[int, int] = (50, 100)
    w_range: Tuple[int, int] = (10, 15)
    m1_range: Tuple[int, int] = (1, 3)
    m2_range: Tuple[int, int] = (1, 3)

    def full_grid(self) -> np.ndarray:
        q_vals = np.arange(self.q_range[0], self.q_range[1] + 1, dtype=np.int64)
        r_vals = np.arange(self.r_range[0], self.r_range[1] + 1, dtype=np.int64)
        w_vals = np.arange(self.w_range[0], self.w_range[1] + 1, dtype=np.int64)
        m1_vals = np.arange(self.m1_range[0], self.m1_range[1] + 1, dtype=np.int64)
        m2_vals = np.arange(self.m2_range[0], self.m2_range[1] + 1, dtype=np.int64)

        QQ, RR, WW, M11, M22 = np.meshgrid(q_vals, r_vals, w_vals, m1_vals, m2_vals, indexing="ij")
        grid = np.column_stack(
            [
                QQ.reshape(-1),
                RR.reshape(-1),
                WW.reshape(-1),
                M11.reshape(-1),
                M22.reshape(-1),
            ]
        )
        return grid.astype(np.int64)

    def clip(self, X_decision: np.ndarray) -> np.ndarray:
        X = np.asarray(X_decision, dtype=np.int64).copy()
        X[:, 0] = np.clip(X[:, 0], *self.q_range)
        X[:, 1] = np.clip(X[:, 1], *self.r_range)
        X[:, 2] = np.clip(X[:, 2], *self.w_range)
        X[:, 3] = np.clip(X[:, 3], *self.m1_range)
        X[:, 4] = np.clip(X[:, 4], *self.m2_range)
        return X

    def decision_to_model(self, X_decision: np.ndarray) -> np.ndarray:
        X = np.asarray(X_decision)
        if X.ndim == 1:
            X = X[None, :]
        q, r, w, m1, m2 = [X[:, i] for i in range(5)]
        return np.column_stack([w, r, m1, m2, q]).astype(np.float64)


@dataclasses.dataclass
class SinThetaSchedule:
    num_batches: int = 80
    hold_batches: int = 3
    center: float = 11.5
    amplitude: float = 8.5
    phase: float = 0.0
    jump_batch: Optional[int] = None
    jump_target: Optional[float] = None
    theta_min: float = 3.0
    theta_max: float = 21.0

    def _base_theta_at(self, batch_idx: int) -> float:
        control_steps = max(int(math.ceil(self.num_batches / self.hold_batches)), 1)
        control_idx = min(batch_idx // self.hold_batches, control_steps - 1)
        angle = 2.0 * math.pi * control_idx / control_steps + self.phase
        return float(self.center + self.amplitude * math.sin(angle))

    def theta_at(self, batch_idx: int) -> float:
        theta = self._base_theta_at(batch_idx)

        if self.jump_batch is not None and self.jump_target is not None and batch_idx >= self.jump_batch:
            jump_offset = float(self.jump_target - self._base_theta_at(self.jump_batch))
            theta = theta + jump_offset

        return float(np.clip(theta, self.theta_min, self.theta_max))


@dataclasses.dataclass
class PlatformThetaSchedule:
    levels: Tuple[float, ...] = (3.0, 11.5, 20.0, 30.0)
    block_batches: int = 20
    noise_std: float = 0.5
    theta_min: float = 0.1
    theta_max: float = 30.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def theta_at(self, batch_idx: int) -> float:
        if self.block_batches <= 0:
            raise ValueError("block_batches must be positive.")
        level_idx = (batch_idx // self.block_batches) % len(self.levels)
        mean_theta = float(self.levels[level_idx])
        theta = float(self._rng.normal(loc=mean_theta, scale=self.noise_std))
        return float(np.clip(theta, self.theta_min, self.theta_max))


def _unique_rows(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.reshape(0, 5)
    return np.unique(np.asarray(X, dtype=np.int64), axis=0)


def _topk(mu: np.ndarray, sigma: np.ndarray, score: np.ndarray, X: np.ndarray, top_k: int) -> Dict[str, np.ndarray]:
    order = np.argsort(-score)
    keep = order[:top_k]
    return {
        "X_decision": X[keep].astype(np.int64),
        "mu_raw": mu[keep],
        "sigma_raw": sigma[keep],
        "ucb": score[keep],
    }


@dataclasses.dataclass
class DiscreteContextualUCBOptimizer:
    decision_space: DecisionSpace
    top_k: int = 10
    beta: float = 0.3
    random_pool_size: int = 4096
    local_pool_size: int = 1024
    eval_chunk_size: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._full_grid = self.decision_space.full_grid()

    def _sample_local_candidates(self, center_points: np.ndarray) -> np.ndarray:
        if center_points is None or len(center_points) == 0 or self.local_pool_size <= 0:
            return np.empty((0, 5), dtype=np.int64)

        per_center = max(self.local_pool_size // len(center_points), 1)
        locals_list = []
        for x in np.asarray(center_points, dtype=np.int64):
            draws = np.tile(x, (per_center, 1))
            draws[:, 0] += self.rng.integers(-6, 7, size=per_center)
            draws[:, 1] += self.rng.integers(-6, 7, size=per_center)
            draws[:, 2] += self.rng.integers(-1, 2, size=per_center)
            draws[:, 3] += self.rng.integers(-1, 2, size=per_center)
            draws[:, 4] += self.rng.integers(-1, 2, size=per_center)
            locals_list.append(self.decision_space.clip(draws))
        return _unique_rows(np.vstack(locals_list))

    def _build_candidate_pool(self, previous_top: Optional[np.ndarray] = None, include_full_grid: bool = False) -> np.ndarray:
        if include_full_grid or len(self._full_grid) <= self.random_pool_size:
            pool = self._full_grid
        else:
            idx = self.rng.choice(len(self._full_grid), size=self.random_pool_size, replace=False)
            pool = self._full_grid[idx]

        if previous_top is not None and len(previous_top) > 0:
            local_pool = self._sample_local_candidates(previous_top)
            pool = np.vstack([pool, previous_top, local_pool])

        return _unique_rows(pool)

    def _score_with_nn(
        self,
        nn_std: NNModelTorchStd,
        gt: GlobalTransformSep,
        X_decision: np.ndarray,
        theta_raw: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_model = self.decision_space.decision_to_model(X_decision)
        X_s = gt.X_base_to_s(X_model)
        theta_s = gt.theta_raw_to_s(np.full(len(X_s), theta_raw, dtype=np.float64)).reshape(-1, 1)
        X_full_s = np.concatenate([X_s, theta_s], axis=1)
        mu_s = nn_std.predict_y_s_from_Xfull_s(X_full_s)
        mu_raw = gt.y_s_to_raw(mu_s)
        sigma_raw = np.zeros_like(mu_raw)
        return mu_raw, sigma_raw

    def _score_with_calibrator(
        self,
        calib,
        gt: GlobalTransformSep,
        X_decision: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu_list = []
        sigma_list = []
        X_model = self.decision_space.decision_to_model(X_decision)
        X_s = gt.X_base_to_s(X_model)

        for start in range(0, len(X_s), self.eval_chunk_size):
            stop = min(start + self.eval_chunk_size, len(X_s))
            batch = torch.tensor(X_s[start:stop], dtype=torch.float64)
            pred = calib.predict_batch(batch)
            mu_s = pred["mu"].detach().cpu().numpy()
            var_s = pred["var"].detach().cpu().numpy()
            mu_raw, sigma_raw = gt.y_s_stats_to_raw(mu_s, var_s)
            mu_list.append(mu_raw)
            sigma_list.append(sigma_raw)

        return np.concatenate(mu_list), np.concatenate(sigma_list)

    def propose_with_nn(
        self,
        nn_std: NNModelTorchStd,
        gt: GlobalTransformSep,
        theta_raw: float,
        previous_top: Optional[np.ndarray] = None,
        include_full_grid: bool = False,
    ) -> Dict[str, np.ndarray]:
        pool = self._build_candidate_pool(previous_top=previous_top, include_full_grid=include_full_grid)
        mu_raw, sigma_raw = self._score_with_nn(nn_std, gt, pool, theta_raw)
        score = mu_raw + self.beta * sigma_raw
        return _topk(mu_raw, sigma_raw, score, pool, self.top_k)

    def propose_with_calibrator(
        self,
        calib,
        nn_std: NNModelTorchStd,
        gt: GlobalTransformSep,
        context_theta_raw: float,
        previous_top: Optional[np.ndarray] = None,
        include_full_grid: bool = False,
    ) -> Dict[str, np.ndarray]:
        if len(getattr(calib.bocpd, "experts", [])) == 0:
            return self.propose_with_nn(
                nn_std=nn_std,
                gt=gt,
                theta_raw=context_theta_raw,
                previous_top=previous_top,
                include_full_grid=include_full_grid,
            )

        pool = self._build_candidate_pool(previous_top=previous_top, include_full_grid=include_full_grid)
        mu_raw, sigma_raw = self._score_with_calibrator(calib, gt, pool)
        score = mu_raw + self.beta * sigma_raw
        return _topk(mu_raw, sigma_raw, score, pool, self.top_k)
