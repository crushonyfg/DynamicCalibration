# =============================================================
# file: calib/moe_calibrator.py
# =============================================================
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional, Tuple
import math
import torch

from .configs import CalibrationConfig, ModelConfig, PFConfig
from .pf import ParticleFilter
from .particles import ParticleSet
from .delta_gp import OnlineGPState
from .kernels import make_kernel
from .emulator import Emulator
from .likelihood import loglik_and_grads, predictive_stats


@dataclass
class MoEExpert:
    """
    一个 expert = (PF over θ) + δ-GP + 自己的历史数据。
    注意：这里不复用 BOCPD 里的 Expert，避免和 run-length / log_mass 逻辑耦合。
    """
    id: int
    pf: ParticleFilter
    delta_state: OnlineGPState
    X_hist: torch.Tensor  # [M, dx]
    y_hist: torch.Tensor  # [M]
    count_assigned: int = 0          # 被分配过多少个 batch
    last_active_batch: int = -1      # 上一次被选中的 batch index

    segments: List[Tuple[int, int]] = None   # <--- 新增字段，用于记录数据片段来源

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


class OnlineMoECalibrator:
    """
    Online Mixture-of-Experts Calibrator (DP/CRP-style hard assignment, per-batch).

    设计要点：
      - 维护 experts 列表，每个 expert 是一个 PF + δ-GP。
      - 每个 batch 只更新一个 expert：
          * 先用「上一轮 expert」做一个适用性测试（likelihood test）。
          * 如果上一轮 expert 仍然 OK，就继续用它。
          * 否则，用 CRP (Chinese Restaurant Process) 在已有 experts + 一个“新 expert”候选之间做选择：
                p(z = k) ∝ n_k * p(y_batch | expert k)
                p(z = new) ∝ α * p(y_batch | new)
      - predict_batch(X_batch) 默认使用当前 active expert 做预测（和你描述的一致：
        “先预测，再 update（决定 expert）”，预测用上一轮的 expert）。
    """

    def __init__(
        self,
        calib_cfg: CalibrationConfig,
        emulator: Emulator,
        prior_sampler: Callable[[int], torch.Tensor],
        alpha: float = 1.0,             # CRP 浓度参数
        bf_threshold: float = 2.0,      # log-Bayes-factor 阈值（>0 表示要有明显优势才换 expert）
        max_experts: Optional[int] = None,   # 最多保留多少个 experts（防止爆炸）
    ):
        self.cfg = calib_cfg
        self.emulator = emulator
        self.prior_sampler = prior_sampler

        self.device = calib_cfg.model.device
        self.dtype = calib_cfg.model.dtype
        self.model_cfg: ModelConfig = calib_cfg.model
        self.pf_cfg: PFConfig = calib_cfg.pf

        self.alpha = float(alpha)
        self.bf_threshold = float(bf_threshold)
        self.max_experts = int(max_experts) if max_experts is not None else int(calib_cfg.bocpd.max_experts)

        self.experts: List[MoEExpert] = []
        self.current_expert_idx: Optional[int] = None  # 当前 active expert index
        self._next_expert_id: int = 0
        self._num_batches_seen: int = 0
        self._t_obs: int = 0  # 观测总数（所有 batch 的样本数和）
        # self._verbose: bool = verbose

    # ------------------------------------------------------------------
    # 内部工具函数
    # ------------------------------------------------------------------
    def _init_empty_delta_state(self, dx: int) -> OnlineGPState:
        """复制 BOCPD 中的初始化逻辑：构造空的 OnlineGPState。"""
        kernel = make_kernel(self.cfg.model.delta_kernel)
        return OnlineGPState(
            X=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y=torch.empty(0, dtype=self.dtype, device=self.device),
            kernel=kernel,
            noise=self.cfg.model.delta_kernel.noise,
            update_mode="exact_full",
            hyperparam_mode="fit",
        )

    def _spawn_new_expert(self, dx: int) -> MoEExpert:
        """从先验采样 θ，创建一个新的 expert。"""
        delta_state = self._init_empty_delta_state(dx, )
        pf = ParticleFilter.from_prior(
            self.prior_sampler,
            self.pf_cfg,
            device=self.device,
            dtype=self.dtype,
        )
        e = MoEExpert(
            id=self._next_expert_id,
            pf=pf,
            delta_state=delta_state,
            X_hist=torch.empty(0, dx, dtype=self.dtype, device=self.device),
            y_hist=torch.empty(0, dtype=self.dtype, device=self.device),
            count_assigned=0,
            last_active_batch=-1,
        )
        self._next_expert_id += 1
        self.experts.append(e)
        return e

    def _append_hist_batch(self, e: MoEExpert, X_batch: torch.Tensor, Y_batch: torch.Tensor, max_len: int) -> None:
        """和 restart_bocpd 中一样，维护一段有限长度的原始历史 + 同步 delta_state."""
        old_t = self._t_obs                    # batch 之前的全局 t
        new_t = self._t_obs + X_batch.shape[0] # batch 之后的全局 t
        e.segments.append((old_t, new_t))
        if e.X_hist.numel() == 0:
            e.X_hist = X_batch.clone()
            e.y_hist = Y_batch.clone()
        else:
            e.X_hist = torch.cat([e.X_hist, X_batch], dim=0)
            e.y_hist = torch.cat([e.y_hist, Y_batch], dim=0)

        if e.X_hist.shape[0] > max_len:
            e.X_hist = e.X_hist[-max_len:, :]
            e.y_hist = e.y_hist[-max_len:]

            if e.delta_state.X.shape[0] > max_len:
                e.delta_state.X = e.delta_state.X[-max_len:, :]
                e.delta_state.y = e.delta_state.y[-max_len:]
                e.delta_state._recompute_cache_full()

    def _predictive_log_mixture_for_expert(
        self,
        e: MoEExpert,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
    ) -> float:
        """
        计算某个 expert 对当前 batch 的 mixture predictive log-likelihood（还没更新 PF 权重前的 UMP）。
        和 restart_bocpd.ump_batch 里完全一致。
        """
        ps: ParticleSet = e.pf.particles
        info = loglik_and_grads(
            Y_batch,
            X_batch,
            ps,
            self.emulator,
            e.delta_state,
            self.model_cfg.rho,
            self.model_cfg.sigma_eps,
            need_grads=False,
        )
        loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)  # [N]
        logmix = torch.logsumexp(ps.logw + loglik, dim=0)
        return float(logmix)

    def _compute_new_expert_log_pred(
        self,
        dx: int,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
    ) -> float:
        """
        对 “新 expert” 候选，基于先验 θ 和空的 δ-GP，近似计算一个 predictive log-likelihood。
        不会修改全局 state，只是用临时 PF + delta_state 做评分。
        """
        tmp_delta = self._init_empty_delta_state(dx)
        tmp_pf = ParticleFilter.from_prior(
            self.prior_sampler, self.pf_cfg, device=self.device, dtype=self.dtype
        )
        ps: ParticleSet = tmp_pf.particles
        info = loglik_and_grads(
            Y_batch,
            X_batch,
            ps,
            self.emulator,
            tmp_delta,
            self.model_cfg.rho,
            self.model_cfg.sigma_eps,
            need_grads=False,
        )
        loglik = torch.clamp(info["loglik"], min=-1e10, max=1e10)
        logmix = torch.logsumexp(ps.logw + loglik, dim=0)
        return float(logmix)

    def _prune_experts(self) -> None:
        """
        限制 experts 数量，避免无限增长。
        策略：
          - 总数 > max_experts 时，保留：
              * 当前 active expert
              * 其余中 count_assigned 最大的若干个
        """
        if len(self.experts) <= self.max_experts:
            return
        if self.current_expert_idx is None:
            # 没有 active 的话，直接按 count_assigned 排序取前 K
            self.experts.sort(key=lambda e: e.count_assigned, reverse=True)
            self.experts = self.experts[: self.max_experts]
            # 重排后 current_expert_idx 置空，由下一次 step_batch 再设定
            self.current_expert_idx = 0 if self.experts else None
            return

        current_id = self.experts[self.current_expert_idx].id
        # 按 count_assigned 排序
        sorted_exps = sorted(self.experts, key=lambda e: e.count_assigned, reverse=True)

        # 当前 expert 必须保留
        kept: List[MoEExpert] = []
        current_first: Optional[MoEExpert] = None
        for e in sorted_exps:
            if e.id == current_id:
                current_first = e
                break
        if current_first is None:
            # 理论上不会发生，但防御性写法
            current_first = sorted_exps[0]

        kept.append(current_first)
        for e in sorted_exps:
            if e.id == current_id:
                continue
            if len(kept) >= self.max_experts:
                break
            kept.append(e)

        self.experts = kept
        # 更新 current_expert_idx
        for i, e in enumerate(self.experts):
            if e.id == current_id:
                self.current_expert_idx = i
                break

    # ------------------------------------------------------------------
    # 对外 API：预测
    # ------------------------------------------------------------------
    def predict_batch(self, X_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        根据“当前 active expert” 对一个 batch 做预测。
        这对应你实验里的：先 predict，再在 step_batch 里根据 Y 决定是不是切换 expert。
        """
        X_batch = X_batch.to(self.device, self.dtype)
        batch_size = X_batch.shape[0]

        # 如果还没有任何 expert，就从先验创建一个（完全先验预测）
        if len(self.experts) == 0:
            dx = X_batch.shape[1]
            self._spawn_new_expert(dx)
            self.current_expert_idx = 0

        if self.current_expert_idx is None:
            self.current_expert_idx = 0

        e = self.experts[self.current_expert_idx]

        # 和 OnlineBayesCalibrator.predict_batch 中的单 expert 部分一致
        mu_eta, var_eta = self.emulator.predict(X_batch, e.pf.particles.theta)  # [b, N]
        mu_delta, var_delta = e.delta_state.predict(X_batch)                    # [b]
        mu, var = predictive_stats(
            self.model_cfg.rho,
            mu_eta,
            var_eta,
            mu_delta,
            var_delta,
            self.model_cfg.sigma_eps,
        )  # [b,N]

        w = e.pf.particles.weights()[None, :]                                   # [1,N]
        mu_mix = (w * mu).sum(dim=1)                                            # [b]
        var_mix = (w * (var + mu ** 2)).sum(dim=1) - mu_mix ** 2               # [b]

        return {"mu": mu_mix, "var": var_mix}

    def predict(self, x_next: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        单点预测（如果你以后想在 streaming 模式下一点一点来，也可以用）。
        """
        x_next = x_next.to(self.device, self.dtype).view(1, -1)
        out = self.predict_batch(x_next)
        return {"mu": out["mu"].squeeze(0), "var": out["var"].squeeze(0)}

    def _aggregate_particles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        和 OnlineBayesCalibrator 保持 API 对齐，返回一个“代表性的 θ 分布”。
        这里简单地用 current expert 的粒子做加权平均（因为我们是 hard assignment）。
        """
        if len(self.experts) == 0:
            return None, None
        if self.current_expert_idx is None:
            self.current_expert_idx = 0
        e = self.experts[self.current_expert_idx]
        th = e.pf.particles.theta
        w = e.pf.particles.weights()
        d = th.shape[1]
        mean = (w[:, None] * th).sum(dim=0)
        C = ((th - mean) * w[:, None]).T @ (th - mean)
        return mean, C

    # ------------------------------------------------------------------
    # 对外 API：更新（每个 batch 一次）
    # ------------------------------------------------------------------
    def step_batch(self, X_batch: torch.Tensor, Y_batch: torch.Tensor, verbose: bool = True) -> Dict[str, Any]:
        """
        对一个 batch (X_batch, Y_batch) 做一次更新：
          1. 如果是第一个 batch，直接用新建的 expert_0，不做切换逻辑。
          2. 否则：
             - 计算当前 expert 的 log_score_curr。
             - 计算所有其他已有 experts 的 log_score_k。
             - 计算 “新 expert 候选” 的 log_score_new。
             - 如果 best_other - curr_score > bf_threshold，则切换到 best。
               best 可能是已有 expert，也可能是 new expert。
             - 否则，继续使用当前 expert。
          3. 只对选中的 expert 执行 PF + δ 更新。
        """
        X_batch = X_batch.to(self.device, self.dtype)
        Y_batch = Y_batch.to(self.device, self.dtype)
        batch_size, dx = X_batch.shape

        # 如果没有 expert，则创建第一个
        if len(self.experts) == 0:
            e0 = self._spawn_new_expert(dx)
            self.current_expert_idx = 0
            if verbose:
                print(f"[MoE] bootstrap expert id={e0.id}")

        if self.current_expert_idx is None:
            self.current_expert_idx = 0

        # 第一个 batch：直接用 current expert，不做 CRP / 切换
        if self._num_batches_seen == 0:
            chosen_idx = self.current_expert_idx
            e = self.experts[chosen_idx]
            e.count_assigned += 1
            e.last_active_batch = self._num_batches_seen

            # 维护历史
            self._append_hist_batch(e, X_batch, Y_batch, max_len=self.cfg.bocpd.max_run_length)

            # PF 更新
            diag = e.pf.step_batch(
                X_batch,
                Y_batch,
                self.emulator,
                e.delta_state,
                self.model_cfg.rho,
                self.model_cfg.sigma_eps,
                grad_info=False,
            )

            # δ-GP 更新：用 mixture residual
            mu_eta, _ = self.emulator.predict(X_batch, e.pf.particles.theta)
            w = e.pf.particles.weights().view(1, -1)
            eta_mix = (w * mu_eta).sum(dim=1)        # [b]
            resid = Y_batch - self.model_cfg.rho * eta_mix
            e.delta_state.append_batch(X_batch, resid)

            self._num_batches_seen += 1
            self._t_obs += batch_size

            return {
                "chosen_expert_index": chosen_idx,
                "chosen_expert_id": e.id,
                "num_experts": len(self.experts),
                "switched": False,
                "bf": 0.0,
                "log_scores": [0.0],  # 第一个 batch 没有可比性
                "pf_diag": diag,
            }

        # ---------- 之后的 batch：执行 likelihood test + CRP 选择 ----------
        # 1) 计算所有现有 experts 的 predictive log-likelihood
        log_pred_existing: List[float] = []
        for e in self.experts:
            lp = self._predictive_log_mixture_for_expert(e, X_batch, Y_batch)
            log_pred_existing.append(lp)

        # 2) 计算“新 expert 候选”的 log-predictive
        log_pred_new = self._compute_new_expert_log_pred(dx, X_batch, Y_batch)

        # 3) 加入 CRP 先验：log_score = log p(y|expert) + log prior(expert)
        log_scores_existing: List[float] = []
        for e, lp in zip(self.experts, log_pred_existing):
            nk = max(e.count_assigned, 1)  # 至少给一个 count，避免 log(0)
            log_scores_existing.append(lp + math.log(float(nk)))

        log_score_new = log_pred_new + math.log(self.alpha)

        # 当前 expert 的 score
        curr_idx = self.current_expert_idx
        curr_score = log_scores_existing[curr_idx]

        # 其他 candidates（所有非 current 的 expert + new）
        best_alt_score = -float("inf")
        best_alt_is_new = False
        best_alt_idx = None  # 如果不是 new，则是 existing expert index

        # 现有其他 experts
        for k, sc in enumerate(log_scores_existing):
            if k == curr_idx:
                continue
            if sc > best_alt_score:
                best_alt_score = sc
                best_alt_is_new = False
                best_alt_idx = k

        # new expert
        if log_score_new > best_alt_score:
            best_alt_score = log_score_new
            best_alt_is_new = True
            best_alt_idx = None

        bf = best_alt_score - curr_score  # log Bayes factor 近似


        # if verbose:
        #     print(f"[MoE] batch={self._num_batches_seen}, curr={self.experts[curr_idx].id}, "
        #           f"curr_score={curr_score:.3f}, best_alt_score={best_alt_score:.3f}, "
        #           f"bf={bf:.3f}, best_alt_is_new={best_alt_is_new}")

        # 4) likelihood test：如果 best_alt 明显更好 (bf > threshold)，才切换
        switched = False
        if bf > self.bf_threshold:
            if verbose:
                print("\n====================== RETRIEVAL DIAGNOSTICS ======================")
                print(f"[MoE] Batch #{self._num_batches_seen}")
                print(f"Current expert = {self.experts[curr_idx].id}")
                print(f"Current_score = {curr_score:.4f}")
                print(f"Best alternative score = {best_alt_score:.4f}")
                print(f"Bayes Factor (bf) = {bf:.4f}")
                print("-------------------------------------------------------------------")
                print("Experts Predictive Likelihood (sorted):")

                # 对所有专家的 likelihood 排序打印
                temp_list = []
                for k, (e, lp, sc) in enumerate(zip(self.experts, log_pred_existing, log_scores_existing)):
                    temp_list.append((k, e.id, lp, sc))
                temp_list_sorted = sorted(temp_list, key=lambda x: x[3], reverse=True)

                for k, eid, lp, sc in temp_list_sorted:
                    print(f"  Expert idx={k}, id={eid}, pred_ll={lp:.4f}, score={sc:.4f}")

                print("-------------------------------------------------------------------")
            print("Expert data segments (history ranges):")
            for k, e in enumerate(self.experts):
                segs = ", ".join([f"[{a},{b})" for a, b in e.segments])
                print(f"  Expert idx={k}, id={e.id}, segments={segs}")

            print("===================================================================\n")
            switched = True
            if best_alt_is_new:
                # 创建一个新 expert
                e_new = self._spawn_new_expert(dx)
                chosen_idx = self.experts.index(e_new)
                if verbose:
                    print(f"[MoE] create NEW expert id={e_new.id}")
            else:
                chosen_idx = best_alt_idx
                if verbose:
                    print(f"[MoE] switch to EXISTING expert id={self.experts[chosen_idx].id}")
        else:
            chosen_idx = curr_idx

        self.current_expert_idx = chosen_idx
        chosen_e = self.experts[chosen_idx]
        chosen_e.count_assigned += 1
        chosen_e.last_active_batch = self._num_batches_seen

        # 5) 对选中的 expert 做 PF + δ 更新
        self._append_hist_batch(chosen_e, X_batch, Y_batch, max_len=self.cfg.bocpd.max_run_length)

        diag = chosen_e.pf.step_batch(
            X_batch,
            Y_batch,
            self.emulator,
            chosen_e.delta_state,
            self.model_cfg.rho,
            self.model_cfg.sigma_eps,
            grad_info=False,
        )

        mu_eta, _ = self.emulator.predict(X_batch, chosen_e.pf.particles.theta)
        w = chosen_e.pf.particles.weights().view(1, -1)
        eta_mix = (w * mu_eta).sum(dim=1)
        resid = Y_batch - self.model_cfg.rho * eta_mix
        chosen_e.delta_state.append_batch(X_batch, resid)

        # 6) 可选：prune 一下 experts
        self._prune_experts()

        self._num_batches_seen += 1
        self._t_obs += batch_size

        return {
            "chosen_expert_index": chosen_idx,
            "chosen_expert_id": chosen_e.id,
            "num_experts": len(self.experts),
            "switched": switched,
            "bf": float(bf),
            "log_scores_existing": log_scores_existing,
            "log_score_new": float(log_score_new),
            "log_pred_existing": log_pred_existing,
            "log_pred_new": float(log_pred_new),
            "pf_diag": diag,
        }
