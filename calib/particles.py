# =============================================================
# file: calib/particles.py
# =============================================================
from dataclasses import dataclass
import torch


@dataclass
class ParticleSet:
    theta: torch.Tensor  # [N, dθ]
    logw: torch.Tensor   # [N]

    def normalize_(self) -> None:
        m = torch.logsumexp(self.logw, dim=0)
        self.logw = self.logw - m

    def weights(self) -> torch.Tensor:
        self.normalize_()
        return torch.exp(self.logw)

    def ess(self) -> torch.Tensor:
        w = self.weights()
        return 1.0 / (w.pow(2).sum() + 1e-16)

    def gini(self) -> torch.Tensor:
        """
        Compute Gini coefficient of particle weights.
        Returns scalar tensor in [0, 1].
        """
        w = self.weights()              # normalized, sum=1
        N = w.numel()

        if N == 0:
            return torch.tensor(0.0, device=w.device)

        # sort weights (ascending)
        w_sorted, _ = torch.sort(w)

        idx = torch.arange(1, N + 1, device=w.device, dtype=w.dtype)

        gini = 1.0 - 2.0 * torch.sum(
            w_sorted * (N - idx + 0.5)
        ) / N

        return gini.clamp(0.0, 1.0)
