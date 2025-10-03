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