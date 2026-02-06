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
        
    def unique_ratio(self, eps: float = 1e-6) -> torch.Tensor:
        """
        Fraction of effectively unique particles.
        For 1D theta: clusters by tolerance eps.
        For multi-D theta: uses L2 distance threshold eps.
        """
        theta = self.theta
        N = theta.shape[0]

        if N <= 1:
            return torch.tensor(1.0, device=theta.device, dtype=theta.dtype)

        if theta.shape[1] == 1:
            # 1D case: sort and count jumps
            x = theta.squeeze()
            x_sorted, _ = torch.sort(x)
            diffs = x_sorted[1:] - x_sorted[:-1]
            num_unique = 1 + (diffs.abs() > eps).sum()
        else:
            # multi-D: greedy uniqueness check (O(N^2), ok for diagnostics)
            num_unique = 0
            used = torch.zeros(N, dtype=torch.bool, device=theta.device)
            for i in range(N):
                if used[i]:
                    continue
                num_unique += 1
                d = torch.norm(theta - theta[i], dim=1)
                used = used | (d < eps)

        return num_unique.to(theta.dtype) / N
    def entropy_1d_histogram(self, bins: int = 30, eps: float = 1e-12) -> torch.Tensor:
        """
        Entropy of histogram of the first dimension of theta.
        Valid for 1D theta; for multi-D, uses theta[:, 0] as a fixed projection.
        """
        x = self.theta[:, 0]

        if x.numel() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # fixed range for stability
        xmin = x.min()
        xmax = x.max()

        if (xmax - xmin).abs() < eps:
            # all particles collapsed
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        hist = torch.histc(x, bins=bins, min=xmin.item(), max=xmax.item())
        p = hist / hist.sum()

        entropy = -(p * (p + eps).log()).sum()
        return entropy
