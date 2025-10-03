# =============================================================
# file: calib/utils.py
# =============================================================
from typing import Dict
import torch


def to_device_dtype(x: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    return x.to(device=device, dtype=dtype)


def summarize_particles(theta: torch.Tensor, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    m = (weights[:, None] * theta).sum(0)
    C = ((theta - m) * weights[:, None]).T @ (theta - m)
    return {"mean": m, "cov": C}

def normal_logpdf(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    # y: [b] or [] ; mu,var: [b,N]
    if y.dim() == 0:
        y = y[None]
    b = y.shape[0]
    yb = y[:, None].expand_as(mu)
    log_det = -0.5 * torch.log(2.0 * torch.pi * var)
    quad = -0.5 * (yb - mu).pow(2) / var
    return log_det + quad  # [b,N]