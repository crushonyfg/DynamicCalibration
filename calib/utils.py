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
    """
    Compute per-(batch, particle) log N(y | mu, var) with optional multi-dimensional y.

    Supported shapes:
      - y:  [b]          , mu,var: [b, N]
      - y:  [b, 1]       , mu,var: [b, N]
      - y:  [b, dy]      , mu,var: [b, N, dy]
      - y:  scalar ([])  , mu,var: [b, N]   (treated as [1])

    Semantics for multi-dimensional y:
      We assume dy output dimensions are conditionally independent, i.e.
        p(y | ...) = ∏_{j=1}^{dy} N(y_j | mu_j, var_j)
      So the returned logpdf is the sum over dimensions, with shape [b, N].
    """
    # Ensure y has at least shape [b, dy]
    if y.dim() == 0:
        y = y[None]  # [1]
    if y.dim() == 1:
        y = y[:, None]  # [b,1]

    # Handle scalar-output case: mu,var in [b,N]
    if mu.dim() == 2:
        # Treat as dy=1
        b = y.shape[0]
        if y.shape[1] != 1:
            # If user passed [b,dy] with dy>1 but mu is [b,N], this is ambiguous.
            # For now, require dy==1 in this case.
            raise ValueError(
                f"normal_logpdf: got y of shape {tuple(y.shape)} but mu/var of shape {tuple(mu.shape)}; "
                "when mu, var are 2D ([b,N]), y must have shape [b] or [b,1]."
            )
        # Broadcast y to [b,N]
        yb = y[:, 0:1].expand_as(mu)
        log_det = -0.5 * torch.log(2.0 * torch.pi * var)
        quad = -0.5 * (yb - mu).pow(2) / var
        return log_det + quad  # [b,N]

    # Multi-dimensional output case: mu,var in [b,N,dy]
    if mu.dim() != 3 or var.dim() != 3:
        raise ValueError(
            f"normal_logpdf: unsupported shapes mu={tuple(mu.shape)}, var={tuple(var.shape)}"
        )

    b, N, dy = mu.shape
    if y.shape[0] != b:
        raise ValueError(
            f"normal_logpdf: batch size mismatch between y {tuple(y.shape)} and mu {tuple(mu.shape)}"
        )
    # y: [b,dy] or [b,1] -> broadcast to [b,1,dy] then [b,N,dy]
    if y.shape[1] == 1 and dy > 1:
        # Allow y[:,0] to broadcast over all output dims (interpret as same value per dim)
        y_exp = y.expand(b, dy)
    elif y.shape[1] == dy:
        y_exp = y
    else:
        raise ValueError(
            f"normal_logpdf: y has shape {tuple(y.shape)} but expected second dim 1 or {dy}"
        )

    yb = y_exp[:, None, :].expand_as(mu)  # [b,N,dy]
    log_det = -0.5 * torch.log(2.0 * torch.pi * var)      # [b,N,dy]
    quad = -0.5 * (yb - mu).pow(2) / var                  # [b,N,dy]
    lp = log_det + quad                                   # [b,N,dy]
    # Sum over output dimensions to get per-(b,N) log-likelihood
    return lp.sum(dim=-1)  # [b,N]