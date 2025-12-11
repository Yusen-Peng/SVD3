import torch
import torch.nn as nn
from typing import Optional

def safe_svd(M: torch.Tensor):
    """
        Apply SVD on M with fallbacks.
    """
    try:
        U, S, VT = torch.linalg.svd(M.detach(), full_matrices=False)
        return U.to(M.device, dtype=M.dtype), S.to(M.device, dtype=M.dtype), VT.to(M.device, dtype=M.dtype)
    except Exception:
        raise RuntimeError("SVD failed. Bad times!")

def sanitize(t: torch.Tensor, replace: float = 0.0) -> torch.Tensor:
    t = t.clone()
    mask = ~torch.isfinite(t)
    if mask.any():
        t[mask] = replace
    return t

def trunc_rank(m: int, n: int, r: float) -> int:
    # r = (ratio * m * n) / (m + n)
    rr = int((m * n * r) / (m + n))
    return max(1, min(rr, min(m, n)))


class TwoFactorLinear(nn.Module):
    """
    A Linear layer represented as two factors: W = U @ V
    where W has shape (out_features, in_features),
    U has shape (out_features, r),
    V has shape (r, in_features).
    """
    def __init__(self, in_features: int, out_features: int,
                 W_u: torch.Tensor, W_v: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        r = W_v.shape[0]
        assert W_v.shape == (r, in_features)
        assert W_u.shape == (out_features, r)
        self.v = nn.Linear(in_features, r, bias=False)
        self.u = nn.Linear(r, out_features, bias=(bias is not None))
        with torch.no_grad():
            self.v.weight.copy_(W_v)
            self.u.weight.copy_(W_u)
            if bias is not None:
                self.u.bias.copy_(bias)

    def forward(self, x):
        """
        Forward pass through the two-factor linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.u(self.v(x))
