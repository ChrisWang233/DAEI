"""Residual Denoising Autoencoder for embedding-space Gaussian denoising.

Architecture: f(x) = x + g(x)  where g is the residual network.
g's output_proj is zero-initialized so the DAE starts as identity.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SigmaConditioner(nn.Module):
    """Encode sigma into a hidden-dim conditioning vector via log + Fourier features."""

    def __init__(self, hidden_dim: int, num_freqs: int = 16) -> None:
        super().__init__()
        input_dim = 1 + 2 * num_freqs
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_freqs))
        self.register_buffer("freqs", freqs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (B,) -> (B, hidden_dim)"""
        log_s = sigma.float().log().unsqueeze(-1)
        scaled = log_s * self.freqs
        feat = torch.cat([log_s, scaled.sin(), scaled.cos()], dim=-1)
        return self.net(feat)


def _maybe_sn(layer: nn.Linear, use_spectral_norm: bool) -> nn.Module:
    """Optionally wrap a Linear layer with spectral normalization."""
    return spectral_norm(layer) if use_spectral_norm else layer


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block with optional sigma conditioning."""

    def __init__(
        self, dim: int, hidden_dim: int, dropout: float = 0.1,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = _maybe_sn(nn.Linear(dim, hidden_dim), use_spectral_norm)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = _maybe_sn(nn.Linear(hidden_dim, dim), use_spectral_norm)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(x)
        if cond is not None:
            h = h + cond
        h = self.fc2(self.dropout(self.act(self.fc1(h))))
        return x + h


class ResidualDAE(nn.Module):
    """Residual DAE: f(x) = x + g(x).

    g(x) is a stack of ResidualBlocks with a zero-initialized output projection,
    so at init the DAE is the identity function.

    Args:
        emb_dim: embedding dimensionality (e.g. 768 for GTR-base)
        hidden_dim: width of hidden layers in the residual blocks
        depth: number of ResidualBlocks
        dropout: dropout rate
        use_sigma_cond: if True, condition on noise sigma (for variable-sigma training)
        use_spectral_norm: if True, apply spectral normalization to all Linear layers
            in the residual blocks (bounds the Lipschitz constant and stabilizes
            Jacobian trace estimation in MC-SURE)
    """

    def __init__(
        self,
        emb_dim: int = 768,
        hidden_dim: int = 1024,
        depth: int = 2,
        dropout: float = 0.1,
        use_sigma_cond: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.use_sigma_cond = use_sigma_cond

        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_proj = _maybe_sn(nn.Linear(emb_dim, hidden_dim), use_spectral_norm)

        if use_sigma_cond:
            self.sigma_cond = SigmaConditioner(hidden_dim)
        else:
            self.sigma_cond = None

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout, use_spectral_norm)
            for _ in range(depth)
        ])

        # output_proj is zero-initialized (identity at init) — skip spectral_norm
        # to avoid division by zero (σ(W)=0 for an all-zero matrix).
        # The Lipschitz bound from other layers is sufficient.
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def get_residual(
        self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute only the residual g(x). Used by MC-SURE for trace estimation."""
        cond = None
        if self.use_sigma_cond and sigma is not None:
            B = x.shape[0]
            if sigma.dim() == 0:
                sigma = sigma.expand(B)
            cond = self.sigma_cond(sigma)

        h = self.input_proj(self.input_norm(x))
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)

    def forward(
        self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denoise: f(x) = x + g(x)."""
        return x + self.get_residual(x, sigma)
