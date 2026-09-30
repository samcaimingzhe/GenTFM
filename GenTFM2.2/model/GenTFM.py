"""Unconditional cell-wise velocity model for encoded tables."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

if __package__:
    from .ColEmb import ColEmbedding
    from .RowInteract import RowInteraction
else:
    from ColEmb import ColEmbedding
    from RowInteract import RowInteraction


class TimeEmbedding(nn.Module):
    """Sinusoidal time features followed by a learned projection."""

    def __init__(self, embed_dim: int):
        super().__init__()
        half_dim = max(1, embed_dim // 2)
        self.register_buffer(
            "frequencies",
            2 * math.pi * torch.exp(torch.linspace(0, math.log(1000), half_dim)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * half_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        angles = t[:, None].to(self.frequencies.dtype) * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))


class GenTFM(nn.Module):
    """Predict velocities (B, K, D) at time t (B,), without context rows.

    D counts encoded dimensions, including one-hot and observed-mask dimensions.
    feature_mask has shape (B, D), where True means a valid dimension.
    """

    def __init__(
        self, max_features: int = 160, embed_dim: int = 128,
        num_col_blocks: int = 2, num_row_blocks: int = 2, nhead: int = 4,
        dim_feedforward: int = 256, num_inds: int = 16,
        dropout: float = 0.0, norm_first: bool = True, recompute: bool = False,
    ):
        super().__init__()
        self.max_features = max_features
        self._config = dict(
            max_features=max_features, embed_dim=embed_dim,
            num_col_blocks=num_col_blocks, num_row_blocks=num_row_blocks,
            nhead=nhead, dim_feedforward=dim_feedforward, num_inds=num_inds,
            dropout=dropout, norm_first=norm_first, recompute=recompute,
        )
        self.col_embedding = ColEmbedding(
            embed_dim=embed_dim, num_blocks=num_col_blocks, nhead=nhead,
            dim_feedforward=dim_feedforward, num_inds=num_inds,
            dropout=dropout, norm_first=norm_first, recompute=recompute,
        )
        self.time_embedding = TimeEmbedding(embed_dim)
        self.row_interaction = RowInteraction(
            embed_dim=embed_dim, num_blocks=num_row_blocks, nhead=nhead,
            dim_feedforward=dim_feedforward, max_features=max_features,
            dropout=dropout, norm_first=norm_first,
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, 1),
        )

    def config(self) -> dict:
        """Constructor arguments for checkpoint reconstruction."""
        return self._config.copy()

    def forward(self, x_t: Tensor, t: Tensor, feature_mask: Tensor) -> Tensor:
        if x_t.ndim != 3:
            raise ValueError("x_t must have shape (B, K, D)")
        B, K, D = x_t.shape
        if K == 0:
            raise ValueError("x_t must contain at least one row")
        if D > self.max_features:
            raise ValueError("D must not exceed max_features")
        if not x_t.is_floating_point():
            raise TypeError("x_t must be floating point")
        if feature_mask.shape != (B, D):
            raise ValueError("feature_mask must have shape (B, D)")
        if feature_mask.dtype != torch.bool:
            raise TypeError("feature_mask must be boolean")
        if feature_mask.device != x_t.device:
            raise ValueError("x_t and feature_mask must be on the same device")
        if t.shape != (B,):
            raise ValueError("t must have shape (B,)")
        if not t.is_floating_point():
            raise TypeError("t must be floating point")
        if t.device != x_t.device:
            raise ValueError("x_t and t must be on the same device")
        if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("t must be finite and within [0, 1]")

        padding = ~feature_mask[:, None, :]
        x_t = x_t.masked_fill(padding, 0.0)
        h = self.col_embedding(x_t, feature_mask)
        time_h = self.time_embedding(t).to(h.dtype)
        h = (h + time_h[:, None, None, :]).masked_fill(padding[..., None], 0.0)
        h = self.row_interaction(h, feature_mask)
        velocity = self.velocity_head(h).squeeze(-1)
        return velocity.masked_fill(padding, 0.0)
