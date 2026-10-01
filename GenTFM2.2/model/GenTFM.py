"""Conditional table flow: column/row encoders and target-to-context attention."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .ColEmb import ColEmbedding, MultiheadAttentionBlock
from .RowInteract import RowInteraction
from data.encoding import Schema, categorical_feature_mask, velocity_feature_mask


class TimeEmbedding(nn.Module):
    """Sinusoidal time features followed by a learned projection."""
    def __init__(self, embed_dim: int):
        super().__init__()
        half_dim = max(1, embed_dim // 2)
        self.register_buffer("frequencies", 2 * math.pi * torch.exp(torch.linspace(0, math.log(1000), half_dim)))
        self.mlp = nn.Sequential(nn.Linear(2 * half_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, t: Tensor) -> Tensor:
        angles = t[:, None].to(self.frequencies.dtype) * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))


class GenTFM(nn.Module):
    """Generate target rows conditioned on a nonempty, clean context table.

    Both branches encode columns across rows, then columns within each row.
    Flattening the fixed-width cell representation preserves feature identity.
    Target row tokens query context row tokens before the output heads.
    """
    def __init__(
        self, max_features: int = 160, embed_dim: int = 128,
        num_col_blocks: int = 2, num_row_blocks: int = 2, nhead: int = 4,
        dim_feedforward: int = 256, num_inds: int = 16,
        dropout: float = 0.0, norm_first: bool = True, recompute: bool = False,
        max_cont: int | None = None, max_cat: int = 0, cat_cardinality: int = 0,
        num_cross_blocks: int = 2, architecture: str = "conditional_v1",
    ):
        super().__init__()
        if architecture != "conditional_v1":
            raise ValueError("only the conditional_v1 architecture is supported")
        if num_cross_blocks < 1:
            raise ValueError("num_cross_blocks must be positive")
        self.max_features, self.embed_dim = max_features, embed_dim
        self.max_cont, self.max_cat, self.cat_cardinality = max_cont, max_cat, cat_cardinality
        if max_cont is not None:
            if max_cont < 1 or max_cat < 1 or cat_cardinality < 2:
                raise ValueError("mixed schema requires max_cont >= 1, max_cat >= 1, cat_cardinality >= 2")
            if max_features != 2 * max_cont + max_cat * cat_cardinality:
                raise ValueError("max_features must match the mixed encoding schema")
        elif max_cat != 0 or cat_cardinality != 0:
            raise ValueError("max_cont must be supplied with a categorical schema")
        self._config = dict(
            max_features=max_features, embed_dim=embed_dim, num_col_blocks=num_col_blocks,
            num_row_blocks=num_row_blocks, nhead=nhead, dim_feedforward=dim_feedforward,
            num_inds=num_inds, dropout=dropout, norm_first=norm_first, recompute=recompute,
            max_cont=max_cont, max_cat=max_cat, cat_cardinality=cat_cardinality,
            num_cross_blocks=num_cross_blocks, architecture=architecture,
        )
        col_kwargs = dict(embed_dim=embed_dim, num_blocks=num_col_blocks, nhead=nhead,
                          dim_feedforward=dim_feedforward, num_inds=num_inds,
                          dropout=dropout, norm_first=norm_first, recompute=recompute)
        row_kwargs = dict(embed_dim=embed_dim, num_blocks=num_row_blocks, nhead=nhead,
                          dim_feedforward=dim_feedforward, max_features=max_features,
                          dropout=dropout, norm_first=norm_first)
        self.context_col_embedding = ColEmbedding(**col_kwargs)
        self.context_row_interaction = RowInteraction(**row_kwargs)
        self.context_projection = nn.Sequential(nn.Linear(max_features * embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.col_embedding = ColEmbedding(**col_kwargs)
        self.row_interaction = RowInteraction(**row_kwargs)
        self.target_projection = nn.Sequential(nn.Linear(max_features * embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.time_embedding = TimeEmbedding(embed_dim)
        self.cross_attention = nn.ModuleList([
            MultiheadAttentionBlock(embed_dim, nhead, dim_feedforward, dropout, norm_first)
            for _ in range(num_cross_blocks)
        ])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.velocity_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, max_features))
        if max_cont is not None:
            self.categorical_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, max_cat * cat_cardinality))

    def config(self) -> dict:
        return self._config.copy()

    def _prepare(self, x: Tensor, feature_mask: Tensor, *, clean: bool = False) -> Tensor:
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.max_features:
            raise ValueError("rows must have shape (B, nonempty_rows, max_features)")
        if not x.is_floating_point():
            raise TypeError("rows must be floating point")
        if feature_mask.shape != (x.shape[0], self.max_features):
            raise ValueError("feature_mask must have shape (B, max_features)")
        if feature_mask.dtype != torch.bool:
            raise TypeError("feature_mask must be boolean")
        if feature_mask.device != x.device:
            raise ValueError("rows and feature_mask must be on the same device")
        x = x.masked_fill(~feature_mask[:, None], 0.0)
        if not torch.isfinite(x).all():
            raise ValueError("active rows must be finite")
        if self.max_cont is not None:
            schema = Schema(self.max_cont, self.max_cat, self.cat_cardinality)
            velocity_feature_mask(feature_mask, schema)
            expected = feature_mask[:, None, schema.mask_start:].to(x.dtype).expand_as(x[..., schema.mask_start:])
            if clean and not torch.equal(x[..., schema.mask_start:], expected):
                raise ValueError("missing observations are unsupported in context")
            if clean:
                classes = categorical_feature_mask(feature_mask, schema)
                cat = x[..., schema.cat_start:schema.mask_start].reshape(x.shape[0], x.shape[1], self.max_cat, self.cat_cardinality)
                active = classes.any(-1)[:, None].expand(cat.shape[:-1])
                selected = cat[active]
                if not (((selected == 0) | (selected == 1)).all() and (selected.sum(-1) == 1).all()):
                    raise ValueError("context categories must be clean one-hot vectors")
            x = x.clone()
            x[..., schema.mask_start:] = expected
        return x

    def encode_context(self, context: Tensor, feature_mask: Tensor) -> Tensor:
        """Time-independent context tokens, reusable across all ODE steps."""
        context = self._prepare(context, feature_mask, clean=True)
        cells = self.context_col_embedding(context, feature_mask)
        cells = self.context_row_interaction(cells, feature_mask)
        rows = self.context_projection(cells.flatten(2))
        return rows.masked_fill(~feature_mask.any(-1)[:, None, None], 0.0)

    def forward(self, x_t: Tensor, t: Tensor, feature_mask: Tensor,
                context: Tensor | None = None, *, context_embeddings: Tensor | None = None,
                return_aux: bool = False):
        """Predict target velocities/logits; raw context or cached tokens are required."""
        x_t = self._prepare(x_t, feature_mask)
        B, N, D = x_t.shape
        if t.shape != (B,) or not t.is_floating_point() or t.device != x_t.device:
            raise ValueError("t must be floating point with shape (B,) on the target device")
        if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("t must be finite and within [0, 1]")
        if (context is None) == (context_embeddings is None):
            raise ValueError("provide exactly one nonempty context or context_embeddings")
        if context is not None:
            if context.shape[0] != B or context.device != x_t.device or context.dtype != x_t.dtype:
                raise ValueError("context and target batch, device and dtype must match")
            context_embeddings = self.encode_context(context, feature_mask)
        if (context_embeddings.ndim != 3 or context_embeddings.shape[0] != B
                or context_embeddings.shape[1] == 0 or context_embeddings.shape[2] != self.embed_dim
                or context_embeddings.device != x_t.device):
            raise ValueError("context_embeddings must have shape (B, K>0, embed_dim) on the target device")
        if not torch.isfinite(context_embeddings).all():
            raise ValueError("context_embeddings must be finite")
        cells = self.col_embedding(x_t, feature_mask)
        cells = cells + self.time_embedding(t).to(cells.dtype)[:, None, None]
        cells = cells.masked_fill(~feature_mask[:, None, :, None], 0.0)
        cells = self.row_interaction(cells, feature_mask)
        rows = self.target_projection(cells.flatten(2))
        for block in self.cross_attention:
            rows = block(rows, context_embeddings, context_embeddings)
        rows = self.output_norm(rows).masked_fill(~feature_mask.any(-1)[:, None, None], 0.0)
        velocity = self.velocity_head(rows).masked_fill(~feature_mask[:, None], 0.0)
        logits = None
        if self.max_cont is not None:
            schema = Schema(self.max_cont, self.max_cat, self.cat_cardinality)
            velocity = velocity.masked_fill(~velocity_feature_mask(feature_mask, schema)[:, None], 0.0)
            if return_aux:
                logits = self.categorical_head(rows).reshape(B, N, self.max_cat, self.cat_cardinality)
                logits = logits.masked_fill(~categorical_feature_mask(feature_mask, schema)[:, None], 0.0)
        return {"velocity": velocity, "categorical_logits": logits} if return_aux else velocity

    @torch.no_grad()
    def generate(self, X_context: Tensor, feature_mask: Tensor, num_gen: int = 500,
                 n_steps: int = 60, method: str = "euler", *,
                 generator: torch.Generator | None = None, categorical_method: str = "sample") -> Tensor:
        """Generate in context-normalized space; use generate_in_context for raw rows."""
        from inference.sampling import sample_table
        if X_context.ndim == 2:
            X_context = X_context.unsqueeze(0)
        return sample_table(self, X_context, feature_mask, num_gen, n_steps, method,
                            generator=generator, categorical_method=categorical_method)
