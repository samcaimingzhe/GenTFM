"""Gen-TFM: an in-context generative model for mixed-type tabular rows.

Architecture
------------
* ``TwoDimensionalContextEncoder`` performs induced attention down each column,
  followed by CLS-mediated attention across fields in each row. It preserves
  the full ``[B, K, F, field_dim]`` table representation.
* ``GenTFM``           ties the two together: ``compute_loss`` implements
  conditional flow matching + categorical cross-entropy, ``generate``
  integrates the learned velocity field from Gaussian noise to rows.
* ``FieldFlowNet``     reads aligned context columns and performs attention
  within each target row, with shared continuous and categorical output heads.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import batch_feature_mask, encoded_dim, slices


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def infer_feature_mask(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fallback: treat padded columns with zero mass across rows as inactive."""
    return x.abs().sum(dim=1) > eps


def metadata_from_feature_mask(feature_mask: torch.Tensor, max_cont: int, max_cat: int, cat_cardinality: int) -> List[Dict[str, object]]:
    sl = slices(max_cont, max_cat, cat_cardinality)
    metadata: List[Dict[str, object]] = []
    for row in feature_mask.detach().cpu().bool():
        n_cont = int(row[:max_cont].sum().item())
        cat_mask = row[sl["cat_start"] : sl["mask_start"]].reshape(max_cat, cat_cardinality)
        n_cat = int(cat_mask.any(dim=-1).sum().item())
        metadata.append({
            "n_cont": n_cont,
            "n_cat": n_cat,
            "cat_cardinality": cat_cardinality,
            "cat_cardinalities": cat_mask.sum(dim=-1)[:n_cat].tolist(),
        })
    return metadata


def normalize_mixed_batch(x: torch.Tensor, metadata: List[Dict[str, object]], max_cont: int, max_cat: int, cat_cardinality: int, eps: float = 1e-6,
                          reference: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-table standardisation of the continuous columns only.

    Statistics come from ``reference`` (the context in training), or ``x``.
    Categorical one-hots and mask bits are left untouched, inactive dimensions
    are zeroed.  This is what makes the model scale-invariant per table.
    """
    reference = x if reference is None else reference
    x_norm = x.clone()
    mean = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    std = torch.ones_like(mean)
    sl = slices(max_cont, max_cat, cat_cardinality)

    for b, item in enumerate(metadata):
        n_cont = int(item["n_cont"])
        n_cat = int(item["n_cat"])
        if n_cont > 0:
            vals = reference[b : b + 1, :, :n_cont]
            mu = vals.mean(dim=1, keepdim=True)
            sigma = vals.std(dim=1, keepdim=True, unbiased=False) + eps
            x_norm[b : b + 1, :, :n_cont] = (x[b : b + 1, :, :n_cont] - mu) / sigma
            mean[b : b + 1, :, :n_cont] = mu
            std[b : b + 1, :, :n_cont] = sigma

        active = torch.zeros(x.shape[2], device=x.device, dtype=torch.bool)
        active[:n_cont] = True
        active[sl["cat_start"] : sl["cat_start"] + n_cat * cat_cardinality] = True
        active[sl["mask_start"] : sl["mask_start"] + n_cont] = True
        x_norm[b, :, ~active] = 0.0

    return x_norm, {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------


class SchemaAwareFieldTokenizer(nn.Module):
    """Tokenize GenTFM's mixed-type encoded rows into semantic field tokens.

    The tokenizer is intentionally schema-aware rather than copying NanoTabICL's
    circular scalar feature grouping: a categorical one-hot block remains one
    field, and continuous values remain paired with their observed-mask bit.
    """

    def __init__(self, max_cont: int, max_cat: int, cat_cardinality: int, field_dim: int):
        super().__init__()
        self.max_cont = int(max_cont)
        self.max_cat = int(max_cat)
        self.cat_cardinality = int(cat_cardinality)
        self.num_fields = self.max_cont + self.max_cat
        self.field_dim = int(field_dim)
        self.cont_projection = nn.Linear(2, field_dim)
        self.cat_projection = nn.Linear(cat_cardinality, field_dim)
        self.continuous_type_embedding = nn.Parameter(torch.zeros(1, 1, field_dim))
        self.categorical_type_embedding = nn.Parameter(torch.zeros(1, 1, field_dim))

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B,R,D], got {tuple(x.shape)}")
        if feature_mask.dim() != 2 or feature_mask.shape[0] != x.shape[0]:
            raise ValueError(f"Expected feature_mask to have shape [B,D], got {tuple(feature_mask.shape)}")

        B, R, D = x.shape
        sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)
        expected_D = encoded_dim(self.max_cont, self.max_cat, self.cat_cardinality)
        if D != expected_D or feature_mask.shape[1] != expected_D:
            raise ValueError(f"Expected encoded dimension {expected_D}, got x={D}, mask={feature_mask.shape[1]}")

        schema_mask = feature_mask.to(device=x.device, dtype=torch.bool)
        value_mask = schema_mask[:, : self.max_cont].to(dtype=x.dtype)
        observed_mask = schema_mask[:, sl["mask_start"] : sl["mask_start"] + self.max_cont].to(dtype=x.dtype)
        cont_input = torch.stack(
            [x[..., : self.max_cont] * value_mask[:, None, :],
             x[..., sl["mask_start"] : sl["mask_start"] + self.max_cont] * observed_mask[:, None, :]],
            dim=-1,
        )
        cont_tokens = self.cont_projection(cont_input)
        cont_tokens = cont_tokens + self.continuous_type_embedding

        cat_mask = schema_mask[:, sl["cat_start"] : sl["mask_start"]]
        cat_values = x[..., sl["cat_start"] : sl["mask_start"]].reshape(B, R, self.max_cat, self.cat_cardinality)
        cat_values = cat_values * cat_mask[:, None, :].reshape(B, 1, self.max_cat, self.cat_cardinality).to(x.dtype)
        cat_tokens = self.cat_projection(cat_values)
        cat_tokens = cat_tokens + self.categorical_type_embedding

        tokens = torch.cat([cont_tokens, cat_tokens], dim=2)
        cont_active = schema_mask[:, : self.max_cont] | schema_mask[:, sl["mask_start"] : sl["mask_start"] + self.max_cont]
        cat_active = cat_mask.reshape(B, self.max_cat, self.cat_cardinality).any(dim=-1)
        field_mask = torch.cat([cont_active, cat_active], dim=1)

        tokens = tokens * field_mask[:, None, :, None].to(dtype=tokens.dtype)
        return tokens, field_mask


class PreNormAttentionBlock(nn.Module):
    """Small pre-norm self-attention block used by the optional 2D encoder."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class InducedSetAttentionBlock(nn.Module):
    """TabICL-style induced attention over an unordered set."""

    def __init__(self, dim: int, n_heads: int, num_inducing: int = 16, dropout: float = 0.0):
        super().__init__()
        if num_inducing < 1:
            raise ValueError(f"num_inducing must be positive, got {num_inducing}")
        self.num_inducing = int(num_inducing)
        self.inducing = nn.Parameter(torch.empty(self.num_inducing, dim))
        nn.init.trunc_normal_(self.inducing, std=0.02)
        self.inducing_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.read_set = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.inducing_mlp_norm = nn.LayerNorm(dim)
        self.inducing_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.read_inducing = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing.unsqueeze(0).expand(x.shape[0], -1, -1)
        norm_x = self.input_norm(x)
        update, _ = self.read_set(
            self.inducing_norm(inducing), norm_x, norm_x, need_weights=False
        )
        inducing = inducing + update
        inducing = inducing + self.inducing_mlp(self.inducing_mlp_norm(inducing))
        norm_inducing = self.inducing_norm(inducing)
        update, _ = self.read_inducing(norm_x, norm_inducing, norm_inducing, need_weights=False)
        x = x + update
        return x + self.output_mlp(self.output_norm(x))


class FieldPreservingRowInteraction(nn.Module):
    """Use CLS tokens to summarize a row, then write the summary back to its fields."""

    def __init__(self, dim: int, n_heads: int, num_layers: int = 1,
                 num_cls_tokens: int = 4, dropout: float = 0.0):
        super().__init__()
        if num_cls_tokens < 1:
            raise ValueError(f"num_cls_tokens must be positive, got {num_cls_tokens}")
        self.num_cls_tokens = int(num_cls_tokens)
        self.cls_tokens = nn.Parameter(torch.empty(self.num_cls_tokens, dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.blocks = nn.ModuleList(
            [PreNormAttentionBlock(dim, n_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.cls_norm = nn.LayerNorm(dim)
        self.global_projection = nn.Sequential(
            nn.Linear(self.num_cls_tokens * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, fields: torch.Tensor, field_mask: torch.Tensor) -> torch.Tensor:
        B, R, F, d = fields.shape
        flat = fields.reshape(B * R, F, d)
        cls = self.cls_tokens.view(1, self.num_cls_tokens, d).expand(B * R, -1, -1)
        sequence = torch.cat([cls, flat], dim=1)
        field_padding = ~field_mask[:, None, :].expand(B, R, F).reshape(B * R, F)
        cls_padding = torch.zeros(B * R, self.num_cls_tokens, dtype=torch.bool, device=fields.device)
        key_padding_mask = torch.cat([cls_padding, field_padding], dim=1)
        for block in self.blocks:
            sequence = block(sequence, key_padding_mask=key_padding_mask)
        cls_output = self.cls_norm(sequence[:, :self.num_cls_tokens])
        global_update = self.global_projection(cls_output.reshape(B * R, self.num_cls_tokens * d))
        field_output = sequence[:, self.num_cls_tokens:] + global_update[:, None, :]
        active = field_mask[:, None, :, None].expand(B, R, F, 1).to(fields.dtype)
        return field_output.reshape(B, R, F, d) * active


class TwoDimensionalContextEncoder(nn.Module):
    """Schema-aware 2D context encoder that always preserves the field axis."""

    def __init__(self, max_cont: int, max_cat: int, cat_cardinality: int,
                 field_dim: int = 128, n_heads: int = 8, n_col_layers: int = 2,
                 n_row_layers: int = 2, dropout: float = 0.0,
                 num_inducing: int = 16, num_cls_tokens: int = 4):
        super().__init__()
        if field_dim % n_heads != 0:
            raise ValueError(f"field_dim={field_dim} must be divisible by n_heads={n_heads}")
        self.max_cont = int(max_cont)
        self.max_cat = int(max_cat)
        self.cat_cardinality = int(cat_cardinality)
        self.field_dim = int(field_dim)
        self.n_col_layers = int(n_col_layers)
        self.n_row_layers = int(n_row_layers)
        self.tokenizer = SchemaAwareFieldTokenizer(max_cont, max_cat, cat_cardinality, field_dim)
        self.col_blocks = nn.ModuleList([
            InducedSetAttentionBlock(field_dim, n_heads, num_inducing, dropout)
            for _ in range(n_col_layers)
        ])
        self.row_interaction = FieldPreservingRowInteraction(
            field_dim, n_heads, n_row_layers, num_cls_tokens, dropout
        )

    def forward(self, x_ctx: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        tokens, field_mask = self.tokenizer(x_ctx, feature_mask)
        B, R, F, d = tokens.shape
        if R == 0:
            return tokens

        # Column stage: each semantic field is an independent sequence of rows.
        col = tokens.permute(0, 2, 1, 3).reshape(B * F, R, d)
        col_active = field_mask.reshape(B * F, 1, 1).to(dtype=col.dtype)
        for block in self.col_blocks:
            col = block(col) * col_active
        tokens = col.reshape(B, F, R, d).permute(0, 2, 1, 3)

        return self.row_interaction(tokens, field_mask)


# ---------------------------------------------------------------------------
# Flow network
# ---------------------------------------------------------------------------


def mask_category_logits(logits: torch.Tensor, feature_mask: torch.Tensor, max_cont: int,
                         max_cat: int, cat_cardinality: int) -> torch.Tensor:
    """Exclude nonexistent categories; finite fill also handles fully padded fields."""
    valid = feature_mask[:, max_cont : max_cont + max_cat * cat_cardinality]
    valid = valid.reshape(feature_mask.shape[0], 1, max_cat, cat_cardinality)
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)


class FieldFlowBlock(nn.Module):
    """Read matching context columns, then exchange information within each target row."""

    def __init__(self, dim: int, n_heads: int, dropout: float, num_cls_tokens: int = 4):
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.row_interaction = FieldPreservingRowInteraction(
            dim, n_heads, num_layers=1, num_cls_tokens=num_cls_tokens, dropout=dropout
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, field_mask: torch.Tensor) -> torch.Tensor:
        B, M, F, d = x.shape
        K = context.shape[1]
        q = self.cross_norm(x).permute(0, 2, 1, 3).reshape(B * F, M, d)
        kv = self.context_norm(context).permute(0, 2, 1, 3).reshape(B * F, K, d)
        update, _ = self.cross_attn(q, kv, kv, need_weights=False)
        active = field_mask[:, None, :, None].to(x.dtype)
        x = (x + update.reshape(B, F, M, d).permute(0, 2, 1, 3)) * active
        return self.row_interaction(x, field_mask) * active


class FieldFlowNet(nn.Module):
    """Shared field heads; equivariant to aligned permutations within feature types.

    No column IDs, RoPE, or whole-row projections. Tables and generated rows
    remain independent batch elements; only fields within a row interact.
    """

    def __init__(self, max_cont: int, max_cat: int, cat_cardinality: int, field_dim: int,
                 time_dim: int, n_layers: int, n_heads: int, dropout: float,
                 num_cls_tokens: int = 4):
        super().__init__()
        self.max_cont, self.max_cat, self.cat_cardinality = max_cont, max_cat, cat_cardinality
        self.tokenizer = SchemaAwareFieldTokenizer(max_cont, max_cat, cat_cardinality, field_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.t_proj = nn.Linear(time_dim, field_dim)
        self.layers = nn.ModuleList([
            FieldFlowBlock(field_dim, n_heads, dropout, num_cls_tokens=num_cls_tokens)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(field_dim)
        self.cont_velocity = nn.Linear(field_dim, 1)
        self.cat_velocity = nn.Linear(field_dim, cat_cardinality)
        self.obs_velocity = nn.Linear(field_dim, 1)
        self.cat_head = nn.Linear(field_dim, cat_cardinality)
        self.mask_head = nn.Linear(field_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
                feature_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, field_mask = self.tokenizer(x_t, feature_mask)
        h = (h + self.t_proj(self.time_embed(t))[:, None, None, :]) * field_mask[:, None, :, None]
        for layer in self.layers:
            h = layer(h, context, field_mask)
        h = self.norm(h)
        cont, cat = h[:, :, :self.max_cont], h[:, :, self.max_cont:]
        velocity = torch.cat([self.cont_velocity(cont).squeeze(-1),
                              self.cat_velocity(cat).flatten(-2),
                              self.obs_velocity(cont).squeeze(-1)], dim=-1)
        velocity = velocity * feature_mask[:, None, :].to(velocity.dtype)
        logits = mask_category_logits(self.cat_head(cat), feature_mask, self.max_cont,
                                      self.max_cat, self.cat_cardinality)
        return velocity, logits, self.mask_head(cont).squeeze(-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class GenTFM(nn.Module):
    """In-context mixed-type tabular generator trained with conditional flow matching.

    Parameters
    ----------
    max_cont, max_cat, cat_cardinality
        Padded schema limits (see :mod:`gen_tfm.encoding`).
    cat_loss_weight
        Weight of the per-field categorical cross-entropy.
    mask_loss_weight
        Weight of the observed-mask BCE (0 under the no-missingness protocol).
    discrete_flow_weight
        How much the categorical / mask dimensions contribute to the flow loss.
    k_max
        Largest context size seen in training; used by the ``logk`` schedule of
        the generation-time categorical calibration.
    field_dim
        Width of every semantic field token. The field axis is preserved from
        context encoding through the generation heads.
    """

    def __init__(self, max_cont: int = 32, max_cat: int = 8, cat_cardinality: int = 12,
                 n_heads: int = 8, time_dim: int = 256, n_flow_layers: int = 6,
                 dropout: float = 0.0, cat_loss_weight: float = 0.8, mask_loss_weight: float = 0.0,
                 discrete_flow_weight: float = 0.05, k_max: int = 200,
                 field_dim: int = 128, n_col_layers: int = 2, n_row_layers: int = 2,
                 num_inducing: int = 16, num_cls_tokens: int = 4):
        super().__init__()
        self.max_cont = max_cont
        self.max_cat = max_cat
        self.cat_cardinality = cat_cardinality
        self.max_features = encoded_dim(max_cont, max_cat, cat_cardinality)
        self.cat_loss_weight = cat_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.discrete_flow_weight = discrete_flow_weight
        self.k_max = max(1, int(k_max))
        self.field_dim = int(field_dim)
        self.n_col_layers = int(n_col_layers)
        self.n_row_layers = int(n_row_layers)
        self.num_inducing = int(num_inducing)
        self.num_cls_tokens = int(num_cls_tokens)
        self.encoder = TwoDimensionalContextEncoder(
            max_cont=max_cont, max_cat=max_cat, cat_cardinality=cat_cardinality,
            field_dim=field_dim, n_heads=n_heads, n_col_layers=n_col_layers, n_row_layers=n_row_layers,
            dropout=dropout, num_inducing=self.num_inducing, num_cls_tokens=self.num_cls_tokens,
        )
        self.flow_net = FieldFlowNet(
            max_cont, max_cat, cat_cardinality, field_dim, time_dim, n_flow_layers,
            n_heads, dropout, num_cls_tokens=self.num_cls_tokens,
        )

    # ----------------------------------------------------------------- utils
    def config(self) -> Dict[str, object]:
        """Constructor arguments, stored inside checkpoints."""
        return {
            "max_cont": self.max_cont,
            "max_cat": self.max_cat,
            "cat_cardinality": self.cat_cardinality,
            "n_heads": self.flow_net.layers[0].cross_attn.num_heads,
            "time_dim": self.flow_net.time_embed.dim,
            "n_flow_layers": len(self.flow_net.layers),
            "dropout": float(self.flow_net.layers[0].cross_attn.dropout),
            "cat_loss_weight": self.cat_loss_weight,
            "mask_loss_weight": self.mask_loss_weight,
            "discrete_flow_weight": self.discrete_flow_weight,
            "k_max": self.k_max,
            "field_dim": self.field_dim,
            "n_col_layers": self.n_col_layers,
            "n_row_layers": self.n_row_layers,
            "num_inducing": self.num_inducing,
            "num_cls_tokens": self.num_cls_tokens,
        }

    def _type_weights(self, feature_mask: torch.Tensor) -> torch.Tensor:
        weights = feature_mask.float()
        sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)
        weights[:, sl["cat_start"] : sl["mask_start"]] *= self.discrete_flow_weight
        weights[:, sl["mask_start"] : sl["mask_start"] + self.max_cont] *= self.discrete_flow_weight
        return weights

    def _apply_categorical_context_calibration(self, cat_logits: torch.Tensor, X_context: torch.Tensor,
                                               feature_mask: torch.Tensor, alpha: float, tau: float) -> torch.Tensor:
        """Add ``alpha * log(smoothed context category frequencies)`` to the categorical logits."""
        alpha = float(alpha)
        if alpha == 0.0:
            return cat_logits
        tau = max(float(tau), 1e-8)
        sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)
        calibrated = cat_logits.clone()
        for j in range(self.max_cat):
            start = sl["cat_start"] + j * self.cat_cardinality
            active = feature_mask[:, start : start + self.cat_cardinality].any(dim=-1)
            if not bool(active.any()):
                continue
            context_counts = X_context[:, :, start : start + self.cat_cardinality].clamp_min(0.0).sum(dim=1)
            probs = (context_counts + tau) / (context_counts.sum(dim=-1, keepdim=True) + tau * self.cat_cardinality)
            log_probs = probs.clamp_min(1e-8).log().to(dtype=cat_logits.dtype)
            calibrated[active, :, j, :] = calibrated[active, :, j, :] + alpha * log_probs[active, None, :]
        return calibrated

    # -------------------------------------------------------------- training
    def compute_loss(self, X_full: torch.Tensor, metadata: Optional[List[Dict[str, object]]] = None,
                     feature_mask: Optional[torch.Tensor] = None, min_ctx: int = 5, max_ctx: int = 200,
                     min_target: int = 512, normalize: bool = True,
                     context_sizes: Optional[List[int]] = None) -> torch.Tensor:
        """One conditional-flow-matching step on a batch of synthetic tables.

        ``X_full`` is ``[B, N, D]``.  Each table is shuffled and split into K
        context rows and N-K target rows, with K drawn uniformly from
        ``[min_ctx, min(max_ctx, N - min_target)]`` (or from ``context_sizes``).
        """
        B, N, D = X_full.shape
        if D != self.max_features:
            raise ValueError(f"Expected D={self.max_features}, got D={D}")
        if feature_mask is None:
            feature_mask = (batch_feature_mask(metadata, self.max_cont, self.max_cat, self.cat_cardinality, X_full.device)
                            if metadata is not None else infer_feature_mask(X_full))
        feature_mask = feature_mask.to(device=X_full.device, dtype=torch.bool)
        if metadata is None:
            metadata = metadata_from_feature_mask(feature_mask, self.max_cont, self.max_cat, self.cat_cardinality)
        if len(metadata) != B:
            raise ValueError(f"Expected {B} metadata entries, got {len(metadata)}")
        mask_f = feature_mask.float()

        high = min(max_ctx, N - min_target)
        if high < min_ctx:
            raise ValueError(f"Not enough rows for context/target split: N={N}, min_ctx={min_ctx}, min_target={min_target}")
        if context_sizes:
            valid = [int(k) for k in context_sizes if min_ctx <= int(k) <= high]
            if not valid:
                raise ValueError(f"No context_sizes fit split constraints: {list(context_sizes)}, min_ctx={min_ctx}, high={high}")
            num_context = valid[torch.randint(len(valid), (1,), device=X_full.device).item()]
        else:
            num_context = torch.randint(min_ctx, high + 1, (1,), device=X_full.device).item()

        idx = torch.argsort(torch.rand(B, N, device=X_full.device), dim=1)
        X_raw_shuffled = torch.gather(X_full, 1, idx.unsqueeze(-1).expand_as(X_full))
        if normalize:
            X_model_shuffled, _ = normalize_mixed_batch(
                X_raw_shuffled, metadata, self.max_cont, self.max_cat, self.cat_cardinality,
                reference=X_raw_shuffled[:, :num_context],
            )
        else:
            X_model_shuffled = X_raw_shuffled
        X_model_shuffled = X_model_shuffled * mask_f[:, None, :]
        X_ctx = X_model_shuffled[:, :num_context, :]
        X_target = X_model_shuffled[:, num_context:, :]
        X_target_raw = X_raw_shuffled[:, num_context:, :]

        context = self.encoder(X_ctx, feature_mask)

        # Conditional flow matching on a straight path from noise to the target row.
        t = torch.rand(B, device=X_full.device)
        x0 = torch.randn_like(X_target) * mask_f[:, None, :]
        x1 = X_target * mask_f[:, None, :]
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        target_v = x1 - x0

        pred_v, cat_logits, mask_logits = self.flow_net(x_t, t, context, feature_mask)

        weights = self._type_weights(feature_mask)
        sq_err = (pred_v - target_v).pow(2) * weights[:, None, :]
        cfm_loss = sq_err.sum() / (weights.sum(dim=1).clamp_min(1.0) * X_target.shape[1]).sum()

        device = X_full.device
        n_cont = torch.as_tensor([int(item["n_cont"]) for item in metadata], device=device)
        n_cat = torch.as_tensor([int(item["n_cat"]) for item in metadata], device=device)
        cat_active = torch.arange(self.max_cat, device=device)[None, :] < n_cat[:, None]
        cont_active = torch.arange(self.max_cont, device=device)[None, :] < n_cont[:, None]
        sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)

        cat_loss_sum = cfm_loss.new_zeros(())
        cat_count = 0
        for j in range(self.max_cat):
            active_b = cat_active[:, j]
            if not bool(active_b.any()):
                continue
            start = sl["cat_start"] + j * self.cat_cardinality
            target = X_target_raw[:, :, start : start + self.cat_cardinality].argmax(dim=-1)
            loss_j = F.cross_entropy(
                cat_logits[:, :, j, :][active_b].reshape(-1, self.cat_cardinality),
                target[active_b].reshape(-1),
                reduction="sum",
            )
            cat_loss_sum = cat_loss_sum + loss_j
            cat_count += int(active_b.sum()) * X_target.shape[1]
        cat_loss = cat_loss_sum / max(cat_count, 1)

        mask_target = X_target_raw[:, :, sl["mask_start"] : sl["mask_start"] + self.max_cont]
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none")
        mask_loss = (bce * cont_active[:, None, :].float()).sum()
        mask_loss = mask_loss / (cont_active.float().sum(dim=1).clamp_min(1.0) * X_target.shape[1]).sum()

        return cfm_loss + self.cat_loss_weight * cat_loss + self.mask_loss_weight * mask_loss

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, X_context: torch.Tensor, feature_mask: Optional[torch.Tensor] = None, num_gen: int = 500,
                 n_steps: int = 60, method: str = "euler", metadata: Optional[List[Dict[str, object]]] = None,
                 sample_discrete: bool = True, categorical_context_calibration: bool = False,
                 cat_context_alpha: float = 0.0, cat_context_alpha_schedule: str = "constant",
                 cat_context_tau: float = 1.0) -> torch.Tensor:
        """Sample ``num_gen`` new rows conditioned on (normalised) context rows.

        ``X_context`` is ``[B, K, D]`` in the *model* space, i.e. continuous
        columns already standardised with the context statistics (see
        :mod:`gen_tfm.generation` for the user-facing wrapper).

        The ODE is integrated from t=0 (noise) to t=1 (rows).  At t=1 the
        categorical head is evaluated once and each categorical block is
        snapped to a valid one-hot by sampling from the (optionally
        calibrated) softmax.
        """
        if X_context.dim() == 2:
            X_context = X_context.unsqueeze(0)
        B, K, D = X_context.shape
        if D != self.max_features:
            raise ValueError(f"Expected D={self.max_features}, got D={D}")
        device = X_context.device
        if feature_mask is None:
            feature_mask = (batch_feature_mask(metadata, self.max_cont, self.max_cat, self.cat_cardinality, device)
                            if metadata is not None else infer_feature_mask(X_context))
        feature_mask = feature_mask.to(device=device, dtype=torch.bool)
        mask_f = feature_mask.float()

        if K == 0:
            X_context = torch.zeros(B, 1, D, device=device, dtype=X_context.dtype)

        context = self.encoder(X_context, feature_mask)
        x = torch.randn(B, num_gen, D, device=device, dtype=X_context.dtype) * mask_f[:, None, :]
        dt = 1.0 / float(n_steps)

        if method == "heun":
            for i in range(n_steps):
                t0 = torch.full((B,), i * dt, device=device, dtype=X_context.dtype)
                v1, _, _ = self.flow_net(x, t0, context, feature_mask)
                x_euler = (x + v1 * dt) * mask_f[:, None, :]
                t1 = torch.full((B,), (i + 1) * dt, device=device, dtype=X_context.dtype)
                v2, _, _ = self.flow_net(x_euler, t1, context, feature_mask)
                x = (x + 0.5 * (v1 + v2) * dt) * mask_f[:, None, :]
        elif method == "euler":
            for i in range(n_steps):
                t0 = torch.full((B,), i * dt, device=device, dtype=X_context.dtype)
                v, _, _ = self.flow_net(x, t0, context, feature_mask)
                x = (x + v * dt) * mask_f[:, None, :]
        else:
            raise ValueError(f"Unsupported ODE method: {method}")

        if sample_discrete:
            if metadata is None:
                metadata = metadata_from_feature_mask(feature_mask, self.max_cont, self.max_cat, self.cat_cardinality)
            t_final = torch.ones(B, device=device, dtype=X_context.dtype)
            _, cat_logits, mask_logits = self.flow_net(x, t_final, context, feature_mask)
            if categorical_context_calibration:
                schedule = str(cat_context_alpha_schedule or "constant").lower()
                effective_alpha = float(cat_context_alpha)
                if schedule == "logk":
                    effective_alpha *= math.log1p(float(K)) / max(math.log1p(float(self.k_max)), 1e-8)
                elif schedule != "constant":
                    raise ValueError(f"Unsupported cat_context_alpha_schedule: {cat_context_alpha_schedule}")
                cat_logits = self._apply_categorical_context_calibration(cat_logits, X_context, feature_mask,
                                                                         alpha=effective_alpha, tau=cat_context_tau)
            sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)
            n_cont = [int(item["n_cont"]) for item in metadata]
            n_cat = [int(item["n_cat"]) for item in metadata]
            obs_probs = torch.sigmoid(mask_logits)
            obs = (torch.rand_like(obs_probs) < obs_probs).to(x.dtype)

            for b in range(B):
                nc, nk = n_cont[b], n_cat[b]
                if nc > 0:
                    fully_observed = (self.mask_loss_weight == 0 or metadata[b].get("no_missingness")
                                      or metadata[b].get("force_observed_mask"))
                    observed = torch.ones_like(obs[b, :, :nc]) if fully_observed else obs[b, :, :nc]
                    x[b, :, sl["mask_start"] : sl["mask_start"] + nc] = observed
                    x[b, :, :nc] = x[b, :, :nc] * observed
                for j in range(nk):
                    start = sl["cat_start"] + j * self.cat_cardinality
                    probs = torch.softmax(cat_logits[b, :, j, :], dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    x[b, :, start : start + self.cat_cardinality] = F.one_hot(sampled, num_classes=self.cat_cardinality).to(x.dtype)

        return x * mask_f[:, None, :]
