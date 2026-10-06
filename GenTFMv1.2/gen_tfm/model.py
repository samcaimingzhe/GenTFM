"""Gen-TFM: an in-context generative model for mixed-type tabular rows.

Architecture
------------
* ``ContextEncoder``   a permutation-invariant Transformer over the K context
  rows (no positional encoding).  Output: ``[B, K, hidden]``.
* ``MixedFlowNet``     a stack of blocks in which the noisy target rows
  cross-attend to the encoded context.  Three heads: a flow-matching velocity
  head for continuous values, a per-field categorical head, and an
  observed-mask head (unused under the no-missingness protocol).
* ``GenTFM``           ties the two together: ``compute_loss`` implements
  conditional flow matching + categorical cross-entropy, ``generate``
  integrates the learned velocity field from Gaussian noise to rows.

Legacy onehot/context_only parameter names are preserved. Binary codecs need new weights.
Query-memory changes only the existing attention memory, not the module stack.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (Schema, encoded_dim, slices, batch_feature_mask, mixed_feature_mask,
                       cat_cardinalities, label_info, decode_category, encode_category, validate_metadata)


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
    schema = Schema(max_cont,max_cat,cat_cardinality)
    metadata = []
    for row in feature_mask.detach().cpu().bool():
        n_cont = int(row[:max_cont].sum())
        cards = []
        inactive_seen = False
        for j in range(max_cat):
            block = row[schema.category_slice(j)]
            if not bool(block.any()):
                inactive_seen = True
                continue
            if inactive_seen:
                raise ValueError("Noncontiguous category slots require explicit metadata")
            card = int(block.sum())
            if not bool(block[:card].all()) or bool(block[card:].any()):
                raise ValueError("Cannot infer cardinality from partial observations; provide metadata")
            cards.append(card)
        metadata.append(dict(n_cont=n_cont,n_cat=len(cards),cat_cardinality=cat_cardinality,
                             cat_cardinalities=cards))
    return metadata


def normalize_mixed_batch(x: torch.Tensor, metadata: List[Dict[str, object]], max_cont: int, max_cat: int, cat_cardinality: int, eps: float = 1e-6, cat_encoding: str = "onehot", binary_bit_order: str = "msb_first") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-table standardisation of the continuous columns only.

    Categorical one-hots and mask bits are left untouched, inactive dimensions
    are zeroed.  This is what makes the model scale-invariant per table.
    """
    x_norm = x.clone()
    mean = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    std = torch.ones_like(mean)
    sl = slices(max_cont, max_cat, cat_cardinality, cat_encoding, binary_bit_order)

    for b, item in enumerate(metadata):
        n_cont = int(item["n_cont"])
        n_cat = int(item["n_cat"])
        if n_cont > 0:
            vals = x[b : b + 1, :, :n_cont]
            mu = vals.mean(dim=1, keepdim=True)
            sigma = vals.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
            x_norm[b : b + 1, :, :n_cont] = (vals - mu) / sigma
            mean[b : b + 1, :, :n_cont] = mu
            std[b : b + 1, :, :n_cont] = sigma

        active = torch.as_tensor(mixed_feature_mask(item, max_cont, max_cat, cat_cardinality,
                                                  cat_encoding, binary_bit_order), device=x.device)
        x_norm[b, :, ~active] = 0.0

    return x_norm, {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Context encoder
# ---------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """Transformer encoder over context rows; returns the full row sequence ``[B, K, hidden]``.

    There is no positional embedding: rows of a table have no order, so the
    representation must be permutation invariant.
    """

    def __init__(self, max_features: int, hidden_dim: int, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.max_features = max_features
        self.proj_in = nn.Sequential(
            nn.Linear(max_features * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_ctx: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        B, K, D = x_ctx.shape
        mask = feature_mask.to(device=x_ctx.device, dtype=x_ctx.dtype)
        x_ctx = x_ctx * mask[:, None, :]
        mask_rows = mask[:, None, :].expand(B, K, D)
        h = self.proj_in(torch.cat([x_ctx, mask_rows], dim=-1))  # the mask tells the encoder which dims are alive
        h = self.transformer(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Flow network
# ---------------------------------------------------------------------------


class TabularFlowBlock(nn.Module):
    """Cross-attention from the (noisy) target rows to the context, followed by an MLP."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, query_conditioning: str = "context_only") -> torch.Tensor:
        query = self.cross_norm(x)
        if query_conditioning == "context_only":
            memory = context
            attn_mask = None
        elif query_conditioning == "context_plus_noisy_query":
            memory = torch.cat([context, query], dim=1)
            n_query = query.shape[1]
            n_context = context.shape[1]
            # Each row may read the context and its own noisy state, but no other query row.
            attn_mask = torch.cat([
                torch.zeros((n_query, n_context), dtype=torch.bool, device=query.device),
                ~torch.eye(n_query, dtype=torch.bool, device=query.device),
            ], dim=1)
        else:
            raise ValueError(f"Unsupported query_conditioning: {query_conditioning}")
        cross_out, _ = self.cross_attn(query=query, key=memory, value=memory,
                                      attn_mask=attn_mask, need_weights=False)
        x = x + cross_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MixedFlowNet(nn.Module):
    """Velocity / categorical / mask heads on top of ``n_layers`` cross-attention blocks."""

    def __init__(self, max_features: int, max_cont: int, max_cat: int, cat_cardinality: int, hidden_dim: int,
                 time_dim: int = 256, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.0,
                 query_conditioning: str = "context_only"):
        super().__init__()
        self.max_features = max_features
        self.max_cont = max_cont
        self.max_cat = max_cat
        self.cat_cardinality = cat_cardinality
        self.query_conditioning = query_conditioning
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.x_proj = nn.Linear(max_features * 2, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)
        # Supervised target embedding.  A separate continuous projection and
        # categorical lookup let one model handle regression and classification.
        self.y_cat_embed = nn.Embedding(cat_cardinality, hidden_dim)
        self.y_cont_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([TabularFlowBlock(hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, max_features)
        self.cat_head = nn.Linear(hidden_dim, max_cat * cat_cardinality)
        self.mask_head = nn.Linear(hidden_dim, max_cont)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor, feature_mask: torch.Tensor,
                y_cond: Optional[torch.Tensor] = None, y_is_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x_t.shape
        mask = feature_mask.to(device=x_t.device, dtype=x_t.dtype)
        x_t = x_t * mask[:, None, :]
        mask_rows = mask[:, None, :].expand(B, N, D)

        t_emb = self.t_proj(self.time_embed(t)).unsqueeze(1)
        h = self.x_proj(torch.cat([x_t, mask_rows], dim=-1)) + t_emb
        if y_cond is not None:
            if y_cond.dim() == 1:
                y_cond = y_cond[:, None].expand(B, N)
            if y_is_cat is None:
                y_is_cat = torch.ones(B, dtype=torch.bool, device=x_t.device)
            y_is_cat = y_is_cat.to(device=x_t.device, dtype=torch.bool).view(B, 1, 1)
            y_cat = self.y_cat_embed(y_cond.long().clamp(0, self.cat_cardinality - 1))
            y_cont = self.y_cont_embed(y_cond.to(x_t.dtype).unsqueeze(-1))
            h = h + torch.where(y_is_cat, y_cat, y_cont)
        for layer in self.layers:
            h = layer(h, context, self.query_conditioning)
        h = self.norm(h)

        velocity = self.velocity_head(h) * mask[:, None, :]
        cat_logits = self.cat_head(h).view(B, N, self.max_cat, self.cat_cardinality)
        mask_logits = self.mask_head(h)
        return velocity, cat_logits, mask_logits


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
    """

    def __init__(self, max_cont: int = 32, max_cat: int = 8, cat_cardinality: int = 12, hidden_dim: int = 384,
                 n_heads: int = 8, n_enc_layers: int = 4, time_dim: int = 256, n_flow_layers: int = 6,
                 dropout: float = 0.0, cat_loss_weight: float = 0.8, mask_loss_weight: float = 0.0,
                 discrete_flow_weight: float = 0.05, k_max: int = 200, cat_encoding: str = "binary",
                 binary_bit_order: str = "msb_first", schema_version: Optional[str] = None,
                 query_conditioning: str = "context_plus_noisy_query"):
        super().__init__()
        self.max_cont = max_cont
        self.max_cat = max_cat
        self.cat_cardinality = cat_cardinality
        self.schema = Schema(max_cont, max_cat, cat_cardinality, cat_encoding, binary_bit_order, schema_version)
        self.cat_encoding = cat_encoding
        self.binary_bit_order = binary_bit_order
        self.query_conditioning = query_conditioning
        if query_conditioning not in {"context_only", "context_plus_noisy_query"}:
            raise ValueError(f"Unsupported query_conditioning: {query_conditioning}")
        self.max_features = self.schema.encoded_dim
        self.hidden_dim = hidden_dim
        self.cat_loss_weight = cat_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.discrete_flow_weight = discrete_flow_weight
        self.k_max = max(1, int(k_max))
        self.encoder = ContextEncoder(self.max_features, hidden_dim, n_heads=n_heads, n_layers=n_enc_layers, dropout=dropout)
        self.flow_net = MixedFlowNet(self.max_features, max_cont, max_cat, cat_cardinality, hidden_dim,
                                     time_dim=time_dim, n_layers=n_flow_layers, n_heads=n_heads, dropout=dropout,
                                     query_conditioning=query_conditioning)

    # ----------------------------------------------------------------- utils
    def config(self) -> Dict[str, object]:
        """Constructor arguments, stored inside checkpoints."""
        return {
            "max_cont": self.max_cont,
            "max_cat": self.max_cat,
            "cat_cardinality": self.cat_cardinality,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.flow_net.layers[0].cross_attn.num_heads,
            "n_enc_layers": len(self.encoder.transformer.layers),
            "time_dim": self.flow_net.time_embed.dim,
            "n_flow_layers": len(self.flow_net.layers),
            "dropout": float(self.flow_net.layers[0].cross_attn.dropout),
            "cat_loss_weight": self.cat_loss_weight,
            "mask_loss_weight": self.mask_loss_weight,
            "discrete_flow_weight": self.discrete_flow_weight,
            "k_max": self.k_max,
            **self.schema.metadata_fields(),
            "query_conditioning": self.query_conditioning,
        }

    def _type_weights(self, feature_mask: torch.Tensor) -> torch.Tensor:
        weights = feature_mask.float()
        sl = slices(*self.schema.as_tuple(), **self.schema.codec_kwargs())
        weights[:, sl["cat_start"] : sl["mask_start"]] *= self.discrete_flow_weight
        weights[:, sl["mask_start"] : sl["mask_start"] + self.max_cont] *= self.discrete_flow_weight
        return weights

    def _label_info(self, metadata: Dict[str, object]) -> Tuple[str, int]:
        """Return supervised label type and its index in the mixed encoding."""
        return label_info(metadata)

    def _prepare_metadata(self, x, metadata, feature_mask):
        if metadata is None:
            if self.cat_encoding == "binary":
                raise ValueError("Binary rows require explicit metadata/cardinalities; zero bits are valid category 0")
            if feature_mask is None:
                feature_mask = infer_feature_mask(x)
            metadata = metadata_from_feature_mask(feature_mask, *self.schema.as_tuple())
            for m in metadata:
                m.update(label_type="categorical", label_cat_index=int(m["n_cat"])-1)
        if len(metadata) != x.shape[0]:
            raise ValueError("One metadata entry required per table")
        for m in metadata:
            validate_metadata(m, self.schema)
            self._label_info(m)
        expected = batch_feature_mask(metadata, *self.schema.as_tuple(), device=x.device, **self.schema.codec_kwargs())
        if feature_mask is not None and (feature_mask.shape != expected.shape or not torch.equal(feature_mask.to(expected), expected)):
            raise ValueError("feature_mask conflicts with schema/metadata")
        return metadata, expected

    def _extract_y(self, rows: torch.Tensor, metadata: List[Dict[str, object]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract per-row labels and build the feature mask used by the flow."""
        B, N, _ = rows.shape
        y = rows.new_zeros((B, N))
        y_is_cat = torch.zeros(B, dtype=torch.bool, device=rows.device)
        feature_keep = torch.ones(B, self.max_features, dtype=torch.bool, device=rows.device)
        sl = slices(*self.schema.as_tuple(), **self.schema.codec_kwargs())
        for b, meta in enumerate(metadata):
            kind, index = self._label_info(meta)
            if kind == "categorical":
                if not 0 <= index < self.max_cat:
                    raise ValueError(f"label_cat_index={index} is outside max_cat={self.max_cat}")
                start = self.schema.category_slice(index).start
                card = int(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)[index])
                y[b] = decode_category(rows[b, :, self.schema.category_slice(index)], self.schema, card).to(rows.dtype)
                feature_keep[b, start:start + self.schema.cat_width] = False
                y_is_cat[b] = True
            else:
                if not 0 <= index < self.max_cont:
                    raise ValueError(f"label_cont_index={index} is outside max_cont={self.max_cont}")
                y[b] = rows[b, :, index]
                feature_keep[b, index] = False
                feature_keep[b, sl["mask_start"] + index] = False
        return y, y_is_cat, feature_keep

    def _apply_categorical_context_calibration(self, cat_logits, X_context, metadata, alpha, tau):
        calibrated = cat_logits.clone()
        tau = max(float(tau), 1e-8)
        for b, meta in enumerate(metadata):
            for j, card in enumerate(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)):
                ids = decode_category(X_context[b, :, self.schema.category_slice(j)], self.schema, card)
                counts = torch.bincount(ids, minlength=int(card)).to(cat_logits.dtype)
                probs = (counts + tau) / (counts.sum() + tau * int(card))
                calibrated[b, :, j, :int(card)] += float(alpha) * probs.log()[None, :]
        return calibrated

    # -------------------------------------------------------------- training
    def compute_loss(self, X_full: torch.Tensor, metadata: Optional[List[Dict[str, object]]] = None,
                     feature_mask: Optional[torch.Tensor] = None, min_ctx: int = 5, max_ctx: int = 200,
                     min_target: int = 512, normalize: bool = True,
                     context_sizes: Optional[List[int]] = None, num_query: Optional[int] = None) -> torch.Tensor:
        """One conditional-flow-matching step on a batch of synthetic tables.

        ``X_full`` is ``[B, N, D]``.  Each table is shuffled and split into K
        context rows and exactly num_query target rows when specified;
        otherwise Q=N-K. Remaining rows are excluded from the loss. Numerical
        normalisation preserves the v1.1 whole-table policy.
        """
        B, N, D = X_full.shape
        if D != self.max_features:
            raise ValueError(f"Expected D={self.max_features}, got D={D}")
        metadata, feature_mask = self._prepare_metadata(X_full, metadata, feature_mask)
        mask_f = feature_mask.float()

        if normalize:
            X_model, _ = normalize_mixed_batch(X_full, metadata, *self.schema.as_tuple(), **self.schema.codec_kwargs())
        else:
            X_model = X_full * mask_f[:, None, :]

        if num_query is not None and int(num_query) <= 0:
            raise ValueError("num_query must be positive or None")
        required_query = int(num_query) if num_query is not None else min_target
        high = min(max_ctx, N - required_query)
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
        X_model_shuffled = torch.gather(X_model, 1, idx.unsqueeze(-1).expand_as(X_model))
        X_raw_shuffled = torch.gather(X_full, 1, idx.unsqueeze(-1).expand_as(X_full))
        X_ctx = X_model_shuffled[:, :num_context, :]
        end = num_context + int(num_query) if num_query is not None else N
        X_target = X_model_shuffled[:, num_context:end, :]
        X_target_raw = X_raw_shuffled[:, num_context:end, :]

        context = self.encoder(X_ctx, feature_mask)
        y_cond, y_is_cat, feature_keep = self._extract_y(X_target, metadata)
        flow_mask = feature_mask & feature_keep

        # Conditional flow matching on a straight path from noise to the target row.
        t = torch.rand(B, device=X_full.device)
        flow_mask_f = flow_mask.float()
        x0 = torch.randn_like(X_target) * flow_mask_f[:, None, :]
        x1 = X_target * flow_mask_f[:, None, :]
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * x0 + t_exp * x1
        target_v = x1 - x0

        pred_v, cat_logits, mask_logits = self.flow_net(x_t, t, context, flow_mask, y_cond, y_is_cat)

        weights = self._type_weights(flow_mask)
        sq_err = (pred_v - target_v).pow(2) * weights[:, None, :]
        cfm_loss = sq_err.sum() / (weights.sum(dim=1).clamp_min(1.0) * X_target.shape[1]).sum()

        device = X_full.device
        n_cont = torch.as_tensor([int(item["n_cont"]) for item in metadata], device=device)
        n_cat = torch.as_tensor([int(item["n_cat"]) for item in metadata], device=device)
        cat_active = torch.arange(self.max_cat, device=device)[None, :] < n_cat[:, None]
        cont_active = torch.arange(self.max_cont, device=device)[None, :] < n_cont[:, None]
        for b, meta in enumerate(metadata):
            kind, index = self._label_info(meta)
            if kind == "continuous":
                cont_active[b, index] = False
        sl = slices(*self.schema.as_tuple(), **self.schema.codec_kwargs())

        cat_losses = []
        for j in range(self.max_cat):
            active_b = cat_active[:, j]
            active_b = active_b & torch.as_tensor([not (self._label_info(m) == ("categorical", j)) for m in metadata], device=device)
            if not bool(active_b.any()):
                continue
            block = self.schema.category_slice(j)
            losses_b = []
            for b in torch.where(active_b)[0].tolist():
                card = int(cat_cardinalities(metadata[b], self.max_cat, self.cat_cardinality)[j])
                target = decode_category(X_target_raw[b, :, block], self.schema, card)
                losses_b.append(F.cross_entropy(cat_logits[b, :, j, :card], target.long()))
            loss_j = torch.stack(losses_b).mean()
            cat_losses.append(loss_j)
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else cfm_loss.new_zeros(())

        mask_target = X_target_raw[:, :, sl["mask_start"] : sl["mask_start"] + self.max_cont]
        bce = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none")
        mask_loss = (bce * cont_active[:, None, :].float()).sum()
        mask_loss = mask_loss / (cont_active.float().sum(dim=1).clamp_min(1.0) * X_target.shape[1]).sum()

        self.last_loss_details = {"cfm_loss": float(cfm_loss.detach()), "cat_loss": float(cat_loss.detach()),
                                  "mask_loss": float(mask_loss.detach()), "K": int(num_context),
                                  "Q": int(X_target.shape[1]), "loss_rows": int(B * X_target.shape[1])}
        return cfm_loss + self.cat_loss_weight * cat_loss + self.mask_loss_weight * mask_loss

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, X_context: torch.Tensor, feature_mask: Optional[torch.Tensor] = None, num_gen: int = 500,
                 n_steps: int = 60, method: str = "euler", metadata: Optional[List[Dict[str, object]]] = None,
                 sample_discrete: bool = True, categorical_context_calibration: bool = False,
                 cat_context_alpha: float = 0.0, cat_context_alpha_schedule: str = "constant",
                 cat_context_tau: float = 1.0, target_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample ``num_gen`` new rows conditioned on (normalised) context rows.

        ``X_context`` is ``[B, K, D]`` in the *model* space, i.e. continuous
        columns already standardised with the context statistics (see
        :mod:`gen_tfm.generation` for the user-facing wrapper).

        The ODE is integrated from t=0 (noise) to t=1 (rows).  At t=1 the
        categorical head is evaluated once and each categorical block is
        encoded as a valid category code by sampling from the (optionally
        calibrated) softmax.
        """
        if X_context.dim() == 2:
            X_context = X_context.unsqueeze(0)
        B, K, D = X_context.shape
        if D != self.max_features:
            raise ValueError(f"Expected D={self.max_features}, got D={D}")
        device = X_context.device
        if num_gen <= 0 or n_steps <= 0:
            raise ValueError("num_gen and n_steps must be positive")
        metadata, feature_mask = self._prepare_metadata(X_context, metadata, feature_mask)
        mask_f = feature_mask.float()
        if K == 0 and target_y is None:
            raise ValueError("Empty context requires explicit target_y")
        y_is_cat = torch.zeros(B, dtype=torch.bool, device=device)
        flow_mask = feature_mask.clone()
        sl = slices(*self.schema.as_tuple(), **self.schema.codec_kwargs())
        if target_y is None:
            ys = []
            for b, meta in enumerate(metadata):
                kind, index = self._label_info(meta)
                y_is_cat[b] = kind == "categorical"
                if kind == "categorical":
                    start = self.schema.category_slice(index).start
                    card = int(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)[index])
                    ids = decode_category(X_context[b, :, self.schema.category_slice(index)], self.schema, card)
                    probs = torch.bincount(ids, minlength=card).to(X_context.dtype)
                    probs = probs / probs.sum()
                    ys.append(torch.multinomial(probs.expand(num_gen, -1), 1).squeeze(-1).to(X_context.dtype))
                    flow_mask[b, start:start + self.schema.cat_width] = False
                else:
                    vals = X_context[b, :, index]
                    pick = torch.randint(vals.shape[0], (num_gen,), device=device)
                    ys.append(vals[pick])
                    flow_mask[b, index] = False
                    flow_mask[b, sl["mask_start"] + index] = False
            y_cond = torch.stack(ys, dim=0)
        else:
            y_cond = torch.as_tensor(target_y, device=device, dtype=X_context.dtype)
            if y_cond.dim() == 0:
                y_cond = y_cond.expand(B, num_gen)
            elif y_cond.dim() == 1:
                if y_cond.numel() == B:
                    y_cond = y_cond[:, None].expand(B, num_gen)
                elif B == 1 and y_cond.numel() == num_gen:
                    y_cond = y_cond[None, :]
                else:
                    raise ValueError("target_y must be scalar, [B], or [B, num_gen]")
            if y_cond.shape != (B, num_gen):
                raise ValueError(f"target_y must have shape {(B, num_gen)}, got {tuple(y_cond.shape)}")
            for b, meta in enumerate(metadata):
                kind, index = self._label_info(meta)
                y_is_cat[b] = kind == "categorical"
                if kind == "categorical":
                    start = self.schema.category_slice(index).start
                    flow_mask[b, start:start + self.schema.cat_width] = False
                else:
                    flow_mask[b, index] = False
                    flow_mask[b, sl["mask_start"] + index] = False
        if not bool(torch.isfinite(y_cond).all()):
            raise ValueError("target_y must be finite")
        for b, meta in enumerate(metadata):
            kind, index = self._label_info(meta)
            if kind == "categorical":
                card = int(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)[index])
                if bool(((y_cond[b] != y_cond[b].long()) | (y_cond[b] < 0) | (y_cond[b] >= card)).any()):
                    raise ValueError("target_y outside valid category IDs")
        flow_mask_f = flow_mask.float()

        if K == 0:
            X_context = torch.zeros(B, 1, D, device=device, dtype=X_context.dtype)

        context = self.encoder(X_context, feature_mask)
        x = torch.randn(B, num_gen, D, device=device, dtype=X_context.dtype) * flow_mask_f[:, None, :]
        dt = 1.0 / float(n_steps)

        if method == "heun":
            for i in range(n_steps):
                t0 = torch.full((B,), i * dt, device=device, dtype=X_context.dtype)
                v1, _, _ = self.flow_net(x, t0, context, flow_mask, y_cond, y_is_cat)
                x_euler = (x + v1 * dt) * flow_mask_f[:, None, :]
                t1 = torch.full((B,), (i + 1) * dt, device=device, dtype=X_context.dtype)
                v2, _, _ = self.flow_net(x_euler, t1, context, flow_mask, y_cond, y_is_cat)
                x = (x + 0.5 * (v1 + v2) * dt) * flow_mask_f[:, None, :]
        elif method == "euler":
            for i in range(n_steps):
                t0 = torch.full((B,), i * dt, device=device, dtype=X_context.dtype)
                v, _, _ = self.flow_net(x, t0, context, flow_mask, y_cond, y_is_cat)
                x = (x + v * dt) * flow_mask_f[:, None, :]
        else:
            raise ValueError(f"Unsupported ODE method: {method}")

        if sample_discrete:
            t_final = torch.ones(B, device=device, dtype=X_context.dtype)
            _, cat_logits, mask_logits = self.flow_net(x, t_final, context, flow_mask, y_cond, y_is_cat)
            if categorical_context_calibration and K > 0:
                schedule = str(cat_context_alpha_schedule or "constant").lower()
                effective_alpha = float(cat_context_alpha)
                if schedule == "logk":
                    effective_alpha *= math.log1p(float(K)) / max(math.log1p(float(self.k_max)), 1e-8)
                elif schedule != "constant":
                    raise ValueError(f"Unsupported cat_context_alpha_schedule: {cat_context_alpha_schedule}")
                cat_logits = self._apply_categorical_context_calibration(cat_logits, X_context, metadata,
                                                                         alpha=effective_alpha, tau=cat_context_tau)
            n_cont = [int(item["n_cont"]) for item in metadata]
            n_cat = [int(item["n_cat"]) for item in metadata]
            obs_probs = torch.sigmoid(mask_logits)
            obs = (torch.rand_like(obs_probs) < obs_probs).to(x.dtype)

            for b in range(B):
                nc, nk = n_cont[b], n_cat[b]
                if nc > 0:
                    if metadata[b].get("no_missingness") or metadata[b].get("force_observed_mask"):
                        obs[b, :, :nc] = 1.0
                    x[b, :, sl["mask_start"] : sl["mask_start"] + nc] = obs[b, :, :nc]
                    x[b, :, :nc] = x[b, :, :nc] * obs[b, :, :nc]
                for j in range(nk):
                    block = self.schema.category_slice(j)
                    card = int(cat_cardinalities(metadata[b], self.max_cat, self.cat_cardinality)[j])
                    probs = torch.softmax(cat_logits[b, :, j, :card], dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    x[b, :, block] = encode_category(sampled, self.schema).to(x)


        x = x * mask_f[:, None, :]
        for b, meta in enumerate(metadata):
            kind, index = self._label_info(meta)
            if kind == "categorical":
                start = self.schema.category_slice(index).start
                labels = y_cond[b].long()
                x[b, :, self.schema.category_slice(index)] = encode_category(labels, self.schema).to(x)
            else:
                x[b, :, index] = y_cond[b]
                x[b, :, sl["mask_start"] + index] = 1.0
        return x
