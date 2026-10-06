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

The parameter names match the frozen checkpoint (target-rich 100k), so the
checkpoint loads with ``strict=True``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import encoded_dim, slices


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
        n_cat = int(row[sl["cat_start"] : sl["mask_start"]].sum().item()) // cat_cardinality
        metadata.append({
            "n_cont": max(1, min(n_cont, max_cont)),
            "n_cat": max(1, min(n_cat, max_cat)),
            "cat_cardinality": cat_cardinality,
        })
    return metadata


def normalize_mixed_batch(x: torch.Tensor, metadata: List[Dict[str, object]], max_cont: int, max_cat: int, cat_cardinality: int, eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-table standardisation of the continuous columns only.

    Categorical one-hots and mask bits are left untouched, inactive dimensions
    are zeroed.  This is what makes the model scale-invariant per table.
    """
    x_norm = x.clone()
    mean = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    std = torch.ones_like(mean)
    sl = slices(max_cont, max_cat, cat_cardinality)

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

        active = torch.zeros(x.shape[2], device=x.device, dtype=torch.bool)
        active[:n_cont] = True
        active[sl["cat_start"] : sl["cat_start"] + n_cat * cat_cardinality] = True
        active[sl["mask_start"] : sl["mask_start"] + n_cont] = True
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

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross_attn(query=self.cross_norm(x), key=context, value=context, need_weights=False)
        x = x + cross_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MixedFlowNet(nn.Module):
    """Velocity / categorical / mask heads on top of ``n_layers`` cross-attention blocks."""

    def __init__(self, max_features: int, max_cont: int, max_cat: int, cat_cardinality: int, hidden_dim: int,
                 time_dim: int = 256, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.max_features = max_features
        self.max_cont = max_cont
        self.max_cat = max_cat
        self.cat_cardinality = cat_cardinality
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.x_proj = nn.Linear(max_features * 2, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)
        self.layers = nn.ModuleList([TabularFlowBlock(hidden_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, max_features)
        self.cat_head = nn.Linear(hidden_dim, max_cat * cat_cardinality)
        self.mask_head = nn.Linear(hidden_dim, max_cont)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor, feature_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x_t.shape
        mask = feature_mask.to(device=x_t.device, dtype=x_t.dtype)
        x_t = x_t * mask[:, None, :]
        mask_rows = mask[:, None, :].expand(B, N, D)

        t_emb = self.t_proj(self.time_embed(t)).unsqueeze(1)
        h = self.x_proj(torch.cat([x_t, mask_rows], dim=-1)) + t_emb
        for layer in self.layers:
            h = layer(h, context)
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
                 discrete_flow_weight: float = 0.05, k_max: int = 200):
        super().__init__()
        self.max_cont = max_cont
        self.max_cat = max_cat
        self.cat_cardinality = cat_cardinality
        self.max_features = encoded_dim(max_cont, max_cat, cat_cardinality)
        self.hidden_dim = hidden_dim
        self.cat_loss_weight = cat_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.discrete_flow_weight = discrete_flow_weight
        self.k_max = max(1, int(k_max))
        self.encoder = ContextEncoder(self.max_features, hidden_dim, n_heads=n_heads, n_layers=n_enc_layers, dropout=dropout)
        self.flow_net = MixedFlowNet(self.max_features, max_cont, max_cat, cat_cardinality, hidden_dim,
                                     time_dim=time_dim, n_layers=n_flow_layers, n_heads=n_heads, dropout=dropout)

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
            feature_mask = infer_feature_mask(X_full)
        feature_mask = feature_mask.to(device=X_full.device, dtype=torch.bool)
        if metadata is None:
            metadata = metadata_from_feature_mask(feature_mask, self.max_cont, self.max_cat, self.cat_cardinality)
        if len(metadata) != B:
            raise ValueError(f"Expected {B} metadata entries, got {len(metadata)}")
        mask_f = feature_mask.float()

        if normalize:
            X_model, _ = normalize_mixed_batch(X_full, metadata, self.max_cont, self.max_cat, self.cat_cardinality)
        else:
            X_model = X_full * mask_f[:, None, :]

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
        X_model_shuffled = torch.gather(X_model, 1, idx.unsqueeze(-1).expand_as(X_model))
        X_raw_shuffled = torch.gather(X_full, 1, idx.unsqueeze(-1).expand_as(X_full))
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

        cat_losses = []
        for j in range(self.max_cat):
            active_b = cat_active[:, j]
            if not bool(active_b.any()):
                continue
            start = sl["cat_start"] + j * self.cat_cardinality
            target = X_target_raw[:, :, start : start + self.cat_cardinality].argmax(dim=-1)
            loss_j = F.cross_entropy(
                cat_logits[:, :, j, :][active_b].reshape(-1, self.cat_cardinality),
                target[active_b].reshape(-1),
            )
            cat_losses.append(loss_j)
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else cfm_loss.new_zeros(())

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
            feature_mask = infer_feature_mask(X_context)
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
                    x[b, :, sl["mask_start"] : sl["mask_start"] + nc] = obs[b, :, :nc]
                    x[b, :, :nc] = x[b, :, :nc] * obs[b, :, :nc]
                for j in range(nk):
                    start = sl["cat_start"] + j * self.cat_cardinality
                    probs = torch.softmax(cat_logits[b, :, j, :], dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    x[b, :, start : start + self.cat_cardinality] = F.one_hot(sampled, num_classes=self.cat_cardinality).to(x.dtype)

        return x * mask_f[:, None, :]
