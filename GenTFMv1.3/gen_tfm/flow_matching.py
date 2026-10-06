"""Binary diffusion path with discrete flow matching, using the unchanged v1.1 network.

Time runs from noise (0) to data (1). Each clean bit is XOR-corrupted with
Bernoulli((1-t)/2) noise. The endpoint-prediction parameterization of discrete
flow matching learns p(x_1 | x_t, t, context, y) with BCE. Its off-diagonal
probability velocity is p(x_1=1-x_t | ...)/(1-t), sampled as binary CTMC jumps.
This adapts the reference paper's binary representation and XOR path to flow
matching; it does not replace the network with the paper's two-head denoiser.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .encoding import (Schema, batch_feature_mask, cat_cardinalities,
                       category_bits_to_ids, category_ids_to_bits,
                       float32_bits_to_values, float32_values_to_bits)
from .model import GenTFM as Network

REPRESENTATION_VERSION = "binary_float32_v1_3"


def xor_corrupt(clean, t):
    """t=0 gives fair binary noise; t=1 leaves the clean bits unchanged."""
    flip = torch.rand_like(clean) < (1.0 - t[:, None, None]) * 0.5
    return torch.logical_xor(clean.bool(), flip).to(clean.dtype)


def binary_flow_step(bits, clean_probs, t: float, dt: float):
    """Euler step for the discrete FM probability velocity; state stays binary."""
    if not 0 <= t < 1 or not 0 < dt <= 1 - t + 1e-7:
        raise ValueError("Invalid discrete flow time step")
    disagree = torch.where(bits.bool(), 1.0 - clean_probs, clean_probs)
    flip_probs = (disagree * (dt / (1.0 - t))).clamp(0.0, 1.0)
    flip = torch.rand_like(bits) < flip_probs
    return torch.logical_xor(bits.bool(), flip).to(bits.dtype)


def validate_checkpoint(ckpt):
    if ckpt.get("model_config", {}).get("representation_version") != REPRESENTATION_VERSION:
        raise ValueError("v1.3 requires a new binary-float32 checkpoint; v1/v1.1/v1.2 weights are incompatible")


class GenTFM(Network):
    """v1.3 training/sampling adapter; inherits every neural module unchanged."""

    def __init__(self, *args, representation_version=REPRESENTATION_VERSION, **kwargs):
        if representation_version != REPRESENTATION_VERSION:
            raise ValueError("Unsupported representation_version")
        super().__init__(*args, **kwargs)
        self.schema = Schema(self.max_cont, self.max_cat, self.cat_cardinality)

    def config(self):
        return dict(super().config(), representation_version=REPRESENTATION_VERSION)

    def _metadata_mask(self, rows, metadata, feature_mask):
        if metadata is None or len(metadata) != rows.shape[0]:
            raise ValueError("v1.3 requires one explicit metadata entry per table")
        if rows.shape[-1] != self.max_features or rows.dtype != torch.float32:
            raise ValueError("Expected v1.3 binary encoded rows stored as float32")
        if not bool(((rows == 0) | (rows == 1)).all()):
            raise ValueError("Expected exact binary bits, not raw numeric values")
        expected = batch_feature_mask(metadata, *self.schema.as_tuple(), rows.device)
        if feature_mask is not None:
            supplied = feature_mask.to(device=rows.device, dtype=torch.bool)
            if supplied.shape != expected.shape or not torch.equal(supplied, expected):
                raise ValueError("feature_mask disagrees with metadata")
        return expected

    def _label_info(self, metadata):
        # Preserve v1.1's supervised target convention, validating active fields.
        kind, index = super()._label_info(metadata)
        limit = int(metadata["n_cat"] if kind == "categorical" else metadata["n_cont"])
        if not 0 <= index < limit:
            raise ValueError("Supervised label index outside active fields")
        return kind, index

    def _extract_y(self, rows, metadata):
        B, N, _ = rows.shape
        y = rows.new_zeros(B, N)
        y_is_cat = torch.zeros(B, dtype=torch.bool, device=rows.device)
        keep = torch.ones(B, self.max_features, dtype=torch.bool, device=rows.device)
        for b, meta in enumerate(metadata):
            kind, index = self._label_info(meta)
            if kind == "categorical":
                start = self.schema.cat_start + index * self.schema.cat_width
                block = slice(start, start + self.schema.cat_width)
                y[b] = category_bits_to_ids(rows[b, :, block]).float()
                y_is_cat[b] = True
            else:
                block = slice(index * 32, (index + 1) * 32)
                y[b] = float32_bits_to_values(rows[b, :, block])
                keep[b, self.schema.mask_start + index] = False
            keep[b, block] = False
        return y, y_is_cat, keep

    def _flow_mask(self, feature_mask, keep, metadata):
        mask = feature_mask & keep
        for b, meta in enumerate(metadata):
            if meta.get("no_missingness") or meta.get("force_observed_mask"):
                mask[b, self.schema.mask_start:] = False
        return mask

    def _type_weights(self, feature_mask):
        weights = feature_mask.float()
        weights[:, self.schema.cat_start:] *= self.discrete_flow_weight
        return weights

    def _legal_logits(self, logits, metadata):
        logits = logits.clone()
        for b, meta in enumerate(metadata):
            for j, card in enumerate(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)):
                logits[b, :, j, int(card):] = -torch.inf
        return logits

    def _apply_categorical_context_calibration(self, cat_logits, X_context, feature_mask, alpha, tau):
        logits = cat_logits.clone()
        tau = max(float(tau), 1e-8)
        for j in range(self.max_cat):
            start = self.schema.cat_start + j * self.schema.cat_width
            block = slice(start, start + self.schema.cat_width)
            for b in range(len(X_context)):
                if not bool(feature_mask[b, block].any()):
                    continue
                ids = category_bits_to_ids(X_context[b, :, block])
                counts = torch.bincount(ids, minlength=self.cat_cardinality).to(logits.dtype)
                probs = (counts + tau) / (counts.sum() + tau * self.cat_cardinality)
                logits[b, :, j] += float(alpha) * probs.log()
        return logits

    def compute_loss(self, X_full, metadata=None, feature_mask=None, min_ctx=5, max_ctx=200,
                     min_target=512, normalize=True, context_sizes=None):
        # Numeric min-max scaling and IEEE-754 encoding happen in the data layer.
        # Never normalize bitstrings, even when legacy callers pass normalize=True.
        feature_mask = self._metadata_mask(X_full, metadata, feature_mask)
        B, N, D = X_full.shape
        high = min(max_ctx, N - min_target)
        if min_ctx < 1 or high < min_ctx:
            raise ValueError("Not enough rows for context/target split")
        valid = list(range(min_ctx, high + 1)) if not context_sizes else [int(k) for k in context_sizes if min_ctx <= int(k) <= high]
        if not valid:
            raise ValueError("No context_sizes fit split constraints")
        K = valid[torch.randint(len(valid), (1,), device=X_full.device).item()]
        idx = torch.argsort(torch.rand(B, N, device=X_full.device), dim=1)
        rows = torch.gather(X_full, 1, idx[..., None].expand(B, N, D))
        X_ctx, target = rows[:, :K], rows[:, K:]
        context = self.encoder(X_ctx, feature_mask)
        y, is_cat, keep = self._extract_y(target, metadata)
        flow_mask = self._flow_mask(feature_mask, keep, metadata)
        clean = target * flow_mask[:, None, :]
        t = torch.rand(B, device=X_full.device)
        noisy = xor_corrupt(clean, t) * flow_mask[:, None, :]
        # The existing velocity head parameterizes the endpoint Bernoulli
        # distribution, which determines the discrete FM probability velocity.
        clean_logits, cat_logits, mask_logits = self.flow_net(noisy, t, context, flow_mask, y, is_cat)
        weights = self._type_weights(flow_mask)
        bit_loss = F.binary_cross_entropy_with_logits(clean_logits, clean, reduction="none")
        dfm_loss = (bit_loss * weights[:, None, :]).sum() / (weights.sum() * target.shape[1]).clamp_min(1)
        cat_logits = self._legal_logits(cat_logits, metadata)
        cat_losses = []
        for b, meta in enumerate(metadata):
            for j in range(int(meta["n_cat"])):
                if self._label_info(meta) == ("categorical", j):
                    continue
                start = self.schema.cat_start + j * self.schema.cat_width
                ids = category_bits_to_ids(target[b, :, start:start + self.schema.cat_width])
                cat_losses.append(F.cross_entropy(cat_logits[b, :, j], ids))
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else dfm_loss.new_zeros(())
        obs_target = target[:, :, self.schema.mask_start:]
        obs_active = feature_mask[:, self.schema.mask_start:] & keep[:, self.schema.mask_start:]
        bce = F.binary_cross_entropy_with_logits(mask_logits, obs_target, reduction="none")
        mask_loss = (bce * obs_active[:, None, :]).sum() / (obs_active.sum() * target.shape[1]).clamp_min(1)
        return dfm_loss + self.cat_loss_weight * cat_loss + self.mask_loss_weight * mask_loss

    @torch.no_grad()
    def generate(self, X_context, feature_mask=None, num_gen=500, n_steps=60, method="euler", metadata=None,
                 sample_discrete=True, categorical_context_calibration=False, cat_context_alpha=0.0,
                 cat_context_alpha_schedule="constant", cat_context_tau=1.0, target_y=None):
        if X_context.dim() == 2:
            X_context = X_context.unsqueeze(0)
        feature_mask = self._metadata_mask(X_context, metadata, feature_mask)
        if n_steps < 1 or num_gen < 1 or method not in {"euler", "heun"}:
            raise ValueError("Positive n_steps/num_gen and euler/heun method required")
        B, K, D = X_context.shape
        device = X_context.device
        context_y, is_cat, keep = self._extract_y(X_context, metadata)
        if target_y is None:
            if K == 0:
                raise ValueError("Empty context requires explicit target_y")
            picks = torch.randint(K, (B, num_gen), device=device)
            y = context_y.gather(1, picks)
        else:
            y = torch.as_tensor(target_y, device=device, dtype=torch.float32)
            if y.dim() == 0:
                y = y.expand(B, num_gen)
            elif y.dim() == 1:
                if y.numel() == B:
                    y = y[:, None].expand(B, num_gen)
                elif B == 1 and y.numel() == num_gen:
                    y = y[None, :]
            if y.shape != (B, num_gen) or not bool(torch.isfinite(y).all()):
                raise ValueError("target_y must be finite scalar, [B], or [B, num_gen]")
            for b, meta in enumerate(metadata):
                kind, index = self._label_info(meta)
                if kind == "categorical":
                    card = int(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)[index])
                    if bool(((y[b] != y[b].long()) | (y[b] < 0) | (y[b] >= card)).any()):
                        raise ValueError("target_y outside categorical cardinality")
                elif bool(((y[b] < 0) | (y[b] > 1)).any()):
                    raise ValueError("Regression target_y must be min-max normalized to [0,1]")
        flow_mask = self._flow_mask(feature_mask, keep, metadata)
        context = self.encoder(X_context if K else X_context.new_zeros(B, 1, D), feature_mask)
        x = torch.randint(2, (B, num_gen, D), device=device).float() * flow_mask[:, None, :]
        # Both legacy solver flags use binary probability transitions. Heun's
        # continuous state update would destroy the exact binary state.
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = i / n_steps
            times = x.new_full((B,), t)
            logits, _, _ = self.flow_net(x, times, context, flow_mask, y, is_cat)
            x = binary_flow_step(x, logits.sigmoid(), t, dt) * flow_mask[:, None, :]
        if sample_discrete:
            _, cat_logits, mask_logits = self.flow_net(x, x.new_ones(B), context, flow_mask, y, is_cat)
            if categorical_context_calibration:
                alpha = float(cat_context_alpha)
                if cat_context_alpha_schedule == "logk":
                    alpha *= math.log1p(K) / math.log1p(self.k_max)
                elif cat_context_alpha_schedule != "constant":
                    raise ValueError("Unsupported calibration schedule")
                cat_logits = self._apply_categorical_context_calibration(cat_logits, X_context, feature_mask, alpha, cat_context_tau)
            cat_logits = self._legal_logits(cat_logits, metadata)
            for b, meta in enumerate(metadata):
                for j in range(int(meta["n_cat"])):
                    start = self.schema.cat_start + j * self.schema.cat_width
                    ids = torch.multinomial(cat_logits[b, :, j].softmax(-1), 1).squeeze(-1)
                    x[b, :, start:start + self.schema.cat_width] = category_ids_to_bits(ids, self.schema.cat_width)
                nc = int(meta["n_cont"])
                x[b, :, self.schema.mask_start:self.schema.mask_start + nc] = (torch.rand_like(mask_logits[b, :, :nc]) < mask_logits[b, :, :nc].sigmoid()).float()
        for b, meta in enumerate(metadata):
            nc = int(meta["n_cont"])
            obs_slice = slice(self.schema.mask_start, self.schema.mask_start + nc)
            if meta.get("no_missingness") or meta.get("force_observed_mask"):
                x[b, :, obs_slice] = 1
            kind, index = self._label_info(meta)
            if kind == "categorical":
                start = self.schema.cat_start + index * self.schema.cat_width
                x[b, :, start:start + self.schema.cat_width] = category_ids_to_bits(y[b], self.schema.cat_width)
            else:
                x[b, :, index * 32:(index + 1) * 32] = float32_values_to_bits(y[b])
                x[b, :, self.schema.mask_start + index] = 1
            # Generated IEEE-754 bitstrings may spell NaN/inf/out-of-range values.
            vals = float32_bits_to_values(x[b, :, :nc * 32].reshape(num_gen, nc, 32))
            vals = torch.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            vals *= x[b, :, obs_slice]
            x[b, :, :nc * 32] = float32_values_to_bits(vals).reshape(num_gen, nc * 32)
            for j, card in enumerate(cat_cardinalities(meta, self.max_cat, self.cat_cardinality)):
                start = self.schema.cat_start + j * self.schema.cat_width
                codes = category_ids_to_bits(torch.arange(int(card), device=device), self.schema.cat_width)
                block = x[b, :, start:start + self.schema.cat_width]
                nearest = ((block[:, None, :] - codes[None]) ** 2).sum(-1).argmin(-1)
                x[b, :, start:start + self.schema.cat_width] = codes[nearest]
        return x * feature_mask[:, None, :]
