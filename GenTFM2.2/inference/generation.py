"""User-facing in-context generation: raw encoded context rows in, valid encoded rows out.

The model works in a per-table standardised space.  This module takes care of

1. computing the context statistics (mean / std of the continuous columns),
2. normalising the context, running :meth:`GenTFM.generate`,
3. un-normalising the result and snapping it to valid mixed rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from data.encoding import Schema, mixed_feature_mask, sanitize_mixed_encoded
from model.GenTFM import GenTFM


@dataclass(frozen=True)
class Calibration:
    """Generation-time categorical context calibration.

    ``cat_logits += alpha_eff * log(smoothed context category frequencies)`` with
    ``alpha_eff = alpha * log(1+K) / log(1+k_max)`` for the ``logk`` schedule.
    The defaults are the frozen final setting (2026-06-01).
    """

    alpha: float = 3.5
    schedule: str = "logk"   # 'logk' or 'constant'
    tau: float = 0.25        # additive smoothing of the frequency estimate

    def as_generate_kwargs(self) -> Dict[str, object]:
        return {
            "categorical_context_calibration": True,
            "cat_context_alpha": float(self.alpha),
            "cat_context_alpha_schedule": str(self.schedule),
            "cat_context_tau": float(self.tau),
        }


DEFAULT_CALIBRATION = Calibration()
NO_CALIBRATION: Optional[Calibration] = None


def context_stats(X_context_raw: np.ndarray, metadata: Dict[str, object], schema: Schema) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std of the active continuous columns, ones/zeros elsewhere."""
    D = schema.encoded_dim
    x = np.asarray(X_context_raw, dtype=np.float32)
    mu = np.zeros((1, D), dtype=np.float32)
    std = np.ones((1, D), dtype=np.float32)
    n_cont = int(metadata["n_cont"])
    if n_cont > 0:
        mu[:, :n_cont] = x[:, :n_cont].mean(axis=0, keepdims=True)
        std[:, :n_cont] = x[:, :n_cont].std(axis=0, keepdims=True) + 1e-6
    return mu, std


@torch.no_grad()
def generate_in_context(model: GenTFM, X_context_raw: np.ndarray, metadata: Dict[str, object], num_gen: int,
                        schema: Optional[Schema] = None, n_steps: int = 60, method: str = "euler",
                        calibration: Optional[Calibration] = DEFAULT_CALIBRATION,
                        device: Optional[torch.device] = None) -> np.ndarray:
    """Generate ``num_gen`` encoded rows that follow the table described by the K context rows.

    Parameters
    ----------
    model
        A :class:`GenTFM` (already on ``device``, in eval mode).
    X_context_raw
        ``[K, D]`` encoded context rows in the original (un-normalised) scale.
    metadata
        The table description produced by the encoder (``n_cont``, ``n_cat``, ...).
    num_gen
        Number of rows to sample.
    n_steps, method
        ODE integration settings (``euler`` with 60 steps was used for all final results).
    calibration
        ``None`` disables the categorical context calibration.
    """
    schema = schema or Schema(model.max_cont, model.max_cat, model.cat_cardinality)
    device = device or next(model.parameters()).device
    feature_mask_np = mixed_feature_mask(metadata, *schema.as_tuple())
    mu, std = context_stats(X_context_raw, metadata, schema)
    X_norm = (np.asarray(X_context_raw, dtype=np.float32) - mu) / std
    X_norm[:, ~feature_mask_np] = 0.0
    feature_mask = torch.as_tensor(feature_mask_np[None, :], dtype=torch.bool, device=device)
    X_ctx = torch.from_numpy(X_norm).unsqueeze(0).to(device)
    kwargs = calibration.as_generate_kwargs() if calibration is not None else {}
    X_gen_norm = model.generate(X_ctx, feature_mask=feature_mask, num_gen=int(num_gen), n_steps=int(n_steps),
                                method=method, metadata=[metadata], **kwargs)[0].cpu().numpy()
    out = X_gen_norm * std + mu
    out[:, ~feature_mask_np] = 0.0
    return sanitize_mixed_encoded(out, metadata, *schema.as_tuple())


@torch.no_grad()
def generate_zero_context(model: GenTFM, X_reference_context: np.ndarray, metadata: Dict[str, object], num_gen: int,
                          schema: Optional[Schema] = None, n_steps: int = 60, method: str = "euler",
                          device: Optional[torch.device] = None) -> np.ndarray:
    """Ablation: same schema and scale, but the model sees an all-zero context (no information)."""
    schema = schema or Schema(model.max_cont, model.max_cat, model.cat_cardinality)
    device = device or next(model.parameters()).device
    feature_mask_np = mixed_feature_mask(metadata, *schema.as_tuple())
    mu, std = context_stats(X_reference_context, metadata, schema)
    feature_mask = torch.as_tensor(feature_mask_np[None, :], dtype=torch.bool, device=device)
    X_ctx = torch.zeros(1, 1, schema.encoded_dim, device=device)
    X_gen_norm = model.generate(X_ctx, feature_mask=feature_mask, num_gen=int(num_gen), n_steps=int(n_steps),
                                method=method, metadata=[metadata])[0].cpu().numpy()
    out = X_gen_norm * std + mu
    out[:, ~feature_mask_np] = 0.0
    return sanitize_mixed_encoded(out, metadata, *schema.as_tuple())
