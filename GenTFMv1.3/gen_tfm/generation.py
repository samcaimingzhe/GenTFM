"""Generation over binary encoded rows; numeric scaling lives in the codec metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .encoding import Schema, mixed_feature_mask, sanitize_mixed_encoded
from .flow_matching import GenTFM


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
        ``[K, D]`` binary encoded context rows from the v1.3 encoder.
    metadata
        The table description produced by the encoder (``n_cont``, ``n_cat``, ...).
    num_gen
        Number of rows to sample.
    n_steps, method
        Binary flow sampling settings; legacy euler/heun names are accepted.
    calibration
        ``None`` disables the categorical context calibration.
    """
    schema = schema or Schema(model.max_cont, model.max_cat, model.cat_cardinality)
    device = device or next(model.parameters()).device
    feature_mask_np = mixed_feature_mask(metadata, *schema.as_tuple())
    X_norm = sanitize_mixed_encoded(X_context_raw, metadata, *schema.as_tuple())
    X_norm[:, ~feature_mask_np] = 0.0
    feature_mask = torch.as_tensor(feature_mask_np[None, :], dtype=torch.bool, device=device)
    X_ctx = torch.from_numpy(X_norm).unsqueeze(0).to(device)
    kwargs = calibration.as_generate_kwargs() if calibration is not None else {}
    X_gen_norm = model.generate(X_ctx, feature_mask=feature_mask, num_gen=int(num_gen), n_steps=int(n_steps),
                                method=method, metadata=[metadata], **kwargs)[0].cpu().numpy()
    out = X_gen_norm
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
    feature_mask = torch.as_tensor(feature_mask_np[None, :], dtype=torch.bool, device=device)
    X_ctx = torch.zeros(1, 1, schema.encoded_dim, device=device)
    X_gen_norm = model.generate(X_ctx, feature_mask=feature_mask, num_gen=int(num_gen), n_steps=int(n_steps),
                                method=method, metadata=[metadata])[0].cpu().numpy()
    out = X_gen_norm
    out[:, ~feature_mask_np] = 0.0
    return sanitize_mixed_encoded(out, metadata, *schema.as_tuple())
