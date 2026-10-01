"""Raw encoded context in; conditional, fully observed encoded rows out."""
from __future__ import annotations

import numpy as np
import torch

from data.encoding import Schema, mixed_feature_mask, sanitize_mixed_encoded
from model.GenTFM import GenTFM


def context_stats(X_context_raw: np.ndarray, metadata: dict, schema: Schema) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std of active continuous context columns only."""
    x = np.asarray(X_context_raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != schema.encoded_dim:
        raise ValueError("context must have shape (K>0, schema.encoded_dim)")
    mask = mixed_feature_mask(metadata, *schema.as_tuple())
    if not np.isfinite(x[:, mask]).all():
        raise ValueError("active context values must be finite")
    mu = np.zeros((1, schema.encoded_dim), dtype=np.float32)
    std = np.ones_like(mu)
    n_cont = int(metadata["n_cont"])
    if n_cont:
        mu[:, :n_cont] = x[:, :n_cont].mean(axis=0, keepdims=True)
        scale = x[:, :n_cont].std(axis=0, keepdims=True)
        std[:, :n_cont] = np.where(scale > 1e-6, scale + 1e-6, 1.0)
    return mu, std


@torch.no_grad()
def generate_in_context(model: GenTFM, X_context_raw: np.ndarray, metadata: dict, num_gen: int,
                        schema: Schema | None = None, n_steps: int = 60, method: str = "euler",
                        device: torch.device | str | None = None, *,
                        generator: torch.Generator | None = None,
                        categorical_method: str = "sample") -> np.ndarray:
    """Fit scale on context, condition each ODE step, then restore original units.

    Context includes the target column. This wrapper does not fit or update
    model weights; a trained conditional checkpoint is required for utility.
    """
    if model.max_cont is None:
        raise ValueError("raw mixed-table generation requires a mixed schema")
    model_schema = Schema(model.max_cont, model.max_cat, model.cat_cardinality)
    schema = schema or model_schema
    if schema != model_schema:
        raise ValueError("schema must match the model")
    parameter = next(model.parameters())
    device = torch.device(device) if device is not None else parameter.device
    if device != parameter.device:
        raise ValueError("device must match the model device")
    n_cont, n_cat = int(metadata["n_cont"]), int(metadata["n_cat"])
    if not 0 <= n_cont <= schema.max_cont or not 0 <= n_cat <= schema.max_cat:
        raise ValueError("metadata exceeds model schema")
    cards = metadata.get("cat_cardinalities", [schema.cat_cardinality] * n_cat)
    if len(cards) != n_cat or any(not 2 <= int(c) <= schema.cat_cardinality for c in cards):
        raise ValueError("invalid categorical cardinalities")
    feature_mask_np = mixed_feature_mask(metadata, *schema.as_tuple())
    mu, std = context_stats(X_context_raw, metadata, schema)
    x = np.asarray(X_context_raw, dtype=np.float32).copy()
    x[:, ~feature_mask_np] = 0.0
    normalized = (x - mu) / std
    normalized[:, ~feature_mask_np] = 0.0
    mask = torch.as_tensor(feature_mask_np[None], dtype=torch.bool, device=device)
    context = torch.as_tensor(normalized[None], dtype=parameter.dtype, device=device)
    generated = model.generate(context, mask, num_gen, n_steps, method,
                               generator=generator, categorical_method=categorical_method)[0].cpu().numpy()
    if not np.isfinite(generated).all():
        raise FloatingPointError("non-finite conditional generation output")
    out = generated * std + mu
    out[:, ~feature_mask_np] = 0.0
    # This model uses fully observed data; never threshold an untrained mask head.
    metadata = {**metadata, "no_missingness": True}
    return sanitize_mixed_encoded(out, metadata, *schema.as_tuple())
