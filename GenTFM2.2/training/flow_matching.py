"""Linear flow paths and masked cell-wise velocity MSE for training."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from data.encoding import Schema, categorical_feature_mask, velocity_feature_mask


def _valid_cells(x: Tensor, feature_mask: Tensor) -> Tensor:
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError("x must have shape (B, K, D) with K > 0")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if feature_mask.shape != (x.shape[0], x.shape[2]):
        raise ValueError("feature_mask must have shape (B, D)")
    if feature_mask.dtype != torch.bool:
        raise TypeError("feature_mask must be boolean")
    if feature_mask.device != x.device:
        raise ValueError("x and feature_mask must be on the same device")
    return feature_mask[:, None, :].expand_as(x)


def sample_flow_batch(
    x_1: Tensor, feature_mask: Tensor, *, generator: torch.Generator | None = None,
    schema: Schema | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, t, target_velocity) for x_t=(1-t)*x_0+t*x_1.

    x_0 is standard Gaussian noise. A single uniform time is sampled per table.
    With a schema, only numerical and one-hot coordinates follow this path;
    observed bits stay fixed at one. Missing observations are rejected.
    Without a schema the original all-coordinate path is retained.
    """
    valid = _valid_cells(x_1, feature_mask)
    x_1 = x_1.masked_fill(~valid, 0.0)
    fixed = torch.zeros_like(valid)
    if schema is not None:
        flow_mask = velocity_feature_mask(feature_mask, schema)
        fixed = valid & ~flow_mask[:, None, :]
        if not (x_1[fixed] == 1).all():
            raise ValueError("missing observations are unsupported; observed bits must be one")
    x_0 = torch.randn(x_1.shape, dtype=x_1.dtype, device=x_1.device, generator=generator)
    x_0 = x_0.masked_fill(~valid, 0.0)
    x_0 = x_0.masked_fill(fixed, 1.0)
    t = torch.rand(x_1.shape[0], dtype=x_1.dtype, device=x_1.device, generator=generator)
    t_view = t[:, None, None]
    return (1 - t_view) * x_0 + t_view * x_1, t, x_1 - x_0


def masked_velocity_loss(prediction: Tensor, target: Tensor, feature_mask: Tensor) -> Tensor:
    """Mean squared velocity error across all valid cells in the batch.

    Every encoded coordinate has equal weight, including categorical one-hots.
    An all-padding batch returns differentiable zero, even with NaN padding.
    """
    valid = _valid_cells(prediction, feature_mask)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape")
    if target.device != prediction.device:
        raise ValueError("target and prediction must be on the same device")
    if not target.is_floating_point():
        raise TypeError("target must be floating point")
    error = prediction.masked_fill(~valid, 0.0) - target.masked_fill(~valid, 0.0)
    if error.dtype in (torch.float16, torch.bfloat16):
        error = error.float()
    return error.square().sum() / valid.sum().clamp_min(1)


def categorical_cross_entropy(logits: Tensor, x_1: Tensor,
                              feature_mask: Tensor, schema: Schema) -> Tensor:
    """Mean CE over valid original categorical cells, with invalid classes removed.

    Targets are clean x_1 one-hots, rather than noisy x_t or velocities.
    Empty categorical fields/tables contribute differentiable zero.
    """
    _valid_cells(x_1, feature_mask)
    B, K, _ = x_1.shape
    if logits.shape != (B, K, schema.max_cat, schema.cat_cardinality):
        raise ValueError("categorical logits have the wrong shape")
    if logits.device != x_1.device or not logits.is_floating_point():
        raise ValueError("logits must be floating point and on the target device")
    classes = categorical_feature_mask(feature_mask, schema)
    active = classes.any(dim=-1)[:, None, :].expand(B, K, schema.max_cat)
    # Select active fields first: no all-masked softmax and no NaN padding.
    selected = logits[active]
    if selected.shape[0] == 0:
        return selected.sum()
    allowed = classes[:, None].expand_as(logits)[active]
    targets = x_1[:, :, schema.cat_start:schema.mask_start].reshape_as(logits)[active]
    targets = targets.masked_fill(~allowed, 0.0)
    if not (((targets == 0) | (targets == 1)).all() and (targets.sum(-1) == 1).all()):
        raise ValueError("active categorical targets must be clean one-hot vectors")
    selected = selected.float().masked_fill(~allowed, -torch.inf)
    return F.cross_entropy(selected, targets.argmax(dim=-1))


def flow_matching_loss(outputs: dict, target_velocity: Tensor, x_1: Tensor,
                       feature_mask: Tensor, schema: Schema,
                       categorical_weight: float = 1.0) -> dict[str, Tensor]:
    """Total = valid numerical/one-hot velocity MSE + weight * categorical CE."""
    if not 0 <= categorical_weight < float("inf"):
        raise ValueError("categorical_weight must be finite and nonnegative")
    mse = masked_velocity_loss(outputs["velocity"], target_velocity,
                               velocity_feature_mask(feature_mask, schema))
    ce = categorical_cross_entropy(outputs["categorical_logits"], x_1, feature_mask, schema)
    return {"loss": mse + categorical_weight * ce, "velocity_mse": mse, "categorical_ce": ce}
