"""Linear flow paths and masked cell-wise velocity MSE for training."""
from __future__ import annotations

import torch
from torch import Tensor


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
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (x_t, t, target_velocity) for x_t=(1-t)*x_0+t*x_1.

    x_0 is standard Gaussian noise. A single uniform time is sampled per table.
    Continuous, one-hot, and observed-mask coordinates all follow this path.
    """
    valid = _valid_cells(x_1, feature_mask)
    x_1 = x_1.masked_fill(~valid, 0.0)
    x_0 = torch.randn(x_1.shape, dtype=x_1.dtype, device=x_1.device, generator=generator)
    x_0 = x_0.masked_fill(~valid, 0.0)
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
