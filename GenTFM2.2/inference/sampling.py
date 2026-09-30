"""Unconditional Euler and Heun ODE sampling."""
from __future__ import annotations

import torch
from torch import Tensor, nn


@torch.no_grad()
def sample_table(
    model: nn.Module, feature_mask: Tensor, num_rows: int,
    n_steps: int = 60, method: str = "euler", *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate Gaussian noise from t=0 to t=1; return (B, num_rows, D).

    Returns continuous encoded coordinates, without categorical sanitization.
    The generator and feature_mask must be on the model's device.
    """
    if not isinstance(num_rows, int) or num_rows < 1:
        raise ValueError("num_rows must be a positive integer")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if method not in ("euler", "heun"):
        raise ValueError("method must be 'euler' or 'heun'")
    if feature_mask.ndim != 2:
        raise ValueError("feature_mask must have shape (B, D)")
    parameter = next(model.parameters())
    B, D = feature_mask.shape
    x = torch.randn(
        B, num_rows, D, device=parameter.device, dtype=parameter.dtype, generator=generator,
    )
    if feature_mask.dtype != torch.bool:
        raise TypeError("feature_mask must be boolean")
    if feature_mask.device != x.device:
        raise ValueError("model and feature_mask must be on the same device")
    valid = feature_mask[:, None, :].expand_as(x)
    x = x.masked_fill(~valid, 0.0)
    if not valid.any():
        return x
    times = torch.linspace(0, 1, n_steps + 1, device=x.device, dtype=x.dtype)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for step in range(n_steps):
            dt = times[step + 1] - times[step]
            v = model(x, times[step].expand(B), feature_mask)
            if method == "euler":
                x = x + dt * v
            else:
                proposal = (x + dt * v).masked_fill(~valid, 0.0)
                v_next = model(proposal, times[step + 1].expand(B), feature_mask)
                x = x + dt * (v + v_next) / 2
            x = x.masked_fill(~valid, 0.0)
    finally:
        for module, training in modes:
            module.training = training
    return x
