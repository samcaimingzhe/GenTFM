"""Unconditional Euler and Heun ODE sampling."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from data.encoding import Schema, categorical_feature_mask, velocity_feature_mask


@torch.no_grad()
def sample_table(
    model: nn.Module, feature_mask: Tensor, num_rows: int,
    n_steps: int = 60, method: str = "euler", *,
    generator: torch.Generator | None = None,
    categorical_method: str = "sample",
) -> Tensor:
    """Integrate Gaussian noise from t=0 to t=1; return (B, num_rows, D).

    Mixed-schema models keep observed bits fixed at one and decode final
    categories with the CE head ('sample' or 'argmax'). 'none' returns raw
    one-hot flow coordinates for ablations. Legacy models return raw coordinates.
    The generator and feature_mask must be on the model's device.
    """
    if not isinstance(num_rows, int) or num_rows < 1:
        raise ValueError("num_rows must be a positive integer")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if method not in ("euler", "heun"):
        raise ValueError("method must be 'euler' or 'heun'")
    if categorical_method not in ("sample", "argmax", "none"):
        raise ValueError("categorical_method must be 'sample', 'argmax', or 'none'")
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
    schema = None
    if getattr(model, "max_cont", None) is not None:
        schema = Schema(model.max_cont, model.max_cat, model.cat_cardinality)
        velocity_feature_mask(feature_mask, schema)
        x[..., schema.mask_start:] = feature_mask[:, None, schema.mask_start:].to(x.dtype)
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
        if schema is not None and categorical_method != "none":
            outputs = model(x, times[-1].expand(B), feature_mask, return_aux=True)
            logits = outputs["categorical_logits"]
            classes = categorical_feature_mask(feature_mask, schema)
            active = classes.any(-1)[:, None, :].expand(B, num_rows, schema.max_cat)
            selected = logits[active].float()
            allowed = classes[:, None].expand_as(logits)[active]
            one_hot = torch.zeros_like(logits, dtype=x.dtype)
            if selected.shape[0] > 0:
                selected = selected.masked_fill(~allowed, -torch.inf)
                if categorical_method == "sample":
                    labels = torch.multinomial(selected.softmax(-1), 1, generator=generator).squeeze(-1)
                else:
                    labels = selected.argmax(-1)
                one_hot[active] = torch.nn.functional.one_hot(labels, schema.cat_cardinality).to(x.dtype)
            x[..., schema.cat_start:schema.mask_start] = one_hot.reshape(B, num_rows, -1)
    finally:
        for module, training in modes:
            module.training = training
    return x
