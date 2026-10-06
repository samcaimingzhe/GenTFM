"""Checkpoint helpers.

Two kinds of files exist:

* *training checkpoints* (``checkpoints/best.pt`` / ``latest.pt``) written by
  ``scripts/train.py``: model + optimizer + scheduler state, plus the model config.
* *slim checkpoints* (e.g. ``gen_tfm_target_rich_100k.pt``): model weights +
  config only.  This is what :func:`load_pretrained` expects, but it also
  accepts training checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .model import GenTFM


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location=device)


def save_training_checkpoint(path, step: int, model: GenTFM, optimizer, scheduler, best_loss: float,
                             train_config: Optional[Dict[str, object]] = None,
                             best_loss_step: Optional[int] = None,
                             best_validation_loss: float = float("inf")) -> None:
    """Save a resumable state; ``step`` counts completed optimizer updates.

    A training loss measured before update ``s`` belongs to the weights at
    ``step=s-1``. ``best_loss_step`` records that evaluated batch number.
    """
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": float(best_loss),
            "best_loss_step": None if best_loss_step is None else int(best_loss_step),
            "best_validation_loss": float(best_validation_loss),
            "train_config": train_config or {},
        },
        path,
    )


def export_slim_checkpoint(src, dst, model_config: Optional[Dict[str, object]] = None,
                           extra: Optional[Dict[str, object]] = None) -> Path:
    """Strip optimizer state from a training checkpoint and attach the model config."""
    ckpt = _torch_load(src, "cpu")
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "model_config": model_config or ckpt.get("model_config"),
        "step": int(ckpt.get("step", -1)),
        "best_loss": float(ckpt.get("best_loss", float("nan"))),
        "train_config": ckpt.get("train_config", {}),
    }
    if "best_loss_step" in ckpt:
        slim["best_loss_step"] = ckpt["best_loss_step"]
    if slim["model_config"] is None:
        raise ValueError("model_config must be given for checkpoints that do not store it")
    if extra:
        slim.update(extra)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    return dst


def load_pretrained(path, device: str = "cpu", model_config: Optional[Dict[str, object]] = None) -> Tuple[GenTFM, Dict[str, object]]:
    """Build a :class:`GenTFM` from a checkpoint and return ``(model, checkpoint_dict)``.

    The model config is read from the checkpoint; ``model_config`` overrides it
    (needed only for legacy checkpoints that do not store a config).
    """
    device = torch.device(device)
    ckpt = _torch_load(path, device)
    stored = dict(ckpt.get("model_config") or {})
    if not stored and not model_config:
        raise ValueError(f"{path} stores no model_config; pass model_config=... explicitly")
    config = dict(stored or model_config)
    config.setdefault("cat_encoding", "onehot")
    config.setdefault("binary_bit_order", "msb_first")
    config.setdefault("query_conditioning", "context_only")
    config.setdefault("schema_version", f"mixed_{config['cat_encoding']}_v1")
    if model_config and stored:
        for key, value in model_config.items():
            if key in config and value != config[key]:
                raise ValueError(f"Checkpoint configuration conflict: {key}")
        config.update(model_config)
    model = GenTFM(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
