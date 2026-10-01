"""Checkpoint helpers.

Two kinds of files exist:

* *training checkpoints* (``checkpoints/best.pt`` / ``latest.pt``) written by
  ``script/train.py``: model + optimizer + scheduler state, plus the model config.
* *slim checkpoints* (e.g. ``gen_tfm_target_rich_100k.pt``): model weights +
  config only.  This is what :func:`load_pretrained` expects, but it also
  accepts training checkpoints.
"""

from __future__ import annotations

from pathlib import Path
import os
import random
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from model.GenTFM import GenTFM


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location=device)


def save_training_checkpoint(path, step: int, model: GenTFM, optimizer, scheduler, best_loss: float,
                             train_config: Optional[Dict[str, object]] = None,
                             training_state: Optional[Dict[str, object]] = None) -> None:
    """Atomically replace a full checkpoint, retaining the previous file on failure."""
    checkpoint = {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": float(best_loss),
            "train_config": train_config or {},
        }
    if training_state is not None:
        checkpoint["training_state"] = training_state
    _atomic_save(checkpoint, path)


def _atomic_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def capture_rng_state() -> dict:
    """Save global Python, NumPy, CPU torch and initialized CUDA RNG states."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore RNGs after reconstructing the model and prior caches."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), index)


def load_training_checkpoint(path) -> dict:
    """Read on CPU and reject inference-only/slim checkpoints before training."""
    checkpoint = _torch_load(path, "cpu")
    required = ("step", "model_config", "model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "train_config", "best_loss")
    missing = [key for key in required if key not in checkpoint]
    if missing or not checkpoint.get("train_config"):
        raise ValueError(f"{path} is not a resumable training checkpoint; missing {missing or ['train_config']}")
    return checkpoint


def copy_best_checkpoint(resume_path, output_dir, best_loss, model_config) -> bool:
    """Carry an available prior best into a different output directory."""
    resume_path, output_dir = Path(resume_path), Path(output_dir)
    source = resume_path if resume_path.name == "best.pt" else resume_path.parent / "best.pt"
    if not source.exists():
        return False
    checkpoint = load_training_checkpoint(source)
    if checkpoint["best_loss"] != best_loss or checkpoint["model_config"] != model_config:
        return False
    _atomic_save(checkpoint, output_dir / "best.pt")
    return True


def export_slim_checkpoint(src, dst, model_config: Optional[Dict[str, object]] = None,
                           extra: Optional[Dict[str, object]] = None) -> Path:
    """Strip optimizer state from a training checkpoint and attach the model config."""
    ckpt = _torch_load(src, "cpu")
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "model_config": model_config or ckpt.get("model_config"),
        "step": int(ckpt.get("step", -1)),
        "best_loss": float(ckpt.get("best_loss", float("nan"))),
    }
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
    config = dict(model_config or ckpt.get("model_config") or {})
    if not config:
        raise ValueError(f"{path} stores no model_config; pass model_config=... explicitly")
    model = GenTFM(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
