"""Fixed-width mixed-type row encoding shared by the prior, the model and the metrics.

Every row of every table (synthetic or real) is encoded as one float vector

    [ continuous values | categorical one-hot blocks | observed-mask bits ]
      max_cont             max_cat * cat_cardinality   max_cont

A table with fewer columns than the maximum is zero padded, and a boolean
``feature_mask`` tells the model which dimensions are alive.  ``metadata`` is a
plain dict that describes a concrete table (``n_cont``, ``n_cat``,
``cat_cardinalities`` ...).  The helpers below never look at column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass(frozen=True)
class Schema:
    """Padded encoding limits.  The frozen checkpoint uses the defaults below."""

    max_cont: int = 32
    max_cat: int = 8
    cat_cardinality: int = 12

    @property
    def encoded_dim(self) -> int:
        return encoded_dim(self.max_cont, self.max_cat, self.cat_cardinality)

    @property
    def cat_start(self) -> int:
        return self.max_cont

    @property
    def mask_start(self) -> int:
        return self.max_cont + self.max_cat * self.cat_cardinality

    def as_tuple(self):
        return self.max_cont, self.max_cat, self.cat_cardinality


def velocity_feature_mask(feature_mask: torch.Tensor, schema: Schema) -> torch.Tensor:
    """Select numerical/one-hot coordinates, excluding fixed observed bits.

    The original feature_mask still describes the complete encoded table.
    Fully observed encoding requires matching continuous and observed slots.
    """
    if feature_mask.ndim != 2 or feature_mask.shape[1] != schema.encoded_dim:
        raise ValueError("feature_mask must have shape (B, schema.encoded_dim)")
    if feature_mask.dtype != torch.bool:
        raise TypeError("feature_mask must be boolean")
    if not torch.equal(feature_mask[:, :schema.max_cont], feature_mask[:, schema.mask_start:]):
        raise ValueError("continuous and observed feature slots must match")
    mask = feature_mask.clone()
    mask[:, schema.mask_start:] = False
    return mask


def categorical_feature_mask(feature_mask: torch.Tensor, schema: Schema) -> torch.Tensor:
    """Return valid classes (B, max_cat, cat_cardinality), including empty fields."""
    velocity_feature_mask(feature_mask, schema)
    return feature_mask[:, schema.cat_start:schema.mask_start].reshape(
        feature_mask.shape[0], schema.max_cat, schema.cat_cardinality,
    )


def encoded_dim(max_cont: int, max_cat: int, cat_cardinality: int) -> int:
    return max_cont + max_cat * cat_cardinality + max_cont


def slices(max_cont: int, max_cat: int, cat_cardinality: int) -> Dict[str, int]:
    cat_start = max_cont
    mask_start = max_cont + max_cat * cat_cardinality
    return {"cat_start": cat_start, "mask_start": mask_start}


def cat_cardinalities(metadata: Dict[str, object], max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Per-field cardinalities of a table, clipped to ``[2, cat_cardinality]``."""
    n_cat = int(metadata["n_cat"])
    raw = metadata.get("cat_cardinalities")
    if raw is None:
        return np.full(n_cat, cat_cardinality, dtype=np.int64)
    cards = np.asarray(raw, dtype=np.int64)[:n_cat]
    if len(cards) < n_cat:
        cards = np.pad(cards, (0, n_cat - len(cards)), constant_values=cat_cardinality)
    return np.clip(cards, 2, cat_cardinality)


def mixed_feature_mask(metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Boolean mask over the padded encoding: which dimensions this table uses."""
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    mask = np.zeros(D, dtype=bool)
    mask[:n_cont] = True
    for j in range(n_cat):
        start = sl["cat_start"] + j * cat_cardinality
        mask[start : start + int(cards[j])] = True
    mask[sl["mask_start"] : sl["mask_start"] + n_cont] = True
    return mask


def batch_feature_mask(metadata: List[Dict[str, object]], max_cont: int, max_cat: int, cat_cardinality: int, device) -> torch.Tensor:
    masks = [mixed_feature_mask(item, max_cont, max_cat, cat_cardinality) for item in metadata]
    return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)


def sanitize_mixed_encoded(encoded: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Project an arbitrary float matrix back onto valid mixed rows.

    Categorical blocks become exact one-hots (argmax), observed-mask bits become
    0/1, and padded dimensions are zeroed.  Under the no-missingness protocol
    (``metadata["no_missingness"]``) every continuous value is marked observed.
    """
    x = np.asarray(encoded, dtype=np.float32).copy()
    if x.ndim != 2:
        raise ValueError("encoded must be 2-D")
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    out = np.zeros((len(x), D), dtype=np.float32)

    if metadata.get("force_observed_mask") or metadata.get("no_missingness"):
        obs = np.ones((len(x), n_cont), dtype=np.float32)
    else:
        obs_scores = x[:, sl["mask_start"] : sl["mask_start"] + n_cont]
        obs = (obs_scores >= 0.5).astype(np.float32)
    cont = x[:, :n_cont] * obs
    out[:, :n_cont] = cont
    out[:, sl["mask_start"] : sl["mask_start"] + n_cont] = obs

    for j in range(n_cat):
        start = sl["cat_start"] + j * cat_cardinality
        group = x[:, start : start + int(cards[j])]
        idx = np.argmax(group, axis=1)
        out[np.arange(len(x)), start + idx] = 1.0
    return out


def decode_components(encoded: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> Dict[str, np.ndarray]:
    """Split a (sanitized) encoded matrix into continuous values, observed bits and integer categories."""
    x = sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    sl = slices(max_cont, max_cat, cat_cardinality)
    cont = x[:, :n_cont]
    obs = x[:, sl["mask_start"] : sl["mask_start"] + n_cont]
    cats = np.zeros((len(x), n_cat), dtype=np.int64)
    for j in range(n_cat):
        start = sl["cat_start"] + j * cat_cardinality
        cats[:, j] = np.argmax(x[:, start : start + cat_cardinality], axis=1)
    return {"encoded": x, "cont": cont, "obs": obs, "cats": cats}


def encode_components(cont: np.ndarray, cats: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Inverse of :func:`decode_components` for fully observed tables."""
    n_rows = len(cont)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    encoded = np.zeros((n_rows, D), dtype=np.float32)
    encoded[:, :n_cont] = np.asarray(cont, dtype=np.float32)[:, :n_cont]
    encoded[:, sl["mask_start"] : sl["mask_start"] + n_cont] = 1.0
    cats = np.asarray(cats, dtype=np.int64)
    for j in range(n_cat):
        start = sl["cat_start"] + j * cat_cardinality
        encoded[np.arange(n_rows), start + cats[:, j]] = 1.0
    return sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality)
