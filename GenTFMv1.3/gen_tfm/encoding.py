"""Fixed-width mixed-type row encoding shared by the prior, the model and the metrics.

Every row of every table (synthetic or real) is encoded as one float vector

    [ float32 IEEE-754 bits | categorical binary blocks | observed-mask bits ]
      max_cont * 32        max_cat * ceil(log2(cat_cardinality))   max_cont

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
    """v1.3 schema: 32 bits per numeric field, binary categorical blocks."""

    max_cont: int = 32
    max_cat: int = 8
    cat_cardinality: int = 12

    def __post_init__(self):
        if self.max_cont < 0 or self.max_cat < 0 or self.cat_cardinality < 2:
            raise ValueError("Invalid schema limits")

    @property
    def cat_width(self) -> int:
        return binary_width(self.cat_cardinality)

    @property
    def encoded_dim(self) -> int:
        return encoded_dim(self.max_cont, self.max_cat, self.cat_cardinality)

    @property
    def cat_start(self) -> int:
        return self.max_cont * 32

    @property
    def mask_start(self) -> int:
        return self.cat_start + self.max_cat * self.cat_width

    def as_tuple(self):
        return self.max_cont, self.max_cat, self.cat_cardinality


def encoded_dim(max_cont: int, max_cat: int, cat_cardinality: int) -> int:
    return max_cont * 32 + max_cat * binary_width(cat_cardinality) + max_cont


def slices(max_cont: int, max_cat: int, cat_cardinality: int) -> Dict[str, int]:
    cat_start = max_cont * 32
    mask_start = cat_start + max_cat * binary_width(cat_cardinality)
    return {"cat_start": cat_start, "mask_start": mask_start, "cat_width": binary_width(cat_cardinality)}


def cat_cardinalities(metadata: Dict[str, object], max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Per-field cardinalities of a table, with explicit schema validation."""
    n_cat = int(metadata["n_cat"])
    if not 0 <= n_cat <= max_cat:
        raise ValueError("Invalid n_cat")
    raw = metadata.get("cat_cardinalities")
    if raw is None:
        return np.full(n_cat, cat_cardinality, dtype=np.int64)
    values = np.asarray(raw)
    cards = np.asarray(raw, dtype=np.int64)
    if (cards.shape != (n_cat,) or np.any(values != cards)
            or np.any(cards < 1) or np.any(cards > cat_cardinality)):
        raise ValueError("Invalid cat_cardinalities")
    return cards


def mixed_feature_mask(metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Boolean mask over the padded encoding: which dimensions this table uses."""
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    if not (0 <= n_cont <= max_cont and 0 <= n_cat <= max_cat):
        raise ValueError("Active columns exceed schema")
    if (metadata.get("cat_encoding", "binary") != "binary"
            or metadata.get("cont_encoding", "float32_bits") != "float32_bits"):
        raise ValueError("v1.3 requires binary encoding")
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    mask = np.zeros(D, dtype=bool)
    mask[:n_cont * 32] = True
    for j in range(n_cat):
        start = sl["cat_start"] + j * sl["cat_width"]
        width = binary_width(int(cards[j]))
        mask[start + sl["cat_width"] - width : start + sl["cat_width"]] = True
    mask[sl["mask_start"] : sl["mask_start"] + n_cont] = True
    return mask


def batch_feature_mask(metadata: List[Dict[str, object]], max_cont: int, max_cat: int, cat_cardinality: int, device) -> torch.Tensor:
    masks = [mixed_feature_mask(item, max_cont, max_cat, cat_cardinality) for item in metadata]
    return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)


def sanitize_mixed_encoded(encoded: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Project an arbitrary float matrix back onto valid mixed rows.

    Categorical blocks become nearest legal binary codes, observed-mask bits become
    0/1, and padded dimensions are zeroed.  Under the no-missingness protocol
    (``metadata["no_missingness"]``) every continuous value is marked observed.
    """
    x = np.asarray(encoded, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] != encoded_dim(max_cont, max_cat, cat_cardinality):
        raise ValueError("encoded must be 2-D")
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    if not (0 <= n_cont <= max_cont and 0 <= n_cat <= max_cat):
        raise ValueError("Active columns exceed schema")
    if (metadata.get("cat_encoding", "binary") != "binary"
            or metadata.get("cont_encoding", "float32_bits") != "float32_bits"):
        raise ValueError("v1.3 requires binary encoding")
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    out = np.zeros((len(x), D), dtype=np.float32)

    if metadata.get("force_observed_mask") or metadata.get("no_missingness"):
        obs = np.ones((len(x), n_cont), dtype=np.float32)
    else:
        obs_scores = x[:, sl["mask_start"] : sl["mask_start"] + n_cont]
        obs = (obs_scores >= 0.5).astype(np.float32)
    bits = (x[:, :n_cont * 32] >= 0.5).astype(np.float32).reshape(len(x), n_cont, 32)
    cont = float32_bits_to_values(bits)
    cont = np.clip(np.nan_to_num(cont, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0) * obs
    out[:, :n_cont * 32] = float32_values_to_bits(cont).reshape(len(x), n_cont * 32)
    out[:, sl["mask_start"] : sl["mask_start"] + n_cont] = obs

    for j in range(n_cat):
        start = sl["cat_start"] + j * sl["cat_width"]
        group = x[:, start : start + sl["cat_width"]]
        codes = category_ids_to_bits(np.arange(int(cards[j])), sl["cat_width"])
        idx = ((group[:, None, :] - codes[None, :, :]) ** 2).sum(axis=-1).argmin(axis=-1)
        out[:, start : start + sl["cat_width"]] = codes[idx]
    return out


def decode_components(encoded: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> Dict[str, np.ndarray]:
    """Split a (sanitized) encoded matrix into continuous values, observed bits and integer categories."""
    x = sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    sl = slices(max_cont, max_cat, cat_cardinality)
    cont = float32_bits_to_values(x[:, :n_cont * 32].reshape(len(x), n_cont, 32))
    obs = x[:, sl["mask_start"] : sl["mask_start"] + n_cont]
    cats = np.zeros((len(x), n_cat), dtype=np.int64)
    for j in range(n_cat):
        start = sl["cat_start"] + j * sl["cat_width"]
        cats[:, j] = category_bits_to_ids(x[:, start : start + sl["cat_width"]])
    return {"encoded": x, "cont": cont, "obs": obs, "cats": cats}


def encode_components(cont: np.ndarray, cats: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    """Inverse of :func:`decode_components` for fully observed tables."""
    n_rows = len(cont)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    encoded = np.zeros((n_rows, D), dtype=np.float32)
    encoded[:, :n_cont * 32] = float32_values_to_bits(np.asarray(cont, dtype=np.float32)[:, :n_cont]).reshape(n_rows, n_cont * 32)
    encoded[:, sl["mask_start"] : sl["mask_start"] + n_cont] = 1.0
    cats = np.asarray(cats, dtype=np.int64)
    for j in range(n_cat):
        start = sl["cat_start"] + j * sl["cat_width"]
        if np.any((cats[:, j] < 0) | (cats[:, j] >= cat_cardinalities(metadata, max_cat, cat_cardinality)[j])):
            raise ValueError("Category ID outside field cardinality")
        encoded[:, start : start + sl["cat_width"]] = category_ids_to_bits(cats[:, j], sl["cat_width"])
    return sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality)


def binary_width(cardinality: int) -> int:
    return max(1, (int(cardinality) - 1).bit_length())


def category_ids_to_bits(ids, width: int):
    """Zero-based category IDs, most significant bit first; float32 storage."""
    if isinstance(ids, torch.Tensor):
        shifts = torch.arange(width - 1, -1, -1, device=ids.device)
        return ((ids.long()[..., None] >> shifts) & 1).to(torch.float32)
    ids = np.asarray(ids, dtype=np.int64)
    return ((ids[..., None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.float32)


def category_bits_to_ids(bits):
    width = bits.shape[-1]
    if isinstance(bits, torch.Tensor):
        weights = 2 ** torch.arange(width - 1, -1, -1, device=bits.device)
        return (bits.long() * weights).sum(dim=-1)
    return (np.asarray(bits, dtype=np.int64) * 2 ** np.arange(width - 1, -1, -1)).sum(axis=-1)


def float32_values_to_bits(values):
    """IEEE-754 float32 bit pattern, MSB first (sign/exponent/mantissa)."""
    if isinstance(values, torch.Tensor):
        raw = values.to(torch.float32).contiguous().view(torch.int32).long()
        shifts = torch.arange(31, -1, -1, device=values.device)
        return ((raw[..., None] >> shifts) & 1).to(torch.float32)
    raw = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((raw[..., None] >> np.arange(31, -1, -1, dtype=np.uint32)) & 1).astype(np.float32)


def float32_bits_to_values(bits):
    if bits.shape[-1] != 32:
        raise ValueError("float32 requires exactly 32 bits")
    if isinstance(bits, torch.Tensor):
        weights = 2 ** torch.arange(31, -1, -1, device=bits.device)
        raw = (bits.long() * weights).sum(dim=-1).to(torch.int32).contiguous()
        return raw.view(torch.float32)
    weights = np.left_shift(np.uint32(1), np.arange(31, -1, -1, dtype=np.uint32))
    raw = (np.asarray(bits, dtype=np.uint32) * weights).sum(axis=-1, dtype=np.uint32)
    return np.ascontiguousarray(raw).view(np.float32)
