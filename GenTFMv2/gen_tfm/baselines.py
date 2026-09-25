"""Simple context-only baselines (no training, no neural network).

They are deliberately weak: their job is to show that Gen-TFM does more than
resampling or fitting independent marginals from the K context rows.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .encoding import cat_cardinalities, decode_components, encoded_dim, sanitize_mixed_encoded, slices


def mixed_bootstrap_baseline(X_context, metadata, max_cont, max_cat, cat_cardinality, num_gen, rng: np.random.Generator) -> np.ndarray:
    """Resample context rows with replacement."""
    idx = rng.integers(0, len(X_context), size=num_gen)
    return sanitize_mixed_encoded(X_context[idx], metadata, max_cont, max_cat, cat_cardinality)


def mixed_independent_baseline(X_context, metadata, max_cont, max_cat, cat_cardinality, num_gen, rng: np.random.Generator) -> np.ndarray:
    """Independent marginals: Gaussian per continuous column, empirical frequencies per categorical column."""
    comp = decode_components(X_context, metadata, max_cont, max_cat, cat_cardinality)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    out = np.zeros((num_gen, D), dtype=np.float32)

    obs_rate = np.clip(comp["obs"].mean(axis=0), 0.02, 0.98)
    obs = (rng.random((num_gen, n_cont)) < obs_rate).astype(np.float32)
    out[:, sl["mask_start"] : sl["mask_start"] + n_cont] = obs
    for j in range(n_cont):
        vals = comp["cont"][comp["obs"][:, j] > 0.5, j]
        if len(vals) < 2:
            vals = comp["cont"][:, j]
        out[:, j] = rng.normal(float(np.mean(vals)), float(np.std(vals) + 1e-4), size=num_gen) * obs[:, j]

    for j in range(n_cat):
        card = int(cards[j])
        counts = np.bincount(comp["cats"][:, j], minlength=card).astype(np.float64)[:card] + 0.5
        cats = rng.choice(card, size=num_gen, p=counts / counts.sum())
        start = sl["cat_start"] + j * cat_cardinality
        out[np.arange(num_gen), start + cats] = 1.0
    return out


def mixed_conditional_baseline(X_context, metadata, max_cont, max_cat, cat_cardinality, num_gen, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the categorical part of a context row, then draw continuous values Gaussian-conditioned on the first category."""
    comp = decode_components(X_context, metadata, max_cont, max_cat, cat_cardinality)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    D = encoded_dim(max_cont, max_cat, cat_cardinality)
    sl = slices(max_cont, max_cat, cat_cardinality)
    out = np.zeros((num_gen, D), dtype=np.float32)

    row_idx = rng.integers(0, len(X_context), size=num_gen)
    sampled_cats = comp["cats"][row_idx]
    for j in range(n_cat):
        start = sl["cat_start"] + j * cat_cardinality
        out[np.arange(num_gen), start + sampled_cats[:, j]] = 1.0

    first_cat = sampled_cats[:, 0]
    global_obs_rate = np.clip(comp["obs"].mean(axis=0), 0.02, 0.98)
    obs = np.zeros((num_gen, n_cont), dtype=np.float32)
    cont = np.zeros((num_gen, n_cont), dtype=np.float32)
    for level in range(cat_cardinality):
        gen_mask = first_cat == level
        ctx_mask = comp["cats"][:, 0] == level
        if not gen_mask.any():
            continue
        obs_rate = global_obs_rate
        if ctx_mask.sum() >= 4:
            obs_rate = np.clip(comp["obs"][ctx_mask].mean(axis=0), 0.02, 0.98)
        obs[gen_mask] = (rng.random((gen_mask.sum(), n_cont)) < obs_rate).astype(np.float32)
        for j in range(n_cont):
            vals = comp["cont"][ctx_mask & (comp["obs"][:, j] > 0.5), j]
            if len(vals) < 3:
                vals = comp["cont"][comp["obs"][:, j] > 0.5, j]
            if len(vals) < 2:
                vals = comp["cont"][:, j]
            cont[gen_mask, j] = rng.normal(float(np.mean(vals)), float(np.std(vals) + 1e-4), gen_mask.sum())
    out[:, :n_cont] = cont * obs
    out[:, sl["mask_start"] : sl["mask_start"] + n_cont] = obs
    return out


MIXED_BASELINES: Dict[str, object] = {
    "mixed_bootstrap": mixed_bootstrap_baseline,
    "mixed_independent": mixed_independent_baseline,
    "mixed_conditional": mixed_conditional_baseline,
}
