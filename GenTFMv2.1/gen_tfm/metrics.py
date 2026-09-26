"""Evaluation metrics for mixed-type synthetic tables.

All functions take *encoded* matrices (see :mod:`gen_tfm.encoding`).  The main
entry point is :func:`evaluate_mixed`, which returns one flat dict of

* fidelity  : ``encoded_mmd`` (primary), sliced Wasserstein, per-column
              Wasserstein on continuous columns, categorical JS divergence,
              categorical-conditional mean differences, coverage / density
* privacy   : nearest-neighbour distances between generated rows and the
              context rows (``dcr_ctx_*``, ``ctx_copy_rate``) vs. held-out rows
              (``dcr_test_*``)

plus :func:`downstream_accuracy` for a train-on-synthetic / test-on-real
(TSTR) proxy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance

from .encoding import cat_cardinalities, decode_components, mixed_feature_mask, sanitize_mixed_encoded


# ----------------------------------------------------------------- distances


def median_gamma(x: np.ndarray, max_points: int = 500) -> float:
    if len(x) > max_points:
        x = x[:max_points]
    pairwise = pdist(x, "sqeuclidean")
    med = float(np.median(pairwise)) if len(pairwise) else 1.0
    return 1.0 / max(med, 1e-6)


def compute_mmd(X_real: np.ndarray, X_gen: np.ndarray, gamma: Optional[float] = None) -> float:
    """Squared MMD with an RBF kernel (median heuristic bandwidth). Lower is better."""
    if gamma is None:
        gamma = median_gamma(X_real)
    XX = cdist(X_real, X_real, "sqeuclidean")
    YY = cdist(X_gen, X_gen, "sqeuclidean")
    XY = cdist(X_real, X_gen, "sqeuclidean")
    val = np.exp(-gamma * XX).mean() + np.exp(-gamma * YY).mean()
    val -= 2.0 * np.exp(-gamma * XY).mean()
    return float(max(val, 0.0))


def compute_sliced_wasserstein(X_real: np.ndarray, X_gen: np.ndarray, n_projections: int = 64, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    d = X_real.shape[1]
    dirs = rng.normal(size=(n_projections, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return float(np.mean([wasserstein_distance(X_real @ u, X_gen @ u) for u in dirs]))


def compute_coverage_density(X_real: np.ndarray, X_gen: np.ndarray, k: int = 5) -> Dict[str, float]:
    from sklearn.neighbors import NearestNeighbors

    k_eff = max(2, min(k + 1, len(X_real)))
    nn_real = NearestNeighbors(n_neighbors=k_eff).fit(X_real)
    dists_real, _ = nn_real.kneighbors(X_real)
    radii = dists_real[:, -1]

    nn_gen = NearestNeighbors(n_neighbors=1).fit(X_gen)
    dists_gen2real, _ = nn_gen.kneighbors(X_real)
    coverage = float(np.mean(dists_gen2real[:, 0] <= radii))

    nn_gen_radius = NearestNeighbors(radius=1.0).fit(X_gen)
    counts = [len(nn_gen_radius.radius_neighbors([x], radius=float(r), return_distance=False)[0]) for x, r in zip(X_real, radii)]
    density = float(np.mean(counts)) / max(k, 1)
    return {"coverage": coverage, "density": density}


def compute_privacy_metrics(X_context_s: np.ndarray, X_test_s: np.ndarray, X_gen_s: np.ndarray, copy_threshold: float = 1e-3) -> Dict[str, float]:
    """Distance-to-closest-record (DCR) style sanity checks.

    ``dcr_ctx_*`` measure how close generated rows are to the rows the model
    actually saw (the context); ``dcr_test_*`` to held-out real rows.  If
    generated rows are systematically closer to the context than to held-out
    rows, the model is copying.
    """
    from sklearn.neighbors import NearestNeighbors

    k_ctx = min(2, max(1, len(X_context_s)))
    nn_ctx = NearestNeighbors(n_neighbors=k_ctx).fit(X_context_s)
    d_ctx_all, _ = nn_ctx.kneighbors(X_gen_s)
    nn_test = NearestNeighbors(n_neighbors=1).fit(X_test_s)
    d_test, _ = nn_test.kneighbors(X_gen_s)
    d_ctx = d_ctx_all[:, 0]
    d_test = d_test[:, 0]
    if k_ctx > 1:
        d_ctx_second = d_ctx_all[:, 1]
        nndr_ctx = d_ctx / (d_ctx_second + 1e-8)
    else:
        d_ctx_second = np.full_like(d_ctx, np.nan)
        nndr_ctx = np.full_like(d_ctx, np.nan)
    d_real = np.minimum(d_ctx, d_test)
    return {
        "nn_ctx_mean": float(np.mean(d_ctx)),
        "nn_ctx_min": float(np.min(d_ctx)),
        "nn_test_mean": float(np.mean(d_test)),
        "nn_test_min": float(np.min(d_test)),
        "ctx_copy_rate": float(np.mean(d_ctx <= copy_threshold)),
        "ctx_closer_than_test_rate": float(np.mean(d_ctx < d_test)),
        "dcr_ctx_mean": float(np.mean(d_ctx)),
        "dcr_ctx_min": float(np.min(d_ctx)),
        "dcr_ctx_q01": float(np.quantile(d_ctx, 0.01)),
        "dcr_ctx_q05": float(np.quantile(d_ctx, 0.05)),
        "dcr_test_mean": float(np.mean(d_test)),
        "dcr_test_min": float(np.min(d_test)),
        "dcr_real_mean": float(np.mean(d_real)),
        "dcr_real_min": float(np.min(d_real)),
        "nndr_ctx_mean": float(np.nanmean(nndr_ctx)),
        "nndr_ctx_min": float(np.nanmin(nndr_ctx)),
        "nn_ctx_second_mean": float(np.nanmean(d_ctx_second)),
    }


# ------------------------------------------------------------ mixed tables


def _active_encoded_view(x: np.ndarray, metadata, max_cont: int, max_cat: int, cat_cardinality: int) -> np.ndarray:
    mask = mixed_feature_mask(metadata, max_cont, max_cat, cat_cardinality)
    return np.asarray(x, dtype=np.float32)[:, mask]


def _standardize_active(X_context: np.ndarray, arrays: List[np.ndarray], metadata, max_cont: int, max_cat: int, cat_cardinality: int) -> List[np.ndarray]:
    Xc = _active_encoded_view(X_context, metadata, max_cont, max_cat, cat_cardinality)
    mu = Xc.mean(axis=0, keepdims=True)
    std = np.maximum(Xc.std(axis=0, keepdims=True), 0.1)
    return [(_active_encoded_view(arr, metadata, max_cont, max_cat, cat_cardinality) - mu) / std for arr in arrays]


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64) + 1e-8
    q = q.astype(np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def evaluate_mixed(X_context: np.ndarray, X_generated: np.ndarray, X_test: np.ndarray, metadata: Dict[str, object],
                   max_cont: int, max_cat: int, cat_cardinality: int, seed: int = 0) -> Dict[str, float]:
    """Compare generated rows with held-out real rows (``X_test``); context is used for scaling and DCR."""
    Xg = sanitize_mixed_encoded(X_generated, metadata, max_cont, max_cat, cat_cardinality)
    Xc = sanitize_mixed_encoded(X_context, metadata, max_cont, max_cat, cat_cardinality)
    Xt = sanitize_mixed_encoded(X_test, metadata, max_cont, max_cat, cat_cardinality)
    Xc_s, Xg_s, Xt_s = _standardize_active(Xc, [Xc, Xg, Xt], metadata, max_cont, max_cat, cat_cardinality)
    metrics = {
        "encoded_mmd": compute_mmd(Xt_s, Xg_s),
        "encoded_sliced_wasserstein": compute_sliced_wasserstein(Xt_s, Xg_s, seed=seed),
    }
    metrics.update(compute_coverage_density(Xt_s, Xg_s))
    metrics.update(compute_privacy_metrics(Xc_s, Xt_s, Xg_s))

    comp_t = decode_components(Xt, metadata, max_cont, max_cat, cat_cardinality)
    comp_g = decode_components(Xg, metadata, max_cont, max_cat, cat_cardinality)
    n_cont = int(metadata["n_cont"])
    n_cat = int(metadata["n_cat"])
    cards = cat_cardinalities(metadata, max_cat, cat_cardinality)

    cat_js = []
    for j in range(n_cat):
        card = int(cards[j])
        pt = np.bincount(comp_t["cats"][:, j], minlength=card)[:card]
        pg = np.bincount(comp_g["cats"][:, j], minlength=card)[:card]
        cat_js.append(js_divergence(pt, pg))
    metrics["cat_js_mean"] = float(np.mean(cat_js)) if cat_js else 0.0

    miss_t = comp_t["obs"].mean(axis=0)
    miss_g = comp_g["obs"].mean(axis=0)
    metrics["missing_rate_l1"] = float(np.mean(np.abs(miss_t - miss_g)))

    w_vals = []
    for j in range(n_cont):
        rt = comp_t["cont"][comp_t["obs"][:, j] > 0.5, j]
        rg = comp_g["cont"][comp_g["obs"][:, j] > 0.5, j]
        if len(rt) < 4 or len(rg) < 4:
            continue
        w_vals.append(wasserstein_distance(rt, rg))
    metrics["cont_observed_wasserstein"] = float(np.mean(w_vals)) if w_vals else 0.0

    assoc = []
    for c in range(n_cat):
        for level in range(cat_cardinality):
            mt = comp_t["cats"][:, c] == level
            mg = comp_g["cats"][:, c] == level
            if mt.sum() < 4 or mg.sum() < 4:
                continue
            mean_t = comp_t["cont"][mt].mean(axis=0)
            mean_g = comp_g["cont"][mg].mean(axis=0)
            scale = comp_t["cont"].std(axis=0) + 1e-6
            assoc.append(np.mean(np.abs((mean_t - mean_g) / scale)))
    metrics["cat_cont_mean_diff"] = float(np.mean(assoc)) if assoc else 0.0
    return metrics


def feature_matrix_without_label(encoded: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> Tuple[np.ndarray, np.ndarray]:
    x = sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality)
    comp = decode_components(x, metadata, max_cont, max_cat, cat_cardinality)
    label_idx = int(metadata.get("label_cat_index", 0))
    y = comp["cats"][:, label_idx].astype(np.int64)
    mask = mixed_feature_mask(metadata, max_cont, max_cat, cat_cardinality)
    start = max_cont + label_idx * cat_cardinality
    mask[start : start + cat_cardinality] = False
    return x[:, mask].astype(np.float32), y


def downstream_accuracy(X_train: np.ndarray, X_test: np.ndarray, metadata: Dict[str, object], max_cont: int, max_cat: int, cat_cardinality: int) -> float:
    """TSTR proxy: logistic regression trained on ``X_train`` (generated), tested on real rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = feature_matrix_without_label(X_train, metadata, max_cont, max_cat, cat_cardinality)
    Xte, yte = feature_matrix_without_label(X_test, metadata, max_cont, max_cat, cat_cardinality)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    try:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
        clf.fit(Xtr, ytr)
        return float(accuracy_score(yte, clf.predict(Xte)))
    except Exception:
        return float("nan")
