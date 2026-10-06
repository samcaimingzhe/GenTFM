"""Real tables in, encoded tables out (and back).

:func:`encode_dataframe` is the entry point for your own data: it picks
continuous / categorical columns from a pandas DataFrame, standardises the
continuous ones, maps categories to integer codes (keeping the most frequent
``cat_cardinality - 1`` levels and an "other" bucket) and writes everything into
the padded encoding.  The returned ``metadata`` stores enough to invert the
transformation with :func:`decode_to_dataframe`.

The ``load_*`` functions build the benchmark suites used in the project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import encode_category, Schema, decode_components, encoded_dim, mixed_feature_mask, sanitize_mixed_encoded

Table = Tuple[str, np.ndarray, Dict[str, object]]  # (name, encoded rows, metadata)

MISSING_TOKENS = {"", "?", "nan", "None", "NaN", "<NA>"}


# ------------------------------------------------------------- primitives


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    qs = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
    return np.searchsorted(qs, x).astype(np.int64)


def _fill_non_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    col_medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0).astype(np.float32)
    bad = ~np.isfinite(x)
    x = x.copy()
    x[bad] = np.take(col_medians, np.where(bad)[1])
    return x


def _to_object_values(values) -> np.ndarray:
    out = []
    for value in np.asarray(values, dtype=object):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="ignore")
        if value is None:
            out.append("missing")
            continue
        text = str(value)
        out.append("missing" if text in MISSING_TOKENS else text)
    return np.asarray(out, dtype=object)


def safe_name(name: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", text).strip("_") or "dataset"


def build_encoded_table(name: str, cont: np.ndarray, cats: Sequence[np.ndarray], cat_names: Sequence[str],
                        schema: Schema, cont_names: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode raw continuous columns + raw categorical columns (any dtype) into the padded format.

    Continuous columns are standardised (mean/std stored in metadata).  For each
    categorical column the most frequent levels are kept and the rest mapped to
    a final "other" level so that the cardinality never exceeds
    ``schema.cat_cardinality``.  Missingness is not modelled: every value counts
    as observed.
    """
    max_cont, max_cat, cat_cardinality = schema.as_tuple()
    cont = _fill_non_finite(np.asarray(cont, dtype=np.float32).reshape(len(cont), -1))
    n_rows, n_cont = cont.shape
    cont_names = list(cont_names) if cont_names is not None else [f"x{j}" for j in range(n_cont)]
    if n_cont > max_cont:
        cont = cont[:, :max_cont]
        cont_names = cont_names[:max_cont]
        n_cont = max_cont
    cont_mean = cont.mean(axis=0, keepdims=True)
    cont_std = cont.std(axis=0, keepdims=True) + 1e-6
    cont = (cont - cont_mean) / cont_std

    cats = list(cats)
    cat_names = list(cat_names)
    if len(cats) > max_cat:
        cats = cats[:max_cat]
        cat_names = cat_names[:max_cat]
    n_cat = len(cats)

    cards, clean_cats, levels = [], [], []
    for values in cats:
        values = _to_object_values(values)
        uniq, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        uniq = uniq[order]
        card = max(2, min(len(uniq), cat_cardinality))
        if len(uniq) <= cat_cardinality:
            kept = list(uniq[:card])
            other_idx = card - 1
        else:
            kept = list(uniq[: cat_cardinality - 1])
            other_idx = cat_cardinality - 1
            card = cat_cardinality
        mapping = {v: i for i, v in enumerate(kept)}
        mapped = np.asarray([mapping.get(v, other_idx) for v in values], dtype=np.int64)
        cards.append(card)
        clean_cats.append(np.clip(mapped, 0, card - 1))
        level_names = [str(v) for v in kept]
        if len(uniq) > cat_cardinality:
            level_names.append("other")
        levels.append(level_names)

    D = schema.encoded_dim
    encoded = np.zeros((n_rows, D), dtype=np.float32)
    encoded[:, :n_cont] = cont
    encoded[:, schema.mask_start : schema.mask_start + n_cont] = 1.0
    for j, values in enumerate(clean_cats):
        encoded[:, schema.category_slice(j)] = encode_category(values, schema)

    metadata: Dict[str, object] = {
        "dataset_name": name,
        "prior_type": f"real_{name}",
        "n_cont": int(n_cont),
        "n_cat": int(n_cat),
        "cat_cardinality": int(cat_cardinality),
        "cat_cardinalities": [int(c) for c in cards],
        "cat_names": cat_names,
        "cat_levels": levels,
        "cont_names": cont_names,
        "cont_mean": cont_mean.ravel().tolist(),
        "cont_std": cont_std.ravel().tolist(),
        "label_cat_index": 0,
        "n_rows": int(n_rows),
        "force_observed_mask": True,
        "no_missingness": True,
    }
    metadata.update(schema.metadata_fields())
    metadata["n_features"] = int(mixed_feature_mask(metadata, *schema.as_tuple(), **schema.codec_kwargs()).sum())
    return sanitize_mixed_encoded(encoded, metadata, *schema.as_tuple(), **schema.codec_kwargs()), metadata


# ---------------------------------------------------------- pandas bridge


def select_columns(df, schema: Schema, target_col: Optional[str] = None, numeric_cat_threshold: int = 20,
                   target_task: str = "classification",
                   max_cat_unique: int = 1000) -> Tuple[np.ndarray, List[str], List[np.ndarray], List[str]]:
    """Heuristic column typing: numeric columns with many unique values are continuous, the rest categorical.

    Returns ``(cont_matrix, cont_names, cat_columns, cat_names)``.  If
    ``target_col`` is given it becomes the supervised field: categorical by
    default, or the first continuous field when ``target_task='regression'``.
    """
    import pandas as pd

    numeric_candidates, cat_candidates = [], []
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1:
            continue
        if pd.api.types.is_numeric_dtype(series) and nunique > numeric_cat_threshold:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
            if not np.isfinite(values).any() or float(np.nanvar(values)) <= 0:
                continue
            numeric_candidates.append((str(col), values, float(np.nanvar(values))))
        elif nunique <= max_cat_unique:
            cat_candidates.append((str(col), _to_object_values(series), nunique))

    numeric_candidates.sort(key=lambda item: (-item[2], item[0]))
    cat_candidates.sort(key=lambda item: (-min(item[2], schema.cat_cardinality), item[0]))
    target_task = str(target_task).lower()
    if target_task not in {"classification", "regression"}:
        raise ValueError("target_task must be 'classification' or 'regression'")
    numeric_target = target_col is not None and target_task == "regression"
    n_extra_cat = schema.max_cat - (1 if target_col is not None and not numeric_target else 0)
    selected_num = numeric_candidates[: max(0, schema.max_cont - int(numeric_target))]
    selected_cat = cat_candidates[: max(0, n_extra_cat)]

    cont = np.column_stack([v for _, v, _ in selected_num]).astype(np.float32) if selected_num else np.zeros((len(df), 0), dtype=np.float32)
    cont_names = [n for n, _, _ in selected_num]
    cats = [v for _, v, _ in selected_cat]
    cat_names = [n for n, _, _ in selected_cat]

    if target_col is not None:
        y = df[target_col]
        numeric = pd.to_numeric(y, errors="coerce")
        numeric_ok = bool(np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan)).mean() > 0.95)
        if numeric_target:
            if not numeric_ok:
                raise ValueError(f"Regression target {target_col!r} must be numeric")
            y_values = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
            cont = np.column_stack([y_values, cont]).astype(np.float32)
            cont_names = [str(target_col), *cont_names]
        else:
            if numeric_ok and int(y.nunique(dropna=True)) > schema.cat_cardinality:
                y_values = quantile_bins(numeric.to_numpy(dtype=np.float64), max(2, min(4, schema.cat_cardinality)))
                label_name = f"{target_col}_bin"
            else:
                y_values = _to_object_values(y)
                label_name = str(target_col)
            cats = [y_values, *cats]
            cat_names = [label_name, *cat_names]
    return cont, cont_names, cats, cat_names


def encode_dataframe(df, schema: Schema = Schema(), target_col: Optional[str] = None, name: str = "table",
                     numeric_cat_threshold: int = 20, target_task: str = "classification") -> Tuple[np.ndarray, Dict[str, object]]:
    """Encode a pandas DataFrame.  See :func:`select_columns` for the typing rules."""
    cont, cont_names, cats, cat_names = select_columns(df, schema, target_col=target_col, numeric_cat_threshold=numeric_cat_threshold,
                                                        target_task=target_task)
    table, metadata = build_encoded_table(name, cont, cats, cat_names, schema, cont_names=cont_names)
    metadata["data_source"] = "dataframe"
    metadata["target_col"] = target_col
    metadata["target_task"] = target_task
    if target_col is not None and target_task == "regression":
        metadata["label_type"] = "continuous"
        metadata["label_cont_index"] = 0
        metadata["label_cat_index"] = None
    elif target_col is not None:
        metadata["label_type"] = "categorical"
        metadata["label_cat_index"] = 0
        metadata["label_cont_index"] = None
    return table, metadata


def decode_to_dataframe(encoded: np.ndarray, metadata: Dict[str, object], schema: Schema = Schema()):
    """Turn encoded rows (real or generated) back into a DataFrame with original column names and category labels."""
    import pandas as pd

    comp = decode_components(encoded, metadata, *schema.as_tuple(), **schema.codec_kwargs())
    n_cont, n_cat = int(metadata["n_cont"]), int(metadata["n_cat"])
    data = {}
    cont_names = metadata.get("cont_names") or [f"x{j}" for j in range(n_cont)]
    mean = np.asarray(metadata.get("cont_mean", np.zeros(n_cont)), dtype=np.float64)
    std = np.asarray(metadata.get("cont_std", np.ones(n_cont)), dtype=np.float64)
    for j in range(n_cont):
        data[cont_names[j]] = comp["cont"][:, j] * std[j] + mean[j]
    cat_names = metadata.get("cat_names") or [f"cat{j}" for j in range(n_cat)]
    levels = metadata.get("cat_levels")
    for j in range(n_cat):
        codes = comp["cats"][:, j]
        if levels is not None and j < len(levels):
            names = list(levels[j])
            data[cat_names[j]] = [names[c] if c < len(names) else f"level_{c}" for c in codes]
        else:
            data[cat_names[j]] = codes
    return pd.DataFrame(data)


# ------------------------------------------------------------ benchmarks


def load_sklearn_tables(schema: Schema, datasets: Sequence[str] = ("iris", "wine", "breast_cancer", "diabetes")) -> List[Table]:
    """Four small bundled datasets, usable offline.  Extra categorical columns are quantile bins."""
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

    tables: List[Table] = []
    if "iris" in datasets:
        data = load_iris()
        cont = data.data.astype(np.float32)
        cats = [data.target, quantile_bins(data.data[:, 0], 4)]
        tables.append(("iris", *build_encoded_table("iris", cont, cats, ["species", "sepal_len_bin"], schema, cont_names=list(data.feature_names))))
    if "wine" in datasets:
        data = load_wine()
        cont = data.data[:, :10].astype(np.float32)
        cats = [data.target, quantile_bins(data.data[:, 9], 4), quantile_bins(data.data[:, 12], 4)]
        tables.append(("wine", *build_encoded_table("wine", cont, cats, ["class", "color_bin", "proline_bin"], schema, cont_names=list(data.feature_names[:10]))))
    if "breast_cancer" in datasets:
        data = load_breast_cancer()
        cont = data.data[:, :10].astype(np.float32)
        cats = [data.target, quantile_bins(data.data[:, 0], 4), quantile_bins(data.data[:, 3], 4)]
        tables.append(("breast_cancer", *build_encoded_table("breast_cancer", cont, cats, ["diagnosis", "radius_bin", "area_bin"], schema, cont_names=list(data.feature_names[:10]))))
    if "diabetes" in datasets:
        data = load_diabetes()
        cont = data.data[:, :10].astype(np.float32)
        sex = (data.data[:, 1] > np.median(data.data[:, 1])).astype(np.int64)
        cats = [quantile_bins(data.target, 4), sex]
        tables.append(("diabetes", *build_encoded_table("diabetes", cont, cats, ["target_bin", "sex_bin"], schema, cont_names=list(data.feature_names[:10]))))
    for _, _, meta in tables:
        meta["data_source"] = "sklearn"
    return tables


def load_tabsyn_tables(schema: Schema, root, datasets: Sequence[str]) -> List[Table]:
    """Datasets preprocessed in the TabSyn format (``X_num_*.npy``, ``X_cat_*.npy``, ``y_*.npy``, ``info.json``)."""
    root = Path(root)
    tables: List[Table] = []
    for name in datasets:
        d = root / name
        required = [d / "X_num_train.npy", d / "X_num_test.npy", d / "y_train.npy", d / "y_test.npy"]
        if not all(p.exists() for p in required):
            print(f"[tabsyn] skipping {name}: missing npy files in {d}")
            continue
        X_num = np.concatenate([np.load(d / "X_num_train.npy"), np.load(d / "X_num_test.npy")], axis=0).astype(np.float32)
        y = np.concatenate([np.load(d / "y_train.npy"), np.load(d / "y_test.npy")], axis=0)
        info = json.loads((d / "info.json").read_text()) if (d / "info.json").exists() else {}
        task_type = str(info.get("task_type", "unknown"))
        regression_task = task_type == "regression" or np.issubdtype(y.dtype, np.floating)
        if regression_task:
            X_num = np.column_stack([y.astype(np.float32), X_num]).astype(np.float32)
            cats, cat_names = [], []
        else:
            cats, cat_names = [y.astype(np.int64)], ["target"]
        if (d / "X_cat_train.npy").exists() and (d / "X_cat_test.npy").exists():
            X_cat = np.concatenate([np.atleast_2d(np.load(d / "X_cat_train.npy", allow_pickle=False)).reshape(len(np.load(d / "y_train.npy")), -1),
                                    np.atleast_2d(np.load(d / "X_cat_test.npy", allow_pickle=False)).reshape(len(np.load(d / "y_test.npy")), -1)], axis=0)
            for j in range(X_cat.shape[1]):
                if len(cats) >= schema.max_cat:
                    break
                values = X_cat[:, j].astype(np.int64)
                if np.unique(values).size <= 1:
                    continue
                cats.append(values)
                cat_names.append(f"cat_{j}")
        dataset_name = f"tabsyn_{name}"
        table, metadata = build_encoded_table(dataset_name, X_num, cats, cat_names, schema)
        metadata.update({"data_source": "tabsyn_npy", "source_dir": str(d), "task_type": task_type})
        if regression_task:
            metadata.update({"target_task": "regression", "label_type": "continuous", "label_cont_index": 0, "label_cat_index": None})
        else:
            metadata.update({"target_task": "classification", "label_type": "categorical", "label_cat_index": 0, "label_cont_index": None})
        tables.append((dataset_name, table, metadata))
    return tables


OPENML_SPECS = {
    # name: (OpenML data id, target column, preferred categorical columns in order)
    "adult": (1590, "class", ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]),
}


def load_openml_tables(schema: Schema, datasets: Sequence[str] = ("adult",), cache_dir: Optional[str] = None,
                       retries: int = 3, retry_delay: float = 30.0) -> List[Table]:
    """Raw OpenML datasets (Adult by default) through ``sklearn.datasets.fetch_openml``."""
    import time

    import pandas as pd
    from sklearn.datasets import fetch_openml

    tables: List[Table] = []
    for name in datasets:
        if name not in OPENML_SPECS:
            print(f"[openml] skipping {name}: no spec; add it to OPENML_SPECS")
            continue
        data_id, target_col, preferred = OPENML_SPECS[name]
        bunch, last_error = None, ""
        for attempt in range(1, max(1, retries) + 1):
            try:
                bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=cache_dir)
                break
            except Exception as exc:  # OpenML is flaky (HTTP 504); retry a few times
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(retry_delay)
        if bunch is None:
            print(f"[openml] skipping {name}: fetch failed ({last_error})")
            continue
        df = bunch.frame
        numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        raw_cat_cols = [c for c in df.columns if c != target_col and c not in numeric_cols]
        cat_cols = [c for c in preferred if c in raw_cat_cols] + [c for c in raw_cat_cols if c not in preferred]
        cont = df[numeric_cols].to_numpy(dtype=np.float32)
        selected = cat_cols[: max(0, schema.max_cat - 1)]
        cats = [_to_object_values(df[target_col])] + [_to_object_values(df[c]) for c in selected]
        cat_names = ["target"] + selected
        dataset_name = f"raw_openml_{name}"
        table, metadata = build_encoded_table(dataset_name, cont, cats, cat_names, schema, cont_names=numeric_cols)
        metadata.update({"data_source": "openml", "openml_id": int(data_id), "task_type": "classification"})
        tables.append((dataset_name, table, metadata))
    return tables


HF_TABARENA_PILOT_DATASETS = [
    "blood_transfusion", "credit_g", "maternal_health_risk", "qsar_fish_toxicity", "website_phishing", "mic",
    "marketing_campaign", "splice", "predict_students_dropout_and_academic_success", "churn", "airfoil_self_noise",
    "bank_marketing",
]

HF_TABARENA_BROAD_DATASETS = HF_TABARENA_PILOT_DATASETS + [
    "heart_disease_cleveland", "ecommerce_shipping", "online_shoppers", "obesity_estimation", "wine_quality",
    "concrete_compressive_strength", "forest_fires", "student_portuguese_performance", "give_me_some_credit", "heloc",
    "jm1", "polish_companies_bankruptcy", "diabetes_130_us", "customer_satisfaction_in_airline",
    "early_stage_diabetes_risk", "cardiotocography", "credit_card_clients_default", "bank_customer_churn",
    "hotel_booking_demand", "in_vehicle_coupon_recommendation", "south_africa_coronary_heart_disease",
    "seismic_bumps", "wine_world_cost",
]


def _find_hf_dataset_prefix(files: List[str], dataset_name: str) -> str:
    candidates = [f.rsplit("/", 1)[0] for f in files if f.startswith(f"{dataset_name}/") and f.endswith("/dataset.parquet")]
    if not candidates:
        raise FileNotFoundError(f"No dataset.parquet found for {dataset_name}")
    candidates.sort(key=lambda path: ("/versions/" in path, path))
    return candidates[0]


def load_hf_tabarena_tables(schema: Schema, datasets: Sequence[str] = HF_TABARENA_PILOT_DATASETS,
                            repo_id: str = "TabArena/BeyondArena", cache_dir: Optional[str] = None,
                            min_rows: int = 128, max_rows: int = 250000, max_raw_features: int = 2000) -> List[Table]:
    """Datasets from the Hugging Face ``TabArena/BeyondArena`` parquet collection."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    tables: List[Table] = []
    for alias in datasets:
        try:
            prefix = _find_hf_dataset_prefix(files, alias)
            parquet = Path(hf_hub_download(repo_id, f"{prefix}/dataset.parquet", repo_type="dataset", cache_dir=cache_dir))
            task = Path(hf_hub_download(repo_id, f"{prefix}/task_metadata.predictive-ml-task-mold-v1.json", repo_type="dataset", cache_dir=cache_dir))
            task_meta = json.loads(task.read_text())
            df = pd.read_parquet(parquet)
        except Exception as exc:
            print(f"[hf_tabarena] skipping {alias}: {exc}")
            continue
        target_col = task_meta.get("target_column_name")
        if target_col not in df.columns:
            print(f"[hf_tabarena] skipping {alias}: target column {target_col!r} not found")
            continue
        if len(df) < min_rows or len(df) > max_rows or (df.shape[1] - 1) > max_raw_features:
            print(f"[hf_tabarena] skipping {alias}: size filter (rows={len(df)}, features={df.shape[1] - 1})")
            continue
        dataset_name = f"hf_tabarena_{safe_name(alias)}"
        table, metadata = encode_dataframe(df, schema, target_col=target_col, name=dataset_name)
        problem_type = str(task_meta.get("problem_type", "classification"))
        metadata.update({
            "data_source": "hf_tabarena_beyondarena",
            "hf_repo_id": repo_id,
            "hf_dataset_alias": alias,
            "task_type": "regression" if "regression" in problem_type else "classification",
            "problem_type": problem_type,
            "raw_n_rows": int(len(df)),
            "raw_n_features": int(df.shape[1] - 1),
        })
        tables.append((dataset_name, table, metadata))
    return tables
