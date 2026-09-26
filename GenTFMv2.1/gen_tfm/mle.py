"""Train-on-synthetic and augmentation evaluation with context-only preprocessing.

Targets are jointly generated but explicitly removed from predictor inputs.
Regression targets occupy a continuous slot and are scored in original units.
Optional predictors are imported only when requested; no tuning uses test data.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from .encoding import Schema, mixed_feature_mask, sanitize_mixed_encoded
from .real_data import _to_object_values, select_columns


class ContextTableEncoder:
    """Fit one schema on context and reuse it unchanged for every experiment arm."""

    def __init__(self, schema: Schema, target: str, task: str):
        if task not in {"classification", "regression"}:
            raise ValueError(f"Unknown task: {task}")
        self.schema, self.target, self.task = schema, target, task

    def labels(self, frame):
        if self.task == "classification":
            return _to_object_values(frame[self.target]).astype(str)
        y = pd.to_numeric(frame[self.target], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite")
        return y

    def fit(self, context, name="table"):
        s = self.schema
        if not context.columns.is_unique or not all(isinstance(c, str) for c in context.columns):
            raise ValueError("MLE expects unique string column names")
        if self.target not in context or len(context) < 2:
            raise ValueError("Context must contain its target and at least two rows")
        regression = self.task == "regression"
        if (regression and s.max_cont < 1) or (not regression and s.max_cat < 1):
            raise ValueError("Schema has no slot for this target type")
        feature_schema = Schema(s.max_cont - int(regression), s.max_cat - int(not regression), s.cat_cardinality)
        _, cont_names, _, cat_names = select_columns(context.drop(columns=[self.target]), feature_schema)
        self.cont_names = ([self.target] if regression else []) + cont_names
        self.cat_names = ([] if regression else [self.target]) + cat_names
        self.medians, self.means, self.stds = [], [], []
        for name_ in self.cont_names:
            values = pd.to_numeric(context[name_], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            finite = np.isfinite(values)
            median = float(np.median(values[finite])) if finite.any() else 0.0
            values = np.where(finite, values, median)
            self.medians.append(median)
            self.means.append(float(values.mean()))
            self.stds.append(float(values.std()) + 1e-6)
        self.levels, self.other_codes, cards = [], [], []
        for name_ in self.cat_names:
            values = _to_object_values(context[name_]).astype(str)
            levels, counts = np.unique(values, return_counts=True)
            if name_ == self.target:
                if len(levels) < 2:
                    raise ValueError("Context has fewer than two target classes; split skipped without resampling")
                if len(levels) > s.cat_cardinality:
                    raise ValueError("Target cardinality exceeds schema; target classes are never merged")
                kept, other = levels.tolist(), None
            else:
                # Reserve an explicit bucket for rare/unseen feature values, fitted on context only.
                order = np.argsort(-counts, kind="stable")
                kept = levels[order][:s.cat_cardinality - 1].tolist()
                other = len(kept)
            self.levels.append(kept)
            self.other_codes.append(other)
            cards.append(len(kept) + int(other is not None))
        self.metadata = {
            "dataset_name": name, "n_cont": len(self.cont_names), "n_cat": len(self.cat_names),
            "cat_cardinality": s.cat_cardinality, "cat_cardinalities": cards,
            "cont_names": self.cont_names, "cat_names": self.cat_names,
            "cont_mean": self.means, "cont_std": self.stds, "cont_medians": self.medians,
            "cat_levels": [levels + (["<other>"] if other is not None else [])
                           for levels, other in zip(self.levels, self.other_codes)],
            "category_other_codes": self.other_codes,
            "target_col": self.target, "task_type": self.task,
            "label_cont_index": 0 if regression else None,
            "label_cat_index": None if regression else 0,
            "no_missingness": True, "force_observed_mask": True,
            "preprocessing_fit": "context_only", "n_rows": len(context),
            "dropped_features": [c for c in context if c not in self.cont_names + self.cat_names],
        }
        self.feature_mask = mixed_feature_mask(self.metadata, *s.as_tuple())
        # Missingness bits are constant under this protocol, so predictors use values/one-hots only.
        self.predictor_mask = self.feature_mask.copy()
        self.predictor_mask[s.mask_start:] = False
        if regression:
            self.predictor_mask[0] = False
        else:
            self.predictor_mask[s.cat_start:s.cat_start + s.cat_cardinality] = False
        if not self.predictor_mask.any():
            raise ValueError("No usable predictor features remain after context-only selection")
        return self

    def transform(self, frame, include_target=False):
        s = self.schema
        out = np.zeros((len(frame), s.encoded_dim), dtype=np.float32)
        for j, col in enumerate(self.cont_names):
            if col == self.target and not include_target:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            values = np.where(np.isfinite(values), values, self.medians[j])
            out[:, j] = (values - self.means[j]) / self.stds[j]
        out[:, s.mask_start:s.mask_start + len(self.cont_names)] = 1
        for j, col in enumerate(self.cat_names):
            if col == self.target and not include_target:
                continue
            mapping = {value: i for i, value in enumerate(self.levels[j])}
            values = _to_object_values(frame[col]).astype(str)
            codes = np.array([mapping.get(value, self.other_codes[j]) for value in values], dtype=object)
            if any(code is None for code in codes):
                raise ValueError("Cannot encode unknown target labels into the context vocabulary")
            out[np.arange(len(frame)), s.cat_start + j * s.cat_cardinality + codes.astype(int)] = 1
        if not np.isfinite(out).all():
            raise ValueError("Non-finite encoded values after context scaling")
        return out

    def real_xy(self, frame):
        # Real target labels, including classes unseen in context, are never collapsed to 'other'.
        return self.transform(frame, include_target=False)[:, self.predictor_mask], self.labels(frame)

    def synthetic_xy(self, encoded):
        if not np.isfinite(encoded).all():
            raise ValueError("Generator returned non-finite values")
        x = sanitize_mixed_encoded(encoded, self.metadata, *self.schema.as_tuple())
        if self.task == "regression":
            y = x[:, 0].astype(np.float64) * self.stds[0] + self.means[0]
        else:
            start = self.schema.cat_start
            codes = x[:, start:start + len(self.levels[0])].argmax(axis=1)
            y = np.asarray(self.levels[0], dtype=str)[codes]
        return x[:, self.predictor_mask], y


@dataclass(frozen=True)
class PredictorConfig:
    n_jobs: int = 1
    tabicl_device: str = "cpu"
    tabicl_n_estimators: int = 8
    tabicl_classifier_path: str | None = None
    tabicl_regressor_path: str | None = None
    tabicl_allow_download: bool = False


def make_predictor(name, task, seed, config):
    if name == "linear":
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        estimator = (LogisticRegression(max_iter=1000, random_state=seed) if task == "classification"
                     else Ridge(alpha=1.0))
        return make_pipeline(StandardScaler(), estimator)
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(n_estimators=200, max_depth=4, learning_rate=0.05, tree_method="hist",
                   random_state=seed, n_jobs=config.n_jobs)
    if name == "tabicl":
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLClassifier if task == "classification" else TabICLRegressor
        path = config.tabicl_classifier_path if task == "classification" else config.tabicl_regressor_path
        return cls(n_estimators=config.tabicl_n_estimators, device=config.tabicl_device,
                   model_path=path, allow_auto_download=config.tabicl_allow_download,
                   random_state=seed, n_jobs=config.n_jobs)
    raise ValueError(f"Unknown predictor {name}")


def evaluate_predictor(name, task, train_xy, test_xy, seed, config, metric_classes=None):
    """Fixed hyperparameters; every call creates a fresh estimator for one arm.

    Returns explicit status/reason on missing dependencies or failed fits. A
    synthetic set with one target class uses a declared constant predictor.
    """
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder
    X, y = train_xy
    X_test, y_test = test_xy
    result = {"status": "ok", "reason": "", "n_train": len(y), "n_test": len(y_test)}
    start = time.perf_counter()
    try:
        if not len(y) or not np.isfinite(X).all() or not np.isfinite(X_test).all():
            raise ValueError("Empty training data or non-finite predictor inputs")
        if task == "classification":
            labels = LabelEncoder().fit(y)
            if len(labels.classes_) == 1:
                from sklearn.dummy import DummyClassifier
                predictor = DummyClassifier(strategy="most_frequent")
                result["constant_predictor"] = True
            else:
                predictor = make_predictor(name, task, seed, config)
                result["constant_predictor"] = False
            predictor.fit(X, labels.transform(y))
            pred = labels.inverse_transform(np.asarray(predictor.predict(X_test), dtype=int))
            probs = predictor.predict_proba(X_test)
            classes = np.asarray(metric_classes if metric_classes is not None else np.union1d(y, y_test))
            aligned = np.zeros((len(y_test), len(classes)))
            positions = {label: i for i, label in enumerate(classes)}
            for j, code in enumerate(predictor.classes_):
                aligned[:, positions[labels.classes_[int(code)]]] = probs[:, j]
            result.update(accuracy=float(accuracy_score(y_test, pred)),
                          f1_macro=float(f1_score(y_test, pred, labels=classes, average="macro", zero_division=0)),
                          f1_weighted=float(f1_score(y_test, pred, labels=classes, average="weighted", zero_division=0)),
                          auroc=None, auroc_reason="")
            if len(classes) < 2 or len(np.unique(y_test)) != len(classes):
                result["auroc_reason"] = "Test set does not contain every scoring class; AUROC undefined"
            elif len(classes) == 2:
                result["auroc"] = float(roc_auc_score(y_test == classes[1], aligned[:, 1]))
            else:
                result["auroc"] = float(roc_auc_score(y_test, aligned, labels=classes, multi_class="ovr", average="macro"))
        else:
            if not np.isfinite(y).all() or not np.isfinite(y_test).all():
                raise ValueError("Non-finite regression targets")
            predictor = make_predictor(name, task, seed, config)
            predictor.fit(X, y)
            pred = predictor.predict(X_test)
            result.update(rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
                          mae=float(mean_absolute_error(y_test, pred)),
                          r2=float(r2_score(y_test, pred)) if len(y_test) > 1 and np.var(y_test) > 0 else None)
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    result["fit_predict_sec"] = time.perf_counter() - start
    return result


def augmentation_arms(context_xy, synthetic_xy, extra_real_xy, seed):
    X, y = context_xy
    Xs, ys = synthetic_xy
    sample = np.random.default_rng(seed).integers(0, len(y), size=len(ys))
    arms = {
        "real_context": context_xy,
        "synthetic_only": synthetic_xy,
        "context_plus_synthetic": (np.concatenate([X, Xs]), np.concatenate([y, ys])),
        "context_plus_bootstrap": (np.concatenate([X, X[sample]]), np.concatenate([y, y[sample]])),
    }
    if extra_real_xy is not None:
        Xe, ye = extra_real_xy
        arms["context_plus_real"] = (np.concatenate([X, Xe]), np.concatenate([y, ye]))
    return arms
