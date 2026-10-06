"""Synthetic data engine: TabICL's SCM prior repurposed for row generation.

TabICL (https://github.com/soda-inria/tabicl) ships a synthetic *supervised*
prior: every call samples a random structural causal model (an MLP-SCM or a
tree-SCM), draws N rows of features ``X`` from it and derives a classification
target ``y``.  Gen-TFM needs *tables*, not prediction tasks, so the adapter
below turns each ``(X, y)`` into one mixed-type table:

* ``y``                       -> the first categorical column
* a few feature columns       -> extra categorical columns (discretised)
* the remaining features      -> continuous columns

Missingness is disabled: every continuous value is observed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .encoding import Schema, mixed_feature_mask, sanitize_mixed_encoded, slices


def _ensure_tabicl_importable() -> None:
    """Make ``import tabicl`` work from the vendored submodule if it is not installed."""
    try:
        import tabicl  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "external" / "tabicl" / "src"]
    if os.environ.get("TABICL_SRC"):
        candidates.append(Path(os.environ["TABICL_SRC"]))
    for candidate in candidates:
        if (candidate / "tabicl").exists():
            sys.path.insert(0, str(candidate))
            return
    raise ImportError(
        "TabICL is not importable. Either `pip install tabicl`, or clone this repo with "
        "`git clone --recursive` (or run `git submodule update --init`) so that external/tabicl exists."
    )


@dataclass
class PriorConfig:
    """Knobs of the synthetic data engine.  Defaults = the 'medium MixSCM' recipe of the frozen checkpoint."""

    prior_type: str = "mix_scm"          # 'mlp_scm', 'tree_scm' or 'mix_scm' (random mixture of both)
    target_task: str = "classification"   # classification or regression supervision for the generated table
    min_features: int = 8                # TabICL feature range before the adapter
    max_features: int = 48
    max_classes: int = 12                # cardinality cap for the target column
    force_feature_cats: int = 6          # how many feature columns are discretised into categorical columns
    feature_cat_min_card: int = 2
    feature_cat_max_card: int = 12
    detect_feature_cats: bool = True     # first reuse columns that are already low-cardinality
    train_min_seq_len: int = 712         # target-rich training: tables are 712..rows_per_dataset rows long
    log_seq_len: bool = True
    replay_small: bool = False
    batch_size_per_gp: int = 2
    device: str = "cpu"                  # device on which TabICL samples the SCMs

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TabICLPriorEngine:
    """Sample batches of encoded synthetic tables.

    Interface used by the training loop::

        X, metadata = engine.sample_batch(batch_size, total_rows, return_metadata=True, split="train")
        # X: [batch_size, total_rows, schema.encoded_dim] float tensor
        # metadata: list of dicts with n_cont, n_cat, cat_cardinalities, ...
    """

    def __init__(self, schema: Schema = Schema(), config: PriorConfig = PriorConfig(), min_cont: int = 6,
                 output_device: str = "cpu"):
        _ensure_tabicl_importable()
        from tabicl.prior import PriorDataset

        self.PriorDataset = PriorDataset
        self.schema = schema
        self.config = config
        self.max_cont = int(schema.max_cont)
        self.min_cont = int(min_cont)
        self.max_cat = int(schema.max_cat)
        self.cat_cardinality = int(schema.cat_cardinality)
        self.max_features = schema.encoded_dim
        self.output_device = torch.device(output_device)

        self.prior_type = config.prior_type
        self.batch_size_per_gp = int(config.batch_size_per_gp)
        self.min_features = int(config.min_features)
        self.tabicl_max_features = int(config.max_features)
        self.max_classes = min(int(config.max_classes), self.cat_cardinality)
        self.train_min_seq_len = int(config.train_min_seq_len)
        self.log_seq_len = bool(config.log_seq_len)
        self.replay_small = bool(config.replay_small)
        self.detect_feature_cats = bool(config.detect_feature_cats)
        self.force_feature_cats = int(config.force_feature_cats)
        self.feature_cat_min_card = max(2, int(config.feature_cat_min_card))
        self.feature_cat_max_card = min(self.cat_cardinality, max(self.feature_cat_min_card, int(config.feature_cat_max_card)))
        self.generated_batches = 0
        self._prior_cache = {}

    # ------------------------------------------------------------ TabICL side
    def _new_prior(self, batch_size: int, total_rows: int, split: Optional[str] = None):
        min_seq_len = None
        if split == "train" and self.train_min_seq_len > 0:
            min_seq_len = max(2, min(int(self.train_min_seq_len), int(total_rows) - 1))
        key = (int(batch_size), int(total_rows), split or "none", min_seq_len or 0)
        if key in self._prior_cache:
            return self._prior_cache[key]
        prior = self.PriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=min(self.batch_size_per_gp, batch_size),
            min_features=self.min_features,
            max_features=self.tabicl_max_features,
            max_classes=self.max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=total_rows,
            log_seq_len=self.log_seq_len,
            replay_small=self.replay_small,
            min_train_size=0.4,
            max_train_size=0.8,
            prior_type=self.prior_type,
            n_jobs=1,
            num_threads_per_generate=1,
            device=self.config.device,
        )
        self._prior_cache[key] = prior
        return prior

    # ------------------------------------------------------------- adapter
    @staticmethod
    def _encode_labels(values: np.ndarray, max_cardinality: int) -> Tuple[np.ndarray, int]:
        """Map arbitrary values to integer codes; rank-bin when there are too many distinct values."""
        values = np.asarray(values)
        uniq = np.unique(values)
        if len(uniq) > max_cardinality:
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(values), dtype=np.int64)
            ranks[order] = np.arange(len(values), dtype=np.int64)
            bins = np.linspace(0, len(values), max_cardinality + 1)[1:-1]
            return np.searchsorted(bins, ranks).astype(np.int64), max_cardinality
        mapping = {v: i for i, v in enumerate(uniq)}
        encoded = np.asarray([mapping[v] for v in values], dtype=np.int64)
        return encoded, max(2, int(len(uniq)))

    def _categorical_feature_columns(self, X: np.ndarray, y: np.ndarray, d: int) -> List[int]:
        """Choose which feature columns become categorical columns of the table."""
        max_extra = max(0, self.max_cat - 1)
        candidates: List[int] = []
        if self.detect_feature_cats:
            for j in range(d):
                col = np.asarray(X[:, j], dtype=np.float32)
                rounded = np.round(col, 6)
                uniq = np.unique(rounded[np.isfinite(rounded)])
                if 2 <= len(uniq) <= self.cat_cardinality:
                    candidates.append(j)
                if len(candidates) >= max_extra:
                    return candidates[:max_extra]

        target_count = min(max_extra, max(self.force_feature_cats, len(candidates)))
        if len(candidates) >= target_count:
            return candidates[:max_extra]

        # Not enough naturally discrete columns: discretise the columns most correlated with y.
        y_num = np.asarray(y, dtype=np.float32)
        y_std = float(np.std(y_num))
        scores = []
        used = set(candidates)
        for j in range(d):
            if j in used:
                continue
            col = np.asarray(X[:, j], dtype=np.float32)
            if not np.isfinite(col).all() or float(np.std(col)) < 1e-6:
                continue
            if y_std > 1e-6:
                corr = np.corrcoef(col, y_num)[0, 1]
                score = abs(float(corr)) if np.isfinite(corr) else 0.0
            else:
                score = float(np.std(col))
            scores.append((score, j))
        scores.sort(reverse=True)
        for _score, j in scores:
            candidates.append(j)
            if len(candidates) >= target_count:
                break
        return candidates[:max_extra]

    def _feature_cardinality(self, dataset_id: int, local_j: int) -> int:
        span = self.feature_cat_max_card - self.feature_cat_min_card + 1
        if span <= 1:
            return self.feature_cat_max_card
        offset = (self.generated_batches + dataset_id + local_j) % span
        return self.feature_cat_min_card + offset

    def _encode_one_dataset(self, X: np.ndarray, y: np.ndarray, d: int, split: Optional[str], dataset_id: int,
                            seq_len: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        d = int(max(1, min(d, X.shape[1])))
        X = np.asarray(X[:, :d], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        target_task = str(self.config.target_task).lower()
        if target_task not in {"classification", "regression"}:
            raise ValueError(f"Unsupported target_task: {self.config.target_task}")
        if target_task == "regression":
            # The first source feature becomes the numeric supervision target.
            # Remaining SCM features retain their dependency on this target.
            y_reg = X[:, 0].copy()
            X = X[:, 1:]
            d = max(0, d - 1)
            y_enc, y_card = None, None
        else:
            y_enc, y_card = self._encode_labels(y, self.cat_cardinality)

        feature_cat_cols = self._categorical_feature_columns(X, y if target_task == "classification" else y_reg, d) if d else []
        cont_cols = [j for j in range(d) if j not in set(feature_cat_cols)]
        cont_cols.extend(j for j in feature_cat_cols if j not in cont_cols)
        cont_cols = cont_cols[: self.max_cont]
        if not cont_cols and target_task == "classification":
            cont_cols = [0]
            feature_cat_cols = [j for j in feature_cat_cols if j != 0]

        cont_capacity = max(0, self.max_cont - (1 if target_task == "regression" else 0))
        cont_cols = cont_cols[:cont_capacity]
        n_cont = min(len(cont_cols) + (1 if target_task == "regression" else 0), self.max_cont)
        feature_cat_cols = feature_cat_cols[: max(0, self.max_cat - (1 if target_task == "classification" else 0))]
        n_cat = len(feature_cat_cols) + (1 if target_task == "classification" else 0)

        D = self.max_features
        sl = slices(self.max_cont, self.max_cat, self.cat_cardinality)
        encoded = np.zeros((len(X), D), dtype=np.float32)
        cont_features = X[:, cont_cols].astype(np.float32) if cont_cols else np.zeros((len(X), 0), dtype=np.float32)
        cont = np.column_stack([y_reg, cont_features]) if target_task == "regression" else cont_features
        mu = cont.mean(axis=0, keepdims=True)
        std = cont.std(axis=0, keepdims=True) + 1e-6
        encoded[:, :n_cont] = (cont - mu) / std
        encoded[:, sl["mask_start"] : sl["mask_start"] + n_cont] = 1.0

        cat_cards = []
        cat_names = []
        for local_j, col_idx in enumerate(feature_cat_cols):
            card_limit = self._feature_cardinality(dataset_id, local_j)
            vals, card = self._encode_labels(np.round(X[:, col_idx], 6), card_limit)
            start = sl["cat_start"] + local_j * self.cat_cardinality
            encoded[np.arange(len(X)), start + vals] = 1.0
            cat_cards.append(int(card))
            cat_names.append(f"tabicl_cat_{col_idx}")
        label_cat_index = None
        if target_task == "classification":
            label_cat_index = len(feature_cat_cols)
            cat_cards.append(int(y_card))
            cat_names.append("target")
            start = sl["cat_start"] + label_cat_index * self.cat_cardinality
            encoded[np.arange(len(X)), start + y_enc] = 1.0

        metadata: Dict[str, object] = {
            "prior_type": f"tabicl_{self.prior_type}",
            "dataset_name": f"tabicl_{self.prior_type}_{self.generated_batches}_{dataset_id}",
            "data_source": "tabicl_prior",
            "split": split or "train",
            "n_cont": int(n_cont),
            "n_cat": int(n_cat),
            "cat_cardinality": int(self.cat_cardinality),
            "cat_cardinalities": cat_cards,
            "cat_names": cat_names,
            "target_task": target_task,
            "label_type": "categorical" if target_task == "classification" else "continuous",
            "label_cat_index": label_cat_index,
            "label_cont_index": 0 if target_task == "regression" else None,
            "force_observed_mask": True,
            "no_missingness": True,
            "tabicl_active_features": int(d),
            "tabicl_cont_cols": [int(j) for j in cont_cols[:n_cont]],
            "tabicl_feature_cat_cols": [int(j) for j in feature_cat_cols],
            "tabicl_seq_len": int(seq_len or len(X)),
        }
        metadata["n_features"] = int(mixed_feature_mask(metadata, self.max_cont, self.max_cat, self.cat_cardinality).sum())
        return sanitize_mixed_encoded(encoded, metadata, self.max_cont, self.max_cat, self.cat_cardinality), metadata

    # ------------------------------------------------------------ public API
    def sample_dataset(self, total_rows: int, **_ignored) -> Tuple[np.ndarray, Dict[str, object]]:
        batch, metadata = self.sample_batch(1, total_rows, return_metadata=True, split="mismatched")
        return batch[0].detach().cpu().numpy(), metadata[0]

    def sample_batch(self, batch_size: int, total_rows: int, return_metadata: bool = False, split: Optional[str] = None):
        prior = self._new_prior(batch_size=batch_size, total_rows=total_rows, split=split)
        X, y, d, seq_lens, _train_sizes = prior.get_batch(batch_size)
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        d_np = d.detach().cpu().numpy()
        seq_len_np = seq_lens.detach().cpu().numpy()
        encoded_tables, metadata = [], []
        for i in range(batch_size):
            seq_len_i = int(seq_len_np[i])
            encoded, meta = self._encode_one_dataset(X_np[i, :seq_len_i], y_np[i, :seq_len_i], int(d_np[i]), split, i, seq_len=seq_len_i)
            encoded_tables.append(torch.from_numpy(encoded))
            metadata.append(meta)
        self.generated_batches += 1
        batch = torch.stack(encoded_tables, dim=0).to(self.output_device)
        if return_metadata:
            return batch, metadata
        return batch
