"""Raw supervised tables for MLE. No full-table encoding, binning or scaling."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .real_data import (HF_TABARENA_BROAD_DATASETS, HF_TABARENA_PILOT_DATASETS,
                        OPENML_SPECS, _find_hf_dataset_prefix, safe_name)


@dataclass
class RawTable:
    name: str
    frame: pd.DataFrame
    target: str
    task: str
    source: str
    test_frame: pd.DataFrame | None = None


def _tabsyn_frame(directory, split):
    y = np.load(directory / f"y_{split}.npy", allow_pickle=False).reshape(-1)
    columns = {}
    for stem, prefix, categorical in [("X_num", "num", False), ("X_cat", "cat", True)]:
        path = directory / f"{stem}_{split}.npy"
        if path.exists():
            x = np.load(path, allow_pickle=False).reshape(len(y), -1)
            for j in range(x.shape[1]):
                columns[f"{prefix}_{j}"] = x[:, j].astype(str) if categorical else x[:, j].astype(float)
    columns["__target__"] = y
    return pd.DataFrame(columns)


def load_raw_tables(args):
    """Yield raw tables, preserving supplied train/test partitions.

    TabSyn NPY inputs must contain raw values, as in its dataset preparation
    output. Any preprocessing already performed by an external source cannot
    be reversed here. Loading failures are surfaced instead of silently omitted.
    """
    source = args.data_source
    if source in {"sklearn", "all"}:
        from sklearn import datasets
        loaders = {"iris": datasets.load_iris, "wine": datasets.load_wine,
                   "breast_cancer": datasets.load_breast_cancer, "diabetes": datasets.load_diabetes}
        for name in args.sklearn_datasets:
            bunch = loaders[name]()
            frame = pd.DataFrame(bunch.data, columns=[str(c) for c in bunch.feature_names])
            frame["__target__"] = bunch.target
            yield RawTable(name, frame, "__target__", "regression" if name == "diabetes" else "classification", "sklearn")
    if source == "csv":
        if not args.csv_path or not args.target_col or not args.task_type:
            raise ValueError("CSV MLE requires --csv_path, --target_col and --task_type")
        frame = pd.read_csv(args.csv_path)
        test = pd.read_csv(args.csv_test_path) if args.csv_test_path else None
        yield RawTable(safe_name(Path(args.csv_path).stem), frame, args.target_col, args.task_type, "csv", test)
    if source in {"tabsyn", "all"}:
        if not args.tabsyn_data_root and source == "tabsyn":
            raise ValueError("--tabsyn_data_root is required")
        if args.tabsyn_data_root:
            for name in args.tabsyn_datasets:
                directory = Path(args.tabsyn_data_root) / name
                info = json.loads((directory / "info.json").read_text())
                task = info.get("task_type")
                if task not in {"regression", "binclass", "multiclass", "classification"}:
                    raise ValueError(f"{name}: info.json must declare task_type")
                yield RawTable(f"tabsyn_{name}", _tabsyn_frame(directory, "train"), "__target__",
                               "regression" if task == "regression" else "classification", "tabsyn_npy",
                               _tabsyn_frame(directory, "test"))
    if source in {"openml", "all"}:
        from sklearn.datasets import fetch_openml
        for name in args.openml_datasets:
            data_id, target, _ = OPENML_SPECS[name]
            bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=args.openml_cache_dir)
            yield RawTable(f"raw_openml_{name}", bunch.frame, target, "classification", "openml")
    if source == "hf_tabarena":
        from huggingface_hub import HfApi, hf_hub_download
        repo = "TabArena/BeyondArena"
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        names = {"pilot": HF_TABARENA_PILOT_DATASETS, "broad": HF_TABARENA_BROAD_DATASETS,
                 "custom": args.hf_tabarena_datasets or []}[args.hf_tabarena_set]
        for alias in names:
            prefix = _find_hf_dataset_prefix(files, alias)
            def download(filename):
                return hf_hub_download(repo, f"{prefix}/{filename}", repo_type="dataset",
                                       cache_dir=args.hf_tabarena_cache_dir)
            meta = json.loads(Path(download("task_metadata.predictive-ml-task-mold-v1.json")).read_text())
            frame = pd.read_parquet(download("dataset.parquet"))
            task = "regression" if "regression" in str(meta["problem_type"]) else "classification"
            yield RawTable(f"hf_tabarena_{safe_name(alias)}", frame, meta["target_column_name"], task, "hf_tabarena")
