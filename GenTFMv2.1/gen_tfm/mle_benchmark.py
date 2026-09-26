"""Paired MLE benchmark orchestration; raw splits precede all learned preprocessing."""

from collections import defaultdict
import csv
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .generation import generate_in_context
from .mle import ContextTableEncoder, PredictorConfig, augmentation_arms, evaluate_predictor
from .mle_data import load_raw_tables
from .real_data import decode_to_dataframe, safe_name


METRICS = ("accuracy", "f1_macro", "f1_weighted", "auroc", "rmse", "mae", "r2")
ARMS = ("real_context", "synthetic_only", "context_plus_synthetic", "context_plus_bootstrap", "context_plus_real")


def stable_seed(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row}) or ["status", "reason"]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_mle(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["dataset_name"], row["context_size"], row["predictor"], row["arm"])
        groups[key].append(row)
    summary = []
    for (dataset, k, predictor, arm), group in groups.items():
        item = {"dataset_name": dataset, "context_size": k, "predictor": predictor, "arm": arm,
                "n_attempted": len(group), "n_ok": sum(r["status"] == "ok" for r in group)}
        for metric in METRICS:
            values = [r[metric] for r in group if r["status"] == "ok" and r.get(metric) is not None
                      and np.isfinite(r[metric])]
            item[metric] = {"mean": float(np.mean(values)) if values else None,
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "n": len(values)}
        summary.append(item)
    return summary


def paired_deltas(rows):
    """Subtract the same-repeat real-context baseline; positive improvement is always better."""
    baselines = {(r["dataset_name"], r["context_size"], r["repeat"], r["predictor"]): r
                 for r in rows if r["arm"] == "real_context" and r["status"] == "ok"}
    deltas = []
    for row in rows:
        key = (row["dataset_name"], row["context_size"], row["repeat"], row["predictor"])
        baseline = baselines.get(key)
        if baseline is None or row["status"] != "ok" or row["arm"] == "real_context":
            continue
        for metric in METRICS:
            a, b = row.get(metric), baseline.get(metric)
            if a is None or b is None or not np.isfinite([a, b]).all():
                continue
            delta = float(a - b)
            deltas.append({**{k: row[k] for k in ("dataset_name", "context_size", "repeat", "predictor", "arm")},
                           "metric": metric, "baseline": b, "score": a, "delta": delta,
                           "improvement": -delta if metric in {"rmse", "mae"} else delta})
    return deltas


def summarize_deltas(deltas):
    groups = defaultdict(list)
    for row in deltas:
        key = (row["dataset_name"], row["context_size"], row["predictor"], row["arm"], row["metric"])
        groups[key].append(row["improvement"])
    return [dict(zip(("dataset_name", "context_size", "predictor", "arm", "metric"), key),
                 n_pairs=len(values), mean_improvement=float(np.mean(values)),
                 std_improvement=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                 win_rate=float(np.mean(np.asarray(values) > 0))) for key, values in groups.items()]


def _clean_frame(frame, target, task):
    if target not in frame:
        raise ValueError(f"Target column {target!r} is missing")
    keep = frame[target].notna().to_numpy()
    if task == "regression":
        keep &= np.isfinite(pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
    # Returned IDs index the original source frame, before removal of missing targets.
    return frame.iloc[np.flatnonzero(keep)].reset_index(drop=True), np.flatnonzero(keep)


def run_mle_benchmark(args, model, schema, calibration, device):
    if args.mle_syn_rows < 1 or args.n_repeats < 1 or min(args.context_sizes) < 2:
        raise ValueError("MLE needs positive synthetic rows/repeats and context sizes >= 2")
    if args.eval_test_rows < args.min_eval_test_rows or args.min_eval_test_rows < 2:
        raise ValueError("eval_test_rows must be >= min_eval_test_rows >= 2")
    output = Path(args.output_dir) / "mle"
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("torch", "numpy", "pandas", "scikit-learn", "xgboost", "tabicl"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    (output / "package_versions.json").write_text(json.dumps(versions, indent=2))
    predictors = list(dict.fromkeys(args.mle_estimators + (["tabicl"] if args.also_tabicl else [])))
    predictor_config = PredictorConfig(
        n_jobs=args.mle_n_jobs, tabicl_device=args.mle_tabicl_device or str(device),
        tabicl_n_estimators=args.mle_tabicl_n_estimators,
        tabicl_classifier_path=args.mle_tabicl_classifier_path,
        tabicl_regressor_path=args.mle_tabicl_regressor_path,
        tabicl_allow_download=args.mle_tabicl_allow_download,
    )
    rows = []
    count = 0
    start = time.perf_counter()
    def save_results():
        write_csv(output / "mle_per_repeat.csv", rows)
        deltas = paired_deltas(rows)
        write_csv(output / "mle_paired_deltas.csv", deltas)
        (output / "mle_summary.json").write_text(json.dumps(summarize_mle(rows), indent=2, allow_nan=False))
        (output / "mle_paired_summary.json").write_text(json.dumps(summarize_deltas(deltas), indent=2, allow_nan=False))

    for table in load_raw_tables(args):
        count += 1
        dataset_dir = output / safe_name(table.name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frame, source_ids = _clean_frame(table.frame, table.target, table.task)
        test_rng = np.random.default_rng(stable_seed(args.seed, table.name, "fixed_test"))
        if table.test_frame is not None:
            test_pool, original_test_ids = _clean_frame(table.test_frame, table.target, table.task)
            test_idx = test_rng.permutation(len(test_pool))[:args.eval_test_rows]
            test = test_pool.iloc[test_idx]
            test_ids = original_test_ids[test_idx]
            pool_idx = np.arange(len(frame))
            split_source = "provided_test_partition"
        else:
            # A fixed held-out set per dataset, reused across K, repeats and every arm.
            n_test = min(args.eval_test_rows, max(args.min_eval_test_rows, len(frame) // 3))
            order = test_rng.permutation(len(frame))
            test_idx, pool_idx = order[:n_test], order[n_test:]
            test = frame.iloc[test_idx]
            test_ids = source_ids[test_idx]
            split_source = "random_holdout"
        (dataset_dir / "test_split.json").write_text(json.dumps({
            "dataset_name": table.name, "source": table.source, "task": table.task, "target": table.target,
            "protocol": split_source, "test_row_ids": test_ids.tolist(),
            "row_id_definition": "zero-based row positions in the source frame; supplied test has its own frame",
        }, indent=2))
        for k in args.context_sizes:
            for rep in range(args.n_repeats):
                seed = stable_seed(args.seed, table.name, k, rep)
                split_dir = dataset_dir / f"K{k}_repeat{rep}"
                split_dir.mkdir(parents=True, exist_ok=True)
                common = {"dataset_name": table.name, "task": table.task, "context_size": k,
                          "repeat": rep, "seed": seed, "n_synthetic": args.mle_syn_rows,
                          "n_test": len(test), "data_source": table.source}
                try:
                    if len(test) < args.min_eval_test_rows or len(pool_idx) < k:
                        raise ValueError("Insufficient rows for the requested context and fixed test; K is not reduced")
                    # Nested context prefixes across K for a given repeat; no target-stratified retries.
                    rng = np.random.default_rng(stable_seed(args.seed, table.name, rep, "context"))
                    chosen = rng.permutation(pool_idx)
                    ctx_idx = chosen[:k]
                    extra_idx = chosen[k:k + args.mle_syn_rows]
                    has_extra = len(extra_idx) == args.mle_syn_rows
                    context = frame.iloc[ctx_idx]
                    extra = frame.iloc[extra_idx] if has_extra else None
                    encoder = ContextTableEncoder(schema, table.target, table.task).fit(context, table.name)
                    (split_dir / "encoder.json").write_text(json.dumps(encoder.metadata, indent=2))
                    (split_dir / "split.json").write_text(json.dumps({
                        **common, "context_row_ids": source_ids[ctx_idx].tolist(),
                        "extra_real_row_ids": source_ids[extra_idx].tolist() if has_extra else [],
                        "test_row_ids": test_ids.tolist(), "extra_real_available": has_extra,
                    }, indent=2))
                    encoded_ctx = encoder.transform(context, include_target=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    gen_start = time.perf_counter()
                    with torch.random.fork_rng():
                        torch.manual_seed(seed)
                        syn = generate_in_context(model, encoded_ctx, encoder.metadata, args.mle_syn_rows,
                                                  schema, args.n_ode_steps, args.ode_method, calibration, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    common["generation_time_sec"] = time.perf_counter() - gen_start
                    if len(syn) != args.mle_syn_rows:
                        raise ValueError("Generator returned the wrong number of synthetic rows")
                    context_xy, test_xy = encoder.real_xy(context), encoder.real_xy(test)
                    extra_xy = encoder.real_xy(extra) if has_extra else None
                    synthetic_xy = encoder.synthetic_xy(syn)
                    arms = augmentation_arms(context_xy, synthetic_xy, extra_xy, seed)
                    if args.mle_save_synthetic:
                        decode_to_dataframe(syn, encoder.metadata, schema).to_csv(split_dir / "synthetic.csv", index=False)
                    classes = None
                    if table.task == "classification":
                        # Used only for scoring/probability alignment, never passed to any fit.
                        classes = np.unique(np.concatenate([context_xy[1], test_xy[1]] + ([extra_xy[1]] if has_extra else [])))
                        common["test_unseen_label_rate"] = float(np.mean(~np.isin(test_xy[1], context_xy[1])))
                    for predictor in predictors:
                        for arm in ARMS:
                            if arm not in arms:
                                result = {"status": "skipped", "reason": "Not enough extra real rows; reference is never drawn from test"}
                            else:
                                result = evaluate_predictor(predictor, table.task, arms[arm], test_xy,
                                                            seed, predictor_config, classes)
                            rows.append({**common, "predictor": predictor, "arm": arm, **result})
                            print(json.dumps({"event": "mle_result", **rows[-1]}), flush=True)
                except ValueError as exc:
                    for predictor in predictors:
                        for arm in ARMS:
                            rows.append({**common, "predictor": predictor, "arm": arm, "status": "skipped", "reason": str(exc)})
                    (split_dir / "skip.json").write_text(json.dumps({**common, "reason": str(exc)}, indent=2))
                finally:
                    save_results()
    if not count:
        raise ValueError("No raw datasets selected for MLE")
    (output / "timing.json").write_text(json.dumps({"seconds": time.perf_counter() - start, "n_datasets": count}, indent=2))
    return rows
