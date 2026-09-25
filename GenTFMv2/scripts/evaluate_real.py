#!/usr/bin/env python3
"""Zero-shot evaluation of a Gen-TFM checkpoint on real tables.

For every table, every context size K and every repeat: sample K context rows,
a held-out test set and a real reference set (disjoint), generate rows from
the K context rows, and compare generated vs. held-out rows.

Data sources
------------
``sklearn``      iris / wine / breast_cancer / diabetes (offline)
``openml``       raw OpenML Adult via ``fetch_openml`` (needs network once)
``tabsyn``       datasets in TabSyn ``.npy`` layout (``--tabsyn_data_root``)
``hf_tabarena``  Hugging Face ``TabArena/BeyondArena`` parquet datasets
``all``          sklearn + tabsyn + openml   (= the "all-real" slice of the reference results)

Example (frozen setting)::

    python scripts/evaluate_real.py --checkpoint_path checkpoints/gen_tfm_target_rich_100k.pt \
        --data_source sklearn --output_dir results/sklearn_calibrated --calibration logk

Outputs: ``eval_per_dataset.csv`` (one row per dataset x K x repeat x method),
``eval_results.json`` (mean/std), ``split_plan.csv``, per-dataset K-curve figures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_tfm.baselines import MIXED_BASELINES  # noqa: E402
from gen_tfm.checkpoint import load_pretrained  # noqa: E402
from gen_tfm.encoding import Schema  # noqa: E402
from gen_tfm.generation import Calibration, generate_in_context, generate_zero_context  # noqa: E402
from gen_tfm.metrics import downstream_accuracy, evaluate_mixed  # noqa: E402
from gen_tfm.real_data import (  # noqa: E402
    HF_TABARENA_BROAD_DATASETS,
    HF_TABARENA_PILOT_DATASETS,
    load_hf_tabarena_tables,
    load_openml_tables,
    load_sklearn_tables,
    load_tabsyn_tables,
)


def log_event(event: str, **payload):
    record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "event": event}
    record.update(payload)
    print(json.dumps(record, sort_keys=True), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-data benchmark for Gen-TFM")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_source", choices=["sklearn", "openml", "tabsyn", "hf_tabarena", "all", "csv"], default="sklearn")
    p.add_argument("--csv_path", default="", help="Raw CSV for MLE-only evaluation")
    p.add_argument("--csv_test_path", default="", help="Optional separate, disjoint test CSV")
    p.add_argument("--target_col", default="")
    p.add_argument("--task_type", choices=["classification", "regression"], default=None)
    p.add_argument("--sklearn_datasets", nargs="+", default=["iris", "wine", "breast_cancer", "diabetes"])
    p.add_argument("--openml_datasets", nargs="+", default=["adult"])
    p.add_argument("--openml_cache_dir", type=str, default=None)
    p.add_argument("--tabsyn_data_root", type=str, default="")
    p.add_argument("--tabsyn_datasets", nargs="+", default=["adult_1590", "default_42477", "magic_1120", "california_housing_ca", "news_uci"])
    p.add_argument("--hf_tabarena_set", choices=["pilot", "broad", "custom"], default="pilot")
    p.add_argument("--hf_tabarena_datasets", nargs="+", default=None, help="Used with --hf_tabarena_set custom")
    p.add_argument("--hf_tabarena_cache_dir", type=str, default=None)
    # protocol
    p.add_argument("--context_sizes", type=int, nargs="+", default=[5, 20, 50, 100, 200])
    p.add_argument("--n_repeats", type=int, default=12)
    p.add_argument("--eval_test_rows", type=int, default=512)
    p.add_argument("--n_gen_rows", type=int, default=512)
    p.add_argument("--min_eval_test_rows", type=int, default=64)
    p.add_argument("--min_gen_rows", type=int, default=64)
    p.add_argument("--n_ode_steps", type=int, default=60)
    p.add_argument("--ode_method", choices=["euler", "heun"], default="euler")
    p.add_argument("--calibration", choices=["none", "logk", "constant"], default="logk",
                   help="Generation-time categorical context calibration for gen_tfm_matched")
    p.add_argument("--cat_context_alpha", type=float, default=3.5)
    p.add_argument("--cat_context_tau", type=float, default=0.25)
    p.add_argument("--also_uncalibrated", action="store_true", help="Add a gen_tfm_uncalibrated method for comparison")
    p.add_argument("--skip_baselines", action="store_true")
    # Strict MLE uses a separate raw-data path, fitted on context only.
    p.add_argument("--also_mle", action="store_true", help="Run TSTR and augmentation using context-only preprocessing")
    p.add_argument("--mle_only", action="store_true", help="Only run MLE; skip legacy distribution metrics and figures")
    p.add_argument("--also_tabicl", action="store_true", help="Enable MLE and add TabICL to the requested predictors")
    p.add_argument("--mle_estimators", nargs="+", choices=["linear", "xgboost", "tabicl"], default=["linear", "xgboost"])
    p.add_argument("--mle_syn_rows", type=int, default=200, help="Exact synthetic and extra-real row count, independent of --n_gen_rows")
    p.add_argument("--mle_n_jobs", type=int, default=1)
    p.add_argument("--mle_save_synthetic", action="store_true")
    p.add_argument("--mle_tabicl_device", default=None, help="Defaults to GenTFM evaluation device")
    p.add_argument("--mle_tabicl_n_estimators", type=int, default=8)
    p.add_argument("--mle_tabicl_classifier_path", default=None)
    p.add_argument("--mle_tabicl_regressor_path", default=None)
    p.add_argument("--mle_tabicl_allow_download", action="store_true", help="Allow TabICL to download weights; otherwise use local/cache weights")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1401)
    return p.parse_args(argv)


def load_tables(args, schema: Schema):
    tables = []
    if args.data_source in {"sklearn", "all"}:
        tables += load_sklearn_tables(schema, args.sklearn_datasets)
    if args.data_source in {"tabsyn", "all"}:
        if args.tabsyn_data_root:
            tables += load_tabsyn_tables(schema, args.tabsyn_data_root, args.tabsyn_datasets)
        elif args.data_source == "tabsyn":
            raise SystemExit("--tabsyn_data_root is required for --data_source tabsyn")
        else:
            log_event("tabsyn_skipped", reason="no --tabsyn_data_root given")
    if args.data_source in {"openml", "all"}:
        tables += load_openml_tables(schema, args.openml_datasets, cache_dir=args.openml_cache_dir)
    if args.data_source == "hf_tabarena":
        names = {"pilot": HF_TABARENA_PILOT_DATASETS, "broad": HF_TABARENA_BROAD_DATASETS, "custom": args.hf_tabarena_datasets or []}[args.hf_tabarena_set]
        tables += load_hf_tabarena_tables(schema, names, cache_dir=args.hf_tabarena_cache_dir)
    if not tables:
        raise SystemExit("No tables loaded")
    for name, table, meta in tables:
        log_event("table_loaded", dataset_name=name, n_rows=int(len(table)), n_cont=meta["n_cont"], n_cat=meta["n_cat"], source=meta.get("data_source"))
    return tables


def make_split(table: np.ndarray, k: int, args, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Disjoint context / test / real-reference split; sizes shrink for small tables."""
    n_rows = len(table)
    remaining = n_rows - k
    if remaining < args.min_eval_test_rows + args.min_gen_rows:
        raise ValueError(f"not enough rows: n={n_rows}, K={k}")
    eval_rows = min(args.eval_test_rows, max(args.min_eval_test_rows, remaining // 2))
    gen_rows = min(args.n_gen_rows, remaining - eval_rows)
    if gen_rows < args.min_gen_rows:
        gen_rows = args.min_gen_rows
        eval_rows = remaining - gen_rows
    if eval_rows < args.min_eval_test_rows:
        raise ValueError(f"not enough eval rows: n={n_rows}, K={k}")
    total = k + eval_rows + gen_rows
    idx = rng.choice(n_rows, size=total, replace=False)
    full = table[idx]
    info = {"source_rows": int(n_rows), "context_size": int(k), "eval_test_rows_actual": int(eval_rows), "n_gen_rows_actual": int(gen_rows)}
    return full[:k], full[k : k + eval_rows], full[k + eval_rows :], info


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    groups: Dict[str, Dict[str, list]] = {}
    skip = {"dataset_name", "context_size", "repeat", "method", "n_cont", "n_cat"}
    for row in rows:
        key = f"{row['dataset_name']}|K={row['context_size']}|{row['method']}"
        g = groups.setdefault(key, defaultdict(list))
        for metric, value in row.items():
            if metric in skip:
                continue
            try:
                g[metric].append(float(value))
            except (TypeError, ValueError):
                pass
    return {key: {"mean": {m: float(np.nanmean(v)) for m, v in g.items()},
                  "std": {m: float(np.nanstd(v)) for m, v in g.items()},
                  "n": {m: int(np.isfinite(v).sum()) for m, v in g.items()}} for key, g in groups.items()}


def plot_results(summary: Dict[str, object], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for key, item in summary.items():
        dataset, k_part, method = key.split("|")
        rows.append({"dataset_name": dataset, "context_size": int(k_part.split("=")[1]), "method": method, **item["mean"]})
    metrics = [("encoded_mmd", "Encoded MMD", True), ("cat_js_mean", "Categorical JS", True),
               ("cont_observed_wasserstein", "Continuous Wasserstein", True), ("downstream_accuracy", "TSTR accuracy", False),
               ("generation_time_sec", "Generation time (s)", True), ("dcr_ctx_q05", "DCR to context (q05)", True)]
    for dataset in sorted({r["dataset_name"] for r in rows}):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
        for ax, (metric, title, logy) in zip(axes.ravel(), metrics):
            for method in sorted({r["method"] for r in rows}):
                items = sorted([r for r in rows if r["dataset_name"] == dataset and r["method"] == method], key=lambda r: r["context_size"])
                if not items or metric not in items[0]:
                    continue
                ax.plot([r["context_size"] for r in items], [r[metric] for r in items], marker="o", label=method)
            ax.set_title(title)
            ax.set_xlabel("K")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if logy:
                ax.set_yscale("symlog", linthresh=1e-4)
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle(f"Real benchmark: {dataset}")
        fig.tight_layout()
        fig.savefig(output_dir / f"real_benchmark_{dataset}.png", dpi=160)
        plt.close(fig)


def main(args=None):
    args = args or parse_args()
    if args.data_source == "csv" and not args.mle_only:
        raise SystemExit("CSV input currently requires --mle_only (or scripts/evaluate_mle.py)")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_pretrained(args.checkpoint_path, device=str(device))
    schema = Schema(model.max_cont, model.max_cat, model.cat_cardinality)
    log_event("checkpoint_loaded", path=args.checkpoint_path, step=int(ckpt.get("step", -1)), device=str(device), **model.config())

    calibration = None if args.calibration == "none" else Calibration(alpha=args.cat_context_alpha, schedule=args.calibration, tau=args.cat_context_tau)
    if args.also_mle or args.mle_only or args.also_tabicl:
        from gen_tfm.mle_benchmark import run_mle_benchmark
        mle_rows = run_mle_benchmark(args, model, schema, calibration, device)
        log_event("mle_complete", n_results=len(mle_rows), n_ok=sum(r["status"] == "ok" for r in mle_rows),
                  output_dir=str(output_dir / "mle"))
        if args.mle_only:
            return
    tables = load_tables(args, schema)
    (output_dir / "dataset_metadata.json").write_text(json.dumps({n: m for n, _, m in tables}, indent=2, default=str))

    rows, split_rows = [], []
    start_all = time.perf_counter()
    for dataset_name, table, metadata in tables:
        for k in args.context_sizes:
            for rep in range(args.n_repeats):
                rng = np.random.default_rng(args.seed + 1000 * rep + 37 * k + len(dataset_name))
                try:
                    X_context, X_test, X_ref, info = make_split(table, k, args, rng)
                except ValueError as exc:
                    split_rows.append({"dataset_name": dataset_name, "context_size": k, "repeat": rep, "status": "skipped", "reason": str(exc)})
                    continue
                split_rows.append({"dataset_name": dataset_name, "repeat": rep, "status": "used", **info})
                n_gen = int(info["n_gen_rows_actual"])

                candidates: Dict[str, np.ndarray] = {}
                times: Dict[str, float] = {}

                def add(name, fn):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    candidates[name] = fn()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    times[name] = time.perf_counter() - t0

                add("gen_tfm_matched", lambda: generate_in_context(model, X_context, metadata, n_gen, schema, args.n_ode_steps, args.ode_method, calibration, device))
                if args.also_uncalibrated and calibration is not None:
                    add("gen_tfm_uncalibrated", lambda: generate_in_context(model, X_context, metadata, n_gen, schema, args.n_ode_steps, args.ode_method, None, device))
                add("gen_tfm_zero", lambda: generate_zero_context(model, X_context, metadata, n_gen, schema, args.n_ode_steps, args.ode_method, device))
                if not args.skip_baselines:
                    for name, fn in MIXED_BASELINES.items():
                        add(name, lambda fn=fn: fn(X_context, metadata, *schema.as_tuple(), n_gen, np.random.default_rng(rng.integers(0, 2**31 - 1))))
                add("real_vs_real", lambda: X_ref)

                for method, cand in candidates.items():
                    metrics = evaluate_mixed(X_context, cand, X_test, metadata, *schema.as_tuple(), seed=args.seed + rep + k)
                    metrics["downstream_accuracy"] = downstream_accuracy(cand, X_test, metadata, *schema.as_tuple())
                    rows.append({"dataset_name": dataset_name, "context_size": k, "repeat": rep, "n_cont": metadata["n_cont"], "n_cat": metadata["n_cat"],
                                 "method": method, "generation_time_sec": times.get(method, 0.0), **info, **metrics})
                    if method == "gen_tfm_matched":
                        log_event("metric_complete", dataset_name=dataset_name, context_size=k, repeat=rep, method=method,
                                  encoded_mmd=metrics["encoded_mmd"], cat_js_mean=metrics["cat_js_mean"], downstream_accuracy=metrics["downstream_accuracy"])

    total_sec = time.perf_counter() - start_all
    summary = summarize(rows)
    keys = sorted({k for r in rows for k in r}) or ["dataset_name", "context_size", "status"]
    with (output_dir / "eval_per_dataset.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with (output_dir / "split_plan.csv").open("w", newline="") as f:
        keys = sorted({k for r in split_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(split_rows)
    (output_dir / "eval_results.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "eval_timing.json").write_text(json.dumps({"eval_total_sec": total_sec, "eval_total_min": total_sec / 60.0, "n_datasets": len(tables)}, indent=2))
    if rows:
        plot_results(summary, output_dir)

    # console summary: mean encoded MMD of gen_tfm_matched per K, averaged over datasets
    for k in args.context_sizes:
        vals = [r["encoded_mmd"] for r in rows if r["method"] == "gen_tfm_matched" and r["context_size"] == k]
        if vals:
            log_event("summary_gen_tfm_matched", context_size=k, mean_encoded_mmd=float(np.mean(vals)), n=len(vals))
    log_event("run_complete", output_dir=str(output_dir), eval_total_min=total_sec / 60.0)


if __name__ == "__main__":
    main()
