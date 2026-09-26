#!/usr/bin/env python3
"""Summarise one or more ``evaluate_real.py`` runs into a K-curve table and figure.

Example::

    python scripts/summarize_results.py results/sklearn_calibrated results/sklearn_uncalibrated \
        --labels calibrated uncalibrated --metric encoded_mmd --output results/k_curve.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(run_dir: Path):
    with (run_dir / "eval_per_dataset.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Directories containing eval_per_dataset.csv")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--method", default="gen_tfm_matched")
    p.add_argument("--metric", default="encoded_mmd")
    p.add_argument("--dataset", default=None, help="Restrict to one dataset (default: average over all)")
    p.add_argument("--output", default=None, help="Path of the PNG to write")
    args = p.parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]

    curves = {}
    for run, label in zip(args.runs, labels):
        rows = [r for r in load_rows(Path(run)) if r["method"] == args.method]
        if args.dataset:
            rows = [r for r in rows if r["dataset_name"] == args.dataset]
        per_k = defaultdict(list)
        for r in rows:
            try:
                per_k[int(r["context_size"])].append(float(r[args.metric]))
            except ValueError:
                pass
        curves[label] = {k: (float(np.nanmean(v)), len(v)) for k, v in sorted(per_k.items())}

    ks = sorted({k for c in curves.values() for k in c})
    header = f"{'K':>6} " + " ".join(f"{lab:>18}" for lab in curves)
    print(f"method={args.method} metric={args.metric} dataset={args.dataset or 'all (mean over datasets x repeats)'}")
    print(header)
    for k in ks:
        print(f"{k:>6} " + " ".join(f"{curves[lab].get(k, (float('nan'), 0))[0]:>18.6f}" for lab in curves))

    if args.output:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.2))
        for lab, c in curves.items():
            ax.plot(list(c.keys()), [v[0] for v in c.values()], marker="o", label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("context size K")
        ax.set_ylabel(args.metric)
        ax.set_title(args.dataset or "mean over datasets")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=160)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
