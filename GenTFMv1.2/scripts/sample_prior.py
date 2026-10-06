#!/usr/bin/env python3
"""Look at the synthetic data engine: sample tables from the TabICL prior and inspect them.

Example::

    python scripts/sample_prior.py --n_tables 16 --rows 512 --output_dir results/prior_samples

Writes ``prior_summary.csv`` (one row per table: number of columns, cardinalities,
prior type), a few decoded tables as CSV, and ``prior_overview.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_tfm.encoding import Schema, decode_components  # noqa: E402
from gen_tfm.prior import PriorConfig, TabICLPriorEngine  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tables", type=int, default=16)
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--prior_type", choices=["mlp_scm", "tree_scm", "mix_scm"], default="mix_scm")
    p.add_argument("--target_task", choices=["classification", "regression"], default="classification")
    p.add_argument("--n_csv", type=int, default=4, help="How many sampled tables to dump as CSV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results/prior_samples")
    p.add_argument("--cat_encoding", choices=["onehot", "binary"], default="binary")
    args = p.parse_args()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    schema = Schema(cat_encoding=args.cat_encoding)
    engine = TabICLPriorEngine(schema=schema, config=PriorConfig(prior_type=args.prior_type, target_task=args.target_task, train_min_seq_len=0))
    X, metadata = engine.sample_batch(args.n_tables, args.rows, return_metadata=True, split="train")
    print(f"batch tensor: {tuple(X.shape)}  (tables, rows, encoded_dim)")

    rows = []
    for i, meta in enumerate(metadata):
        rows.append({
            "table": i,
            "prior_type": meta["prior_type"],
            "tabicl_active_features": meta["tabicl_active_features"],
            "n_cont": meta["n_cont"],
            "n_cat": meta["n_cat"],
            "target_task": meta["target_task"],
            "cat_cardinalities": " ".join(str(c) for c in meta["cat_cardinalities"]),
            "seq_len": meta["tabicl_seq_len"],
        })
        print(f"table {i:2d}: {meta['prior_type']:16s} raw features={meta['tabicl_active_features']:3d} -> "
              f"{meta['n_cont']:2d} continuous + {meta['n_cat']} categorical (cards {meta['cat_cardinalities']})")
    with (out / "prior_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    import json
    for i, meta in enumerate(metadata):
        np.save(out / f"table_{i}_encoded.npy", X[i].cpu().numpy())
        (out / f"table_{i}_metadata.json").write_text(json.dumps(meta, indent=2))

    for i in range(min(args.n_csv, args.n_tables)):
        comp = decode_components(X[i].numpy(), metadata[i], *schema.as_tuple(), **schema.codec_kwargs())
        df = pd.DataFrame(comp["cont"], columns=[f"x{j}" for j in range(metadata[i]["n_cont"])])
        for j, name in enumerate(metadata[i]["cat_names"]):
            df[name] = comp["cats"][:, j]
        df.to_csv(out / f"table_{i}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist([r["n_cont"] for r in rows], bins=range(0, schema.max_cont + 2), align="left")
    axes[0].set_title("continuous columns per table")
    axes[1].hist([r["n_cat"] for r in rows], bins=range(0, schema.max_cat + 2), align="left")
    axes[1].set_title("categorical columns per table")
    comp = decode_components(X[0].numpy(), metadata[0], *schema.as_tuple(), **schema.codec_kwargs())
    if metadata[0]["n_cont"] >= 2:
        color = comp["cats"][:, int(metadata[0]["label_cat_index"])] if metadata[0]["label_type"] == "categorical" else comp["cont"][:, int(metadata[0]["label_cont_index"])]
        axes[2].scatter(comp["cont"][:, 0], comp["cont"][:, 1], c=color, s=8, cmap="tab10")
        axes[2].set_title("table 0: first two continuous fields, coloured by target")
    fig.tight_layout()
    fig.savefig(out / "prior_overview.png", dpi=150)
    print(f"wrote {out / 'prior_summary.csv'}, {out / 'prior_overview.png'} and {min(args.n_csv, args.n_tables)} CSV tables")


if __name__ == "__main__":
    main()
