#!/usr/bin/env python3
"""Pre-train Gen-TFM on synthetic tables from the TabICL prior.

The v1.2 defaults use binary categories, noisy-query memory, K=Q=200,
768 prior rows, batch 8 and 100k steps. Unused rows do not enter the loss.
A full run takes about 15 h on one LUMI MI250x GCD; ``--n_iterations 2000``
is a good smoke test (~20 min on GPU).

Example::

    python scripts/train.py --output_dir /scratch/<project>/gen_tfm_runs/my_run --n_iterations 100000

Outputs (in ``output_dir``): ``config.json``, ``checkpoints/best.pt``,
``checkpoints/latest.pt``, ``train_losses.npy``, ``train_loss.png``,
``train_timing.json`` and, unless ``--skip_synthetic_eval``, a synthetic
held-out evaluation (``synthetic_eval/``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_tfm.checkpoint import export_slim_checkpoint, save_training_checkpoint  # noqa: E402
from gen_tfm.encoding import Schema, batch_feature_mask  # noqa: E402
from gen_tfm.model import GenTFM  # noqa: E402
from gen_tfm.prior import PriorConfig, TabICLPriorEngine  # noqa: E402


def log_event(event: str, **payload):
    record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "event": event}
    record.update(payload)
    print(json.dumps(record, sort_keys=True), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train Gen-TFM on the TabICL synthetic prior")
    p.add_argument("--config", type=str, default="", help="JSON training defaults; explicit CLI flags override")
    # schema (must match the checkpoint you later evaluate)
    p.add_argument("--max_cont_features", type=int, default=32)
    p.add_argument("--min_cont_features", type=int, default=6)
    p.add_argument("--max_cat_features", type=int, default=8)
    p.add_argument("--cat_cardinality", type=int, default=12)
    p.add_argument("--cat_encoding", choices=["onehot", "binary"], default="binary")
    p.add_argument("--binary_bit_order", choices=["msb_first"], default="msb_first")
    p.add_argument("--query_conditioning", choices=["context_only", "context_plus_noisy_query"], default="context_plus_noisy_query")
    # model
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_enc_layers", type=int, default=4)
    p.add_argument("--n_flow_layers", type=int, default=6)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cat_loss_weight", type=float, default=0.8)
    p.add_argument("--mask_loss_weight", type=float, default=0.0)
    p.add_argument("--discrete_flow_weight", type=float, default=0.05)
    # prior (synthetic data engine)
    p.add_argument("--prior_type", choices=["mlp_scm", "tree_scm", "mix_scm"], default="mix_scm")
    p.add_argument("--target_task", choices=["classification", "regression"], default="classification",
                   help="Whether TabICL prior tables use a categorical or continuous supervised y column")
    p.add_argument("--prior_min_features", type=int, default=8)
    p.add_argument("--prior_max_features", type=int, default=48)
    p.add_argument("--prior_max_classes", type=int, default=12)
    p.add_argument("--prior_force_feature_cats", type=int, default=6)
    p.add_argument("--prior_feature_cat_min_card", type=int, default=2)
    p.add_argument("--prior_feature_cat_max_card", type=int, default=12)
    p.add_argument("--prior_train_min_seq_len", type=int, default=712)
    p.add_argument("--prior_no_log_seq_len", action="store_true")
    p.add_argument("--prior_device", type=str, default="cpu")
    # training / context split
    p.add_argument("--n_iterations", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--rows_per_dataset", type=int, default=768)
    p.add_argument("--min_ctx", type=int, default=200)
    p.add_argument("--max_ctx", type=int, default=200)
    p.add_argument("--min_target", type=int, default=200)
    p.add_argument("--train_context_sizes", type=int, nargs="+", default=None, help="Optional discrete K grid sampled uniformly")
    p.add_argument("--num_query", type=int, default=200, help="Exact Q; 0 selects legacy Q=N-K")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=10000)
    p.add_argument("--resume", type=str, default="", help="Path to latest.pt to resume from")
    # synthetic held-out eval
    p.add_argument("--skip_synthetic_eval", action="store_true")
    p.add_argument("--eval_tabicl", action="store_true", help="Also compute TabICL TSTR on held-out prior tables")
    p.add_argument("--context_sizes", type=int, nargs="+", default=[5, 20, 50, 100, 200])
    p.add_argument("--n_eval_datasets", type=int, default=48)
    p.add_argument("--eval_test_rows", type=int, default=384)
    p.add_argument("--n_gen_rows", type=int, default=384)
    p.add_argument("--n_ode_steps", type=int, default=60)
    # misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1400)
    p.add_argument("--output_dir", type=str, default=None)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    preliminary, _ = pre.parse_known_args()
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        actions = {a.dest: a for a in p._actions}
        if not isinstance(config, dict) or set(config) - set(actions):
            p.error("Unknown arguments in config JSON")
        for key, value in config.items():
            action = actions[key]
            if action.choices is not None and value not in action.choices:
                p.error(f"Invalid config choice for {key}: {value}")
        p.set_defaults(**config)
    args = p.parse_args()
    if not args.output_dir:
        p.error("--output_dir is required (or provide it in --config)")
    if args.num_query < 0 or args.min_ctx < 1 or args.max_ctx < args.min_ctx or args.min_target < 1:
        p.error("Invalid K/Q configuration")
    required_q = args.num_query or args.min_target
    if args.rows_per_dataset < args.max_ctx + required_q:
        p.error("rows_per_dataset must be >= max_ctx + Q")
    if args.prior_train_min_seq_len > args.rows_per_dataset:
        p.error("prior_train_min_seq_len exceeds rows_per_dataset")
    if args.n_iterations < 1 or args.batch_size < 1:
        p.error("n_iterations and batch_size must be positive")
    return args


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_prior(args, schema: Schema, output_device: str) -> TabICLPriorEngine:
    config = PriorConfig(
        prior_type=args.prior_type,
        target_task=args.target_task,
        min_features=args.prior_min_features,
        max_features=args.prior_max_features,
        max_classes=args.prior_max_classes,
        force_feature_cats=args.prior_force_feature_cats,
        feature_cat_min_card=args.prior_feature_cat_min_card,
        feature_cat_max_card=args.prior_feature_cat_max_card,
        train_min_seq_len=args.prior_train_min_seq_len,
        log_seq_len=not args.prior_no_log_seq_len,
        device=args.prior_device,
    )
    return TabICLPriorEngine(schema=schema, config=config, min_cont=args.min_cont_features, output_device=output_device)


def build_model(args, device) -> GenTFM:
    return GenTFM(
        max_cont=args.max_cont_features, max_cat=args.max_cat_features, cat_cardinality=args.cat_cardinality,
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, n_enc_layers=args.n_enc_layers,
        time_dim=args.time_dim, n_flow_layers=args.n_flow_layers, dropout=args.dropout,
        cat_loss_weight=args.cat_loss_weight, mask_loss_weight=args.mask_loss_weight,
        discrete_flow_weight=args.discrete_flow_weight, k_max=args.max_ctx,
        cat_encoding=args.cat_encoding, binary_bit_order=args.binary_bit_order,
        query_conditioning=args.query_conditioning,
    ).to(device)


def train(args, device):
    schema = Schema(args.max_cont_features, args.max_cat_features, args.cat_cardinality, args.cat_encoding, args.binary_bit_order)
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_event("model_created", n_params=n_params, encoded_dim=model.max_features)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler(optimizer, args.warmup_steps, args.n_iterations)
    prior = build_prior(args, schema, output_device=str(device))
    log_event("prior_created", **prior.config.to_dict())

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    start_step, best_loss, losses = 0, float("inf"), []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        stored_config = dict(ckpt.get("model_config") or {})
        stored_config.setdefault("cat_encoding", "onehot")
        stored_config.setdefault("binary_bit_order", "msb_first")
        stored_config.setdefault("query_conditioning", "context_only")
        if stored_config != model.config():
            # Old configs lack schema_version; add its codec-derived value before checking.
            stored_config.setdefault("schema_version", f"mixed_{stored_config['cat_encoding']}_v1")
        if stored_config != model.config():
            raise ValueError("Resume model config differs; use the original codec/query/model configuration")
        previous = ckpt.get("train_config", {})
        for key in ("target_task", "num_query", "min_ctx", "max_ctx", "rows_per_dataset"):
            old_value = previous.get(key, 0 if key == "num_query" else getattr(args, key))
            if old_value != getattr(args,key):
                raise ValueError(f"Resume training config differs: {key}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step, best_loss = int(ckpt["step"]), float(ckpt.get("best_loss", float("inf")))
        log_event("resumed", path=args.resume, step=start_step, best_loss=best_loss)

    train_config = vars(args)
    train_start = time.perf_counter()
    interval_start = train_start
    model.train()
    for step in range(start_step, args.n_iterations):
        data_start = time.perf_counter()
        X_batch, metadata = prior.sample_batch(args.batch_size, args.rows_per_dataset, return_metadata=True, split="train")
        feature_mask = batch_feature_mask(metadata, *schema.as_tuple(), device=device, **schema.codec_kwargs())
        data_time = time.perf_counter() - data_start

        compute_start = time.perf_counter()
        loss = model.compute_loss(
            X_batch, metadata=metadata, feature_mask=feature_mask, min_ctx=args.min_ctx, max_ctx=args.max_ctx,
            min_target=args.min_target, normalize=True, context_sizes=args.train_context_sizes, num_query=args.num_query or None,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        compute_time = time.perf_counter() - compute_start

        loss_val = float(loss.item())
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            save_training_checkpoint(best_path, step + 1, model, optimizer, scheduler, best_loss, train_config)

        if (step + 1) % args.log_every == 0 or step == start_step:
            now = time.perf_counter()
            log_event("train_step", step=step + 1, total_steps=args.n_iterations, loss=loss_val, best_loss=best_loss,
                      lr=scheduler.get_last_lr()[0], elapsed_min=(now - train_start) / 60.0,
                      sec_per_step=(now - interval_start) / max(1, args.log_every if step > start_step else 1),
                      data_time_sec=data_time, compute_time_sec=compute_time, **model.last_loss_details)
            interval_start = now
        if (step + 1) % args.checkpoint_every == 0:
            save_training_checkpoint(latest_path, step + 1, model, optimizer, scheduler, best_loss, train_config)
            log_event("checkpoint_saved", step=step + 1, path=str(latest_path))

    save_training_checkpoint(latest_path, args.n_iterations, model, optimizer, scheduler, best_loss, train_config)
    total_sec = time.perf_counter() - train_start
    np.save(Path(args.output_dir) / "train_losses.npy", np.asarray(losses, dtype=np.float32))
    timing = {"train_total_sec": total_sec, "train_total_min": total_sec / 60.0, "steps": args.n_iterations,
              "avg_sec_per_step": total_sec / max(args.n_iterations - start_step, 1), "best_loss": best_loss,
              "synthetic_tables_seen": args.n_iterations * args.batch_size}
    (Path(args.output_dir) / "train_timing.json").write_text(json.dumps(timing, indent=2))
    log_event("train_complete", **timing)

    # reload best weights and export a slim, self-describing checkpoint for evaluation / sharing
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if ckpt.get("model_config") != model.config():
            raise ValueError("Best checkpoint configuration differs from this run")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        slim = export_slim_checkpoint(best_path, Path(args.output_dir) / "gen_tfm_best_slim.pt", model_config=model.config(),
                                      extra={"train_config": train_config, "recommended_calibration": {"alpha": 3.5, "schedule": "logk", "tau": 0.25}})
        log_event("slim_checkpoint_exported", path=str(slim))
    return model, losses, prior


def plot_training_curve(losses, output_dir):
    if not losses:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(losses, linewidth=0.5, alpha=0.35, label="step")
    window = min(200, max(5, len(losses) // 20))
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(losses)), smooth, linewidth=2, label=f"smooth {window}")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("loss (CFM + 0.8 * categorical CE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "train_loss.png", dpi=160)
    plt.close(fig)


def synthetic_eval(model, prior, args, device):
    """Held-out synthetic tables: how well does the model fit *new* draws from the prior?"""
    from gen_tfm.baselines import MIXED_BASELINES
    from gen_tfm.generation import DEFAULT_CALIBRATION, generate_in_context, generate_zero_context
    from gen_tfm.metrics import evaluate_mixed, downstream_tabicl
    from gen_tfm.encoding import label_info

    schema = Schema(args.max_cont_features, args.max_cat_features, args.cat_cardinality, args.cat_encoding, args.binary_bit_order)
    out_dir = Path(args.output_dir) / "synthetic_eval"
    out_dir.mkdir(exist_ok=True)
    model.eval()
    rows = []
    for k in args.context_sizes:
        total_rows = k + args.eval_test_rows + args.n_gen_rows
        for i in range(args.n_eval_datasets):
            seed = args.seed + 10000 + i + k * 17
            np.random.seed(seed)
            torch.manual_seed(seed)
            X_batch, meta_batch = prior.sample_batch(1, total_rows, return_metadata=True, split="test")
            X_full, meta = X_batch[0].cpu().numpy(), meta_batch[0]
            X_ctx = X_full[:k]
            X_test = X_full[k : k + args.eval_test_rows]
            X_ref = X_full[k + args.eval_test_rows :]
            rng = np.random.default_rng(seed)
            candidates = {
                "gen_tfm_matched": generate_in_context(model, X_ctx, meta, args.n_gen_rows, schema, n_steps=args.n_ode_steps, calibration=None, device=device),
                "gen_tfm_calibrated": generate_in_context(model, X_ctx, meta, args.n_gen_rows, schema, n_steps=args.n_ode_steps, calibration=DEFAULT_CALIBRATION, device=device),
                "gen_tfm_zero": generate_zero_context(model, X_ctx, meta, args.n_gen_rows, schema, n_steps=args.n_ode_steps, device=device),
                "real_vs_real": X_ref,
            }
            for name, fn in MIXED_BASELINES.items():
                candidates[name] = fn(X_ctx, meta, *schema.as_tuple(), args.n_gen_rows, np.random.default_rng(rng.integers(0, 2**31 - 1)), **schema.codec_kwargs())
            for method, X_cand in candidates.items():
                metrics = evaluate_mixed(X_ctx, X_cand, X_test, meta, *schema.as_tuple(), seed=seed, **schema.codec_kwargs())
                if args.eval_tabicl:
                    key = "downstream_accuracy" if label_info(meta)[0] == "categorical" else "downstream_r2"
                    metrics[key] = downstream_tabicl(X_cand, X_test, meta, *schema.as_tuple(), **schema.codec_kwargs())
                rows.append({"context_size": k, "dataset_id": i, "method": method, "n_cont": meta["n_cont"], "n_cat": meta["n_cat"], **metrics})
        for method in sorted({r["method"] for r in rows}):
            vals = [r["encoded_mmd"] for r in rows if r["context_size"] == k and r["method"] == method]
            log_event("synthetic_eval_summary", context_size=k, method=method, encoded_mmd=float(np.mean(vals)))
    import csv

    with (out_dir / "eval_per_dataset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log_event("synthetic_eval_complete", path=str(out_dir / "eval_per_dataset.csv"))


def main():
    args = parse_args()
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "config.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_event("run_init", output_dir=args.output_dir, device=str(device), seed=args.seed)
    model, losses, prior = train(args, device)
    plot_training_curve(losses, args.output_dir)
    if not args.skip_synthetic_eval:
        synthetic_eval(model, prior, args, device)
    log_event("run_complete", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
