#!/usr/bin/env python3
"""Pre-train Gen-TFM on synthetic tables from the TabICL prior.

The defaults use field-level context and generation with the target-rich data
recipe (medium MixSCM prior, batch 8, 768 rows, at least 512 target rows).
The field architecture needs new weights and has not been performance-tuned.
best.pt is selected by fixed held-out validation loss, not training loss.

Example::

    python scripts/train.py --output_dir /scratch/<project>/gen_tfm_runs/my_run --n_iterations 100000

Outputs (in ``output_dir``): ``config.json``, ``checkpoints/best.pt``,
``checkpoints/latest.pt``, append-only ``loss_history.csv``,
``train_loss_steps.npy``, ``train_losses.npy``, ``train_loss.png``,
``train_timing.json``, ``validation_set.pt``, ``validation_history.csv`` and,
unless ``--skip_synthetic_eval``, a synthetic
held-out evaluation (``synthetic_eval/``).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import json
import os
import random
import shutil
import sys
import time
import uuid
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


LOSS_HISTORY_FIELDS = ["session_id", "resume_from_step", "step", "loss", "best_loss", "lr", "time"]


def load_loss_prefix(path: Path, through_step: int):
    """Return the latest recorded loss for every step up to a resume point.

    The append-only CSV may contain several attempts at the same step after a
    restart.  Later rows win, while the original rows remain on disk for audit.
    """
    by_step = {}
    if through_step <= 0 or not path.exists():
        return [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= step <= through_step:
                by_step[step] = loss
    steps = sorted(by_step)
    return steps, [by_step[step] for step in steps]


def open_loss_history(path: Path):
    """Open the loss journal in append mode and write its header if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    f = path.open("a", newline="", buffering=1)
    writer = csv.DictWriter(f, fieldnames=LOSS_HISTORY_FIELDS)
    if needs_header:
        writer.writeheader()
        f.flush()
    return f, writer


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train Gen-TFM on the TabICL synthetic prior")
    # schema (must match the checkpoint you later evaluate)
    p.add_argument("--max_cont_features", type=int, default=32)
    p.add_argument("--min_cont_features", type=int, default=6)
    p.add_argument("--max_cat_features", type=int, default=8)
    p.add_argument("--cat_cardinality", type=int, default=12)
    # model
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_flow_layers", type=int, default=6)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cat_loss_weight", type=float, default=0.8)
    p.add_argument("--mask_loss_weight", type=float, default=0.0)
    p.add_argument("--discrete_flow_weight", type=float, default=0.05)
    p.add_argument("--field_dim", type=int, default=128)
    p.add_argument("--n_col_layers", type=int, default=2)
    p.add_argument("--n_row_layers", type=int, default=2)
    p.add_argument("--num_inducing", type=int, default=16,
                   help="learned inducing points per column-attention block")
    p.add_argument("--num_cls_tokens", type=int, default=4,
                   help="CLS tokens used for field-preserving row interaction")
    # prior (synthetic data engine)
    p.add_argument("--prior_type", choices=["mlp_scm", "tree_scm", "mix_scm"], default="mix_scm")
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
    p.add_argument("--min_ctx", type=int, default=5)
    p.add_argument("--max_ctx", type=int, default=200)
    p.add_argument("--min_target", type=int, default=512)
    p.add_argument("--train_context_sizes", type=int, nargs="+", default=None, help="Optional discrete K grid sampled uniformly")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=10000)
    p.add_argument("--validation_every", type=int, default=1000)
    p.add_argument("--validation_tables", type=int, default=8)
    p.add_argument("--validation_seed", type=int, default=271828)
    p.add_argument("--resume", type=str, default="", help="Path to latest.pt to resume from")
    # synthetic held-out eval
    p.add_argument("--skip_synthetic_eval", action="store_true")
    p.add_argument("--context_sizes", type=int, nargs="+", default=[5, 20, 50, 100, 200])
    p.add_argument("--n_eval_datasets", type=int, default=48)
    p.add_argument("--eval_test_rows", type=int, default=384)
    p.add_argument("--n_gen_rows", type=int, default=384)
    p.add_argument("--n_ode_steps", type=int, default=60)
    # misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1400)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


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
        n_heads=args.n_heads,
        time_dim=args.time_dim, n_flow_layers=args.n_flow_layers, dropout=args.dropout,
        cat_loss_weight=args.cat_loss_weight, mask_loss_weight=args.mask_loss_weight,
        discrete_flow_weight=args.discrete_flow_weight, k_max=args.max_ctx,
        field_dim=args.field_dim,
        n_col_layers=args.n_col_layers, n_row_layers=args.n_row_layers,
        num_inducing=args.num_inducing, num_cls_tokens=args.num_cls_tokens,
    ).to(device)


@contextmanager
def isolated_seed(seed):
    """Validation must not change subsequent training tables, splits or noise."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng():
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def fixed_validation_set(args, schema, resume_checkpoint=None):
    """Persist held-out tables so validation and resumed runs use identical tasks."""
    path = Path(args.output_dir) / "validation_set.pt"
    settings = {
        "schema": schema.as_tuple(), "tables": args.validation_tables,
        "rows": args.rows_per_dataset, "seed": args.validation_seed,
        "min_ctx": args.min_ctx, "max_ctx": args.max_ctx, "min_target": args.min_target,
        "context_sizes": args.train_context_sizes,
    }
    source = path
    if args.resume and not source.exists():
        source = Path(args.resume).resolve().parent.parent / "validation_set.pt"
    if source.exists():
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if saved["settings"] != settings:
            raise ValueError("Validation settings changed; use the original settings for comparable checkpoint selection")
    else:
        if resume_checkpoint and resume_checkpoint.get("train_config", {}).get("checkpoint_selection") == "fixed_validation":
            raise FileNotFoundError(f"Resuming validation-selected weights requires {source}")
        with isolated_seed(args.validation_seed):
            validation_prior = build_prior(args, schema, output_device="cpu")
            tables, metadata = validation_prior.sample_batch(
                args.validation_tables, args.rows_per_dataset, return_metadata=True, split="test"
            )
        saved = {"settings": settings, "tables": tables.cpu(), "metadata": metadata}
    if not path.exists():
        torch.save(saved, path)
    return saved["tables"], saved["metadata"]


@torch.no_grad()
def validation_loss(model, tables, metadata, args, device):
    """Equal table/K weighting with fixed splits, times and noise on every evaluation."""
    high = min(args.max_ctx, tables.shape[1] - args.min_target)
    sizes = args.train_context_sizes or [args.min_ctx, (args.min_ctx + high) // 2, high]
    sizes = sorted({int(k) for k in sizes if args.min_ctx <= k <= high})
    if not sizes:
        raise ValueError("No validation context size fits the context/target split")
    was_training = model.training
    model.eval()
    losses = []
    try:
        with isolated_seed(args.validation_seed + 1):
            for i, meta in enumerate(metadata):
                x = tables[i:i + 1].to(device)
                for k in sizes:
                    loss = model.compute_loss(x, metadata=[meta], min_ctx=k, max_ctx=k,
                                              min_target=args.min_target, normalize=True)
                    losses.append(float(loss))
    finally:
        model.train(was_training)
    return float(np.mean(losses))


def train(args, device):
    if args.validation_every < 1 or args.validation_tables < 1:
        raise ValueError("validation_every and validation_tables must be positive")
    schema = Schema(args.max_cont_features, args.max_cat_features, args.cat_cardinality)
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

    start_step, best_loss = 0, float("inf")
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step, best_loss = int(ckpt["step"]), float(ckpt.get("best_loss", float("inf")))
        if ckpt.get("train_config", {}).get("checkpoint_selection") != "fixed_validation":
            best_loss = float("inf")
            log_event("selection_reset", reason="legacy checkpoint selected by training loss")
        elif not best_path.exists():
            previous_best = Path(args.resume).resolve().parent / "best.pt"
            if previous_best.exists():
                shutil.copy2(previous_best, best_path)
            else:
                best_loss = float("inf")
                log_event("selection_reset", reason="previous best weights unavailable")
        log_event("resumed", path=args.resume, step=start_step, best_loss=best_loss)

    validation_tables, validation_metadata = fixed_validation_set(args, schema, ckpt)
    log_event("validation_ready", tables=len(validation_metadata), path=str(Path(args.output_dir) / "validation_set.pt"))

    loss_history_path = Path(args.output_dir) / "loss_history.csv"
    loss_steps, losses = load_loss_prefix(loss_history_path, start_step)
    session_id = f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    loss_file, loss_writer = open_loss_history(loss_history_path)
    log_event("loss_history_opened", path=str(loss_history_path), session_id=session_id,
              resume_from_step=start_step, recovered_points=len(losses))

    train_config = {**vars(args), "checkpoint_selection": "fixed_validation"}
    train_start = time.perf_counter()
    interval_start = train_start
    model.train()
    try:
        for step in range(start_step, args.n_iterations):
            data_start = time.perf_counter()
            X_batch, metadata = prior.sample_batch(args.batch_size, args.rows_per_dataset, return_metadata=True, split="train")
            feature_mask = batch_feature_mask(metadata, *schema.as_tuple(), device)
            data_time = time.perf_counter() - data_start

            compute_start = time.perf_counter()
            loss = model.compute_loss(
                X_batch, metadata=metadata, feature_mask=feature_mask, min_ctx=args.min_ctx, max_ctx=args.max_ctx,
                min_target=args.min_target, normalize=True, context_sizes=args.train_context_sizes,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            compute_time = time.perf_counter() - compute_start

            completed_step = step + 1
            loss_val = float(loss.item())
            if (step == start_step or completed_step % args.validation_every == 0
                    or completed_step == args.n_iterations):
                val_loss = validation_loss(model, validation_tables, validation_metadata, args, device)
                if not np.isfinite(val_loss):
                    raise FloatingPointError(f"Non-finite validation loss at step {completed_step}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_training_checkpoint(best_path, completed_step, model, optimizer, scheduler, best_loss, train_config)
                val_path = Path(args.output_dir) / "validation_history.csv"
                needs_header = not val_path.exists() or val_path.stat().st_size == 0
                with val_path.open("a", newline="") as val_file:
                    writer = csv.writer(val_file)
                    if needs_header:
                        writer.writerow(["step", "validation_loss", "best_validation_loss"])
                    writer.writerow([completed_step, val_loss, best_loss])
                log_event("validation", step=completed_step, validation_loss=val_loss, best_validation_loss=best_loss)

            current_lr = scheduler.get_last_lr()[0]
            loss_writer.writerow({
                "session_id": session_id,
                "resume_from_step": start_step,
                "step": completed_step,
                "loss": repr(loss_val),
                "best_loss": repr(best_loss),
                "lr": repr(current_lr),
                "time": dt.datetime.now().isoformat(timespec="seconds"),
            })
            # Make every completed step visible even if the process is interrupted.
            loss_file.flush()
            loss_steps.append(completed_step)
            losses.append(loss_val)

            if completed_step % args.log_every == 0 or step == start_step:
                now = time.perf_counter()
                log_event("train_step", step=completed_step, total_steps=args.n_iterations, loss=loss_val,
                          best_loss=best_loss, lr=current_lr, elapsed_min=(now - train_start) / 60.0,
                          sec_per_step=(now - interval_start) / max(1, args.log_every if step > start_step else 1),
                          data_time_sec=data_time, compute_time_sec=compute_time)
                interval_start = now
            if completed_step % args.checkpoint_every == 0:
                save_training_checkpoint(latest_path, completed_step, model, optimizer, scheduler, best_loss, train_config)
                log_event("checkpoint_saved", step=completed_step, path=str(latest_path))
    finally:
        loss_file.close()

    save_training_checkpoint(latest_path, args.n_iterations, model, optimizer, scheduler, best_loss, train_config)
    total_sec = time.perf_counter() - train_start
    np.save(Path(args.output_dir) / "train_loss_steps.npy", np.asarray(loss_steps, dtype=np.int64))
    np.save(Path(args.output_dir) / "train_losses.npy", np.asarray(losses, dtype=np.float32))
    timing = {"train_total_sec": total_sec, "train_total_min": total_sec / 60.0, "steps": args.n_iterations,
              "avg_sec_per_step": total_sec / max(args.n_iterations - start_step, 1), "best_loss": best_loss,
              "synthetic_tables_seen": args.n_iterations * args.batch_size}
    (Path(args.output_dir) / "train_timing.json").write_text(json.dumps(timing, indent=2))
    log_event("train_complete", **timing)

    # reload best weights and export a slim, self-describing checkpoint for evaluation / sharing
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        extra = {"train_config": train_config}
        slim = export_slim_checkpoint(best_path, Path(args.output_dir) / "gen_tfm_best_slim.pt", model_config=model.config(),
                                      extra=extra)
        log_event("slim_checkpoint_exported", path=str(slim))
    return model, loss_steps, losses, prior


def plot_training_curve(loss_steps, losses, output_dir):
    if not losses:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(loss_steps, losses, linewidth=0.5, alpha=0.35, label="step")
    window = min(200, max(5, len(losses) // 20))
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(loss_steps[window - 1:], smooth, linewidth=2, label=f"smooth {window}")
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
    from gen_tfm.metrics import evaluate_mixed

    schema = Schema(args.max_cont_features, args.max_cat_features, args.cat_cardinality)
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
                candidates[name] = fn(X_ctx, meta, *schema.as_tuple(), args.n_gen_rows, np.random.default_rng(rng.integers(0, 2**31 - 1)))
            for method, X_cand in candidates.items():
                metrics = evaluate_mixed(X_ctx, X_cand, X_test, meta, *schema.as_tuple(), seed=seed)
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
    model, loss_steps, losses, prior = train(args, device)
    plot_training_curve(loss_steps, losses, args.output_dir)
    if not args.skip_synthetic_eval:
        synthetic_eval(model, prior, args, device)
    log_event("run_complete", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
