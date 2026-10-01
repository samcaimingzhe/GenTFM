"""Train the conditional GenTFM on TabICL synthetic tables.

Run from the project root: python -m script.train --steps 1000
Or run directly from the project root: python script/train.py --steps 1000
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Direct execution adds the project root so sibling packages can be imported.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GenTFM
from training.checkpoint import (capture_rng_state, copy_best_checkpoint, load_training_checkpoint,
                                 restore_rng_state, save_training_checkpoint)
from training.plotting import save_loss_curve
from data.encoding import Schema, batch_feature_mask, categorical_feature_mask, velocity_feature_mask
from training.flow_matching import (flow_matching_loss, masked_velocity_loss,
                                    sample_conditional_flow_batch, velocity_weights)
from data.prior import PriorConfig, TabICLPriorEngine


def build_training_batch(prior, schema, batch_size, num_rows, device, split="train"):
    tables, metadata = prior.sample_batch(
        batch_size, num_rows, return_metadata=True, split=split,
    )
    if tables.shape != (batch_size, num_rows, schema.encoded_dim):
        raise ValueError(f"prior returned {tuple(tables.shape)}; expected {(batch_size, num_rows, schema.encoded_dim)}")
    mask = batch_feature_mask(metadata, *schema.as_tuple(), device=device)
    return tables.to(device), mask


def model_schema(model):
    """Mixed schema, or None for a numerical-only conditional model."""
    if model.max_cont is None:
        return None
    return Schema(model.max_cont, model.max_cat, model.cat_cardinality)


def train_step(model, optimizer, tables, feature_mask, max_grad_norm=1.0,
               categorical_weight=0.8, *, return_metrics=False,
               discrete_flow_weight=0.05, min_context=1, max_context=500,
               min_target=1, context_sizes=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    schema = model_schema(model)
    x_t, t, target, _, clean_target, context = sample_conditional_flow_batch(
        tables, feature_mask, schema=schema, min_context=min_context,
        max_context=max_context, min_target=min_target, context_sizes=context_sizes,
    )
    if schema is None:
        mse = masked_velocity_loss(model(x_t, t, feature_mask, context), target, feature_mask)
        metrics = {"loss": mse, "velocity_mse": mse, "categorical_ce": mse.new_zeros(())}
    else:
        metrics = flow_matching_loss(model(x_t, t, feature_mask, context, return_aux=True),
                                     target, clean_target, feature_mask, schema, categorical_weight, discrete_flow_weight)
    loss = metrics["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if return_metrics:
        return {name: value.detach() for name, value in metrics.items()}
    return loss.detach()


@torch.no_grad()
def validate(model, validation_batches, categorical_weight=0.8, *, return_metrics=False, discrete_flow_weight=0.05):
    """Aggregate MSE and CE by their respective valid cell counts.

    Batches contain (x_t, t, target_velocity, mask, clean_target, context).
    Context is fixed and excluded from both MSE and CE.
    """
    modes = [(module, module.training) for module in model.modules()]
    total_error, total_cells = 0.0, 0
    total_ce, total_cat_cells = 0.0, 0
    schema = model_schema(model)
    model.eval()
    try:
        for batch in validation_batches:
            if len(batch) != 6:
                raise ValueError("conditional validation requires clean targets and context")
            x_t, t, target, mask, clean_target, context = batch
            if schema is None:
                loss = masked_velocity_loss(model(x_t, t, mask, context), target, mask)
                mse, ce = loss, loss.new_zeros(())
                count = int(mask.sum()) * x_t.shape[1]
                cat_count = 0
            else:
                metrics = flow_matching_loss(model(x_t, t, mask, context, return_aux=True),
                                             target, clean_target, mask, schema, categorical_weight, discrete_flow_weight)
                loss, mse, ce = metrics["loss"], metrics["velocity_mse"], metrics["categorical_ce"]
                count = float(velocity_weights(mask, schema, discrete_flow_weight).sum()) * x_t.shape[1]
                cat_count = int(categorical_feature_mask(mask, schema).any(-1).sum()) * x_t.shape[1]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total_error += float(mse) * count
            total_cells += count
            total_ce += float(ce) * cat_count
            total_cat_cells += cat_count
    finally:
        for module, training in modes:
            module.training = training
    if total_cells == 0 and total_cat_cells == 0:
        raise ValueError("validation requires at least one valid cell")
    mse = total_error / max(total_cells, 1e-12)
    ce = total_ce / max(total_cat_cells, 1)
    metrics = {"loss": mse + categorical_weight * ce, "velocity_mse": mse, "categorical_ce": ce}
    return metrics if return_metrics else metrics["loss"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--steps", type=int, default=1000, help="Total planned optimizer steps, including completed steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rows", type=int, default=1024, help="Total context + target rows per prior table")
    parser.add_argument("--min-context", type=int, default=5)
    parser.add_argument("--max-context", type=int, default=500)
    parser.add_argument("--min-target", type=int, default=512)
    parser.add_argument("--context-sizes", type=int, nargs="+", help="Optional fixed set of training context sizes")
    parser.add_argument("--num-cross-blocks", type=int, default=2)
    parser.add_argument("--discrete-flow-weight", type=float, default=0.05, help="One-hot velocity weight relative to numerical coordinates")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-col-blocks", type=int, default=2)
    parser.add_argument("--num-row-blocks", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--num-inds", type=int, default=16)
    parser.add_argument("--max-cont", type=int, default=32)
    parser.add_argument("--max-cat", type=int, default=8)
    parser.add_argument("--cat-cardinality", type=int, default=12)
    parser.add_argument("--prior-type", choices=("mlp_scm", "tree_scm", "mix_scm"), default="mix_scm")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--categorical-loss-weight", type=float, default=0.8,
                        help="Weight of clean-category CE in MSE + weight * CE")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, help="Resume a full training checkpoint; inherit its training configuration")
    parser.add_argument("--save-every", type=int, default=100, help="Save latest.pt every N completed steps")
    args = parser.parse_args()
    args._resume_checkpoint = None
    if args.resume is not None:
        try:
            checkpoint = load_training_checkpoint(args.resume)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        explicit = {token.split('=', 1)[0][2:].replace('-', '_')
                    for token in sys.argv[1:] if token.startswith('--')}
        adjustable = {"resume", "device", "output_dir", "log_every", "val_every", "save_every"}
        for name, value in checkpoint['train_config'].items():
            if not hasattr(args, name) or name.startswith('_') or name == 'resume':
                continue
            if name in explicit:
                if name not in adjustable and getattr(args, name) != value:
                    parser.error(f"--{name.replace('_', '-')} must match checkpoint value {value!r}; resume preserves the original training plan")
            else:
                setattr(args, name, Path(value) if name == 'output_dir' else value)
        if 'output_dir' not in explicit:
            args.output_dir = args.resume.resolve().parent
        args._resume_checkpoint = checkpoint
    for name in ("steps", "batch_size", "num_rows", "val_every", "val_batches", "log_every", "save_every", "min_context", "max_context", "min_target", "num_cross_blocks"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_cont < 1 or args.max_cat < 1 or args.cat_cardinality < 2:
        parser.error("schema requires max-cont >= 1, max-cat >= 1, cat-cardinality >= 2")
    if args.lr <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        parser.error("lr and max-grad-norm must be positive; weight-decay must be nonnegative")
    if not 0 <= args.categorical_loss_weight < float("inf"):
        parser.error("categorical-loss-weight must be finite and nonnegative")
    if args.max_context < args.min_context or args.num_rows < args.max_context + args.min_target:
        parser.error("num-rows must cover max-context + min-target; context limits must be ordered")
    if args.context_sizes and any(k < args.min_context or k > args.max_context for k in args.context_sizes):
        parser.error("context-sizes must fall between min-context and max-context")
    if not 0 <= args.discrete_flow_weight < float("inf"):
        parser.error("discrete-flow-weight must be finite and nonnegative")
    return args


def snapshot_training_state(prior, validation_batches, train_history, validation_history):
    """Store validation tensors on CPU and preserve prior cache construction order."""
    requests = [key[:3] for key in getattr(prior, "_prior_cache", {})]
    return {
        "rng": capture_rng_state(),
        "prior": {"generated_batches": getattr(prior, "generated_batches", 0),
                  "cached_requests": requests},
        "validation_batches": [tuple(t.detach().cpu() for t in batch) for batch in validation_batches],
        "train_history": list(train_history),
        "validation_history": list(validation_history),
    }


def restore_prior_state(prior, state):
    # Construct cached TabICL datasets before restoring global RNGs. Replay is
    # disabled by this trainer, so the adapter counter is its persistent state.
    for batch_size, num_rows, split in state.get("cached_requests", []):
        prior._new_prior(batch_size, num_rows, None if split == "none" else split)
    prior.generated_batches = state.get("generated_batches", 0)


def main():
    args = parse_args()
    checkpoint = args._resume_checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    schema = Schema(args.max_cont, args.max_cat, args.cat_cardinality)
    prior_config = PriorConfig(**checkpoint["train_config"]["prior"]) if checkpoint and "prior" in checkpoint["train_config"] else PriorConfig(
        prior_type=args.prior_type, train_min_seq_len=0,
        log_seq_len=False, replay_small=False,
    )
    if checkpoint:
        start_step = int(checkpoint["step"])
        if start_step < 0 or start_step > args.steps:
            raise ValueError("checkpoint step must be between zero and the planned total steps")
        if int(checkpoint["scheduler_state_dict"]["T_max"]) != args.steps:
            raise ValueError("--steps must match the original cosine scheduler T_max")
        model = GenTFM(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        start_step = 0
        model = GenTFM(
            max_features=schema.encoded_dim, embed_dim=args.embed_dim,
            num_col_blocks=args.num_col_blocks, num_row_blocks=args.num_row_blocks,
            nhead=args.nhead, dim_feedforward=args.dim_feedforward, num_inds=args.num_inds,
            max_cont=schema.max_cont, max_cat=schema.max_cat, cat_cardinality=schema.cat_cardinality,
            num_cross_blocks=args.num_cross_blocks,
        ).to(device)
    prior = TabICLPriorEngine(schema, prior_config, output_device=str(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    state = checkpoint.get("training_state") if checkpoint else None
    best_loss = float(checkpoint["best_loss"]) if checkpoint else float("inf")
    train_history = list(state["train_history"]) if state else []
    validation_history = list(state["validation_history"]) if state else []
    if state:
        validation_batches = [tuple(t.to(device) for t in batch) for batch in state["validation_batches"]]
        restore_prior_state(prior, state["prior"])
        restore_rng_state(state["rng"])
    else:
        validation_batches = []
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        flow_schema = model_schema(model)
        for _ in range(args.val_batches):
            tables, mask = build_training_batch(
                prior, schema, args.batch_size, args.num_rows, device, split="validation",
            )
            validation_batches.append(sample_conditional_flow_batch(
                tables, mask, generator=generator, schema=flow_schema,
                min_context=args.min_context, max_context=args.max_context,
                min_target=args.min_target, context_sizes=args.context_sizes,
            ))
        if checkpoint:
            # Old full checkpoints lack validation/RNG history. Resume weights,
            # optimizer and schedule, but start a new validation comparison.
            best_loss = float("inf")
            prior.generated_batches = args.val_batches + start_step
            print("Legacy checkpoint: RNG, validation batches and loss history were not saved. "
                  "Validation has been regenerated; best loss is reset.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint and state and args.output_dir.resolve() != args.resume.resolve().parent:
        if not copy_best_checkpoint(args.resume, args.output_dir, best_loss, checkpoint["model_config"]):
            best_loss = float("inf")
            print("Prior best weights are unavailable in the new output directory; best loss is reset.", flush=True)
    train_config = {key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items() if not key.startswith('_')}
    train_config["prior"] = prior_config.to_dict()
    last_saved_step = start_step
    if checkpoint:
        print(f"Resumed {args.resume}: completed_step={start_step} next_step={start_step + 1} "
              f"total_steps={args.steps} lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
    # Establish latest.pt even before the first update, or when resuming into a
    # new output directory. Never save partially completed optimizer updates.
    def save_at(path, step):
        save_training_checkpoint(
            path, step, model, optimizer, scheduler, best_loss, train_config,
            training_state=snapshot_training_state(prior, validation_batches, train_history, validation_history),
        )
    save_at(args.output_dir / "latest.pt", start_step)
    try:
        for step in range(start_step + 1, args.steps + 1):
            tables, mask = build_training_batch(prior, schema, args.batch_size, args.num_rows, device)
            metrics = train_step(model, optimizer, tables, mask, args.max_grad_norm,
                                 args.categorical_loss_weight, return_metrics=True,
                                 discrete_flow_weight=args.discrete_flow_weight,
                                 min_context=args.min_context, max_context=args.max_context,
                                 min_target=args.min_target, context_sizes=args.context_sizes)
            loss = metrics["loss"]
            scheduler.step()
            train_history.append((step, float(loss)))
            if step == 1 or step % args.log_every == 0:
                print(f"step={step} train_loss={float(loss):.6f} "
                      f"velocity_mse={float(metrics['velocity_mse']):.6f} "
                      f"categorical_ce={float(metrics['categorical_ce']):.6f}", flush=True)
            validation_due = step % args.val_every == 0 or step == args.steps
            if validation_due:
                val_metrics = validate(model, validation_batches, args.categorical_loss_weight, return_metrics=True,
                                       discrete_flow_weight=args.discrete_flow_weight)
                val_loss = val_metrics["loss"]
                validation_history.append((step, val_loss))
                print(f"step={step} validation_loss={val_loss:.6f} "
                      f"velocity_mse={val_metrics['velocity_mse']:.6f} "
                      f"categorical_ce={val_metrics['categorical_ce']:.6f}", flush=True)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_at(args.output_dir / "best.pt", step)
            if step == 1 or step % args.save_every == 0 or validation_due:
                save_at(args.output_dir / "latest.pt", step)
                last_saved_step = step
    except KeyboardInterrupt:
        print(f"Training interrupted. Latest completed checkpoint is step={last_saved_step}: "
              f"{args.output_dir / 'latest.pt'}. Unsaved steps will be repeated on resume.", flush=True)
    if train_history:
        curve_path = save_loss_curve(
            train_history, validation_history, args.output_dir / "loss_curve.png",
            ylabel="Total loss (velocity MSE + weighted categorical CE)",
        )
        print(f"Loss curve saved to {curve_path}", flush=True)
    elif start_step == args.steps:
        print("Training already reached the planned total steps.", flush=True)


if __name__ == "__main__":
    main()
