"""Train the unconditional GenTFM on TabICL synthetic tables.

Run from the project root: python -m script.train --steps 1000
Or run directly from the project root: python script/train.py --steps 1000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Direct execution adds the project root so sibling packages can be imported.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import GenTFM
from training.checkpoint import save_training_checkpoint
from training.plotting import save_loss_curve
from data.encoding import Schema, batch_feature_mask
from training.flow_matching import masked_velocity_loss, sample_flow_batch
from data.prior import PriorConfig, TabICLPriorEngine


def build_training_batch(prior, schema, batch_size, num_rows, device, split="train"):
    tables, metadata = prior.sample_batch(
        batch_size, num_rows, return_metadata=True, split=split,
    )
    mask = batch_feature_mask(metadata, *schema.as_tuple(), device=device)
    return tables.to(device), mask


def train_step(model, optimizer, tables, feature_mask, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_t, t, target = sample_flow_batch(tables, feature_mask)
    loss = masked_velocity_loss(model(x_t, t, feature_mask), target, feature_mask)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def validate(model, validation_batches):
    """Evaluate fixed (x_t, t, target, mask) batches, weighted by valid cells."""
    modes = [(module, module.training) for module in model.modules()]
    total_error, total_cells = 0.0, 0
    model.eval()
    try:
        for x_t, t, target, mask in validation_batches:
            loss = masked_velocity_loss(model(x_t, t, mask), target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            count = int(mask.sum()) * x_t.shape[1]
            total_error += float(loss) * count
            total_cells += count
    finally:
        for module, training in modes:
            module.training = training
    if total_cells == 0:
        raise ValueError("validation requires at least one valid cell")
    return total_error / total_cells


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rows", type=int, default=128)
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
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    for name in ("steps", "batch_size", "num_rows", "val_every", "val_batches", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_cont < 1 or args.max_cat < 1 or args.cat_cardinality < 2:
        parser.error("schema requires max-cont >= 1, max-cat >= 1, cat-cardinality >= 2")
    if args.lr <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        parser.error("lr and max-grad-norm must be positive; weight-decay must be nonnegative")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    schema = Schema(args.max_cont, args.max_cat, args.cat_cardinality)
    # Fixed row counts in this first version: no row padding or row mask.
    prior_config = PriorConfig(
        prior_type=args.prior_type, train_min_seq_len=0,
        log_seq_len=False, replay_small=False,
    )
    prior = TabICLPriorEngine(schema, prior_config, output_device=str(device))
    model = GenTFM(
        max_features=schema.encoded_dim, embed_dim=args.embed_dim,
        num_col_blocks=args.num_col_blocks, num_row_blocks=args.num_row_blocks,
        nhead=args.nhead, dim_feedforward=args.dim_feedforward, num_inds=args.num_inds,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    validation_batches = []
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    for _ in range(args.val_batches):
        tables, mask = build_training_batch(
            prior, schema, args.batch_size, args.num_rows, device, split="validation",
        )
        validation_batches.append((*sample_flow_batch(tables, mask, generator=generator), mask))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_config = {**vars(args), "output_dir": str(args.output_dir), "prior": prior_config.to_dict()}
    best_loss = float("inf")
    train_history = []
    validation_history = []
    for step in range(1, args.steps + 1):
        tables, mask = build_training_batch(prior, schema, args.batch_size, args.num_rows, device)
        loss = train_step(model, optimizer, tables, mask, args.max_grad_norm)
        scheduler.step()
        train_history.append((step, float(loss)))
        if step == 1 or step % args.log_every == 0:
            print(f"step={step} train_loss={float(loss):.6f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            val_loss = validate(model, validation_batches)
            validation_history.append((step, val_loss))
            print(f"step={step} validation_loss={val_loss:.6f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                save_training_checkpoint(
                    args.output_dir / "best.pt", step, model, optimizer, scheduler,
                    best_loss, train_config,
                )
            save_training_checkpoint(
                args.output_dir / "latest.pt", step, model, optimizer, scheduler,
                best_loss, train_config,
            )

    curve_path = save_loss_curve(
        train_history, validation_history, args.output_dir / "loss_curve.png",
    )
    print(f"Loss curve saved to {curve_path}", flush=True)


if __name__ == "__main__":
    main()
