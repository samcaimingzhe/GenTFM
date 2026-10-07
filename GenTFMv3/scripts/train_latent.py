#!/usr/bin/env python3
"""Train schema-conditioned whole-table latent flow; load the trained decoder representation first."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_tfm.table import Schema
from gen_tfm.prior import PriorConfig, TabICLPriorEngine
from gen_tfm.latent import DEFAULT_ENCODER, FrozenTabICLEncoder, LatentNormalizer, LatentFlow, schema_condition
from gen_tfm.training_state import parse_resume_args, seed_all, capture_rng, restore_rng, cpu_tree, to_device_tree, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--decoder_checkpoint', default='', help='Stage-1 best.pt; reuses encoder and fixed latent statistics')
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=1400)
    p.add_argument('--prior_type', choices=['mix_scm', 'mlp_scm', 'tree_scm'], default='mix_scm')
    p.add_argument('--save_every', type=int, default=100)
    args, resumed = parse_resume_args(p, 'flow')
    if not resumed and not args.decoder_checkpoint:
        p.error('--decoder_checkpoint is required for a new flow run')
    if min(args.steps, args.batch_size, args.save_every) < 1 or args.rows < 4:
        p.error('Positive step/batch/save counts and rows >= 4 required')
    seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    decoder_payload = resumed if resumed else torch.load(args.decoder_checkpoint, map_location='cpu', weights_only=True)
    if not resumed and decoder_payload.get('stage') != 'decoder':
        raise ValueError('Expected a stage-1 decoder checkpoint')
    schema = Schema(**decoder_payload['schema'])
    encoder = FrozenTabICLEncoder(decoder_payload, schema).to(device)
    prior_config = PriorConfig(**decoder_payload['prior_config'])
    if args.prior_type != prior_config.prior_type:
        p.error('prior_type differs from decoder checkpoint')
    prior = TabICLPriorEngine(schema, prior_config, output_device=str(device))

    def draw():
        # One prior draw per table: no assumption that SCM tables have equal lengths.
        zs, metas = [], []
        for _ in range(args.batch_size):
            x, meta = prior.sample_batch(1, args.rows, return_metadata=True)
            z, mask = encoder(x, meta)
            zs.append(z[0, mask[0]])
            metas.append(meta[0])
        n = max(map(len, zs))
        z = torch.zeros(len(zs), n, encoder.hidden_dim, device=device)
        mask = torch.zeros(len(zs), n, device=device, dtype=torch.bool)
        for i, item in enumerate(zs):
            z[i, :len(item)] = item
            mask[i, :len(item)] = True
        return z, mask, metas

    normalizer = LatentNormalizer(encoder.hidden_dim).to(device)
    normalizer.load_state_dict(decoder_payload['normalizer_state'], strict=True)
    print(json.dumps({'event': 'loaded_decoder_representation', 'latent_dim': encoder.hidden_dim}), flush=True)
    # Fixed independent draws for monitoring; these do not enter normalization fitting.
    if resumed:
        val_z, val_mask, val_meta, val_condition, val_noise, val_t = to_device_tree(resumed['validation'],device)
    else:
        val_z, val_mask, val_meta = draw()
        val_z = normalizer(val_z).masked_fill(~val_mask[..., None], 0)
        val_condition = schema_condition(val_meta, schema, device)
        val_noise = torch.randn_like(val_z)
        val_t = torch.rand(len(val_z), device=device)
    flow = LatentFlow(encoder.hidden_dim, schema.max_cont + schema.max_cat, args.width, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.lr)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    history, start = [], 0
    if resumed:
        flow.load_state_dict(resumed['flow_state'],strict=True)
        optimizer.load_state_dict(resumed['optimizer_state'])
        history, start = list(resumed['history']), int(resumed['step'])
        prior.generated_batches = int(resumed['generated_batches'])
        prior._new_prior(1,args.rows)
        restore_rng(resumed['rng_state'])
        print(json.dumps(dict(event='resumed',stage='flow',step=start)),flush=True)
    for step in range(start+1, args.steps + 1):
        z, mask, meta = draw()
        z = normalizer(z).masked_fill(~mask[..., None], 0)
        condition = schema_condition(meta, schema, device)
        flow.train()
        loss = flow.compute_loss(z, condition, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite flow loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.save_every == 0 or step == args.steps:
            flow.eval()
            with torch.no_grad():
                val_loss = flow.compute_loss(val_z, val_condition, val_mask, z0=val_noise, t=val_t)
            record = dict(step=step, loss=float(loss.detach()), validation_loss=float(val_loss))
            history.append(record)
            print(json.dumps(record), flush=True)
            # Encoder weights included so later use does not depend on a cached download.
            payload = dict(stage="flow",step=step, flow_config=flow.config, flow_state=flow.state_dict(),
                           normalizer_state=normalizer.state_dict(), optimizer_state=optimizer.state_dict(),
                           schema=asdict(schema), encoder_checkpoint=encoder.checkpoint,
                           encoder_state=encoder.state_dict(), encoder_config=encoder.config,
                           decoder_config=decoder_payload['decoder_config'], decoder_state=decoder_payload['decoder_state'],
                           train_config=vars(args), prior_config=prior_config.to_dict(), history=history)
            payload.update(resume_version=1, rng_state=capture_rng(), generated_batches=prior.generated_batches,
                           validation=cpu_tree((val_z,val_mask,val_meta,val_condition,val_noise,val_t)))
            atomic_save(payload,out/'latest.pt')
    generated = flow.sample(val_condition, val_mask)
    embeddings = normalizer.inverse(generated).masked_fill(~val_mask[..., None], 0)
    torch.save(dict(embeddings=embeddings.cpu(), row_mask=val_mask.cpu(), metadata=val_meta), out / 'sample_embeddings.pt')
    (out / 'losses.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
