#!/usr/bin/env python3
"""Stage 1: frozen pretrained encoder, train mixed-table decoder only."""
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
from gen_tfm.latent import DEFAULT_ENCODER, FrozenTabICLEncoder, LatentNormalizer, schema_condition
from gen_tfm.latent_data import draw_batch
from gen_tfm.decoder import TableDecoder
from gen_tfm.training_state import parse_resume_args, seed_all, capture_rng, restore_rng, cpu_tree, to_device_tree, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--encoder_checkpoint', default='')
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--calibration_tables', type=int, default=16)
    p.add_argument('--validation_batches', type=int, default=4)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--cat_weight', type=float, default=1.)
    p.add_argument('--seed', type=int, default=1400)
    p.add_argument('--prior_type', choices=['mix_scm','mlp_scm','tree_scm'], default='mix_scm')
    p.add_argument('--save_every', type=int, default=100)
    a, resumed = parse_resume_args(p, 'decoder')
    if min(a.steps,a.batch_size,a.calibration_tables,a.validation_batches,a.save_every,a.width) < 1 or a.rows < 4 or a.lr <= 0 or a.cat_weight < 0:
        p.error('Invalid training counts/rows/learning rate/category weight')
    seed_all(a.seed)
    device = torch.device(a.device)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = Schema(**resumed['schema']) if resumed else Schema()
    checkpoint = a.encoder_checkpoint
    if resumed:
        encoder = FrozenTabICLEncoder(resumed, schema).to(device)
    else:
        if not checkpoint:
            from huggingface_hub import hf_hub_download
            checkpoint = hf_hub_download('jingang/TabICL', DEFAULT_ENCODER, cache_dir=str(out/'encoder_cache'))
        encoder = FrozenTabICLEncoder(checkpoint, schema).to(device)
    config = PriorConfig(**resumed['prior_config']) if resumed else PriorConfig(prior_type=a.prior_type,train_min_seq_len=0,log_seq_len=False)
    prior = TabICLPriorEngine(schema, config, output_device=str(device))
    def draw():
        return draw_batch(prior, encoder, a.batch_size, a.rows)
    normalizer = LatentNormalizer(encoder.hidden_dim).to(device)
    if resumed:
        normalizer.load_state_dict(resumed['normalizer_state'])
        validation = to_device_tree(resumed['validation'],device)
    else:
        def calibration():
            for _ in range(a.calibration_tables):
                _, z, mask, _ = draw()
                yield z, mask
        normalizer.fit(calibration())
        validation = [draw() for _ in range(a.validation_batches)]
    decoder = TableDecoder(encoder.hidden_dim,schema.max_cont,schema.max_cat,schema.cat_cardinality,a.width).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=a.lr)
    history, best, start = [], float('inf'), 0
    if resumed:
        decoder.load_state_dict(resumed['decoder_state'], strict=True)
        optimizer.load_state_dict(resumed['optimizer_state'])
        history, best, start = list(resumed['history']), resumed['best_loss'], int(resumed['step'])
        prior.generated_batches = int(resumed['generated_batches'])
        prior._new_prior(1,a.rows)
        restore_rng(resumed['rng_state'])
        print(json.dumps(dict(event='resumed',stage='decoder',step=start)),flush=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    for step in range(start+1,a.steps+1):
        x,z,mask,meta = draw()
        z = normalizer(z).masked_fill(~mask[...,None],0)
        condition = schema_condition(meta,schema,device)
        decoder.train()
        loss, details = decoder.compute_loss(z,condition,x,meta,schema,mask,a.cat_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite decoder loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % a.save_every == 0 or step == a.steps:
            decoder.eval()
            metrics = []
            with torch.no_grad():
                for vx,vz,vm,vmeta in validation:
                    vz = normalizer(vz).masked_fill(~vm[...,None],0)
                    vc = schema_condition(vmeta,schema,device)
                    vl, vd = decoder.compute_loss(vz,vc,vx,vmeta,schema,vm,a.cat_weight)
                    metrics.append(dict(validation_loss=float(vl),**{k:float(v) for k,v in vd.items()}))
            record = dict(step=step,train_loss=float(loss.detach()),**{
                k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]})
            history.append(record)
            print(json.dumps(record),flush=True)
            payload = dict(stage='decoder',step=step,decoder_config=decoder.config,decoder_state=decoder.state_dict(),
                           normalizer_state=normalizer.state_dict(),schema=asdict(schema),encoder_config=encoder.config,
                           encoder_state=encoder.state_dict(),encoder_checkpoint=encoder.checkpoint,
                           prior_config=config.to_dict(),train_config=vars(a),optimizer_state=optimizer.state_dict(),history=history)
            improved = record['validation_loss'] < best
            best = min(best, record['validation_loss'])
            payload.update(resume_version=1, best_loss=best, rng_state=capture_rng(),
                           generated_batches=prior.generated_batches, validation=cpu_tree(validation))
            atomic_save(payload,out/'latest.pt')
            if improved:
                atomic_save(payload,out/'best.pt')
    vx,vz,vm,vmeta = validation[0]
    with torch.no_grad():
        reconstructed = decoder.reconstruct(normalizer(vz),schema_condition(vmeta,schema,device),vmeta,schema,vm)
    torch.save(dict(original=vx.cpu(),reconstructed=reconstructed.cpu(),row_mask=vm.cpu(),metadata=vmeta),out/'reconstruction.pt')
    (out/'losses.json').write_text(json.dumps(history,indent=2))

if __name__ == '__main__':
    main()
