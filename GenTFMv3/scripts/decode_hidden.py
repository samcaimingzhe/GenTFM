#!/usr/bin/env python3
"""Decode saved raw hidden embeddings using a trained stage-1 checkpoint."""
import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gen_tfm.table import Schema
from gen_tfm.latent import LatentNormalizer, schema_condition
from gen_tfm.decoder import TableDecoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decoder_checkpoint',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--sample_categories',action='store_true')
    a=p.parse_args()
    payload=torch.load(a.decoder_checkpoint,map_location='cpu',weights_only=True)
    data=torch.load(a.input,map_location='cpu',weights_only=True)
    schema=Schema(**payload['schema'])
    model=TableDecoder(**payload['decoder_config']).to(a.device).eval()
    model.load_state_dict(payload['decoder_state'])
    norm=LatentNormalizer(payload['decoder_config']['latent_dim']).to(a.device)
    norm.load_state_dict(payload['normalizer_state'])
    z=data['embeddings'].to(a.device)
    mask=data['row_mask'].to(a.device)
    condition=schema_condition(data['metadata'],schema,a.device)
    reconstructed=model.reconstruct(norm(z),condition,data['metadata'],schema,mask,a.sample_categories)
    output=Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(tables=reconstructed.cpu(),row_mask=mask.cpu(),metadata=data['metadata'],schema=payload['schema']),output)

if __name__=='__main__':
    main()
