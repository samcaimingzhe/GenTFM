#!/usr/bin/env python3
"""Encode a saved engine batch: {'tables': Tensor, 'metadata': list, 'schema': dict}."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_tfm.table import Schema
from gen_tfm.latent import FrozenTabICLEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--encoder_checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    schema = Schema(**data['schema'])
    encoder = FrozenTabICLEncoder(args.encoder_checkpoint, schema).to(args.device)
    z, mask = encoder(data['tables'], data['metadata'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(embeddings=z.cpu(), row_mask=mask.cpu(), metadata=data['metadata'],
                    schema=asdict(schema), encoder_checkpoint=encoder.checkpoint,
                    representation='raw_tabicl_row_embedding'), output)
    print(f'Saved embeddings {tuple(z.shape)} to {output}')


if __name__ == '__main__':
    main()
