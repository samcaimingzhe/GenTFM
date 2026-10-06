#!/usr/bin/env python3
"""Generate encoded rows from a saved context and matching schema metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_tfm.checkpoint import load_pretrained
from gen_tfm.generation import Calibration, generate_in_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--context', required=True, help='Encoded context .npy in the checkpoint codec')
    p.add_argument('--metadata', required=True)
    p.add_argument('--output', required=True, help='New synthetic .npy path')
    p.add_argument('--num_gen', type=int, default=200)
    p.add_argument('--n_steps', type=int, default=60)
    p.add_argument('--method', choices=['euler', 'heun'], default='euler')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target_y', type=float, help='Optional constant y in the input context numerical scale / category ID')
    p.add_argument('--calibration_alpha', type=float, default=0.0)
    args = p.parse_args()
    output = Path(args.output)
    if output.suffix != '.npy':
        p.error('--output must end in .npy')
    manifest = output.with_suffix('.json')
    if output.exists() or manifest.exists():
        p.error('Output or manifest already exists; choose a new path')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, _ = load_pretrained(args.checkpoint, args.device)
    context = np.load(args.context, allow_pickle=False)
    metadata = json.loads(Path(args.metadata).read_text())
    calibration = Calibration(alpha=args.calibration_alpha) if args.calibration_alpha else None
    generated = generate_in_context(model, context, metadata, args.num_gen, n_steps=args.n_steps,
                                    method=args.method, calibration=calibration, target_y=args.target_y)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, generated)
    manifest.write_text(json.dumps(dict(model_config=model.config(), generation=vars(args),
                                        metadata=metadata, generation_chunk=args.num_gen), indent=2))
    print(f'Saved {generated.shape} to {output}')


if __name__ == '__main__':
    main()
