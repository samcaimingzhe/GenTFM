"""Safe training-state serialization and strict resume configuration."""
import argparse
import random
import numpy as np
import torch


def parse_resume_args(parser, stage):
    parser.add_argument('--resume', default='', help='Resume a latest.pt; steps is the total target step count')
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--resume', default='')
    first, _ = preliminary.parse_known_args()
    payload = None
    if first.resume:
        payload = torch.load(first.resume, map_location='cpu', weights_only=True)
        required = {'resume_version','rng_state','validation','generated_batches','optimizer_state','train_config'}
        if payload.get('stage') != stage or payload.get('resume_version') != 1 or not required.issubset(payload):
            parser.error('Checkpoint has no complete resume state for this stage; use a new resume-enabled checkpoint')
        actions = {a.dest:a for a in parser._actions}
        saved = {k:v for k,v in payload['train_config'].items() if k in actions and k != 'resume'}
        parser.set_defaults(**saved)
        for key in saved:
            actions[key].required = False
    args = parser.parse_args()
    if payload:
        # Only runtime destination/device, logging interval and total target steps may change.
        allowed = {'resume','output_dir','device','save_every','steps'}
        for k,v in payload['train_config'].items():
            if k not in allowed and hasattr(args,k) and getattr(args,k) != v:
                parser.error(f'Resume configuration differs: {k}; expected {v!r}')
        if args.steps <= int(payload['step']):
            parser.error('--steps must exceed saved step; it is a total target, not additional steps')
    return args, payload


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=n[0],keys=torch.tensor(n[1].astype(np.int64)),
                position=n[2],has_gauss=n[3],cached_gaussian=n[4]), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n['kind'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from saved random state')
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def to_device_tree(value,device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:to_device_tree(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [to_device_tree(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(to_device_tree(v,device) for v in value)
    return value


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)
