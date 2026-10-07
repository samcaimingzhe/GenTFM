import os
from pathlib import Path
import subprocess
import torch
import argparse
import sys
parser=argparse.ArgumentParser(description='Verify continuous training equals split/resume training on CPU')
parser.add_argument('--encoder_checkpoint',required=True)
parser.add_argument('--work_dir',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
python=sys.executable
checkpoint=args.encoder_checkpoint
env=dict(os.environ,OMP_NUM_THREADS='1',NUMBA_CACHE_DIR=str(Path(args.work_dir).absolute()/'numba'))

def run(stage,out,steps,extra):
    subprocess.run([python,str(root/'scripts'/f'train_{stage}.py'),'--output_dir',str(out),'--steps',str(steps),'--batch_size','1','--rows','16','--width','32','--save_every','1',*extra],env=env,check=True)

def compare(a,b,key):
    left=torch.load(a/'latest.pt',weights_only=True)
    right=torch.load(b/'latest.pt',weights_only=True)
    for name in left[key]:
        torch.testing.assert_close(left[key][name],right[key][name],rtol=0,atol=0)
    for name in left['normalizer_state']:
        torch.testing.assert_close(left['normalizer_state'][name],right['normalizer_state'][name],rtol=0,atol=0)
    assert left['generated_batches']==right['generated_batches']
    assert left['history']==right['history']
    print('EXACT MATCH:',key,flush=True)

base=Path(args.work_dir)
d1,d2=base/'decoder_full',base/'decoder_split'
opts=['--encoder_checkpoint',checkpoint,'--calibration_tables','1','--validation_batches','1']
run('decoder',d1,4,opts)
run('decoder',d2,2,opts)
# Resume inherits all training arguments; source encoder is embedded.
subprocess.run([python,str(root/'scripts/train_decoder.py'),'--resume',str(d2/'latest.pt'),'--steps','4'],env=env,check=True)
compare(d1,d2,'decoder_state')
f1,f2=base/'flow_full',base/'flow_split'
opts=['--decoder_checkpoint',str(d1/'best.pt'),'--heads','4','--layers','1']
run('latent',f1,4,opts)
run('latent',f2,2,opts)
subprocess.run([python,str(root/'scripts/train_latent.py'),'--resume',str(f2/'latest.pt'),'--steps','4'],env=env,check=True)
compare(f1,f2,'flow_state')
# Configuration changes must fail before training.
r=subprocess.run([python,str(root/'scripts/train_decoder.py'),'--resume',str(d2/'latest.pt'),'--steps','5','--rows','20'],env=env,capture_output=True,text=True)
assert r.returncode!=0 and 'configuration differs: rows' in r.stderr
print('Mismatched configuration rejected.',flush=True)
