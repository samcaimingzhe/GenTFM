import argparse
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from gen_tfm.training_state import seed_all,capture_rng,restore_rng,atomic_save,parse_resume_args

class ResumeTests(unittest.TestCase):
    def test_rng_roundtrip_safe_checkpoint(self):
        seed_all(47)
        state=capture_rng()
        expected=(random.random(),np.random.normal(size=4),torch.randn(4))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            atomic_save({'rng_state':state},path)
            restore_rng(torch.load(path,weights_only=True)['rng_state'])
        self.assertEqual(expected[0],random.random())
        np.testing.assert_array_equal(expected[1],np.random.normal(size=4))
        torch.testing.assert_close(expected[2],torch.randn(4),rtol=0,atol=0)

    def test_resume_inherits_and_rejects_changed_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latest.pt'
            payload=dict(stage='decoder',resume_version=1,step=2,rng_state={},validation=[],generated_batches=3,
                         optimizer_state={},train_config=dict(output_dir=directory,rows=16,steps=2))
            atomic_save(payload,path)
            def parser():
                p=argparse.ArgumentParser()
                p.add_argument('--output_dir',required=True)
                p.add_argument('--rows',type=int,default=128)
                p.add_argument('--steps',type=int,default=10000)
                return p
            with patch.object(sys,'argv',['train','--resume',str(path),'--steps','4']):
                a,_=parse_resume_args(parser(),'decoder')
                self.assertEqual(a.rows,16)
                self.assertEqual(a.steps,4)
            with patch.object(sys,'argv',['train','--resume',str(path),'--steps','4','--rows','20']):
                with self.assertRaises(SystemExit):
                    parse_resume_args(parser(),'decoder')
            payload.pop('rng_state')
            atomic_save(payload,path)
            with patch.object(sys,'argv',['train','--resume',str(path),'--steps','4']):
                with self.assertRaises(SystemExit):
                    parse_resume_args(parser(),'decoder')

if __name__=='__main__':
    unittest.main()
