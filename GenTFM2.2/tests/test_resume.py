"""Check interrupted training against an uninterrupted reference trajectory."""
import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from script.train import main, train_step
from training.checkpoint import load_training_checkpoint, save_training_checkpoint


class RandomPrior:
    """Exercise adapter counters, cache warmup and all three CPU RNGs."""
    def __init__(self, schema, config, **kwargs):
        self.schema = schema
        self.generated_batches = 0
        self._prior_cache = {}

    def _new_prior(self, batch_size, rows, split=None):
        key = (batch_size, rows, split or 'none', 0)
        if key not in self._prior_cache:
            torch.rand(1)
            random.random()
            np.random.rand()
            self._prior_cache[key] = object()

    def sample_batch(self, batch_size, rows, split='train', **kwargs):
        self._new_prior(batch_size, rows, split)
        x = torch.zeros(batch_size, rows, self.schema.encoded_dim)
        x[..., 0] = torch.randn(batch_size, rows) + random.random() + np.random.rand()
        # The counter changes the category distribution, as in the real adapter.
        labels = (torch.randint(0, 2, (batch_size, rows)) + self.generated_batches) % 2
        x[..., 2:4] = torch.nn.functional.one_hot(labels, 2).float()
        x[..., self.schema.mask_start] = 1
        self.generated_batches += 1
        metadata = [{"n_cont": 1, "n_cat": 1, "cat_cardinalities": [2]} for _ in range(batch_size)]
        return x, metadata


class ResumeTests(unittest.TestCase):
    def run_main(self, directory, extra=None, interrupted=False):
        argv = ['train', '--steps', '4', '--batch-size', '2', '--num-rows', '3',
                '--max-cont', '2', '--max-cat', '2', '--cat-cardinality', '3',
                '--embed-dim', '8', '--num-col-blocks', '1', '--num-row-blocks', '1',
                '--nhead', '2', '--dim-feedforward', '16', '--num-inds', '3',
                '--val-batches', '1', '--val-every', '2', '--log-every', '1',
                '--save-every', '2', '--output-dir', str(directory)]
        if extra is not None:
            argv = ['train'] + extra
        count = 0
        def interrupt_before_third_update(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise KeyboardInterrupt()
            return train_step(*args, **kwargs)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), patch('script.train.TabICLPriorEngine', RandomPrior), patch('sys.argv', argv):
            if interrupted:
                with patch('script.train.train_step', interrupt_before_third_update):
                    main()
            else:
                main()
        return stream.getvalue()

    def test_resume_matches_uninterrupted_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, resumed = root/'reference', root/'resumed'
            self.run_main(reference)
            output = self.run_main(resumed, interrupted=True)
            self.assertIn('Training interrupted', output)
            halfway = load_training_checkpoint(resumed/'latest.pt')
            self.assertEqual(halfway['step'], 2)
            output = self.run_main(resumed, ['--resume', str(resumed/'latest.pt'), '--device', 'cpu'])
            self.assertIn('next_step=3 total_steps=4', output)
            expected = load_training_checkpoint(reference/'latest.pt')
            actual = load_training_checkpoint(resumed/'latest.pt')
            self.assertEqual(actual['step'], 4)
            self.assertEqual(actual['best_loss'], expected['best_loss'])
            self.assertEqual(actual['scheduler_state_dict'], expected['scheduler_state_dict'])
            self.assertEqual(actual['training_state']['train_history'], expected['training_state']['train_history'])
            self.assertEqual(actual['training_state']['validation_history'], expected['training_state']['validation_history'])
            self.assertEqual(actual['training_state']['prior'], expected['training_state']['prior'])
            for key in expected['model_state_dict']:
                torch.testing.assert_close(actual['model_state_dict'][key], expected['model_state_dict'][key], rtol=0, atol=0)
            for key, value in expected['optimizer_state_dict']['state'].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(actual['optimizer_state_dict']['state'][key][name], tensor, rtol=0, atol=0)
            for left, right in zip(actual['training_state']['validation_batches'], expected['training_state']['validation_batches']):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue((resumed/'loss_curve.png').exists())
            # A finished checkpoint must not perform extra optimizer updates.
            self.run_main(resumed, ['--resume', str(resumed/'latest.pt')])
            self.assertEqual(load_training_checkpoint(resumed/'latest.pt')['step'], 4)
            moved = root/'moved'
            self.run_main(moved, ['--resume', str(resumed/'latest.pt'), '--output-dir', str(moved)])
            self.assertEqual(load_training_checkpoint(moved/'best.pt')['best_loss'], actual['best_loss'])
            self.assertEqual(load_training_checkpoint(moved/'latest.pt')['step'], 4)

    def test_resume_rejects_changed_plan_and_slim_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_main(root, interrupted=True)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_main(root, ['--resume', str(root/'latest.pt'), '--steps', '6'])
                with self.assertRaises(SystemExit):
                    self.run_main(root, ['--resume', str(root/'latest.pt'), '--batch-size', '3'])
            slim = root/'slim.pt'
            torch.save({'model_state_dict': {}, 'model_config': {}}, slim)
            with self.assertRaisesRegex(ValueError, 'not a resumable'):
                load_training_checkpoint(slim)

    def test_legacy_full_checkpoint_can_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_main(root, interrupted=True)
            checkpoint = load_training_checkpoint(root/'latest.pt')
            del checkpoint['training_state']
            torch.save(checkpoint, root/'latest.pt')
            output = self.run_main(root, ['--resume', str(root/'latest.pt')])
            self.assertIn('Legacy checkpoint', output)
            self.assertIn('best loss is reset', output)
            self.assertEqual(load_training_checkpoint(root/'latest.pt')['step'], 4)

    def test_failed_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.pt'
            path.write_bytes(b'previous complete checkpoint')
            class Model:
                def state_dict(self): return {}
                def config(self): return {}
            class State:
                def state_dict(self): return {}
            def fail_save(checkpoint, handle):
                handle.write(b'partial new checkpoint')
                raise OSError('simulated disk write failure')
            with patch('training.checkpoint.torch.save', fail_save):
                with self.assertRaises(OSError):
                    save_training_checkpoint(path, 1, Model(), State(), State(), 1.)
            self.assertEqual(path.read_bytes(), b'previous complete checkpoint')
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
