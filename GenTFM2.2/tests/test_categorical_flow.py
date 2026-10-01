"""Mixed flow/CE regression checks without downloading or running TabICL."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data.encoding import Schema, batch_feature_mask, velocity_feature_mask
from model import GenTFM
from training.flow_matching import categorical_cross_entropy, flow_matching_loss, sample_flow_batch
from training.checkpoint import save_training_checkpoint, load_pretrained
from inference.sampling import sample_table
from script.train import train_step, validate, main


class CategoricalFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.schema = Schema(2, 2, 3)
        self.metadata = [
            {"n_cont": 1, "n_cat": 1, "cat_cardinalities": [2]},
            {"n_cont": 2, "n_cat": 2, "cat_cardinalities": [3, 2]},
            {"n_cont": 0, "n_cat": 0},
        ]
        self.mask = batch_feature_mask(self.metadata, *self.schema.as_tuple(), device="cpu")
        self.x = torch.zeros(3, 3, self.schema.encoded_dim)
        self.x[0, :, 0] = torch.tensor([-1., 0., 1.])
        self.x[1, :, :2] = torch.randn(3, 2)
        self.x[0, :, 2] = 1
        self.x[1, :, 3] = 1
        self.x[1, :, 5] = 1
        self.x[:, :, 8:] = self.mask[:, None, 8:]

    def model(self):
        return GenTFM(max_features=10, embed_dim=8, num_col_blocks=1,
                      num_row_blocks=1, nhead=2, dim_feedforward=16, num_inds=3,
                      max_cont=2, max_cat=2, cat_cardinality=3)

    def test_ce_formula_invalid_classes_and_empty_fields(self):
        logits = torch.zeros(3, 3, 2, 3, requires_grad=True)
        allowed = self.mask[:, 2:8].reshape(3, 2, 3)[:, None].expand_as(logits)
        dirty = logits.masked_fill(~allowed, float('nan'))
        loss = categorical_cross_entropy(dirty, self.x, self.mask, self.schema)
        self.assertAlmostEqual(float(loss), (2*math.log(2)+math.log(3))/3, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(int(torch.count_nonzero(logits.grad[~allowed])), 0)
        empty = torch.full_like(logits, float('nan'), requires_grad=True)
        zero = categorical_cross_entropy(empty, self.x, torch.zeros_like(self.mask), self.schema)
        self.assertEqual(float(zero), 0)
        zero.backward()
        self.assertEqual(int(torch.count_nonzero(empty.grad)), 0)

    def test_fixed_observations_and_missing_data_rejection(self):
        xt, t, target = sample_flow_batch(self.x, self.mask, schema=self.schema)
        torch.testing.assert_close(xt[..., 8:], self.x[..., 8:])
        self.assertEqual(int(torch.count_nonzero(target[..., 8:])), 0)
        x0 = self.x - target
        torch.testing.assert_close(xt, (1-t[:, None, None])*x0+t[:, None, None]*self.x)
        missing = self.x.clone()
        missing[0, 0, 8] = 0
        with self.assertRaisesRegex(ValueError, "missing observations"):
            sample_flow_batch(missing, self.mask, schema=self.schema)

    def test_heads_padding_permutation_and_gradients(self):
        model = self.model()
        xt, t, target = sample_flow_batch(self.x, self.mask, schema=self.schema)
        outputs = model(xt, t, self.mask, return_aux=True)
        self.assertEqual(outputs['velocity'].shape, self.x.shape)
        self.assertEqual(outputs['categorical_logits'].shape, (3, 3, 2, 3))
        dirty = xt.masked_fill(~self.mask[:, None], float('nan'))
        dirty_outputs = model(dirty, t, self.mask, return_aux=True)
        perm = torch.tensor([2, 0, 1])
        permuted = model(xt[:, perm], t, self.mask, return_aux=True)
        for key, value in outputs.items():
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(int(torch.count_nonzero(value[2])), 0)
            torch.testing.assert_close(dirty_outputs[key], value)
            torch.testing.assert_close(permuted[key], value[:, perm], atol=2e-6, rtol=2e-5)
        metrics = flow_matching_loss(outputs, target, self.x, self.mask, self.schema, .8)
        torch.testing.assert_close(metrics['loss'], metrics['velocity_mse']+.8*metrics['categorical_ce'])
        metrics['loss'].backward()
        for head in (model.velocity_head, model.categorical_head):
            self.assertGreater(float(head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        empty_mask = torch.zeros_like(self.mask)
        empty_outputs = model(torch.full_like(xt, float('nan')), t, empty_mask, return_aux=True)
        zero = flow_matching_loss(empty_outputs, target, self.x, empty_mask, self.schema)['loss']
        self.assertEqual(float(zero), 0)
        zero.backward()

    def test_training_validation_checkpoint_and_category_sampling(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        before = model.categorical_head[-1].weight.detach().clone()
        metrics = train_step(model, optimizer, self.x, self.mask, categorical_weight=.8, return_metrics=True)
        self.assertTrue(torch.isfinite(metrics['loss']))
        self.assertFalse(torch.equal(before, model.categorical_head[-1].weight))
        xt, t, target = sample_flow_batch(self.x, self.mask, schema=self.schema)
        combined = [(xt, t, target, self.mask, self.x)]
        split = [(xt[i:i+1], t[i:i+1], target[i:i+1], self.mask[i:i+1], self.x[i:i+1]) for i in range(3)]
        a = validate(model, combined, .8, return_metrics=True)
        b = validate(model, split, .8, return_metrics=True)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_training_checkpoint(path, 1, model, optimizer, scheduler, a['loss'])
            restored, _ = load_pretrained(path)
            for key, value in model(xt, t, self.mask, return_aux=True).items():
                torch.testing.assert_close(restored(xt, t, self.mask, return_aux=True)[key], value)
            # Force the valid class 0: invalid high-logit classes must never win.
            with torch.no_grad():
                for p in restored.categorical_head.parameters():
                    p.zero_()
                restored.categorical_head[-1].bias.copy_(torch.tensor([100., -100., 1000.]))
            for method in ('euler', 'heun'):
                for decoding in ('sample', 'argmax'):
                    y = sample_table(restored, self.mask, 4, n_steps=2, method=method,
                                     categorical_method=decoding, generator=torch.Generator().manual_seed(9))
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertEqual(int(torch.count_nonzero(y.masked_select(~self.mask[:, None]))), 0)
                    torch.testing.assert_close(y[..., 8:], self.mask[:, None, 8:].expand(3, 4, 2).float())
                    self.assertTrue((y[0, :, 2] == 1).all())
                    self.assertTrue((y[1, :, 4] == 1).all())
                    self.assertTrue((y[1, :, 5] == 1).all())
            self.assertTrue(restored.training is False)

    def test_main_writes_best_latest_and_final_curve(self):
        x, metadata = self.x[:2], self.metadata[:2]
        class LocalPrior:
            def __init__(self, *args, **kwargs):
                pass
            def sample_batch(self, *args, **kwargs):
                return x.clone(), metadata
        with tempfile.TemporaryDirectory() as directory:
            argv = ['train', '--steps', '2', '--batch-size', '2', '--num-rows', '3',
                    '--max-cont', '2', '--max-cat', '2', '--cat-cardinality', '3',
                    '--embed-dim', '8', '--num-col-blocks', '1', '--num-row-blocks', '1',
                    '--nhead', '2', '--dim-feedforward', '16', '--num-inds', '3',
                    '--val-batches', '1', '--val-every', '1', '--log-every', '1',
                    '--categorical-loss-weight', '.8', '--output-dir', directory]
            with patch('script.train.TabICLPriorEngine', LocalPrior), patch('sys.argv', argv):
                main()
            for name in ('best.pt', 'latest.pt', 'loss_curve.png'):
                self.assertGreater((Path(directory)/name).stat().st_size, 0)
            model, checkpoint = load_pretrained(Path(directory)/'latest.pt')
            self.assertEqual(model.max_cat, 2)
            self.assertEqual(checkpoint['train_config']['categorical_loss_weight'], .8)


if __name__ == '__main__':
    unittest.main()
