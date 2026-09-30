"""Run with: python -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from model import GenTFM
from training.flow_matching import masked_velocity_loss, sample_flow_batch
from inference.sampling import sample_table
from training.checkpoint import export_slim_checkpoint, load_pretrained, save_training_checkpoint
from script.train import train_step, validate


def small_model(**kwargs):
    return GenTFM(max_features=5, embed_dim=8, num_col_blocks=1,
                  num_row_blocks=1, nhead=2, dim_feedforward=16, num_inds=3, **kwargs)


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.mask = torch.tensor([[True, False, True, False, True], [False]*5])
        self.x = torch.randn(2, 4, 5)

    def test_loss_formula_and_padding_gradients(self):
        pred = torch.tensor([[[1., float('nan'), 3.], [2., float('inf'), 4.]]], requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.tensor([[True, False, True]])
        loss = masked_velocity_loss(pred, target, mask)
        self.assertEqual(float(loss), 7.5)  # (1+9+4+16)/4
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[[.5, 0., 1.5], [1., 0., 2.]]]))
        empty = torch.full((2, 4, 5), float('nan'), requires_grad=True)
        loss = masked_velocity_loss(empty, empty.detach(), torch.zeros_like(self.mask))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(empty.grad)), 0)

    def test_flow_path(self):
        xt, t, target = sample_flow_batch(self.x, self.mask, generator=torch.Generator().manual_seed(10))
        valid = self.mask[:, None, :].expand_as(self.x)
        x1 = self.x.masked_fill(~valid, 0.)
        x0 = x1 - target
        torch.testing.assert_close(xt, (1-t[:, None, None])*x0 + t[:, None, None]*x1)
        self.assertTrue(((t >= 0) & (t <= 1)).all())
        self.assertEqual(int(torch.count_nonzero(xt[~valid])), 0)
        self.assertEqual(int(torch.count_nonzero(target[~valid])), 0)

    def test_model_padding_time_permutation_and_gradients(self):
        for norm_first in (True, False):
            for recompute in (False, True):
                with self.subTest(norm_first=norm_first, recompute=recompute):
                    model = small_model(norm_first=norm_first, recompute=recompute)
                    x = self.x.clone().requires_grad_()
                    t = torch.tensor([.2, .7])
                    y = model(x, t, self.mask)
                    self.assertEqual(y.shape, x.shape)
                    self.assertTrue(torch.isfinite(y).all())
                    valid = self.mask[:, None, :].expand_as(x)
                    self.assertEqual(int(torch.count_nonzero(y[~valid])), 0)
                    dirty = x.detach().masked_fill(~valid, float('nan'))
                    torch.testing.assert_close(model(dirty, t, self.mask), y)
                    perm = torch.tensor([2, 0, 3, 1])
                    torch.testing.assert_close(model(x[:, perm], t, self.mask), y[:, perm], atol=2e-6, rtol=2e-5)
                    self.assertFalse(torch.allclose(y[0], model(x, torch.tensor([.6, .7]), self.mask)[0]))
                    y.square().sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertEqual(int(torch.count_nonzero(x.grad[~valid])), 0)
                    self.assertGreater(float(model.time_embedding.mlp[0].weight.grad.abs().sum()), 0.)
                    empty = model(torch.full_like(x, float('nan')), t, torch.zeros_like(self.mask))
                    self.assertTrue(torch.isfinite(empty).all())
                    self.assertEqual(int(torch.count_nonzero(empty)), 0)
                    empty.sum().backward()

    def test_ode_known_velocity_and_mode_restoration(self):
        class TimeVelocity(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.child = nn.Dropout()
            def forward(self, x, t, mask):
                return t[:, None, None].expand_as(x)
        model = TimeVelocity()
        model.train()
        model.child.eval()
        valid = self.mask[:, None, :].expand_as(self.x)
        noise = torch.randn(self.x.shape, generator=torch.Generator().manual_seed(12)).masked_fill(~valid, 0.)
        for method, shift in (("euler", .375), ("heun", .5)):
            y = sample_table(model, self.mask, 4, n_steps=4, method=method,
                             generator=torch.Generator().manual_seed(12))
            torch.testing.assert_close(y, (noise + shift).masked_fill(~valid, 0.))
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
        with self.assertRaises(ValueError):
            sample_table(model, self.mask, 4, n_steps=0)
        with self.assertRaises(ValueError):
            sample_table(model, self.mask, 4, method='unknown')

    def test_training_validation_checkpoint_and_sampling(self):
        model = small_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        before = model.velocity_head[-1].weight.detach().clone()
        loss = train_step(model, optimizer, self.x, self.mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.velocity_head[-1].weight))
        scheduler.step()
        batch = (*sample_flow_batch(self.x, self.mask), self.mask)
        validation = validate(model, [batch])
        self.assertEqual(validation, validate(model, [batch]))
        self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'full.pt'
            slim = Path(directory) / 'slim.pt'
            save_training_checkpoint(full, 1, model, optimizer, scheduler, validation)
            export_slim_checkpoint(full, slim)
            for path in (full, slim):
                restored, ckpt = load_pretrained(path)
                self.assertEqual(ckpt['step'], 1)
                model.eval()
                torch.testing.assert_close(restored(*batch[:2], self.mask), model(*batch[:2], self.mask))
                for method in ('euler', 'heun'):
                    y = sample_table(restored, self.mask, 3, n_steps=2, method=method)
                    self.assertEqual(y.shape, (2, 3, 5))
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertEqual(int(torch.count_nonzero(y[1])), 0)


if __name__ == '__main__':
    unittest.main()
