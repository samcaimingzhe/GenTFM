"""Run with: python -m unittest discover -s tests -p test_latent.py"""
import tempfile
import unittest
from pathlib import Path
import torch
from gen_tfm.table import Schema
from gen_tfm.latent import FrozenTabICLEncoder, LatentFlow, LatentNormalizer


class LatentTests(unittest.TestCase):
    def test_velocity_target_and_equal_table_weight(self):
        flow = LatentFlow(4, 2, 8, 2, 1)
        for param in flow.parameters():
            param.data.zero_()
        z = torch.ones(2, 3, 4)
        z[1] = 2
        mask = torch.tensor([[True, False, False], [True, True, True]])
        loss = flow.compute_loss(z, torch.zeros(2, 2), mask, z0=torch.zeros_like(z), t=torch.ones(2) * .5)
        self.assertAlmostEqual(loss.item(), 2.5)  # (1^2 + 2^2) / 2, not row-weighted
        loss.backward()
        self.assertIsNotNone(flow.output.bias.grad)

    def test_padding_and_permutation(self):
        torch.manual_seed(4)
        flow = LatentFlow(4, 2, 8, 2, 1).eval()
        z = torch.randn(1, 4, 4)
        mask = torch.tensor([[True, True, True, False]])
        c, t = torch.randn(1, 2), torch.ones(1) * .3
        before = flow(z, t, c, mask)
        padded = z.clone()
        padded[:, 3] = 1000
        torch.testing.assert_close(before[:, :3], flow(padded, t, c, mask)[:, :3])
        order = torch.tensor([2, 0, 1, 3])
        torch.testing.assert_close(before[:, order], flow(z[:, order], t, c, mask[:, order]), atol=1e-6, rtol=1e-5)
        sampled = flow.sample(c, mask, steps=2)
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertEqual(sampled[:, 3].abs().sum().item(), 0)

    def test_normalizer_masks_and_roundtrip(self):
        norm = LatentNormalizer(2)
        z = torch.tensor([[[1., 2.], [3., 6.], [100., 100.]]])
        mask = torch.tensor([[True, True, False]])
        norm.fit([(z, mask)])
        torch.testing.assert_close(norm.mean, torch.tensor([2., 4.]))
        torch.testing.assert_close(norm.std, torch.tensor([1., 2.]))
        torch.testing.assert_close(norm.inverse(norm(z)), z)

    def test_real_tabicl_api_freeze_codecs_variable_rows(self):
        from tabicl._model.tabicl import TabICL
        # Small randomly initialized model tests the real library interface;
        # pretrained-weight behavior is separately covered by the smoke run.
        config = dict(embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
                      col_feature_group=False, col_target_aware=False, row_num_blocks=1,
                      row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2)
        model = TabICL(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'encoder.pt'
            torch.save(dict(config=config, state_dict=model.state_dict()), checkpoint)
            for _ in range(1):
                schema = Schema(2, 1, 4)
                encoder = FrozenTabICLEncoder(checkpoint, schema)
                encoder.train()
                self.assertFalse(encoder.training)
                meta = dict(n_cont=1, n_cat=1, cat_cardinalities=[3], **schema.metadata_fields())
                x = torch.zeros(2, 5, schema.table_dim)
                x[:, :, 0] = torch.linspace(-1, 1, 5)
                x[:, :, schema.category_index(0)] = torch.tensor([0, 1, 2, 0, 1])
                metas = [dict(meta, tabicl_seq_len=5), dict(meta, tabicl_seq_len=3)]
                expected = torch.tensor([[-1., 0.], [-.5, 1.], [0., 2.], [.5, 0.], [1., 1.]])
                torch.testing.assert_close(encoder.table_input(x[0], metas[0]), expected)
                z, mask = encoder(x.requires_grad_(), metas)
                self.assertEqual(z.shape, (2, 5, 32))
                self.assertEqual(mask.sum(1).tolist(), [5, 3])
                self.assertFalse(z.requires_grad)
                self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            config['col_target_aware'] = True
            model = TabICL(**config)
            torch.save(dict(config=config, state_dict=model.state_dict()), checkpoint)
            with self.assertRaisesRegex(ValueError, 'non-target-aware'):
                FrozenTabICLEncoder(checkpoint, schema)


if __name__ == '__main__':
    unittest.main()
