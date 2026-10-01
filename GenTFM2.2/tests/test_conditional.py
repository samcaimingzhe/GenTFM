"""Conditional information flow, split isolation and raw-scale generation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data.encoding import Schema, batch_feature_mask
from inference.generation import generate_in_context
from inference.sampling import sample_table
from model import GenTFM
from script.train import parse_args
from training.checkpoint import load_pretrained, load_training_checkpoint
from training.flow_matching import split_context_target, masked_velocity_loss, velocity_weights


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.schema = Schema(2, 1, 2)
        self.metadata = {"n_cont": 2, "n_cat": 1, "cat_cardinalities": [2]}
        self.mask = batch_feature_mask([self.metadata], *self.schema.as_tuple(), device="cpu")
        self.context = torch.zeros(1, 8, 6)
        self.context[0, :, 0] = torch.linspace(-2, 2, 8)
        self.context[0, :, 1] = self.context[0, :, 0].square()
        self.context[..., 2] = 1
        self.context[..., 4:] = 1
        self.model = GenTFM(max_features=6, embed_dim=8, num_col_blocks=1,
                            num_row_blocks=1, num_cross_blocks=1, nhead=2,
                            dim_feedforward=16, num_inds=3,
                            max_cont=2, max_cat=1, cat_cardinality=2)
        self.model.eval()

    def test_context_changes_output_permutation_and_cache_equivalence(self):
        x = torch.randn(1, 3, 6)
        t = torch.tensor([.4])
        a = self.model(x, t, self.mask, self.context)
        changed = self.context.clone()
        changed[..., 1] = -changed[..., 1]
        b = self.model(x, t, self.mask, changed)
        self.assertGreater(float((a-b).abs().max()), 1e-5)
        perm = torch.tensor([7, 2, 1, 5, 0, 3, 6, 4])
        torch.testing.assert_close(a, self.model(x, t, self.mask, self.context[:, perm]), atol=2e-6, rtol=2e-5)
        cached = self.model.encode_context(self.context, self.mask)
        torch.testing.assert_close(a, self.model(x, t, self.mask, context_embeddings=cached))
        a.square().mean().backward()
        for module in (self.model.context_col_embedding, self.model.context_row_interaction,
                       self.model.context_projection, self.model.cross_attention):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_context_joint_relationship_changes_output_with_equal_marginals(self):
        positive = self.context.clone()
        positive[..., 1] = positive[..., 0]
        negative = positive.clone()
        negative[..., 1] = -negative[..., 0]
        for j in (0, 1):
            torch.testing.assert_close(positive[..., j].sort().values, negative[..., j].sort().values)
        x, t = torch.randn(1, 3, 6), torch.tensor([.4])
        a = self.model(x, t, self.mask, positive)
        b = self.model(x, t, self.mask, negative)
        self.assertGreater(float((a-b).abs().max()), 1e-6)

    def test_missing_or_empty_context_rejected(self):
        x = torch.zeros(1, 3, 6)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.model(x, torch.tensor([.5]), self.mask)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.model(x, torch.tensor([.5]), self.mask, self.context[:, :0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.model(x, torch.tensor([.5]), self.mask, self.context,
                       context_embeddings=self.model.encode_context(self.context, self.mask))
        missing = self.context.clone()
        missing[0, 0, 4] = 0
        with self.assertRaisesRegex(ValueError, "missing observations"):
            self.model.encode_context(missing, self.mask)
        with self.assertRaises(ValueError):
            sample_table(self.model, self.context[:, :0], self.mask, 3)

    def test_disjoint_split_and_scaling_does_not_read_target(self):
        kwargs = dict(min_context=3, max_context=3, min_target=2)
        # Recover the same sampled raw context rows (using no schema scaling).
        raw_context, raw_target = split_context_target(self.context, self.mask,
            generator=torch.Generator().manual_seed(10), **kwargs)
        ctx_ids = set(raw_context[0, :, 0].tolist())
        target_ids = set(raw_target[0, :, 0].tolist())
        self.assertFalse(ctx_ids & target_ids)
        self.assertEqual(len(ctx_ids | target_ids), 8)
        context, target = split_context_target(self.context, self.mask, schema=self.schema,
            generator=torch.Generator().manual_seed(10), **kwargs)
        mu = raw_context[..., :2].mean(1, keepdim=True)
        std = raw_context[..., :2].std(1, keepdim=True, correction=0) + 1e-6
        torch.testing.assert_close(context[..., :2], (raw_context[..., :2]-mu)/std)
        torch.testing.assert_close(target[..., :2], (raw_target[..., :2]-mu)/std)
        modified = self.context.clone()
        for i in range(8):
            if float(modified[0, i, 0]) in target_ids:
                modified[0, i, :2] += 100
        context2, target2 = split_context_target(modified, self.mask, schema=self.schema,
            generator=torch.Generator().manual_seed(10), **kwargs)
        torch.testing.assert_close(context, context2, rtol=0, atol=0)
        self.assertFalse(torch.equal(target, target2))
        with self.assertRaisesRegex(ValueError, "context_sizes"):
            split_context_target(self.context, self.mask, context_sizes=[500])

    def test_weighted_velocity_formula(self):
        weights = velocity_weights(self.mask, self.schema, .05)
        loss = masked_velocity_loss(torch.ones(1, 3, 6), torch.zeros(1, 3, 6),
                                    weights.bool(), weights)
        self.assertAlmostEqual(float(loss), 1, places=6)
        pred = torch.zeros(1, 3, 6)
        pred[..., 2:4] = 2
        loss = masked_velocity_loss(pred, torch.zeros_like(pred), weights.bool(), weights)
        self.assertAlmostEqual(float(loss), .4/2.1, places=6)

    def test_sampling_encodes_once_reproducible_and_restores_mode(self):
        self.model.train()
        self.model.context_col_embedding.eval()
        with patch.object(self.model, 'encode_context', wraps=self.model.encode_context) as encode:
            a = sample_table(self.model, self.context, self.mask, 4, n_steps=3,
                             generator=torch.Generator().manual_seed(18))
            self.assertEqual(encode.call_count, 1)
        b = sample_table(self.model, self.context, self.mask, 4, n_steps=3,
                         generator=torch.Generator().manual_seed(18))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.context_col_embedding.training)

    def test_raw_wrapper_restores_scale_and_500_row_shapes(self):
        # Zero flow velocity => generated numerical coordinates remain noise.
        for p in self.model.velocity_head.parameters():
            p.data.zero_()
        raw = self.context[0].numpy().copy()
        raw[:, :2] = raw[:, :2] * np.array([3., 5.]) + np.array([10., -20.])
        result = generate_in_context(self.model, raw, self.metadata, 5, n_steps=2,
                                    generator=torch.Generator().manual_seed(19))
        noise = torch.randn(1, 5, 6, generator=torch.Generator().manual_seed(19)).numpy()[0]
        expected = noise[:, :2] * (raw[:, :2].std(0)+1e-6) + raw[:, :2].mean(0)
        np.testing.assert_allclose(result[:, :2], expected, rtol=2e-6, atol=2e-6)
        large_context = np.tile(raw, (63, 1))[:500]
        result = generate_in_context(self.model, large_context, self.metadata, 500, n_steps=1,
                                    generator=torch.Generator().manual_seed(20))
        self.assertEqual(result.shape, (500, 6))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result[:, 4:] == 1).all())
        self.assertTrue((result[:, 2:4].sum(1) == 1).all())

    def test_old_checkpoint_and_invalid_training_plan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.pt'
            torch.save({'model_config': {'max_features': 6}, 'model_state_dict': {}}, path)
            with self.assertRaisesRegex(ValueError, "not a conditional"):
                load_pretrained(path)
            torch.save({'step': 1, 'model_config': {'max_features': 6}, 'model_state_dict': {},
                        'optimizer_state_dict': {}, 'scheduler_state_dict': {},
                        'best_loss': 1., 'train_config': {'steps': 2}}, path)
            with self.assertRaisesRegex(ValueError, "not a conditional"):
                load_training_checkpoint(path)
        with patch('sys.argv', ['train']):
            args = parse_args()
        self.assertEqual((args.num_rows, args.max_context, args.min_target), (1024, 500, 512))
        with patch('sys.argv', ['train', '--num-rows', '500']), patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == '__main__':
    unittest.main()
