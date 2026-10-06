import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gen_tfm import (GenTFM, Schema, build_encoded_table, decode_components,
                     decode_to_dataframe, encode_dataframe, generate_in_context,
                     generate_zero_context, load_pretrained)
from gen_tfm.baselines import MIXED_BASELINES
from gen_tfm.checkpoint import export_slim_checkpoint, save_training_checkpoint
from gen_tfm.encoding import (category_bits_to_ids, category_ids_to_bits,
                              float32_bits_to_values, float32_values_to_bits,
                              mixed_feature_mask, sanitize_mixed_encoded)
from gen_tfm.flow_matching import binary_flow_step, xor_corrupt
from gen_tfm.metrics import evaluate_mixed, feature_matrix_without_label
from gen_tfm.model import GenTFM as Network
from gen_tfm.prior import PriorConfig, TabICLPriorEngine


class BinaryFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)
        self.schema = Schema(2, 2, 5)
        self.values = np.column_stack([np.linspace(-10, 20, 48), np.full(48, 7)]).astype(np.float32)
        self.x, self.meta = build_encoded_table(
            "table", self.values, [np.arange(48) % 3, np.arange(48) % 5], ["y", "cat"], self.schema)

    def model(self):
        return GenTFM(max_cont=2, max_cat=2, cat_cardinality=5, hidden_dim=16,
                      n_heads=2, n_enc_layers=1, n_flow_layers=1, time_dim=8)

    def test_ieee754_roundtrip_is_bit_exact(self):
        # Includes subnormal, negative zero and signs; no quantization to integers.
        values = np.array([0, -0.0, 1, -2.5, np.nextafter(np.float32(0), np.float32(1)),
                           np.finfo(np.float32).max, np.inf, -np.inf], dtype=np.float32)
        bits = float32_values_to_bits(values)
        self.assertEqual(bits.shape, (8, 32))
        np.testing.assert_array_equal(float32_bits_to_values(bits).view(np.uint32), values.view(np.uint32))
        tb = float32_values_to_bits(torch.from_numpy(values))
        np.testing.assert_array_equal(tb.numpy(), bits)
        np.testing.assert_array_equal(float32_bits_to_values(tb).numpy().view(np.uint32), values.view(np.uint32))
        np.testing.assert_array_equal(bits[2, :9], [0, 0, 1, 1, 1, 1, 1, 1, 1])

    def test_minmax_float32_and_constant_column(self):
        comp = decode_components(self.x, self.meta, *self.schema.as_tuple())
        self.assertEqual(self.x.dtype, np.float32)
        self.assertEqual(comp["cont"].dtype, np.float32)
        self.assertEqual(self.x.shape[1], 2 * 32 + 2 * 3 + 2)
        np.testing.assert_array_equal(np.unique(self.x), [0, 1])
        np.testing.assert_allclose(comp["cont"][:, 0], np.linspace(0, 1, 48), atol=1e-7)
        np.testing.assert_array_equal(comp["cont"][:, 1], 0)
        decoded = decode_to_dataframe(self.x, self.meta, self.schema)
        np.testing.assert_allclose(decoded[["x0", "x1"]], self.values, atol=3e-6)
        self.assertEqual(decoded.x0.dtype, np.float32)

    def test_categories_zero_based_binary_and_padding(self):
        ids = np.arange(5)
        bits = category_ids_to_bits(ids, 3)
        np.testing.assert_array_equal(category_bits_to_ids(bits), ids)
        np.testing.assert_array_equal(bits[0], [0, 0, 0])
        np.testing.assert_array_equal(bits[4], [1, 0, 0])
        meta = dict(self.meta, n_cont=1, cat_cardinalities=[2, 5])
        mask = mixed_feature_mask(meta, *self.schema.as_tuple())
        self.assertTrue(mask[:32].all())
        self.assertFalse(mask[32:64].any())
        np.testing.assert_array_equal(mask[64:67], [False, False, True])

    def test_invalid_float_and_category_patterns_are_repaired(self):
        x = self.x.copy()
        x[:3, :32] = float32_values_to_bits(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
        x[:, self.schema.cat_start:self.schema.cat_start + 3] = 1
        x[:, self.schema.mask_start:] = 0
        cleaned = sanitize_mixed_encoded(x, self.meta, *self.schema.as_tuple())
        comp = decode_components(cleaned, self.meta, *self.schema.as_tuple())
        np.testing.assert_array_equal(comp["cont"][:3, 0], [0, 1, 0])
        self.assertTrue(np.all(comp["cats"][:, 0] < 3))
        self.assertTrue(np.all(comp["obs"] == 1))
        np.testing.assert_array_equal(cleaned, sanitize_mixed_encoded(cleaned, self.meta, *self.schema.as_tuple()))

    def test_xor_probability_path_endpoints_and_marginal(self):
        clean = torch.ones(1, 100000, 1)
        self.assertTrue(torch.equal(xor_corrupt(clean, torch.ones(1)), clean))
        noisy = xor_corrupt(clean, torch.zeros(1))
        self.assertAlmostEqual(noisy.mean().item(), 0.5, delta=0.008)
        halfway = xor_corrupt(clean, torch.full((1,), 0.5))
        self.assertAlmostEqual(halfway.mean().item(), 0.75, delta=0.008)

    def test_flow_probability_velocity_advances_correct_marginal(self):
        # Oracle clean bit=1: p_t(1)=0.5+0.5*t, so .4 -> .6 gives .7 -> .8.
        bits = (torch.rand(100000) < 0.7).float()
        advanced = binary_flow_step(bits, torch.ones_like(bits), t=0.4, dt=0.2)
        self.assertAlmostEqual(advanced.mean().item(), 0.8, delta=0.008)
        final = binary_flow_step(advanced, torch.ones_like(bits), t=0.6, dt=0.4)
        self.assertTrue(bool((final == 1).all()))
        final = binary_flow_step(advanced, torch.zeros_like(bits), t=0.6, dt=0.4)
        self.assertTrue(bool((final == 0).all()))

    def test_dataframe_regression_and_mixed_task_loss_backward(self):
        df = pd.DataFrame({"y": np.linspace(100, 200, 48), "x": np.linspace(-3, 4, 48),
                           "cat": ["a", "b", "c"] * 16})
        reg, meta = encode_dataframe(df, self.schema, target_col="y", target_task="regression")
        np.testing.assert_allclose(decode_to_dataframe(reg, meta, self.schema).y, df.y, atol=1e-5)
        model = self.model()
        batch = torch.from_numpy(np.stack([self.x, reg]))
        loss = model.compute_loss(batch, [self.meta, meta], min_ctx=4, max_ctx=8, min_target=16, context_sizes=[4, 8])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.flow_net.velocity_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.flow_net.y_cat_embed.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.flow_net.y_cont_embed[0].weight.grad.abs().sum().item(), 0)
        y, is_cat, keep = model._extract_y(batch, [self.meta, meta])
        self.assertFalse(bool(keep[1, :32].any()))
        np.testing.assert_allclose(y[1], np.linspace(0, 1, 48), atol=1e-7)
        self.assertEqual(is_cat.tolist(), [True, False])

    def test_binary_sampling_both_targets_and_both_legacy_solver_flags(self):
        model = self.model().eval()
        seen = []
        def inspect_state(module, args):
            state = args[0]
            self.assertTrue(bool(((state == 0) | (state == 1)).all()))
            seen.append(1)
        hook = model.flow_net.register_forward_pre_hook(inspect_state)
        for method in ["euler", "heun"]:
            result = model.generate(torch.from_numpy(self.x[:8]), metadata=[self.meta], num_gen=16,
                                    n_steps=4, method=method, target_y=2,
                                    categorical_context_calibration=True, cat_context_alpha=3.5)
            comp = decode_components(result[0].numpy(), self.meta, *self.schema.as_tuple())
            np.testing.assert_array_equal(comp["cats"][:, 0], 2)
            self.assertTrue(np.all(comp["cats"][:, 1] < 5))
            self.assertTrue(np.isfinite(comp["cont"]).all())
        reg_meta = dict(self.meta, label_type="continuous", label_cont_index=0, label_cat_index=None)
        result = model.generate(torch.from_numpy(self.x[:8]), metadata=[reg_meta], num_gen=16,
                                n_steps=4, target_y=0.3, sample_discrete=False)
        vals = float32_bits_to_values(result[0, :, :32]).numpy()
        np.testing.assert_array_equal(vals, np.full(16, 0.3, dtype=np.float32))
        hook.remove()
        self.assertGreater(len(seen), 8)

    def test_wrappers_baselines_and_metrics_use_new_codec(self):
        model = self.model().eval()
        for wrapper in [generate_in_context, generate_zero_context]:
            out = wrapper(model, self.x[:12], self.meta, 16, self.schema, n_steps=3)
            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(out.shape, (16, self.schema.encoded_dim))
        for baseline in MIXED_BASELINES.values():
            out = baseline(self.x[:12], self.meta, *self.schema.as_tuple(), 16, np.random.default_rng(1))
            self.assertTrue(np.all((out == 0) | (out == 1)))
            metrics = evaluate_mixed(self.x[:12], out, self.x[12:28], self.meta, *self.schema.as_tuple())
            self.assertTrue(np.isfinite(metrics["encoded_mmd"]))
        features, labels = feature_matrix_without_label(self.x, self.meta, *self.schema.as_tuple())
        keep = mixed_feature_mask(self.meta, *self.schema.as_tuple())
        keep[self.schema.cat_start:self.schema.cat_start + self.schema.cat_width] = False
        np.testing.assert_array_equal(features, self.x[:, keep])
        np.testing.assert_array_equal(labels, decode_components(self.x, self.meta, *self.schema.as_tuple())["cats"][:, 0])

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        loss = model.compute_loss(torch.from_numpy(self.x)[None], [self.meta], min_ctx=4, max_ctx=8, min_target=16)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "latest.pt"
            slim = Path(tmp) / "slim.pt"
            save_training_checkpoint(train, 1, model, optimizer, scheduler, float(loss.detach()))
            export_slim_checkpoint(train, slim)
            loaded, ckpt = load_pretrained(slim)
            self.assertEqual(ckpt["step"], 1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            old = torch.load(train, weights_only=False)
            old["model_config"].pop("representation_version")
            torch.save(old, train)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_pretrained(train)

    def test_network_file_and_modules_unchanged(self):
        self.assertEqual((ROOT / "gen_tfm/model.py").read_bytes(),
                         (ROOT.parent / "GenTFMv1.1/gen_tfm/model.py").read_bytes())
        adapter = self.model()
        config = adapter.config()
        config.pop("representation_version")
        original = Network(**config)
        self.assertEqual([(k, tuple(v.shape)) for k, v in adapter.state_dict().items()],
                         [(k, tuple(v.shape)) for k, v in original.state_dict().items()])

    def test_synthetic_adapter_classification_and_regression(self):
        # Isolate the codec adapter from the optional external prior dependency.
        X = np.random.default_rng(0).normal(size=(48, 8)).astype(np.float32)
        for task in ["classification", "regression"]:
            engine = TabICLPriorEngine.__new__(TabICLPriorEngine)
            engine.config = PriorConfig(target_task=task)
            engine.schema = self.schema
            engine.max_cont, engine.max_cat, engine.cat_cardinality = self.schema.as_tuple()
            engine.max_features = self.schema.encoded_dim
            engine.detect_feature_cats = False
            engine.force_feature_cats = 1
            engine.feature_cat_min_card, engine.feature_cat_max_card = 2, 5
            engine.generated_batches = 0
            engine.prior_type = "mix_scm"
            rows, meta = engine._encode_one_dataset(X, np.arange(48) % 3, 8, "train", 0)
            comp = decode_components(rows, meta, *self.schema.as_tuple())
            self.assertTrue(np.all((rows == 0) | (rows == 1)))
            self.assertTrue(np.all((comp["cont"] >= 0) & (comp["cont"] <= 1)))
            loss = self.model().compute_loss(torch.from_numpy(rows)[None], [meta], min_ctx=4, max_ctx=8, min_target=16)
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
