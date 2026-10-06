import unittest

import torch

from gen_tfm.model import TabularFlowBlock


class QueryAttentionTest(unittest.TestCase):
    def test_other_query_rows_cannot_change_a_query_output(self):
        torch.manual_seed(0)
        block = TabularFlowBlock(dim=16, n_heads=4).eval()
        context = torch.randn(2, 3, 16)
        queries = torch.randn(2, 4, 16)

        with torch.no_grad():
            original = block(queries, context, "context_plus_noisy_query")
            changed = queries.clone()
            changed[:, 1:] += 100 * torch.randn_like(changed[:, 1:])
            result = block(changed, context, "context_plus_noisy_query")

        torch.testing.assert_close(result[:, 0], original[:, 0], rtol=0, atol=1e-6)

    def test_query_output_is_independent_of_batch_chunking(self):
        torch.manual_seed(1)
        block = TabularFlowBlock(dim=16, n_heads=4).eval()
        context = torch.randn(1, 3, 16)
        queries = torch.randn(1, 5, 16)

        with torch.no_grad():
            together = block(queries, context, "context_plus_noisy_query")
            separate = torch.cat([
                block(queries[:, :2], context, "context_plus_noisy_query"),
                block(queries[:, 2:], context, "context_plus_noisy_query"),
            ], dim=1)

        torch.testing.assert_close(together, separate, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
