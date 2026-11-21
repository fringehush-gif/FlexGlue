import math
import unittest

import torch

from flexglue.utils.ops import sorted_entmax15_threshold


class EntmaxThresholdGradientTest(unittest.TestCase):
    def test_inactive_zero_root_does_not_poison_selected_gradient(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                scores = torch.tensor(
                    [0, -0.5, -0.5, -1, -1, -1, -1, -1],
                    dtype=dtype,
                    requires_grad=True,
                )
                # Prefix eight has radicand zero; selected prefix three has 5/18.
                threshold = sorted_entmax15_threshold(scores)
                threshold.sum().backward()
                tau = -1 / 3 - math.sqrt(5 / 18)
                active = [-tau, -0.5 - tau, -0.5 - tau]
                expected = scores.new_tensor(
                    [value / sum(active) for value in active] + [0] * 5
                )
                self.assertTrue(torch.isfinite(scores.grad).all())
                torch.testing.assert_close(threshold, threshold.new_tensor([tau]))
                torch.testing.assert_close(scores.grad, expected)


if __name__ == "__main__":
    unittest.main()
