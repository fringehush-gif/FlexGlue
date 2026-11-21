import unittest

import torch

from flexglue.utils import ops


def legacy_readout(logits, grid, hw, radius=1, k=512):
    # Frozen eager formula: center the full volume before gathering neighbors.
    if logits.shape[-1] > k:
        values, indices = logits.topk(k, dim=-1)
        maximum, index = values[..., :1], indices[..., :1]
        centered = (logits - maximum) / 2
        threshold = ops.sorted_entmax15_threshold((values - maximum) / 2)
    else:
        maximum, index = logits.max(-1, keepdim=True)
        centered = (logits - maximum) / 2
        threshold = ops.entmax15_threshold(centered, k)
    h, w = hw
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    y = (index // w)[..., None] + offsets[:, None]
    x = (index % w)[..., None] + offsets[None, :]
    inside = ((y >= 0) & (y < h) & (x >= 0) & (x < w)).flatten(-2)
    indices = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).flatten(-2)
    weights = (centered.gather(-1, indices) - threshold).clamp_min(0).square()
    position = (weights[..., None] * grid[indices]).sum(-2) / weights.sum(
        -1, keepdim=True
    )
    return position, (weights * inside).sum(-1)


class EntmaxReadoutTest(unittest.TestCase):
    def test_forward_matches_full_volume_reference(self):
        for dtype in (torch.float32, torch.float64):
            for batch, queries, h, w, radius, k in (
                (1, 0, 3, 5, 1, 8),
                (1, 1, 1, 1, 0, 8),
                (2, 7, 3, 5, 1, 8),
                (1, 9, 8, 9, 2, 16),
                (1, 5, 3, 7, 4, 32),
                (1, 3, 17, 31, 1, 512),
            ):
                with self.subTest(
                    dtype=dtype, shape=(batch, queries, h, w), radius=radius, k=k
                ):
                    torch.manual_seed(19)
                    logits = torch.rand(batch, h * w, queries, dtype=dtype).transpose(
                        1, 2
                    )
                    grid = ops.cell_centers(h, w, (w, h), device=logits.device).to(
                        dtype
                    )
                    actual = ops.entmax15_window_readout(
                        logits, grid, (h, w), radius, k
                    )
                    expected = legacy_readout(logits, grid, (h, w), radius, k)
                    for got, reference in zip(actual, expected):
                        self.assertTrue(torch.equal(got, reference))

    def test_ties_and_clamped_edges_preserve_selection(self):
        for k in (4, 32):
            logits = torch.zeros(1, 3, 15)
            grid = ops.cell_centers(3, 5, (5, 3), device=logits.device)
            for radius in (0, 1, 3):
                expected = legacy_readout(logits, grid, (3, 5), radius, k)
                actual = ops.entmax15_window_readout(logits, grid, (3, 5), radius, k)
                for got, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(got, reference))

    def test_gradients_match_full_volume_reference(self):
        torch.manual_seed(29)
        grid = ops.cell_centers(8, 9, (9, 8), device="cpu").double()
        logits = torch.rand(2, 7, 72, dtype=torch.float64, requires_grad=True)
        actual = ops.entmax15_window_readout(logits, grid, (8, 9), k=32)
        expected = legacy_readout(logits, grid, (8, 9), k=32)
        got = torch.autograd.grad(sum(x.sum() for x in actual), logits)[0]
        reference = torch.autograd.grad(sum(x.sum() for x in expected), logits)[0]
        torch.testing.assert_close(got, reference, rtol=1e-10, atol=1e-12)

    def test_float32_edge_and_tie_gradients(self):
        grid = ops.cell_centers(3, 5, (5, 3), device="cpu")
        for k in (4, 32):
            for radius in (0, 2, 4):
                with self.subTest(k=k, radius=radius):
                    logits = torch.zeros(1, 3, 15, requires_grad=True)
                    actual = ops.entmax15_window_readout(
                        logits, grid, (3, 5), radius, k
                    )
                    expected = legacy_readout(logits, grid, (3, 5), radius, k)
                    got = torch.autograd.grad(sum(x.sum() for x in actual), logits)[0]
                    reference = torch.autograd.grad(
                        sum(x.sum() for x in expected), logits
                    )[0]
                    torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
