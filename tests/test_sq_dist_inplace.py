"""Equivalence checks for squaring temporary distance buffers in place."""
import unittest
import warnings

import torch

from flexglue.utils import ops


def original(x, y):
    if x.shape[-1] == 0:
        return 0
    result = (x[..., 0, None] - y[..., 0][:, None]).square()
    for i in range(1, x.shape[-1]):
        result.add_((x[..., i, None] - y[..., i][:, None]).square())
    return result


class SquaredDistanceTemporaryTest(unittest.TestCase):
    def test_values_inputs_empty_nonfinite(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for m, n, d in ((5, 7, 4), (0, 7, 4), (5, 0, 4), (5, 7, 1)):
                with self.subTest(dtype=dtype, shape=(m, n, d)):
                    x = torch.randn(1, d, m, dtype=dtype).transpose(1, 2)
                    y = torch.randn(1, d, n, dtype=dtype).transpose(1, 2)
                    x0, y0 = x.clone(), y.clone()
                    torch.testing.assert_close(
                        ops.sq_dist(x, y),
                        original(x, y),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    self.assertTrue(torch.equal(x, x0) and torch.equal(y, y0))
        self.assertEqual(ops.sq_dist(torch.empty(1, 2, 0), torch.empty(1, 3, 0)), 0)
        x = torch.tensor([[[0.0, -0.0], [float("inf"), float("nan")]]])
        y = torch.tensor([[[-0.0, 0.0], [float("inf"), 1.0]]])
        actual, expected = ops.sq_dist(x, y), original(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(
            torch.equal(
                torch.signbit(actual[actual == 0]),
                torch.signbit(expected[expected == 0]),
            )
        )

    def test_first_and_second_derivatives(self):
        torch.manual_seed(8)
        x = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        y = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
        out, ref = ops.sq_dist(x, y), original(x, y)
        g = torch.autograd.grad(out.sum(), (x, y), create_graph=True)
        h = torch.autograd.grad(ref.sum(), (x, y), create_graph=True)
        for actual, expected in zip(g, h):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        vx = torch.arange(1, x.numel() + 1, dtype=x.dtype).reshape_as(x) / 7
        vy = torch.arange(1, y.numel() + 1, dtype=y.dtype).reshape_as(y).square() / 13
        gg = torch.autograd.grad((g[0] * vx).sum() + (g[1] * vy).sum(), (x, y))
        hh = torch.autograd.grad((h[0] * vx).sum() + (h[1] * vy).sum(), (x, y))
        self.assertTrue(any(bool(torch.count_nonzero(value)) for value in hh))
        for actual, expected in zip(gg, hh):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vmap_jvp_and_fullgraph_eager(self):
        torch.manual_seed(9)
        x = torch.randn(3, 1, 4, 3)
        y = torch.randn(1, 5, 3)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            actual = torch.vmap(ops.sq_dist, in_dims=(0, None))(x, y)
            actual_y = torch.vmap(ops.sq_dist, in_dims=(None, 0))(y, x)
        self.assertFalse(any("batching rule" in str(w.message).lower() for w in seen))
        torch.testing.assert_close(
            actual, torch.vmap(original, in_dims=(0, None))(x, y), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_y, torch.vmap(original, in_dims=(None, 0))(y, x), rtol=0, atol=0
        )
        dx, dy = torch.randn_like(x[0]), torch.randn_like(y)
        primal, tangent = torch.func.jvp(ops.sq_dist, (x[0], y), (dx, dy))
        ref, ref_tangent = torch.func.jvp(original, (x[0], y), (dx, dy))
        torch.testing.assert_close(primal, ref, rtol=0, atol=0)
        torch.testing.assert_close(tangent, ref_tangent, rtol=0, atol=0)
        compiled = torch.compile(ops.sq_dist, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x[0], y), original(x[0], y), rtol=0, atol=0)
