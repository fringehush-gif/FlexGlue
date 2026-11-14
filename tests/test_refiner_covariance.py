"""Regression tests for refinement precision mixing and covariance."""
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from flexglue.refiner import (
    ConvRefiner,
    RoMav2Refiner,
    _covariance_factor,
    _interpolate_precision_factor,
)


def decimal(value):
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, Decimal):
        return value
    return Decimal.from_float(float(value))


def covariance_reference(terms, scale=(1, 1)):
    """Invert a weighted precision sum without subtracting its determinant."""
    with localcontext() as ctx:
        ctx.prec = 100
        terms = [(decimal(w), tuple(map(decimal, factor))) for w, factor in terms]
        p00 = sum((w * a * a for w, (a, b, c) in terms), Decimal(0))
        p01 = sum((w * a * b for w, (a, b, c) in terms), Decimal(0))
        p11 = sum((w * (b * b + c * c) for w, (a, b, c) in terms), Decimal(0))
        det = sum((w * w * a * a * c * c for w, (a, b, c) in terms), Decimal(0))
        for i, (wi, (a, b, c)) in enumerate(terms):
            for wj, (d, e, f) in terms[i + 1 :]:
                det += wi * wj * ((a * e - d * b) ** 2 + (a * f) ** 2 + (d * c) ** 2)
        sx, sy = map(decimal, scale)
        return (
            (sx * sx * p11 / det, -sx * sy * p01 / det),
            (-sx * sy * p01 / det, sy * sy * p00 / det),
        )


def bilinear_terms(source, y, x, size):
    h, w = source.shape[1:3]
    py = max(
        Fraction(0),
        min(
            Fraction(h - 1),
            Fraction(2 * y + 1, 2) * Fraction(h, size[0]) - Fraction(1, 2),
        ),
    )
    px = max(
        Fraction(0),
        min(
            Fraction(w - 1),
            Fraction(2 * x + 1, 2) * Fraction(w, size[1]) - Fraction(1, 2),
        ),
    )
    y0, x0 = py.numerator // py.denominator, px.numerator // px.denominator
    wy, wx = py - y0, px - x0
    return [
        (ay * ax, source[0, iy, ix].tolist())
        for iy, ay in ((y0, 1 - wy), (min(y0 + 1, h - 1), wy))
        for ix, ax in ((x0, 1 - wx), (min(x0 + 1, w - 1), wx))
    ]


class RefinerCovarianceTest(unittest.TestCase):
    def assert_covariance(
        self, actual, reference, directions, weak_tol=2e-6, tol=2e-12
    ):
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        with localcontext() as ctx:
            ctx.prec = 100
            got = tuple(tuple(decimal(v) for v in row) for row in actual.tolist())
            error = sum(
                (got[i][j] - reference[i][j]) ** 2 for i in range(2) for j in range(2)
            )
            norm = sum(v * v for row in reference for v in row)
            self.assertLessEqual((error / norm).sqrt(), decimal(tol))
            for i in range(2):
                for j in range(2):
                    self.assertLessEqual(
                        abs(got[i][j] - reference[i][j]),
                        decimal(tol) * (1 + abs(reference[i][j])),
                    )
            self.assertGreater(got[0][0], 0)
            self.assertGreater(got[0][0] * got[1][1] - got[0][1] * got[1][0], 0)
            for direction in directions:
                v = tuple(map(decimal, direction))
                q = sum(v[i] * got[i][j] * v[j] for i in range(2) for j in range(2))
                expected = sum(
                    v[i] * reference[i][j] * v[j] for i in range(2) for j in range(2)
                )
                self.assertLessEqual(
                    abs(q - expected), decimal(weak_tol) * (1 + abs(expected))
                )

    @staticmethod
    def tiny_refiner(last):
        ref = RoMav2Refiner.__new__(RoMav2Refiner)
        nn.Module.__init__(ref)
        ref.scales = (2, 1)
        ref.refiners = nn.ModuleDict(
            {str(s): ConvRefiner(2, 2, 2, None, hidden_blocks=0) for s in ref.scales}
        )
        with torch.no_grad():
            for stage in ref.refiners.values():
                for parameter in stage.parameters():
                    parameter.zero_()
                stage.confidence_head.bias.copy_(torch.tensor([0.2, 0.0, 1.0, last]))
        return ref.eval()

    def test_actual_two_stage_adverse_precision(self):
        for last in (-9.0, -10.0):
            for size in ((13, 20), (19, 14)):
                with self.subTest(last=last, size=size):
                    ref = self.tiny_refiner(last)
                    raw = []
                    hooks = [
                        stage.confidence_head.register_forward_hook(
                            lambda module, inputs, output: raw.append(
                                output.detach().permute(0, 2, 3, 1)
                            )
                        )
                        for stage in ref.refiners.values()
                    ]
                    features = {1: torch.zeros(1, 6, 6, 2), 2: torch.zeros(1, 3, 3, 2)}
                    try:
                        with torch.no_grad():
                            out = ref.refine(
                                torch.zeros(1, 2, 2, 2),
                                torch.zeros(1, 2, 2),
                                features,
                                features,
                                torch.tensor([size]),
                            )
                    finally:
                        for hook in hooks:
                            hook.remove()
                    self.assertEqual(len(raw), 2)
                    factors = [
                        torch.stack(
                            (
                                F.softplus(d[..., 1]) + 1e-6,
                                d[..., 2],
                                F.softplus(d[..., 3]) + 1e-6,
                            ),
                            -1,
                        )
                        for d in raw
                    ]
                    # Packing these FP32 factors loses the positive minor term.
                    for factor in factors:
                        self.assertTrue(
                            torch.equal(
                                factor[..., 1].square() + factor[..., 2].square(),
                                factor[..., 1].square(),
                            )
                        )
                    scale = (Fraction(size[0], 6), Fraction(size[1], 6))
                    with localcontext() as ctx:
                        ctx.prec = 100
                        for y in range(6):
                            for x in range(6):
                                terms = bilinear_terms(factors[0], y, x, (6, 6))
                                terms.append(
                                    (Fraction(1), factors[1][0, y, x].tolist())
                                )
                                reference = covariance_reference(terms, scale)
                                a, b, _ = terms[-1][1]
                                weak = (
                                    decimal(a) / decimal(scale[0]),
                                    decimal(b) / decimal(scale[1]),
                                )
                                self.assert_covariance(
                                    out["dense_covi_ij"][0, y, x], reference, [weak]
                                )

    def test_nonuniform_bilinear_precision_transport(self):
        source = torch.tensor(
            [
                [
                    [[1.25, 0.3, 0.7], [0.6, -0.8, 1.1], [1.5, 0.5, 0.8]],
                    [[0.9, 0.4, 0.6], [1.2, -0.6, 0.5], [0.7, 0.2, 1.3]],
                ]
            ],
            dtype=torch.float64,
        )
        moved = _interpolate_precision_factor(source, (5, 7))
        for y in range(5):
            for x in range(7):
                reference = covariance_reference(bilinear_terms(source, y, x, (5, 7)))
                factor = _covariance_factor(
                    moved[0, y, x], torch.ones(2, dtype=torch.float64)
                )
                self.assert_covariance(
                    factor @ factor.mT,
                    reference,
                    [(1, 0), (0, 1), (0.7, -0.4)],
                    weak_tol=2e-12,
                    tol=3e-13,
                )

    def test_tensor_only_stage_hook_requires_migration(self):
        features = {1: torch.zeros(1, 6, 6, 2), 2: torch.zeros(1, 3, 3, 2)}
        for scale in (2, 1):
            with self.subTest(scale=scale):
                ref = self.tiny_refiner(-9.0)
                hook = ref.refiners[str(scale)].register_forward_hook(
                    lambda module, inputs, output: (output[0], output[1].logits)
                )
                try:
                    with torch.no_grad(), self.assertRaisesRegex(
                        TypeError, "internal uncertainty state"
                    ):
                        ref.refine(
                            torch.zeros(1, 2, 2, 2),
                            torch.zeros(1, 2, 2),
                            features,
                            features,
                            torch.tensor([[13, 20]]),
                        )
                finally:
                    hook.remove()

    def test_direct_stage_tensor_contract(self):
        stage = ConvRefiner(2, 2, 2, None, hidden_blocks=0).eval()
        with torch.no_grad():
            for parameter in stage.parameters():
                parameter.zero_()
            stage.confidence_head.bias.copy_(torch.tensor([0.25, 0.0, 1.0, -9.0]))
        features = torch.zeros(1, 2, 3, 2)
        warp = torch.zeros(1, 2, 3, 2)
        logits = torch.full((1, 2, 3, 1), 0.1, requires_grad=True)
        raw = []
        hook = stage.confidence_head.register_forward_hook(
            lambda module, inputs, output: raw.append(output.permute(0, 2, 3, 1))
        )
        try:
            output_warp, output_logits = stage(
                features, features, warp, logits, (1.0, 1.0)
            )
        finally:
            hook.remove()
        self.assertIsInstance(output_warp, torch.Tensor)
        self.assertIsInstance(output_logits, torch.Tensor)
        self.assertTrue(torch.equal(output_warp, warp))
        self.assertEqual(len(raw), 1)
        d = raw[0]
        a = F.softplus(d[..., 1]) + 1e-6
        b = d[..., 2]
        c = F.softplus(d[..., 3]) + 1e-6
        packed = torch.stack((d[..., 0], a * a, a * b, b * b + c * c), -1)
        expected = F.pad(logits, (0, 3)) + packed
        torch.testing.assert_close(output_logits, expected, rtol=0, atol=0)
        variables = (logits, stage.confidence_head.bias)
        actual_grad = torch.autograd.grad(
            output_logits.square().sum(), variables, retain_graph=True
        )
        expected_grad = torch.autograd.grad(expected.square().sum(), variables)
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
