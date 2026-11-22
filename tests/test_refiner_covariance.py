"""Inversion checks against the precision stored by the stock refiner."""

import math
import unittest

import torch
from torch import nn

from flexglue.refiner import ConvRefiner, RoMav2Refiner


def legacy_covariance(precision, scale):
    p00, p10, p11 = precision.unbind(-1)
    adjugate = torch.stack([p11, -p10, -p10, p00], -1).unflatten(-1, (2, 2))
    cov = adjugate / (p00 * p11 - p10 * p10)[..., None, None]
    return cov * scale[..., :, None] * scale[..., None, :]


def stock_refiner(n, device):
    model = RoMav2Refiner.__new__(RoMav2Refiner)
    nn.Module.__init__(model)
    model.scales = (2, 1)
    model.refiners = nn.ModuleDict(
        {str(s): ConvRefiner(*model.configs[s]) for s in model.scales}
    )
    model = model.to(device).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        biases = {
            "2": (0.2, n, n - 1, -100),
            "1": (0.2, math.log(math.expm1(1 - 1e-6)), 1, -100),
        }
        for stride, bias in biases.items():
            target = model.refiners[stride].confidence_head.bias
            target.copy_(target.new_tensor(bias))
    features = {
        s: torch.zeros(1, 6 // s, 6 // s, model.configs[s][0], device=device)
        for s in model.scales
    }
    return model, features


def refine_precision(precision, size=None):
    class TensorRefiner(RoMav2Refiner):
        def __init__(self):
            nn.Module.__init__(self)

        def forward(self, feats_i, feats_j, warp, logits):
            return warp, torch.cat((logits[..., None], precision), -1)

    batch, h, w = precision.shape[:3]
    shape = (batch, h, w)
    features = {1: precision.new_empty((*shape, 1))}
    if size is None:
        size = precision.new_tensor([w, h]).expand(batch, -1)
    return TensorRefiner().refine(
        precision.new_zeros((*shape, 2)),
        precision.new_zeros(shape),
        features,
        features,
        size,
    )["dense_covi_ij"]


class RefinerCovarianceTest(unittest.TestCase):
    def test_actual_stages_return_finite_correct_covariance(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for n, stored in ((100, (10001, 9901, 9802)), (65, (4226, 4161, 4097))):
                with self.subTest(n=n, device=device), torch.inference_mode():
                    model, features = stock_refiner(n, device)
                    warp = torch.zeros(1, 1, 1, 2, device=device)
                    logits = torch.zeros(1, 1, 1, device=device)
                    size = torch.tensor([[6, 6]], device=device)
                    stage_calls = []

                    def observe(_stage, inputs, outputs):
                        self.assertIsInstance(inputs[3], torch.Tensor)
                        self.assertIsInstance(outputs[1], torch.Tensor)
                        stage_calls.append(inputs[3].shape[-1])

                    handles = [
                        stage.register_forward_hook(observe)
                        for stage in model.refiners.values()
                    ]
                    try:
                        _, packed = model(features, features, warp, logits)
                        p, q, r = stored
                        torch.testing.assert_close(
                            packed[..., 1:],
                            packed.new_tensor(stored).expand(1, 6, 6, 3),
                            rtol=0,
                            atol=0,
                        )
                        out = model.refine(warp, logits, features, features, size)
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(stage_calls, [1, 4, 1, 4])
                    expected = packed.new_tensor([[r, -q], [-q, p]]).expand(
                        1, 6, 6, 2, 2
                    )
                    torch.testing.assert_close(
                        out["dense_covi_ij"], expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        out["dense_confidencei"],
                        packed[..., 0].sigmoid(),
                        rtol=0,
                        atol=0,
                    )

    def test_mixed_ordinary_and_rescued_values_and_gradients(self):
        ordinary = torch.tensor([3.2, 0.35, 2.1], requires_grad=True)
        rescued = torch.tensor([4226.0, 4161.0, 4097.0], requires_grad=True)
        rows = torch.stack((ordinary, rescued)).reshape(1, 1, 2, 3)
        scale = torch.tensor([1.75, 2.25])
        position_size = (scale * torch.tensor([2, 1]))[None].double()
        actual = refine_precision(rows, position_size).reshape(2, 2, 2)
        expected = torch.stack(
            (
                legacy_covariance(ordinary, scale.double()),
                legacy_covariance(rescued.double(), scale.double()),
            )
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual.sum(), (ordinary, rescued))
        expected_grad = torch.autograd.grad(expected.sum(), (ordinary, rescued))
        self.assertTrue(all(value.isfinite().all() for value in actual_grad))
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=0)

    def test_determinant_range_and_non_fp32_contract(self):
        large, small = 2.0**100, 2.0**-100
        precision = torch.tensor([[[[large, 0, large], [small, 0, small]]]])
        expected = torch.diag_embed(precision[..., [0, 2]].reciprocal())
        torch.testing.assert_close(
            refine_precision(precision), expected, rtol=0, atol=0
        )
        for dtype in (torch.float16, torch.float64):
            value = torch.tensor([[[[1.5, 0.5, 2.0]]]], dtype=dtype)
            expected = legacy_covariance(value, value.new_ones(2))
            torch.testing.assert_close(
                refine_precision(value), expected, rtol=0, atol=0
            )

    def test_corrected_scaling_has_finite_size_gradients(self):
        large = 2.0**100
        precision = torch.tensor([[[[large, 0.0, large]]]])
        size = torch.full((1, 2), 2.0**40, requires_grad=True)
        covariance = refine_precision(precision, size)
        expected = torch.eye(2).reshape(1, 1, 1, 2, 2) * 2.0**-20
        torch.testing.assert_close(covariance, expected, rtol=0, atol=0)
        covariance.sum().backward()
        torch.testing.assert_close(
            size.grad, torch.full_like(size, 2.0**-59), rtol=0, atol=0
        )

    def test_already_singular_precision_is_not_reinterpreted(self):
        precision = torch.tensor([[[[1.0, 1.0, 1.0]]]])
        expected = legacy_covariance(precision, precision.new_ones(2))
        torch.testing.assert_close(
            refine_precision(precision), expected, rtol=0, atol=0, equal_nan=True
        )

    def test_forward_gradient_under_no_grad(self):
        precision = torch.tensor([[[[4226.0, 4161.0, 4097.0]]]])
        direction = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        forward_ad = torch.autograd.forward_ad
        with forward_ad.dual_level(), torch.no_grad():
            dual = forward_ad.make_dual(precision, direction)
            cov, tangent = forward_ad.unpack_dual(refine_precision(dual))
            column = cov[..., :1].double()
            expected = -(column @ column.mT)
            torch.testing.assert_close(tangent, expected.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
