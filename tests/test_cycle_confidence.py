import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue


class CycleConfidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=1,
            mp=None,
        ).eval()
        self.data = {
            "features": torch.randn(1, 4, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
            "keypoints": torch.tensor([[[3.0, 3.0], [8.0, 8.0], [12.0, 12.0]]]),
        }

    def test_saturated_cycle_score_and_raw_confidence_have_finite_gradients(self):
        with torch.no_grad():
            self.model.heads[1].matchability.weight.zero_()
            self.model.heads[1].matchability.bias.fill_(-100)
        for distance in (1.0, 3.0):
            data = self.data
            with torch.no_grad():
                expected = self.model.match(data, data, max_distance_px=distance)
            for field in ("matching_scores", "confidence"):
                with self.subTest(distance=distance, field=field):
                    self.model.zero_grad(set_to_none=True)
                    pred = self.model.match(data, data, max_distance_px=distance)
                    loss = sum(pred[f"{field}{i}"].sum() for i in (0, 1))
                    self.assertEqual(loss.item(), 0)
                    for name in expected:
                        torch.testing.assert_close(
                            pred[name], expected[name], rtol=0, atol=0
                        )
                    loss.backward()
                    grads = [
                        p.grad for p in self.model.parameters() if p.grad is not None
                    ]
                    self.assertTrue(grads)
                    self.assertTrue(all(g.count_nonzero() == 0 for g in grads))

    def cycle_confidence(self, forward, backward):
        points = self.data["keypoints"] / 8 - 1
        with (
            patch.object(self.model, "run_head", return_value=(points, forward)),
            patch.object(
                self.model,
                "cycle_error",
                return_value=(torch.zeros_like(forward), backward),
            ),
        ):
            return self.model.match(self.data, self.data)["confidence0"]

    def test_zero_derivative_preserves_signed_zero_and_nonzero_forward_values(self):
        forward = torch.tensor(
            [[0.0, -0.0, 4.0]], dtype=torch.float64, requires_grad=True
        )
        backward = torch.tensor([[3.0, 2.0, 9.0]], requires_grad=True)
        raw = self.cycle_confidence(forward, backward)
        expected = (forward * backward).sqrt()
        torch.testing.assert_close(raw, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(raw.signbit(), expected.signbit()))
        for grad in torch.autograd.grad(raw.sum(), (forward, backward)):
            torch.testing.assert_close(grad[:, :2], torch.zeros_like(grad[:, :2]))

        forward = torch.tensor([[0.0, -1.0, float("nan")]], requires_grad=True)
        raw = self.cycle_confidence(forward, backward)
        torch.testing.assert_close(raw, (forward * backward).sqrt(), equal_nan=True)

    def test_positive_products_preserve_first_and_second_derivatives(self):
        forward = torch.tensor(
            [[0.25, 0.5, 0.75]], dtype=torch.float64, requires_grad=True
        )
        backward = torch.tensor(
            [[0.9, 0.6, 0.3]], dtype=torch.float64, requires_grad=True
        )
        inputs = (forward, backward)
        actual = self.cycle_confidence(*inputs)
        expected = (forward * backward).sqrt()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
        for grad, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(grad, reference, rtol=0, atol=0)

        # A stationary loss still has curvature when the product is positive.
        target = expected.detach()
        actual_grad = torch.autograd.grad(
            (actual - target).square().sum(), inputs, create_graph=True
        )
        expected_grad = torch.autograd.grad(
            (expected - target).square().sum(), inputs, create_graph=True
        )
        for grad, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(grad, reference, rtol=0, atol=0)
        actual_hess = torch.autograd.grad(sum(g.sum() for g in actual_grad), inputs)
        expected_hess = torch.autograd.grad(sum(g.sum() for g in expected_grad), inputs)
        for hess, reference in zip(actual_hess, expected_hess):
            torch.testing.assert_close(hess, reference, rtol=0, atol=0)
            self.assertTrue((hess != 0).any())


if __name__ == "__main__":
    unittest.main()
