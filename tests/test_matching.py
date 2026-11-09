import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from flexglue import FlexGlue


class MatchingContractsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
            mp=None,
        ).eval()
        self.data = [
            {
                "features": torch.rand(2, 4, 2, 3),
                "image_size": torch.tensor([[24.0, 16.0], [24.0, 16.0]]),
            }
            for _ in range(2)
        ]

    def test_invalid_attention_dimensions_fail_before_allocation(self):
        for options in (
            {"num_heads": 0},
            {"num_heads": -1},
            {"num_heads": True},
            {"descriptor_dim": 0},
            {"descriptor_dim": 15, "num_heads": 2},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError), patch(
                "torch.load", side_effect=AssertionError("loaded weights")
            ):
                FlexGlue(**options)

    def test_empty_compile_lengths_keep_eager_head(self):
        points = torch.zeros(1, 2, 2)
        features = torch.ones(1, 8, 2, 3)
        expected = self.model.heads[0](points, features, features)
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
            self.model.compile(static_lengths=())
        with patch.object(
            self.model.heads[0],
            "compiled_forward",
            side_effect=AssertionError("compiled head"),
        ):
            actual = self.model.run_head(0, points, features, features)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))

    def test_compile_lengths_keep_padding_and_large_query_fallback(self):
        points = torch.zeros(1, 3, 2)
        features = torch.ones(1, 8, 2, 3)
        expected = self.model.run_head(0, points, features, features)
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
            self.model.compile(static_lengths=(2,))
        with patch.object(
            self.model.heads[0],
            "compiled_forward",
            side_effect=AssertionError("padded beyond maximum"),
        ):
            actual = self.model.run_head(0, points, features, features)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))
        with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
            self.model.compile(static_lengths=(4,))
        with patch.object(
            self.model.heads[0],
            "compiled_forward",
            wraps=self.model.heads[0].compiled_forward,
        ) as head:
            output = self.model.run_head(0, points, features, features)
        self.assertEqual(head.call_args.args[0].shape[1], 4)
        self.assertEqual(output[0].shape[1], 3)

    def test_sparse_empty_keypoints(self):
        for n0, n1 in ((0, 3), (3, 0), (0, 0)):
            for gemm in (False, True):
                with self.subTest(n0=n0, n1=n1, gemm=gemm), torch.no_grad():
                    inputs = [
                        {**d, "keypoints": torch.rand(2, n, 2) * 10 + 1}
                        for d, n in zip(self.data, (n0, n1))
                    ]
                    result = self.model.match(*inputs, sparse=True, gemm_nn=gemm)
                    for i, n in enumerate((n0, n1)):
                        matches = result[f"matches{i}"]
                        scores = result[f"matching_scores{i}"]
                        self.assertEqual(matches.shape, (2, n))
                        self.assertEqual(matches.dtype, torch.long)
                        self.assertEqual(matches.device, inputs[i]["keypoints"].device)
                        self.assertTrue(torch.all(matches == -1))
                        self.assertEqual(scores.shape, (2, n))
                        self.assertEqual(scores.dtype, result[f"confidence{i}"].dtype)
                        self.assertTrue(torch.all(scores == 0))

    def test_negative_distance_disables_dense_cycle(self):
        for symmetric in (False, True):
            with self.subTest(symmetric=symmetric), torch.no_grad():
                zero = self.model.match_dense(
                    *self.data, symmetric=symmetric, max_distance_px=0
                )
                negative = self.model.match_dense(
                    *self.data, symmetric=symmetric, max_distance_px=-1
                )
                self.assertEqual(zero.keys(), negative.keys())
                for key in zero:
                    if isinstance(zero[key], torch.Tensor):
                        self.assertTrue(torch.equal(zero[key], negative[key]), key)
                    else:
                        self.assertEqual(zero[key], negative[key])

    def test_negative_distance_refine_does_not_add_reverse_grid(self):
        context = object()
        conf = SimpleNamespace(refine=True, max_distance_px=-1)
        with patch.object(self.model, "grid_correspondences", return_value={}) as grid:
            self.model.prepare_dense(context, *self.data, [(0, 1)], conf)
        grid.assert_called_once()

    def test_refinement_reuses_each_supplied_feature_packet(self):
        self.model.refiner = torch.nn.Module()
        self.model.refiner.extract = Mock(return_value={"fallback": torch.ones(1)})
        cached = [{"cached": torch.ones(1)}, {"cached": torch.zeros(1)}]
        conf = self.model.resolve_conf({"refine": True})
        for missing in (None, 0, 1):
            with self.subTest(missing=missing):
                self.model.refiner.extract.reset_mock()
                inputs = [
                    {**data, "fine_features": features}
                    for data, features in zip(self.data, cached)
                ]
                if missing is not None:
                    inputs[missing]["fine_features"] = None
                    inputs[missing]["image"] = torch.zeros(2, 3, 8, 8)
                context = self.model.decode_pair(*inputs, conf)
                for i in (0, 1):
                    expected = (
                        self.model.refiner.extract.return_value
                        if i == missing
                        else cached[i]
                    )
                    self.assertIs(context.fine[i], expected)
                if missing is None:
                    self.model.refiner.extract.assert_not_called()
                else:
                    self.model.refiner.extract.assert_called_once_with(
                        inputs[missing]["image"], resize=conf.refine_resize
                    )


if __name__ == "__main__":
    unittest.main()
