import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
