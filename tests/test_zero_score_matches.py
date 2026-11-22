import unittest

import torch

from flexglue import FlexGlue
from flexglue.utils.ops import identity_matches


class MatchingValidityTest(unittest.TestCase):
    def setUp(self):
        self.model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
            mp=None,
            max_distance_px=0,
            max_num_samples=8,
        ).eval()
        # Uniform readouts predict the image center with confidence one.
        for parameter in self.model.parameters():
            torch.nn.init.zeros_(parameter)
        self.data = {
            "features": torch.zeros(1, 4, 2, 2),
            "image_size": torch.tensor([[16.0, 16.0]]),
        }

    def test_nonzero_scores_still_follow_the_threshold(self):
        scores = torch.tensor([[-0.2, -0.05, -0.0, 0.0, 0.1]])
        kpts = torch.zeros(1, 5, 2)
        result = identity_matches(kpts, kpts, scores, -0.1)
        expected_scores = torch.tensor([[0.0, -0.05, 0.0, 0.0, 0.1]])
        for i in (0, 1):
            torch.testing.assert_close(
                result[f"matches{i}"], torch.tensor([[-1, 1, -1, -1, 4]])
            )
            torch.testing.assert_close(result[f"matching_scores{i}"], expected_scores)

    @torch.inference_mode()
    def test_rejected_sampled_pairs_stay_unmatched(self):
        for kde_balance in (False, True):
            for threshold in (-0.1, torch.tensor(-0.1), 0.0, 0.1):
                with self.subTest(kde_balance=kde_balance, threshold=threshold):
                    result = self.model.match(
                        self.data,
                        self.data,
                        min_covisibility=2.0,  # Layer 0 has covisibility one.
                        filter_threshold=threshold,
                        kde_balance=kde_balance,
                    )
                    for i in (0, 1):
                        torch.testing.assert_close(
                            result[f"matches{i}"], torch.full((1, 8), -1)
                        )
                        torch.testing.assert_close(
                            result[f"matching_scores{i}"], torch.zeros(1, 8)
                        )

    @torch.inference_mode()
    def test_rejected_keypoints_do_not_hide_valid_matches(self):
        data = {
            **self.data,
            "keypoints": torch.tensor([[[-1.0, -1.0], [8.0, 8.0], [17.0, 17.0]]]),
        }
        cases = (
            (self.data, [-1, 1, -1]),
            (data, [-1, 1, -1, -1, 4, -1]),
        )
        for data1, matches in cases:
            expected = torch.tensor([matches])
            for threshold in (-0.1, torch.tensor(-0.1), 0.0, 0.1):
                with self.subTest(two_sided=data1 is data, threshold=threshold):
                    result = self.model.match(data, data1, filter_threshold=threshold)
                    for i in (0, 1):
                        torch.testing.assert_close(result[f"matches{i}"], expected)
                        torch.testing.assert_close(
                            result[f"matching_scores{i}"], (expected >= 0).float()
                        )


if __name__ == "__main__":
    unittest.main()
