import unittest
from types import SimpleNamespace

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


class DenseCycleTest(unittest.TestCase):
    @torch.inference_mode()
    def test_cycle_gate_with_different_grid_and_image_sizes(self):
        model = FlexGlue(
            weights=None,
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=0,
        ).eval()
        data = [
            {
                "image_size": torch.tensor([[96.0, 72.0]]),
                "resized_size": torch.tensor([[48.0, 24.0]]),
            },
            {
                "image_size": torch.tensor([[160.0, 180.0]]),
                "resized_size": torch.tensor([[80.0, 60.0]]),
            },
        ]
        warps = [
            ops.cell_centers(h, w, (w, h)).view(1, h, w, 2) for h, w in ((2, 3), (3, 4))
        ]
        warps[0][0, 0, 1, 0] += 0.25
        pred = {}
        for i, j in ((0, 1), (1, 0)):
            pred[f"dense_p2d{i}_n{j}"] = warps[i]
            pred[f"dense_p2d{i}_i{j}"] = ops.denormalize_keypoints(
                warps[i], data[j]["image_size"]
            )
            pred[f"dense_confidence{i}"] = torch.full(warps[i].shape[:-1], 0.75)
        expected = [
            torch.tensor([[[True, False, True], [True, True, True]]]),
            torch.tensor(
                [
                    [
                        [False, False, False, False],
                        [True, True, True, True],
                        [False, True, True, False],
                    ]
                ]
            ),
        ]
        conf = SimpleNamespace(**{**model.default_conf, "max_distance_px": 5.5})
        context = SimpleNamespace(covisibility_score=torch.ones(1))
        for i, j in ((0, 1), (1, 0)):
            dense_j = {
                "dense_p2di_nj": warps[j],
                "dense_confidencei": pred[f"dense_confidence{j}"],
            }
            error, _ = model.cycle_error(
                context,
                i,
                ops.grid_sources(warps[i]).flatten(1, 2),
                warps[i].flatten(1, 2),
                data[i]["resized_size"],
                dense_j,
            )
            torch.testing.assert_close(
                (error < conf.max_distance_px).view_as(expected[i]), expected[i]
            )
        result = model.gate_dense(dict(pred), context, *data, conf)
        for i in (0, 1):
            torch.testing.assert_close(result[f"dense_valid{i}"], expected[i])
            torch.testing.assert_close(
                result[f"dense_confidence{i}"], expected[i].float() * 0.75
            )


if __name__ == "__main__":
    unittest.main()
