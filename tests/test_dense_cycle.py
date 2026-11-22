import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.utils import ops


class DenseCycleTest(unittest.TestCase):
    def test_dense_gate_respects_cycle_and_sampling_overrides(self):
        class CycleOverride(FlexGlue):
            def cycle_error(self, context, i, points, target, size, dense=None):
                self.calls.append(i)
                return points.new_full(points.shape[:-1], 100), points.new_ones(
                    points.shape[:-1]
                )

        class SamplingOverride(FlexGlue):
            def sample_correspondences(self, dense, points):
                self.calls.append(points.shape[1])
                return points.new_full(points.shape, 2), points.new_ones(
                    points.shape[:-1]
                )

        cases = (
            (CycleOverride, nullcontext(), False),
            (SamplingOverride, nullcontext(), False),
            (
                FlexGlue,
                patch.object(FlexGlue, "cycle_error", CycleOverride.cycle_error),
                False,
            ),
            (
                FlexGlue,
                patch.object(
                    FlexGlue,
                    "sample_correspondences",
                    SamplingOverride.sample_correspondences,
                ),
                False,
            ),
            (FlexGlue, nullcontext(), True),
        )
        for cls, override, rebound in cases:
            with self.subTest(
                override=cls.__name__, patch=type(override).__name__, rebound=rebound
            ), override:
                model = cls(
                    weights=None,
                    describe=False,
                    input_dim=4,
                    descriptor_dim=8,
                    search_dim=4,
                    num_heads=2,
                    n_layers=0,
                ).eval()
                model.calls = []
                calls = model.calls
                if rebound:
                    other = SamplingOverride(
                        weights=None,
                        describe=False,
                        input_dim=4,
                        descriptor_dim=8,
                        search_dim=4,
                        num_heads=2,
                        n_layers=0,
                    ).eval()
                    other.calls = []
                    calls = other.calls
                    model.cycle_error = MethodType(FlexGlue.cycle_error, other)
                size = torch.tensor([[16.0, 16.0]])
                data = {"image_size": size, "resized_size": size}
                warp = ops.cell_centers(2, 2, (2, 2)).view(1, 2, 2, 2)
                pred = {}
                for i, j in ((0, 1), (1, 0)):
                    pred[f"dense_p2d{i}_n{j}"] = warp.clone()
                    pred[f"dense_p2d{i}_i{j}"] = ops.denormalize_keypoints(warp, size)
                    pred[f"dense_confidence{i}"] = torch.ones(1, 2, 2)
                conf = SimpleNamespace(**{**model.default_conf, "max_distance_px": 1.0})
                result = model.gate_dense(
                    pred,
                    SimpleNamespace(covisibility_score=torch.ones(1)),
                    data,
                    data,
                    conf,
                )
                self.assertEqual(len(calls), 2)
                for i in (0, 1):
                    self.assertFalse(result[f"dense_valid{i}"].any())

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
        with patch.object(ops, "sample_features", wraps=ops.sample_features) as sample:
            result = model.gate_dense(dict(pred), context, *data, conf)
        self.assertEqual(sample.call_count, 2)  # one location sample per direction
        for i in (0, 1):
            torch.testing.assert_close(result[f"dense_valid{i}"], expected[i])
            torch.testing.assert_close(
                result[f"dense_confidence{i}"], expected[i].float() * 0.75
            )


if __name__ == "__main__":
    unittest.main()
