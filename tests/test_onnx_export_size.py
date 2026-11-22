"""Export sizes must preserve the requested descriptor coordinate frame."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from onnx_test_utils import load_export
from torch import nn

from flexglue import FlexGlue
from flexglue.descriptor import ConvNeXt, DINOv3Descriptor, LinearConvHead


def small_descriptor():
    # Use the actual encode method and spatial operations without a checkpoint.
    descriptor = DINOv3Descriptor.__new__(DINOv3Descriptor)
    nn.Module.__init__(descriptor)
    descriptor.backbone = ConvNeXt(depths=(1, 1, 1, 1), dims=(8, 16, 24, 32))
    descriptor.head = LinearConvHead(in_dims=(8, 16, 24, 32), dim=16, out_dim=256)
    descriptor.register_buffer(
        "mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    )
    descriptor.register_buffer(
        "std", torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    )
    return descriptor


class ExportSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        import onnxruntime as ort

        ort.disable_telemetry_events()
        cls.export = load_export()

    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(20261122)
        self.matcher = FlexGlue(
            weights=None,
            describe=False,
            n_layers=1,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            mp=None,
        ).eval()

    def test_invalid_sizes_refuse_before_model_or_file_changes(self):
        class Descriptor(nn.Module):
            def encode(self, image):
                return image

        descriptor = Descriptor().train()
        invalid = (
            (33, 40),
            (32, 41),
            (32,),
            (32, 40, 8),
            (0, 32),
            (-8, 32),
            (True, 32),
            (32.0, 40),
            (float("nan"), 32),
            32,
            None,
            "32,40",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            path.write_bytes(b"existing graph")
            with patch.object(self.export.copy, "deepcopy") as copy_model, patch.object(
                torch.onnx, "export"
            ) as write:
                for size in invalid:
                    for name, model in (
                        ("export_matcher", self.matcher),
                        ("export_descriptor", descriptor),
                    ):
                        with self.subTest(size=size, export=name):
                            with self.assertRaisesRegex(ValueError, "Export size"):
                                getattr(self.export, name)(model, path, size, 18)
                            self.assertEqual(path.read_bytes(), b"existing graph")
                            self.assertEqual(list(path.parent.iterdir()), [path])
                            self.assertTrue(descriptor.training)
                copy_model.assert_not_called()
                write.assert_not_called()

    def test_matcher_refuses_stride_the_shipped_readers_do_not_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matcher.onnx"
            path.write_bytes(b"existing graph")
            for stride in (4, 16, 8.0, "8"):
                with self.subTest(stride=stride), patch.object(
                    self.matcher, "stride", stride
                ), patch.object(self.export.copy, "deepcopy") as copy_model:
                    with self.assertRaisesRegex(ValueError, "descriptor stride 8"):
                        self.export.export_matcher(self.matcher, path, (32, 32), 18)
                    copy_model.assert_not_called()
                    self.assertEqual(path.read_bytes(), b"existing graph")

    def test_dummy_inputs_keep_valid_rectangular_sizes(self):
        for size in (
            (16, 32),
            [32, 40],
            torch.Size((40, 48)),
            (np.int64(24), np.int32(56)),
        ):
            with self.subTest(size=size):
                values = self.export.dummy_matcher_inputs(size, n0=1, n1=2)
                frame = tuple(FlexGlue.stride * s for s in values[0].shape[-2:])
                self.assertEqual(frame, tuple(size))

    def test_cli_validates_before_loading_weights_or_creating_output(self):
        for matcher_only, size in (
            (False, (33, 41)),
            (False, (40, 48)),
            (True, (33, 41)),
            (True, (0, 32)),
        ):
            with self.subTest(matcher_only=matcher_only, size=size):
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "not_created"
                    argv = [
                        "export.py",
                        "--size",
                        *(str(s) for s in size),
                        "--weights",
                        "absent-local-checkpoint",
                        "--output_dir",
                        str(destination),
                    ]
                    if matcher_only:
                        argv.append("--matcher_only")
                    with patch.object(sys, "argv", argv), patch.object(
                        self.export, "FlexGlue", wraps=FlexGlue
                    ) as matcher, patch.object(
                        self.export, "DINOv3Descriptor", wraps=DINOv3Descriptor
                    ) as descriptor, contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        # Preserve class constants while observing construction.
                        matcher.stride = FlexGlue.stride
                        descriptor.edge_divisible_by = (
                            DINOv3Descriptor.edge_divisible_by
                        )
                        with self.assertRaises(SystemExit) as raised:
                            self.export.main()
                        self.assertEqual(raised.exception.code, 2)
                        matcher.assert_not_called()
                        descriptor.assert_not_called()
                    self.assertFalse(destination.exists())

    def test_cli_keeps_defaults_and_matcher_only_stride_sizes(self):
        for argv, expected in (
            ([], (512, 512)),
            (["--size", "32", "64"], (32, 64)),
            (["--matcher_only", "--size", "40", "48"], (40, 48)),
        ):
            with self.subTest(argv=argv), patch.object(
                sys, "argv", ["export.py", *argv]
            ):
                self.assertEqual(self.export.parse_args().size, expected)

    def test_public_rectangular_exports_keep_exact_grid_and_runtime_values(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        with torch.no_grad():
            for parameter in self.matcher.heads[1].offset_head.parameters():
                parameter.zero_()
            self.matcher.heads[1].matchability.weight.zero_()
            self.matcher.heads[1].matchability.bias.fill_(2.0)
        with tempfile.TemporaryDirectory() as temporary, torch.inference_mode():
            descriptor = small_descriptor().eval()
            path = Path(temporary) / "descriptor.onnx"
            with contextlib.redirect_stdout(io.StringIO()):
                self.export.export_descriptor(descriptor, path, (40, 48), 18)
            session = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
            self.assertEqual(session.get_inputs()[0].shape, [1, 3, 40, 48])
            image = torch.rand(1, 3, 40, 48)
            actual = session.run(None, {"image": image.numpy()})[0]
            self.assertEqual(actual.shape, (1, 256, 5, 6))
            np.testing.assert_allclose(
                actual, descriptor.encode(image).numpy(), rtol=1e-4, atol=1e-5
            )

            for size in ((16, 32), (32, 40)):
                path = Path(temporary) / "matcher.onnx"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.export.export_matcher(
                        self.matcher, path, size, 18, fused=False
                    )
                session = ort.InferenceSession(
                    str(path), options, providers=["CPUExecutionProvider"]
                )
                self.assertEqual(
                    session.get_inputs()[0].shape, [1, 256, size[0] // 8, size[1] // 8]
                )
                for n0, n1 in ((1, 1), (3, 5)):
                    values = list(self.export.dummy_matcher_inputs(size, n0, n1))
                    values[4] = values[5] = torch.tensor(
                        [[size[1], size[0]]], dtype=torch.float32
                    )
                    for index, count in ((2, n0), (3, n1)):
                        points = torch.arange(count, dtype=torch.float32) + 3
                        values[index] = torch.stack((points, points), -1)[None]
                    values[-3:] = [
                        torch.tensor(0.0),
                        torch.tensor(float("inf")),
                        torch.tensor(0.0),
                    ]
                    data = [
                        {
                            "features": values[k],
                            "keypoints": values[2 + k],
                            "image_size": values[4 + k],
                        }
                        for k in (0, 1)
                    ]
                    expected = self.matcher.match(
                        *data,
                        sparse=True,
                        exact_nn=True,
                        filter_threshold=0.0,
                        max_distance_px=float("inf"),
                    )
                    actual = session.run(
                        None,
                        {
                            key: value.numpy()
                            for key, value in zip(self.export.MATCHER_INPUTS, values)
                        },
                    )
                    self.assertGreater(int((actual[0] >= 0).sum()), 0)
                    np.testing.assert_array_equal(
                        actual[0], expected["matches0"].numpy()
                    )
                    for got, key in zip(
                        actual[1:], ("matching_scores0", "covisibility_score")
                    ):
                        np.testing.assert_allclose(
                            got, expected[key].numpy(), rtol=1e-4, atol=1e-5
                        )


if __name__ == "__main__":
    unittest.main()
