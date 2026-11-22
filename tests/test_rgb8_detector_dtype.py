"""Generated detector tests for byte input precision and ownership."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_rgb8 import tiny_matcher
from torch import nn
from torch.nn import functional as F


class DetectorDtypeTest(unittest.TestCase):
    def test_numpy_uses_detector_floating_state(self):
        class Detector(nn.Module):
            def __init__(self, dtype, mode):
                super().__init__()
                self.register_buffer("counter", torch.zeros((), dtype=torch.int64))
                self.mode = mode
                self.seen = []
                if mode == "parameter":
                    self.conv = nn.Conv2d(3, 2, 1, bias=False).to(dtype=dtype)
                elif mode == "buffer":
                    self.register_buffer("kernel", torch.ones(2, 3, 1, 1, dtype=dtype))

            def extract(self, image):
                self.seen.append((image.dtype, image.device))
                if self.mode == "parameter":
                    result = self.conv(image.unsqueeze(0))
                else:
                    kernel = (
                        self.kernel
                        if self.mode == "buffer"
                        else image.new_ones(2, 3, 1, 1)
                    )
                    result = F.conv2d(image.unsqueeze(0), kernel)
                return {"keypoints": result.mean(dim=(-2, -1)).unsqueeze(1)}

        image = np.arange(8 * 7 * 3, dtype=np.uint8).reshape(8, 7, 3)
        for dtype, mode in (
            (torch.float16, "parameter"),
            (torch.float64, "parameter"),
            (torch.float64, "buffer"),
            (torch.float32, "integer"),
            (torch.float32, "stateless"),
        ):
            with self.subTest(dtype=dtype, mode=mode):
                detector = Detector(dtype, mode).eval()
                if mode == "stateless":
                    del detector._buffers["counter"]
                matcher = tiny_matcher()
                with torch.inference_mode(), patch.object(
                    matcher, "match", side_effect=lambda *data, **kw: data
                ):
                    packet = matcher.match_pair(image, image, detector=detector)
                self.assertEqual([v[0] for v in detector.seen], [dtype, dtype])
                self.assertEqual([v[1].type for v in detector.seen], ["cpu", "cpu"])
                self.assertEqual(packet[0]["keypoints"].shape, (1, 1, 2))
                self.assertTrue(torch.isfinite(packet[0]["keypoints"]).all())

    def test_numpy_detector_geometry_runs_real_matcher(self):
        class Detector(nn.Module):
            def __init__(self, dtype, size_dtype):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, 1, bias=False).to(dtype=dtype)
                self.size_dtype = size_dtype
                self.seen = []
                self.outputs = []

            def extract(self, image):
                self.seen.append(image.dtype)
                response = self.conv(image.unsqueeze(0)).mean((-2, -1))
                scale = response.new_tensor([image.shape[-1] - 1, image.shape[-2] - 1])
                output = {
                    "keypoints": (response.sigmoid() * scale).unsqueeze(1),
                    "descriptors": response.unsqueeze(1),
                    "scores": response.square().sum(-1, keepdim=True),
                    "indices": torch.zeros((1, 1), dtype=torch.int64),
                }
                if self.size_dtype is not None:
                    output["image_size"] = response.new_tensor(
                        [[image.shape[-1], image.shape[-2]]], dtype=self.size_dtype
                    )
                self.outputs.append(output)
                return output

        image = np.arange(8 * 7 * 3, dtype=np.uint8).reshape(8, 7, 3)
        for dtype in (torch.float16, torch.float64):
            for size_dtype in (None, dtype, torch.int64):
                with self.subTest(dtype=dtype, size_dtype=size_dtype):
                    matcher = tiny_matcher()
                    detector = Detector(dtype, size_dtype).eval()
                    # Observe the handoff while retaining the real matching consumer.
                    with torch.inference_mode(), patch.object(
                        matcher, "match", wraps=matcher.match
                    ) as match:
                        pred = matcher.match_pair(
                            image,
                            image,
                            detector=detector,
                            describe=True,
                            describe_resize=(32, 32),
                            early_stop_decay=None,
                            max_distance_px=0,
                            min_covisibility=0.0,
                        )
                    match.assert_called_once()
                    self.assertEqual(detector.seen, [dtype, dtype])
                    for packet, original in zip(match.call_args.args, detector.outputs):
                        for key in ("keypoints", "image_size"):
                            if key in original:
                                expected = (
                                    original[key].float()
                                    if original[key].is_floating_point()
                                    else original[key]
                                )
                                self.assertEqual(packet[key].dtype, expected.dtype)
                                torch.testing.assert_close(
                                    packet[key], expected, rtol=0, atol=0
                                )
                        for key in ("descriptors", "scores", "indices"):
                            self.assertEqual(packet[key].dtype, original[key].dtype)
                            torch.testing.assert_close(
                                packet[key], original[key], rtol=0, atol=0
                            )
                    for i, j in ((0, 1), (1, 0)):
                        self.assertEqual(pred[f"p2d{i}_n{j}"].shape, (1, 1, 2))
                        self.assertTrue(torch.isfinite(pred[f"p2d{i}_n{j}"]).all())
                        self.assertTrue(torch.isfinite(pred[f"confidence{i}"]).all())

    def test_numpy_recovers_exact_size_from_half_detector(self):
        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("anchor", torch.zeros((), dtype=torch.float16))

            def extract(self, image):
                shape = image.shape[-2:][::-1]
                return {
                    "keypoints": image.new_tensor([[[1536, 4]]]),
                    "image_size": torch.tensor(shape)[None].to(image).float(),
                }

        image = np.zeros((8, 3073, 3), dtype=np.uint8)
        matcher = tiny_matcher()
        with torch.inference_mode(), patch.object(
            matcher, "match", side_effect=lambda *data, **kw: data
        ):
            packets = matcher.match_pair(image, image, detector=Detector())
        expected = torch.tensor([[3073, 8]], dtype=torch.float32)
        for packet in packets:
            torch.testing.assert_close(packet["image_size"], expected, rtol=0, atol=0)

    def test_tensor_input_is_not_retyped(self):
        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 2, 1, bias=False).double()
                self.seen = []

            def extract(self, image):
                self.seen.append(image)
                return {
                    "keypoints": self.conv(image.unsqueeze(0))
                    .mean((-2, -1))
                    .unsqueeze(1),
                    "image_size": image.new_tensor(
                        [[image.shape[-1], image.shape[-2]]]
                    ),
                }

        image = torch.rand(3, 8, 7, dtype=torch.float64)
        detector = Detector().eval()
        matcher = tiny_matcher()
        with torch.inference_mode(), patch.object(
            matcher, "match", side_effect=lambda *data, **kw: data
        ):
            packets = matcher.match_pair(image, image, detector=detector)
        for packet in packets:
            self.assertEqual(packet["keypoints"].dtype, torch.float64)
            self.assertEqual(packet["image_size"].dtype, torch.float64)
        self.assertTrue(
            all(
                x.dtype == torch.float64 and x.data_ptr() == image.data_ptr()
                for x in detector.seen
            )
        )
