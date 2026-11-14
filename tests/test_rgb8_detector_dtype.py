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
                    .unsqueeze(1)
                }

        image = torch.rand(3, 8, 7, dtype=torch.float64)
        detector = Detector().eval()
        matcher = tiny_matcher()
        with torch.inference_mode(), patch.object(
            matcher, "match", side_effect=lambda *data, **kw: data
        ):
            matcher.match_pair(image, image, detector=detector)
        self.assertTrue(
            all(
                x.dtype == torch.float64 and x.data_ptr() == image.data_ptr()
                for x in detector.seen
            )
        )
