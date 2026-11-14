"""CPU layout check for the benchmark's full-resolution resize path."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flexglue.utils import rgb8_torch
from flexglue.utils._rgb8 import fp32_codebook


class BenchmarkLayoutTest(unittest.TestCase):
    def test_full_resize_one_pixel_nchw_and_bits(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_rgb8.py"
        spec = importlib.util.spec_from_file_location("_rgb8_benchmark_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = torch.from_numpy(np.array([[[3, 129, 252]]], dtype=np.uint8))
        table = torch.from_numpy(fp32_codebook())
        actual = module.full_resize(
            source, table, (512, 512), False, converter=rgb8_torch._convert_rgb8
        )
        image = table[source.long()].permute(2, 0, 1).unsqueeze(0).contiguous()
        expected = F.interpolate(
            image, size=(512, 512), mode="bilinear", align_corners=False
        ).contiguous()
        self.assertEqual(tuple(actual.shape), (1, 3, 512, 512))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )
