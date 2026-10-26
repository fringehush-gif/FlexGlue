import unittest
from unittest.mock import patch

import torch
from torch import nn

from flexglue.descriptor import DINOv3Descriptor


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return [self.conv(x)]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 1)

    def forward(self, features):
        return self.conv(features[0])


class DescriptorPrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        backbone = TinyBackbone()
        head = TinyHead()
        descriptor_head = nn.Linear(8, 4)
        self.state = {
            **{"extractor.backbone." + k: v for k, v in backbone.state_dict().items()},
            **{"extractor.head." + k: v for k, v in head.state_dict().items()},
            **{
                "extractor.descriptor_head." + k: v
                for k, v in descriptor_head.state_dict().items()
            },
        }

    def build(self, mp=None):
        with patch("flexglue.descriptor.ConvNeXt", TinyBackbone), patch(
            "flexglue.descriptor.LinearConvHead", TinyHead
        ):
            return DINOv3Descriptor(
                weights="unused", state_dict=self.state, resize=(32, 32), mp=mp
            ).eval()

    def test_default_remains_float32(self):
        descriptor = self.build()
        image = torch.rand(1, 3, 32, 32)
        keypoints = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]])
        pred = descriptor.describe(image, keypoints=keypoints)
        self.assertEqual(descriptor.backbone.conv.weight.dtype, torch.float32)
        self.assertEqual(descriptor.head.conv.weight.dtype, torch.float32)
        self.assertEqual(pred["features"].dtype, torch.float32)
        self.assertEqual(pred["descriptors"].dtype, torch.float32)

    def test_resident_half_keeps_normalization_and_sparse_output_float32(self):
        descriptor = self.build("float16")
        image = torch.rand(1, 3, 32, 32)
        keypoints = torch.tensor([[[8.0, 8.0], [24.0, 24.0]]])
        pred = descriptor.describe(image, keypoints=keypoints)
        self.assertEqual(descriptor.backbone.conv.weight.dtype, torch.float16)
        self.assertEqual(descriptor.head.conv.weight.dtype, torch.float16)
        self.assertEqual(descriptor.mean.dtype, torch.float32)
        self.assertEqual(descriptor.std.dtype, torch.float32)
        self.assertEqual(descriptor.descriptor_head.weight.dtype, torch.float32)
        self.assertEqual(pred["features"].dtype, torch.float32)
        self.assertEqual(pred["descriptors"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(pred["descriptors"]).all())

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "descriptor mp"):
            self.build("bfloat16")


if __name__ == "__main__":
    unittest.main()
