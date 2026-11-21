import unittest
from types import SimpleNamespace

import torch

from flexglue.descriptor import DINOv3Descriptor
from flexglue.utils.preprocess import image_sizes


class ImageSizesTest(unittest.TestCase):
    def test_dimensions_do_not_inherit_half_pixel_rounding(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                image = torch.ones(1, 1, 1, 1, dtype=dtype).expand(2, 3, 33, 4111)
                expected_dtype = (
                    torch.int64 if dtype in (torch.float16, torch.bfloat16) else dtype
                )
                expected = torch.tensor([[4111, 33]] * 2, dtype=expected_dtype)
                sizes = image_sizes(image)
                torch.testing.assert_close(sizes, expected, rtol=0, atol=0)
                sizes[0, 0] = 0
                torch.testing.assert_close(image_sizes(image), expected, rtol=0, atol=0)
                self.assertTrue((image == 1).all())

                descriptor = DINOv3Descriptor.__new__(DINOv3Descriptor)
                torch.nn.Module.__init__(descriptor)
                descriptor.conf = SimpleNamespace(
                    resize=(32, 32), describe=False, mp=None
                )
                descriptor.backbone = torch.nn.Identity()
                descriptor.head = torch.nn.Identity()
                descriptor.register_buffer("mean", torch.zeros(3, 1, 1))
                descriptor.register_buffer("std", torch.ones(3, 1, 1))
                result = descriptor({"image": image})
                torch.testing.assert_close(
                    result["image_size"], expected, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
