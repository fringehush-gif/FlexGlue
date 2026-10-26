import unittest

import numpy as np
import torch

from flexglue.utils.preprocess import numpy_image_to_torch


def legacy_image_to_torch(image):
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 2:
        image = image[None]
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


class ImageConversionTest(unittest.TestCase):
    def assert_legacy_equal(self, image):
        before = image.copy()
        expected = legacy_image_to_torch(image)
        actual = numpy_image_to_torch(image)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        np.testing.assert_array_equal(image, before)
        self.assertFalse(np.shares_memory(actual.numpy(), image))
        return actual

    def test_every_byte_value_gray_and_rgb(self):
        codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
        self.assert_legacy_equal(codes)
        rgb = np.stack((codes, np.flipud(codes), np.fliplr(codes)), axis=-1)
        self.assert_legacy_equal(rgb)
        self.assert_legacy_equal(rgb[..., ::-1])

    def test_strided_readonly_input_has_independent_output(self):
        source = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
        view = source[1::2, 2::3, ::-1]
        view.setflags(write=False)
        converted = self.assert_legacy_equal(view)
        saved = converted.clone()
        source.fill(0)
        self.assertTrue(torch.equal(converted, saved))
        converted.fill_(0.5)
        self.assertEqual(int(source.sum()), 0)

    def test_empty_and_single_pixel(self):
        self.assert_legacy_equal(np.empty((0, 0), dtype=np.uint8))
        self.assert_legacy_equal(np.array([[255]], dtype=np.uint8))

    def test_non_byte_inputs_keep_legacy_behavior(self):
        for dtype in (np.uint16, np.int16, np.float32, np.float64, np.bool_):
            with self.subTest(dtype=dtype):
                self.assert_legacy_equal(np.arange(36).reshape(3, 4, 3).astype(dtype))

    def test_default_device_context(self):
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rgb = np.stack((gray, np.flipud(gray), np.fliplr(gray)), axis=-1)[..., ::-1]
        for image in (gray, rgb):
            with self.subTest(shape=image.shape), torch.device("meta"):
                expected = legacy_image_to_torch(image)
                actual = numpy_image_to_torch(image)
                self.assertEqual(actual.device.type, "meta")
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.stride(), expected.stride())

    def test_invalid_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not an image"):
            numpy_image_to_torch(np.zeros((2, 2, 2, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
