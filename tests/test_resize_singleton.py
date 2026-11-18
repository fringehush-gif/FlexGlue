"""Singleton resize ownership, layout and gradient regression."""
import unittest

import torch

from flexglue.utils import preprocess as pp


def original(images, resize):
    resized = [pp.resize_tensor(im, resize) for im in pp.batch_images(images)]
    if len({tuple(im.shape[-2:]) for im in resized}) > 1:
        raise ValueError("Images of different sizes need a (h, w) resize to batch.")
    return torch.cat(resized)


class ResizeSingletonTest(unittest.TestCase):
    def test_singleton_values_layout_and_ownership(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for batch in (0, 1, 2):
                for channels in (1, 3):
                    for source, target in (
                        ((1, 1), (5, 7)),
                        ((3, 5), (3, 5)),
                        ((5, 7), (2, 4)),
                    ):
                        for channels_last in (False, True):
                            with self.subTest(
                                dtype=dtype,
                                batch=batch,
                                channels=channels,
                                source=source,
                                target=target,
                                channels_last=channels_last,
                            ):
                                image = torch.randn(
                                    batch, channels, *source, dtype=dtype
                                )
                                if channels_last:
                                    image = image.contiguous(
                                        memory_format=torch.channels_last
                                    )
                                before = image.clone()
                                actual, expected = pp.resize_images(
                                    image, target
                                ), original(image, target)
                                self.assertEqual(actual.shape, expected.shape)
                                self.assertEqual(actual.dtype, expected.dtype)
                                self.assertEqual(actual.stride(), expected.stride())
                                torch.testing.assert_close(
                                    actual, expected, rtol=0, atol=0, equal_nan=True
                                )
                                self.assertTrue(torch.equal(image, before))
                                if actual.numel():
                                    self.assertNotEqual(
                                        actual.untyped_storage().data_ptr(),
                                        image.untyped_storage().data_ptr(),
                                    )

    def test_nonfinite_and_signed_zero(self):
        image = torch.tensor([[[[0.0, -0.0], [float("inf"), float("nan")]]]])
        actual, expected = pp.resize_images(image, (3, 3)), original(image, (3, 3))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(
            torch.equal(
                torch.signbit(actual[actual == 0]),
                torch.signbit(expected[expected == 0]),
            )
        )

    def test_multiple_and_empty_lists(self):
        images = [torch.randn(3, 3, 4), torch.randn(3, 3, 4)]
        torch.testing.assert_close(
            pp.resize_images(images, (5, 6)), original(images, (5, 6)), rtol=0, atol=0
        )
        with self.assertRaises(Exception) as prior:
            original([], (5, 6))
        with self.assertRaises(type(prior.exception)) as actual:
            pp.resize_images([], (5, 6))
        self.assertEqual(str(actual.exception), str(prior.exception))

    def test_gradients_and_vmap(self):
        image = torch.randn(2, 3, 3, 5, dtype=torch.float64, requires_grad=True)
        actual, expected = pp.resize_images(image, (4, 6)), original(image, (4, 6))
        ga = torch.autograd.grad(actual.square().sum(), image, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.square().sum(), image)[0]
        torch.testing.assert_close(ga, ge, rtol=0, atol=0)
        batch = torch.randn(2, 3, 3, 5)
        mapped = torch.vmap(lambda x: pp.resize_images(x, (4, 6)))(batch)
        ref = torch.vmap(lambda x: original(x, (4, 6)))(batch)
        torch.testing.assert_close(mapped, ref, rtol=0, atol=0)
