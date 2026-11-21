"""Public resize ownership and differentiation contracts."""

import unittest

import torch
import torch.nn.functional as F

from flexglue.utils.preprocess import resize_images


class ResizeImagesTests(unittest.TestCase):
    def test_owned_result_and_gradients(self):
        # Equal-size interpolation must still own its result, as must resized
        # CHW, grayscale, strided and batched input, including singleton lists.
        source = torch.linspace(-1, 1, 2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        for image in (
            source,
            source[:1],
            source[0],
            source[0, :1],
            source.transpose(-1, -2),
            source.contiguous(memory_format=torch.channels_last),
        ):
            for resize in (image.shape[-2:], (3, 4)):
                for as_list in (False, True):
                    with self.subTest(
                        shape=image.shape, resize=resize, as_list=as_list
                    ):
                        x = image.detach().clone().requires_grad_()
                        before = x.detach().clone()
                        batch = x[None] if x.ndim == 3 else x
                        expected = F.interpolate(
                            batch.expand(-1, 3, -1, -1),
                            resize,
                            mode="bilinear",
                            align_corners=False,
                        )
                        actual = resize_images([x] if as_list else x, resize)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        grad = torch.autograd.grad(
                            actual.square().sum(), x, create_graph=True
                        )[0]
                        reference_grad = torch.autograd.grad(
                            expected.square().sum(), x, create_graph=True
                        )[0]
                        torch.testing.assert_close(grad, reference_grad, rtol=0, atol=0)
                        second = torch.autograd.grad(grad.square().sum(), x)[0]
                        reference_second = torch.autograd.grad(
                            reference_grad.square().sum(), x
                        )[0]
                        torch.testing.assert_close(
                            second, reference_second, rtol=0, atol=0
                        )
                        with torch.no_grad():
                            actual.add_(1)
                        torch.testing.assert_close(x, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
