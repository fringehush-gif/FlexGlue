"""Portable RGB8 API checks; no checkpoint, download or Triton is needed.

Run with: python -m unittest discover -s tests -p test_rgb8.py
Optional Linux CUDA checks: set FLEXGLUE_TEST_RGB8_CUDA=1 explicitly.
"""
import builtins
from contextlib import contextmanager
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from flexglue import FlexGlue
from flexglue.descriptor import DINOv3Descriptor
from flexglue.utils.preprocess import numpy_image_to_torch
from test_descriptor_mp import TinyBackbone, TinyHead


class TinyStride8Backbone(TinyBackbone):
    def forward(self, x):
        return [torch.nn.functional.avg_pool2d(self.conv(x), 8)]


def tiny_descriptor(resize=(32, 32)):
    # Reuse the existing descriptor precision test's generated mock modules.
    torch.manual_seed(7)
    backbone, head, descriptor_head = TinyStride8Backbone(), TinyHead(), nn.Linear(8, 4)
    state = {
        **{"extractor.backbone." + k: v for k, v in backbone.state_dict().items()},
        **{"extractor.head." + k: v for k, v in head.state_dict().items()},
        **{"extractor.descriptor_head." + k: v for k, v in descriptor_head.state_dict().items()},
    }
    with patch("flexglue.descriptor.ConvNeXt", TinyStride8Backbone), patch(
        "flexglue.descriptor.LinearConvHead", TinyHead
    ):
        return DINOv3Descriptor(weights="unused", state_dict=state, resize=resize).eval()


def tiny_matcher():
    matcher = FlexGlue(
        weights=None, describe=False, input_dim=8, descriptor_dim=8,
        search_dim=4, num_heads=2, n_layers=1, mp=None,
    ).eval()
    matcher.descriptor = tiny_descriptor()
    return matcher


@contextmanager
def forbid_triton():
    original = builtins.__import__
    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton" or name.startswith("triton.") or "rgb8_triton" in (fromlist or ()):
            raise AssertionError("portable CPU path imported optional Triton")
        return original(name, globals, locals, fromlist, level)
    with patch("builtins.__import__", side_effect=blocked):
        yield


class PacketAssertions:
    def assertPacketEqual(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.stride(), right.stride())
            self.assertTrue(torch.equal(
                left.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
                right.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
            ))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertPacketEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertPacketEqual(a, b)
        else:
            self.assertEqual(left, right)


class RGB8PortableTest(PacketAssertions, unittest.TestCase):
    def setUp(self):
        for target in ('torch.load', 'torch.hub.load_state_dict_from_url'):
            guard = patch(target, side_effect=AssertionError('checkpoint access is forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        self.image = np.arange(13 * 17 * 3, dtype=np.uint32).astype(np.uint8).reshape(13, 17, 3)
        self.descriptor = tiny_descriptor()

    def test_cpu_layouts_match_original_forward_without_triton(self):
        class ImageSubclass(np.ndarray):
            pass
        layouts = [self.image.copy(), self.image[:, :, ::-1],
                   self.image[:, ::-1, :], self.image.view(ImageSubclass)]
        with forbid_triton(), torch.no_grad():
            for image in layouts:
                with self.subTest(strides=image.strides, type=type(image).__name__):
                    expected = self.descriptor({"image": numpy_image_to_torch(image)})
                    actual = self.descriptor.describe_rgb8(image)
                    self.assertPacketEqual(expected, actual)
                    self.assertTrue(torch.equal(actual["image_size"], torch.tensor([[17., 13.]])))

    def test_keypoints_keep_original_descriptor_values(self):
        keypoints = torch.tensor([[[4., 4.], [12., 9.]]])
        expected = self.descriptor({"image": numpy_image_to_torch(self.image),
                                    "keypoints": keypoints})
        expected["keypoints"] = keypoints
        with forbid_triton():
            self.assertPacketEqual(expected, self.descriptor.describe_rgb8(self.image, keypoints))

    def test_training_and_grad_fallback_construct_real_autograd_graph(self):
        self.descriptor.train()
        with forbid_triton(), torch.inference_mode(False), torch.enable_grad():
            result = self.descriptor.describe_rgb8(self.image)
            self.assertTrue(result["features"].requires_grad)
            self.assertIsNotNone(result["features"].grad_fn)
            self.assertFalse(torch.is_inference(result["features"]))
            result["features"].sum().backward()
        self.assertIsNotNone(self.descriptor.backbone.conv.weight.grad)
        self.assertTrue(torch.isfinite(self.descriptor.backbone.conv.weight.grad).all())

    def test_invalid_direct_inputs_are_rejected(self):
        for image in (self.image.astype(np.float32), self.image[:, :, 0], self.image[:, :, :2]):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaisesRegex(ValueError, "HWC RGB8"):
                    self.descriptor.describe_rgb8(image)

    def test_codebook_is_derived_nonpersistent_and_invalidates_on_transforms(self):
        for transform in (lambda d: d.to("cpu"), lambda d: d.to(dtype=torch.float64),
                          lambda d: d.to(dtype=torch.float32), lambda d: d.half()):
            self.descriptor._rgb8_codebook = torch.arange(256, dtype=torch.float32)
            self.assertFalse(any("_rgb8_codebook" in key for key in self.descriptor.state_dict()))
            self.assertIs(transform(self.descriptor), self.descriptor)
            self.assertIsNone(self.descriptor._rgb8_codebook)
        self.descriptor._rgb8_codebook = torch.arange(256, dtype=torch.float32)
        def fail(tensor):
            raise RuntimeError("test transform failure")
        with self.assertRaisesRegex(RuntimeError, "test transform failure"):
            self.descriptor._apply(fail)
        self.assertIsNone(self.descriptor._rgb8_codebook)

    def test_default_off_and_boolean_validation(self):
        self.assertIs(FlexGlue.default_conf["rgb8_triton"], False)
        matcher = tiny_matcher()
        self.assertIs(matcher.conf.rgb8_triton, False)
        with self.assertRaisesRegex(ValueError, "rgb8_triton"):
            matcher.resolve_conf({"rgb8_triton": "yes"})
        with self.assertRaisesRegex(ValueError, "rgb8_triton"):
            FlexGlue(weights=None, describe=False, rgb8_triton=1)

    def test_actual_cpu_match_optin_matches_torch_and_feature_packets(self):
        matcher = tiny_matcher()
        images = [self.image, self.image[:, :, ::-1]]
        tensors = [numpy_image_to_torch(image) for image in images]
        policy = dict(describe=True, describe_resize=(32, 32), max_num_samples=4,
                      min_covisibility=0., early_stop_decay=None)
        def run(data, **flags):
            torch.manual_seed(42)
            output = matcher.match(*data, **policy, **flags)
            return output, torch.get_rng_state()
        with forbid_triton(), torch.no_grad():
            reference, rng = run([{"image": tensor} for tensor in tensors])
            actual, actual_rng = run([{"image": image} for image in images], rgb8_triton=True)
            self.assertEqual(len(reference), 16)
            self.assertPacketEqual(reference, actual)
            self.assertTrue(torch.equal(rng, actual_rng))
            packets = [matcher.descriptor({"image": tensor}, resize=(32, 32)) for tensor in tensors]
            torch.manual_seed(42)
            expected = matcher.match(*packets, describe=False, max_num_samples=4,
                                     min_covisibility=0., early_stop_decay=None)
            torch.manual_seed(42)
            candidate = matcher.match(*packets, describe=False, rgb8_triton=True,
                                      max_num_samples=4, min_covisibility=0., early_stop_decay=None)
            self.assertPacketEqual(expected, candidate)

    def test_refiner_description_retains_torch_image(self):
        matcher = tiny_matcher()
        conf = matcher.resolve_conf({"describe": True, "describe_resize": (32, 32), "rgb8_triton": True})
        conf.refine = True  # routing only; no refiner weights or downloads
        with forbid_triton(), torch.no_grad():
            packets = matcher.describe_inputs({"image": self.image}, {"image": self.image}, conf)
        for packet in packets:
            self.assertIsInstance(packet["image"], torch.Tensor)
            self.assertPacketEqual(packet["image"], numpy_image_to_torch(self.image))


@unittest.skipUnless(os.environ.get("FLEXGLUE_TEST_RGB8_CUDA") == "1",
                     "set FLEXGLUE_TEST_RGB8_CUDA=1 for optional Linux CUDA checks")
class RGB8CudaTest(PacketAssertions, unittest.TestCase):
    def setUp(self):
        if not sys.platform.startswith("linux") or not torch.cuda.is_available():
            self.skipTest("Linux CUDA is required for the optional Triton path")
        if importlib.util.find_spec("triton") is None:
            self.skipTest("optional Triton is not installed")
        self.descriptor = tiny_descriptor((512, 512)).cuda()

    def test_accelerated_pixels_features_and_original_size(self):
        from flexglue.utils import rgb8_triton
        for height, width in ((512, 512), (513, 769)):
            rgb = np.arange(height * width * 3, dtype=np.uint32).astype(np.uint8).reshape(height, width, 3)[:, :, ::-1]
            with self.subTest(shape=rgb.shape), torch.inference_mode():
                image = numpy_image_to_torch(rgb).cuda()
                expected_pixels = self.descriptor.preprocess(image, (512, 512))
                actual = self.descriptor.describe_rgb8(rgb)
                prepared = rgb8_triton.prepare_rgb8(
                    rgb, codebook_device=self.descriptor._rgb8_codebook,
                )
                self.assertPacketEqual(expected_pixels, prepared.tensor)
                self.assertPacketEqual(self.descriptor({"image": image}), actual)
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
