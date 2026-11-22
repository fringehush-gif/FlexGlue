import copy
import unittest

import torch
from torch import nn

from flexglue.descriptor import ConvNeXtBlock


class ConvNeXtLifetimeTest(unittest.TestCase):
    def test_pwconv1_hook_rebinding_preserves_inflight_forward_and_gradients(self):
        ref = ConvNeXtBlock(4).double()
        block = copy.deepcopy(ref)
        generator = torch.Generator().manual_seed(41)
        image = torch.rand(1, 4, 5, 7, dtype=torch.float64, generator=generator)
        ref_image = image.clone().requires_grad_()
        image.requires_grad_()
        params = tuple(block.parameters())
        gamma = nn.Parameter(torch.zeros_like(block.gamma))
        act = nn.Identity()
        pwconv2 = nn.Linear(16, 4).double()

        def rebind(_module, _inputs, _output):
            block.gamma = gamma
            block.act = act
            block.pwconv2 = pwconv2

        expected = ref(ref_image)
        with block.pwconv1.register_forward_hook(rebind):
            actual = block(image)
        self.assertIs(block.gamma, gamma)
        self.assertIs(block.act, act)
        self.assertIs(block.pwconv2, pwconv2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        actual_grads = torch.autograd.grad(actual.square().sum(), (image, *params))
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (ref_image, *ref.parameters())
        )
        self.assertEqual(len(actual_grads), len(expected_grads))
        for observed, target in zip(actual_grads, expected_grads):
            torch.testing.assert_close(observed, target, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
