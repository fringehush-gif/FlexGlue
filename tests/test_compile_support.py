import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue


class CompileSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self):
        return FlexGlue(
            weights=None,
            describe=False,
            refine=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=1,
            mp=None,
            max_distance_px=None,
            filter_threshold=0.0,
        ).eval()

    def test_missing_capability_preserves_all_dispatch(self):
        for compiler in (None, SimpleNamespace()):
            for already_configured in (False, True):
                with self.subTest(compiler=compiler, configured=already_configured):
                    model = self.make_model()
                    model.descriptor = nn.Linear(4, 4)
                    model.descriptor.encode = model.descriptor.forward
                    model.refiner = nn.Linear(4, 4)
                    model.refiner.features = nn.Linear(4, 4)
                    if already_configured:
                        model.static_lengths = (3,)
                        model.heads[0].compiled_forward = model.heads[0].forward
                    forwards = [(module, module.forward) for module in model.modules()]
                    encode = model.descriptor.encode
                    lengths = model.static_lengths
                    missing = object()
                    heads = [
                        getattr(head, "compiled_forward", missing)
                        for head in model.heads
                    ]
                    with patch.object(torch, "compiler", compiler), patch.object(
                        torch, "compile"
                    ) as compile_mock:
                        with self.assertRaisesRegex(RuntimeError, "PyTorch >= 2.2"):
                            model.compile(static_lengths=(4, 8))
                    compile_mock.assert_not_called()
                    self.assertIs(model.static_lengths, lengths)
                    self.assertIs(model.descriptor.encode, encode)
                    for module, forward in forwards:
                        self.assertEqual(module.forward, forward)
                    for head, old in zip(model.heads, heads):
                        self.assertIs(getattr(head, "compiled_forward", missing), old)

    def test_supported_compile_preserves_public_matching(self):
        torch.manual_seed(0)
        model = self.make_model()
        data = tuple(
            {
                "features": torch.randn(1, 4, 2, 2),
                "image_size": torch.tensor([[16.0, 16.0]]),
                "keypoints": torch.tensor([[[3.0, 4.0], [11.0, 12.0]]]),
            }
            for _ in range(2)
        )
        with torch.no_grad():
            expected = model.match(*data)
        captures, executions = [], []

        def backend(graph, inputs, **kwargs):
            captures.append(graph)

            def run(*values):
                executions.append(True)
                return graph.forward(*values)

            return run

        real_compile = torch.compile

        def compile_cpu(function, **kwargs):
            return real_compile(function, backend=backend, **kwargs)

        with patch.object(torch, "compile", side_effect=compile_cpu), patch.object(
            torch.compiler,
            "cudagraph_mark_step_begin",
            wraps=torch.compiler.cudagraph_mark_step_begin,
        ) as mark_step, torch.no_grad():
            self.assertIs(model.compile(mode="default", static_lengths=(4,)), model)
            for _ in range(2):
                actual = model.match(*data)
                for key, value in expected.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(actual[key], value), key)
                    else:
                        self.assertEqual(actual[key], value, key)
        self.assertEqual(mark_step.call_count, 2)
        self.assertGreater(len(captures), 0)
        self.assertGreater(len(executions), 0)


if __name__ == "__main__":
    unittest.main()
