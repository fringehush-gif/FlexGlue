import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from flexglue.utils import modules


class TinyMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformers = nn.ModuleList([modules.TransformerLayer(8, 2)])
        self.heads = nn.ModuleList()


class ExportIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        path = Path(__file__).resolve().parents[1] / "onnx" / "export.py"
        with patch.object(sys, "path", [str(path.parent), *sys.path]):
            cls.export = importlib.import_module("export")
        if Path(cls.export.__file__).resolve() != path:
            raise ImportError("export module came from another checkout")

    def test_timing_rejects_zero_runs_before_session_execution(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
        spec = importlib.util.spec_from_file_location("flexglue_check_test", path)
        check = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"export": self.export}):
            with patch.object(
                sys, "path", [str(Path(check.__file__).parent), *sys.path]
            ):
                spec.loader.exec_module(check)
        session = unittest.mock.Mock()
        for runs, warmup in ((0, 2), (-1, 2), (1, -1)):
            with self.subTest(runs=runs, warmup=warmup), self.assertRaises(ValueError):
                check.time_session(session, {}, runs=runs, warmup=warmup)
        session.run.assert_not_called()

    def test_descriptor_frames_in_runtime(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        import onnxruntime as ort

        from flexglue import FlexGlue

        export = self.export
        for size, fused in (((16, 32), False), ((32, 16), True)):
            with self.subTest(
                size=size, fused=fused
            ), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(17)
                model = FlexGlue(
                    weights=None,
                    describe=False,
                    n_layers=1,
                    descriptor_dim=32,
                    search_dim=8,
                    num_heads=4,
                    mp=None,
                    query_chunk_size=None,
                ).eval()
                with torch.no_grad():
                    for parameter in model.heads[1].offset_head.parameters():
                        parameter.zero_()
                    model.heads[1].matchability.weight.zero_()
                    model.heads[1].matchability.bias.fill_(2.0)
                path = Path(tmp) / "matcher.onnx"
                with torch.no_grad():
                    export.export_matcher(model, path, size, 18, fused=fused)
                    session = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                    for n0, n1, resolution in ((3, 5, 1536), (11, 2, 3072)):
                        inputs = list(export.dummy_matcher_inputs(size, n0, n1))
                        for k, dimensions in (
                            (0, (resolution, resolution // 2)),
                            (1, (resolution // 2, resolution)),
                        ):
                            scale = inputs[4 + k].new_tensor([dimensions])
                            inputs[2 + k] = (
                                inputs[2 + k] / inputs[4 + k][:, None] * scale[:, None]
                            )
                            inputs[4 + k] = scale
                        data = [
                            {
                                "features": inputs[k],
                                "keypoints": inputs[2 + k],
                                "image_size": inputs[4 + k],
                            }
                            for k in (0, 1)
                        ]
                        for limit in (float("inf"), 0.0):
                            inputs[6], inputs[7] = torch.tensor(0.0), torch.tensor(
                                limit
                            )
                            pred = model.match(
                                *data,
                                sparse=True,
                                exact_nn=True,
                                filter_threshold=inputs[6],
                                max_distance_px=inputs[7],
                            )
                            outputs = session.run(
                                None,
                                {
                                    key: value.numpy()
                                    for key, value in zip(export.MATCHER_INPUTS, inputs)
                                },
                            )
                            self.assertTrue(
                                torch.equal(
                                    torch.from_numpy(outputs[0]), pred["matches0"]
                                )
                            )
                            accepted = int((pred["matches0"] >= 0).sum())
                            if limit == 0.0:
                                self.assertEqual(accepted, 0)
                            else:
                                self.assertGreater(accepted, 0)
                            for result, expected in zip(
                                outputs[1:],
                                (pred["matching_scores0"], pred["covisibility_score"]),
                            ):
                                torch.testing.assert_close(
                                    torch.from_numpy(result),
                                    expected,
                                    rtol=1e-4,
                                    atol=1e-5,
                                )

    def test_fused_context_is_local_and_native_backward_works(self):
        export_model, native = TinyMatcher(), TinyMatcher()
        exported = export_model.transformers[0]
        native_cross = native.transformers[0].cross_attn
        original = modules.attention
        calls = []

        def fake_fused(q, k, v):
            calls.append(1)
            return original(q, k, v)

        with patch.object(self.export, "fused_attention", fake_fused):
            with self.export.exportable(export_model):
                self.assertIs(modules.attention, original)
                self.assertIs(exported.self_attn.attention, fake_fused)
                self.assertIs(exported.cross_attn.attention, fake_fused)
                x = torch.randn(1, 3, 8, requires_grad=True)
                y = torch.randn(1, 4, 8, requires_grad=True)
                outputs = native_cross(x, y)
                sum(t.sum() for t in outputs).backward()
                self.assertIsNotNone(x.grad)
                self.assertEqual(calls, [])
                with torch.no_grad():
                    exported.cross_attn(x.detach(), y.detach())
                self.assertEqual(len(calls), 2)
        self.assertIsNone(exported.self_attn.attention)
        self.assertIsNone(exported.cross_attn.attention)
        self.assertIs(modules.attention, original)

    def test_nested_overlapping_and_exception_restore(self):
        first, second = TinyMatcher(), TinyMatcher()
        a, b = first.transformers[0].cross_attn, second.transformers[0].cross_attn
        native = modules.attention
        with self.export.exportable(first):
            self.assertIs(a.attention, self.export.fused_attention)
            with self.export.exportable(first):
                self.assertIs(a.attention, self.export.fused_attention)
            self.assertIs(a.attention, self.export.fused_attention)
        self.assertIsNone(a.attention)

        outer = self.export.exportable(first)
        inner = self.export.exportable(second)
        outer.__enter__()
        inner.__enter__()
        outer.__exit__(None, None, None)
        self.assertIsNone(a.attention)
        self.assertIs(b.attention, self.export.fused_attention)
        inner.__exit__(None, None, None)
        self.assertIsNone(b.attention)
        self.assertIs(modules.attention, native)

        with self.assertRaisesRegex(RuntimeError, "export failed"):
            with self.export.exportable(first):
                raise RuntimeError("export failed")
        self.assertIsNone(a.attention)
        self.assertIs(modules.attention, native)


if __name__ == "__main__":
    unittest.main()
