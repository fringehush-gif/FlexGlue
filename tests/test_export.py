import importlib.util
import math
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
        spec = importlib.util.spec_from_file_location("flexglue_export_test", path)
        cls.export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.export)

    def test_early_stop_export_checks_only_required_heads_before_copy(self):
        from flexglue import FlexGlue

        export = self.export
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
            early_stop_decay=0.5,
            min_covisibility=0.5,
        ).eval()
        original_conf = vars(model.conf).copy()
        model.loaded_heads = {2, 4, 5}  # unchecked intermediate heads can be absent
        original_heads = model.loaded_heads.copy()
        path = Path("unused.onnx")

        with patch.object(
            export, "dummy_matcher_inputs", return_value=()
        ), patch.object(torch.onnx, "export") as write, patch.object(
            export, "make_keypoint_dims_dynamic"
        ), patch.object(
            export, "add_early_stopping"
        ):
            export.export_matcher(model, path, (32, 32), 18, early_stop_layers=(2, 4))
            self.assertIsInstance(
                write.call_args.args[0], export.LayerScoresMatcherONNX
            )
            self.assertIsNot(write.call_args.args[0].matcher, model)
            self.assertEqual(write.call_args.args[0].check_layers, (2, 4))
            self.assertIsNone(write.call_args.args[0].conf.early_stop_decay)
        self.assertEqual(vars(model.conf), original_conf)
        self.assertEqual(model.loaded_heads, original_heads)

        for heads, schedule, debug in (
            ({4, 5}, (2, 4), False),  # missing a checked head
            ({2, 4}, (2, 4), False),  # missing the final head
            ({2, 4, 5}, (2, 2), False),
            ({2, 4, 5}, (True,), False),
            ({2, 4, 5}, (5,), False),
            ({2, 4, 5}, (0,), False),
            ({2, 4, 5}, (2, 4), True),
        ):
            with self.subTest(heads=heads, schedule=schedule, debug=debug):
                model.loaded_heads = heads
                with patch.object(
                    export.copy, "deepcopy", wraps=export.copy.deepcopy
                ) as copy_model:
                    with self.assertRaises(ValueError):
                        export.export_matcher(
                            model,
                            path,
                            (32, 32),
                            18,
                            early_stop_layers=schedule,
                            debug_outputs=debug,
                        )
                    copy_model.assert_not_called()
                self.assertEqual(vars(model.conf), original_conf)
                self.assertEqual(model.loaded_heads, heads)

    def test_layer_scores_wrapper_requires_selected_heads(self):
        from flexglue import FlexGlue

        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()
        model.loaded_heads = {2, 4, 5}
        wrapper = self.export.LayerScoresMatcherONNX(model, (2, 4))
        self.assertEqual(wrapper.check_layers, (2, 4))
        model.loaded_heads = {4, 5}
        with self.assertRaises(ValueError):
            self.export.LayerScoresMatcherONNX(model, (2, 4))

    def test_early_stop_checker_rejects_wrong_score(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
        spec = importlib.util.spec_from_file_location("flexglue_check_test", path)
        check = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"export": self.export}):
            spec.loader.exec_module(check)

        class Matcher:
            def match(self, data0, data1, **kwargs):
                self.kwargs = kwargs
                return {
                    "matches0": torch.tensor([[-1, -1]]),
                    "matching_scores0": torch.zeros(1, 2),
                    "covisibility_score": torch.tensor([0.12]),
                }

        class Session:
            def __init__(self, score):
                self.score = score

            def run(self, names, feeds):
                self.decay = float(feeds["early_stop_decay"])
                return [
                    torch.tensor([[-1, -1]]).numpy(),
                    torch.zeros(1, 2).numpy(),
                    torch.tensor([self.score]).numpy(),
                ]

        features = [
            {
                "features": torch.zeros(1, 256, 4, 4),
                "image_size": torch.tensor([[32.0, 32.0]]),
            }
            for _ in range(2)
        ]
        keypoints = [torch.ones(1, 2, 2), torch.ones(1, 3, 2)]
        feeds = check.matcher_feeds(*features, *keypoints, 0.5)
        matcher = Matcher()
        for score, expected in ((0.12, True), (0.8, False)):
            with self.subTest(score=score):
                session = Session(score)
                result = check.check_early_stop(
                    session,
                    matcher,
                    features,
                    keypoints,
                    feeds,
                    0.3,
                    check.TOLERANCES["cpu"],
                )
                self.assertIs(result, expected)
                self.assertEqual(matcher.kwargs["early_stop_decay"], 0.3)
                self.assertAlmostEqual(session.decay, 0.3)

    def test_scheduled_early_stops_match_native_runtime(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("optional dependency onnxruntime is unavailable")
        import onnx
        import onnxruntime as ort

        from flexglue import FlexGlue

        export = self.export
        schedule = (2, 4, 6, 8)
        scores = {2: 0.12, 4: 0.13, 6: 0.14, 8: 0.15, 9: 0.9}
        torch.manual_seed(31)
        model = FlexGlue(
            weights=None,
            describe=False,
            n_layers=9,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
            query_chunk_size=None,
        ).eval()
        with torch.no_grad():
            for head in model.heads:
                for parameter in head.offset_head.parameters():
                    parameter.zero_()
                if head.matchability is not None:
                    head.matchability.weight.zero_()
                    head.matchability.bias.fill_(2.0)
            for layer, score in scores.items():
                model.heads[layer].covisibility.weight.zero_()
                model.heads[layer].covisibility.bias.fill_(
                    math.log(score / (1 - score))
                )

        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            path = Path(tmp) / "scheduled.onnx"
            export.export_matcher(
                model, path, (32, 32), 18, fused=False, early_stop_layers=schedule
            )
            check_path = Path(__file__).resolve().parents[1] / "onnx" / "check.py"
            spec = importlib.util.spec_from_file_location(
                "flexglue_check_test", check_path
            )
            check = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"export": export}):
                spec.loader.exec_module(check)
            graph_model = onnx.load(str(path))
            graphs = list(check.subgraphs(graph_model.graph))
            if_nodes = [
                node for graph in graphs for node in graph.node if node.op_type == "If"
            ]
            self.assertEqual(len(if_nodes), len(schedule))
            for index, node in enumerate(if_nodes):
                node.name = f"renamed_if_{index}"
            onnx.checker.check_model(graph_model)
            renamed_path = Path(tmp) / "renamed.onnx"
            onnx.save(graph_model, str(renamed_path))
            self.assertEqual(
                check.matcher_options(renamed_path).early_stop_layers, list(schedule)
            )

            # Rename the score consistently throughout all graph scopes, so the
            # model remains valid but this stop predicate cannot be identified.
            old, new = "covisibility_score_layer4", "anonymous_stop_score"
            replacements = 0
            for graph in graphs:
                for node in graph.node:
                    for names in (node.input, node.output):
                        for index, name in enumerate(names):
                            if name == old:
                                names[index] = new
                                replacements += 1
                for value in (
                    list(graph.input) + list(graph.output) + list(graph.value_info)
                ):
                    if value.name == old:
                        value.name = new
                        replacements += 1
            self.assertGreater(replacements, 0)
            onnx.checker.check_model(graph_model)
            unknown_path = Path(tmp) / "unknown.onnx"
            onnx.save(graph_model, str(unknown_path))
            with self.assertRaisesRegex(ValueError, "Cannot identify"):
                check.matcher_options(unknown_path)
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            for n0, n1 in ((1, 1), (3, 5)):
                inputs = list(export.dummy_matcher_inputs((32, 32), n0, n1))
                inputs[6], inputs[7], inputs[8] = (
                    torch.tensor(0.0),
                    torch.tensor(float("inf")),
                    torch.tensor(0.5),
                )
                data = [
                    {
                        "features": inputs[k],
                        "keypoints": inputs[2 + k],
                        "image_size": inputs[4 + k],
                    }
                    for k in (0, 1)
                ]
                for decay, layer in ((0.1, 2), (0.2, 4), (0.3, 6), (0.5, 8), (1.0, 9)):
                    with self.subTest(decay=decay, layer=layer):
                        pred = model.match(
                            *data,
                            sparse=True,
                            exact_nn=True,
                            filter_threshold=0.0,
                            max_distance_px=float("inf"),
                            min_covisibility=0.5,
                            early_stop_layers=schedule,
                            early_stop_decay=None if decay == 1 else decay,
                        )
                        feed = {
                            key: value.numpy()
                            for key, value in zip(export.MATCHER_INPUTS, inputs)
                        }
                        feed["early_stop_decay"] = torch.tensor(decay).numpy()
                        matches, match_scores, covisibility = session.run(None, feed)
                        self.assertEqual(pred["stop_layer"], layer)
                        self.assertTrue(
                            torch.equal(torch.from_numpy(matches), pred["matches0"])
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(match_scores),
                            pred["matching_scores0"],
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(covisibility),
                            pred["covisibility_score"],
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        torch.testing.assert_close(
                            torch.from_numpy(covisibility),
                            torch.tensor([scores[layer]]),
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        if layer == 9:
                            self.assertGreater(int((pred["matches0"] >= 0).sum()), 0)
                        else:
                            self.assertTrue(torch.all(torch.from_numpy(matches) == -1))
                            self.assertTrue(
                                torch.all(torch.from_numpy(match_scores) == 0)
                            )

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
