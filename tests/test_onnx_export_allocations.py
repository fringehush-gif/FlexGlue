"""Focused publication and graph ownership contracts for the ONNX exporter."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from onnx_test_utils import load_export


def fixture_model(payload_elements=1):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    inits = [
        numpy_helper.from_array(np.full(payload_elements, value, np.float32), name)
        for name, value in (("weight2", 0.2), ("weight4", 0.3), ("weight_final", 0.4))
    ]
    nodes = [
        helper.make_node("ReduceMean", ["weight2"], ["covisibility_score_layer2"]),
        helper.make_node("ReduceMean", ["weight4"], ["covisibility_score_layer4"]),
        helper.make_node("Shape", ["kpts0"], ["shape"], start=0, end=2),
        helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["matches0"],
            value=helper.make_tensor("v", TensorProto.INT64, [1], [-1]),
        ),
        helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["mscores0"],
            value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0]),
        ),
        helper.make_node("ReduceMean", ["weight_final"], ["covisibility_score"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [
            helper.make_tensor_value_info("kpts0", TensorProto.FLOAT, [1, 3, 2]),
            helper.make_tensor_value_info("min_covisibility", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("matches0", TensorProto.INT64, [1, 3]),
            helper.make_tensor_value_info("mscores0", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("covisibility_score", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info(
                "covisibility_score_layer2", TensorProto.FLOAT, [1]
            ),
            helper.make_tensor_value_info(
                "covisibility_score_layer4", TensorProto.FLOAT, [1]
            ),
        ],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


class ExportAllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        cls.export = load_export()

    def matcher(self):
        from flexglue import FlexGlue

        return FlexGlue(
            weights=None,
            describe=False,
            n_layers=5,
            descriptor_dim=32,
            search_dim=8,
            num_heads=4,
            mp=None,
        ).eval()

    def test_one_round_trip_without_early_stopping(self):
        import onnx

        export = self.export
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matcher.onnx"
            path.write_bytes(fixture_model().SerializeToString())
            with patch.object(onnx, "load", wraps=onnx.load) as load, patch.object(
                onnx, "save", wraps=onnx.save
            ) as save, patch.object(
                onnx.checker, "check_model", wraps=onnx.checker.check_model
            ) as check:
                export._postprocess_matcher_graph(path, 5, 8, ())
                self.assertEqual(
                    (load.call_count, check.call_count, save.call_count), (1, 1, 1)
                )
            model = onnx.load(path)
            self.assertNotIn("early_stop_decay", [x.name for x in model.graph.input])
            self.assertEqual(
                {p.key: p.value for p in model.metadata_props}["flexglue.n_layers"], "5"
            )

    def test_one_round_trip_publishes_transformed_graph(self):
        import onnx

        export = self.export
        raw = fixture_model().SerializeToString()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matcher.onnx"
            path.write_bytes(b"original")

            def write(*args, **kwargs):
                self.assertEqual(path.read_bytes(), b"original")
                Path(args[2]).write_bytes(raw)

            with patch.object(
                export, "dummy_matcher_inputs", return_value=()
            ), patch.object(torch.onnx, "export", side_effect=write), patch.object(
                onnx, "load", wraps=onnx.load
            ) as load, patch.object(
                onnx, "save", wraps=onnx.save
            ) as save, patch.object(
                onnx.checker, "check_model", wraps=onnx.checker.check_model
            ) as check:
                export.export_matcher(
                    self.matcher(), path, (32, 32), 18, early_stop_layers=(2, 4)
                )
                self.assertEqual(
                    (load.call_count, check.call_count, save.call_count), (1, 1, 1)
                )
            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.assertEqual(
                {p.key: p.value for p in model.metadata_props}["flexglue.n_layers"], "5"
            )
            self.assertIn("early_stop_decay", [x.name for x in model.graph.input])
            self.assertEqual(
                [x.name for x in model.graph.output], export.MATCHER_OUTPUTS
            )
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_real_checker_failure_preserves_destination(self):
        import onnx

        export = self.export
        invalid = fixture_model()
        invalid.graph.node[0].op_type = "DefinitelyNotAnOnnxOperator"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matcher.onnx"
            path.write_bytes(b"original")

            def write(*args, **kwargs):
                Path(args[2]).write_bytes(invalid.SerializeToString())

            with patch.object(
                export, "dummy_matcher_inputs", return_value=()
            ), patch.object(torch.onnx, "export", side_effect=write):
                with self.assertRaises(onnx.checker.ValidationError):
                    export.export_matcher(
                        self.matcher(), path, (32, 32), 18, early_stop_layers=(2, 4)
                    )
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_completed_continuations_are_not_passed_to_if_constructors(self):
        from onnx import helper

        export = self.export
        model = fixture_model(payload_elements=1_048_576)
        constructor_payloads = []
        original = helper.make_node

        def payload_bytes(graph):
            total = sum(len(t.raw_data) for t in graph.initializer)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == attr.GRAPH:
                        total += payload_bytes(attr.g)
            return total

        def make_node(op_type, *args, **kwargs):
            if op_type == "If":
                constructor_payloads.append(payload_bytes(kwargs["else_branch"]))
            return original(op_type, *args, **kwargs)

        with patch.object(helper, "make_node", side_effect=make_node):
            export._add_early_stopping_model(model, (2, 4), 5)
        self.assertEqual(constructor_payloads, [0, 0])
        graph = model.graph
        for expected in ("weight2", "weight4", "weight_final"):
            self.assertEqual([t.name for t in graph.initializer], [expected])
            conditionals = [node for node in graph.node if node.op_type == "If"]
            if conditionals:
                graph = next(
                    a.g for a in conditionals[0].attribute if a.name == "else_branch"
                )
        self.assertEqual(len(graph.initializer[0].raw_data), 4_194_304)

    def test_path_wrappers_remain_usable(self):
        import onnx

        export = self.export
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matcher.onnx"
            path.write_bytes(fixture_model().SerializeToString())
            export.finalize_matcher_graph(path, 5, 8)
            export.add_early_stopping(path, (2, 4), 5)
            model = onnx.load(path)
            onnx.checker.check_model(model)
            self.assertEqual(
                {p.key: p.value for p in model.metadata_props}["flexglue.stride"], "8"
            )
            self.assertIn("early_stop_decay", [x.name for x in model.graph.input])
