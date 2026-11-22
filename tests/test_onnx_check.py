import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class ONNXCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("onnx", "onnxscript", "onnxruntime"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"optional dependency {name} is unavailable")
        onnx_dir = Path(__file__).resolve().parents[1] / "onnx"
        spec = importlib.util.spec_from_file_location(
            "flexglue_onnx_check", onnx_dir / "check.py"
        )
        cls.check = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(onnx_dir), *sys.path]):
            spec.loader.exec_module(cls.check)

    def setUp(self):
        n = 512
        self.pred = {
            "matches0": np.arange(n)[None],
            "matching_scores0": np.full((1, n), 0.5),
            "covisibility_score": np.array([0.7]),
            "p2d0_i1": np.zeros((1, n, 2)),
            "p2d1_i0": np.zeros((1, n, 2)),
            "confidence0": np.full((1, n), 0.5),
            "confidence1": np.full((1, n), 0.5),
        }
        self.tol = self.check.TOLERANCES["cpu"]

    def test_descriptor_failure_is_independent_of_image_order(self):
        for error in (np.nan, np.inf, 2 * self.tol["features"]):
            for index in (0, 1):
                with self.subTest(error=error, index=index):
                    errors = [0.0, 0.0]
                    errors[index] = error
                    with (
                        patch.object(self.check, "load_image"),
                        patch.object(self.check, "describe_pytorch"),
                        patch.object(
                            self.check, "descriptor_error", side_effect=errors
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        self.assertFalse(
                            self.check.check_pair(
                                SimpleNamespace(descriptor=object(), matcher=None),
                                SimpleNamespace(descriptor=None),
                                ("first.png", "second.png"),
                                (8, 8),
                                (),
                                self.tol,
                                False,
                            )
                        )

    def test_nonfinite_scores_fail_even_when_unmatched(self):
        for invalid in (np.nan, np.inf, -np.inf):
            for side in (0, 1):
                for matched in (False, True):
                    with self.subTest(invalid=invalid, side=side, matched=matched):
                        pred = {k: v.copy() for k, v in self.pred.items()}
                        outputs = [
                            pred[k].copy()
                            for k in (
                                "matches0",
                                "matching_scores0",
                                "covisibility_score",
                            )
                        ]
                        if not matched:
                            pred["matches0"][0, 0] = outputs[0][0, 0] = -1
                            pred["matching_scores0"][0, 0] = outputs[1][0, 0] = 0
                        (pred["matching_scores0"] if side == 0 else outputs[1])[
                            0, 0
                        ] = invalid
                        self.assertFalse(
                            self.check.compare_matches(pred, outputs, self.tol)[0]
                        )

    def test_nonfinite_compared_points_and_confidences_fail(self):
        for field in ("p2d0_i1", "confidence0"):
            for invalid in (np.nan, np.inf, -np.inf):
                for side in (0, 1):
                    with self.subTest(field=field, invalid=invalid, side=side):
                        pred = {k: v.copy() for k, v in self.pred.items()}
                        debug = {k: v.copy() for k, v in pred.items()}
                        target = pred if side == 0 else debug
                        index = (0, 0, 0) if field == "p2d0_i1" else (0, 0)
                        target[field][index] = invalid
                        self.assertFalse(
                            self.check.compare_points(pred, debug, self.tol)[0]
                        )

    def test_finite_outlier_and_confidence_mask_tolerances_are_preserved(self):
        outputs = [
            self.pred[k].copy()
            for k in ("matches0", "matching_scores0", "covisibility_score")
        ]
        outputs[1][0, 0] += 0.02
        self.assertTrue(self.check.compare_matches(self.pred, outputs, self.tol)[0])
        debug = {k: v.copy() for k, v in self.pred.items()}
        debug["p2d0_i1"][0, 0, 0] = 2
        debug["confidence0"][0, 0] += 0.02
        self.assertTrue(self.check.compare_points(self.pred, debug, self.tol)[0])
        self.pred["confidence0"].fill(0)
        self.pred["confidence1"].fill(0)
        debug = {k: v.copy() for k, v in self.pred.items()}
        debug["p2d0_i1"].fill(np.inf)
        self.assertTrue(self.check.compare_points(self.pred, debug, self.tol)[0])

    def test_truncated_debug_output_does_not_broadcast(self):
        pred = {k: v.copy() for k, v in self.pred.items()}
        # Different keypoint counts across directions are valid.
        for key in ("p2d1_i0", "confidence1"):
            pred[key] = pred[key][:, :3]
        self.assertTrue(self.check.compare_points(pred, pred, self.tol)[0])
        for key in ("p2d0_i1", "confidence0", "p2d1_i0", "confidence1"):
            with self.subTest(key=key):
                debug = {**pred, key: pred[key][:, :1]}
                accepted, message = self.check.compare_points(pred, debug, self.tol)
                self.assertFalse(accepted)
                self.assertIn(key, message)


if __name__ == "__main__":
    unittest.main()
