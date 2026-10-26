import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from flexglue import FlexGlue
from flexglue.utils import ops


class WeightLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "weights.pth")
        self.conf = dict(
            describe=False,
            input_dim=4,
            descriptor_dim=8,
            search_dim=4,
            num_heads=2,
            n_layers=2,
        )
        torch.manual_seed(0)
        self.original = FlexGlue(weights=None, **self.conf)
        self.state = {
            "matcher." + k: v for k, v in self.original.state_dict().items()
        }

    def build(self, state=None, **conf):
        torch.save(self.state if state is None else state, self.path)
        return FlexGlue(weights=self.path, **{**self.conf, **conf})

    def test_local_prefix_and_shared_mapping(self):
        raw = {"matcher.x": torch.tensor([1.0]), "extractor.y": torch.tensor([2.0])}
        torch.save(raw, self.path)
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(
                torch.equal(
                    ops.load_state_dict(self.path, "matcher.")["x"], raw["matcher.x"]
                )
            )
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        with patch.object(torch, "load", side_effect=AssertionError("unexpected read")):
            self.assertIs(
                ops.load_state_dict(self.path, "extractor.", state_dict=raw)["y"],
                raw["extractor.y"],
            )
        self.assertEqual(set(raw), {"matcher.x", "extractor.y"})

    def test_missing_prefix_in_preloaded_mapping(self):
        with self.assertRaisesRegex(KeyError, "No weights with prefix"):
            ops.load_state_dict(self.path, "matcher.", state_dict={})

    def test_url_options(self):
        url = "https://example.invalid/weights.pth"
        with patch.object(
            torch.hub, "load_state_dict_from_url", return_value={"matcher.x": 1}
        ) as load:
            self.assertEqual(ops.load_state_dict(url, "matcher."), {"x": 1})
            load.assert_called_once_with(url, map_location="cpu")

    def test_standalone_matcher_reads_once(self):
        with patch.object(torch, "load", wraps=torch.load) as load:
            actual = self.build()
            self.assertEqual(load.call_count, 1)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_combined_constructor_reads_once(self):
        class Descriptor(nn.Module):
            def __init__(self, *, weights, state_dict, mp):
                super().__init__()
                self.mp = mp
                self.raw = ops.load_state_dict(
                    weights, "extractor.", state_dict=state_dict
                )

        state = {**self.state, "extractor.x": torch.tensor([3.0])}
        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            with patch.object(torch, "load", wraps=torch.load) as load:
                actual = self.build(state, describe=True)
                self.assertEqual(load.call_count, 1)
        self.assertTrue(torch.equal(actual.descriptor.raw["x"], state["extractor.x"]))
        self.assertIsNone(actual.descriptor.mp)
        self.assertIsInstance(actual.conf.weights, str)
        self.assertNotIn("state_dict", vars(actual.conf))

    def test_descriptor_precision_is_fixed_and_passed_at_construction(self):
        class Descriptor(nn.Module):
            def __init__(self, *, weights, state_dict, mp):
                super().__init__()
                self.mp = mp

        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            actual = FlexGlue(weights=None, describe=True, describe_mp="float16")
        self.assertEqual(actual.descriptor.mp, "float16")
        with self.assertRaisesRegex(KeyError, "Fixed at construction"):
            actual.resolve_conf({"describe_mp": None})

    def test_descriptor_precision_rejects_unknown_setting(self):
        with self.assertRaisesRegex(ValueError, "describe_mp"):
            FlexGlue(weights=None, describe=False, describe_mp="bfloat16")

    def test_truncated_layers(self):
        actual = self.build(n_layers=1)
        self.assertEqual(actual.loaded_heads, {0, 1})
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_whole_missing_head_allowed(self):
        state = {
            k: v for k, v in self.state.items() if not k.startswith("matcher.heads.1.")
        }
        self.assertEqual(self.build(state).loaded_heads, {0, 2})

    def test_partial_head_rejected(self):
        state = dict(self.state)
        del state[next(k for k in state if k.startswith("matcher.heads.1."))]
        with self.assertRaisesRegex(KeyError, "only partially"):
            self.build(state)

    def test_non_head_missing_rejected(self):
        state = dict(self.state)
        del state["matcher.input_proj.weight"]
        with self.assertRaisesRegex(KeyError, "Missing weights"):
            self.build(state)

    def test_unexpected_key_rejected(self):
        with self.assertRaisesRegex(KeyError, "Unexpected weights"):
            self.build({**self.state, "matcher.unexpected": torch.tensor(1)})

    def test_empty_checkpoint_rejected(self):
        with self.assertRaisesRegex(KeyError, "No weights with prefix 'matcher.'"):
            self.build({})


if __name__ == "__main__":
    unittest.main()
