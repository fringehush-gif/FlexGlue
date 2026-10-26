import mmap
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_default_copy_and_opted_local_mapping(self):
        raw = {"matcher.x": torch.tensor([1.0]), "extractor.y": torch.tensor([2.0])}
        torch.save(raw, self.path)
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(torch.equal(
                ops.load_state_dict(self.path, "matcher.")["x"], raw["matcher.x"]
            ))
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        with patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(
                torch.equal(
                    ops.load_state_dict(self.path, "matcher.", allow_mmap=True)["x"],
                    raw["matcher.x"],
                )
            )
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True, mmap=True
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

    def test_legacy_format_uses_copy_loader(self):
        torch.save(self.state, self.path, _use_new_zipfile_serialization=False)
        with patch.object(torch, "load", wraps=torch.load) as load:
            actual = FlexGlue(weights=self.path, **self.conf)
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True
        )
        self.assertEqual(actual.loaded_heads, {0, 1, 2})

    def test_exact_module_assignment_preserves_grad_and_storage_lifetime(self):
        original = nn.Linear(4, 3)
        original.weight.requires_grad_(False)
        torch.save(original.state_dict(), self.path)
        checkpoint = ops._read_state_dict(self.path, allow_mmap=True)
        target = nn.Linear(4, 3)
        target.weight.requires_grad_(False)
        (_, assigned) = ops.load_into_module(
            target, checkpoint, self.path, allow_assign=True
        )
        self.assertTrue(assigned)
        self.assertEqual(target.weight.data_ptr(), checkpoint["weight"].data_ptr())
        self.assertFalse(target.weight.requires_grad)
        self.assertTrue(target.bias.requires_grad)
        del checkpoint
        self.assertTrue(torch.equal(target.weight, original.weight))
        self.assertTrue(torch.equal(target.bias, original.bias))

    def test_alias_or_dtype_mismatch_uses_copy(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(3))
                self.weight_copy = self.weight

        torch.save({"weight": torch.ones(3), "weight_copy": torch.ones(3)}, self.path)
        tied = Tied()
        before = tied.weight
        (_, assigned) = ops.load_into_module(
            tied, ops._read_state_dict(self.path, allow_mmap=True), self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)
        self.assertIs(tied.weight, before)
        self.assertIs(tied.weight_copy, before)

        source = nn.Linear(4, 3)
        torch.save(source.state_dict(), self.path)
        target = nn.Linear(4, 3, dtype=torch.float64)
        (_, assigned) = ops.load_into_module(
            target, ops._read_state_dict(self.path, allow_mmap=True), self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)
        self.assertEqual(target.weight.dtype, torch.float64)
        self.assertTrue(torch.equal(target.weight, source.weight.double()))

    def test_nondefault_global_dtype_disables_mapping(self):
        torch.save(self.state, self.path)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        finally:
            torch.set_default_dtype(previous)

    def test_nondefault_global_device_disables_mapping(self):
        torch.save(self.state, self.path)
        with patch.object(torch, "get_default_device", return_value=torch.device("meta")):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True
        )

    def test_missing_new_torch_apis_and_legacy_load_signature_use_copy(self):
        torch.save(nn.Linear(4, 3).state_dict(), self.path)
        with patch.object(torch, "get_default_device", None):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        with patch.object(torch.serialization, "get_default_mmap_options", None):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )

        class LegacyLinear(nn.Linear):
            def load_state_dict(self, state_dict, strict=True):
                return super().load_state_dict(state_dict, strict=strict)

        legacy = LegacyLinear(4, 3)
        (_, assigned) = ops.load_into_module(
            legacy, ops._read_state_dict(self.path, allow_mmap=True), self.path,
            allow_assign=True,
        )
        self.assertFalse(assigned)

    def test_nonposix_platform_uses_ordinary_loader(self):
        torch.save(self.state, self.path)
        with patch.object(ops, "os", SimpleNamespace(name="nt")):
            with patch.object(torch, "load", wraps=torch.load) as load:
                ops._read_state_dict(self.path, allow_mmap=True)
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True
        )

    def test_shared_mapping_policy_falls_back_without_mutating_checkpoint(self):
        torch.save({"weight": torch.ones(3)}, self.path)
        original_bytes = Path(self.path).read_bytes()
        previous = torch.serialization.get_default_mmap_options()
        try:
            torch.serialization.set_default_mmap_options(mmap.MAP_SHARED)
            with patch.object(torch, "load", wraps=torch.load) as load:
                state = ops._read_state_dict(self.path, allow_mmap=True)
            load.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )

            class OneWeight(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.zeros(3))

            model = OneWeight()
            (_, assigned) = ops.load_into_module(
                model, state, self.path, allow_assign=True
            )
            self.assertFalse(assigned)
            with torch.no_grad():
                model.weight.add_(2)
            self.assertEqual(torch.serialization.get_default_mmap_options(), mmap.MAP_SHARED)
            self.assertEqual(Path(self.path).read_bytes(), original_bytes)
            self.assertTrue(torch.equal(state["weight"], torch.ones(3)))
        finally:
            torch.serialization.set_default_mmap_options(previous)

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
            self.assertNotIn("mmap", load.call_args.kwargs)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_default_constructor_copies_independent_of_checkpoint_rewrite(self):
        actual = self.build()
        before = {k: v.detach().clone() for k, v in actual.state_dict().items()}
        torch.save({k: torch.zeros_like(v) for k, v in self.state.items()}, self.path)
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_opted_constructor_assigns_exact_state(self):
        assigned = []
        original = ops.load_into_module

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            assigned.append(result[1])
            return result

        with patch.object(ops, "load_into_module", side_effect=observed):
            with patch.object(torch, "load", wraps=torch.load) as load:
                actual = self.build(mmap_weights=True)
        load.assert_called_once_with(
            self.path, map_location="cpu", weights_only=True, mmap=True
        )
        self.assertEqual(assigned, [True])
        for name, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, self.original.state_dict()[name]))

    def test_explicit_reload_preserves_parameter_owners(self):
        actual = self.build(mmap_weights=True)
        owners = {name: id(value) for name, value in actual.named_parameters()}
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
        optimizer_owners = [id(value) for group in optimizer.param_groups for value in group["params"]]
        actual.load_weights(self.path)
        self.assertEqual(
            owners, {name: id(value) for name, value in actual.named_parameters()}
        )
        self.assertEqual(
            optimizer_owners,
            [id(value) for group in optimizer.param_groups for value in group["params"]],
        )

    def test_combined_constructor_reads_once(self):
        class Descriptor(nn.Module):
            def __init__(
                self, *, weights, state_dict, mp, mmap_weights, _assign_source_unique,
                _assign_owned_state
            ):
                super().__init__()
                self.mp = mp
                self.mmap_weights = mmap_weights
                self.assign_source_unique = _assign_source_unique
                self.assign_owned_state = _assign_owned_state
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
        self.assertFalse(actual.descriptor.assign_source_unique)
        self.assertTrue(actual.descriptor.assign_owned_state)
        self.assertFalse(actual.descriptor.mmap_weights)
        self.assertIsInstance(actual.conf.weights, str)
        self.assertNotIn("state_dict", vars(actual.conf))

    def test_cross_prefix_checkpoint_alias_disables_assignment(self):
        class Descriptor(nn.Module):
            def __init__(
                self, *, weights, state_dict, mp, mmap_weights, _assign_source_unique,
                _assign_owned_state
            ):
                super().__init__()
                self.assign_source_unique = _assign_source_unique

        state = {
            **self.state,
            "extractor.x": self.state["matcher.input_proj.weight"],
        }
        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            actual = self.build(state, describe=True, mmap_weights=True)
        self.assertFalse(actual.descriptor.assign_source_unique)
        self.assertEqual(actual.loaded_heads, {0, 1, 2})

    def test_descriptor_precision_is_fixed_and_passed_at_construction(self):
        class Descriptor(nn.Module):
            def __init__(
                self, *, weights, state_dict, mp, mmap_weights, _assign_source_unique,
                _assign_owned_state
            ):
                super().__init__()
                self.mp = mp

        with patch("flexglue.flexglue.DINOv3Descriptor", Descriptor):
            actual = FlexGlue(weights=None, describe=True, describe_mp="float16")
        self.assertEqual(actual.descriptor.mp, "float16")
        with self.assertRaisesRegex(KeyError, "Fixed at construction"):
            actual.resolve_conf({"describe_mp": None})
        with self.assertRaisesRegex(KeyError, "Fixed at construction"):
            actual.resolve_conf({"mmap_weights": True})

    def test_mmap_option_rejects_nonboolean(self):
        with self.assertRaisesRegex(ValueError, "mmap_weights"):
            FlexGlue(weights=None, describe=False, mmap_weights="yes")

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
        assigned = []
        original = ops.load_into_module

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            assigned.append(result[1])
            return result

        with patch.object(ops, "load_into_module", side_effect=observed):
            actual = self.build(state)
        self.assertEqual(actual.loaded_heads, {0, 2})
        self.assertEqual(assigned, [False])

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
