"""Optional eager fusion contracts; GPU cases require CUDA and Triton."""

import functools
import importlib.util
import unittest
from unittest.mock import patch

import torch

from flexglue import FlexGlue
from flexglue.refiner import ConvRefiner, RoMav2Refiner, local_correlation
from flexglue.utils import ops

HAS_TRITON = importlib.util.find_spec("triton") is not None


class BackendSelectionTest(unittest.TestCase):
    def test_invalid_backend_before_weights(self):
        for name in ("refine_correlation", "readout_backend"):
            with (
                self.subTest(name=name),
                patch.object(
                    ops, "load_state_dict", side_effect=AssertionError("weight read")
                ),
            ):
                with self.assertRaisesRegex(ValueError, name):
                    FlexGlue(**{name: "invalid"})
        with self.assertRaisesRegex(ValueError, "correlation_backend"):
            RoMav2Refiner(correlation_backend="invalid")

    def test_default_and_fixed_configuration(self):
        model = FlexGlue(weights=None, describe=False, n_layers=0)
        self.assertIs(model.heads[0].threshold, ops.sorted_entmax15_threshold)
        for name in ("refine_correlation", "readout_backend"):
            with self.subTest(name=name), self.assertRaisesRegex(KeyError, "Fixed"):
                model.resolve_conf({name: "triton"})
        block = ConvRefiner(4, 48, 4, 1, hidden_blocks=0)
        self.assertIs(block.correlation, local_correlation)
        before = set(block.state_dict())
        block.correlation = functools.partial(local_correlation)
        self.assertEqual(before, set(block.state_dict()))

    def test_default_does_not_import_triton(self):
        import builtins

        original = builtins.__import__

        def checked(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("default imported Triton")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=checked):
            FlexGlue(weights=None, describe=False, n_layers=0)

    @unittest.skipUnless(HAS_TRITON, "Triton not installed")
    def test_cpu_and_ad_fallback(self):
        from flexglue.utils.correlation_triton import local_correlation as fused
        from flexglue.utils.entmax_triton import threshold

        a = torch.randn(1, 48, 3, 5, requires_grad=True)
        b = torch.randn_like(a)
        w = torch.zeros(1, 3, 5, 2)
        torch.testing.assert_close(
            fused(a, b, w, 1, native=local_correlation),
            local_correlation(a, b, w, 1),
            rtol=0,
            atol=0,
        )
        ys = torch.randn(2, 512).sort(descending=True).values
        for state in (torch.enable_grad(), torch.no_grad()):
            with state:
                torch.testing.assert_close(
                    threshold(ys, native=ops.sorted_entmax15_threshold),
                    ops.sorted_entmax15_threshold(ys),
                    rtol=0,
                    atol=0,
                )


@unittest.skipUnless(
    HAS_TRITON and torch.cuda.is_available(), "CUDA and Triton required"
)
class FusedCUDATest(unittest.TestCase):
    def test_correlation_layouts_and_nonfinite_coordinates(self):
        from flexglue.utils import correlation_triton
        from flexglue.utils.correlation_triton import local_correlation as fused

        torch.manual_seed(1701)
        with (
            torch.no_grad(),
            patch.object(
                correlation_triton,
                "_correlation",
                wraps=correlation_triton._correlation,
            ) as launch,
        ):
            for layout in (torch.contiguous_format, torch.channels_last):
                a = torch.randn(1, 48, 11, 13, device="cuda").contiguous(
                    memory_format=layout
                )
                b = torch.randn_like(a)
                w = torch.rand(1, 11, 13, 2, device="cuda") * 2 - 1
                w[0, 0, :3] = torch.tensor(
                    [[float("nan"), 0], [float("inf"), 0], [-1, -1]], device="cuda"
                )
                before = launch.call_count
                torch.testing.assert_close(
                    fused(a, b, w, 1, native=local_correlation),
                    local_correlation(a, b, w, 1),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

                self.assertEqual(launch.call_count, before + 1)

    def test_threshold_supported_and_fallback_shapes(self):
        from flexglue.utils import entmax_triton
        from flexglue.utils.entmax_triton import threshold

        kernel = entmax_triton._threshold
        launches = []

        class Spy:
            def __getitem__(self, grid):
                launches.append(grid)
                return kernel[grid]

        torch.manual_seed(1701)
        with torch.no_grad(), patch.object(entmax_triton, "_threshold", Spy()):
            for shape in ((1, 512), (2, 3, 512), (2, 511), (0, 512)):
                before = len(launches)
                ys = torch.randn(shape, device="cuda").sort(descending=True).values
                torch.testing.assert_close(
                    threshold(ys, native=ops.sorted_entmax15_threshold),
                    ops.sorted_entmax15_threshold(ys),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                self.assertEqual(
                    len(launches) - before, int(shape[-1] == 512 and ys.numel() > 0)
                )
            ys = torch.zeros(2, 512, device="cuda")
            torch.testing.assert_close(
                threshold(ys, native=ops.sorted_entmax15_threshold),
                ops.sorted_entmax15_threshold(ys),
                rtol=0,
                atol=0,
            )

    def test_ineligible_cuda_delegates(self):
        from flexglue.utils import correlation_triton, entmax_triton

        a = torch.randn(1, 48, 3, 5, device="cuda")
        w = torch.zeros(1, 3, 5, 2, device="cuda")
        sentinel = object()
        with patch.object(
            correlation_triton,
            "_correlation",
            side_effect=AssertionError("unsupported launch"),
        ):
            with torch.enable_grad():
                self.assertIs(
                    correlation_triton.local_correlation(
                        a, a, w, 1, native=lambda *args: sentinel
                    ),
                    sentinel,
                )
            with torch.no_grad():
                for args in [
                    (a, a, w, 2),
                    (a.double(), a.double(), w.double(), 1),
                    (a[:, :47], a[:, :47], w, 1),
                ]:
                    self.assertIs(
                        correlation_triton.local_correlation(
                            *args, native=lambda *xs: sentinel
                        ),
                        sentinel,
                    )
        with patch.object(
            entmax_triton,
            "_threshold",
            side_effect=AssertionError("unsupported launch"),
        ):
            ys = torch.randn(2, 512, device="cuda")
            with torch.enable_grad():
                self.assertIs(
                    entmax_triton.threshold(ys, native=lambda x: sentinel), sentinel
                )
            with torch.no_grad():
                for x in [ys.double(), ys[:, :511], ys[:, ::2]]:
                    self.assertIs(
                        entmax_triton.threshold(x, native=lambda x: sentinel), sentinel
                    )

    def test_compiled_refinement_selects_native(self):
        from flexglue.utils.correlation_triton import local_correlation as fused
        from flexglue.utils.entmax_triton import threshold

        a = torch.randn(1, 48, 3, 5, device="cuda")
        w = torch.zeros(1, 3, 5, 2, device="cuda")
        ys = torch.randn(2, 512, device="cuda").sort(descending=True).values
        with torch.no_grad():
            with patch(
                "flexglue.utils.correlation_triton._correlation",
                side_effect=AssertionError("compiled custom correlation"),
            ):
                fn = torch.compile(
                    functools.partial(fused, native=local_correlation),
                    backend="eager",
                    fullgraph=True,
                )
                torch.testing.assert_close(
                    fn(a, a, w, 1), local_correlation(a, a, w, 1), rtol=0, atol=0
                )
            with patch(
                "flexglue.utils.entmax_triton._threshold",
                side_effect=AssertionError("compiled custom threshold"),
            ):
                fn = torch.compile(
                    functools.partial(threshold, native=ops.sorted_entmax15_threshold),
                    backend="eager",
                    fullgraph=True,
                )
                torch.testing.assert_close(
                    fn(ys), ops.sorted_entmax15_threshold(ys), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
