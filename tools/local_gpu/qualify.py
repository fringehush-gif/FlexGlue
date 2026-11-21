# SPDX-License-Identifier: MIT
"""Small, explicit-device qualification supplement for the local GPU queue.

Run one target checkout per process. This is correctness qualification, not a
benchmark or an automatic decision to clear a PR's compatibility/quality hold.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import unittest
from contextlib import nullcontext
from fractions import Fraction
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from references import correlation_control, legacy_lattice_kde, selected_root_control

HERE = Path(__file__).resolve().parent
FIXTURE_SHA = "524a5aa6d202da43aaf47d0a614b54e2ff9d9583861025fb555253c7ac103322"
PROFILES = {
    19: ("kde", None, False, False, "d26476031d94e7b7442f9915ea78fefa3c3c62d0"),
    22: ("readout", "native", False, False, "91d0aea051a84e5ea500d192d6af9f0decb1b9b6"),
    23: (
        "readout",
        "promoted",
        False,
        False,
        "a62db4d54f5eaec0d812616993d2fef16e2dcb64",
    ),
    24: (
        "readout",
        "promoted",
        True,
        False,
        "d801d271920f64953f071f6fe735424aca2875b0",
    ),
    25: ("refiner", None, False, False, "ad76dcf8f814e9cfcbada7fe66e2b2c00752a0f9"),
    26: ("refiner", None, True, False, "d25662444adaa70dcc16aa0e4314f63fd94a1c07"),
    36: ("covariance", None, False, True, "3cd16e8dda47aea1052e5daee93f5bbee7c22bd4"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_compilation(function):
    def wrapped(*args, **kwargs):
        assert (
            torch.compiler.is_compiling()
        ), "Required consumer ran without compilation"
        return function(*args, **kwargs)

    return wrapped


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, function) for v in value)
    return value


def compare(actual, reference, *, rtol=1e-5, atol=1e-5):
    """Finite comparison; tolerances are small-fixture guards, not quality claims."""
    if isinstance(actual, torch.Tensor):
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        return (
            float((actual.detach().double() - reference.detach().double()).abs().max())
            if actual.numel()
            else 0.0
        )
    if isinstance(actual, dict):
        assert actual.keys() == reference.keys()
        return max(
            (compare(actual[k], reference[k], rtol=rtol, atol=atol) for k in actual),
            default=0.0,
        )
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(reference)
        return max(
            (compare(a, b, rtol=rtol, atol=atol) for a, b in zip(actual, reference)),
            default=0.0,
        )
    assert actual == reference
    return 0.0


def test_module(repo, filename):
    path = repo / "tests" / filename
    spec = importlib.util.spec_from_file_location("qualification_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def maintained(repo, filename, selection):
    module = test_module(repo, filename)
    suite = unittest.defaultTestLoader.loadTestsFromName(selection, module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert (
        result.testsRun > 0 and result.wasSuccessful() and not result.skipped
    ), "Required CUDA tests failed or skipped"
    return {
        "file": filename,
        "selection": selection,
        "run": result.testsRun,
        "skipped": 0,
    }


def readout(args):
    from flexglue.utils import modules, ops

    promote = args.precision == "promoted"
    rows = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        scores = torch.tensor(
            [0, -0.5, -0.5, -1, -1, -1, -1, -1],
            device=args.device,
            dtype=dtype,
            requires_grad=True,
        )
        threshold = ops.sorted_entmax15_threshold(scores)
        expected_dtype = (
            torch.float32
            if promote and dtype in (torch.float16, torch.bfloat16)
            else dtype
        )
        assert threshold.dtype == expected_dtype, "Wrong precision-policy checkout"
        (gradient,) = torch.autograd.grad(threshold.sum(), scores)
        tau = -1 / 3 - math.sqrt(5 / 18)
        active = [-tau, -0.5 - tau, -0.5 - tau]
        expected = scores.new_tensor([x / sum(active) for x in active] + [0] * 5)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(threshold, threshold.new_tensor([tau]))
        torch.testing.assert_close(gradient, expected)
        rows.append(
            {
                "dtype": str(dtype),
                "threshold_dtype": str(threshold.dtype),
                "gradient_finite": True,
            }
        )

    fixture_path = HERE / "fixtures" / "readout_bf16.pt"
    assert digest(fixture_path) == FIXTURE_SHA, "Frozen BF16 fixture digest mismatch"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    results = []
    retained_topk = 0
    for control in (False, True):
        model = modules.MatchHead(16, 8, None, cross=True).eval().to(args.device)
        model.load_state_dict(fixture["state"], strict=True)
        inputs = tuple(
            x.detach().to(args.device).clone().requires_grad_() for x in fixture["data"]
        )
        saved = map_tensors(inputs, lambda x: x.detach().clone())
        versions = [x._version for x in inputs]
        state = map_tensors(model.state_dict(), lambda x: x.detach().clone())
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if args.device == "cuda" else None
        retained = []
        topk_results = []
        original_topk = torch.Tensor.topk

        def retain_topk(tensor, *positional, **keywords):
            result = original_topk(tensor, *positional, **keywords)
            topk_results.extend((x, x.detach().clone()) for x in result)
            return result

        hook = model.to_search.register_forward_hook(
            lambda m, a, out: retained.append((out, out.detach().clone()))
        )
        context = (
            patch.object(
                ops,
                "sorted_entmax15_threshold",
                lambda ys: selected_root_control(ys, promote),
            )
            if control
            else nullcontext()
        )
        try:
            with (
                context,
                patch.object(torch.Tensor, "topk", retain_topk),
                torch.autocast(args.device, torch.bfloat16),
            ):
                outputs = model(*inputs)
                loss = sum(x.square().sum() for x in outputs)
            names = ["points", "features_i", "features_j"] + [
                name for name, _ in model.named_parameters()
            ]
            gradients = torch.autograd.grad(
                loss, (*inputs, *model.parameters()), allow_unused=True
            )
            for tensor in (*outputs, *[x for x in gradients if x is not None]):
                assert (
                    tensor.device.type == args.device and torch.isfinite(tensor).all()
                )
            temperature = gradients[names.index("temperature")]
            assert temperature is not None and temperature.abs().item() > 0
            assert retained and versions == [x._version for x in inputs]
            compare(inputs, saved, rtol=0, atol=0)
            compare(model.state_dict(), state, rtol=0, atol=0)
            for tensor, snapshot in retained:
                compare(tensor, snapshot, rtol=0, atol=0)
            assert topk_results, "Actual head did not exercise top-k"
            for tensor, snapshot in topk_results:
                compare(tensor, snapshot, rtol=0, atol=0)
            retained_topk += len(topk_results)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            if cuda_rng is not None:
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            results.append((outputs, gradients))
        finally:
            hook.remove()
    error = compare(results[0], results[1])
    precision = []
    if promote:
        fixtures = test_module(args.repo, "test_entmax.py")
        for dtype in (torch.float16, torch.bfloat16):
            logits = fixtures.precision_fixture(args.device, dtype)
            for grid_dtype in (dtype, torch.float32):
                grid = ops.cell_centers(32, 32, (32, 32), device=args.device).to(
                    grid_dtype
                )
                actual = ops.entmax15_window_readout(logits, grid, (32, 32))
                reference = ops.entmax15_window_readout(
                    logits.float(), grid.float(), (32, 32)
                )
                assert actual[0].dtype == torch.promote_types(dtype, grid_dtype)
                assert actual[1].dtype == dtype
                assert ((actual[1] >= 0) & (actual[1] <= 1.01)).all()
                difference = compare(
                    tuple(x.float() for x in actual),
                    reference,
                    rtol=0.01,
                    atol=0.004,
                )
                precision.append(
                    {
                        "logits_dtype": str(dtype),
                        "grid_dtype": str(grid_dtype),
                        "maximum_mass": float(actual[1].max()),
                        "fp32_replay_max_abs": difference,
                    }
                )
    tests = []
    if args.device == "cuda" and promote:
        tests.append(
            maintained(
                args.repo,
                "test_entmax.py",
                "EntmaxPrecisionTest.test_original_fp16_cuda_failure_is_fixed",
            )
        )
    if args.device == "cuda" and args.triton:
        tests.append(
            maintained(args.repo, "test_readout_backend.py", "ReadoutCUDATest")
        )
    return {
        "analytic_thresholds": rows,
        "actual_head_max_abs_difference": error,
        "active_gradients": sum(x is not None for x in results[0][1]),
        "retained_topk_tensors_preserved": retained_topk,
        "promoted_mass_dtype_replays": precision,
        "fixture_sha256": FIXTURE_SHA,
        "maintained_cuda_tests": tests,
    }


def kde(args):
    from flexglue import FlexGlue
    from flexglue.utils import ops

    torch.manual_seed(841)
    edge = torch.tensor([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0])
    cases = [
        (
            torch.stack([edge.roll(i) for i in range(4)], -1)[None],
            torch.rand(1, 9),
            0.5,
        ),
        (torch.rand(2, 31, 4) * 2 - 1, torch.rand(2, 31), 0.25),
        (torch.rand(1, 13, 5) * 2 - 1, torch.rand(1, 13), 1.0),
        (
            torch.rand(1, 19, 2, dtype=torch.float64) * 2 - 1,
            torch.rand(1, 19, dtype=torch.float64),
            0.5,
        ),
    ]
    rows = []
    for points, weights, std in cases:
        values = []
        for function in (ops.lattice_kde, legacy_lattice_kde):
            p, w = [
                x.to(args.device).clone().requires_grad_() for x in (points, weights)
            ]
            versions = [p._version, w._version]
            out = function(p, w, std)
            assert out.device.type == args.device and out.dtype == torch.float32
            coefficient = torch.linspace(
                0.3, 1.7, out.numel(), device=args.device
            ).reshape_as(out)
            grads = torch.autograd.grad((out * coefficient).sum(), (p, w))
            assert versions == [p._version, w._version]
            compare(
                (p, w),
                (points.to(args.device), weights.to(args.device)),
                rtol=0,
                atol=0,
            )
            values.append((out, *grads))
        rows.append(
            {
                "dimensions": points.shape[-1],
                "input_dtype": str(points.dtype),
                "max_abs_difference": compare(*values),
            }
        )

    module = test_module(args.repo, "test_sample_endpoint_lifetime.py")
    torch.manual_seed(104)
    pred, data = module.inputs(((5, 7), (7, 5)), batch=1)
    pred, data = map_tensors((pred, data), lambda x: x.to(args.device))
    saved = map_tensors((pred, data), lambda x: x.clone())
    conf = SimpleNamespace(kde_balance=True, max_num_samples=31, filter_threshold=0.1)
    values = []
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for control in (False, True):
            torch.manual_seed(104)
            context = (
                patch.object(ops, "lattice_kde", legacy_lattice_kde)
                if control
                else nullcontext()
            )
            with context, torch.inference_mode():
                out = FlexGlue.sample_matches(None, pred, *data, conf)
            rng = [torch.get_rng_state()]
            if args.device == "cuda":
                rng.append(torch.cuda.get_rng_state())
            values.append((out, rng))
        compare(*values, rtol=0, atol=0)
        compare((pred, data), saved, rtol=0, atol=0)
        retained = int((values[0][0]["matches0"] >= 0).sum())
        assert retained > 0
    finally:
        torch.use_deterministic_algorithms(deterministic)
    return {
        "helper_gradients": rows,
        "seeded_sampler_outputs_rng_exact": True,
        "retained_matches": retained,
        "scatter_mode": "deterministic; default-scatter variability is separate",
    }


def refiner(args):
    from flexglue import refiner as module

    torch.manual_seed(703)
    helper = []
    for layout in ("contiguous", "projected"):
        a, b = [torch.randn(1, 48, 5, 7).to(args.device) for _ in range(2)]
        if layout == "projected":
            a, b = [
                x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2) for x in (a, b)
            ]
        warp = (torch.rand(1, 5, 7, 2) * 1.8 - 0.9).to(args.device)
        values = []
        for function in (module.local_correlation, correlation_control):
            leaves = [x.detach().clone().requires_grad_() for x in (a, b, warp)]
            out = function(*leaves, 1)
            gradients = torch.autograd.grad(out.square().mean(), leaves)
            compare(leaves, (a, b, warp), rtol=0, atol=0)
            values.append((out, gradients))
        oracle = correlation_control(a, b, warp, 1, fp64_sum=True)
        oracle_error = compare(values[0][0].double(), oracle)
        helper.append(
            {
                "layout": layout,
                "max_abs_difference": compare(*values),
                "fp64_product_sum_max_abs": oracle_error,
            }
        )

    consumers = []
    for dtype in (None, torch.bfloat16, torch.float16):
        torch.manual_seed(709)
        models = [
            module.ConvRefiner(128, 48, 23, 1, hidden_blocks=8).eval().to(args.device)
            for _ in range(2)
        ]
        models[1].load_state_dict(models[0].state_dict())
        data = [
            torch.randn(1, 5, 7, 128),
            torch.randn(1, 5, 7, 128),
            torch.rand(1, 5, 7, 2) * 1.8 - 0.9,
            torch.randn(1, 5, 7, 4),
        ]
        results = []
        for i, model in enumerate(models):
            leaves = [x.to(args.device).clone().requires_grad_() for x in data]
            if i and hasattr(model, "correlation"):
                model.correlation = correlation_control
            control = (
                patch.object(module, "local_correlation", correlation_control)
                if i
                else nullcontext()
            )
            autocast = torch.autocast(args.device, dtype) if dtype else nullcontext()
            with control, autocast:
                out = model(*leaves, (0.8, 1.1))
                loss = sum(x.square().mean() for x in out)
            gradients = torch.autograd.grad(loss, (*leaves, *model.parameters()))
            compare(leaves, [x.to(args.device) for x in data], rtol=0, atol=0)
            results.append((out, gradients))
        consumers.append(
            {"autocast": str(dtype), "max_abs_difference": compare(*results)}
        )
    fused_consumers = []
    tests = []
    if args.device == "cuda":
        with patch.object(
            module.VGG19BN, "forward", require_compilation(module.VGG19BN.forward)
        ):
            tests.append(
                maintained(
                    args.repo,
                    "test_refiner_features.py",
                    "SelectedFeaturesTest."
                    "test_owned_compiled_features_survive_pair_generations",
                )
            )
        if args.triton:
            from flexglue.utils import correlation_triton

            native = module.ConvRefiner(128, 48, 23, 1, hidden_blocks=8)
            fused = module.ConvRefiner(
                128,
                48,
                23,
                1,
                hidden_blocks=8,
                correlation=partial(
                    correlation_triton.local_correlation,
                    native=module.local_correlation,
                ),
            )
            native, fused = [model.eval().to(args.device) for model in (native, fused)]
            fused.load_state_dict(native.state_dict())
            inputs = [x.to(args.device) for x in data]
            saved = map_tensors(inputs, lambda x: x.clone())
            for dtype in (None, torch.bfloat16, torch.float16):
                autocast = (
                    torch.autocast(args.device, dtype) if dtype else nullcontext()
                )
                with (
                    torch.inference_mode(),
                    autocast,
                    patch.object(
                        correlation_triton,
                        "_correlation",
                        wraps=correlation_triton._correlation,
                    ) as launch,
                ):
                    expected = native(*inputs, (0.8, 1.1))
                    actual = fused(*inputs, (0.8, 1.1))
                assert launch.call_count == (1 if dtype is None else 0)
                compare(inputs, saved, rtol=0, atol=0)
                fused_consumers.append(
                    {
                        "autocast": str(dtype),
                        "fused_launches": launch.call_count,
                        "native_max_abs_difference": compare(actual, expected),
                    }
                )
            tests.append(
                maintained(args.repo, "test_correlation_backend.py", "FusedCUDATest")
            )
    return {
        "native_helper": helper,
        "actual_8_block_consumers": consumers,
        "actual_fused_fallback_consumers": fused_consumers,
        "maintained_cuda_tests": tests,
    }


def covariance(args):
    oracle = test_module(args.repo, "test_refiner_covariance.py")
    ref = oracle.RefinerCovarianceTest.tiny_refiner(-9.0)
    raw_fields = []

    class SpatialHead(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            # Above the softplus threshold, a/c are exact FP32 identity values;
            # 1e-6 rounds away. Large b still loses c*c when packed into FP32.
            raw = torch.stack(
                (
                    torch.full_like(x, 0.2, dtype=torch.float32),
                    2.0**18 + 16 * x,
                    2.0**18 + 16 * y,
                    32.0 + 2 * (x + y),
                ),
                0,
            )[None]
            self.register_buffer("raw", raw)
            raw_fields.append(raw.permute(0, 2, 3, 1).clone())

        def forward(self, inputs):
            return self.raw.expand(inputs.shape[0], -1, -1, -1)

    for scale, stage in ref.refiners.items():
        stage.confidence_head = SpatialHead(6 // int(scale))
    ref = ref.to(args.device)
    features = {
        1: torch.zeros(1, 6, 6, 2, device=args.device),
        2: torch.zeros(1, 3, 3, 2, device=args.device),
    }
    values = (
        torch.zeros(1, 2, 2, 2, device=args.device),
        torch.zeros(1, 2, 2, device=args.device),
        features,
        features,
        torch.tensor([[13, 20]], device=args.device),
    )

    def execute(*inputs):
        result = ref.refine(*inputs)
        origins = []
        for stage in ref.refiners.values():
            raw = stage.confidence_head.raw.permute(0, 2, 3, 1)
            origins.append(
                torch.stack(
                    (
                        F.softplus(raw[..., 1]) + 1e-6,
                        raw[..., 2],
                        F.softplus(raw[..., 3]) + 1e-6,
                    ),
                    -1,
                )
            )
        return result, origins

    phases = [("eager", execute)]
    if args.compile:
        # CPU mode checks Dynamo capture only, never CUDA qualification.
        backend = "inductor" if args.device == "cuda" else "eager"
        phases.append(
            (
                "compiled",
                torch.compile(
                    require_compilation(execute), backend=backend, fullgraph=True
                ),
            )
        )
    factors = [raw[..., 1:] for raw in raw_fields]
    for factor in factors:
        assert torch.equal(
            factor[..., 1].square() + factor[..., 2].square(), factor[..., 1].square()
        )
    rows = []
    checker = oracle.RefinerCovarianceTest()
    for name, function in phases:
        with torch.inference_mode():
            for _ in range(3):
                out, origins = function(*values)
            assert out["dense_covi_ij"].device.type == args.device
            if args.device == "cuda":
                torch.cuda.synchronize()
            if name == "compiled" and args.device == "cuda":
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    out, origins = function(*values)
                    torch.cuda.synchronize()
                kernels = [
                    event.name
                    for event in profile.events()
                    if str(event.device_type) == "DeviceType.CUDA"
                ]
                assert (
                    kernels
                ), "No CUDA activity was observed for the compiled consumer"
            else:
                kernels = []
        output = out["dense_covi_ij"].cpu()
        compare([factor.cpu() for factor in origins], factors, rtol=0, atol=0)
        for y in range(6):
            for x in range(6):
                terms = oracle.bilinear_terms(factors[0], y, x, (6, 6))
                terms.append((Fraction(1), factors[1][0, y, x].tolist()))
                wanted = oracle.covariance_reference(
                    terms, (Fraction(13, 6), Fraction(20, 6))
                )
                a, b, _ = terms[-1][1]
                weak = (a / (13 / 6), b / (20 / 6))
                checker.assert_covariance(
                    output[0, y, x], wanted, [(1, 0), (0, 1), weak]
                )
        rows.append(
            {
                "phase": name,
                "decimal_checked_pixels": 36,
                "cuda_kernel_events": len(kernels),
            }
        )
    return {
        "nonuniform_actual_two_stage": rows,
        "compiled_cuda_qualified": args.compile and args.device == "cuda",
        "remaining_hold": (
            "FP64 covariance output and structured-state API/default acceptance; "
            "broader real-checkpoint quality remains unqualified"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--expected-head",
        help="Explicit reviewed replacement for this profile's pinned head",
    )
    parser.add_argument("--pr", type=int, choices=tuple(PROFILES), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.case, args.precision, args.triton, args.compile, pinned = PROFILES[args.pr]
    triton_required = (
        args.triton
        or args.case == "refiner"
        or (args.case == "covariance" and args.compile)
    )
    expected_head = args.expected_head or pinned
    args.repo = args.repo.resolve()
    report = {
        "pr": args.pr,
        "case": args.case,
        "device": args.device,
        "precision": args.precision,
        "triton_required": triton_required,
        "expected_head": expected_head,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "harness_sha256": digest(Path(__file__)),
        "references_sha256": digest(HERE / "references.py"),
        "status": "incomplete",
        "cuda_qualified": False,
    }
    try:
        if not __debug__:
            raise RuntimeError(
                "Do not run qualification with Python -O; assertions are required"
            )
        if args.device == "cuda" and triton_required:
            assert (
                os.environ.get("TRITON_INTERPRET") != "1"
            ), "Triton interpreter mode cannot qualify CUDA kernels"
        assert (args.repo / "flexglue").is_dir(), "Target source checkout is missing"
        proc = subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        report["head"] = proc.stdout.strip() if proc.returncode == 0 else None
        proc = subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
        )
        report["git_toplevel"] = proc.stdout.strip() if proc.returncode == 0 else None
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(args.repo),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            capture_output=True,
        )
        report["dirty_state"] = (
            proc.stdout.splitlines() if proc.returncode == 0 else None
        )
        report["source_sha256"] = {
            str(p.relative_to(args.repo)): digest(p)
            for p in sorted((args.repo / "flexglue").rglob("*.py"))
        }
        report["test_source_sha256"] = {
            str(p.relative_to(args.repo)): digest(p)
            for p in sorted((args.repo / "tests").glob("*.py"))
        }
        if args.device == "cuda":
            assert (
                report["git_toplevel"]
                and Path(report["git_toplevel"]).resolve() == args.repo
            ), "Target must be the actual Git checkout root"
            assert (
                report["head"] == expected_head
            ), "Git HEAD differs from the reviewed profile; inspect before overriding"
            assert (
                report["dirty_state"] is not None
            ), "CUDA qualification requires Git provenance"
            changes = [
                line
                for line in report["dirty_state"]
                if not line.startswith("?? ")
                or line[3:].startswith(("flexglue/", "tests/"))
            ]
            assert (
                not changes
            ), "Tracked or executable source edits require a reviewed clean commit"
            assert (
                torch.cuda.is_available()
            ), "CUDA is unavailable; this cannot clear a GPU gate"
            report["gpu"] = torch.cuda.get_device_name()
            report["cuda_runtime"] = torch.version.cuda
            report["gpu_memory_bytes"] = torch.cuda.get_device_properties(
                0
            ).total_memory
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            if triton_required:
                import triton

                report["triton"] = triton.__version__
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        sys.path.insert(0, str(args.repo))
        package = importlib.import_module("flexglue")
        assert (
            Path(package.__file__).resolve().is_relative_to(args.repo)
        ), "Imported an installed FlexGlue instead of the target checkout"
        report["evidence"] = {
            "readout": readout,
            "kde": kde,
            "refiner": refiner,
            "covariance": covariance,
        }[args.case](args)
        for name, module in tuple(sys.modules.items()):
            if name == "flexglue" or name.startswith("flexglue."):
                assert (
                    Path(module.__file__).resolve().is_relative_to(args.repo)
                ), f"Wrong source origin: {name}"
        report["cuda_qualified"] = args.device == "cuda" and (
            args.case != "covariance" or args.compile
        )
        report["status"] = (
            "passed"
            if report["cuda_qualified"]
            else "cpu-smoke-passed"
            if args.device == "cpu"
            else "incomplete"
        )
    except Exception:
        report["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("case", "device", "head", "status", "cuda_qualified")
                if k in report
            }
        )
    )
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return 0 if report["status"] in ("passed", "cpu-smoke-passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
