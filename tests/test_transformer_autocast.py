"""Keep caller autocast compatible with the selected transformer precision."""

import copy
import io
import unittest
import warnings
from unittest.mock import patch

import torch

from flexglue import FlexGlue


class TransformerAutocastTest(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(461)
            self.model = (
                FlexGlue(
                    weights=None,
                    describe=False,
                    input_dim=4,
                    descriptor_dim=8,
                    search_dim=4,
                    num_heads=2,
                    n_layers=2,
                    query_chunk_size=2,
                    mp=None,
                )
                .eval()
                .to(self.device)
            )
            self.data0 = {
                "features": torch.randn(1, 4, 2, 3, device=self.device),
                "image_size": torch.tensor([[24.0, 16.0]], device=self.device),
                "keypoints": torch.tensor(
                    [[[4.0, 4.0], [12.0, 8.0], [20.0, 12.0]]], device=self.device
                ),
            }
            self.data1 = {
                "features": torch.randn(1, 4, 3, 2, device=self.device),
                "image_size": torch.tensor([[16.0, 24.0]], device=self.device),
            }

    def test_public_calls_cross_outer_and_inner_autocast(self):
        # Wqkv keeps caller inputs under enabled AMP; disabled AMP only adapts
        # the plain linear input to its weights. Transformer hooks keep raw tokens.
        cases = (
            (None, None, torch.float32),
            (None, "bfloat16", torch.float32),
            (None, "float16", torch.float32),
            (torch.bfloat16, None, torch.float32),
            (torch.bfloat16, "bfloat16", torch.bfloat16),
            (torch.bfloat16, "float16", torch.bfloat16),
            (torch.float16, None, torch.float32),
            (torch.float16, "bfloat16", torch.float16),
            (torch.float16, "float16", torch.float16),
        )
        observed = []
        qkv_inputs = []
        hook = self.model.transformers[0].register_forward_pre_hook(
            lambda module, args: observed.append(tuple(x.dtype for x in args))
        )
        self.addCleanup(hook.remove)
        qkv_hook = self.model.transformers[0].self_attn.Wqkv.register_forward_hook(
            lambda module, args, output: qkv_inputs.append(args[0].dtype)
        )
        self.addCleanup(qkv_hook.remove)
        for outer, mp, tokens in cases:
            for layers in (1, 2):
                for dense in (False, True):
                    with self.subTest(outer=outer, mp=mp, layers=layers, dense=dense):
                        observed.clear()
                        qkv_inputs.clear()
                        conf = dict(n_layers=layers, mp=mp, max_distance_px=0)
                        with torch.no_grad(), torch.autocast(
                            self.device,
                            dtype=outer or torch.bfloat16,
                            enabled=outer is not None,
                        ):
                            if dense:
                                output = self.model.match_dense(
                                    self.data0, self.data1, **conf
                                )
                            else:
                                output = self.model(
                                    {"image0": self.data0, "image1": self.data1}, **conf
                                )
                        encoding = outer or torch.float32
                        self.assertEqual(observed, [(encoding,) * 4])
                        self.assertEqual(qkv_inputs, [tokens, tokens])
                        for value in output.values():
                            if isinstance(value, torch.Tensor):
                                self.assertTrue(torch.isfinite(value).all())
                        self.assertEqual(output["stop_layer"], layers)
                        if dense:
                            self.assertEqual(
                                output["dense_confidence0"].shape, (1, 2, 3)
                            )
                            self.assertEqual(
                                output["dense_confidence1"].shape, (1, 3, 2)
                            )
                        else:
                            self.assertEqual(output["matches0"].shape, (1, 3))
                            self.assertEqual(output["keypoints1"].shape, (1, 3, 2))

    def test_zero_layers_preserve_pair_context_inputs(self):
        original = self.model.pair_context
        observed = []

        def capture(x0, x1, *args):
            observed.append((x0.dtype, x1.dtype))
            return original(x0, x1, *args)

        self.model.pair_context = capture
        for outer in (None, torch.bfloat16, torch.float16):
            for mp in (None, "bfloat16", "float16"):
                with self.subTest(outer=outer, mp=mp):
                    observed.clear()
                    with torch.no_grad(), torch.autocast(
                        self.device,
                        dtype=outer or torch.bfloat16,
                        enabled=outer is not None,
                    ):
                        output = self.model.match_dense(
                            self.data0, self.data1, n_layers=0, mp=mp, max_distance_px=0
                        )
                    self.assertEqual(
                        observed, [(outer or torch.float32, outer or torch.float32)]
                    )
                    self.assertEqual(output["stop_layer"], 0)

    def test_disabled_inner_autocast_preserves_typed_transformers(self):
        for dtype in (torch.bfloat16, torch.float16):
            for whole_model in (False, True):
                with self.subTest(dtype=dtype, whole_model=whole_model):
                    model = copy.deepcopy(self.model)
                    (model if whole_model else model.transformers).to(dtype)
                    observed = []
                    hook = model.transformers[0].register_forward_pre_hook(
                        lambda module, args: observed.append(
                            tuple(x.dtype for x in args)
                        )
                    )
                    with torch.no_grad(), torch.autocast(self.device, dtype=dtype):
                        for layers in (1, 2):
                            output = model.match_dense(
                                self.data0,
                                self.data1,
                                n_layers=layers,
                                mp=None,
                                max_distance_px=0,
                            )
                            self.assertTrue(
                                torch.isfinite(output["dense_confidence0"]).all()
                            )
                            self.assertEqual(output["stop_layer"], layers)
                    hook.remove()
                    self.assertEqual(observed, [(dtype,) * 4] * 2)

    def test_precision_switches_preserve_outer_autocast_cache(self):
        observed_cache = []
        block_hook = self.model.transformers[0].register_forward_pre_hook(
            lambda module, args: observed_cache.append(
                torch.is_autocast_cache_enabled()
            )
        )
        self.addCleanup(block_hook.remove)
        for outer in (torch.bfloat16, torch.float16):
            same = "bfloat16" if outer == torch.bfloat16 else "float16"
            other = "float16" if outer == torch.bfloat16 else "bfloat16"
            for cache in (False, True):
                # A projection hook can change token dtype without changing the
                # outer weight cache, so cache policy must use autocast state.
                for fp32_projection in (False, True):
                    hook = (
                        self.model.input_proj.register_forward_hook(
                            lambda module, args, result: result.float()
                        )
                        if fp32_projection
                        else None
                    )
                    try:
                        for sequence in (
                            (same, None, other, same),
                            (other, None, same, other),
                        ):
                            with self.subTest(
                                outer=outer,
                                cache=cache,
                                sequence=sequence,
                                fp32_projection=fp32_projection,
                            ), torch.no_grad(), torch.autocast(
                                self.device, dtype=outer, cache_enabled=cache
                            ):
                                for mp in sequence:
                                    observed_cache.clear()
                                    output = self.model.match_dense(
                                        self.data0,
                                        self.data1,
                                        mp=mp,
                                        max_distance_px=0,
                                    )
                                    self.assertTrue(
                                        torch.isfinite(
                                            output["dense_confidence0"]
                                        ).all()
                                    )
                                    self.assertEqual(
                                        torch.is_autocast_cache_enabled(), cache
                                    )
                                    self.assertEqual(
                                        observed_cache,
                                        [False if mp == other else cache],
                                    )
                    finally:
                        if hook is not None:
                            hook.remove()

    def test_legacy_autocast_queries_match_current_queries(self):
        # Exercise the PyTorch 2.0-2.3 query branch on the current CPU runtime.
        # Keeping the real APIs installed lets torch.autocast itself run normally.
        if self.device != "cpu":
            self.skipTest("This method specifically controls legacy CPU AMP queries.")

        def legacy_queries(obj, name):
            return (
                False
                if obj is torch and name == "get_autocast_dtype"
                else hasattr(obj, name)
            )

        for outer in (torch.bfloat16, torch.float16):
            runs = []
            for legacy in (False, True):
                outputs = []
                query = legacy_queries if legacy else hasattr
                with patch(
                    "flexglue.utils.modules.hasattr", query, create=True
                ), warnings.catch_warnings(), torch.no_grad():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    with torch.autocast("cpu", dtype=outer):
                        for mp in ("bfloat16", None, "float16", "bfloat16"):
                            outputs.append(
                                self.model.match_dense(
                                    self.data0,
                                    self.data1,
                                    mp=mp,
                                    max_distance_px=0,
                                )
                            )
                runs.append(outputs)
            for current, legacy in zip(*runs):
                for name in current:
                    with self.subTest(outer=outer, output=name):
                        if isinstance(current[name], torch.Tensor):
                            torch.testing.assert_close(
                                current[name], legacy[name], rtol=0, atol=0
                            )
                        else:
                            self.assertEqual(current[name], legacy[name])

    def test_custom_transformers_keep_their_token_precision(self):
        class SquareTokens(torch.nn.Module):
            def __init__(self, weighted):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0)) if weighted else 1

            def forward(self, x0, x1, enc0, enc1):
                return x0.square() * self.scale, x1.square() * self.scale

        x0 = torch.linspace(-1, 1, 48, device=self.device).reshape(1, 6, 8).bfloat16()
        x1 = torch.linspace(-2, 1, 48, device=self.device).reshape(1, 6, 8).half()

        def check_tokens(module, tokens):
            self.assertIs(tokens[0], x0)
            self.assertIs(tokens[1], x1)

        for weighted in (False, True):
            block = SquareTokens(weighted).to(self.device)
            self.model.transformers[0] = block
            for outer, mp in ((None, None), (torch.bfloat16, "float16")):
                with self.subTest(outer=outer, mp=mp, weighted=weighted):
                    with block.register_forward_pre_hook(check_tokens), patch.object(
                        self.model, "encode_pair", return_value=(x0, x1, None, None)
                    ), torch.no_grad(), torch.autocast(
                        self.device,
                        dtype=outer or torch.bfloat16,
                        enabled=outer is not None,
                    ):
                        context = self.model.decode_pair(
                            self.data0,
                            self.data1,
                            self.model.resolve_conf({"n_layers": 1, "mp": mp}),
                        )

                    # Autocast can change the precision of the custom operation
                    # without changing the tokens passed to it.
                    with torch.no_grad(), torch.autocast(
                        self.device, dtype=torch.float16, enabled=mp is not None
                    ):
                        y0, y1 = block(x0, x1, None, None)
                    expected = (
                        y0.float().mT.unflatten(-1, (2, 3)),
                        y1.float().mT.unflatten(-1, (3, 2)),
                    )
                    for actual, reference in zip(context.cf, expected):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_projection_pre_hooks_keep_precision_and_gradients(self):
        for outer in (torch.bfloat16, torch.float16):
            for normalize in (False, True):
                with self.subTest(outer=outer, normalize=normalize):
                    model = copy.deepcopy(self.model)
                    block = model.transformers[0].self_attn
                    events, retained = [], []
                    raw = []

                    def observe_block(module, args):
                        events.append("block")
                        raw.append(args[0])

                    def prepare_input(module, args):
                        events.append("prepare")
                        self.assertIs(args[0], raw[-1])
                        self.assertEqual(args[0].dtype, outer)
                        x = args[0]
                        if normalize:
                            x = torch.nn.functional.normalize(x, dim=-1)
                        return (x.float(),)

                    def observe_input(module, args):
                        events.append("input")
                        args[0].retain_grad()
                        retained.append(args[0])

                    def observe_output(module, args, output):
                        events.append("output")
                        output.retain_grad()
                        retained.append(output)

                    handles = (
                        block.register_forward_pre_hook(observe_block),
                        block.Wqkv.register_forward_pre_hook(prepare_input),
                        block.Wqkv.register_forward_pre_hook(observe_input),
                        block.Wqkv.register_forward_hook(observe_output),
                    )
                    for handle in handles:
                        self.addCleanup(handle.remove)
                    with torch.autocast(self.device, dtype=outer):
                        context = model.decode_pair(
                            self.data0,
                            self.data1,
                            model.resolve_conf({"n_layers": 1, "mp": None}),
                        )
                        loss = sum(x.square().mean() for x in context.cf)
                    loss.backward()
                    self.assertEqual(
                        events, ["block", "prepare", "input", "output"] * 2
                    )
                    self.assertEqual(len(retained), 4)
                    for value in retained:
                        self.assertIsNotNone(value.grad)
                        self.assertTrue(torch.isfinite(value.grad).all())

        class NormalizingLinear(torch.nn.Linear):
            def __call__(self, x):
                return super().__call__(
                    torch.nn.functional.normalize(x, dim=-1).float()
                )

        # A subclass may own input conversion in __call__ while inheriting forward.
        # Its result must match the same conversion supplied through a pre-hook.
        for outer in (torch.bfloat16, torch.float16):
            models = [copy.deepcopy(self.model) for _ in range(2)]
            original = models[0].transformers[0].self_attn.Wqkv
            replacement = NormalizingLinear(original.in_features, original.out_features)
            replacement.load_state_dict(original.state_dict())
            models[0].transformers[0].self_attn.Wqkv = replacement.to(self.device)
            handle = (
                models[1]
                .transformers[0]
                .self_attn.Wqkv.register_forward_pre_hook(
                    lambda module, args: (
                        torch.nn.functional.normalize(args[0], dim=-1).float(),
                    )
                )
            )
            self.addCleanup(handle.remove)
            outputs = []
            for model in models:
                with torch.autocast(self.device, dtype=outer):
                    context = model.decode_pair(
                        self.data0,
                        self.data1,
                        model.resolve_conf({"n_layers": 1, "mp": None}),
                    )
                sum(x.square().mean() for x in context.cf).backward()
                outputs.append(context.cf)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for (name, actual), (reference_name, reference) in zip(
                models[0].named_parameters(), models[1].named_parameters()
            ):
                with self.subTest(outer=outer, parameter=name):
                    self.assertEqual(name, reference_name)
                    if reference.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        torch.testing.assert_close(
                            actual.grad, reference.grad, rtol=0, atol=0
                        )

    def test_scripted_blocks_save_and_load(self):
        with torch.no_grad():
            x0, x1, enc0, enc1 = self.model.encode_pair(
                self.data0["features"], self.data1["features"]
            )
            for blocked in (False, True):
                layer = copy.deepcopy(self.model.transformers[0])
                if blocked:
                    layer.self_attn.use_blocked_qkv()
                for block, args in (
                    (layer.self_attn, (x0, enc0)),
                    (layer, (x0, x1, enc0, enc1)),
                ):
                    with self.subTest(block=type(block).__name__, blocked=blocked):
                        expected = block(*args)
                        stream = io.BytesIO()
                        torch.jit.save(torch.jit.script(block), stream)
                        stream.seek(0)
                        loaded = torch.jit.load(stream, map_location=self.device)
                        torch.testing.assert_close(loaded(*args), expected)


if __name__ == "__main__":
    unittest.main()
