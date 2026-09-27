# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file

import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.engine.target_muse_glimmer import MuseGlimmerTargetOps
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.models import muse_glimmer as muse_module
from dflash_mlx.models.muse_glimmer import Model, ModelArgs

_CHECKPOINT = Path(
    "~/Workspace/models/meta-models/Muse-Glimmer-30B"
).expanduser()


def _tiny_model(num_layers: int = 8, sliding_window: int = 8) -> Model:
    mx.random.seed(0)
    args = ModelArgs(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=256,
        sliding_window=sliding_window,
    )
    return Model(args)


class TestModule:
    def test_layer_pattern_and_nope(self):
        model = _tiny_model()
        assert model.args.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ] * 2
        assert model.model.layers[3].self_attn.use_rope is False
        assert model.model.layers[0].self_attn.use_rope is True

    def test_make_cache_pattern(self):
        caches = _tiny_model().make_cache()
        kinds = [type(c) for c in caches]
        assert kinds == [
            RotatingKVCache,
            RotatingKVCache,
            RotatingKVCache,
            KVCache,
        ] * 2
        assert caches[0].max_size == 8

    def test_centered_rms_norm_preserves_transformers_fp32_order(self):
        # Mirrors the vendored mlx-vlm port (PR #1838 commit edfb0ef1): the
        # centered scale is applied in FP32 before the single downcast, so
        # verify logits keep matching serving logits bit-for-bit.
        from dflash_mlx.models.muse_glimmer import CenteredRMSNorm

        norm = CenteredRMSNorm(4, eps=1e-6)
        norm.weight = (mx.arange(4, dtype=mx.float32) * 0.031 - 0.2).astype(
            mx.bfloat16
        )
        inputs = (
            (mx.arange(4, dtype=mx.float32) * 0.37 - 1.13)
            .reshape(1, 4)
            .astype(mx.bfloat16)
        )

        inputs32 = inputs.astype(mx.float32)
        variance = mx.mean(mx.square(inputs32), axis=-1, keepdims=True)
        expected = inputs32 * mx.rsqrt(variance + 1e-6)
        expected = expected * (1.0 + norm.weight.astype(mx.float32))
        expected = expected.astype(mx.bfloat16)

        assert bool(mx.array_equal(norm(inputs), expected))

    def test_logits_tail_matches_call(self):
        model = _tiny_model()
        ids = mx.array([[1, 2, 3]])
        logits = model(ids)
        hidden = model.model(ids)
        assert bool(mx.allclose(logits, model.logits_tail(hidden)))
        assert float(mx.abs(logits).max()) <= model.args.final_logit_softcapping

    def test_sanitize_drops_vision_and_strips_prefix(self):
        model = _tiny_model()
        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "model.language_model.embed_tokens.weight": mx.zeros((1,)),
            "model.language_model.norm.weight": mx.zeros((1,)),
            "model.vision_tower.ln_pre.weight": mx.zeros((1,)),
            "model.vision_adapter.fc1.weight": mx.zeros((1,)),
            "model.vision_projection.weight": mx.zeros((1,)),
            "lm_head.weight": mx.zeros((1,)),
        }
        sanitized = model.sanitize(weights)
        assert set(sanitized) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        }

    def test_sanitize_handles_mlx_vlm_artifact_layout(self):
        # oMLX oQ artifacts are saved in the mlx-vlm runtime layout
        # (language_model.model.* / language_model.lm_head.* / vision_*),
        # including quantized suffixes.
        model = _tiny_model()
        weights = {
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros((1,)),
            "language_model.model.layers.0.self_attn.q_proj.scales": mx.zeros((1,)),
            "language_model.model.embed_tokens.weight": mx.zeros((1,)),
            "language_model.lm_head.weight": mx.zeros((1,)),
            "vision_tower.ln_pre.weight": mx.zeros((1,)),
            "vision_adapter.fc1.weight": mx.zeros((1,)),
            "vision_projection.weight": mx.zeros((1,)),
        }
        sanitized = model.sanitize(weights)
        assert set(sanitized) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.embed_tokens.weight",
            "lm_head.weight",
        }

    def test_register_yields_to_existing_module(self):
        name = "mlx_lm.models.muse_glimmer"
        saved = sys.modules.pop(name, None)
        sentinel = object()
        sys.modules[name] = sentinel
        try:
            assert muse_module.register_into_mlx_lm() is False
            assert sys.modules[name] is sentinel
        finally:
            del sys.modules[name]
            if saved is not None:
                sys.modules[name] = saved

    def test_register_yields_to_usable_upstream(self):
        # An upstream module that carries the surface MuseGlimmerTargetOps
        # needs (Model.logits_tail) is the one registration should stand down
        # for.
        name = "mlx_lm.models.muse_glimmer"
        saved = sys.modules.pop(name, None)
        upstream = types.ModuleType(name)

        class Model:
            def logits_tail(self, hidden_states):
                return hidden_states

        upstream.Model = Model
        sys.modules[name] = upstream
        try:
            assert muse_module.register_into_mlx_lm() is False
            assert sys.modules[name] is upstream
        finally:
            del sys.modules[name]
            if saved is not None:
                sys.modules[name] = saved

    def test_register_overrides_unusable_upstream(self):
        # oMLX 0.7.0rc1 pins mlx-lm 872ae88, which ships models/muse_glimmer.py
        # with a MuseGlimmerModel that has neither logits_tail nor
        # full_attention_idx. Yielding to it leaves the DFlash target on a model
        # whose masks cannot be built: the first decode raises
        # AttributeError: 'MuseGlimmerModel' object has no attribute
        # 'full_attention_idx'. The bundled implementation has to keep
        # ownership until upstream carries the tail.
        name = "mlx_lm.models.muse_glimmer"
        saved = sys.modules.pop(name, None)
        upstream = types.ModuleType(name)

        class Model:
            def __call__(self, inputs, cache=None):
                return inputs

        upstream.Model = Model
        sys.modules[name] = upstream
        try:
            assert muse_module.register_into_mlx_lm() is True
            assert sys.modules[name] is muse_module
        finally:
            del sys.modules[name]
            if saved is not None:
                sys.modules[name] = saved

    def test_register_installs_this_module(self):
        name = "mlx_lm.models.muse_glimmer"
        saved = sys.modules.pop(name, None)
        try:
            assert muse_module.register_into_mlx_lm() is True
            assert sys.modules[name] is muse_module
        finally:
            sys.modules.pop(name, None)
            if saved is not None:
                sys.modules[name] = saved

    def test_from_dict_translates_quant_override_paths(self):
        # oQ artifact configs key per-layer overrides by mlx-vlm runtime
        # paths; mlx-lm's class_predicate resolves them against this
        # module's paths on the same config dict.
        config = {
            "model_type": "muse_glimmer",
            "text_config": {"model_type": "muse_glimmer_text"},
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "language_model.model.embed_tokens": {"group_size": 64, "bits": 8},
                "language_model.lm_head": {"group_size": 64, "bits": 6},
            },
        }
        ModelArgs.from_dict(config)
        overrides = config["quantization"]
        assert overrides["model.embed_tokens"] == {"group_size": 64, "bits": 8}
        assert overrides["lm_head"] == {"group_size": 64, "bits": 6}
        assert "language_model.model.embed_tokens" not in overrides
        assert overrides["bits"] == 4

    @pytest.mark.skipif(
        not _CHECKPOINT.exists(), reason="Muse Glimmer checkpoint not available"
    )
    def test_real_config_flattens(self):
        config = json.loads((_CHECKPOINT / "config.json").read_text())
        args = ModelArgs.from_dict(config)
        assert args.num_hidden_layers == 52
        assert args.hidden_size == 6656
        assert args.num_key_value_heads == 2
        assert args.sliding_window == 2048
        assert args.qk_scale_factor == 3.87
        assert args.final_logit_softcapping == 20.0
        assert args.tie_word_embeddings is False
        assert args.layer_types[3] == "full_attention"
        assert args.layer_rope_theta[3] == 0


class TestTargetOps:
    def test_resolver_picks_muse_backend(self):
        model = _tiny_model()
        ops = resolve_target_ops(model)
        assert isinstance(ops, MuseGlimmerTargetOps)
        assert ops.family(model) == "muse_glimmer_swa"
        caps = ops.capabilities_for(model)
        assert caps.supports_dflash
        assert caps.supports_kv_trim
        assert not caps.supports_prefix_snapshot

    def test_make_cache_rejects_unsupported_options(self):
        model = _tiny_model()
        ops = MuseGlimmerTargetOps()
        with pytest.raises(ValueError, match="quantization"):
            ops.make_cache(
                model,
                enable_speculative_linear_cache=False,
                quantize_kv_cache=True,
            )
        with pytest.raises(ValueError, match="SWA"):
            ops.make_cache(
                model,
                enable_speculative_linear_cache=False,
                target_fa_window=1024,
            )

    def test_install_speculative_hooks_is_instance_local(self):
        model = _tiny_model()
        ops = MuseGlimmerTargetOps()
        ops.install_speculative_hooks(model)
        assert model.model._dflash_speculative_hooks_installed is True
        # No class-level patching happened.
        other = _tiny_model()
        assert not getattr(
            other.model, "_dflash_speculative_hooks_installed", False
        )

    def test_capture_parity_with_model_forward(self):
        model = _tiny_model()
        ops = MuseGlimmerTargetOps()
        # Long enough to make the sliding mask differ from the causal mask.
        ids = mx.array([[(i * 7) % 60 for i in range(24)]])

        logits, captured = ops.forward_with_hidden_capture(
            model,
            input_ids=ids,
            cache=model.make_cache(),
            capture_layer_ids={0, 2, 4, 8},
        )
        reference = model(ids, cache=model.make_cache())
        mx.eval(logits, reference)
        assert bool(mx.allclose(logits, reference, atol=1e-5))

        # captured[k+1] is the output of layer k: replay manually.
        h = model.model.embed_tokens(ids)
        assert bool(mx.allclose(captured[0], h))
        replay_cache = model.make_cache()
        masks = ops._layer_masks(model.model, h, replay_cache)
        for idx, (layer, mask, layer_cache) in enumerate(
            zip(model.model.layers, masks, replay_cache)
        ):
            h = layer(h, mask=mask, cache=layer_cache)
            if idx + 1 in captured:
                assert bool(mx.allclose(captured[idx + 1], h, atol=1e-5))

    def test_extract_context_feature_concatenates(self):
        ops = MuseGlimmerTargetOps()
        captured = {
            2: mx.ones((1, 3, 4)),
            5: mx.ones((1, 3, 4)) * 2,
        }
        feature = ops.extract_context_feature(captured, [1, 4])
        assert feature.shape == (1, 3, 8)
        assert float(feature[0, 0, 0]) == 1.0
        assert float(feature[0, 0, 4]) == 2.0

    def test_rotating_rollback_round_trip(self):
        # After restore_after_acceptance the DFlash cycle always feeds the
        # target another multi-token verify block (never a single-token
        # decode — the AR fallback rebuilds a fresh cache), so the
        # post-rollback contract is checked with a 2-token step.
        model = _tiny_model()
        ops = MuseGlimmerTargetOps()
        prompt = mx.array([[(i * 5) % 60 for i in range(20)]])
        step = mx.array([[7, 8]])

        ref_cache = model.make_cache()
        model(prompt, cache=ref_cache)
        ref_logits = model(step, cache=ref_cache)

        # Speculative path: prompt, verify 4 extra tokens, roll back, verify.
        cache = model.make_cache()
        model(prompt, cache=cache)
        verify_ids = mx.array([[9, 11, 13, 15]])
        ops.verify_block(
            target_model=model,
            verify_ids=verify_ids,
            target_cache=cache,
            capture_layer_ids={0},
        )
        ops.restore_after_acceptance(
            cache, target_len=prompt.shape[1], acceptance_length=0
        )
        for entry in cache:
            assert int(entry.offset) == prompt.shape[1]
        logits = model(step, cache=cache)
        mx.eval(logits, ref_logits)
        assert bool(mx.allclose(logits, ref_logits, atol=1e-5))

    def test_rollback_past_window_matches_reference(self):
        # Roll back while the rotating ring has already wrapped (offset >
        # window): trims must operate in temporal order.
        model = _tiny_model(sliding_window=8)
        ops = MuseGlimmerTargetOps()
        prompt = mx.array([[(i * 3) % 60 for i in range(14)]])
        step = mx.array([[21, 22]])

        ref_cache = model.make_cache()
        model(prompt, cache=ref_cache)
        ref_logits = model(step, cache=ref_cache)

        cache = model.make_cache()
        model(prompt, cache=cache)
        model(mx.array([[1, 2, 3]]), cache=cache)
        ops.restore_after_acceptance(
            cache, target_len=prompt.shape[1], acceptance_length=0
        )
        logits = model(step, cache=cache)
        mx.eval(logits, ref_logits)
        assert bool(mx.allclose(logits, ref_logits, atol=1e-5))
