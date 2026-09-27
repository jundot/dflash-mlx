# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Text-only Muse Glimmer (Meta) backbone, loadable through mlx-lm.

Neither mlx-lm's ``muse_glimmer`` (added in 0.32.0, oMLX 0.7.0rc1's pin) nor
mlx-vlm's carries the logit tail and attention-layer indices that the DFlash
target backend needs, and ``load_target_bundle`` loads DFlash targets through
``mlx_lm.utils.load`` only. This module implements the 52-layer text
backbone of ``MuseGlimmerForConditionalGeneration`` checkpoints (vision
weights are dropped in ``sanitize``) and registers itself into
``sys.modules["mlx_lm.models.muse_glimmer"]``, yielding to upstream only once
that module actually carries the tail (see ``register_into_mlx_lm``).

The layer implementation is adapted from the mlx-vlm muse_glimmer port
(mlx-vlm PR #1838 + #1839): gated GQA with a shared weightless qk norm
and a query-side ``qk_scale_factor``, NoPE on full-attention layers
(per-layer rope theta 0), centered (1+w) RMS norms in a 4-norm sandwich,
a normalized token embedding, and an lm_head -> output_multiplier ->
tanh softcap logit tail. ``Model.logits_tail`` is the single home of
that tail: ``Model.__call__`` and the DFlash target backend both call
it, so verify logits can never drift from generation logits.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.base import scaled_dot_product_attention as _sdpa
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_glimmer"
    vocab_size: int = 202048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    attention_bias: bool = False
    sliding_window: int = 2048
    rope_parameters: Optional[dict] = None
    layer_types: Optional[list] = None
    layer_rope_theta: Optional[list[Union[int, float]]] = None
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    tie_word_embeddings: bool = False
    quantization: Optional[dict] = field(default=None)

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 500000.0, "rope_type": "default"}
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - idx) % 4 == 0
                    else "sliding_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            theta = self.rope_parameters.get("rope_theta", 500000.0)
            self.layer_rope_theta = [
                0 if layer_type == "full_attention" else theta
                for layer_type in self.layer_types
            ]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        # mlx-vlm-sanitized artifacts (oMLX oQ outputs) carry per-layer
        # quantization overrides keyed by their runtime paths
        # (language_model.model.* / language_model.lm_head.*). mlx-lm's
        # load_model resolves overrides against THIS module's paths
        # (model.* / lm_head.*) on the same config dict from_dict receives,
        # so translate the keys in place — otherwise e.g. an 8-bit
        # embed_tokens override is missed and the 4-bit global config
        # fails strict loading with a packed-shape mismatch.
        for quant_key in ("quantization", "quantization_config"):
            overrides = params.get(quant_key)
            if not isinstance(overrides, dict):
                continue
            for key in [k for k in overrides if k.startswith("language_model.")]:
                if key.startswith("language_model.model."):
                    new_key = key.replace("language_model.model.", "model.", 1)
                else:
                    new_key = key.replace("language_model.", "", 1)
                overrides[new_key] = overrides.pop(key)

        # VLM checkpoints nest the backbone under text_config; flatten it
        # (nested values win over top-level ones like the VLM's own
        # eos/bos ids, which do not appear in the annotations anyway).
        data = dict(params)
        text_config = data.pop("text_config", None)
        if isinstance(text_config, dict):
            merged = dict(data)
            merged.update(text_config)
            data = merged
        data["model_type"] = "muse_glimmer"
        return cls(
            **{key: value for key, value in data.items() if key in cls.__annotations__}
        )


@partial(mx.compile, shapeless=True)
def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


@mx.compile
def _centered_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    x = x * (1.0 + weight.astype(mx.float32))
    return x.astype(dtype)


class RMSNormNoScale(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint scale is centered at zero (effective scale 1+w)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Transformers applies the centered scale in FP32 before casting back
        # to the activation dtype. Casting earlier changes BF16 decode choices.
        return _centered_rms_norm(x, self.weight, self.eps)


class QuantizedNormedEmbedding(nn.QuantizedEmbedding):
    """Quantized NormedEmbedding that keeps the weightless embedding norm."""

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(super().__call__(inputs))


class NormedEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, hidden_size: int, eps: float):
        super().__init__(vocab_size, hidden_size)
        self.embed_norm = RMSNormNoScale(eps)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(super().__call__(inputs))

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        **kwargs,
    ) -> "QuantizedNormedEmbedding":
        quantized = QuantizedNormedEmbedding.from_embedding(
            self, group_size, bits, mode=mode
        )
        quantized.embed_norm = self.embed_norm
        return quantized


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.use_rope = bool(args.layer_rope_theta[layer_idx])
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)

        theta = (
            float(args.layer_rope_theta[layer_idx])
            if self.use_rope
            else float(args.rope_parameters.get("rope_theta", 500000.0))
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": theta},
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)

        queries = self.qk_norm(queries)
        queries = (queries.astype(mx.float32) * self.qk_scale_factor).astype(
            queries.dtype
        )
        queries = queries.transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = _sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = residual + self.post_attention_layernorm(x)

        residual = x
        x = self.mlp(self.pre_feedforward_layernorm(x))
        return residual + self.post_feedforward_layernorm(x)


class MuseGlimmerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = NormedEmbedding(
            args.vocab_size, args.hidden_size, args.rms_norm_eps
        )
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window

        self.full_attention_idx = self.layer_types.index("full_attention")
        self.sliding_attention_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(hidden_states, cache[self.full_attention_idx])
        sliding_mask = None
        if self.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseGlimmerModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def logits_tail(self, hidden_states: mx.array) -> mx.array:
        """lm_head -> output_multiplier -> tanh softcap, in reference order.

        The single implementation shared by ``__call__`` and the DFlash
        target backend's ``logits_from_hidden``; a divergence here breaks
        draft acceptance silently (drafted and verified distributions
        stop matching while output stays correct).
        """
        logits = self.lm_head(hidden_states) * self.args.output_multiplier
        softcap = self.args.final_logit_softcapping
        return mx.tanh(logits / softcap) * softcap

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return self.logits_tail(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # Two checkpoint layouts exist: the original HF export
        # (model.language_model.* / model.vision_* / lm_head.*) and
        # mlx-vlm-sanitized artifacts such as oMLX oQ outputs
        # (language_model.model.* / language_model.lm_head.* / vision_*).
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key:
                continue
            if key.startswith(
                (
                    "model.vision_tower.",
                    "model.vision_adapter.",
                    "model.vision_projection.",
                    "vision_tower.",
                    "vision_adapter.",
                    "vision_projection.",
                )
            ):
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "model.", 1)
            elif key.startswith("language_model.lm_head."):
                key = key.replace("language_model.", "", 1)
            elif key.startswith("language_model.model."):
                key = key.replace("language_model.", "", 1)
            sanitized[key] = value
        return sanitized


def _upstream_impl_supports_dflash(module: Any) -> bool:
    """True when an upstream ``mlx_lm.models.muse_glimmer`` can serve as a target.

    ``MuseGlimmerTargetOps`` delegates the logit tail to ``Model.logits_tail``
    (lm_head -> output_multiplier -> tanh softcap) so verify logits can never
    drift from generation logits, and builds its sliding/full masks from the
    ``full_attention_idx`` / ``sliding_attention_idx`` the bundled model exposes.
    mlx-lm 872ae88 (0.32.0) ships its own ``models/muse_glimmer.py`` with
    neither, so yielding to it leaves every DFlash decode raising AttributeError
    on ``full_attention_idx``.

    A module with no ``Model`` at all is not an mlx-lm model module (partially
    initialised import, test double): don't argue with it, stand down.
    """
    model_cls = getattr(module, "Model", None)
    if model_cls is None:
        return True
    return hasattr(model_cls, "logits_tail")


def register_into_mlx_lm() -> bool:
    """Seed ``mlx_lm.models.muse_glimmer`` so mlx-lm can load the target.

    Yields to a real upstream module only while that module is usable as a
    DFlash target (see ``_upstream_impl_supports_dflash``); otherwise this
    bundled implementation keeps ownership of the module name.
    """
    import importlib.util

    name = "mlx_lm.models.muse_glimmer"
    bundled = sys.modules[__name__]

    existing = sys.modules.get(name)
    if existing is not None and existing is not bundled:
        if _upstream_impl_supports_dflash(existing):
            return False
    else:
        try:
            has_upstream = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            has_upstream = False
        if has_upstream:
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - unimportable is not usable
                module = None
            if module is not None and _upstream_impl_supports_dflash(module):
                return False

    sys.modules[name] = bundled
    return True
