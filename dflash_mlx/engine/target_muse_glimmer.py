# Copyright 2026 jundot
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""DFlash target backend for Meta Muse Glimmer (muse_glimmer / _text).

The target loads through ``dflash_mlx.models.muse_glimmer`` (text-only
mlx-lm module). Architecture specifics — gated attention, the query-side
qk_scale_factor, NoPE full layers, the centered 4-norm sandwich — live
inside that module's layers; this backend only routes masks, captures
hidden states, and delegates the logit tail to ``Model.logits_tail`` so
verify logits can never diverge from generation logits.

No class-level hooks are installed (``install_speculative_hooks`` sets a
per-instance flag only, Laguna-style). If a future change patches any
``__call__`` at class level, the installer MUST be registered in oMLX's
``omlx/patches/dflash_lifecycle.py`` (issues #1510/#2252) or engine swaps
poison shared classes.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from dflash_mlx.engine.target_gemma4 import _trim_recent_cache
from dflash_mlx.engine.target_ops import TargetCapabilities

_MUSE_MODEL_TYPES = ("muse_glimmer", "muse_glimmer_text")


class MuseGlimmerTargetOps:
    backend_name = "muse_glimmer"

    def model_type(self, target_model: Any) -> str:
        args = getattr(target_model, "args", None)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            text_config = config.get("text_config", config)
            return str(
                text_config.get("model_type", config.get("model_type", ""))
            ).lower()
        return ""

    def supports_model(self, target_model: Any) -> bool:
        if self.model_type(target_model) not in _MUSE_MODEL_TYPES:
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        args = getattr(self.text_wrapper(target_model), "args", None)
        layer_types = tuple(getattr(args, "layer_types", None) or ())
        # _layer_masks routes sliding/full masks through the attention-layer
        # indices on the text model, so those attributes are part of the
        # contract, not an implementation detail. mlx-lm 0.32 ships its own
        # models/muse_glimmer.py that builds masks per layer type and exposes
        # none of them: declining here means resolve_target_ops fails fast at
        # load with a readable message, instead of raising AttributeError
        # mid-decode and stranding the engine in fallback.
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "full_attention_idx")
            and hasattr(inner, "sliding_attention_idx")
            and hasattr(inner, "sliding_window")
            and "sliding_attention" in layer_types
            and "full_attention" in layer_types
        )

    def family(self, target_model: Any) -> str:
        return "muse_glimmer_swa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        # Snapshot round-trip coverage landed with
        # test_snapshot_round_trip_matches_fresh_continuation (mixed
        # sliding/full caches, rotating ring index, offsets). Verify
        # linears stay off (no muse-specific linear-cache coverage).
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=True,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            supports_verify_linear=False,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_wrapper(self, target_model: Any) -> Any:
        # mlx-lm loads the text-only Model directly; tolerate a VLM-style
        # wrapper for future reuse.
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(
            f"Unsupported Muse Glimmer model wrapper: {type(target_model)!r}"
        )

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported Muse Glimmer text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        logits_tail = getattr(wrapper, "logits_tail", None)
        if logits_tail is None:
            raise AttributeError(
                "Muse Glimmer target must expose logits_tail() "
                "(lm_head + output_multiplier + softcap)"
            )
        return logits_tail(hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        del enable_speculative_linear_cache
        if quantize_kv_cache:
            raise ValueError("Muse Glimmer target KV quantization is not supported yet")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError(
                "Muse Glimmer uses its config-defined SWA/full attention cache"
            )
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "make_cache"):
            return wrapper.make_cache()
        raise AttributeError("Muse Glimmer target must expose make_cache()")

    def install_speculative_hooks(self, target_model: Any) -> None:
        # Per-instance no-op: nothing is patched at class level.
        text_model = self.text_model(target_model)
        text_model._dflash_speculative_hooks_installed = True

    def _layer_masks(
        self, inner: Any, h: mx.array, cache: list[Any]
    ) -> list[Any]:
        full_mask = create_attention_mask(h, cache[inner.full_attention_idx])
        sliding_mask = full_mask
        if inner.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                h,
                cache[inner.sliding_attention_idx],
                window_size=inner.sliding_window,
            )
        return [
            sliding_mask if layer.is_sliding else full_mask for layer in inner.layers
        ]

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        if input_embeddings is None:
            input_embeddings = inner.embed_tokens(input_ids)
        h = input_embeddings

        if cache is None:
            cache = [None] * len(inner.layers)
        else:
            cache = list(cache) + [None] * (len(inner.layers) - len(cache))

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [h]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: h} if 0 in capture_layer_ids else {}

        masks = self._layer_masks(inner, h, cache)
        for idx, (layer, layer_cache, mask) in enumerate(
            zip(inner.layers, cache, masks, strict=True)
        ):
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = idx + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h

        normalized = inner.norm(h)
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError(
            "Muse Glimmer DDTree target-tree verification is not implemented"
        )

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError(
            "Muse Glimmer DDTree target-tree cache commit is not implemented"
        )

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array] | list[mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[int(layer_id) + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        return None

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        replay_ns_total = 0
        for cache_entry in cache_entries:
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset <= target_len:
                continue
            trim_n = offset - int(target_len)
            replay_start_ns = time.perf_counter_ns()
            _trim_recent_cache(cache_entry, trim_n)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        return replay_ns_total

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        draft_cache.clear()
        target_cache.clear()
