# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

import mlx.core as mx


def prepare_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: list[int] | None,
) -> mx.array | None:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: mx.array | None = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def sampling_probs(
    logits: mx.array,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    suppress_token_mask: mx.array | None = None,
) -> mx.array:
    if temperature <= 0:
        raise ValueError("sampling_probs requires temperature > 0")
    scores = logits.astype(mx.float32) / float(temperature)
    if suppress_token_mask is not None:
        scores = mx.where(suppress_token_mask, -mx.inf, scores)
    vocab_size = int(scores.shape[-1])
    if 0 < int(top_k) < vocab_size:
        indices = mx.argpartition(-scores, int(top_k) - 1, axis=-1)[..., : int(top_k)]
        scores = mx.take_along_axis(scores, indices, axis=-1)
    else:
        indices = None
    probs = mx.softmax(scores, axis=-1)
    if top_p < 1.0:
        order = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, order, axis=-1)
        keep = mx.cumsum(sorted_probs, axis=-1) - sorted_probs < float(top_p)
        sorted_probs = mx.where(keep, sorted_probs, 0)
        probs = mx.put_along_axis(mx.zeros_like(probs), order, sorted_probs, axis=-1)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    if indices is not None:
        probs = mx.put_along_axis(
            mx.zeros(logits.shape, dtype=probs.dtype),
            indices,
            probs,
            axis=-1,
        )
    return probs


def sample_probs(probs: mx.array) -> mx.array:
    return mx.random.categorical(mx.log(probs)).astype(mx.uint32)


def sample_logits(
    logits: mx.array,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    suppress_token_mask: mx.array | None = None,
) -> mx.array:
    if temperature <= 0:
        return greedy_tokens_with_mask(logits, suppress_token_mask)
    return sample_probs(
        sampling_probs(logits, temperature, top_p, top_k, suppress_token_mask)
    )


def masked_topk_arrays(
    logits_2d: mx.array,
    suppress_token_mask: mx.array | None,
    *,
    width: int,
) -> tuple[mx.array, mx.array]:
    """Per-row top-`width` ids (desc) and masked log-softmax values, as lazy arrays.

    Mirrors greedy_tokens_with_mask masking, so row argmax is always id 0 (up to
    bf16 ties). No eval here: callers fold both arrays into an existing eval point.
    """
    top_width = int(width)
    if top_width <= 0:
        raise ValueError("width must be positive")
    masked = logits_2d
    if suppress_token_mask is not None:
        floor = mx.array(-1e9, dtype=logits_2d.dtype)
        masked = mx.where(suppress_token_mask, floor, logits_2d)
    top = mx.argpartition(masked, kth=-top_width, axis=-1)[:, -top_width:]
    top_logits = mx.take_along_axis(masked, top, axis=-1)
    order = mx.argsort(top_logits, axis=-1)[:, ::-1]
    top = mx.take_along_axis(top, order, axis=-1)
    log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    values = mx.take_along_axis(log_probs, top, axis=-1)
    return top, values


def eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0
