# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

from dflash_mlx.engine.sampling import sample_probs

def match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))


def match_acceptance_length_host(
    drafted_tokens: list[int],
    posterior_tokens: list[int],
) -> int:
    accepted = 0
    for drafted, posterior in zip(drafted_tokens, posterior_tokens):
        if drafted != posterior:
            break
        accepted += 1
    return accepted


def rejection_sample(
    draft_tokens: mx.array,
    target_probs: mx.array,
    draft_probs: mx.array,
    draft_indices: mx.array | None = None,
) -> tuple[int, int]:
    gamma = int(draft_tokens.shape[1])
    p = mx.take_along_axis(
        target_probs[:, :gamma], draft_tokens[..., None], axis=-1
    )[..., 0]
    if draft_indices is None:
        q = mx.take_along_axis(draft_probs, draft_tokens[..., None], axis=-1)[..., 0]
    else:
        q = mx.sum(
            draft_probs * (draft_indices == draft_tokens[..., None]),
            axis=-1,
        )
    accepted = int(
        mx.sum(
            mx.cumprod(
                (mx.random.uniform(shape=q.shape) * q < p).astype(mx.int32),
                axis=-1,
            ),
            axis=-1,
        )[0].item()
    )
    if accepted == gamma:
        return accepted, int(sample_probs(target_probs[:, -1])[0].item())

    residual = target_probs[0, accepted]
    if draft_indices is None:
        residual = residual - draft_probs[0, accepted]
    else:
        indices = draft_indices[0, accepted]
        values = mx.take(residual, indices) - draft_probs[0, accepted]
        residual = mx.put_along_axis(
            residual[None], indices[None], values[None], axis=-1
        )[0]
    residual = mx.maximum(residual, 0)
    total = mx.sum(residual)
    residual = mx.where(
        total > 0,
        residual / mx.maximum(total, 1e-30),
        target_probs[0, accepted],
    )
    return accepted, int(sample_probs(residual[None])[0].item())
