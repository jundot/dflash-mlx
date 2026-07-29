# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

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
