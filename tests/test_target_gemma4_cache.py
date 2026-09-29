# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps


def _kv(start: int, length: int) -> tuple[mx.array, mx.array]:
    keys = mx.arange(start, start + length, dtype=mx.float32).reshape(1, 1, length, 1)
    values = keys + 1000
    return keys, values


def _append_tokens_one_by_one(cache: RotatingKVCache, start: int, count: int) -> None:
    for token in range(start, start + count):
        keys, values = _kv(token, 1)
        cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)


def _clone_rotating_cache(cache: RotatingKVCache) -> RotatingKVCache:
    clone = RotatingKVCache(max_size=cache.max_size, keep=cache.keep)
    clone.offset = cache.offset
    clone._idx = cache._idx
    if cache.keys is not None:
        clone.keys = mx.array(cache.keys)
        clone.values = mx.array(cache.values)
        mx.eval(clone.keys, clone.values)
    return clone


def _reference_temporal_trim(cache: RotatingKVCache, n: int) -> None:
    n = min(cache.offset, max(0, int(n)))
    keys = cache._temporal_order(cache.keys)
    values = cache._temporal_order(cache.values)
    keep_len = max(0, int(keys.shape[2]) - n)
    cache.keys = keys[..., :keep_len, :]
    cache.values = values[..., :keep_len, :]
    cache.offset -= n
    cache._idx = keep_len
    mx.eval(cache.keys, cache.values)


def _temporal_key_values(cache: RotatingKVCache) -> list[int]:
    keys = cache._temporal_order(cache.keys)
    mx.eval(keys)
    return np.array(keys).reshape(-1).astype(int).tolist()

def _state_key_values(cache: KVCache) -> list[int]:
    # mlx-lm 0.32 widened ``state`` to (keys, values, offset) and keeps the
    # padded ``step``-block buffers in it; the live tokens come from
    # ``keys_and_values()``. 0.31's ``state`` pair is already the live view.
    if hasattr(cache, "keys_and_values") and cache.keys is not None:
        keys, _ = cache.keys_and_values()
    else:
        keys = cache.state[0]
    mx.eval(keys)
    return np.array(keys).reshape(-1).astype(int).tolist()


def test_gemma4_restore_after_acceptance_temporal_trims_rotating_cache() -> None:
    cache = RotatingKVCache(max_size=5, keep=0)
    _append_tokens_one_by_one(cache, 0, 8)

    verify_keys, verify_values = _kv(8, 3)
    cache.update_and_fetch(verify_keys, verify_values)
    mx.eval(cache.keys, cache.values)

    expected = _clone_rotating_cache(cache)
    _reference_temporal_trim(expected, 2)

    naive = _clone_rotating_cache(cache)
    naive.trim(2)
    mx.eval(naive.keys, naive.values)

    actual = _clone_rotating_cache(cache)
    target_len = actual.offset - 2
    Gemma4TargetOps().restore_after_acceptance(
        [actual],
        target_len=target_len,
        acceptance_length=1,
        drafted_tokens=2,
    )
    mx.eval(actual.keys, actual.values)

    assert actual.offset == expected.offset == 9
    assert actual._idx == expected._idx == 5
    assert _temporal_key_values(actual) == _temporal_key_values(expected) == [4, 5, 6, 7, 8]

    naive_temporal = _temporal_key_values(naive)
    assert naive_temporal != [4, 5, 6, 7, 8]
    assert 9 in naive_temporal and 10 in naive_temporal

def test_gemma4_restore_after_acceptance_trims_mixed_rotating_and_full_kv_cache() -> None:
    rotating = RotatingKVCache(max_size=5, keep=0)
    full = KVCache()

    _append_tokens_one_by_one(rotating, 0, 8)
    full.update_and_fetch(*_kv(0, 8))
    verify_keys, verify_values = _kv(8, 3)
    rotating.update_and_fetch(verify_keys, verify_values)
    full.update_and_fetch(verify_keys, verify_values)
    mx.eval(rotating.keys, rotating.values, full.keys, full.values)

    Gemma4TargetOps().restore_after_acceptance(
        [rotating, full],
        target_len=9,
        acceptance_length=1,
        drafted_tokens=2,
    )
    mx.eval(rotating.keys, rotating.values, full.keys, full.values)

    assert rotating.offset == 9
    assert rotating._idx == 5
    assert _temporal_key_values(rotating) == [4, 5, 6, 7, 8]
    assert full.offset == 9
    assert _state_key_values(full) == list(range(9))

def test_gemma4_restore_after_acceptance_noops_when_cache_is_already_at_target_len() -> None:
    rotating = RotatingKVCache(max_size=5, keep=0)
    _append_tokens_one_by_one(rotating, 0, 4)
    before = _temporal_key_values(rotating)
    before_offset = rotating.offset
    before_idx = rotating._idx

    replay_ns = Gemma4TargetOps().restore_after_acceptance(
        [rotating],
        target_len=4,
        acceptance_length=0,
        drafted_tokens=0,
    )

    assert replay_ns == 0
    assert rotating.offset == before_offset
    assert rotating._idx == before_idx
    assert _temporal_key_values(rotating) == before
