# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import random

import mlx.core as mx

from dflash_mlx.engine.acceptance import (
    match_acceptance_length,
    match_acceptance_length_host,
)

def test_acceptance_length_cases():
    cases = [
        ([], [], 0),
        ([1, 2, 3, 4], [1, 2, 3, 4], 4),
        ([5, 5, 5], [1, 5, 5], 0),
        ([1, 2, 9, 4, 5], [1, 2, 3, 4, 5], 2),
        ([7, 7, 7, 0], [7, 7, 7, 1], 3),
    ]
    for drafted, posterior, expected in cases:
        out = match_acceptance_length(
            mx.array(drafted, dtype=mx.uint32),
            mx.array(posterior, dtype=mx.uint32),
        )
        assert int(out.item()) == expected
        assert match_acceptance_length_host(drafted, posterior) == expected


def test_host_matches_device_on_random_cases():
    rng = random.Random(20260729)
    for _ in range(200):
        length = rng.randrange(0, 17)
        drafted = [rng.randrange(0, 5) for _ in range(length)]
        posterior = [rng.randrange(0, 5) for _ in range(length)]
        device = int(
            match_acceptance_length(
                mx.array(drafted, dtype=mx.uint32),
                mx.array(posterior, dtype=mx.uint32),
            ).item()
        )
        assert match_acceptance_length_host(drafted, posterior) == device
