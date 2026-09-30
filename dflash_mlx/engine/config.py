# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.runtime.chip_detect import ChipProfile, detect_chip


def _gemv_cap_disabled() -> bool:
    return os.environ.get("DFLASH_DISABLE_GEMV_BLOCK_CAP", "") == "1"

# MLX dispatches quantized_matmul (transpose=True, the serving layout) to a
# batched-GEMV kernel only while the row count stays below a per-chip limit
# (mlx/backend/metal/quantized.cpp, get_qmv_batch_limit). Above the limit it
# falls back to a bm=32 tiled GEMM whose small-M large-matrix bandwidth
# collapses on older GPUs: measured on M1 Max with a 4-bit 5120x248320
# matmul, 245 GB/s at M=1 vs 48 GB/s at M=6 (the chip's limit), so a verify
# block of limit+1 rows runs ~5x slower than the same pass one row shorter.
# Cap the verify block so M stays under the limit unless the caller asked
# for something smaller. Mirrors the upstream limits for large matrices
# (D>4096 or O>4096); smaller ones are rare in decode-time layers and the
# cap stays conservative there anyway.
_QMV_LARGE_MATRIX_LIMITS = {
    13: 6,   # M1 (non-Ultra); Ultra ('d') 12 → safe block 5 / 11
    14: 6,   # M2 (non-Ultra); Ultra ('d') 12
    15: 13,  # M3/M4 → safe block 12
    16: 13,
    17: 33,  # M5 → safe block 32
}


def gemv_verify_block_cap(chip_profile: Optional[ChipProfile] = None) -> int:
    """Max verify rows that keep MLX on the batched-GEMV quantized kernel.

    MLX switches to the tiled GEMM when M >= the qmv batch limit, so the
    safe row count is one below the limit. Returns 0 when unknown (no cap).
    Measured on M1 Max (limit 6): verify pass costs ~55 ms/row at M=3..5
    and 79 ms/row at M=6.
    """
    try:
        profile = chip_profile or detect_chip()
    except Exception:
        return 0
    gen = int(getattr(profile, "arch_gen", 0) or 0)
    tier = str(getattr(profile, "tier", "") or "")
    limit = _QMV_LARGE_MATRIX_LIMITS.get(gen, 0)
    if limit and tier == "ultra":
        limit *= 2
    return max(0, limit - 1)


@dataclass(frozen=True)
class SpeculativeCycleConfig:
    draft_block_size: int
    requested_block_tokens: int
    effective_block_tokens: int
    verify_len_cap: int


def resolve_verify_len_cap(runtime_config: Any, block_tokens: int) -> int:
    requested = int(getattr(runtime_config, "verify_len_cap", 0) or 0)
    if requested <= 0:
        return int(block_tokens)
    return max(1, min(int(block_tokens), requested))

def verify_token_count_for_block(block_len: int, verify_len_cap: int) -> int:
    return max(1, min(int(block_len), int(verify_len_cap)))


def resolve_speculative_cycle_config(
    runtime_config: Any,
    draft_model: Any,
    block_tokens: Optional[int],
    *,
    chip_profile: Optional[ChipProfile] = None,
    gemv_cap_rows: Optional[int] = None,
) -> SpeculativeCycleConfig:
    draft_block_size = int(draft_model.block_size)
    requested_block_tokens = (
        draft_block_size if block_tokens is None else int(block_tokens)
    )
    effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
    # Keep the verify pass on the batched-GEMV quantized kernel: M=block rows
    # must stay strictly under the per-chip qmv batch limit, else every target
    # matmul falls to the tiled GEMM path (see _QMV_LARGE_MATRIX_LIMITS).
    # Only applied when callers pass the real chip profile or an explicit cap,
    # keeping unit tests and CPU/reference paths chip-independent.
    cap = gemv_cap_rows
    if cap is None and chip_profile is not None and not _gemv_cap_disabled():
        cap = gemv_verify_block_cap(chip_profile)
    if cap is not None and 0 < cap < effective_block_tokens:
        effective_block_tokens = max(1, cap)
    return SpeculativeCycleConfig(
        draft_block_size=draft_block_size,
        requested_block_tokens=requested_block_tokens,
        effective_block_tokens=effective_block_tokens,
        verify_len_cap=resolve_verify_len_cap(runtime_config, effective_block_tokens),
    )


def resolve_draft_window(
    runtime_config: Any,
    draft_model: Any,
    *,
    context_len: Optional[int] = None,
    allow_full_attention_context: bool = False,
) -> tuple[int, int]:
    sink = int(getattr(runtime_config, "draft_sink_size", 64))
    requested_window = int(getattr(runtime_config, "draft_window_size", 1024))
    return sink, _effective_draft_window_size(
        draft_model,
        requested_window,
        context_len=context_len,
        allow_full_attention_context=allow_full_attention_context,
    )

def _is_unwindowed_full_attention_draft(draft_model: Any) -> bool:
    args = getattr(draft_model, "args", None)
    if args is None:
        return False
    if int(getattr(args, "sliding_window", 0) or 0) > 0:
        return False
    layer_types = tuple(str(kind) for kind in (getattr(args, "layer_types", ()) or ()))
    if not layer_types:
        return False
    return all(kind == "full_attention" for kind in layer_types)

def _effective_draft_window_size(
    draft_model: Any,
    requested_window: int,
    *,
    context_len: Optional[int] = None,
    allow_full_attention_context: bool = False,
) -> int:
    sliding_window = int(getattr(getattr(draft_model, "args", None), "sliding_window", 0) or 0)
    window = max(1, int(requested_window), sliding_window)
    if (
        allow_full_attention_context
        and context_len is not None
        and _is_unwindowed_full_attention_draft(draft_model)
    ):
        window = max(window, int(context_len))
    return window

def _profile_dflash_cycles_enabled(
    diagnostics: Optional[DiagnosticsConfig] = None,
) -> bool:
    return bool(diagnostics is not None and diagnostics.trace.cycle_events)
