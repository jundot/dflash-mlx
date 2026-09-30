# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class DFlashPrefixKey:
    target_model_id: str
    draft_model_id: str
    capture_layer_ids: tuple[int, ...]
    draft_sink_size: int
    draft_window_size: int
    template_hash: str
    prompt_policy_hash: str
    target_fa_window: int = 0
    # 4: target GDN verify follows mlx-lm normalize_qk, so older snapshots
    # no longer match a fresh prefill.
    format_version: int = 4
