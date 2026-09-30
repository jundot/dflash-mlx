# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dflash_mlx.draft_backend import DraftBackend, EagerDraftBackend
from dflash_mlx.engine.target_ops import TargetOps, bind_draft_to_target
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime.loading import (
    load_draft_bundle,
    load_target_bundle,
)
from dflash_mlx.runtime.registry import (
    ModelSupportSpec,
    resolve_effective_draft_quant,
    resolve_model_support_spec,
    resolve_optional_draft_ref,
    supported_base_models,
    validate_support_spec_family,
)


@dataclass(frozen=True)
class RuntimeBundle:
    target_model: Any
    tokenizer: Any
    target_meta: dict[str, Any]
    draft_model: Any
    draft_meta: dict[str, Any]
    draft_backend: DraftBackend
    target_ops: TargetOps
    resolved_model_ref: str
    resolved_draft_ref: str
    effective_draft_quant: str | None
    support_spec: ModelSupportSpec | None


def load_runtime_bundle(
    *,
    model_ref: str | Path,
    draft_ref: str | None,
    draft_quant: str | None = None,
    verify_config: VerifyConfig | None = None,
    quantize_kv_cache: bool = False,
    lazy: bool = True,
) -> RuntimeBundle:
    support_spec = resolve_model_support_spec(str(model_ref))
    resolved_draft_ref = resolve_optional_draft_ref(str(model_ref), draft_ref)
    if not resolved_draft_ref:
        raise ValueError(
            f"No DFlash draft model found for '{model_ref}'.\n"
            f"Use --draft to specify one, or check https://huggingface.co/z-lab for available drafts.\n"
            f"Supported base models: {supported_base_models()}"
        )
    target_bundle = load_target_bundle(
        model_ref,
        lazy=lazy,
        quantize_kv_cache=quantize_kv_cache,
        verify_config=verify_config,
    )
    target_model = target_bundle.model
    tokenizer = target_bundle.tokenizer
    target_meta = target_bundle.meta
    target_ops = target_bundle.target_ops
    resolved_model_ref = str(target_meta.get("resolved_model_ref") or model_ref)
    if support_spec is None:
        support_spec = resolve_model_support_spec(resolved_model_ref)

    if support_spec is not None:
        validate_support_spec_family(
            model_ref=resolved_model_ref,
            support_spec=support_spec,
            actual_family=target_ops.family(target_model),
        )
    effective_draft_quant = resolve_effective_draft_quant(
        draft_quant=draft_quant,
        resolved_draft_ref=resolved_draft_ref,
        support_spec=support_spec,
    )

    draft_model, draft_meta = load_draft_bundle(
        resolved_draft_ref,
        lazy=lazy,
        draft_quant=effective_draft_quant,
    )
    draft_meta = dict(draft_meta)
    effective_draft_quant = draft_meta.get(
        "draft_quant_spec",
        effective_draft_quant,
    )
    draft_meta["draft_quant_spec"] = effective_draft_quant
    draft_meta["draft_quant_source"] = (
        "explicit"
        if (draft_quant or "").strip() and (draft_quant or "").strip().lower() != "none"
        else "model_default"
        if effective_draft_quant is not None
        else "none"
    )
    draft_logits_w4a32_swapped = 0
    if (
        bool(getattr(draft_model, "is_dflash2", False))
        and draft_meta.get("draft_load_dtype") == "float32"
    ):
        from dflash_mlx.verify_linear import install_w4a32_draft_logits_linear

        draft_logits_w4a32_swapped = install_w4a32_draft_logits_linear(
            target_model,
            target_ops=target_ops,
            enable_qmm=bool(target_meta.get("verify_qmm_enabled", False)),
        )
    draft_meta["draft_logits_w4a32_swapped"] = draft_logits_w4a32_swapped
    draft_backend = EagerDraftBackend()
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)
    return RuntimeBundle(
        target_model=target_model,
        tokenizer=tokenizer,
        target_meta=target_meta,
        draft_model=draft_model,
        draft_meta=draft_meta,
        draft_backend=draft_backend,
        target_ops=target_ops,
        resolved_model_ref=resolved_model_ref,
        resolved_draft_ref=resolved_draft_ref,
        effective_draft_quant=effective_draft_quant,
        support_spec=support_spec,
    )
