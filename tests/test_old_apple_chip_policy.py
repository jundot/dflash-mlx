# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import mlx.core as mx
import mlx.nn as nn
import pytest

import dflash_mlx.verify_linear as verify_linear
from dflash_mlx.runtime import loading as runtime_loading
from dflash_mlx.runtime.chip_detect import (
    ChipDetectionError,
    chip_profile_from_device_info,
)


def _profile(architecture: str):
    return chip_profile_from_device_info(
        {"architecture": architecture},
        metal_available=True,
        macos_version="26.2",
    )


def test_chip_profile_parses_apple_gpu_family_and_tier():
    profile = _profile("applegpu_g13s")

    assert profile.arch_gen == 13
    assert profile.family == "M1"
    assert profile.tier == "max"
    assert profile.bf16_emulated is True
    assert profile.nax_capable is False


def test_chip_profile_marks_m5_nax_capable_on_supported_macos():
    profile = _profile("applegpu_g17s")

    assert profile.family == "M5"
    assert profile.tier == "max"
    assert profile.bf16_native is True
    assert profile.nax_capable is True


def test_resolve_draft_load_dtype_only_casts_quantized_m1_m2_drafts():
    quant = runtime_loading.parse_draft_quant_spec("w4")

    assert (
        runtime_loading.resolve_draft_load_dtype(
            quant,
            chip_profile=_profile("applegpu_g13s"),
        )
        == mx.float16
    )
    assert (
        runtime_loading.resolve_draft_load_dtype(
            quant,
            chip_profile=_profile("applegpu_g14d"),
        )
        == mx.float16
    )
    assert (
        runtime_loading.resolve_draft_load_dtype(
            quant,
            chip_profile=_profile("applegpu_g15s"),
        )
        is None
    )


def test_resolve_draft_load_dtype_fails_if_chip_detection_fails(monkeypatch):
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: (_ for _ in ()).throw(ChipDetectionError("device info failed")),
    )

    with pytest.raises(ChipDetectionError, match="device info failed"):
        runtime_loading.resolve_draft_load_dtype(
            runtime_loading.parse_draft_quant_spec("w4")
        )
    assert (
        runtime_loading.resolve_draft_load_dtype(
            runtime_loading.parse_draft_quant_spec("w4a32"),
            chip_profile=_profile("applegpu_g13s"),
        )
        is None
    )
    assert (
        runtime_loading.resolve_draft_load_dtype(
            None,
            chip_profile=_profile("applegpu_g13s"),
        )
        is None
    )


def test_cast_floating_model_uses_mlx_apply_callback_signature():
    layer = nn.Linear(2, 2, bias=False)
    layer.weight = mx.ones((2, 2), dtype=mx.bfloat16)

    runtime_loading._cast_floating_model(layer, mx.float16)

    assert layer.weight.dtype == mx.float16


def test_load_draft_bundle_casts_quantized_old_apple_floating_tensors(
    tmp_path,
    monkeypatch,
):
    class FakeDraftModel:
        def __init__(self):
            self.float_value = mx.array([1.0], dtype=mx.bfloat16)
            self.int_value = mx.array([1], dtype=mx.uint32)

        def apply(self, fn):
            self.float_value = fn(self.float_value)
            self.int_value = fn(self.int_value)

    fake_model = FakeDraftModel()
    quant_calls = []

    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (fake_model, {"model_type": "qwen3"}),
    )
    monkeypatch.setattr(
        runtime_loading.nn,
        "quantize",
        lambda model, *, bits, group_size: quant_calls.append((model, bits, group_size)),
    )
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: _profile("applegpu_g13s"),
    )

    install_calls = []
    prewarm_dtypes = []
    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm, w4a32_only: install_calls.append(
            (model, enable_qmm, w4a32_only)
        )
        or 0,
    )
    monkeypatch.setattr(
        verify_linear,
        "prewarm_verify_kernels",
        lambda model, *, input_dtype: prewarm_dtypes.append((model, input_dtype)) or 0,
    )

    model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant="w4")

    assert model is fake_model
    assert quant_calls == [(fake_model, 4, 64)]
    assert fake_model.float_value.dtype == mx.float16
    assert fake_model.int_value.dtype == mx.uint32
    assert install_calls == [(fake_model, True, False)]
    assert prewarm_dtypes == [(fake_model, mx.float16)]
    assert meta["draft_load_dtype"] == "float16"
    assert meta["draft_load_dtype_source"] == "old_apple_bf16_emulation"


def test_load_draft_bundle_casts_w4a32_floating_tensors_to_f32(
    tmp_path,
    monkeypatch,
):
    class FakeDraftModel:
        def __init__(self):
            self.float_value = mx.array([1.0], dtype=mx.float16)
            self.int_value = mx.array([1], dtype=mx.uint32)

        def apply(self, fn):
            self.float_value = fn(self.float_value)
            self.int_value = fn(self.int_value)

    fake_model = FakeDraftModel()
    install_calls = []

    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (fake_model, {"model_type": "qwen3"}),
    )
    monkeypatch.setattr(runtime_loading.nn, "quantize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: _profile("applegpu_g13s"),
    )
    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm, w4a32_only: install_calls.append(
            (model, enable_qmm, w4a32_only)
        ),
    )
    monkeypatch.setattr(verify_linear, "prewarm_verify_kernels", lambda *args, **kwargs: None)

    model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant="w4a32")

    assert model is fake_model
    assert fake_model.float_value.dtype == mx.float32
    assert fake_model.int_value.dtype == mx.uint32
    assert install_calls == [(fake_model, True, True)]
    assert meta["draft_load_dtype"] == "float32"
    assert meta["draft_load_dtype_source"] is None


def test_load_draft_bundle_promotes_dflash2_and_reports_runtime_block(
    tmp_path,
    monkeypatch,
):
    class FakeDFlash2Model:
        is_dflash2 = True
        block_size = 8

        def __init__(self):
            self.float_value = mx.array([1.0], dtype=mx.bfloat16)
            self.int_value = mx.array([1], dtype=mx.uint32)

        def apply(self, fn):
            self.float_value = fn(self.float_value)
            self.int_value = fn(self.int_value)

    fake_model = FakeDFlash2Model()
    install_calls = []

    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (fake_model, {"model_type": "qwen3"}),
    )
    monkeypatch.setattr(runtime_loading.nn, "quantize", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: _profile("applegpu_g13s"),
    )
    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm, w4a32_only: install_calls.append(
            (model, enable_qmm, w4a32_only)
        ),
    )
    prewarm_dtypes = []
    monkeypatch.setattr(
        verify_linear,
        "prewarm_verify_kernels",
        lambda model, *, input_dtype: prewarm_dtypes.append(input_dtype),
    )

    model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant="w4")

    assert model is fake_model
    assert fake_model.float_value.dtype == mx.float32
    assert fake_model.int_value.dtype == mx.uint32
    assert install_calls == [(fake_model, True, True)]
    assert prewarm_dtypes == [mx.float32]
    assert meta["draft_load_dtype"] == "float32"
    assert meta["draft_load_dtype_source"] == "dflash2_bf16_emulation"
    assert meta["draft_quant_spec"] == "w4a32:gs64"
    assert meta["draft_quant_adjustment"] == "dflash2_bf16_emulation"
    assert meta["draft_quant"]["act_bits"] == 32
    assert meta["draft_quant_requested"]["act_bits"] == 16
    assert meta["runtime_block_size"] == 8


def test_load_draft_bundle_preserves_checkpoint_dtype_without_quant(
    tmp_path,
    monkeypatch,
):
    class FakeDraftModel:
        def __init__(self):
            self.float_value = mx.array([1.0], dtype=mx.bfloat16)

        def apply(self, fn):
            self.float_value = fn("float_value", self.float_value)

    fake_model = FakeDraftModel()
    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (fake_model, {"model_type": "qwen3"}),
    )
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: _profile("applegpu_g13s"),
    )

    _model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant=None)

    assert fake_model.float_value.dtype == mx.bfloat16
    assert meta["draft_load_dtype"] is None
    assert meta["draft_load_dtype_source"] is None
