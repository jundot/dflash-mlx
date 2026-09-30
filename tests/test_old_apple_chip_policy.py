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
        lambda model, *, enable_qmm: install_calls.append((model, enable_qmm)) or 0,
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
    assert install_calls == [(fake_model, True)]
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
    monkeypatch.setattr(verify_linear, "install_verify_linears", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_linear, "prewarm_verify_kernels", lambda *args, **kwargs: None)

    model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant="w4a32")

    assert model is fake_model
    assert fake_model.float_value.dtype == mx.float32
    assert fake_model.int_value.dtype == mx.uint32
    assert meta["draft_load_dtype"] == "float32"
    assert meta["draft_load_dtype_source"] is None


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


def test_dflash2_drafts_keep_fp16_with_fp32_quant_metadata_on_m1():
    """The naive fp16 cast (which also downcasts quantization scales/biases)
    makes DFlash2 block forwards emit all-NaN logits on M1/M2 (0% acceptance).
    DFlash2 therefore loads with scales/biases pinned to fp32 while the rest
    of the floating weights keep the fp16 fast path; DFlash (1) drafts are
    unaffected and keep the plain fp16 cast."""
    quant = runtime_loading.parse_draft_quant_spec("w4")

    # Both draft generations on M1/M2 select fp16 as the load dtype; the
    # DFlash2 difference is applied at cast time in load_draft_bundle.
    for architecture in ("applegpu_g13s", "applegpu_g14d"):
        for is_dflash2 in (True, False):
            assert (
                runtime_loading.resolve_draft_load_dtype(
                    quant,
                    chip_profile=_profile(architecture),
                    is_dflash2=is_dflash2,
                )
                == mx.float16
            )

    # bf16-native chips (M3+): no cast at all.
    assert (
        runtime_loading.resolve_draft_load_dtype(
            quant,
            chip_profile=_profile("applegpu_g15s"),
            is_dflash2=True,
        )
        is None
    )


def test_is_dflash2_config_detects_architecture():
    assert runtime_loading.is_dflash2_config(
        {"architectures": ["DFlash2DraftModel"]}
    )
    assert not runtime_loading.is_dflash2_config(
        {"architectures": ["DFlashDraftModel"]}
    )
    assert not runtime_loading.is_dflash2_config({})


def test_load_draft_bundle_pins_dflash2_scales_to_fp32_on_m1(tmp_path, monkeypatch):
    import mlx.nn as nn

    class FakeDraftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128, bias=False)
            self.norm = nn.RMSNorm(128)
            self.is_dflash2 = True

    fake_model = FakeDraftModel()
    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (
            fake_model,
            {"architectures": ["DFlash2DraftModel"]},
        ),
    )
    monkeypatch.setattr(
        runtime_loading,
        "detect_chip",
        lambda: _profile("applegpu_g13s"),
    )

    _model, meta = runtime_loading.load_draft_bundle(tmp_path, draft_quant="w4")

    assert meta["draft_load_dtype"] == "float16"
    assert meta["draft_load_dtype_source"] == "old_apple_bf16_emulation"
    # Weights take the fp16 fast path...
    assert fake_model.norm.weight.dtype == mx.float16
    # ...but quantization metadata stays fp32 (the NaN root cause).
    assert fake_model.linear.scales.dtype == mx.float32
    assert fake_model.linear.biases.dtype == mx.float32
