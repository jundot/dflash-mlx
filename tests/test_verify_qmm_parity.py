# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import dflash_mlx.verify_qmm as verify_qmm
from dflash_mlx.verify_qmm import is_enabled, verify_matmul

GROUP_SIZE = 64
BITS = 4

REAL_MLP_SHAPES = [

    ("gate_proj", 16, 5120, 17408),
    ("up_proj",   16, 5120, 17408),
    ("down_proj", 16, 17408, 5120),
]

REAL_MLP_M4_SHAPES = [
    ("gate_proj_m4", 4, 5120, 17408),
    ("up_proj_m4", 4, 5120, 17408),
    ("down_proj_m4", 4, 17408, 5120),
]

REAL_MLP_M8_SHAPES = [
    ("gate_proj_m8", 8, 5120, 17408),
    ("up_proj_m8", 8, 5120, 17408),
    ("down_proj_m8", 8, 17408, 5120),
]


@pytest.mark.parametrize(
    ("K", "N", "expected"),
    [
        (5120, 17408, ("scalar", 4, 1)),
        (17408, 5120, ("scalar", 4, 2)),
        (6144, 5120, ("scalar", 4, 1)),
        (5120, 2048, ("scalar", 4, 2)),
        (5120, 1024, ("scalar", 2, 1)),
        (5120, 48, ("scalar", 1, 8)),
        (5120, 10, None),
    ],
)
def test_m8_tuned_config_selects_profiled_m1_shapes(K, N, expected, monkeypatch):
    monkeypatch.setattr(
        verify_qmm.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g13s"},
    )
    assert verify_qmm._m8_tuned_config(K, N, BITS) == expected


def test_m8_tuned_config_keeps_newer_apple_gpus_on_stock(monkeypatch):
    monkeypatch.setattr(
        verify_qmm.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g15s"},
    )
    assert verify_qmm._m8_tuned_config(5120, 17408, BITS) is None


@pytest.mark.parametrize(
    ("K", "N", "expected"),
    [
        (5120, 17408, ("matrix_fp16", 32, 8)),
        (17408, 5120, ("matrix_fp16", 32, 2)),
        (5120, 4096, ("matrix_fp16", 16, 8)),
        (4096, 5120, ("matrix_fp16", 16, 16)),
        (5120, 1024, ("matrix_fp16", 32, 8)),
        (5120, 1280, ("scalar", 4, 1)),
        (25600, 5120, ("matrix_fp16", 32, 2)),
        (5120, 248320, ("matrix_fp16", 32, 8)),
    ],
)
def test_m8_w4a32_config_selects_profiled_dflash2_shapes(
    K, N, expected, monkeypatch
):
    monkeypatch.setattr(
        verify_qmm.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g13s"},
    )
    assert verify_qmm._m8_w4a32_config(K, N, BITS) == expected


@pytest.mark.parametrize("n_tile", [16, 32])
def test_m8_w4a32_matrix_is_finite_and_close_to_stock(n_tile):
    if verify_qmm._m8_w4a32_config(5120, 17408, BITS) is None:
        pytest.skip("W4A32 M=8 kernel is only enabled on M1/M2 GPUs")
    M, K, N = 8, 512, 1024
    rng = np.random.default_rng(0xA32)
    scale = 1.0 / np.sqrt(K)
    x = mx.array((rng.standard_normal((M, K)) * scale).astype(np.float32))
    w_fp = mx.array(
        (rng.standard_normal((N, K)) * scale).astype(np.float32)
    )
    w_q, scales, biases = mx.quantize(
        w_fp, group_size=GROUP_SIZE, bits=BITS
    )
    kernel = verify_qmm._build_kernel_m8_ksplit_fp16(
        GROUP_SIZE,
        mx.float32,
        k_parts=8,
        n_tile=n_tile,
        k_value=K,
        n_value=N,
    )
    (y_kernel,) = kernel(
        inputs=[x, w_q, scales, biases, K, N],
        template=[("T", mx.float32)],
        grid=(32 * 8, N // n_tile, 1),
        threadgroup=(32 * 8, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
    )
    y_stock = mx.quantized_matmul(
        x,
        w_q,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
    )
    mx.eval(y_kernel, y_stock)
    deviation = float(mx.max(mx.abs(y_kernel - y_stock)).item())
    reference_max = float(mx.max(mx.abs(y_stock)).item())
    assert mx.all(mx.isfinite(y_kernel)).item()
    assert deviation <= 2e-3
    assert deviation / (reference_max + 1e-3) <= 5e-3


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (("scalar", 4, 1), "kmajor"),
        (("scalar", 2, 1), "tiled"),
        (("matrix_fp16", 16, 16), "matrix"),
    ],
)
def test_m8_tuned_kernel_dispatches_profiled_morphology(
    config, expected, monkeypatch
):
    calls = []

    def builder(name):
        def build(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return build

    monkeypatch.setattr(
        verify_qmm, "_build_kernel_m8_scalar_kmajor", builder("kmajor")
    )
    monkeypatch.setattr(
        verify_qmm, "_build_kernel_m8_scalar_tiled", builder("tiled")
    )
    monkeypatch.setattr(
        verify_qmm, "_build_kernel_m8_ksplit_fp16", builder("matrix")
    )

    kernel = verify_qmm._build_kernel_m8_tuned(
        GROUP_SIZE, mx.bfloat16, 5120, 2048, config
    )
    assert kernel == expected
    assert calls[0][0] == expected
    assert calls[0][2]["k_value"] == 5120
    assert calls[0][2]["n_value"] == 2048


def _quantize_ref(w_fp, gs, bits):
    return mx.quantize(w_fp, group_size=gs, bits=bits)

def _gen_random_shapes(n=30, seed=0xDF1A):
    rng = np.random.default_rng(seed)
    shapes = []
    M_levels = [1, 4, 8, 12, 15, 16]
    for i in range(n):
        M = M_levels[i % len(M_levels)]

        K = int(rng.integers(2, 40)) * GROUP_SIZE
        N = int(rng.integers(1, 64)) * 8
        shapes.append((f"rnd{i:02d}", M, K, N))
    return shapes

def _run_case(name, M, K, N, dtype, gs):
    rng = np.random.default_rng(hash((name, M, K, N, gs)) & 0xFFFF)
    scale = 1.0 / np.sqrt(max(K, 1))
    x_np = (rng.standard_normal((M, K)) * scale).astype(np.float32)
    w_np = (rng.standard_normal((N, K)) * scale).astype(np.float32)

    x = mx.array(x_np).astype(dtype)
    w_fp = mx.array(w_np).astype(dtype)
    w_q, scales, biases = _quantize_ref(w_fp, gs, BITS)

    y_ref = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases,
        transpose=True, group_size=gs, bits=BITS,
    )
    mx.eval(y_ref)

    y_verify = verify_matmul(
        x, w_q, scales, biases,
        transpose=True, group_size=gs, bits=BITS,
    )
    mx.eval(y_verify)

    y_ref_np = np.array(y_ref.astype(mx.float32))
    y_verify_np = np.array(y_verify.astype(mx.float32))

    max_abs = float(np.max(np.abs(y_verify_np - y_ref_np)))
    max_rel = float(max_abs / (np.max(np.abs(y_ref_np)) + 1e-3))

    return max_abs, max_rel, y_ref_np.shape

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("name,M,K,N", REAL_MLP_SHAPES)
def test_mlp_real_shapes(name, M, K, N, dtype):
    abs_tol = 8e-3 if dtype == mx.bfloat16 else 4e-3
    rel_tol = 2e-2
    max_abs, max_rel, shape = _run_case(name, M, K, N, dtype, GROUP_SIZE)
    assert max_abs <= abs_tol, f"{name}[{dtype}] max_abs={max_abs:.4g} > {abs_tol}"
    assert max_rel <= rel_tol, f"{name}[{dtype}] max_rel={max_rel:.4g} > {rel_tol}"

def test_m16_auto_prefers_nax_on_apple_g17(monkeypatch):
    monkeypatch.setattr(
        verify_qmm.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g17s"},
    )
    monkeypatch.setattr(
        verify_qmm.platform,
        "mac_ver",
        lambda: ("26.4.0", ("", "", ""), ""),
    )

    assert verify_qmm._resolve_m16_ktmpl_variant(5120, 17408, 4) == "nax_ktmpl"

def test_m16_auto_keeps_steel_before_apple_g17(monkeypatch):
    monkeypatch.setattr(
        verify_qmm.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g16s"},
    )
    monkeypatch.setattr(
        verify_qmm.platform,
        "mac_ver",
        lambda: ("26.4.0", ("", "", ""), ""),
    )

    assert (
        verify_qmm._resolve_m16_ktmpl_variant(5120, 17408, 4)
        == "super_tree_fp16_ktmpl"
    )

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_m16_nax_variant_matches_stock_when_forced(dtype, monkeypatch):
    if not str(mx.device_info().get("architecture", "")).startswith("applegpu_g17"):
        pytest.skip("Metal 4 NAX verify path is only enabled on applegpu_g17")
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    monkeypatch.setenv("DFLASH_VERIFY_VARIANT", "nax_ktmpl")
    abs_tol = 8e-3 if dtype == mx.bfloat16 else 4e-3
    rel_tol = 2e-2
    max_abs, max_rel, shape = _run_case(
        "forced_nax_ktmpl", 16, 512, 1024, dtype, GROUP_SIZE
    )
    assert shape == (16, 1024)
    assert max_abs <= abs_tol, f"nax[{dtype}] max_abs={max_abs:.4g} > {abs_tol}"
    assert max_rel <= rel_tol, f"nax[{dtype}] max_rel={max_rel:.4g} > {rel_tol}"

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("name,M,K,N", REAL_MLP_M4_SHAPES)
def test_mlp_real_shapes_m4_ksplit_np(name, M, K, N, dtype, monkeypatch):
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    abs_tol = 8e-3 if dtype == mx.bfloat16 else 4e-3
    rel_tol = 2e-2
    max_abs, max_rel, shape = _run_case(name, M, K, N, dtype, GROUP_SIZE)
    assert shape == (M, N)
    assert max_abs <= abs_tol, f"{name}[{dtype}] max_abs={max_abs:.4g} > {abs_tol}"
    assert max_rel <= rel_tol, f"{name}[{dtype}] max_rel={max_rel:.4g} > {rel_tol}"

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("name,M,K,N", REAL_MLP_M8_SHAPES)
def test_mlp_real_shapes_m8_ksplit_fp16(name, M, K, N, dtype, monkeypatch):
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    abs_tol = 2e-2 if dtype == mx.bfloat16 else 1e-2
    rel_tol = 3e-2
    max_abs, max_rel, shape = _run_case(name, M, K, N, dtype, GROUP_SIZE)
    assert shape == (M, N)
    assert max_abs <= abs_tol, f"{name}[{dtype}] max_abs={max_abs:.4g} > {abs_tol}"
    assert max_rel <= rel_tol, f"{name}[{dtype}] max_rel={max_rel:.4g} > {rel_tol}"

@pytest.mark.parametrize("name,M,K,N", _gen_random_shapes(30))
def test_random_shapes(name, M, K, N):
    abs_tol = 8e-3
    rel_tol = 2e-2
    max_abs, max_rel, _ = _run_case(name, M, K, N, mx.bfloat16, GROUP_SIZE)
    assert max_abs <= abs_tol, f"{name} M={M} K={K} N={N} abs={max_abs:.4g}"
    assert max_rel <= rel_tol, f"{name} M={M} K={K} N={N} rel={max_rel:.4g}"

def test_stub_mode_reports_identity_when_disabled():
    if is_enabled():
        pytest.skip("DFLASH_VERIFY_QMM=1, skip stub sanity")
    max_abs, max_rel, _ = _run_case("stub_sanity", 16, 5120, 8, mx.bfloat16, GROUP_SIZE)
    assert max_abs == 0.0, "Stub path must return exact stock output, got delta"
