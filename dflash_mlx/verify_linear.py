# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from dflash_mlx.internal_debug import (
    verify_include as _debug_verify_include,
    verify_max_n as _debug_verify_max_n,
    verify_qmm_enabled as _debug_verify_qmm_enabled,
)
from dflash_mlx.verify_qmm import (
    verify_matmul,
    _auto_variant,
    _build_kernel_m16_combo_ktmpl,
    _build_kernel_m16_nax_ktmpl,
    _build_kernel_m16_super_tree_fp16_ktmpl,
    _build_kernel_mma2big,
    _build_kernel_mma2big_8bit,
    _build_kernel_m4_ksplit_np,
    _build_kernel_m8_tuned,
    _build_kernel_mma2big_pipe,
    _build_kernel_mma2big_pipe_8bit,
    _m4_ksplit_np_kparts,
    _m4_ksplit_np_shape,
    _m8_tuned_config,
    _m8_w4a32_config,
    _resolve_m16_ktmpl_variant,
)

_VERIFY_MAX_N_DEFAULT = 100_000

_PROJ_TAGS = {
    "mlp.gate_proj":        "mlp_gate",
    "mlp.up_proj":          "mlp_up",
    "mlp.down_proj":        "mlp_down",
    "self_attn.q_proj":     "attn_q",
    "self_attn.k_proj":     "attn_k",
    "self_attn.v_proj":     "attn_v",
    "self_attn.o_proj":     "attn_o",
    "linear_attn.in_proj_qkv": "gdn_qkv",
    "linear_attn.in_proj_z":   "gdn_z",
    "linear_attn.out_proj":    "gdn_o",
}

def _path_tag(path: str) -> str:
    for suffix, tag in _PROJ_TAGS.items():
        if path.endswith(suffix):
            return tag
    return "other"

def is_verify_eligible(ql: nn.QuantizedLinear, path: str = "") -> bool:
    if not isinstance(ql, nn.QuantizedLinear):
        return False
    if getattr(ql, "bits", None) not in (4, 8):
        return False
    if getattr(ql, "group_size", None) not in (32, 64, 128):
        return False
    if getattr(ql, "mode", "affine") != "affine":
        return False
    w = ql.weight
    N = int(w.shape[0])
    K = int(w.shape[1]) * (32 // ql.bits)
    if K % 32 != 0:
        return False
    if ql.bits == 4:
        if N % 4 != 0:
            return False
    elif N % 32 != 0:
        return False
    if N >= _debug_verify_max_n(_VERIFY_MAX_N_DEFAULT):
        return False
    include = _debug_verify_include()
    if include not in ("", "all"):
        tag = _path_tag(path)
        allowed = {s.strip() for s in include.split(",") if s.strip()}
        if "mlp" in allowed:
            allowed.update({"mlp_gate", "mlp_up", "mlp_down"})
        if "attn" in allowed:
            allowed.update({"attn_q", "attn_k", "attn_v", "attn_o"})
        if "gdn" in allowed:
            allowed.update({"gdn_qkv", "gdn_z", "gdn_o"})
        if tag not in allowed:
            return False
    return True

class VerifyQuantizedLinear(nn.QuantizedLinear):

    @classmethod
    def from_quantized(
        cls,
        ql: nn.QuantizedLinear,
        *,
        enable_qmm: Optional[bool] = None,
        w4a32_only: bool = False,
        enable_bf16_m8: bool = False,
    ) -> "VerifyQuantizedLinear":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.group_size = ql.group_size
        obj.bits = ql.bits
        obj.mode = getattr(ql, "mode", "affine")
        obj.weight = ql.weight
        obj.scales = ql.scales
        if getattr(ql, "biases", None) is not None:
            obj.biases = ql.biases
        if "bias" in ql:
            obj.bias = ql.bias

        object.__setattr__(
            obj,
            "_call_fn",
            _build_dispatch(
                obj,
                enable_qmm=enable_qmm,
                w4a32_only=w4a32_only,
                enable_bf16_m8=enable_bf16_m8,
            ),
        )
        object.__setattr__(obj, "_w4a32_only", bool(w4a32_only))

        obj.freeze()
        return obj

    def __call__(self, x: mx.array) -> mx.array:
        return self._call_fn(x)

def _build_dispatch(
    obj: "VerifyQuantizedLinear",
    *,
    enable_qmm: Optional[bool] = None,
    w4a32_only: bool = False,
    enable_bf16_m8: bool = False,
):
    w = obj.weight
    s = obj.scales
    b = getattr(obj, "biases", None)
    gs = obj.group_size
    bits = obj.bits
    mode = obj.mode
    has_bias = "bias" in obj
    bias = obj.bias if has_bias else None

    N = int(w.shape[0])
    K = int(w.shape[1]) * (32 // bits)

    qmm_enabled = (
        _debug_verify_qmm_enabled()
        if enable_qmm is None
        else bool(enable_qmm)
    )

    ktmpl_variant = _resolve_m16_ktmpl_variant(K, N, bits) if qmm_enabled else None
    m16_qmm = qmm_enabled and not w4a32_only and (
        ktmpl_variant is not None
        or (N % 32 == 0 and K % 32 == 0)
    )
    m4_qmm = qmm_enabled and not w4a32_only and _m4_ksplit_np_shape(K, N, bits)
    m8_config = (
        _m8_tuned_config(K, N, bits)
        if qmm_enabled and (not w4a32_only or enable_bf16_m8)
        else None
    )
    m8_qmm = m8_config is not None
    m8_w4a32_config = (
        _m8_w4a32_config(K, N, bits)
        if qmm_enabled and w4a32_only
        else None
    )
    m8_w4a32_qmm = m8_w4a32_config is not None

    if m16_qmm or m8_qmm or m8_w4a32_qmm or m4_qmm:
        w_c = mx.contiguous(w)
        s_c = mx.contiguous(s)
        b_c = mx.contiguous(b) if b is not None else b
        mx.eval(w_c, s_c)
        if b_c is not None:
            mx.eval(b_c)

        if m16_qmm:
            if ktmpl_variant is not None:
                if ktmpl_variant == "nax_ktmpl":
                    _kern_fn = _build_kernel_m16_nax_ktmpl
                    _grid = (256, N // 32, 1)
                else:
                    _kern_fn = (
                        _build_kernel_m16_combo_ktmpl
                        if ktmpl_variant == "combo_ktmpl" else
                        _build_kernel_m16_super_tree_fp16_ktmpl
                    )
                    _grid = (256, N // 16, 1)
                kern_bf16 = _kern_fn(K, gs, mx.bfloat16)
                kern_fp16 = _kern_fn(K, gs, mx.float16)

                def _run(x2: mx.array, kern) -> mx.array:
                    (y,) = kern(
                        inputs=[x2, w_c, s_c, b_c, N],
                        template=[("T", x2.dtype), ("KCONST", K)],
                        grid=_grid,
                        threadgroup=(256, 1, 1),
                        output_shapes=[(16, N)],
                        output_dtypes=[x2.dtype],
                    )
                    return y
            else:
                variant, auto_kp = _auto_variant(K, N)
                K_PARTS = auto_kp
                if variant == "mma2big_pipe" and K % (32 * K_PARTS) != 0:
                    variant = "mma2big"
                    K_PARTS = 1
                _kern_fn = (
                    (_build_kernel_mma2big_pipe_8bit if bits == 8 else _build_kernel_mma2big_pipe)
                    if variant == "mma2big_pipe" else
                    (_build_kernel_mma2big_8bit if bits == 8 else _build_kernel_mma2big)
                )
                kern_bf16 = _kern_fn(gs, mx.bfloat16)
                kern_fp16 = _kern_fn(gs, mx.float16)

                if variant == "mma2big_pipe":
                    def _run(x2: mx.array, kern) -> mx.array:
                        (partials,) = kern(
                            inputs=[x2, w_c, s_c, b_c, 16, K, N, K_PARTS],
                            template=[("T", x2.dtype)],
                            grid=(64, N // 32, K_PARTS),
                            threadgroup=(64, 1, 1),
                            output_shapes=[(K_PARTS, 16, N)],
                            output_dtypes=[mx.float32],
                        )
                        return partials.sum(axis=0).astype(x2.dtype)
                else:
                    def _run(x2: mx.array, kern) -> mx.array:
                        (y,) = kern(
                            inputs=[x2, w_c, s_c, b_c, 16, K, N],
                            template=[("T", x2.dtype)],
                            grid=(64, N // 32, 1),
                            threadgroup=(64, 1, 1),
                            output_shapes=[(16, N)],
                            output_dtypes=[x2.dtype],
                        )
                        return y

        m4_k_parts = _m4_ksplit_np_kparts(N)
        kern_m4_bf16 = (
            _build_kernel_m4_ksplit_np(gs, mx.bfloat16, k_parts=m4_k_parts)
            if m4_qmm else None
        )
        kern_m4_fp16 = (
            _build_kernel_m4_ksplit_np(gs, mx.float16, k_parts=m4_k_parts)
            if m4_qmm else None
        )

        def _run_m4(x2: mx.array, kern) -> mx.array:
            (y,) = kern(
                inputs=[x2, w_c, s_c, b_c, K, N],
                template=[("T", x2.dtype)],
                grid=(32 * m4_k_parts, N // 4, 1),
                threadgroup=(32 * m4_k_parts, 1, 1),
                output_shapes=[(4, N)],
                output_dtypes=[x2.dtype],
            )
            return y

        _m8_variant, m8_n_tile, m8_k_parts = (
            m8_config if m8_config is not None else ("none", 1, 1)
        )

        kern_m8_bf16 = (
            _build_kernel_m8_tuned(gs, mx.bfloat16, K, N, m8_config)
            if m8_qmm else None
        )
        kern_m8_fp16 = (
            _build_kernel_m8_tuned(gs, mx.float16, K, N, m8_config)
            if m8_qmm and not w4a32_only else None
        )
        _m8_w4a32_variant, m8_w4a32_n_tile, m8_w4a32_k_parts = (
            m8_w4a32_config
            if m8_w4a32_config is not None
            else ("none", 1, 1)
        )
        kern_m8_fp32 = (
            _build_kernel_m8_tuned(gs, mx.float32, K, N, m8_w4a32_config)
            if m8_w4a32_qmm else None
        )

        def _run_m8(x2: mx.array, kern) -> mx.array:
            (y,) = kern(
                inputs=[x2, w_c, s_c, b_c, K, N],
                template=[("T", x2.dtype)],
                grid=(32 * m8_k_parts, N // m8_n_tile, 1),
                threadgroup=(32 * m8_k_parts, 1, 1),
                output_shapes=[(8, N)],
                output_dtypes=[x2.dtype],
            )
            return y

        def _run_m8_fp32(x2: mx.array) -> mx.array:
            (y,) = kern_m8_fp32(
                inputs=[x2, w_c, s_c, b_c, K, N],
                template=[("T", x2.dtype)],
                grid=(32 * m8_w4a32_k_parts, N // m8_w4a32_n_tile, 1),
                threadgroup=(32 * m8_w4a32_k_parts, 1, 1),
                output_shapes=[(8, N)],
                output_dtypes=[mx.float32],
            )
            return y

        def _run_m8_fp32_rows(x2: mx.array, rows: int) -> mx.array:
            if rows < 8:
                x2 = mx.concatenate(
                    [x2, mx.zeros((8 - rows, K), dtype=mx.float32)],
                    axis=0,
                )
            return _run_m8_fp32(x2)[:rows]

        if has_bias:
            def call(x: mx.array) -> mx.array:
                orig = x.shape
                m = 1
                for d in orig[:-1]:
                    m *= d
                if m == 16 and m16_qmm:
                    x2 = mx.contiguous(x.reshape(16, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        y = _run(x2, kern_bf16)
                    elif dtype == mx.float16:
                        y = _run(x2, kern_fp16)
                    else:
                        y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                                transpose=True, group_size=gs, bits=bits, mode=mode)
                    return y.reshape(*orig[:-1], N) + bias
                if m == 4 and m4_qmm:
                    x2 = mx.contiguous(x.reshape(4, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        y = _run_m4(x2, kern_m4_bf16)
                    elif dtype == mx.float16:
                        y = _run_m4(x2, kern_m4_fp16)
                    else:
                        y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                                transpose=True, group_size=gs, bits=bits, mode=mode)
                    return y.reshape(*orig[:-1], N) + bias
                if m == 8 and m8_qmm:
                    x2 = mx.contiguous(x.reshape(8, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        y = _run_m8(x2, kern_m8_bf16)
                    elif dtype == mx.float16 and not w4a32_only:
                        y = _run_m8(x2, kern_m8_fp16)
                    elif dtype == mx.float32 and m8_w4a32_qmm:
                        y = _run_m8_fp32(x2)
                    else:
                        y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                                transpose=True, group_size=gs, bits=bits, mode=mode)
                    return y.reshape(*orig[:-1], N) + bias
                if m == 7 and m8_w4a32_qmm and x.dtype == mx.float32:
                    x2 = mx.contiguous(x.reshape(7, orig[-1]))
                    y = _run_m8_fp32_rows(x2, 7)
                    return y.reshape(*orig[:-1], N) + bias
                y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                        transpose=True, group_size=gs, bits=bits, mode=mode)
                return y + bias
        else:
            def call(x: mx.array) -> mx.array:
                orig = x.shape
                m = 1
                for d in orig[:-1]:
                    m *= d
                if m == 16 and m16_qmm:
                    x2 = mx.contiguous(x.reshape(16, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        return _run(x2, kern_bf16).reshape(*orig[:-1], N)
                    if dtype == mx.float16:
                        return _run(x2, kern_fp16).reshape(*orig[:-1], N)
                if m == 4 and m4_qmm:
                    x2 = mx.contiguous(x.reshape(4, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        return _run_m4(x2, kern_m4_bf16).reshape(*orig[:-1], N)
                    if dtype == mx.float16:
                        return _run_m4(x2, kern_m4_fp16).reshape(*orig[:-1], N)
                if m == 8 and m8_qmm:
                    x2 = mx.contiguous(x.reshape(8, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        return _run_m8(x2, kern_m8_bf16).reshape(*orig[:-1], N)
                    if dtype == mx.float16 and not w4a32_only:
                        return _run_m8(x2, kern_m8_fp16).reshape(*orig[:-1], N)
                    if dtype == mx.float32 and m8_w4a32_qmm:
                        return _run_m8_fp32(x2).reshape(*orig[:-1], N)
                if m == 7 and m8_w4a32_qmm and x.dtype == mx.float32:
                    x2 = mx.contiguous(x.reshape(7, orig[-1]))
                    return _run_m8_fp32_rows(x2, 7).reshape(*orig[:-1], N)
                return mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                           transpose=True, group_size=gs, bits=bits, mode=mode)
        return call

    if has_bias:
        def call(x):
            m = 1
            for d in x.shape[:-1]:
                m *= d
            if m == 16:
                y = verify_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
            else:
                y = mx.quantized_matmul(x, w, scales=s, biases=b,
                                        transpose=True, group_size=gs, bits=bits, mode=mode)
            return y + bias
    else:
        def call(x):
            m = 1
            for d in x.shape[:-1]:
                m *= d
            if m == 16:
                return verify_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
            return mx.quantized_matmul(x, w, scales=s, biases=b,
                                       transpose=True, group_size=gs, bits=bits, mode=mode)
    return call

def prewarm_verify_kernels(
    model: nn.Module,
    *,
    input_dtype=mx.bfloat16,
) -> int:
    from mlx.utils import tree_flatten

    seen: set[tuple] = set()
    warmed = 0
    for _, m in tree_flatten(model.leaf_modules()):
        if not isinstance(m, VerifyQuantizedLinear):
            continue
        K = int(m.weight.shape[1]) * (32 // m.bits)
        N = int(m.weight.shape[0])
        key = (K, N, m.bits, m.group_size)
        if key in seen:
            continue
        seen.add(key)
        if bool(getattr(m, "_w4a32_only", False)):
            dummy_m8 = mx.zeros((1, 8, K), dtype=mx.float32)
            mx.eval(m(dummy_m8))
            warmed += 1
            continue
        dummy = mx.zeros((1, 16, K), dtype=input_dtype)
        mx.eval(m(dummy))
        warmed += 1
        if _m4_ksplit_np_shape(K, N, m.bits):
            dummy_m4 = mx.zeros((1, 4, K), dtype=input_dtype)
            mx.eval(m(dummy_m4))
            warmed += 1
        if _m8_tuned_config(K, N, m.bits) is not None:
            dummy_m8 = mx.zeros((1, 8, K), dtype=input_dtype)
            mx.eval(m(dummy_m8))
            warmed += 1
    return warmed

def install_verify_linears(
    model: nn.Module,
    *,
    predicate: Optional[Callable[[str, nn.QuantizedLinear], bool]] = None,
    enable_qmm: Optional[bool] = None,
    w4a32_only: bool = False,
) -> int:
    if predicate is None:
        predicate = lambda path, m: is_verify_eligible(m, path=path)

    count = 0

    def _maybe_swap(path, m):
        nonlocal count
        if isinstance(m, VerifyQuantizedLinear):
            return m
        if isinstance(m, nn.QuantizedLinear) and predicate(path, m):
            count += 1
            return VerifyQuantizedLinear.from_quantized(
                m,
                enable_qmm=enable_qmm,
                w4a32_only=w4a32_only,
            )
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_swap, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return count


def install_w4a32_draft_logits_linear(
    target_model,
    *,
    target_ops,
    enable_qmm: bool,
) -> int:
    """Install the DFlash2-specific M=7/M=8 vocabulary projections.

    The wrapper accelerates the FP32 M=7 draft projection and BF16 M=8 target
    verification. Every other shape and dtype stays on stock MLX QMM, keeping
    ordinary target decoding unchanged. The one missing draft row is
    zero-padded inside the wrapper to reuse the profiled M=8 matrix kernel.
    """
    if not enable_qmm:
        return 0
    try:
        wrapper = target_ops.text_wrapper(target_model)
    except (AttributeError, TypeError):
        return 0
    if bool(getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True)):
        return 0
    linear = getattr(wrapper, "lm_head", None)
    if not isinstance(linear, nn.QuantizedLinear):
        return 0
    if isinstance(linear, VerifyQuantizedLinear):
        return 0
    if (
        getattr(linear, "bits", None) != 4
        or getattr(linear, "group_size", None) not in (32, 64, 128)
        or getattr(linear, "mode", "affine") != "affine"
    ):
        return 0
    K = int(linear.weight.shape[1]) * 8
    N = int(linear.weight.shape[0])
    if _m8_w4a32_config(K, N, 4) is None:
        return 0

    replacement = VerifyQuantizedLinear.from_quantized(
        linear,
        enable_qmm=True,
        w4a32_only=True,
        enable_bf16_m8=True,
    )
    wrapper.lm_head = replacement
    draft_dummy = mx.zeros((1, 7, K), dtype=mx.float32)
    target_dummy = mx.zeros((1, 8, K), dtype=mx.bfloat16)
    mx.eval(replacement(draft_dummy), replacement(target_dummy))
    return 1
