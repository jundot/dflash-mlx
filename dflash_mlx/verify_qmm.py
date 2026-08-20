# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import platform

import mlx.core as mx

from dflash_mlx.internal_debug import (
    verify_qmm_enabled as _debug_verify_qmm_enabled,
    verify_qmm_kparts as _debug_verify_qmm_kparts,
    verify_qmm_variant as _debug_verify_qmm_variant,
)

def is_enabled() -> bool:
    return _debug_verify_qmm_enabled()

def _variant() -> str:
    return _debug_verify_qmm_variant()

def _auto_variant(K: int, N: int) -> tuple[str, int]:
    if K >= 8192 or N <= 8192:
        return ("mma2big_pipe", 8)
    return ("mma2big", 1)

_VERIFY_KERNEL_CACHE: dict[tuple, object] = {}

def _m4_ksplit_np_shape(K: int, N: int, bits: int) -> bool:
    return int(bits) == 4 and int(N) % 4 == 0 and int(K) % 32 == 0

def _m4_ksplit_np_kparts(N: int) -> int:
    return 2 if int(N) >= 4096 else 4

def _m8_tuned_available() -> bool:
    arch = str(mx.device_info().get("architecture", "")).lower()
    return arch.startswith(("applegpu_g13", "applegpu_g14"))


def _m8_tuned_config(K: int, N: int, bits: int) -> tuple[str, int, int] | None:
    """Select the M1/M2 M=8 kernel morphology for a quantized projection.

    The tuple is ``(variant, n_tile, k_parts)``. Qwen 3.5/3.8 target-verifier
    shapes favor a scalar register tile on M1/M2; the 6144-wide output
    projection is fastest without K splitting.
    """
    K = int(K)
    N = int(N)
    if (
        not _m8_tuned_available()
        or int(bits) != 4
        or K % 32 != 0
        or N % 4 != 0
    ):
        return None
    if (K, N) == (6144, 5120):
        return ("scalar", 4, 1)
    if N < 256:
        desired_k_parts = 8
        n_tile = 1
    elif N < 2048:
        desired_k_parts = 1
        n_tile = 2
    else:
        desired_k_parts = 2 if K >= 8192 or N < 12288 else 1
        n_tile = 4
    while desired_k_parts > 1 and K % (32 * desired_k_parts) != 0:
        desired_k_parts //= 2
    return ("scalar", n_tile, desired_k_parts)


def _m8_w4a32_config(K: int, N: int, bits: int) -> tuple[str, int, int] | None:
    """Select the profiled M=8 W4A32 kernel on M1/M2 GPUs.

    DFlash2 uses FP32 activations on these GPUs because BF16 is emulated. Its
    fixed projection shapes have a different optimum from target verification:
    large projections benefit from FP16 matrix operands with FP32 accumulation,
    while the dynamic-convolution projections remain faster in the exact scalar
    kernel. The matrix path only affects draft predictions; target verification
    still determines the emitted tokens.
    """
    K = int(K)
    N = int(N)
    if (
        not _m8_tuned_available()
        or int(bits) != 4
        or K % 32 != 0
        or N % 4 != 0
    ):
        return None

    matrix_config = {
        (5120, 17408): (32, 8),
        (17408, 5120): (32, 2),
        (5120, 4096): (16, 8),
        (4096, 5120): (16, 16),
        (5120, 1024): (32, 8),
        (25600, 5120): (32, 2),
        (5120, 248320): (32, 8),
    }.get((K, N))
    if matrix_config is not None:
        n_tile, k_parts = matrix_config
        return ("matrix_fp16", n_tile, k_parts)
    if (K, N) == (5120, 1280):
        return ("scalar", 4, 1)
    return _m8_tuned_config(K, N, bits)

def _m16_ktmpl_variant(K: int, N: int, bits: int) -> str | None:
    if int(bits) != 4:
        return None
    if int(K) % 256 != 0 or int(N) % 16 != 0:
        return None
    if int(N) % 32 == 0 and _nax_verify_available():
        return "nax_ktmpl"
    if int(K) >= 8192 or int(N) <= 5120:
        return "combo_ktmpl"
    return "super_tree_fp16_ktmpl"

def _resolve_m16_ktmpl_variant(K: int, N: int, bits: int, variant: str | None = None) -> str | None:
    if int(bits) != 4 or int(K) % 256 != 0 or int(N) % 16 != 0:
        return None
    selected = _variant() if variant is None else str(variant)
    if selected == "auto":
        return _m16_ktmpl_variant(K, N, bits)
    if selected == "nax_ktmpl" and int(N) % 32 == 0 and _nax_verify_available():
        return selected
    if selected in ("combo_ktmpl", "super_tree_fp16_ktmpl"):
        return selected
    return None

def _nax_verify_available() -> bool:
    arch = str(mx.device_info().get("architecture", "")).lower()
    if not arch.startswith("applegpu_g17"):
        return False
    parts = platform.mac_ver()[0].split(".")
    try:
        major = int(parts[0]) if parts and parts[0] else 0
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        minor = 0
    return major > 26 or (major == 26 and minor >= 2)

def _build_kernel_m16_nax_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_nax_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 16;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};
        constexpr int K = KCONST;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float partial[NSG][BM * BN];

        constexpr auto desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<device T, dextents<int, 2>, tensor_inline> A(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
            B_tile[sg_id],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
            partial[sg_id],
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, BN}});

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
        }}

        int n_global = n0 + int(lane);
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            uint32_t p0 = w_q[n_global * K_by_8 + ((k0 + 0) >> 3)];
            uint32_t p1 = w_q[n_global * K_by_8 + ((k0 + 8) >> 3)];
            float s0 = float(scales[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float s1 = float(scales[n_global * K_by_gs + ((k0 + 8) / GS)]);
            float b0 = float(biases[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float b1 = float(biases[n_global * K_by_gs + ((k0 + 8) / GS)]);

            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p0 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][ki * BN + int(lane)] = T(float(nib) * s0 + b0);
            }}
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p1 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float(nib) * s1 + b1);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);

            auto tA = A.template slice<16, 16>(k0, 0);
            auto tB = B.template slice<32, 16>(0, 0);
            op.run(tA, tB, ct_c);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        auto tC = C.template slice<32, 16>(0, 0);
        ct_c.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc01 = partial[0][off] + partial[1][off];
            float acc23 = partial[2][off] + partial[3][off];
            float acc45 = partial[4][off] + partial[5][off];
            float acc67 = partial[6][off] + partial[7][off];
            float acc = (acc01 + acc23) + (acc45 + acc67);
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_nax_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        header="""
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_m4_ksplit_np(
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int = 4,
):
    key = ("m4_ksplit_np", group_size, dtype, int(k_parts))
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 4;
        constexpr int BN = 4;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {group_size};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

        float acc[BN * M];
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
            float s0 = float(scales[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + (k_base / GS)]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + (k_base / GS)]);

            {{
                uint32_t packed = p0;
                float s = s0;
                float b = b0;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[0 * M + 0] += float(v0[ki]) * wv;
                    acc[0 * M + 1] += float(v1[ki]) * wv;
                    acc[0 * M + 2] += float(v2[ki]) * wv;
                    acc[0 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p1;
                float s = s1;
                float b = b1;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[1 * M + 0] += float(v0[ki]) * wv;
                    acc[1 * M + 1] += float(v1[ki]) * wv;
                    acc[1 * M + 2] += float(v2[ki]) * wv;
                    acc[1 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p2;
                float s = s2;
                float b = b2;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[2 * M + 0] += float(v0[ki]) * wv;
                    acc[2 * M + 1] += float(v1[ki]) * wv;
                    acc[2 * M + 2] += float(v2[ki]) * wv;
                    acc[2 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p3;
                float s = s3;
                float b = b3;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[3 * M + 0] += float(v0[ki]) * wv;
                    acc[3 * M + 1] += float(v1[ki]) * wv;
                    acc[3 * M + 2] += float(v2[ki]) * wv;
                    acc[3 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
        }}

        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {{
            for (int i = 0; i < BN * M; ++i) {{
                partial[int(part) * BN * M + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {{
            float total = 0.0f;
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p * BN * M + int(lane)];
            }}
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(total);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m4_ksplit_np_kp{int(k_parts)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m8_scalar_tiled(
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int,
    n_tile: int,
    k_value: int | None = None,
    n_value: int | None = None,
):
    """Build the narrow-output register-tiled M=8 verifier."""
    if n_tile not in (1, 2, 4):
        raise ValueError(f"unsupported scalar M=8 N tile: {n_tile}")
    if 32 * int(k_parts) > 1024:
        raise ValueError("scalar M=8 threadgroup exceeds 1024 threads")

    activation_loads = "\n".join(
        f"Vec8 v{row} = xv[({row} * K + k_base) / 8];"
        for row in range(8)
    )
    weight_steps = []
    for col in range(n_tile):
        fmas = "\n".join(
            f"acc[{col * 8 + row}] += float(v{row}[ki]) * wv;"
            for row in range(8)
        )
        weight_steps.append(
            f"""
            {{
                int n_global = n0 + {col};
                uint32_t packed = w_q[n_global * K_by_8 + pack];
                float scale = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float bias = float(biases[n_global * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * scale + bias;
                    {fmas}
                }}
            }}
            """
        )
    dequant_and_fma = "\n".join(weight_steps)

    key = (
        "m8_scalar_tiled",
        group_size,
        dtype,
        int(k_parts),
        int(n_tile),
        int(k_value) if k_value is not None else None,
        int(n_value) if n_value is not None else None,
    )
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 8;
        constexpr int RM = 8;
        constexpr int BN = {int(n_tile)};
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {group_size};
        constexpr int ACC_COUNT = RM * BN;

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        constexpr bool SPECIALIZED_K = {'true' if k_value is not None else 'false'};
        constexpr bool SPECIALIZED_N = {'true' if n_value is not None else 'false'};
        int K = SPECIALIZED_K ? {int(k_value or 0)} : int(K_size);
        int N = SPECIALIZED_N ? {int(n_value or 0)} : int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = pack_start + packs_per_part;

        float acc[ACC_COUNT];
        _Pragma("unroll")
        for (int i = 0; i < ACC_COUNT; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = reinterpret_cast<const device Vec8 *>(x);

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            {activation_loads}
            {dequant_and_fma}
        }}

        _Pragma("unroll")
        for (int i = 0; i < ACC_COUNT; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS][ACC_COUNT];
        if (lane == 0) {{
            _Pragma("unroll")
            for (int i = 0; i < ACC_COUNT; ++i) {{
                partial[part][i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < ACC_COUNT) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p][lane];
            }}
            int col = int(lane) / RM;
            int row = int(lane) - col * RM;
            y[row * N + n0 + col] = T(total);
        }}
    """

    dtype_tag = {
        mx.bfloat16: "bf16",
        mx.float16: "fp16",
        mx.float32: "fp32",
    }.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=(
            f"verify_m8_scalar_n{int(n_tile)}_kp{int(k_parts)}_"
            f"ks{int(k_value) if k_value is not None else 'dyn'}_"
            f"ns{int(n_value) if n_value is not None else 'dyn'}_"
            f"gs{group_size}_{dtype_tag}"
        ),
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m8_scalar_kmajor(
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int,
    k_value: int | None = None,
    n_value: int | None = None,
):
    """Build the profiled 8x4 verifier for M1/M2-class Apple GPUs.

    Each lane walks a disjoint slice of packed K and keeps an 8x4 FP32
    accumulator tile. Loading activations in K-major order lets all four
    output columns reuse the same BF16-to-FP32 conversions. Production shapes
    specialize K and N so Metal can fold indexing and bounds arithmetic.
    """
    if int(group_size) not in (32, 64, 128):
        raise ValueError(f"unsupported group size: {group_size}")
    if int(k_parts) < 1 or 32 * int(k_parts) > 1024:
        raise ValueError(f"unsupported K partition count: {k_parts}")

    key = (
        "m8_scalar_kmajor",
        int(group_size),
        dtype,
        int(k_parts),
        int(k_value) if k_value is not None else None,
        int(n_value) if n_value is not None else None,
    )
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    activation_conversions = "\n".join(
        f"float a{row} = float(v{row}[ki]);" for row in range(8)
    )
    weight_conversions = "\n".join(
        f"float w{col} = float((packed{col} >> (ki * 4)) & 0xFu) "
        f"* scale{col} + bias{col};"
        for col in range(4)
    )
    fmas = "\n".join(
        f"acc[{col * 8 + row}] += a{row} * w{col};"
        for col in range(4)
        for row in range(8)
    )
    group_shift = {32: 2, 64: 3, 128: 4}[int(group_size)]

    if int(k_parts) == 1:
        reduction_and_store = """
        _Pragma("unroll")
        for (int i = 0; i < ACC_COUNT; ++i) {
            acc[i] = simd_sum(acc[i]);
        }

        int output_idx = int(lane);
        int col = output_idx / M;
        int row = output_idx - col * M;
        y[row * N + n0 + col] = T(acc[output_idx]);
        """
    else:
        reduction_and_store = """
        _Pragma("unroll")
        for (int i = 0; i < ACC_COUNT; ++i) {
            acc[i] = simd_sum(acc[i]);
        }

        threadgroup float partial[K_PARTS][ACC_COUNT];
        if (lane == 0) {
            _Pragma("unroll")
            for (int i = 0; i < ACC_COUNT; ++i) {
                partial[part][i] = acc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0) {
            int output_idx = int(lane);
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {
                total += partial[p][output_idx];
            }
            int col = output_idx / M;
            int row = output_idx - col * M;
            y[row * N + n0 + col] = T(total);
        }
        """

    source = f"""
        using namespace metal;
        constexpr int M = 8;
        constexpr int BN = 4;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {int(group_size)};
        constexpr int ACC_COUNT = M * BN;

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = {'int(K_size)' if k_value is None else int(k_value)};
        int N = {'int(N_size)' if n_value is None else int(n_value)};
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = pack_start + packs_per_part;

        float acc[ACC_COUNT];
        _Pragma("unroll")
        for (int i = 0; i < ACC_COUNT; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = reinterpret_cast<const device Vec8 *>(x);

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            Vec8 v0 = xv[0 * K_by_8 + pack];
            Vec8 v1 = xv[1 * K_by_8 + pack];
            Vec8 v2 = xv[2 * K_by_8 + pack];
            Vec8 v3 = xv[3 * K_by_8 + pack];
            Vec8 v4 = xv[4 * K_by_8 + pack];
            Vec8 v5 = xv[5 * K_by_8 + pack];
            Vec8 v6 = xv[6 * K_by_8 + pack];
            Vec8 v7 = xv[7 * K_by_8 + pack];

            uint32_t packed0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t packed1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t packed2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t packed3 = w_q[(n0 + 3) * K_by_8 + pack];
            int group = pack >> {group_shift};
            float scale0 = float(scales[(n0 + 0) * K_by_gs + group]);
            float scale1 = float(scales[(n0 + 1) * K_by_gs + group]);
            float scale2 = float(scales[(n0 + 2) * K_by_gs + group]);
            float scale3 = float(scales[(n0 + 3) * K_by_gs + group]);
            float bias0 = float(biases[(n0 + 0) * K_by_gs + group]);
            float bias1 = float(biases[(n0 + 1) * K_by_gs + group]);
            float bias2 = float(biases[(n0 + 2) * K_by_gs + group]);
            float bias3 = float(biases[(n0 + 3) * K_by_gs + group]);

            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                {activation_conversions}
                {weight_conversions}
                {fmas}
            }}
        }}

        {reduction_and_store}
    """

    dtype_tag = {
        mx.bfloat16: "bf16",
        mx.float16: "fp16",
        mx.float32: "fp32",
    }.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=(
            f"verify_m8_scalar_kmajor_kp{int(k_parts)}_"
            f"k{int(k_value) if k_value is not None else 'dyn'}_"
            f"n{int(n_value) if n_value is not None else 'dyn'}_"
            f"gs{group_size}_{dtype_tag}"
        ),
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m8_ksplit_fp16(
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int,
    n_tile: int = 16,
    k_value: int | None = None,
    n_value: int | None = None,
):
    """Build the profiled M=8 matrix verifier for M1/M2-class Apple GPUs.

    Each simdgroup owns one K partition and computes an 8x16 or 8x32 output
    tile with native 8x8 matrix accumulators. Activations and dequantized
    weights are staged as FP16, while accumulation and cross-part reduction
    remain FP32. This morphology is selected only for the measured 6144-wide
    output-projection shape.
    """
    if int(k_parts) < 1 or 32 * int(k_parts) > 1024:
        raise ValueError(f"unsupported K partition count: {k_parts}")
    if int(n_tile) not in (16, 32):
        raise ValueError(f"unsupported matrix M=8 N tile: {n_tile}")
    threadgroup_bytes = int(k_parts) * (
        8 * 32 * 2
        + 32 * int(n_tile) * 2
        + 8 * int(n_tile) * 4
    )
    if threadgroup_bytes > 32 * 1024:
        raise ValueError(
            f"matrix M=8 threadgroup uses {threadgroup_bytes} bytes"
        )

    key = (
        "m8_ksplit_fp16",
        int(group_size),
        dtype,
        int(k_parts),
        int(n_tile),
        int(k_value) if k_value is not None else None,
        int(n_value) if n_value is not None else None,
    )
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    accumulators = "\n".join(
        f"simdgroup_matrix<float, 8, 8> c{index} = "
        "simdgroup_matrix<float, 8, 8>(0.0f);"
        for index in range(int(n_tile) // 8)
    )
    matrix_steps = "\n".join(
        f"""
                simdgroup_load(
                    b, b_tile[part] + ks * BK_SUB * BN + {index * 8}, BN
                );
                simdgroup_multiply_accumulate(c{index}, a, b, c{index});
        """
        for index in range(int(n_tile) // 8)
    )
    matrix_stores = "\n".join(
        f"simdgroup_store(c{index}, partial[part] + {index * 8}, BN);"
        for index in range(int(n_tile) // 8)
    )
    source = f"""
        using namespace metal;
        constexpr int M = 8;
        constexpr int BN = {int(n_tile)};
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {int(group_size)};

        uint tid = thread_position_in_threadgroup.x;
        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = {'int(K_size)' if k_value is None else int(k_value)};
        int N = {'int(N_size)' if n_value is None else int(n_value)};
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_chunk = K / K_PARTS;
        int k_begin = int(part) * k_chunk;
        int k_end = k_begin + k_chunk;

        threadgroup half x_tile[K_PARTS][M * BK];
        threadgroup half b_tile[K_PARTS][BK * BN];
        threadgroup float partial[K_PARTS][M * BN];

        simdgroup_matrix<half, 8, 8> a, b;
        {accumulators}

        using XVec = vec<T, 8>;
        using HVec = vec<half, 8>;
        const device XVec *x_vec = reinterpret_cast<const device XVec *>(x);

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int x_slot = int(lane); x_slot < M * BK / 8;
                 x_slot += 32) {{
                int x_row = x_slot / (BK / 8);
                int x_pack = x_slot - x_row * (BK / 8);
                XVec x_values =
                    x_vec[(x_row * K + k0 + x_pack * 8) / 8];
                *reinterpret_cast<threadgroup HVec *>(
                    x_tile[part] + x_row * BK + x_pack * 8
                ) = HVec(x_values);
            }}

            float lane_scale = 0.0f;
            float lane_bias = 0.0f;
            if (lane < BN) {{
                int group_index = int(k0 / GS);
                lane_scale =
                    float(scales[(n0 + int(lane)) * K_by_gs + group_index]);
                lane_bias =
                    float(biases[(n0 + int(lane)) * K_by_gs + group_index]);
            }}

            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < BK * BN / (32 * 8);
                 ++pack_idx) {{
                int packed_slot = pack_idx * 32 + int(lane);
                int dq_k = packed_slot / BN;
                int dq_n = packed_slot - dq_k * BN;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                uint32_t packed =
                    w_q[n_global * K_by_8 + (k_base >> 3)];
                float scale = simd_shuffle(lane_scale, dq_n);
                float bias = simd_shuffle(lane_bias, dq_n);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nibble = (packed >> (ki * 4)) & 0xFu;
                    b_tile[part][(dq_k * 8 + ki) * BN + dq_n] =
                        half(float(nibble) * scale + bias);
                }}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);

            _Pragma("unroll")
            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a, x_tile[part] + ks * BK_SUB, BK);
                {matrix_steps}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        {matrix_stores}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint output_idx = tid; output_idx < M * BN;
             output_idx += K_PARTS * 32) {{
            float total = 0.0f;
            _Pragma("unroll")
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p][output_idx];
            }}
            int row = int(output_idx) / BN;
            int col = int(output_idx) - row * BN;
            y[row * N + n0 + col] = T(total);
        }}
    """

    dtype_tag = {
        mx.bfloat16: "bf16",
        mx.float16: "fp16",
        mx.float32: "fp32",
    }.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=(
            f"verify_m8_ksplit_fp16_kp{int(k_parts)}_"
            f"nt{int(n_tile)}_"
            f"k{int(k_value) if k_value is not None else 'dyn'}_"
            f"n{int(n_value) if n_value is not None else 'dyn'}_"
            f"gs{group_size}_{dtype_tag}"
        ),
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel_m8_tuned(
    group_size: int,
    dtype: mx.Dtype,
    K: int,
    N: int,
    config: tuple[str, int, int],
):
    """Build the kernel selected by :func:`_m8_tuned_config`."""
    variant, n_tile, k_parts = config
    if variant == "matrix_fp16":
        return _build_kernel_m8_ksplit_fp16(
            group_size,
            dtype,
            k_parts=k_parts,
            n_tile=n_tile,
            k_value=K,
            n_value=N,
        )
    if variant != "scalar":
        raise ValueError(f"unsupported tuned M=8 variant: {variant}")
    if n_tile == 4:
        return _build_kernel_m8_scalar_kmajor(
            group_size,
            dtype,
            k_parts=k_parts,
            k_value=K,
            n_value=N,
        )
    return _build_kernel_m8_scalar_tiled(
        group_size,
        dtype,
        k_parts=k_parts,
        n_tile=n_tile,
        k_value=K,
        n_value=N,
    )


def _build_kernel_m16_combo_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_combo_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};

        constexpr int K       = KCONST;
        constexpr int K_by_8  = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane  = tid % 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB, K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB, K);
                simdgroup_load(b_L, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_store(c_tL, tg_partials[sg_id], BN);
        simdgroup_store(c_tR, tg_partials[sg_id] + 8, BN);
        simdgroup_store(c_bL, tg_partials[sg_id] + 8 * BN, BN);
        simdgroup_store(c_bR, tg_partials[sg_id] + 8 * BN + 8, BN);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc = 0.0f;
            _Pragma("unroll")
            for (int g = 0; g < NSG; ++g) {{
                acc += tg_partials[g][off];
            }}
            int row = off / BN;
            int col = off - row * BN;
            int n_global = n0 + col;
            y[row * N + n_global] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_combo_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_m16_super_tree_fp16_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_super_tree_fp16_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};

        constexpr int K       = KCONST;
        constexpr int K_by_8  = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane  = tid % 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup half B_tile[NSG][BK * BN];
        threadgroup half x_half[NSG][BM * BK];
        threadgroup half h_scratch[NSG][BM * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<half, 8, 8> a_top_h, a_bot_h, b_L_h, b_R_h;
        simdgroup_matrix<half, 8, 8> c_tL_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_tR_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_bL_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_bR_h = simdgroup_matrix<half, 8, 8>(half(0));

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int t = 0; t < BM * BK / 32; ++t) {{
                int slot = t * 32 + int(lane);
                int row = slot / BK;
                int kk = slot - row * BK;
                x_half[sg_id][row * BK + kk] = half(float(x[row * K + k0 + kk]));
            }}

            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = half(float(nib) * s + b);
                }}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top_h, x_half[sg_id] + ks * BK_SUB, BK);
                simdgroup_load(a_bot_h, x_half[sg_id] + 8 * BK + ks * BK_SUB, BK);
                simdgroup_load(b_L_h, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R_h, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                simdgroup_multiply_accumulate(c_tL_h, a_top_h, b_L_h, c_tL_h);
                simdgroup_multiply_accumulate(c_tR_h, a_top_h, b_R_h, c_tR_h);
                simdgroup_multiply_accumulate(c_bL_h, a_bot_h, b_L_h, c_bL_h);
                simdgroup_multiply_accumulate(c_bR_h, a_bot_h, b_R_h, c_bR_h);
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_store(c_tL_h, h_scratch[sg_id], BN);
        simdgroup_store(c_tR_h, h_scratch[sg_id] + 8, BN);
        simdgroup_store(c_bL_h, h_scratch[sg_id] + 8 * BN, BN);
        simdgroup_store(c_bR_h, h_scratch[sg_id] + 8 * BN + 8, BN);

        simdgroup_barrier(mem_flags::mem_threadgroup);

        _Pragma("unroll")
        for (uint i = 0; i < BM * BN / 32; ++i) {{
            uint off = i * 32u + lane;
            tg_partials[sg_id][off] = float(h_scratch[sg_id][off]);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sg_id & 1u) == 0u) {{
            uint src_sg = sg_id + 1u;
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[sg_id][off] += tg_partials[src_sg][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sg_id & 3u) == 0u) {{
            uint src_sg = sg_id + 2u;
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[sg_id][off] += tg_partials[src_sg][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {{
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[0][off] += tg_partials[4][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {{
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                int row = int(off) / BN;
                int col = int(off) - row * BN;
                int n_global = n0 + col;
                y[row * N + n_global] = T(tg_partials[0][off]);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_super_tree_fp16_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big(group_size: int, dtype: mx.Dtype):
    key = ("mma2big", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;

        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
                }}
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,                  N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,              N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,          N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8,      N);
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_8bit(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_8bit", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;

        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base   = k0 + dq_k_a * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_a * 8 + ki)     * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base   = k0 + dq_k_b * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_b * 8 + ki)     * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,             N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,         N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,     N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8, N);
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_8bit_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_pipe(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_pipe", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                              \\
            {{                                                                          \\
                int n_global = n0 + dq_n_a;                                             \\
                int k_base = (k0_stage) + dq_k_a * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
            {{                                                                          \\
                int n_global = n0 + dq_n_b;                                             \\
                int k_base = (k0_stage) + dq_k_b * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_pipe_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
        output_names=["partials"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_pipe_8bit(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_pipe_8bit", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                                                                  \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_a;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_a * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + ki)     * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_b;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_b * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + ki)     * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_pipe_8bit_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
        output_names=["partials"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _should_use_verify(
    x: mx.array,
    group_size: int,
    bits: int,
    transpose: bool,
) -> bool:
    if not is_enabled():
        return False
    if bits not in (4, 8) or group_size not in (32, 64, 128):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if not transpose:
        return False
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m in (4, 8, 16)

def verify_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    if not _should_use_verify(x, group_size, bits, transpose):
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=transpose, group_size=group_size, bits=bits,
        )

    orig_shape = x.shape
    m = 1
    for d in orig_shape[:-1]:
        m *= d
    x2 = mx.contiguous(x.reshape(m, orig_shape[-1]))
    w_q = mx.contiguous(w)
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)

    M = int(m)
    K = int(x2.shape[-1])
    N = int(w_q.shape[0])

    if M == 4:
        if not _m4_ksplit_np_shape(K, N, bits):
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        k_parts = _m4_ksplit_np_kparts(N)
        kernel = _build_kernel_m4_ksplit_np(group_size, x.dtype, k_parts=k_parts)
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, K, N],
            template=[("T", x.dtype)],
            grid=(32 * k_parts, N // 4, 1),
            threadgroup=(32 * k_parts, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )
        return y.reshape(*orig_shape[:-1], N)

    if M == 8:
        m8_config = _m8_tuned_config(K, N, bits)
        if m8_config is None:
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        _, n_tile, k_parts = m8_config
        kernel = _build_kernel_m8_tuned(
            group_size,
            x.dtype,
            K,
            N,
            m8_config,
        )
        threadgroup_size = 32 * k_parts
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, K, N],
            template=[("T", x.dtype)],
            grid=(threadgroup_size, N // n_tile, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )
        return y.reshape(*orig_shape[:-1], N)

    variant = _variant()
    ktmpl_variant = _resolve_m16_ktmpl_variant(K, N, bits, variant)

    if ktmpl_variant is not None:
        if K % 256 != 0 or N % 16 != 0 or bits != 4:
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        if ktmpl_variant == "nax_ktmpl":
            kernel = _build_kernel_m16_nax_ktmpl(K, group_size, x.dtype)
            grid = (256, N // 32, 1)
        else:
            kernel = (
                _build_kernel_m16_combo_ktmpl(K, group_size, x.dtype)
                if ktmpl_variant == "combo_ktmpl" else
                _build_kernel_m16_super_tree_fp16_ktmpl(K, group_size, x.dtype)
            )
            grid = (256, N // 16, 1)
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, N],
            template=[("T", x.dtype), ("KCONST", K)],
            grid=grid,
            threadgroup=(256, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )
        return y.reshape(*orig_shape[:-1], N)

    auto_kp: int | None = None
    if variant == "auto":
        variant, auto_kp = _auto_variant(K, N)

    K_PARTS = auto_kp if auto_kp is not None else _debug_verify_qmm_kparts(4)

    if variant == "mma2big_pipe":
        if N % 32 != 0 or K % (32 * K_PARTS) != 0:
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        kernel = (
            _build_kernel_mma2big_pipe_8bit(group_size, x.dtype)
            if bits == 8 else
            _build_kernel_mma2big_pipe(group_size, x.dtype)
        )
        (partials,) = kernel(
            inputs=[x2, w_q, scales, biases, M, K, N, K_PARTS],
            template=[("T", x.dtype)],
            grid=(64, N // 32, K_PARTS),
            threadgroup=(64, 1, 1),
            output_shapes=[(K_PARTS, M, N)],
            output_dtypes=[mx.float32],
        )
        y = partials.sum(axis=0).astype(x.dtype)
        return y.reshape(*orig_shape[:-1], N)

    if N % 32 != 0 or K % 32 != 0:
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=transpose, group_size=group_size, bits=bits,
        )
    kernel = (
        _build_kernel_mma2big_8bit(group_size, x.dtype)
        if bits == 8 else
        _build_kernel_mma2big(group_size, x.dtype)
    )
    (y,) = kernel(
        inputs=[x2, w_q, scales, biases, M, K, N],
        template=[("T", x.dtype)],
        grid=(64, N // 32, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*orig_shape[:-1], N)
