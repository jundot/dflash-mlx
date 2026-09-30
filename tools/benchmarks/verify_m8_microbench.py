"""Compare the production M=8 verifier with M=4 and stock MLX QMM."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import mlx.core as mx

from dflash_mlx.verify_qmm import (
    _build_kernel_m4_ksplit_np,
    _build_kernel_m8_tuned,
    _m4_ksplit_np_kparts,
    _m8_tuned_config,
)


def _measure(
    fn: Callable[[], mx.array],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        mx.eval(fn())

    samples_ms = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        mx.eval(fn())
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return statistics.median(samples_ms)


def _custom_runner(
    kernel,
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    k: int,
    n: int,
    m: int,
    n_tile: int,
    k_parts: int,
) -> Callable[[], mx.array]:
    threads = 32 * k_parts

    def run() -> mx.array:
        (output,) = kernel(
            inputs=[x, weight, scales, biases, k, n],
            template=[("T", x.dtype)],
            grid=(threads, n // n_tile, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(m, n)],
            output_dtypes=[x.dtype],
        )
        return output

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=17408)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--variant",
        choices=("all", "stock-m8", "production-m4", "production-m8"),
        default="all",
        help="Select one kernel when collecting a Metal System Trace.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.k % 64 or args.n % 4:
        parser.error("--k must be divisible by 64 and --n by 4")

    dtype = {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
    }[args.dtype]
    group_size = 64
    x = mx.random.normal((8, args.k)).astype(dtype)
    weight = mx.zeros((args.n, args.k // 8), dtype=mx.uint32)
    scales = mx.ones((args.n, args.k // group_size), dtype=dtype)
    biases = mx.zeros((args.n, args.k // group_size), dtype=dtype)
    mx.eval(x, weight, scales, biases)

    def stock_m8() -> mx.array:
        return mx.quantized_matmul(
            x,
            weight,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=group_size,
            bits=4,
        )

    m4_k_parts = _m4_ksplit_np_kparts(args.n)
    m4_kernel = _build_kernel_m4_ksplit_np(
        group_size,
        dtype,
        k_parts=m4_k_parts,
    )
    production_m4 = _custom_runner(
        m4_kernel,
        x[:4],
        weight,
        scales,
        biases,
        k=args.k,
        n=args.n,
        m=4,
        n_tile=4,
        k_parts=m4_k_parts,
    )

    m8_config = _m8_tuned_config(args.k, args.n, 4)
    if m8_config is None:
        parser.error("this GPU or shape has no tuned production M=8 kernel")
    _, m8_n_tile, m8_k_parts = m8_config
    m8_kernel = _build_kernel_m8_tuned(
        group_size,
        dtype,
        args.k,
        args.n,
        m8_config,
    )
    production_m8 = _custom_runner(
        m8_kernel,
        x,
        weight,
        scales,
        biases,
        k=args.k,
        n=args.n,
        m=8,
        n_tile=m8_n_tile,
        k_parts=m8_k_parts,
    )

    runners = {
        "stock_m8": stock_m8,
        "production_m4": production_m4,
        "production_m8": production_m8,
    }
    if args.variant != "all":
        selected = args.variant.replace("-", "_")
        runners = {selected: runners[selected]}

    latency_ms = {
        name: _measure(
            runner,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for name, runner in runners.items()
    }
    rows_per_ms = {
        name: (4 if name == "production_m4" else 8) / duration
        for name, duration in latency_ms.items()
    }
    result = {
        "architecture": str(mx.device_info().get("architecture", "unknown")),
        "shape": {"k": args.k, "n": args.n},
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "m8_config": m8_config,
        "latency_ms": latency_ms,
        "rows_per_ms": rows_per_ms,
    }
    if "production_m4" in rows_per_ms and "production_m8" in rows_per_ms:
        result["m8_vs_m4_row_throughput"] = (
            rows_per_ms["production_m8"] / rows_per_ms["production_m4"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
