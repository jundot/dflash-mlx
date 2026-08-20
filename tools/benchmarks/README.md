# Internal Benchmark Tools

This directory contains lab harnesses. They are useful for diagnosis, but they
are not the public benchmark contract.

Use `dflash benchmark` first for public baseline-vs-DFlash claims.
Named suites such as `smoke`, `humaneval`, `gsm8k`, `math500`, and `longctx`
belong to that public command, not to these lab tools.

## Surfaces

```bash
PYTHONPATH=$PWD python -m tools.benchmarks.agentic_trace --help
PYTHONPATH=$PWD python -m tools.benchmarks.context_grid --help
PYTHONPATH=$PWD python -m tools.benchmarks.prefix_cache_survival_gate --help
PYTHONPATH=$PWD python -m tools.benchmarks.prefix_cache_probe --help
PYTHONPATH=$PWD python -m tools.benchmarks.analyze_trace --help
PYTHONPATH=$PWD python -m tools.benchmarks.verify_m8_microbench --help
```

`agentic_trace.py`
: Agentic session/proxy trace tooling. Use it to study server behavior under a
real client shape, not as a public speed claim by default.

`context_grid.py`
: Long-context speed grid. It runs `mlx_lm` and DFlash as separate backend
phases, one heavy backend at a time, and writes durable `rows.jsonl`,
`memory_samples.jsonl`, `summary.json`, and `summary.md` with
prompt-processing speed, generation speed, MLX peak memory, and Darwin physical
footprint start/peak/end/delta by context bucket. Treat this as private
roofline/memory evidence, not a public throughput claim. Use
`--clear-cache-between-cases` to separate per-bucket allocator pressure from a
long-lived process ladder.

`prefix_cache_survival_gate.py`
: Long-context prefix-cache correctness gate. It is not a statistical NIAH benchmark.

`prefix_cache_probe.py`
: Prefix-cache and L2 mechanism probes.

`analyze_trace.py`
: Trace and prompt/memory analyzers.

`verify_m8_microbench.py`
: Compares the production M=8 verifier against production M=4 and stock MLX
QMM. Select a single `--variant` when wrapping it in Metal System Trace.

## Capture vs replay

`agentic_trace run` is a capture/orchestration harness. It starts a server,
starts the recording proxy, drives a real OpenCode or pi session, and writes
the observed requests, SSE stream, server diagnostics, and workspace output.
By default it writes one run directory under `.artifacts/dflash/traces/...`.
Pass `--workspace-source <repo>` to copy a real local project into the run
workspace before OpenCode starts. The copy excludes common VCS, cache, vendor,
artifact, and secret-like paths by default; add `--workspace-exclude <pattern>`
for project-specific skips. Use this mode for long real-project sessions where
prompt growth, tool outputs, and cache behavior must look like a user working
inside a non-empty repository.
Pass `--system-sample-interval-s <seconds>` on live runs or replays when the
artifact needs server RSS, VM wired memory, pageout, and `pmset` samples aligned
with the request timeline. Keep it off for clean public throughput claims.

`agentic_trace replay` is a fixed-body replay harness. It starts a fresh
DFlash or `mlx_lm.server`, replays captured `requests/*.json` bodies directly,
and writes a fresh `summary.json` / `compare.md`. Use replay when a live
OpenCode/pi trajectory would add noise to a runtime or cache A/B. Replay fixes
the request bodies, not the model's generated token stream, so decode length can
still vary.

Both `run` and `replay` also write table-ready `rows.jsonl` and `rows.md`.
Those files normalize one row per OpenAI-compatible POST: `cold` / `warm` /
`warm-l2` cache class, `cache_hit_source` (`L1`, `L2`, `disk`, or `unknown`),
prompt/cached/computed tokens, TTFT, prefill/decode timing, decode TPS,
acceptance/cycles, tool calls, finish reason, and server-process physical
footprint start/end/delta when DFlash diagnostics emit boundary samples.
Prefer `--diagnostics basic` for throughput rows. `--diagnostics full` is for
mechanism evidence and can depress TPS because it records heavier cycle/memory
events.

## Cross-runtime opencode comparison (dflash vs mlxlm)

Run baseline first, then dflash with `--compare-to`:

```bash
python -m tools.benchmarks.agentic_trace run --backend mlxlm --target <model> \
    --task-file <task.txt> --label mlx_baseline
python -m tools.benchmarks.agentic_trace run --backend dflash --target <model> \
    --draft <draft> --task-file <task.txt> --label dflash_run \
    --draft-quant w4 --prefill-step-size 1024 \
    --fastpath-max-tokens 0 --prefix-cache --prefix-cache-l2 \
    --diagnostics basic \
    --compare-to .artifacts/dflash/traces/<mlx_baseline_dir>
```

The dflash run emits `compare.md` with:

- **Trajectory-robust aggregate metrics** (`decode_tps_avg`,
  `prefill_tokens_saved`, `weighted_acceptance`,
  `observed_response_ms_per_output_token`) — aggregate rates that remain
  readable when trajectories diverge. `observed_response_ms_per_output_token`
  is response wall time divided by streamed output tokens; it is not a prefill
  metric.
- **Trajectory-dependent metrics** (`wall_s`, POST count) shown for reference
  with caveats — if trajectories diverge, these metrics describe the captured
  sessions, not a direct runtime-speed comparison.
- **Per-POST gap table** emitted only when trajectories align: same POST
  count and decode tokens within ±5%. Otherwise it is omitted with an explicit
  `TRAJECTORY DIVERGED` warning.

For deterministic per-token A/B with no trajectory divergence by construction,
use `dflash benchmark` on a fixed prompt. That benchmark runs both runtimes
in-process on identical input.

For real captured POST bodies with fixed prompt shape:

```bash
python -m tools.benchmarks.agentic_trace replay \
    --source-trace .artifacts/dflash/traces/<captured_dir> \
    --backend mlxlm --target <model> --label mlx_replay

python -m tools.benchmarks.agentic_trace replay \
    --source-trace .artifacts/dflash/traces/<captured_dir> \
    --backend dflash --target <model> --draft <draft> \
    --draft-quant w4 --prefill-step-size 1024 \
    --prefix-cache --prefix-cache-l2 \
    --diagnostics basic --label dflash_replay \
    --compare-to .artifacts/dflash/traces/<mlx_replay_dir>
```

For DFlash trace/replay, the harness passes `--fastpath-max-tokens 0` by
default so short POSTs still exercise the DFlash path and emit prefix-cache and
boundary-memory events. Override it only when deliberately measuring target-only
fastpath behavior.

Private `_*.py` files are implementation modules for these wrappers.

## Tool audit

| Tool | Verdict | One-sentence question |
| --- | --- | --- |
| `agentic_trace.py` | KEEP | On a real OpenCode/pi loop, how many POSTs, tokens, cache hits, tok/s, and acceptance did the runtime produce? |
| `context_grid.py` | KEEP | How do prompt-processing speed, generation speed, wall time, and peak memory scale across long-context buckets for `mlx_lm` vs DFlash? |
| `prefix_cache_survival_gate.py` | KEEP | Does a warm prefix-cache request preserve the correct long-context record after a divergent suffix and reject a stale wrong-haystack answer? |
| `_agentic_trace.py` | KEEP | How does the orchestrated server+proxy+client capture build the trace and compare it to a peer run? |
| `_agentic_proxy.py` | KEEP | What exact OpenAI-compatible requests and SSE events did the client send and receive? |
| `_agentic_session.py` | FOLD later | What did a raw OpenCode/pi client session emit without the server/proxy orchestration? |
| `analyze_trace.py` | KEEP | In a captured trace, where did prompt tokens, memory buckets, and request costs concentrate? |
| `prefix_cache_probe.py` | KEEP | Does prefix-cache L1/L2 hit, evict, restore, and account bytes mechanically? |
| `_prefix_cache_multiturn.py` | KEEP | Does an in-process multi-turn prefix-cache path produce expected hit/reuse behavior? |
| `_prefix_l2_long_session.py` | KEEP | Does a real-model long-session L2 path restore snapshots across revisits? |
| `_prefix_l2_synthetic.py` | KEEP | What are L2 write/read/size mechanics without loading a model? |

## Output Policy

New lab outputs should go under `.artifacts/dflash/...` or an explicit local
`--out` path. Do not add new benchmark outputs to Git.

Some harnesses are intentionally narrower than `dflash benchmark` and may not
write a full public manifest. If a result will be quoted outside local
debugging, record the exact command, model refs, git hash, runtime flags, prompt
tokenization mode, and output directory.

## Rules

- Do not compare numbers across harnesses as if they were one protocol.
- Do not use full tracing for throughput claims.
- Do not overlap heavy model loads.
- Keep mechanism probes separate from product benchmark results.
- Prefer the smallest harness that answers the question.
