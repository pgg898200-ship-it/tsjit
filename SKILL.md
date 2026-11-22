---
name: tsjit
description: Profile, compile, verify strict FP32 parity, and deploy sub-microsecond streaming time-series foundation models (Toto-2.0, Chronos-Bolt, Chronos-2) and multi-agent streaming pipelines using tsjit and its stdio MCP server.
---

# tsjit Agent Skill

Use this skill when optimizing time-series foundation model (TSFM) inference latency/throughput, building causal online streaming or conformal calibration loops, isolating multi-agent sensor feeds against `NaN` corruption, or scaling online Ridge / cross-series fusion across NVIDIA GPUs.

## Step 1: Inspect Host Capabilities First

Before picking a compiler precision or runtime mode, run `tsjit info --json` (or invoke the `tsjit_info` MCP tool):

```bash
tsjit info --json
tsjit verify --json
```

Confirm:
- `native.available == true` and note `native.simd_isa` (`AVX2` or `NEON`).
- `optional_backends.cuda_available` and `optional_backends.triton` before invoking `tsjit.compile` GPU passes.

## Step 2: Compile Public TSFMs Safely (`tsjit.compile`)

Always start with `precision="a10-auto", mode="exact", verify=True` so `tsjit` audits compiled output against unmodified strict-FP32 eager execution and auto-escalates outlier-sensitive layers to `fp16x3`:

```python
import tsjit

compiled = tsjit.compile(
    model,
    horizon=16,
    context_length=512,
    precision="a10-auto",
    mode="exact",
    verify=True,
    verify_tol=0.05,
)
```

### Precision & Pass Rules
- **`Datadog/Toto-2.0-4m` / `22m`:** Keep `precision="a10-auto"` (escalates to `fp16x3`) whenever inputs may contain `NaN` missingness. Switch to `precision="fast-tf32"` only after confirming clean streams and domain-acceptable tail error (`fast-tf32` is `3x` faster than `fp16x3` but fails `10/512` on `25%-NaN` stress slices). `tsjit` also resolves upstream `B >= 4096` CUDA launch grid crashes automatically.
- **`amazon/chronos-2`:** Independent/singleton streams automatically collapse `GroupSelfAttention` from `O(B^2)` to `O(B)` (`26.4x` vs eager, `35.8x` vs `torch.compile` at `B=4096`) while preserving exact per-patch validity masking under full all-NaN patch outages. Multi-series grouped calls fall back safely to exact grouped attention.
- **`autogluon/chronos-bolt-small`:** Passes `0/1,536` strict parity across `fast-tf32`, `a10-auto`, and `fp16x3`.
- **Standalone `fp16x3_linear` / `scaled_fp16_linear`:** Use for `M >= 2048` or inside CUDA Graphs (`1.58x–3.12x` faster than `cuBLAS FP32` with `FP32`-grade `FroRelErr ~ 1e-6`). Avoid ungraphed `M <= 1024` calls where ~`120 us` launch overhead dominates.

## Step 3: Build Strictly Causal Streaming Pipelines (`tsjit.stream`)

1. Define a positive-allowlist `StreamSpec` (unknown keys and `lookahead != 0` are rejected):

```python
import tsjit

spec = tsjit.StreamSpec(
    name="telemetry_stream",
    window=64,
    max_age=4,
    alpha=0.10,
    revision_mode="soft_v2",
    gate_expr="abs_sg >= 2.5 or e_cusum >= 20.0 or age >= max_age",
)
```

2. Always verify zero future leakage with `assert_causal` before shipping:

```python
from tsjit.runtime.arrival import assert_causal

report = assert_causal(
    lambda: tsjit.stream(compiled_or_none, spec=spec),
    historical_series,
    split_step=len(historical_series) // 2,
)
assert report["causal"] and report["max_diff"] == 0.0
```

3. Understand latency vs CRPS trade-offs:
   - If the SLA allows `0.3–1.0 ms` per tick (single-series GPU), calling `tsjit.compile` synchronously avoids the 1-tick gap lag (`+1.9%` to `+20.7%` CRPS on volatile series).
   - If the SLA requires sub-microsecond arrival (`0.121 us/decision` batched C11) or `~74%` GPU compute elision with bullseye OnlineACI coverage (`80%` / `90%`), use `tsjit.stream` with `event_gated` or `gap_async`.

## Step 4: Multi-Agent Isolation & Distributed Scaling

- **Multiple agents reading shared sensor channels:** Use `tsjit.MultiAgentStreamHub(n_channels, window=64)` so agents share one EBR arena without `NaN` bleed across disjoint channel subsets, and check `hub.verify_ledger()` in tests.
- **High-rate GPU online Ridge (`K <= 136`):** Refresh the IPCG preconditioner `M0` every `<= 8` steps (FP64) or `<= 16` steps (FP32); non-converged rows fall back to GPU Cholesky automatically.
- **2–4+ GPU cross-series fusion:** Use `AsyncFuseRunner` (`16.64 KB/step`, `99.1%` 4-GPU weak scaling, `1/W` straggler amortization) for 1-step-lagged streaming, or `PipeFuseRunner` (`<= 5e-7` central parity) for strict zero-lag exactness.

## Step 5: MCP Server Integration

Start `tsjit mcp` over stdio to let agents invoke `tsjit_info`, `tsjit_verify`, `tsjit_benchmark`, `tsjit_validate_spec`, `tsjit_stream_sim`, and `tsjit_hub_demo` with strict RFC 8259 JSON and bounded schemas.
