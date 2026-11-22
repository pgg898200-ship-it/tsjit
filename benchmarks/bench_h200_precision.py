# Copyright 2026 tsjit contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live H200 benchmark for Crest-Guarded Scaled-FP16 & FP16x3 vs FP32/TF32/BF16/FP8.

Evaluates:
- Shapes (B=64..4096, K=256..3072, N=256..3072) including non-power-of-2
  688, 1368, 3072 and unaligned N % 4 != 0.
- Adversarial numerical battery without RMSNorm pre-scaling:
  1000x weight/activation outliers, raw 1e38, raw 1e-40 subnormals,
  ex + ew > 127 zero-cancelled dot products, and NaN/Inf propagation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from tsjit.cuda.loader import is_cublas_fp16_available
from tsjit.cuda.scaled_fp16_triton import (
    SM90_VERIFIED_REPORTS,
    TRITON_AVAILABLE,
    CrestGuardedLinear,
    fp16x3_linear,
    scaled_fp16_linear,
)


def _time_cuda(fn: Any, warmup: int, iters: int, device: Any) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times: list[float] = []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize(device)
        times.append(float(ev0.elapsed_time(ev1)))
    return float(np.median(times))


def _p99_1d(rel: Any) -> float:
    n = int(rel.numel())
    if n == 0:
        return 0.0
    if n <= 16_000_000:
        return float(torch.quantile(rel, 0.99).item())
    k = max(1, min(n, int(0.99 * (n - 1)) + 1))
    return float(torch.kthvalue(rel, k).values.item())


def _rel_error_vs_f64(out: Any, ref_f64: Any) -> tuple[float, float]:
    diff = (out.to(torch.float64) - ref_f64).abs()
    denom = ref_f64.abs().clamp(min=1e-12)
    rel = (diff / denom).reshape(-1)
    return float(rel.max().item()), _p99_1d(rel)


def _error_metrics_vs_f64(out: Any, ref_f64: Any) -> tuple[float, float, float]:
    """Return ``(frobenius_rel_err, p99_rel_err, max_rel_err)`` against FP64 reference."""
    out_f64 = out.to(torch.float64)
    diff = out_f64 - ref_f64
    fro_ref = float(torch.linalg.vector_norm(ref_f64).item())
    fro_rel = float(torch.linalg.vector_norm(diff).item()) / max(fro_ref, 1e-12)
    denom = ref_f64.abs().clamp(min=1e-12)
    rel = (diff.abs() / denom).reshape(-1)
    return fro_rel, _p99_1d(rel), float(rel.max().item())


def _prepare_fp8_weight(w: Any) -> tuple[Any | None, Any | None, str | None]:
    """Pre-quantize static FP8 e4m3fn weight offline outside the benchmark timer."""
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return None, None, "torch._scaled_mm unavailable"
    N, K = w.shape
    if (K % 16) != 0 or (N % 16) != 0:
        return None, None, "unaligned_for_fp8_16"
    try:
        w_scale = (w.abs().amax() / 448.0).clamp(min=1e-12).to(torch.float32)
        w_fp8 = (w / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        # torch._scaled_mm expects column-major (K, N) view transposed from contiguous (N, K).
        return w_fp8.t(), w_scale, None
    except Exception as exc:
        return None, None, str(exc)


def _run_fp8_scaled_mm_prequant(
    x: Any, w_fp8_t: Any, w_scale: Any, bias: Any | None = None
) -> tuple[Any | None, str | None]:
    """Run FP8 e4m3fn scaled_mm with pre-quantized weights and online activation quantization."""
    M, K = x.shape
    if (M % 16) != 0 or (K % 16) != 0:
        return None, "unaligned_for_fp8_16"
    try:
        x_scale = (x.abs().amax() / 448.0).clamp(min=1e-12).to(torch.float32)
        x_fp8 = (x / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            w_fp8_t,
            scale_a=x_scale,
            scale_b=w_scale,
            out_dtype=torch.float32,
        )
        if isinstance(out, tuple):
            out = out[0]
        if bias is not None:
            out = out + bias
        return out, None
    except Exception as exc:
        return None, str(exc)


def _run_adversarial_battery(device: Any) -> list[dict[str, Any]]:
    """Run raw (un-RMSNormed) adversarial numerical cases on GPU."""
    results: list[dict[str, Any]] = []
    gen = torch.Generator(device="cpu").manual_seed(2026)

    # 1. 1000x weight & activation outliers with cancellation (K=512 -> crest=16 > 12)
    M, K, N = 64, 512, 64
    x_out = (0.5 + 0.1 * torch.randn((M, K), generator=gen)).to(device=device, dtype=torch.float32)
    w_out = (0.5 + 0.1 * torch.randn((N, K), generator=gen)).to(device=device, dtype=torch.float32)
    x_out[:, 0] = 1000.1234
    x_out[:, 1] = -1000.1234
    w_out[:, 0] = 1000.5678
    w_out[:, 1] = 1000.5678
    ref_out = x_out.to(torch.float64) @ w_out.to(torch.float64).t()

    lin_auto = CrestGuardedLinear.from_linear(nn.Linear(K, N, bias=False).to(device), mode="auto")
    with torch.no_grad():
        lin_auto.weight.copy_(w_out)
    lin_auto.repack()
    y_auto = lin_auto(x_out)
    y_bf16 = F.linear(x_out.to(torch.bfloat16), w_out.to(torch.bfloat16)).to(torch.float32)
    err_auto, _ = _rel_error_vs_f64(y_auto, ref_out)
    err_bf16, _ = _rel_error_vs_f64(y_bf16, ref_out)
    results.append(
        {
            "case": "1000x_weight_and_act_outlier_cancel",
            "escalated_w": lin_auto.escalated_w,
            "max_crest_w": lin_auto.max_crest_w,
            "crest_guarded_max_rel_err": err_auto,
            "bf16_max_rel_err": err_bf16,
            "passed": bool(lin_auto.escalated_w and err_auto < 1e-3),
        }
    )

    # 2. Activation-only 1000x outlier (2-GEMM act-escalated, zero W1 VRAM!)
    w_norm = (0.5 + 0.1 * torch.randn((N, K), generator=gen)).to(device=device, dtype=torch.float32)
    w_norm[:, 0] = 1.0
    w_norm[:, 1] = 1.0
    ref_act = x_out.to(torch.float64) @ w_norm.to(torch.float64).t()
    lin_act = CrestGuardedLinear.from_linear(nn.Linear(K, N, bias=False).to(device), mode="auto")
    with torch.no_grad():
        lin_act.weight.copy_(w_norm)
    lin_act.repack()
    y_act = lin_act(x_out)
    err_act, _ = _rel_error_vs_f64(y_act, ref_act)
    results.append(
        {
            "case": "1000x_activation_only_2gemm_escalation",
            "escalated_w": lin_act.escalated_w,
            "max_crest_w": lin_act.max_crest_w,
            "crest_guarded_max_rel_err": err_act,
            "passed": bool((not lin_act.escalated_w) and err_act < 5e-3),
        }
    )

    # 3. Raw 1e38 inputs
    x_hi = (1e38 * (0.1 + 0.05 * torch.randn((32, 256), generator=gen))).to(
        device=device, dtype=torch.float32
    )
    w_hi = (1e-3 * (0.2 + 0.05 * torch.randn((32, 256), generator=gen))).to(
        device=device, dtype=torch.float32
    )
    ref_hi = x_hi.to(torch.float64) @ w_hi.to(torch.float64).t()
    y_hi = fp16x3_linear(x_hi, w_hi)
    err_hi, _ = _rel_error_vs_f64(y_hi, ref_hi)
    results.append(
        {
            "case": "raw_1e38_near_overflow",
            "all_finite": bool(torch.isfinite(y_hi).all().item()),
            "fp16x3_max_rel_err": err_hi,
            "passed": bool(torch.isfinite(y_hi).all().item() and err_hi < 1e-4),
        }
    )

    # 4. Raw 1e-40 subnormals
    x_sub = (1e-40 * (1.0 + 0.3 * torch.randn((32, 256), generator=gen))).to(
        device=device, dtype=torch.float32
    )
    w_sub = (1e12 * (0.5 + 0.2 * torch.randn((32, 256), generator=gen))).to(
        device=device, dtype=torch.float32
    )
    ref_sub = x_sub.to(torch.float64) @ w_sub.to(torch.float64).t()
    y_sub = fp16x3_linear(x_sub, w_sub)
    err_sub, _ = _rel_error_vs_f64(y_sub, ref_sub)
    results.append(
        {
            "case": "raw_1e-40_subnormal",
            "all_finite": bool(torch.isfinite(y_sub).all().item()),
            "fp16x3_max_rel_err": err_sub,
            "passed": bool(torch.isfinite(y_sub).all().item() and err_sub < 1e-4),
        }
    )

    # 5. ex + ew > 127 zero-cancelled dot product (zero 0*Inf = NaN!)
    x_zc = torch.tensor([[1e25, -1e25, 1e25, -1e25]], dtype=torch.float32, device=device)
    w_zc = torch.tensor([[1e25, 1e25, 1e25, 1e25]], dtype=torch.float32, device=device)
    y_zc = scaled_fp16_linear(x_zc, w_zc)
    results.append(
        {
            "case": "ex_plus_ew_gt_127_zero_cancel",
            "output_val": float(y_zc.item()),
            "is_exact_zero": bool(float(y_zc.item()) == 0.0),
            "passed": bool(torch.isfinite(y_zc).all().item() and float(y_zc.item()) == 0.0),
        }
    )

    # 6. Non-finite NaN propagation under graph_safe=True
    x_nan = x_hi.clone()
    x_nan[5, 13] = float("nan")
    y_nan = scaled_fp16_linear(x_nan, w_hi, graph_safe=True)
    results.append(
        {
            "case": "nan_row_propagation_graph_safe",
            "bad_row_all_nan": bool(torch.isnan(y_nan[5]).all().item()),
            "good_rows_all_finite": bool(torch.isfinite(y_nan[0]).all().item()),
            "passed": bool(
                torch.isnan(y_nan[5]).all().item() and torch.isfinite(y_nan[0]).all().item()
            ),
        }
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live H200 Crest-Guarded Scaled-FP16 & FP16x3 benchmark."
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", type=str, default="results/bench_h200_precision.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("bench_h200_precision requires PyTorch with CUDA.")

    device = torch.device("cuda:0")
    shapes: list[tuple[int, int, int]] = [
        (64, 256, 256),
        (512, 688, 688),  # non-pow2
        (1024, 1368, 1369),  # non-pow2 & N % 4 != 0
        (2048, 3072, 1024),  # K=3072 non-pow2
        (4096, 1024, 3072),
    ]

    cublas_wgmma = is_cublas_fp16_available(device)
    print(
        f"Device: {torch.cuda.get_device_name(device)} | "
        f"Triton: {TRITON_AVAILABLE} | cuBLAS WGMMA: {cublas_wgmma}",
        flush=True,
    )
    print(
        "| (B, K, N) | FP32 (ms) | TF32 (ms) | BF16 (ms) | FP8 (ms) | "
        "Scaled-FP16 (ms) | FP16x3 (ms) | S-FP16 vs FP32 | FP16x3 vs FP32 | "
        "FP16x3 FroRelErr | FP16x3 p99RelErr |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    bench_rows: list[dict[str, Any]] = []
    for B, K, N in shapes:
        gen = torch.Generator(device="cpu").manual_seed(B + K + N)
        x = torch.randn((B, K), generator=gen, dtype=torch.float32).to(device)
        base_lin = nn.Linear(K, N, bias=True).to(device=device, dtype=torch.float32).eval()
        w = base_lin.weight.detach()
        b = base_lin.bias.detach()

        with torch.inference_mode():
            ref_f64 = x.to(torch.float64) @ w.to(torch.float64).t() + b.to(torch.float64)

            # Strict FP32 cuBLAS
            torch.backends.cuda.matmul.allow_tf32 = False
            ms_fp32 = _time_cuda(
                lambda x_=x, w_=w, b_=b: F.linear(x_, w_, b_), args.warmup, args.iters, device
            )
            out_fp32 = F.linear(x, w, b)

            # TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            ms_tf32 = _time_cuda(
                lambda x_=x, w_=w, b_=b: F.linear(x_, w_, b_), args.warmup, args.iters, device
            )
            out_tf32 = F.linear(x, w, b)
            torch.backends.cuda.matmul.allow_tf32 = False

            # BF16
            x_bf = x.to(torch.bfloat16)
            w_bf = w.to(torch.bfloat16)
            b_bf = b.to(torch.bfloat16)
            ms_bf16 = _time_cuda(
                lambda x_=x_bf, w_=w_bf, b_=b_bf: F.linear(x_, w_, b_),
                args.warmup,
                args.iters,
                device,
            )
            out_bf16 = F.linear(x_bf, w_bf, b_bf)

            # FP8 (Hopper SM90): pre-quantize static weights ONCE outside the timer.
            w_fp8_t, w_fp8_scale, fp8_err = _prepare_fp8_weight(w)
            out_fp8 = None
            if w_fp8_t is not None and w_fp8_scale is not None:
                out_fp8, fp8_err = _run_fp8_scaled_mm_prequant(x, w_fp8_t, w_fp8_scale, b)
            if out_fp8 is not None:
                ms_fp8: float | None = _time_cuda(
                    lambda x_=x, wt_=w_fp8_t, ws_=w_fp8_scale, b_=b: _run_fp8_scaled_mm_prequant(
                        x_, wt_, ws_, b_
                    ),
                    args.warmup,
                    args.iters,
                    device,
                )
            else:
                ms_fp8 = None

            # Scaled-FP16 (1-GEMM, pre-packed W0)
            lin_s16 = CrestGuardedLinear.from_linear(base_lin, mode="scaled-fp16", graph_safe=True)
            ms_s16 = _time_cuda(lambda fn_=lin_s16, x_=x: fn_(x_), args.warmup, args.iters, device)
            out_s16 = lin_s16(x)

            # Crest-Guarded Auto (2-GEMM activation-safe on cuBLAS / 3-GEMM on weight outlier)
            lin_auto = CrestGuardedLinear.from_linear(base_lin, mode="auto", graph_safe=True)
            ms_auto = _time_cuda(
                lambda fn_=lin_auto, x_=x: fn_(x_), args.warmup, args.iters, device
            )

            # FP16x3 Ozaki (3-GEMM exact FP32)
            lin_oz = CrestGuardedLinear.from_linear(base_lin, mode="fp16x3", graph_safe=True)
            ms_oz = _time_cuda(lambda fn_=lin_oz, x_=x: fn_(x_), args.warmup, args.iters, device)
            out_oz = lin_oz(x)

            flops = 2.0 * B * K * N
            fro_fp32, p99_fp32, e_fp32 = _error_metrics_vs_f64(out_fp32, ref_f64)
            fro_tf32, p99_tf32, e_tf32 = _error_metrics_vs_f64(out_tf32, ref_f64)
            fro_bf16, p99_bf16, e_bf16 = _error_metrics_vs_f64(out_bf16, ref_f64)
            fro_s16, p99_s16, e_s16 = _error_metrics_vs_f64(out_s16, ref_f64)
            fro_oz, p99_oz, e_oz = _error_metrics_vs_f64(out_oz, ref_f64)
            if out_fp8 is not None:
                fro_fp8: float | None
                p99_fp8: float | None
                e_fp8: float | None
                fro_fp8, p99_fp8, e_fp8 = _error_metrics_vs_f64(out_fp8, ref_f64)
            else:
                fro_fp8, p99_fp8, e_fp8 = None, None, None
            sp_s16 = ms_fp32 / max(ms_s16, 1e-9)
            sp_oz = ms_fp32 / max(ms_oz, 1e-9)

            row = {
                "shape": [B, K, N],
                "fp32_ms": ms_fp32,
                "tf32_ms": ms_tf32,
                "bf16_ms": ms_bf16,
                "fp8_ms": ms_fp8,
                "fp8_prequantized_weight": True,
                "fp8_note": fp8_err,
                "fp8_frobenius_rel_err": fro_fp8,
                "fp8_p99_rel_err": p99_fp8,
                "fp8_max_rel_err": e_fp8,
                "scaled_fp16_ms": ms_s16,
                "crest_auto_ms": ms_auto,
                "fp16x3_ms": ms_oz,
                "scaled_fp16_speedup_vs_fp32": sp_s16,
                "fp16x3_speedup_vs_fp32": sp_oz,
                "scaled_fp16_tflops": flops / (ms_s16 * 1e9),
                "fp16x3_tflops": flops / (ms_oz * 1e9),
                "fp32_frobenius_rel_err": fro_fp32,
                "fp32_p99_rel_err": p99_fp32,
                "fp32_max_rel_err": e_fp32,
                "tf32_frobenius_rel_err": fro_tf32,
                "tf32_p99_rel_err": p99_tf32,
                "tf32_max_rel_err": e_tf32,
                "bf16_frobenius_rel_err": fro_bf16,
                "bf16_p99_rel_err": p99_bf16,
                "bf16_max_rel_err": e_bf16,
                "scaled_fp16_frobenius_rel_err": fro_s16,
                "scaled_fp16_p99_rel_err": p99_s16,
                "scaled_fp16_max_rel_err": e_s16,
                "fp16x3_frobenius_rel_err": fro_oz,
                "fp16x3_p99_rel_err": p99_oz,
                "fp16x3_max_rel_err": e_oz,
            }
            bench_rows.append(row)
            fp8_disp = f"{ms_fp8:.3f}" if ms_fp8 is not None else "n/a"
            print(
                f"| ({B}, {K}, {N}) | {ms_fp32:.3f} | {ms_tf32:.3f} | {ms_bf16:.3f} | "
                f"{fp8_disp} | {ms_s16:.3f} | {ms_oz:.3f} | {sp_s16:.2f}x | {sp_oz:.2f}x | "
                f"{fro_oz:.2e} | {p99_oz:.2e} |",
                flush=True,
            )

    print("\n### Numerical Exactness vs FP64 Reference (Frobenius RelErr / p99 RelErr)", flush=True)
    print(
        "| (B, K, N) | FP32 Fro / p99 | TF32 Fro / p99 | BF16 Fro / p99 | "
        "FP8 Fro / p99 | Scaled-FP16 Fro / p99 | FP16x3 Fro / p99 |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|", flush=True)
    for r in bench_rows:
        B, K, N = r["shape"]
        fp8_err_disp = (
            f"{r['fp8_frobenius_rel_err']:.2e} / {r['fp8_p99_rel_err']:.2e}"
            if r["fp8_frobenius_rel_err"] is not None
            else "n/a"
        )
        print(
            f"| ({B}, {K}, {N}) | "
            f"{r['fp32_frobenius_rel_err']:.2e} / {r['fp32_p99_rel_err']:.2e} | "
            f"{r['tf32_frobenius_rel_err']:.2e} / {r['tf32_p99_rel_err']:.2e} | "
            f"{r['bf16_frobenius_rel_err']:.2e} / {r['bf16_p99_rel_err']:.2e} | "
            f"{fp8_err_disp} | "
            f"{r['scaled_fp16_frobenius_rel_err']:.2e} / {r['scaled_fp16_p99_rel_err']:.2e} | "
            f"{r['fp16x3_frobenius_rel_err']:.2e} / {r['fp16x3_p99_rel_err']:.2e} |",
            flush=True,
        )

    adversarial = _run_adversarial_battery(device)
    n_pass = sum(1 for item in adversarial if item["passed"])
    print(f"\nAdversarial Battery: {n_pass}/{len(adversarial)} PASS", flush=True)
    for item in adversarial:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  [{status}] {item['case']}", flush=True)

    payload = {
        "device": torch.cuda.get_device_name(device),
        "triton_available": TRITON_AVAILABLE,
        "cublas_wgmma_available": cublas_wgmma,
        "sm90_verified_configs": [
            {
                "tile": [r.bm, r.bn, r.bk],
                "stages": r.num_stages,
                "warps": r.num_warps,
                "smem_bytes": r.smem_bytes,
                "elems_per_thr": r.elems_per_thr,
                "est_regs": r.est_regs,
            }
            for r in SM90_VERIFIED_REPORTS
        ],
        "shapes": bench_rows,
        "adversarial_battery": adversarial,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved JSON report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
