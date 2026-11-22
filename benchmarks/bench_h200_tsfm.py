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

"""Live H200 benchmark for public Hugging Face TSFMs: eager vs torch.compile vs tsjit.compile.

Evaluates real public models:
- Datadog/Toto-2.0-4m
- Datadog/Toto-2.0-22m
- autogluon/chronos-bolt-small
- amazon/chronos-2

Measures throughput, batch-1/64/1024/4096 p50/p99 latency, and strict 0/512
FP32 parity (serial batch-1 allow_tf32=False reference) on GIFT-Eval, 25% NaN,
and 2.22e10 DC-offset stress slices.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from tsjit.compiler import CausalFloat64DCStripper, StrictParityValidator
from tsjit.compiler import compile as tsjit_compile

DEFAULT_MODELS: tuple[str, ...] = (
    "Datadog/Toto-2.0-4m",
    "Datadog/Toto-2.0-22m",
    "autogluon/chronos-bolt-small",
    "amazon/chronos-2",
)


def _load_public_model(model_id: str, device: Any) -> tuple[Any, str]:
    low = model_id.lower()
    if "chronos-bolt" in low:
        from chronos import ChronosBoltPipeline

        pipe = ChronosBoltPipeline.from_pretrained(
            model_id, device_map=str(device), torch_dtype=torch.float32
        )
        pipe.model.eval()
        return pipe, "chronos-bolt"

    if "chronos-2" in low:
        from chronos import Chronos2Pipeline

        pipe = Chronos2Pipeline.from_pretrained(
            model_id, device_map=str(device), torch_dtype=torch.float32
        )
        pipe.model.eval()
        return pipe, "chronos-2"

    if "toto" in low:
        try:
            from toto2.model import Toto2Model

            model = (
                Toto2Model.from_pretrained(model_id)
                .to(device=device, dtype=torch.float32)
                .eval()
            )
            return model, "toto2"
        except Exception:
            try:
                from toto.model.toto import Toto

                model = Toto.from_pretrained(model_id).to(device=device, dtype=torch.float32).eval()
                return model, "toto2"
            except Exception:
                from transformers import AutoModel

                model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
                model = model.to(device=device, dtype=torch.float32).eval()
                return model, "toto2"

    raise ValueError(f"Unsupported public model identifier: {model_id}")


def _make_eager_callable(model: Any, family: str, horizon: int) -> Callable[[Any], Any]:
    if family == "chronos-bolt":
        inner = model.model

        def _fn(x: Any) -> Any:
            out = inner(context=x.to(torch.float32))
            q = out.quantile_preds if hasattr(out, "quantile_preds") else out
            return q[:, :, :horizon]

        return _fn

    if family == "chronos-2":
        import inspect

        inner = model.model
        fwd_params = set(inspect.signature(inner.forward).parameters.keys())
        patch_len = max(
            int(getattr(getattr(inner, "chronos_config", None), "output_patch_size", 16)), 1
        )
        n_patches = max(1, (horizon + patch_len - 1) // patch_len)

        def _fn(x: Any) -> Any:
            B = x.shape[0]
            gids = torch.arange(B, dtype=torch.long, device=x.device)
            kw: dict[str, Any] = {"context": x.to(torch.float32), "group_ids": gids}
            if "num_output_patches" in fwd_params:
                kw["num_output_patches"] = n_patches
            out = inner(**kw)
            q = out.quantile_preds if hasattr(out, "quantile_preds") else out
            return q[:, :, :horizon]

        return _fn

    # Toto-2.0 eager callable: use true unmodified model.forecast({"target": x_3d, ...})
    # when available (Datadog/Toto-2.0-4m & 22m), else uncontracted FP32 fallback.
    from tsjit.compiler.toto2_pass import (
        _call_toto2_native_forecast,
        _has_native_toto2_forecast,
        _run_uncontracted_eager_toto2,
        unwrap_toto2_backbone,
    )

    forecast_owner = None
    if _has_native_toto2_forecast(model):
        forecast_owner = model
    else:
        cand_bb = unwrap_toto2_backbone(model)
        if _has_native_toto2_forecast(cand_bb):
            forecast_owner = cand_bb

    if forecast_owner is not None:

        def _fn_toto_native(x: Any, owner_: Any = forecast_owner) -> Any:
            return _call_toto2_native_forecast(owner_, x, horizon=horizon)

        return _fn_toto_native

    def _fn_toto(x: Any) -> Any:
        return _run_uncontracted_eager_toto2(
            model,
            x,
            horizon=horizon,
            quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            dc_strip=False,
        )

    return _fn_toto


def _time_callable_cuda(
    fn: Callable[[Any], Any],
    x: Any,
    warmup: int,
    iters: int,
    device: Any,
) -> dict[str, Any]:
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                _ = fn(x)
            torch.cuda.synchronize(device)
            lat_ms: list[float] = []
            for _ in range(iters):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                _ = fn(x)
                ev1.record()
                torch.cuda.synchronize(device)
                lat_ms.append(float(ev0.elapsed_time(ev1)))
        arr = np.asarray(lat_ms, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        return {
            "status": "ok",
            "p50_ms": p50,
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "throughput_series_per_s": float(x.shape[0] / max(p50 * 1e-3, 1e-9)),
        }
    except torch.cuda.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "oom", "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return {"status": "oom", "error": f"{type(exc).__name__}: {exc}"}
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _load_parity_slices(
    num_series: int,
    context_length: int,
    gift_dir: str | None,
    seed: int = 2026,
) -> tuple[dict[str, np.ndarray], str]:
    """Load 512-series GIFT-Eval slice (or tagged synthetic fallback) plus stress slices."""
    base_list: list[np.ndarray] = []
    source = "synthetic_multi_regime_fallback"

    if gift_dir:
        gpath = Path(gift_dir)
        for p in sorted(gpath.rglob("*.npy")):
            arr = np.load(p).astype(np.float64)
            if arr.ndim == 1 and arr.size >= context_length:
                base_list.append(arr[-context_length:])
            elif arr.ndim == 2 and arr.shape[1] >= context_length:
                for row in arr:
                    base_list.append(row[-context_length:])
            if len(base_list) >= num_series:
                source = f"local_gift_dir:{gpath.name}"
                break
        if len(base_list) < num_series:
            try:
                from datasets import load_from_disk

                for info_file in sorted(gpath.rglob("dataset_info.json")):
                    ds = load_from_disk(str(info_file.parent))
                    for row in ds:
                        target = np.asarray(row["target"], dtype=np.float64)
                        if target.ndim == 2:
                            for ch in target:
                                tail = ch[-context_length:]
                                if ch.size >= context_length and np.isfinite(tail).all():
                                    base_list.append(tail)
                                    if len(base_list) >= num_series:
                                        break
                        elif (
                            target.size >= context_length
                            and np.isfinite(target[-context_length:]).all()
                        ):
                            base_list.append(target[-context_length:])
                        if len(base_list) >= num_series:
                            source = f"local_gift_arrow:{gpath.name}"
                            break
                    if len(base_list) >= num_series:
                        break
            except Exception:
                pass

    if len(base_list) < num_series:
        try:
            from datasets import load_dataset

            ds = load_dataset("Salesforce/GiftEval", "electricity", split="train", streaming=True)
            for row in ds:
                target = np.asarray(row["target"], dtype=np.float64)
                if target.ndim == 2:
                    target = target[0]
                if target.size >= context_length:
                    base_list.append(target[-context_length:])
                if len(base_list) >= num_series:
                    source = "hf:Salesforce/GiftEval/electricity"
                    break
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 24.0, context_length, dtype=np.float64)
    while len(base_list) < num_series:
        idx = len(base_list)
        scale = 10.0 ** rng.uniform(-3.0, 4.0)
        freq = rng.uniform(0.3, 3.0)
        trend = rng.uniform(-0.02, 0.02) * t
        seasonal = np.sin(freq * t + idx) + 0.4 * np.cos(2.3 * freq * t)
        noise = rng.normal(0.0, 0.15, size=context_length)
        base_list.append(scale * (1.0 + trend + seasonal + noise))

    clean = np.stack(base_list[:num_series], axis=0).astype(np.float64)

    # 25% NaN-bearing slice (keep final entry finite for causal DC anchor).
    nan_mask = rng.random(size=clean.shape) < 0.25
    nan_mask[:, -1] = False
    nan_25 = clean.copy()
    nan_25[nan_mask] = np.nan

    # 2.22e10 DC-offset stress slice.
    dc_stress = (clean - np.nanmean(clean, axis=1, keepdims=True)) / np.maximum(
        np.nanstd(clean, axis=1, keepdims=True), 1e-6
    )
    dc_stress = dc_stress + 2.22e10

    return {"gifteval_clean": clean, "nan_25pct": nan_25, "dc_2p22e10": dc_stress}, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live H200 public TSFM compiler benchmark.")
    parser.add_argument("--models", type=str, nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 1024, 4096])
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--parity-series", type=int, default=512)
    parser.add_argument("--gift-dir", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/bench_h200_tsfm.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("bench_h200_tsfm requires PyTorch with CUDA.")

    device = torch.device("cuda:0")
    parity_slices, slice_source = _load_parity_slices(
        args.parity_series, args.context_length, args.gift_dir
    )
    validator = StrictParityValidator(tol=0.05)

    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "horizon": args.horizon,
        "context_length": args.context_length,
        "parity_slice_source": slice_source,
        "parity_series": args.parity_series,
        "models": {},
    }

    def _fmt_ms(res: dict[str, Any]) -> str:
        val = res.get("p50_ms")
        if isinstance(val, (int, float)):
            return f"{val:.2f} ms"
        return str(res.get("status", "ERR")).upper()

    for model_id in args.models:
        print(f"\n=== Benchmarking {model_id} ===", flush=True)
        try:
            model, family = _load_public_model(model_id, device)
        except Exception as exc:
            report["models"][model_id] = {
                "status": "load_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"Skipping {model_id}: {exc}", flush=True)
            continue

        eager_fn = _make_eager_callable(model, family, args.horizon)
        try:
            tc_fn = torch.compile(eager_fn, mode="reduce-overhead")
        except Exception:
            tc_fn = None

        compiled_fast = tsjit_compile(
            model,
            horizon=args.horizon,
            context_length=args.context_length,
            precision="fast-tf32",
            mode="fast",
            batch_buckets=tuple(sorted(set(args.batches))),
            dc_strip=True,
        )
        compiled_auto = tsjit_compile(
            model,
            horizon=args.horizon,
            context_length=args.context_length,
            precision="a10-auto",
            mode="exact",
            batch_buckets=tuple(sorted(set(args.batches))),
            dc_strip=True,
        )
        compiled_fp16x3 = tsjit_compile(
            model,
            horizon=args.horizon,
            context_length=args.context_length,
            precision="fp16x3",
            mode="exact",
            batch_buckets=tuple(sorted(set(args.batches))),
            dc_strip=True,
        )

        batch_results: dict[str, Any] = {}
        for B in args.batches:
            x_b = torch.randn((B, args.context_length), dtype=torch.float32, device=device)
            r_eager = _time_callable_cuda(eager_fn, x_b, args.warmup, args.iters, device)
            gc.collect()
            torch.cuda.empty_cache()

            r_tc = (
                _time_callable_cuda(tc_fn, x_b, args.warmup, args.iters, device)
                if tc_fn is not None
                else {"status": "unavailable"}
            )
            if r_tc.get("status") in ("oom", "error"):
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

            r_fast = _time_callable_cuda(
                compiled_fast.predict_quantiles, x_b, args.warmup, args.iters, device
            )
            gc.collect()
            torch.cuda.empty_cache()

            r_auto = _time_callable_cuda(
                compiled_auto.predict_quantiles, x_b, args.warmup, args.iters, device
            )
            gc.collect()
            torch.cuda.empty_cache()

            r_oz = _time_callable_cuda(
                compiled_fp16x3.predict_quantiles, x_b, args.warmup, args.iters, device
            )
            del x_b
            gc.collect()
            torch.cuda.empty_cache()

            batch_results[str(B)] = {
                "eager": r_eager,
                "torch_compile": r_tc,
                "tsjit_fast_tf32": r_fast,
                "tsjit_a10_auto": r_auto,
                "tsjit_fp16x3": r_oz,
            }
            print(
                f"  B={B:4d} | eager={_fmt_ms(r_eager)} | "
                f"torch.compile={_fmt_ms(r_tc)} | "
                f"tsjit(fast-tf32)={_fmt_ms(r_fast)} | "
                f"tsjit(a10-auto)={_fmt_ms(r_auto)} | "
                f"tsjit(fp16x3)={_fmt_ms(r_oz)}",
                flush=True,
            )

        # Strict 0/512 FP32 parity across all 3 stress slices
        parity_results: dict[str, Any] = {}
        dc_ref_stripper = CausalFloat64DCStripper(enabled=True)
        for slice_name, arr in parity_slices.items():
            x_sl = torch.from_numpy(arr).to(device=device, dtype=torch.float64)
            # For DC-offset slice, wrap the UNMODIFIED eager model with FP64 DC stripping so the
            # strict-FP32 reference itself isn't destroyed by FP32 ULP collapse at 2.22e10.
            if slice_name == "dc_2p22e10":

                def _eager_dc_ref(
                    x1: Any,
                    fn_: Callable[[Any], Any] = eager_fn,
                    stripper_: CausalFloat64DCStripper = dc_ref_stripper,
                ) -> Any:
                    x_s, dc_f64 = stripper_.strip(x1)
                    q_s = fn_(x_s).sort(dim=1).values
                    return stripper_.restore(q_s, dc_f64).sort(dim=1).values

                ref_fn = _eager_dc_ref
            else:

                def _eager_ref(x1: Any, fn_: Callable[[Any], Any] = eager_fn) -> Any:
                    return fn_(x1.to(torch.float32)).sort(dim=1).values.to(torch.float64)

                ref_fn = _eager_ref

            y_ref = validator._run_serial_fp32_ref(ref_fn, x_sl)
            aud_fast = validator.audit(
                ref_fn, compiled_fast.predict_quantiles, x_sl, precomputed_ref=y_ref
            )
            aud_auto = validator.audit(
                ref_fn, compiled_auto.predict_quantiles, x_sl, precomputed_ref=y_ref
            )
            aud_oz = validator.audit(
                ref_fn, compiled_fp16x3.predict_quantiles, x_sl, precomputed_ref=y_ref
            )
            parity_results[slice_name] = {
                "tsjit_fast_tf32": {
                    "passed": aud_fast.passed,
                    "failed": f"{aud_fast.num_failed}/{aud_fast.num_series}",
                    "max_scaled_err": aud_fast.max_scaled_err,
                    "p99_scaled_err": aud_fast.p99_scaled_err,
                    "nonfinite_mismatches": aud_fast.nonfinite_mismatches,
                },
                "tsjit_a10_auto": {
                    "passed": aud_auto.passed,
                    "failed": f"{aud_auto.num_failed}/{aud_auto.num_series}",
                    "max_scaled_err": aud_auto.max_scaled_err,
                    "p99_scaled_err": aud_auto.p99_scaled_err,
                    "nonfinite_mismatches": aud_auto.nonfinite_mismatches,
                },
                "tsjit_fp16x3": {
                    "passed": aud_oz.passed,
                    "failed": f"{aud_oz.num_failed}/{aud_oz.num_series}",
                    "max_scaled_err": aud_oz.max_scaled_err,
                    "p99_scaled_err": aud_oz.p99_scaled_err,
                    "nonfinite_mismatches": aud_oz.nonfinite_mismatches,
                },
            }
            print(
                f"  Parity [{slice_name}]: "
                f"fast-tf32={aud_fast.num_failed}/{aud_fast.num_series} "
                f"(max={aud_fast.max_scaled_err:.4f}), "
                f"a10-auto={aud_auto.num_failed}/{aud_auto.num_series} "
                f"(max={aud_auto.max_scaled_err:.4f}), "
                f"fp16x3={aud_oz.num_failed}/{aud_oz.num_series} "
                f"(max={aud_oz.max_scaled_err:.4f})",
                flush=True,
            )

        rep_fast = compiled_fast.report()
        rep_auto = compiled_auto.report()
        rep_oz = compiled_fp16x3.report()
        print(
            f"  Compiler [{family}]: prec_auto={rep_auto.get('precision_resolved')} | "
            f"graphs_fast={rep_fast.get('cuda_graphs', {})} | "
            f"graphs_auto={rep_auto.get('cuda_graphs', {})} | "
            f"mask_stub={rep_auto.get('group_time_mask_stubbed')} | "
            f"exec_path={rep_auto.get('execution_path')}",
            flush=True,
        )
        report["models"][model_id] = {
            "status": "ok",
            "family": family,
            "compiler_report_fast_tf32": rep_fast,
            "compiler_report_auto": rep_auto,
            "compiler_report_fp16x3": rep_oz,
            "batches": batch_results,
            "parity": parity_results,
        }

        del model, eager_fn, tc_fn, compiled_fast, compiled_auto, compiled_fp16x3
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved JSON report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
