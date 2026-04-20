"""Synthetic benchmark: FP8 IMMA kernel vs naive BF16-upcast path.

Measures three execution strategies for a linear layer whose weights live
in FP8 E4M3 on-device (the default ComfyUI safetensors load format):

  baseline   : weight.view(float8) → .to(bfloat16) → F.linear(x, w_bf16)
               Mirrors exactly what ComfyUI's LowVramPatch does today on Ampere.

  fp8_kernel : fp8_ampere_mm(x_bf16, weight_u8, weight_scale)
               Our INT8 IMMA kernel with LUT-transcode and dynamic act quant.

  fp8_fallback: Fp8AmpereLinear without the CUDA extension (BF16 F.linear after
               FP8 decode).  Shows that the module itself isn't the bottleneck
               when the kernel is not built.

Metrics per configuration (M × K × N):
  - Median latency (ms)
  - Throughput (TFLOPS effective, using BF16 FLOP count 2·M·K·N)
  - Peak VRAM delta (MB): extra memory allocated during the call vs at rest
  - Speedup vs baseline

Usage:
    python benchmarks/bench_fp8_vs_bf16.py [--csv]
"""

import os
import sys
import argparse
import gc
import statistics
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from raylight.distributed_modules.fp8_ampere.packing import quantize_to_fp8_e4m3
from raylight.distributed_modules.fp8_ampere.fp8_ampere_linear import (
    Fp8AmpereLinear,
    _is_ampere_or_newer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 20
ITERS  = 100

# (M=batch_tokens, K=in_features, N=out_features, label)
# Shapes inspired by Flux-dev attention / MLP layers and video-model token counts.
SHAPES = [
    # (M,    K,     N,     label)
    (  256,  3072,  3072,  "attn_proj  (256 tok)"),
    ( 1024,  3072,  3072,  "attn_proj  (1k tok)"),
    ( 4096,  3072,  3072,  "attn_proj  (4k tok)"),
    (  256,  3072, 12288,  "mlp_up     (256 tok)"),
    ( 1024,  3072, 12288,  "mlp_up     (1k tok)"),
    ( 4096,  3072, 12288,  "mlp_up     (4k tok)"),
    (  256, 12288,  3072,  "mlp_down   (256 tok)"),
    ( 1024, 12288,  3072,  "mlp_down   (1k tok)"),
    ( 4096, 12288,  3072,  "mlp_down   (4k tok)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_time(fn, warmup: int, iters: int) -> list[float]:
    """Return per-iteration wall-time in ms (CUDA-synchronised)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _peak_vram_delta_mb(fn) -> float:
    """Extra VRAM allocated during a single call (MB)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 1024 ** 2


def _tflops(M: int, K: int, N: int, ms: float) -> float:
    flops = 2 * M * K * N
    return flops / (ms * 1e-3) / 1e12


def _median(xs):
    return statistics.median(xs)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_baseline(x_bf16: torch.Tensor, weight_fp8_u8: torch.Tensor) -> dict:
    """weight.view(float8_e4m3fn) → .to(bfloat16) → F.linear.  Mirrors LowVramPatch."""
    def fn():
        w_bf16 = weight_fp8_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
        return F.linear(x_bf16, w_bf16)

    times = _sync_time(fn, WARMUP, ITERS)
    vram  = _peak_vram_delta_mb(fn)
    return {"times": times, "vram_mb": vram}


def bench_fp8_kernel(
    x_bf16: torch.Tensor,
    weight_fp8_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    ext,
) -> dict:
    """fp8_ampere_mm — INT8 IMMA kernel."""
    def fn():
        return ext.fp8_ampere_mm(x_bf16, weight_fp8_u8, weight_scale)

    times = _sync_time(fn, WARMUP, ITERS)
    vram  = _peak_vram_delta_mb(fn)
    return {"times": times, "vram_mb": vram}


def bench_fp8_fallback(fp8_lin: Fp8AmpereLinear, x_bf16: torch.Tensor) -> dict:
    """Fp8AmpereLinear._forward_fallback — BF16 F.linear after FP8 decode."""
    def fn():
        return fp8_lin._forward_fallback(x_bf16)

    times = _sync_time(fn, WARMUP, ITERS)
    vram  = _peak_vram_delta_mb(fn)
    return {"times": times, "vram_mb": vram}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_mode: bool = False):
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}  SM {props.major}.{props.minor}  "
          f"VRAM {props.total_memory // 1024**3} GiB")

    if not _is_ampere_or_newer(device):
        print("WARNING: Not an Ampere GPU — IMMA path will not be used by the module, "
              "but the benchmark will still run the kernel directly.")

    # Try to load the CUDA extension
    ext = None
    try:
        from raylight.distributed_modules.fp8_ampere import _C_fp8ampere as _ext
        ext = _ext
        print("Extension: _C_fp8ampere loaded ✓")
    except ImportError:
        print("Extension: _C_fp8ampere NOT built — kernel benchmark skipped")

    print(f"Warmup={WARMUP}  Iterations={ITERS}\n")

    sep = "-" * 108
    header = (
        f"{'Shape':<28}  {'Method':<18}  "
        f"{'Median(ms)':>10}  {'p99(ms)':>9}  "
        f"{'TFLOPS':>8}  {'VRAM Δ(MB)':>10}  {'vs baseline':>11}"
    )

    if csv_mode:
        print("shape,M,K,N,method,median_ms,p99_ms,tflops,vram_delta_mb,speedup_vs_baseline")
    else:
        print(sep)
        print(header)
        print(sep)

    for (M, K, N, label) in SHAPES:
        # Enforce K divisible by 64 (kernel requirement)
        K_aligned = ((K + 63) // 64) * 64
        N_aligned = N  # N doesn't need alignment for WMMA

        torch.manual_seed(42)
        x_bf16        = torch.randn(M, K_aligned, dtype=torch.bfloat16, device=device)
        weight_float  = torch.randn(N_aligned, K_aligned, dtype=torch.bfloat16, device=device)
        weight_fp8_u8, weight_scale = quantize_to_fp8_e4m3(weight_float)

        # Build Fp8AmpereLinear for fallback path
        fp8_lin = Fp8AmpereLinear.from_fp8_weight(
            weight_fp8_u8, weight_scale, bias=None
        ).to(device)

        results = {}

        # ---- baseline ----
        r = bench_baseline(x_bf16, weight_fp8_u8)
        results["baseline"] = r

        # ---- fp8 kernel ----
        if ext is not None:
            r = bench_fp8_kernel(x_bf16, weight_fp8_u8, weight_scale, ext)
            results["fp8_kernel"] = r

        # ---- fp8 fallback ----
        r = bench_fp8_fallback(fp8_lin, x_bf16)
        results["fp8_fallback"] = r

        baseline_median = _median(results["baseline"]["times"])

        for method, r in results.items():
            times    = r["times"]
            med      = _median(times)
            p99      = sorted(times)[int(0.99 * len(times))]
            tf       = _tflops(M, K_aligned, N_aligned, med)
            vram     = r["vram_mb"]
            speedup  = baseline_median / med

            if csv_mode:
                print(
                    f"{label},{M},{K_aligned},{N_aligned},{method},"
                    f"{med:.4f},{p99:.4f},{tf:.3f},{vram:.1f},{speedup:.3f}"
                )
            else:
                print(
                    f"{label:<28}  {method:<18}  "
                    f"{med:>10.3f}  {p99:>9.3f}  "
                    f"{tf:>8.3f}  {vram:>10.1f}  {speedup:>10.2f}×"
                )

        if not csv_mode:
            print()

        # Free tensors explicitly to keep VRAM tidy between shapes
        del x_bf16, weight_float, weight_fp8_u8, weight_scale, fp8_lin
        gc.collect()
        torch.cuda.empty_cache()

    if not csv_mode:
        print(sep)
        print("Notes:")
        print("  TFLOPS uses 2·M·K·N (same as BF16 reference, for fair comparison).")
        print("  VRAM Δ = peak_allocated − base_allocated during a single call.")
        print("  baseline = weight.view(fp8).to(bf16) + F.linear  (current ComfyUI path)")
        print("  fp8_kernel = fp8_ampere_mm INT8 IMMA  (this work)")
        print("  fp8_fallback = Fp8AmpereLinear without CUDA ext (BF16 F.linear)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP8 kernel vs BF16-upcast benchmark")
    parser.add_argument("--csv", action="store_true", help="Output CSV instead of table")
    args = parser.parse_args()
    run(csv_mode=args.csv)
