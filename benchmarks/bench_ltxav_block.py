"""Simulated LTXAV-22B block benchmark: FP8 IMMA kernel vs BF16-upcast baseline.

Simulates a full forward pass through ONE BasicAVTransformerBlock of the
LTX 2.3-22B Audio-Video model.  Instead of isolated GEMM microbenchmarks,
this measures the realistic cost of executing ALL linear layers in a block
consecutively — matching the actual execution pattern during inference.

Model architecture (48 blocks, per block):
  Video self-attn (4 layers):  to_q/k/v/out  (4096, 4096)  — FP8
  Video cross-attn (4 layers): to_q/k/v/out  (4096, 4096)  — FP8
  Video FFN (2 layers):        up (16384, 4096), down (4096, 16384)  — FP8
  Audio self-attn (4 layers):  to_q/k/v/out  (2048, 2048)  — BF16
  Audio cross-attn (4 layers): to_q/k/v/out  (2048, 2048)  — BF16
  Audio FFN (2 layers):        up (8192, 2048), down (2048, 8192)  — BF16

Only the 10 FP8 layers are candidates for the IMMA kernel.
Audio layers stay BF16 regardless — they're included for total block timing.

Token counts from real LTXAV inference:
  512×768, 97 frames (~4s @ 25fps):   ~5k video tokens, ~640 audio tokens
  768×512, 161 frames (~6.5s):        ~8k video tokens, ~1k audio tokens
  768×512, 257 frames (~10s):         ~13k video tokens, ~1.6k audio tokens

Usage:
    python benchmarks/bench_ltxav_block.py [--csv]
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

from raylight.distributed_modules.fp8_ampere.packing import (
    quantize_to_fp8_e4m3,
    compute_fp8_int8_scales,
)
from raylight.distributed_modules.fp8_ampere.fp8_ampere_linear import (
    _is_ampere_or_newer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 10
ITERS  = 50   # fewer iters since each measures a full block

# (video_tokens, audio_tokens, label)
# Typical LTXAV latent sequence lengths for common resolutions.
SCENARIOS = [
    (  3328,   640, "512×512 97f (~4s)"),
    (  4992,   640, "512×768 97f (~4s)"),
    (  8064,  1050, "768×512 161f (~6.5s)"),
    ( 12672,  1600, "768×512 257f (~10s)"),
]

# All linear layers in one BasicAVTransformerBlock.
# (name, out_features, in_features, is_fp8, "video" | "audio")
# gate_logits (32, 4096) are too small to matter — omitted.
BLOCK_LAYERS = [
    # Video self-attention
    ("v_self_q",     4096, 4096, True,  "video"),
    ("v_self_k",     4096, 4096, True,  "video"),
    ("v_self_v",     4096, 4096, True,  "video"),
    ("v_self_out",   4096, 4096, True,  "video"),
    # Video cross-attention
    ("v_cross_q",    4096, 4096, True,  "video"),
    ("v_cross_k",    4096, 4096, True,  "video"),
    ("v_cross_v",    4096, 4096, True,  "video"),
    ("v_cross_out",  4096, 4096, True,  "video"),
    # Video FFN
    ("v_ffn_up",    16384, 4096, True,  "video"),
    ("v_ffn_down",   4096,16384, True,  "video"),
    # Audio self-attention
    ("a_self_q",     2048, 2048, False, "audio"),
    ("a_self_k",     2048, 2048, False, "audio"),
    ("a_self_v",     2048, 2048, False, "audio"),
    ("a_self_out",   2048, 2048, False, "audio"),
    # Audio cross-attention
    ("a_cross_q",    2048, 2048, False, "audio"),
    ("a_cross_k",    2048, 2048, False, "audio"),
    ("a_cross_v",    2048, 2048, False, "audio"),
    ("a_cross_out",  2048, 2048, False, "audio"),
    # Audio FFN
    ("a_ffn_up",     8192, 2048, False, "audio"),
    ("a_ffn_down",   2048, 8192, False, "audio"),
]

NUM_BLOCKS = 48


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_time(fn, warmup: int, iters: int) -> list[float]:
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


def _median(xs):
    return statistics.median(xs)


def _peak_vram_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 1024 ** 2


def _total_flops(video_M, audio_M):
    """Total FLOPs for one block = Σ 2·M·K·N for each layer."""
    total = 0
    for name, N, K, is_fp8, stream in BLOCK_LAYERS:
        M = video_M if stream == "video" else audio_M
        total += 2 * M * K * N
    return total


# ---------------------------------------------------------------------------
# Block simulation: baseline (LowVramPatch style)
# ---------------------------------------------------------------------------

def make_block_baseline(video_M, audio_M, device):
    """Build weight tensors and a callable that simulates one full block.

    Baseline path: each FP8 layer does weight.view(fp8).to(bf16) + F.linear.
    BF16 layers do plain F.linear.
    """
    weights = {}
    activations = {}

    for name, N, K, is_fp8, stream in BLOCK_LAYERS:
        M = video_M if stream == "video" else audio_M
        K_aligned = ((K + 63) // 64) * 64  # kernel needs K%64==0
        torch.manual_seed(hash(name) & 0x7FFFFFFF)
        w = torch.randn(N, K_aligned, dtype=torch.bfloat16, device=device)
        if is_fp8:
            # Store as uint8 (FP8 bit-patterns) — matches safetensors load
            w_fp8 = w.to(torch.float8_e4m3fn)
            weights[name] = w_fp8.view(torch.uint8)
        else:
            weights[name] = w

        # Pre-allocate activation — in reality these flow from previous layers,
        # but for timing purposes we just need the right shape and dtype.
        if f"x_{stream}" not in activations:
            activations[f"x_{stream}"] = torch.randn(
                M, K_aligned if K_aligned == K else K_aligned,
                dtype=torch.bfloat16, device=device,
            )

    # Pre-allocate contiguous activations for each unique (stream, K) pair
    act_cache = {}
    for name, N, K, is_fp8, stream in BLOCK_LAYERS:
        K_aligned = ((K + 63) // 64) * 64
        key = (stream, K_aligned)
        if key not in act_cache:
            M = video_M if stream == "video" else audio_M
            act_cache[key] = torch.randn(M, K_aligned, dtype=torch.bfloat16, device=device)

    def run_block():
        for name, N, K, is_fp8, stream in BLOCK_LAYERS:
            K_aligned = ((K + 63) // 64) * 64
            x = act_cache[(stream, K_aligned)]
            w = weights[name]
            if is_fp8:
                w_bf16 = w.view(torch.float8_e4m3fn).to(torch.bfloat16)
                F.linear(x, w_bf16)
            else:
                F.linear(x, w)

    return run_block


# ---------------------------------------------------------------------------
# Block simulation: FP8 IMMA kernel for FP8 layers, cuBLAS for BF16 layers
# ---------------------------------------------------------------------------

def make_block_fp8_kernel(video_M, audio_M, device, ext):
    """FP8 layers use fp8_ampere_mm; BF16 layers use F.linear."""
    weights = {}
    scales = {}

    for name, N, K, is_fp8, stream in BLOCK_LAYERS:
        K_aligned = ((K + 63) // 64) * 64
        torch.manual_seed(hash(name) & 0x7FFFFFFF)
        w = torch.randn(N, K_aligned, dtype=torch.bfloat16, device=device)
        if is_fp8:
            w_u8, w_scale = quantize_to_fp8_e4m3(w)
            # Kernel expects [K, N] transposed layout
            weights[name] = w_u8.t().contiguous()
            scales[name] = w_scale
        else:
            weights[name] = w

    # Pre-allocate contiguous activations for each unique (stream, K) pair
    act_cache = {}
    for name, N, K, is_fp8, stream in BLOCK_LAYERS:
        K_aligned = ((K + 63) // 64) * 64
        key = (stream, K_aligned)
        if key not in act_cache:
            M = video_M if stream == "video" else audio_M
            act_cache[key] = torch.randn(M, K_aligned, dtype=torch.bfloat16, device=device)

    def run_block():
        for name, N, K, is_fp8, stream in BLOCK_LAYERS:
            K_aligned = ((K + 63) // 64) * 64
            x = act_cache[(stream, K_aligned)]
            if is_fp8:
                ext.fp8_ampere_mm(x, weights[name], scales[name])
            else:
                F.linear(x, weights[name])

    return run_block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_mode: bool = False):
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}  SM {props.major}.{props.minor}  "
          f"VRAM {props.total_memory // 1024**3} GiB")

    ext = None
    try:
        from raylight.distributed_modules.fp8_ampere import _C_fp8ampere as _ext
        ext = _ext
        print("Extension: _C_fp8ampere loaded ✓")
    except ImportError:
        print("Extension: _C_fp8ampere NOT built — kernel benchmark skipped")

    print(f"Warmup={WARMUP}  Iterations={ITERS}")
    print(f"Model: LTXAV-22B  Blocks: {NUM_BLOCKS}")
    print()

    sep = "-" * 118
    header = (
        f"{'Scenario':<26}  {'Method':<18}  "
        f"{'Block(ms)':>10}  {'Full model':>12}  "
        f"{'TFLOPS':>8}  {'VRAM Δ(MB)':>10}  {'vs baseline':>11}"
    )

    if csv_mode:
        print("scenario,video_M,audio_M,method,block_median_ms,model_48x_ms,tflops,vram_delta_mb,speedup")
    else:
        print(sep)
        print(header)
        print(sep)

    for video_M, audio_M, label in SCENARIOS:
        torch.cuda.empty_cache()
        gc.collect()

        total_flops = _total_flops(video_M, audio_M)

        results = {}

        # Baseline
        fn_base = make_block_baseline(video_M, audio_M, device)
        times_base = _sync_time(fn_base, WARMUP, ITERS)
        vram_base = _peak_vram_mb(fn_base)
        results["baseline"] = {"times": times_base, "vram_mb": vram_base}
        del fn_base
        torch.cuda.empty_cache()

        # FP8 kernel
        if ext is not None:
            fn_kern = make_block_fp8_kernel(video_M, audio_M, device, ext)
            times_kern = _sync_time(fn_kern, WARMUP, ITERS)
            vram_kern = _peak_vram_mb(fn_kern)
            results["fp8_kernel"] = {"times": times_kern, "vram_mb": vram_kern}
            del fn_kern
            torch.cuda.empty_cache()

        baseline_med = _median(results["baseline"]["times"])

        for method, r in results.items():
            times = r["times"]
            med    = _median(times)
            p99    = sorted(times)[int(0.99 * len(times))]
            model_ms = med * NUM_BLOCKS
            tf     = total_flops / (med * 1e-3) / 1e12
            vram   = r["vram_mb"]
            speedup = baseline_med / med

            if csv_mode:
                print(
                    f"{label},{video_M},{audio_M},{method},"
                    f"{med:.3f},{model_ms:.1f},{tf:.2f},{vram:.1f},{speedup:.3f}"
                )
            else:
                model_str = f"{model_ms / 1000:.2f}s" if model_ms >= 1000 else f"{model_ms:.0f}ms"
                print(
                    f"{label:<26}  {method:<18}  "
                    f"{med:>10.3f}  {model_str:>12}  "
                    f"{tf:>8.2f}  {vram:>10.1f}  {speedup:>10.2f}×"
                )

        if not csv_mode:
            # FP8-only layers breakdown
            fp8_count = sum(1 for _, _, _, is_fp8, _ in BLOCK_LAYERS if is_fp8)
            bf16_count = len(BLOCK_LAYERS) - fp8_count
            fp8_flops = sum(
                2 * (video_M if s == "video" else audio_M) * K * N
                for _, N, K, is_fp8, s in BLOCK_LAYERS if is_fp8
            )
            print(f"  ↳ {fp8_count} FP8 layers ({fp8_flops/1e9:.1f} GFLOP), "
                  f"{bf16_count} BF16 layers ({(total_flops-fp8_flops)/1e9:.1f} GFLOP)")
            print()

    if not csv_mode:
        print(sep)
        print("Notes:")
        print(f"  Block = {len(BLOCK_LAYERS)} linear layers executed sequentially (no attention compute).")
        print(f"  Full model ≈ block × {NUM_BLOCKS} (one diffusion step, excludes attention/norms/etc).")
        print("  TFLOPS = total 2·M·K·N / block_time.")
        print("  baseline = weight.view(fp8).to(bf16) + F.linear per FP8 layer (ComfyUI LowVramPatch).")
        print("  fp8_kernel = fp8_ampere_mm for FP8 layers, F.linear for BF16 layers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    run(csv_mode=args.csv)
