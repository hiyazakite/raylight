"""Micro-benchmark: fused mmvq/mmq vs dequant+cuBLAS for realistic shapes.

Measures the two paths head-to-head to identify where time is spent.
Also benchmarks scratch-buffer vs fresh-alloc in ggml_dequant_mm.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F
import gguf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from raylight.expansion.comfyui_gguf.fused_kernels import (
    HAS_FUSED_GGUF_CUDA,
    fused_ggml_gemm,
    _C,
)
from raylight.expansion.comfyui_gguf.dequant import dequantize as pytorch_dequantize

assert HAS_FUSED_GGUF_CUDA, "Need _C_gguf compiled"
assert torch.cuda.is_available()

# Typical diffusion model layer dims (Flux-like)
SHAPES = [
    (3072, 3072,  "attn_proj"),
    (12288, 3072, "mlp_up"),
    (3072, 12288, "mlp_down"),
    (768, 768,    "small_proj"),
]

QTYPES = [
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q8_0,
]

BATCH_SIZES = [1, 2, 4, 8, 16]

WARMUP = 5
ITERS = 50


def make_quant_weight(out_dim, in_dim, qtype, rng):
    w = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    qw_bytes = gguf.quants.quantize(w, qtype)
    return torch.from_numpy(np.frombuffer(qw_bytes, dtype=np.uint8)).cuda()


def bench_fused(qw, x, qtype, shape, warmup=WARMUP, iters=ITERS):
    """Benchmark the fused mmvq/mmq path."""
    for _ in range(warmup):
        fused_ggml_gemm(x, qw, qtype, shape)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fused_ggml_gemm(x, qw, qtype, shape)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms


def bench_dequant_cublas(qw_cpu, x, qtype, shape, warmup=WARMUP, iters=ITERS):
    """Benchmark the old path: PyTorch dequant (CPU) + cuBLAS F.linear."""
    for _ in range(warmup):
        w = pytorch_dequantize(qw_cpu, qtype, shape).to(device="cuda", dtype=x.dtype)
        F.linear(x, w)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        w = pytorch_dequantize(qw_cpu, qtype, shape).to(device="cuda", dtype=x.dtype)
        F.linear(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_cuda_dequant_cublas(qw_cuda, x, qtype, shape, warmup=WARMUP, iters=ITERS):
    """Benchmark: CUDA dequant + cuBLAS F.linear (what dequant.py fast-path does)."""
    m, n = shape
    type_int = qtype.value
    for _ in range(warmup):
        w = _C.ggml_dequantize(qw_cuda, type_int, m, n, x.dtype)
        F.linear(x, w)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        w = _C.ggml_dequantize(qw_cuda, type_int, m, n, x.dtype)
        F.linear(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_cublas_only(qw_cpu, x, qtype, shape, warmup=WARMUP, iters=ITERS):
    """Benchmark just cuBLAS F.linear with pre-dequantized weight (ceiling)."""
    w = pytorch_dequantize(qw_cpu, qtype, shape).to(device="cuda", dtype=x.dtype)
    for _ in range(warmup):
        F.linear(x, w)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        F.linear(x, w)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    rng = np.random.default_rng(42)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print()
    
    header = f"{'Shape':>16} {'QType':>6} {'Batch':>5} │ {'Fused':>8} {'CUDADeq+cuB':>12} {'PyDeq+cuB':>10} {'cuBLAS':>8} │ {'Fused/CUDADeq':>13}"
    print(header)
    print("─" * len(header))
    
    for out_dim, in_dim, name in SHAPES:
        for qtype in QTYPES:
            qw_cuda = make_quant_weight(out_dim, in_dim, qtype, rng)
            qw_cpu = qw_cuda.cpu()
            shape = (out_dim, in_dim)
            
            for batch in BATCH_SIZES:
                x = torch.randn(batch, in_dim, device="cuda", dtype=torch.float16)
                
                t_fused = bench_fused(qw_cuda, x, qtype, shape)
                t_cuda_deq = bench_cuda_dequant_cublas(qw_cuda, x, qtype, shape)
                t_py_deq = bench_dequant_cublas(qw_cpu, x, qtype, shape)
                t_cublas = bench_cublas_only(qw_cpu, x, qtype, shape)
                
                ratio = t_fused / t_cuda_deq
                marker = "✓" if ratio < 1.0 else "✗"
                
                print(
                    f"{out_dim}x{in_dim:>5} ({name[:8]:>8}) "
                    f"{qtype.name:>6} {batch:>5} │ "
                    f"{t_fused:>7.3f}ms {t_cuda_deq:>11.3f}ms {t_py_deq:>9.3f}ms {t_cublas:>7.3f}ms │ "
                    f"{ratio:>6.2f}x {marker}"
                )
            print()


# ---------------------------------------------------------------------------
# Scratch-buffer benchmark — scratch (zero-alloc) vs fresh alloc in ggml_dequant_mm
# ---------------------------------------------------------------------------

SCRATCH_SHAPES = [
    # (out_dim, in_dim, label)
    (3072,  3072,  "attn_proj"),
    (12288, 3072,  "mlp_up"),
    (3072,  12288, "mlp_down"),
    (4096,  4096,  "large_proj"),
    (6144,  4096,  "xl_up"),
    (256,   256,   "tiny"),
]

# Realistic token counts: Flux ~4k, Wan ~14k, HunyuanVideo ~42k
SCRATCH_BATCHES = [8, 256, 4096, 14000]

SCRATCH_QTYPES = [
    gguf.GGMLQuantizationType.Q4_0,
    gguf.GGMLQuantizationType.Q8_0,
]


def bench_dequant_mm(qw, x, type_int, rows, cols, bias, scratch, warmup=WARMUP, iters=ITERS):
    """Benchmark ggml_dequant_mm with or without scratch."""
    for _ in range(warmup):
        _C.ggml_dequant_mm(qw, x, type_int, rows, cols, bias, scratch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _C.ggml_dequant_mm(qw, x, type_int, rows, cols, bias, scratch)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main_scratch():
    rng = np.random.default_rng(42)

    print(f"\n{'='*80}")
    print(f"SCRATCH BUFFER BENCHMARK — scratch (zero-alloc) vs fresh alloc")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP}, Iters: {ITERS}")
    print()

    header = (
        f"{'Shape':>16} {'QType':>5} {'Batch':>5} │ "
        f"{'NoScratch':>10} {'Scratch':>8} {'cuBLAS':>8} │ "
        f"{'Speedup':>8}"
    )
    print(header)
    print("─" * len(header))

    for out_dim, in_dim, name in SCRATCH_SHAPES:
        for qtype in SCRATCH_QTYPES:
            qw_cuda = make_quant_weight(out_dim, in_dim, qtype, rng)
            rows, cols = out_dim, in_dim
            type_int = qtype.value
            total = rows * cols

            scratch = torch.empty(total, dtype=torch.float16, device="cuda")

            # Pre-dequantized weight for cuBLAS-only ceiling
            w_fp16 = _C.ggml_dequantize(qw_cuda, type_int, rows, cols, torch.float16)

            for batch in SCRATCH_BATCHES:
                x = torch.randn(batch, in_dim, device="cuda", dtype=torch.float16)

                # No scratch: allocates fresh buffer each call
                t_alloc = bench_dequant_mm(
                    qw_cuda, x, type_int, rows, cols, None, None,
                )

                # With scratch: zero-alloc path
                t_scratch = bench_dequant_mm(
                    qw_cuda, x, type_int, rows, cols, None, scratch,
                )

                # cuBLAS-only ceiling (no dequant at all)
                for _ in range(WARMUP):
                    torch.mm(x, w_fp16.t())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    torch.mm(x, w_fp16.t())
                torch.cuda.synchronize()
                t_cublas = (time.perf_counter() - t0) / ITERS * 1000

                speedup = t_alloc / t_scratch if t_scratch > 0 else float('inf')
                marker = "✓" if speedup > 1.02 else ("─" if speedup > 0.98 else "✗")

                print(
                    f" {out_dim}x{in_dim:>5} ({name[:8]:>8}) "
                    f"{qtype.name:>5} {batch:>5} │ "
                    f"{t_alloc:>9.3f}ms {t_scratch:>7.3f}ms {t_cublas:>7.3f}ms │ "
                    f"{speedup:>6.2f}x {marker}"
                )
            print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", action="store_true",
                        help="Run scratch-buffer benchmark only")
    parser.add_argument("--all", action="store_true",
                        help="Run both original and scratch benchmarks")
    args = parser.parse_args()

    if args.scratch:
        main_scratch()
    elif args.all:
        main()
        main_scratch()
    else:
        main()
