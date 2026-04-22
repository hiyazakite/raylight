"""Roofline analysis for fp8_ampere_mm on RTX 3090.

Computes arithmetic intensity for each benchmark shape, overlays measured
TFLOPS from bench_fp8_vs_bf16.py, and classifies each kernel as memory-bound
or compute-bound relative to the RTX 3090 roofline.

RTX 3090 hardware limits:
  Peak BF16 TensorCore (dense):  71.0 TFLOPS
  Peak DRAM bandwidth:          936.2 GB/s
  Ridge point:                   71e12 / 936e9 ≈ 75.8 FLOP/byte

Kernel memory traffic (fp8_ampere_mm):
  Read  A:  M*K * 2 bytes  (BF16 activations)
  Read  B:  K*N * 1 byte   (FP8/INT8 weights)
  Read  s:  N   * 2 bytes  (BF16 per-column scales, one per output col)
  Write C:  M*N * 2 bytes  (BF16 output)

Baseline memory traffic (view(fp8).to(bf16) + F.linear):
  Read  W_fp8: K*N * 1 byte    (read FP8 weights once for upcast)
  Write W_bf16: K*N * 2 bytes  (write upcasted BF16 weights to VRAM)
  Read  W_bf16: K*N * 2 bytes  (read again for F.linear)
  Read  A:  M*K * 2 bytes  (BF16 activations)
  Write C:  M*N * 2 bytes  (BF16 output)
  Total: K*N*(1+2+2) + M*K*2 + M*N*2 = K*N*5 + M*(K+N)*2
  (In practice PyTorch may fuse the view+to, so a lower-bound is 3×K*N)

Usage:
    python benchmarks/roofline.py
    python benchmarks/roofline.py --plot      # requires matplotlib
"""

import argparse

# ---------------------------------------------------------------------------
# RTX 3090 hardware constants
# ---------------------------------------------------------------------------
PEAK_TFLOPS_BF16  = 71.0          # dense BF16 TensorCore, TFLOPS
PEAK_BW_GBS       = 936.2         # GDDR6X peak memory bandwidth, GB/s
RIDGE_FLOP_BYTE   = (PEAK_TFLOPS_BF16 * 1e12) / (PEAK_BW_GBS * 1e9)  # ~75.8

# ---------------------------------------------------------------------------
# Shapes: (M, K, N, label)
# From bench_fp8_vs_bf16.py SHAPES list
# ---------------------------------------------------------------------------
SHAPES = [
    ( 128, 3072, 3072, "attn_proj 128tok"),
    ( 256, 3072, 3072, "attn_proj 256tok"),
    ( 512, 3072, 3072, "attn_proj 512tok"),
    (1024, 3072, 3072, "attn_proj 1k tok"),
    ( 128, 3072,12288, "mlp_up    128tok"),
    ( 256, 3072,12288, "mlp_up    256tok"),
    ( 128,12288, 3072, "mlp_down  128tok"),
    ( 256,12288, 3072, "mlp_down  256tok"),
]

# Measured median TFLOPS from most recent bench_fp8_vs_bf16.py run (STAGES=3).
# Update these if you re-run the benchmark.
MEASURED = {
    # label: (baseline_tflops, fp8_kernel_tflops)
    "attn_proj 128tok": (18.44, 16.88),
    "attn_proj 256tok": (29.15, 27.54),
    "attn_proj 512tok": (31.19, 33.83),
    "attn_proj 1k tok": (49.29, 48.39),
    "mlp_up    128tok": (22.27, 21.88),
    "mlp_up    256tok": (28.49, 47.28),
    "mlp_down  128tok": (22.32, 17.87),
    "mlp_down  256tok": (32.80, 35.33),
}


def arithmetic_intensity_kernel(M: int, K: int, N: int) -> float:
    """Arithmetic intensity for fp8_ampere_mm (FP8 weights, BF16 activations/output).

    Traffic:
      Read  A (BF16):     M*K*2
      Read  B (FP8/INT8): K*N*1
      Read  s (BF16):     N*2      (per-column scale)
      Write C (BF16):     M*N*2
    """
    flops = 2 * M * K * N
    bytes_a = M * K * 2
    bytes_b = K * N * 1
    bytes_s = N * 2
    bytes_c = M * N * 2
    total_bytes = bytes_a + bytes_b + bytes_s + bytes_c
    return flops / total_bytes


def arithmetic_intensity_baseline(M: int, K: int, N: int) -> float:
    """Arithmetic intensity for view(fp8).to(bf16) + F.linear baseline.

    Approximate traffic (conservative — assumes upcasted weight not cached):
      Read  W_fp8  (FP8): K*N*1
      Write W_bf16 (BF16): K*N*2   (materialize upcasted weight)
      Read  W_bf16 (BF16): K*N*2   (F.linear reads it)
      Read  A (BF16):     M*K*2
      Write C (BF16):     M*N*2
    """
    flops = 2 * M * K * N
    bytes_w = K * N * (1 + 2 + 2)  # read fp8 + write bf16 + read bf16
    bytes_a = M * K * 2
    bytes_c = M * N * 2
    total_bytes = bytes_w + bytes_a + bytes_c
    return flops / total_bytes


def roofline_peak(ai: float) -> float:
    """Predicted peak TFLOPS from roofline model given arithmetic intensity."""
    memory_bound_peak = ai * PEAK_BW_GBS / 1e3   # TFLOPS
    return min(memory_bound_peak, PEAK_TFLOPS_BF16)


def classify(ai: float) -> str:
    if ai < RIDGE_FLOP_BYTE * 0.8:
        return "MEM-BOUND"
    elif ai < RIDGE_FLOP_BYTE * 1.2:
        return "RIDGE"
    else:
        return "COMPUTE-BOUND"


def run(plot: bool = False):
    print(f"RTX 3090 Roofline")
    print(f"  Peak BF16 TensorCore : {PEAK_TFLOPS_BF16:.1f} TFLOPS")
    print(f"  Peak DRAM BW         : {PEAK_BW_GBS:.1f} GB/s")
    print(f"  Ridge point          : {RIDGE_FLOP_BYTE:.1f} FLOP/byte")
    print()

    col = "{:<20}  {:>8}  {:>9}  {:>9}  {:>7}  {:>11}  {:>11}  {:>9}  {:>9}  {:>9}"
    hdr = col.format(
        "Shape", "AI(kern)", "AI(base)", "Roof(TFLOP)",
        "Bound?",
        "Base(TFLOP)", "Kern(TFLOP)",
        "Base eff%", "Kern eff%", "Kern/Base"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    plot_data = []

    for M, K, N, label in SHAPES:
        ai_k  = arithmetic_intensity_kernel(M, K, N)
        ai_b  = arithmetic_intensity_baseline(M, K, N)
        roof  = roofline_peak(ai_k)
        bound = classify(ai_k)

        base_tf, kern_tf = MEASURED.get(label, (None, None))

        base_eff = f"{base_tf / PEAK_TFLOPS_BF16 * 100:.1f}%" if base_tf else "N/A"
        kern_eff = f"{kern_tf / PEAK_TFLOPS_BF16 * 100:.1f}%" if kern_tf else "N/A"
        kern_of_roof = f"{kern_tf / roof * 100:.1f}%" if kern_tf else "N/A"
        speedup  = f"{kern_tf / base_tf:.2f}×" if (kern_tf and base_tf) else "N/A"

        print(col.format(
            label,
            f"{ai_k:.1f}",
            f"{ai_b:.1f}",
            f"{roof:.1f}",
            bound,
            f"{base_tf:.2f}" if base_tf else "N/A",
            f"{kern_tf:.2f}" if kern_tf else "N/A",
            base_eff,
            kern_of_roof,
            speedup,
        ))

        if base_tf and kern_tf:
            plot_data.append((label, ai_k, ai_b, roof, base_tf, kern_tf))

    print(sep)
    print()
    print("Columns:")
    print("  AI(kern)    = arithmetic intensity of fp8_ampere_mm  [FLOP/byte]")
    print("  AI(base)    = arithmetic intensity of baseline upcast [FLOP/byte]")
    print("  Roof(TFLOP) = roofline ceiling for kernel AI          [TFLOPS]")
    print("  Bound?      = MEM-BOUND if AI < 0.8×ridge, COMPUTE-BOUND if AI > 1.2×ridge")
    print("  Base eff%   = baseline TFLOPS / peak BF16 TensorCore")
    print("  Kern eff%   = kernel TFLOPS / roofline ceiling  (hardware efficiency)")
    print("  Kern/Base   = measured speedup")
    print()

    # Print bottleneck summary
    print("Bottleneck analysis:")
    for M, K, N, label in SHAPES:
        ai_k = arithmetic_intensity_kernel(M, K, N)
        roof = roofline_peak(ai_k)
        _, kern_tf = MEASURED.get(label, (None, None))
        if kern_tf is None:
            continue
        gap_to_roof = (roof - kern_tf) / roof * 100
        bound = classify(ai_k)
        if bound == "MEM-BOUND":
            limiting = f"BW ceiling={roof:.1f} TFLOPS, gap={gap_to_roof:.0f}%"
        else:
            limiting = f"Compute ceiling={PEAK_TFLOPS_BF16:.1f} TFLOPS, gap={gap_to_roof:.0f}%"
        print(f"  {label:<20}  [{bound:^12}]  achieved={kern_tf:.1f}  roof={roof:.1f}  gap={gap_to_roof:.0f}%")

    if plot:
        _plot(plot_data)


def _plot(data):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n[plot] matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline
    ai_range = np.logspace(-1, 3, 500)
    roof_line = np.minimum(ai_range * PEAK_BW_GBS / 1e3, PEAK_TFLOPS_BF16)
    ax.loglog(ai_range, roof_line, "k-", linewidth=2, label="RTX 3090 roofline (BF16 TC)")
    ax.axvline(RIDGE_FLOP_BYTE, color="gray", linestyle="--", alpha=0.5, label=f"Ridge ({RIDGE_FLOP_BYTE:.0f} FLOP/B)")

    colors = plt.cm.tab10.colors
    for i, (label, ai_k, ai_b, roof, base_tf, kern_tf) in enumerate(data):
        c = colors[i % len(colors)]
        ax.scatter([ai_b], [base_tf], marker="o", color=c, s=80, zorder=5)
        ax.scatter([ai_k], [kern_tf], marker="^", color=c, s=80, zorder=5)
        ax.annotate(label, (ai_k, kern_tf), textcoords="offset points", xytext=(4, 4), fontsize=7, color=c)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", label="baseline", linestyle="None"),
        Line2D([0], [0], marker="^", color="gray", label="fp8_kernel", linestyle="None"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:2], fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Roofline: fp8_ampere_mm vs baseline — RTX 3090")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, 500)
    ax.set_ylim(1, 100)

    outpath = "benchmarks/ncu_profiles/roofline.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"\n[plot] Saved to {outpath}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roofline analysis for fp8_ampere_mm")
    parser.add_argument("--plot", action="store_true", help="Save roofline plot (requires matplotlib)")
    args = parser.parse_args()
    run(plot=args.plot)
