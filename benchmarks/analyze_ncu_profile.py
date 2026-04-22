"""Analyze NCU profile and suggest kernel optimizations.

Reads a .ncu-rep file and outputs:
  1. Key performance metrics
  2. Bottleneck classification (memory-bound vs compute-bound)
  3. Specific optimization recommendations

Usage:
    python benchmarks/analyze_ncu_profile.py benchmarks/ncu_profiles/profile.ncu-rep
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_ncu_csv_export(rep_file: Path, output_csv: Path):
    """Export NCU metrics to CSV format."""
    cmd = [
        "ncu",
        "--import",
        "ncu-rep",
        "--export",
        str(output_csv),
        "--import-source",
        "yes",
        "--csv",
        str(rep_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error exporting NCU data: {result.stderr}")
        sys.exit(1)


def parse_metrics(csv_file: Path) -> dict[str, float]:
    """Parse CSV and extract key metrics."""
    metrics = {}

    with open(csv_file) as f:
        for line in f:
            # Skip header
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    value = float(parts[1])
                    metrics[parts[0]] = value
                except ValueError:
                    pass

    return metrics


def classify_bottleneck(metrics: dict[str, float]) -> str:
    """Classify whether kernel is memory-bound or compute-bound."""
    # Check Speed of Light metrics
    compute_bound_threshold = 0.7  # If compute efficiency > 70%, compute-bound
    memory_bound_threshold = 0.8  # If memory efficiency > 80%, memory-bound

    compute_efficiency = metrics.get("smspc__throughput_fp32_tensor_cores_percent", 0)
    memory_efficiency = metrics.get("smspc__throughput_l2_read_throughput_percent", 0)

    if compute_efficiency > compute_bound_threshold:
        return "COMPUTE-BOUND"
    elif memory_efficiency > memory_bound_threshold:
        return "MEMORY-BOUND"
    else:
        return "MIXED"


def get_recommendations(bottleneck: str, metrics: dict[str, float]) -> list[str]:
    """Generate optimization recommendations based on bottleneck."""
    recommendations = []

    if bottleneck == "COMPUTE-BOUND":
        # Check register usage
        registers_per_thread = metrics.get("smspc__registers_per_thread_avg", 0)
        if registers_per_thread > 64:
            recommendations.append(
                f"HIGH REGISTER USAGE ({registers_per_thread:.0f} regs/thread):\n"
                "  - Consider using BF16 accumulators instead of FP32 (halves register count)\n"
                "  - Vectorize accumulator storage (float2 instead of float)"
            )

        # Check tensor core utilization
        wmma_util = metrics.get("smspc__smsp__inst_exec_cmplx_tensor_cores_ratio", 0)
        if wmma_util < 0.6:
            recommendations.append(
                f"LOW TENSOR CORE UTILIZATION ({wmma_util * 100:.1f}%):\n"
                "  - Increase tile size (BLOCK_M, BLOCK_N) to reduce kernel launch overhead\n"
                "  - Verify ldmatrix loads are not stalling (check ldmatrix latency)\n"
                "  - Consider warp-specialization (separate copy/compute warps)"
            )

        # Check instruction mix
        mma_insts = metrics.get("smspc__inst_exec_cmplx_tensor_cores_total", 0)
        alu_insts = metrics.get("smspc__inst_exec_cmplx_fp32_total", 0)
        if mma_insts > 0 and alu_insts / mma_insts > 0.5:
            recommendations.append(
                "HIGH ALU OVERHEAD relative to MMA:\n"
                "  - Dequant function is too expensive - consider pre-scaling weights\n"
                "  - Fuse dequant with weight loading (load BF16 directly from repacked data)\n"
                "  - Use simpler quantization (INT8 instead of FP8 E4M3)"
            )

    elif bottleneck == "MEMORY-BOUND":
        # Check L2 cache hit rate
        l2_hit_rate = metrics.get("smspc__l2_cache_hit_rate_throughput", 0)
        if l2_hit_rate < 0.7:
            recommendations.append(
                f"LOW L2 HIT RATE ({l2_hit_rate * 100:.1f}%):\n"
                "  - Increase shared memory staging (add more pipeline stages)\n"
                "  - Improve data locality with better tiling\n"
                "  - Consider prefetching next tile earlier"
            )

        # Check DRAM throughput
        dram_throughput = metrics.get("smspc__dram_throughput_bytes_per_second", 0)
        peak_dram = 20e12  # Approximate peak for A100 (20 TB/s)
        dram_efficiency = dram_throughput / peak_dram if peak_dram > 0 else 0

        if dram_efficiency < 0.3:
            recommendations.append(
                f"LOW DRAM EFFICIENCY ({dram_efficiency * 100:.1f}%):\n"
                "  - Increase vectorization (use bf16x4 instead of bf16x2 writes)\n"
                "  - Ensure coalesced access patterns\n"
                "  - Pad shared memory to avoid bank conflicts"
            )

        # Check shared memory usage
        smem_usage = metrics.get("smspc__shared_memory_per_block_bytes", 0)
        if smem_usage > 48000:
            recommendations.append(
                f"HIGH SHARED MEMORY USAGE ({smem_usage:.0f} bytes):\n"
                "  - Reduce pipeline stages (3 -> 2) to free shared memory\n"
                "  - Use smaller tile sizes\n"
                "  - Consider using registers for scale array"
            )

    else:  # MIXED
        recommendations.append(
            "MIXED BOTTLENECK - Multiple factors:\n"
            "  - Profile individual kernel launches for specific layer types\n"
            "  - Consider hybrid approach: use cuBLAS for large matrices, custom kernel for small\n"
            "  - Tune tile sizes per layer type (attention vs MLP)"
        )

    # General recommendations
    occupancy = metrics.get("smspc__occupancy_percent", 0)
    if occupancy < 50:
        recommendations.append(
            f"LOW OCCUPANCY ({occupancy:.1f}%):\n"
            "  - Reduce register usage (see above)\n"
            "  - Reduce shared memory per block\n"
            "  - Increase block size (256 -> 512 threads)"
        )

    return recommendations


def print_analysis(rep_file: Path):
    """Run full analysis and print results."""
    csv_file = Path("/tmp/ncu_metrics.csv")

    print(f"\n{'=' * 70}")
    print(f"NCU PROFILE ANALYSIS: {rep_file.name}")
    print(f"{'=' * 70}\n")

    # Export to CSV
    print("Exporting NCU data...")
    run_ncu_csv_export(rep_file, csv_file)

    # Parse metrics
    metrics = parse_metrics(csv_file)

    # Print key metrics
    print("\nKEY METRICS:")
    print("-" * 70)

    metric_names = {
        "smspc__occupancy_percent": "Occupancy",
        "smspc__registers_per_thread_avg": "Registers per thread",
        "smspc__shared_memory_per_block_bytes": "Shared memory per block",
        "smspc__smsp__inst_exec_cmplx_tensor_cores_ratio": "Tensor core utilization",
        "smspc__l2_cache_hit_rate_throughput": "L2 hit rate",
        "smspc__dram_throughput_bytes_per_second": "DRAM throughput (B/s)",
        "smspc__throughput_fp32_tensor_cores_percent": "Compute efficiency",
    }

    for key, label in metric_names.items():
        if key in metrics:
            value = metrics[key]
            if "throughput" in key and "bytes" in key:
                print(f"{label:35} {value / 1e12:8.2f} TB/s")
            elif "rate" in key or "ratio" in key or "percent" in key:
                print(f"{label:35} {value * 100:8.2f}%")
            elif "bytes" in key:
                print(f"{label:35} {value:8.0f} bytes")
            else:
                print(f"{label:35} {value:8.2f}")

    # Classify bottleneck
    bottleneck = classify_bottleneck(metrics)
    print(f"\n{'=' * 70}")
    print(f"BOTTLENECK: {bottleneck}")
    print(f"{'=' * 70}\n")

    # Print recommendations
    recommendations = get_recommendations(bottleneck, metrics)

    print("OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 70)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec}")

    if not recommendations:
        print("No specific recommendations - kernel appears well-optimized!")

    print(f"\n{'=' * 70}")
    print(f"View full profile: ncu {rep_file}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze NCU profile and suggest optimizations")
    parser.add_argument("profile", type=Path, help="Path to .ncu-rep profile file")

    args = parser.parse_args()

    if not args.profile.exists():
        print(f"Error: Profile file not found: {args.profile}")
        sys.exit(1)

    print_analysis(args.profile)


if __name__ == "__main__":
    main()
