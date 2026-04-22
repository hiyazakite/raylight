"""NCU profiling wrapper for FP8 kernel benchmark.

Runs the FP8 vs BF16 benchmark under Nsight Compute to collect:
  - Occupancy (active warps per SM, registers per thread, shared memory usage)
  - Speed of Light (roofline analysis: memory-bound vs compute-bound)
  - Tensor Core utilization (mma instructions issued vs theoretical max)
  - Memory workload (L1/L2 cache hit rates, global memory throughput)
  - Instruction mix (ALU vs tensor core vs memory ops)

Usage:
    # Full profile (slow, comprehensive)
    python benchmarks/bench_fp8_ncu.py

    # Quick profile (only key metrics)
    python benchmarks/bench_fp8_ncu.py --quick

    # Profile specific shape
    python benchmarks/bench_fp8_ncu.py --shape 256 3072 3072

Output:
    Prints key metrics to stdout and saves .ncu-rep files in benchmarks/ncu_profiles/
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ---------------------------------------------------------------------------
# NCU configuration
# ---------------------------------------------------------------------------

NCU_DIR = Path(__file__).parent / "ncu_profiles"
NCU_DIR.mkdir(exist_ok=True)

# Full section list for comprehensive profiling
FULL_SECTIONS = [
    "Occupancy",
    "SpeedOfLight",
    "TensorCores",
    "MemoryWorkloadAnalysis",
    "LaunchTiming",
    "SchedulerStats",
    "InstructionStats",
    "L1Cache",
    "L2Cache",
    "DRAM",
]

# Quick sections for fast iteration
QUICK_SECTIONS = [
    "SpeedOfLight",
    "Occupancy",
    "MemoryWorkloadAnalysis",
]


def build_ncu_command(
    script: str,
    sections: list[str],
    shape: tuple[int, int, int] | None = None,
    quick: bool = False,
) -> list[str]:
    """Build the ncu command line."""
    cmd = [
        "ncu",
        "--set",
        "full" if not quick else "quick",
        "--import-source",
        "yes",
        "--target-processes",
        "all",
        "--launch-skip",
        "0",
        "--launch-count",
        "1",
    ]

    for section in sections:
        cmd.extend(["--section", section])

    cmd.extend(
        [
            "--export",
            str(NCU_DIR / "profile.ncu-rep"),
            "python3.13",
            script,
        ]
    )

    if shape is not None:
        M, K, N = shape
        cmd.extend(["--shape", str(M), str(K), str(N)])

    return cmd


def parse_ncu_output(rep_file: Path) -> dict[str, float | str]:
    """Extract key metrics from .ncu-rep file using ncu --import."""
    metrics = {}

    # Use ncu to export metrics as CSV
    csv_file = NCU_DIR / "metrics.csv"

    cmd = [
        "ncu",
        "--import",
        "ncu-rep",
        "--export",
        str(csv_file),
        "--import-source",
        "yes",
        str(rep_file),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error parsing NCU output: {e.stderr}")
        return metrics

    # Parse the CSV for key metrics
    if csv_file.exists():
        with open(csv_file) as f:
            for line in f:
                if "smspc_" in line.lower() or "throughput" in line.lower():
                    # Extract metric name and value
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        metrics[parts[0]] = parts[1]

    return metrics


def print_profile_summary(metrics: dict[str, float | str]):
    """Print a human-readable summary of profile results."""
    print("\n" + "=" * 60)
    print("NCU PROFILE SUMMARY")
    print("=" * 60)

    # Occupancy
    if "smspc__occupancy_percent" in metrics:
        print(f"\nOccupancy: {metrics['smspc__occupancy_percent']}%")

    # Tensor Core utilization
    if "smspc__tensor_cores_exec_half_throughput_avg" in metrics:
        print(f"Tensor Core throughput: {metrics['smspc__tensor_cores_exec_half_throughput_avg']}")

    # Memory
    if "smspc__l2_cache_hit_rate_throughput" in metrics:
        print(f"L2 hit rate: {metrics['smspc__l2_cache_hit_rate_throughput']}")

    if "smspc__dram_throughput" in metrics:
        print(f"DRAM throughput: {metrics['smspc__dram_throughput']}")

    print("\n" + "=" * 60)


def run_profile(quick: bool = False, shape: tuple[int, int, int] | None = None):
    """Run the benchmark under NCU."""
    script = str(Path(__file__).parent / "bench_fp8_vs_bf16.py")

    sections = QUICK_SECTIONS if quick else FULL_SECTIONS

    cmd = build_ncu_command(script, sections, shape, quick)

    print(f"Running NCU profile (quick={quick})...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            rep_file = NCU_DIR / "profile.ncu-rep"
            if rep_file.exists():
                print(f"\nProfile saved to: {rep_file}")
                metrics = parse_ncu_output(rep_file)
                print_profile_summary(metrics)

                # Also open in NCU GUI if on desktop
                if os.environ.get("DISPLAY"):
                    print(f"\nTo view in NCU GUI: ncu {rep_file}")
    except subprocess.CalledProcessError as e:
        print(f"NCU failed with return code {e.returncode}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Profile FP8 kernel with Nsight Compute")
    parser.add_argument("--quick", action="store_true", help="Run quick profile (fewer sections, faster)")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "K", "N"), help="Profile specific shape (e.g., --shape 256 3072 3072)")
    parser.add_argument("--view", action="store_true", help="Open profile in NCU GUI after running")

    args = parser.parse_args()

    run_profile(quick=args.quick, shape=tuple(args.shape) if args.shape else None)

    if args.view:
        rep_file = NCU_DIR / "profile.ncu-rep"
        if rep_file.exists():
            subprocess.run(["ncu", str(rep_file)])


if __name__ == "__main__":
    main()
