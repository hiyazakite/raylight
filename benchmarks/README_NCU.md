# FP8 Kernel Performance Profiling

## Quick Start

### 1. Run basic benchmark (no NCU)
```bash
python benchmarks/bench_fp8_vs_bf16.py
```

### 2. Run NCU profile (comprehensive)
```bash
python benchmarks/bench_fp8_ncu.py
```

### 3. Run NCU profile (quick iteration)
```bash
python benchmarks/bench_fp8_ncu.py --quick
```

### 4. Profile specific layer shape
```bash
python benchmarks/bench_fp8_ncu.py --shape 256 3072 3072
```

### 5. Analyze profile results
```bash
python benchmarks/analyze_ncu_profile.py benchmarks/ncu_profiles/profile.ncu-rep
```

### 6. View in NCU GUI
```bash
ncu benchmarks/ncu_profiles/profile.ncu-rep
```

## Output Files

- `benchmarks/ncu_profiles/profile.ncu-rep` - Full NCU profile
- `benchmarks/ncu_profiles/metrics.csv` - Extracted metrics
- Console output - Key metrics and recommendations

## Key Metrics to Watch

| Metric | Good | Bad | Action |
|--------|------|-----|--------|
| Occupancy | >50% | <30% | Reduce registers/shared memory |
| Tensor Core Util | >60% | <40% | Increase tile size, fix ldmatrix |
| L2 Hit Rate | >70% | <50% | Add pipeline stages, improve locality |
| DRAM Efficiency | >30% | <15% | Vectorize writes, coalesce access |

## Typical Bottlenecks & Fixes

### Compute-bound (Tensor Core Util < 60%)
- **Problem**: Dequant ALU overhead too high
- **Fix**: Pre-scale weights during repack, eliminate runtime dequant

### Memory-bound (L2 Hit Rate < 70%)
- **Problem**: Not enough data reuse
- **Fix**: Increase pipeline stages (3→4), larger tiles

### Low Occupancy (< 30%)
- **Problem**: Too many registers/shared memory per block
- **Fix**: Use BF16 accumulators, reduce tile size

## Hybrid Approach

If kernel is slower than cuBLAS for large matrices:

```python
# In fp8_ampere_linear.py:257
CUBLAS_THRESHOLD = 10_000_000  # 10M elements

def _forward_marlin(self, x, ext):
    total_elements = x.shape[0] * self.out_features * self.in_features
    
    if total_elements > CUBLAS_THRESHOLD:
        # cuBLAS is faster for large matrices
        return self._forward_fallback(x)
    
    # Use custom kernel for smaller matrices
    ...
```

Adjust threshold based on your profile results.
