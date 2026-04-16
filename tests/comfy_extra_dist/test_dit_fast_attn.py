"""Tests for the CacheDiT lightweight cache (replaces old test_dit_fast_attn.py)."""
import os
import sys
import types
import torch

# Stub out comfy imports so the module loads without a full ComfyUI install
for name in ("comfy", "comfy.model_patcher", "comfy.patcher_extension"):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []  # Mark as package so sub-imports work
        sys.modules[name] = mod

# Provide the minimal stubs needed
import comfy.model_patcher as _cmp
import comfy.patcher_extension as _cpe
if not hasattr(_cmp, "create_model_options_clone"):
    _cmp.create_model_options_clone = lambda x: dict(x)
if not hasattr(_cpe, "WrappersMP"):
    class _WMP:
        OUTER_SAMPLE = "outer_sample"
        DIFFUSION_MODEL = "diffusion_model"
    _cpe.WrappersMP = _WMP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from raylight.comfy_extra_dist.nodes_cachedit import (
    _enable_lightweight_cache,
    _cleanup_transformer_cache,
    _get_lightweight_cache_stats,
    _lightweight_cache_state,
)

print("✅ Imports successful!")

# ── Build a tiny fake transformer ──────────────────────────────────────────────
class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_log = []

    def forward(self, x):
        self.call_log.append("compute")
        return x * 2.0

dummy = DummyTransformer()

# ── Test: skip_interval=2 (compute on calls 1,3,5,…; skip on 2,4,6,…) ──────
print("\nTesting warmup=2, skip_interval=2 over 8 calls:")
_enable_lightweight_cache(dummy, warmup_steps=2, skip_interval=2, noise_scale=0.0)

x = torch.ones(1, 4)
for i in range(1, 9):
    result = dummy(x)
    state_snap = _get_lightweight_cache_stats()
    action = "COMPUTE" if result is not x * 2.0 or i <= 2 else "SKIP"
    # Determine from call log changes
    n_compute = len(dummy.call_log)

import torch
_enable_lightweight_cache(dummy, warmup_steps=2, skip_interval=2, noise_scale=0.0)
compute_calls = []
x = torch.ones(1, 4)
for i in range(1, 9):
    before = len(dummy.call_log)
    result = dummy(x)
    after = len(dummy.call_log)
    computed = after > before
    print(f"  Call {i}: {'COMPUTE' if computed else 'CACHE  '}")

stats = _get_lightweight_cache_stats()
print(f"\nStats: total={stats['total_steps']}, computed={stats['computed_steps']}, "
      f"cached={stats['cached_steps']}, hit_rate={stats['cache_hit_rate']:.1f}%, "
      f"speedup={stats['estimated_speedup']:.2f}x")

assert stats["computed_steps"] == 5, f"Expected 5 compute steps, got {stats['computed_steps']}"
assert stats["cached_steps"] == 3, f"Expected 3 cached steps, got {stats['cached_steps']}"
print("✅ skip_interval=2 logic correct")

# ── Test: cleanup restores original forward ──────────────────────────────────
_cleanup_transformer_cache(dummy)
assert not hasattr(dummy, "_cachedit_original_forward"), "original_forward sentinel should be removed"
assert _lightweight_cache_state["enabled"] is False, "Global state should be disabled after cleanup"
print("✅ Cleanup restores original forward correctly")

print("\n✅ All CacheDiT lightweight cache tests passed.")
