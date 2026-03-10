import sys
import os

# Add custom_nodes to path to allow importing raylight directly
sys.path.insert(0, "/root/ComfyUI/custom_nodes")

class DummyTracker:
    def __init__(self):
        self.step = -1
        self.total_steps = -1
    def update(self, s):
        self.step = s
    def reset(self):
        self.step = 0

# Bypass __init__ imports by loading files directly for testing
from raylight.distributed_modules.temporal_cache import DenoisingStepTracker, CachedAttentionWrapper

print("✅ Imports successful!")
tracker = DenoisingStepTracker()

print("Verifying skip logic (1 compute, 1 skip)...")
for step in range(6):
    tracker.current_step = step
    should = tracker.should_cache(skip_interval=1, start_pct=0.0, end_pct=1.0)
    print(f"Step {step}: {'COMPUTE' if not should else 'CACHE'}")

print("\nVerifying skip logic with 20% start/end bounds (Total=10)...")
tracker.total_steps = 10
for step in range(10):
    tracker.current_step = step
    should = tracker.should_cache(skip_interval=1, start_pct=0.2, end_pct=0.8)
    print(f"Step {step} (Progress {step/10}): {'COMPUTE' if not should else 'CACHE'}")
