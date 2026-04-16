import os
import json
import sys
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from raylight.config import (
    RaylightConfig, 
    ExecutionStrategy, 
    CompactConfig, 
    ClusterConfig, 
    DeviceConfig, 
    DebugConfig, 
    SystemConfig, 
    RaylightAttnType, 
    CompactCompressType
)

def test_config_defaults():
    print("Running test_config_defaults...")
    config = RaylightConfig()
    
    assert config.strategy.ulysses_degree == 1
    assert config.strategy.ring_degree == 1
    assert config.strategy.attention_backend == "STANDARD"
    assert config.strategy.attention_type == RaylightAttnType.TORCH
    
    assert config.compact.enabled is False
    assert config.compact.warmup_steps == 1
    
    assert config.cluster.address == "local"
    assert config.cluster.namespace == "default"
    
    assert config.device.use_mmap is True
    assert config.device.vram_limit_gb == 0.0
    
    assert config.debug.verbose_attn is False
    assert config.debug.overlap_decomp is True
    print("  PASS")

def test_config_validation():
    print("Running test_config_validation...")
    # This should be valid
    RaylightConfig(
        strategy=ExecutionStrategy(ulysses_degree=2, ring_degree=1),
        device=DeviceConfig(world_size=2)
    )
    
    # This should raise ValueError: total_sp_degree (4) > world_size (2)
    try:
        RaylightConfig(
            strategy=ExecutionStrategy(ulysses_degree=2, ring_degree=2),
            device=DeviceConfig(world_size=2)
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot exceed available world size" in str(e)
    print("  PASS")

def test_config_from_env():
    print("Running test_config_from_env...")
    os.environ["RAYLIGHT_VERBOSE_ATTN"] = "1"
    os.environ["RAYLIGHT_OVERLAP_DECOMP"] = "0"
    os.environ["MASTER_ADDR"] = "10.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    
    try:
        config = RaylightConfig.from_env()
        assert config.debug.verbose_attn is True
        assert config.debug.overlap_decomp is False
        assert config.system.master_addr == "10.0.0.1"
        assert config.system.master_port == "1234"
    finally:
        # Cleanup
        for k in ["RAYLIGHT_VERBOSE_ATTN", "RAYLIGHT_OVERLAP_DECOMP", "MASTER_ADDR", "MASTER_PORT"]:
            if k in os.environ:
                del os.environ[k]
    print("  PASS")

def test_legacy_dict_mapping():
    print("Running test_legacy_dict_mapping...")
    config = RaylightConfig(
        strategy=ExecutionStrategy(ulysses_degree=2, attention_backend="COMPACT"),
        device=DeviceConfig(world_size=4, vram_limit_gb=8.5)
    )
    
    legacy = config.to_legacy_dict()
    
    assert legacy["ulysses_degree"] == 2
    assert legacy["attention_backend"] == "COMPACT"
    assert legacy["global_world_size"] == 4
    assert legacy["vram_limit_bytes"] == int(8.5 * 1024**3)
    assert legacy["is_xdit"] is True
    print("  PASS")

def test_serialization():
    print("Running test_serialization...")
    config = RaylightConfig(
        strategy=ExecutionStrategy(attention_type=RaylightAttnType.FLASH_ATTN)
    )
    
    # JSON Verification
    js = config.to_json()
    data = json.loads(js)
    assert data["strategy"]["attention_type"] == "FLASH_ATTN"
    
    # YAML-like Verification
    yml = config.to_yaml()
    assert "attention_type: FLASH_ATTN" in yml
    assert "ulysses_degree: 1" in yml
    print("  PASS")

def test_apply_to_env():
    print("Running test_apply_to_env...")
    config = RaylightConfig(
        system=SystemConfig(master_addr="1.2.3.4", python_gil=False)
    )
    
    # Backup
    orig_addr = os.environ.get("MASTER_ADDR")
    orig_gil = os.environ.get("PYTHON_GIL")
    
    try:
        config.apply_to_env()
        assert os.environ["MASTER_ADDR"] == "1.2.3.4"
        assert os.environ["PYTHON_GIL"] == "0"
    finally:
        # Restore
        if orig_addr: os.environ["MASTER_ADDR"] = orig_addr
        if orig_gil: os.environ["PYTHON_GIL"] = orig_gil
    print("  PASS")

if __name__ == "__main__":
    try:
        test_config_defaults()
        test_config_validation()
        test_config_from_env()
        test_legacy_dict_mapping()
        test_serialization()
        test_apply_to_env()
        print("\nALL CONFIG TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
