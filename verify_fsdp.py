import sys
import os
sys.path.append("/root/ComfyUI/custom_nodes/raylight/src")
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Mock raylight modules
sys.modules["raylight.distributed_modules.utils"] = MagicMock()
sys.modules["raylight.distributed_modules.quantize"] = MagicMock()
sys.modules["raylight.distributed_modules.kitchen_patch"] = MagicMock()

# Import the module to test
from raylight.diffusion_models.flux.fsdp import shard_model_fsdp2

def test_shard_logic():
    print("Testing shard_model_fsdp2 logic...")
    
    # Mock Model
    model = MagicMock()
    diffusion_model = MagicMock()
    model.diffusion_model = diffusion_model
    
    # Mock Blocks
    block = nn.Linear(10, 10)
    diffusion_model.single_blocks = [block]
    diffusion_model.double_blocks = [block]
    
    # Mock FSDP dependencies
    with patch("raylight.diffusion_models.flux.fsdp.fully_shard") as mock_shard, \
         patch("raylight.diffusion_models.flux.fsdp.checkpoint_wrapper") as mock_ckpt, \
         patch("raylight.diffusion_models.flux.fsdp.dist") as mock_dist, \
         patch("raylight.diffusion_models.flux.fsdp.set_model_state_dict") as mock_set_sd:
         
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
         
        # Test Default
        shard_model_fsdp2(model, {}, False)
        print("Default run passed.")
        
        # Test Optimizations
        os.environ["RAYLIGHT_FSDP_GRADIENT_CHECKPOINTING"] = "1"
        shard_model_fsdp2(model, {}, False, patcher=MagicMock())
        print("Optimized run passed.")
        
        # Check assertions
        assert mock_shard.call_count > 0, "fully_shard should be called"
        assert mock_ckpt.call_count > 0, "checkpoint_wrapper should be called"

if __name__ == "__main__":
    test_shard_logic()
