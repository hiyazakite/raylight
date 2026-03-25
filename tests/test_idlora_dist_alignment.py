import torch
import pytest
from unittest.mock import MagicMock
from raylight.comfy_extra_dist.idlora_patches import ensure_patches_applied

# Mock ComfyUI components
import sys
from unittest.mock import patch

def test_idlora_mask_and_seq_alignment():
    """Verify that prepending tokens to ax works with sharding."""
    ensure_patches_applied()
    
    from comfy.ldm.lightricks.av_model import LTXAVModel
    
    # Create a mock model
    model = LTXAVModel(
        in_channels=128, audio_in_channels=128,
        num_layers=1, num_attention_heads=4,
        attention_head_dim=32, audio_num_attention_heads=4,
        audio_attention_head_dim=16
    ).to("cpu")
    
    # Mock some parameters to avoid init issues
    model.audio_patchify_proj = MagicMock(side_effect=lambda x: x)
    model.a_patchifier = MagicMock()
    model.a_patchifier.hop_length = 160
    model.a_patchifier.audio_latent_downsample_factor = 1
    model.a_patchifier.sample_rate = 16000
    model.a_patchifier.start_end = False
    model.a_patchifier._get_audio_latent_time_in_sec = MagicMock(return_value=torch.zeros(10))
    model.a_patchifier.patchify = MagicMock(side_effect=lambda ax: (ax, torch.zeros(1, 1, ax.shape[1], 1)))
    
    # Input: [vx, ax]
    vx = torch.randn(1, 100, 128)
    ax = torch.randn(1, 50, 64)
    ref_audio = {"tokens": torch.randn(1, 10, 64)}
    
    # Test _process_input
    [vx_out, ax_out], [v_coords, a_coords], additional_args = model._process_input(
        [vx, ax], None, None, ref_audio=ref_audio
    )
    
    assert ax_out.shape[1] == 50 + 10, "ax should have prepended tokens"
    assert additional_args["ref_audio_seq_len"] == 10
    
    # Simulate sp_chunk_group (sharding)
    # world_size = 2
    shard_0 = ax_out[:, :30]
    shard_1 = ax_out[:, 30:]
    
    assert shard_0.shape[1] == 30
    assert shard_1.shape[1] == 30
    
    # Check _prepare_timestep
    v_ts = torch.randn(1, 1)
    a_ts = torch.randn(1, 50, 128)
    
    # Mock parent calls
    model.adaln_single = MagicMock(return_value=(torch.zeros(1, 100, 128), torch.zeros(1, 100, 128)))
    model.prompt_adaln_single = MagicMock()
    model.audio_adaln_single = MagicMock(return_value=(torch.zeros(1, 60, 128), torch.zeros(1, 60, 128)))
    model.av_ca_audio_scale_shift_adaln_single = MagicMock(return_value=(torch.zeros(1, 1), None))
    model.av_ca_video_scale_shift_adaln_single = MagicMock(return_value=(torch.zeros(1, 1), None))
    model.av_ca_a2v_gate_adaln_single = MagicMock(return_value=(torch.zeros(1, 1), None))
    model.av_ca_v2a_gate_adaln_single = MagicMock(return_value=(torch.zeros(1, 1), None))
    
    # We need to simulate the environment where additional_args are available
    res_ts, res_emb, _ = model._prepare_timestep(
        v_ts, 1, torch.float32, a_timestep=a_ts, **additional_args
    )
    
    # Audio timestep should now include the 10 zeros
    # LTXAV returns [v_ts, a_ts, ...]
    a_ts_out = res_ts[1]
    assert a_ts_out.shape[1] == 60, f"a_timestep should be extended, got {a_ts_out.shape}"
    
    print("Test passed: Sequence alignment looks correct.")

if __name__ == "__main__":
    test_idlora_mask_and_seq_alignment()
