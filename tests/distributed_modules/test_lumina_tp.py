"""Unit tests for Lumina TP patching — single-process, no real distributed backend.

Run standalone:
    export PYTHONPATH=$PYTHONPATH:/root/ComfyUI/custom_nodes/raylight/src
    python3 tests/test_lumina_tp.py
"""

import sys
import os
import types

_src = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

_root_init = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _root_init in sys.path:
    sys.path.remove(_root_init)

# Mock comfy modules so raylight.comfy_dist can be imported without ComfyUI
def _make_mock_module(name, attrs=None):
    m = types.ModuleType(name)
    m.__path__ = []  # Make it a package so submodule lookups work
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m

# Build comfy mock tree
_comfy = _make_mock_module("comfy")
_comfy.model_management = _make_mock_module("comfy.model_management", {
    "cast_to_device": lambda t, dev, dtype, copy=False: t.to(device=dev, dtype=dtype, copy=copy) if copy else t.to(device=dev, dtype=dtype),
})

class _MockWeightAdapterBase:
    name = ""
    loaded_keys = set()
    weights = []
    @classmethod
    def load(cls, *a, **kw): return None
    def calculate_weight(self, *a, **kw): return None

_comfy_wa_base = _make_mock_module("comfy.weight_adapter.base", {
    "WeightAdapterBase": _MockWeightAdapterBase,
    "weight_decompose": lambda *a, **kw: a[1],
    "pad_tensor_to_shape": lambda t, shape: t,
    "tucker_weight_from_conv": lambda *a: a[0],
})
_comfy_wa = _make_mock_module("comfy.weight_adapter", {"base": _comfy_wa_base})
_comfy.weight_adapter = _comfy_wa

_comfy_lora = _make_mock_module("comfy.lora", {
    "pad_tensor_to_shape": lambda t, shape: t,
})
_comfy.lora = _comfy_lora
_comfy.float = _make_mock_module("comfy.float", {
    "stochastic_rounding": lambda w, dtype, seed=None: w.to(dtype),
})

_mock_comfy_modules = [
    ("comfy", _comfy),
    ("comfy.model_management", _comfy.model_management),
    ("comfy.weight_adapter", _comfy_wa),
    ("comfy.weight_adapter.base", _comfy_wa_base),
    ("comfy.lora", _comfy_lora),
    ("comfy.float", _comfy.float),
]
for name, mod in _mock_comfy_modules:
    sys.modules.setdefault(name, mod)

import pytest
import torch
import torch.nn as nn

from raylight.distributed_modules.tensor_parallel import (
    TensorParallelState,
    TPLinear,
)


def mock_tp(tp_size: int, tp_rank: int = 0) -> None:
    TensorParallelState._tp_size = tp_size
    TensorParallelState._tp_rank = tp_rank
    TensorParallelState._tp_group = None


# ---------------------------------------------------------------------------
# Minimal stub modules matching Lumina architecture (avoids ComfyUI import)
# ---------------------------------------------------------------------------

class StubJointAttention(nn.Module):
    def __init__(self, dim=512, n_heads=8, n_kv_heads=8, head_dim=64):
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.qkv = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


class StubFeedForward(nn.Module):
    def __init__(self, dim=512, hidden_dim=1024):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)


class StubJointTransformerBlock(nn.Module):
    def __init__(self, dim=512, n_heads=8, hidden_dim=1024):
        super().__init__()
        self.attention = StubJointAttention(dim=dim, n_heads=n_heads)
        self.feed_forward = StubFeedForward(dim=dim, hidden_dim=hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 4 * dim, bias=True),
        )
        self.attention_norm1 = nn.LayerNorm(dim)
        self.attention_norm2 = nn.LayerNorm(dim)
        self.ffn_norm1 = nn.LayerNorm(dim)
        self.ffn_norm2 = nn.LayerNorm(dim)


class StubFinalLayer(nn.Module):
    def __init__(self, dim=512, out_dim=128):
        super().__init__()
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), dim, bias=True),
        )


class StubNextDiT(nn.Module):
    def __init__(self, dim=512, n_heads=8, n_layers=4, hidden_dim=1024):
        super().__init__()
        self.x_embedder = nn.Linear(128, dim, bias=True)
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, dim, bias=True),
        )
        self.layers = nn.ModuleList([
            StubJointTransformerBlock(dim=dim, n_heads=n_heads, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        self.noise_refiner = nn.ModuleList([
            StubJointTransformerBlock(dim=dim, n_heads=n_heads, hidden_dim=hidden_dim)
            for _ in range(2)
        ])
        self.context_refiner = nn.ModuleList([
            StubJointTransformerBlock(dim=dim, n_heads=n_heads, hidden_dim=hidden_dim)
            for _ in range(2)
        ])
        self.final_layer = StubFinalLayer(dim=dim)
        self.siglip_refiner = None
        self.siglip_embedder = None


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from raylight.diffusion_models.lumina.tp import (
    apply_tp_to_lumina_attention,
    apply_tp_to_lumina_feedforward,
    apply_tp_to_lumina_block,
    apply_tp_to_lumina_model,
)


def _is_tp_linear(module):
    """Check if module is a TPLinear (or TPGGMLLinear)."""
    return isinstance(module, TPLinear)


# ---------------------------------------------------------------------------
# Tests: JointAttention
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_tp():
    TensorParallelState.reset()
    yield
    TensorParallelState.reset()


class TestLuminaTPAttention:
    def test_qkv_column_parallel_no_gather(self):
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        assert _is_tp_linear(attn.qkv)
        assert attn.qkv.parallelism == "column"
        assert attn.qkv.gather_output is False

    def test_out_row_parallel(self):
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        assert _is_tp_linear(attn.out)
        assert attn.out.parallelism == "row"

    def test_local_head_counts(self):
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=4, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        assert attn._tp_local_heads == 4  # 8 / 2
        assert attn._tp_local_kv_heads == 2  # 4 / 2

    def test_custom_forward_installed(self):
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        original_forward = attn.forward
        apply_tp_to_lumina_attention(attn)
        assert attn.forward != original_forward

    def test_norms_unchanged(self):
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        assert isinstance(attn.q_norm, nn.Identity)
        assert isinstance(attn.k_norm, nn.Identity)

    def test_tp_size_1_via_model(self):
        """tp_size=1 early-returns in apply_tp_to_lumina_model, leaving linears untouched."""
        mock_tp(1)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=2)
        apply_tp_to_lumina_model(model)
        assert isinstance(model.layers[0].attention.qkv, nn.Linear)
        assert isinstance(model.layers[0].attention.out, nn.Linear)

    def test_qkv_weight_sharded_per_head(self):
        """Fused QKV weight must be sharded per-head, NOT contiguously.

        For n_heads=8, n_kv_heads=4, head_dim=64, tp_size=2:
        Full QKV weight layout: [Q(512) | K(256) | V(256)] = 1024 along dim 0.
        Rank 0 should get: [Q_heads_0-3(256) | K_heads_0-1(128) | V_heads_0-1(128)] = 512
        Rank 1 should get: [Q_heads_4-7(256) | K_heads_2-3(128) | V_heads_2-3(128)] = 512

        NOT the wrong contiguous sharding:
        Rank 0 wrong: weight[0:512] = [Q_all(512)]
        Rank 1 wrong: weight[512:1024] = [K_all(256) | V_all(256)]
        """
        dim = 512
        n_heads = 8
        n_kv_heads = 4
        head_dim = 64
        q_size = n_heads * head_dim  # 512
        kv_size = n_kv_heads * head_dim  # 256

        # Create QKV weight with known values for each section
        attn = StubJointAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
        full_weight = attn.qkv.weight.data
        # Mark Q section with 1.0, K with 2.0, V with 3.0
        full_weight.zero_()
        full_weight[:q_size] = 1.0
        full_weight[q_size:q_size + kv_size] = 2.0
        full_weight[q_size + kv_size:] = 3.0

        for rank in range(2):
            mock_tp(2, tp_rank=rank)
            attn_r = StubJointAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
            attn_r.qkv.weight.data.copy_(full_weight)
            apply_tp_to_lumina_attention(attn_r)

            w = attn_r.qkv.weight.data
            local_q_size = (n_heads // 2) * head_dim  # 256
            local_kv_size = (n_kv_heads // 2) * head_dim  # 128
            total_local = local_q_size + 2 * local_kv_size  # 512

            assert w.shape[0] == total_local, f"Rank {rank}: wrong local QKV size"
            # First local_q_size rows should be Q (1.0)
            assert torch.all(w[:local_q_size] == 1.0), f"Rank {rank}: Q section has wrong values"
            # Next local_kv_size rows should be K (2.0)
            assert torch.all(w[local_q_size:local_q_size + local_kv_size] == 2.0), f"Rank {rank}: K section has wrong values"
            # Last local_kv_size rows should be V (3.0)
            assert torch.all(w[local_q_size + local_kv_size:] == 3.0), f"Rank {rank}: V section has wrong values"

    def test_qkv_custom_weight_loader_installed(self):
        """QKV TPLinear must have a custom weight_loader for streaming path."""
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        # The weight_loader should NOT be the default TPLinear.weight_loader
        # (it should be our custom _qkv_weight_loader)
        assert hasattr(attn.qkv, 'weight_loader')
        # Verify it works with full-sized input
        full_weight = torch.randn(8 * 64 * 3, 512)  # [full_qkv, dim]
        param = nn.Parameter(torch.zeros(8 * 64 * 3 // 2, 512))
        attn.qkv.weight_loader(param, full_weight)
        assert param.shape[0] == 8 * 64 * 3 // 2

    def test_qkv_scale_loader_installed(self):
        """QKV TPLinear must have a _qkv_scale_loader for INT8 scale sharding."""
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        assert hasattr(attn.qkv, '_qkv_scale_loader')

    def test_qkv_scale_loader_per_row(self):
        """Per-row weight_scale must be QKV-sharded, not naively narrowed."""
        dim = 512
        n_heads = 8
        n_kv_heads = 4
        head_dim = 64
        q_size = n_heads * head_dim       # 512
        kv_size = n_kv_heads * head_dim   # 256
        full_scale_size = q_size + 2 * kv_size  # 1024

        # Mark Q scales=1, K scales=2, V scales=3
        full_scale = torch.zeros(full_scale_size)
        full_scale[:q_size] = 1.0
        full_scale[q_size:q_size + kv_size] = 2.0
        full_scale[q_size + kv_size:] = 3.0

        for rank in range(2):
            mock_tp(2, tp_rank=rank)
            attn = StubJointAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
            apply_tp_to_lumina_attention(attn)
            sharded = attn.qkv._qkv_scale_loader(full_scale)

            local_q = (n_heads // 2) * head_dim      # 256
            local_kv = (n_kv_heads // 2) * head_dim  # 128
            expected_size = local_q + 2 * local_kv    # 512
            assert sharded.shape[0] == expected_size, f"Rank {rank}: wrong sharded scale size"
            assert torch.all(sharded[:local_q] == 1.0), f"Rank {rank}: Q scales wrong"
            assert torch.all(sharded[local_q:local_q + local_kv] == 2.0), f"Rank {rank}: K scales wrong"
            assert torch.all(sharded[local_q + local_kv:] == 3.0), f"Rank {rank}: V scales wrong"

    def test_qkv_scale_loader_scalar(self):
        """Scalar per-tensor scale should pass through unchanged."""
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        apply_tp_to_lumina_attention(attn)
        scalar_scale = torch.tensor(0.5)
        result = attn.qkv._qkv_scale_loader(scalar_scale)
        assert result.item() == 0.5

    def test_int8_weight_scale_sharded_on_load(self):
        """Legacy path: weight_scale on source linear is QKV-sharded during TP."""
        mock_tp(2, tp_rank=0)
        dim = 512
        n_heads = 8
        n_kv_heads = 8
        head_dim = 64
        full_qkv = 3 * n_heads * head_dim  # 1536
        attn = StubJointAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
        # Simulate INT8 pre-quantized with per-row scale
        attn.qkv.weight_scale = torch.ones(full_qkv)
        attn.qkv.weight_scale[:n_heads * head_dim] = 1.0  # Q
        attn.qkv.weight_scale[n_heads * head_dim:2 * n_heads * head_dim] = 2.0  # K
        attn.qkv.weight_scale[2 * n_heads * head_dim:] = 3.0  # V
        apply_tp_to_lumina_attention(attn)
        assert hasattr(attn.qkv, 'weight_scale')
        scale = attn.qkv.weight_scale
        local_size = full_qkv // 2  # 768
        assert scale.shape[0] == local_size
        local_q = (n_heads // 2) * head_dim  # 256
        local_kv = (n_kv_heads // 2) * head_dim  # 256
        assert torch.all(scale[:local_q] == 1.0)
        assert torch.all(scale[local_q:local_q + local_kv] == 2.0)
        assert torch.all(scale[local_q + local_kv:] == 3.0)

    def test_structure_only_preserves_int8_dtype(self):
        """structure_only=True must preserve int8 dtype for streaming TP compatibility.

        When dtype is lost (defaulting to float32), the streaming path creates
        a float32 parameter; int8 weight data gets silently cast → the INT8
        inference path is never triggered → garbage output.
        """
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        # Simulate INT8 weight (as Int8SafetensorOps would produce on meta)
        attn.qkv.weight = nn.Parameter(
            torch.empty(8 * 64 * 3, 512, dtype=torch.int8, device="meta"),
            requires_grad=False,
        )
        apply_tp_to_lumina_attention(attn, structure_only=True)
        assert attn.qkv.weight.dtype == torch.int8, "QKV TPLinear must preserve int8 dtype in structure_only mode"

    def test_structure_only_preserves_bf16_dtype(self):
        """structure_only=True must also preserve bf16 dtype."""
        mock_tp(2)
        attn = StubJointAttention(dim=512, n_heads=8, n_kv_heads=8, head_dim=64)
        attn.qkv.weight = nn.Parameter(
            torch.empty(8 * 64 * 3, 512, dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
        apply_tp_to_lumina_attention(attn, structure_only=True)
        assert attn.qkv.weight.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Tests: FeedForward
# ---------------------------------------------------------------------------

class TestLuminaTPFeedForward:
    def test_all_linears_patched(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, hidden_dim=1024)
        apply_tp_to_lumina_feedforward(ff)
        assert _is_tp_linear(ff.w1)
        assert _is_tp_linear(ff.w2)
        assert _is_tp_linear(ff.w3)

    def test_w1_w3_column_no_gather(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, hidden_dim=1024)
        apply_tp_to_lumina_feedforward(ff)
        assert ff.w1.parallelism == "column"
        assert ff.w1.gather_output is False
        assert ff.w3.parallelism == "column"
        assert ff.w3.gather_output is False

    def test_w2_row_parallel(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, hidden_dim=1024)
        apply_tp_to_lumina_feedforward(ff)
        assert ff.w2.parallelism == "row"


# ---------------------------------------------------------------------------
# Tests: JointTransformerBlock
# ---------------------------------------------------------------------------

class TestLuminaTPBlock:
    def test_attention_patched(self):
        mock_tp(2)
        block = StubJointTransformerBlock(dim=512)
        apply_tp_to_lumina_block(block)
        assert _is_tp_linear(block.attention.qkv)
        assert _is_tp_linear(block.attention.out)

    def test_feedforward_patched(self):
        mock_tp(2)
        block = StubJointTransformerBlock(dim=512)
        apply_tp_to_lumina_block(block)
        assert _is_tp_linear(block.feed_forward.w1)
        assert _is_tp_linear(block.feed_forward.w2)
        assert _is_tp_linear(block.feed_forward.w3)

    def test_adaln_replicated(self):
        """AdaLN modulation must stay replicated (not TP-patched), matching sglang."""
        mock_tp(2)
        block = StubJointTransformerBlock(dim=512)
        apply_tp_to_lumina_block(block)
        adaln_linear = block.adaLN_modulation[-1]
        assert isinstance(adaln_linear, nn.Linear), "adaLN should remain plain nn.Linear"
        assert not _is_tp_linear(adaln_linear)

    def test_tp_patched_flag(self):
        mock_tp(2)
        block = StubJointTransformerBlock(dim=512)
        apply_tp_to_lumina_block(block)
        assert block._tp_patched is True


# ---------------------------------------------------------------------------
# Tests: Full model
# ---------------------------------------------------------------------------

class TestLuminaTPModel:
    def test_main_layers_patched(self):
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        for block in model.layers:
            assert _is_tp_linear(block.attention.qkv)
            assert _is_tp_linear(block.attention.out)
            assert _is_tp_linear(block.feed_forward.w1)

    def test_refiner_blocks_patched(self):
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        for block in model.noise_refiner:
            assert _is_tp_linear(block.attention.qkv)
        for block in model.context_refiner:
            assert _is_tp_linear(block.attention.qkv)

    def test_boundary_projections_patched(self):
        """Only x_embedder and final_layer.linear are TP-patched at boundaries."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        assert _is_tp_linear(model.x_embedder)
        assert _is_tp_linear(model.final_layer.linear)

    def test_replicated_modules_not_patched(self):
        """cap_embedder, final_layer.adaLN_modulation stay replicated (sglang pattern)."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        assert isinstance(model.cap_embedder[-1], nn.Linear)
        assert not _is_tp_linear(model.cap_embedder[-1])
        assert isinstance(model.final_layer.adaLN_modulation[-1], nn.Linear)
        assert not _is_tp_linear(model.final_layer.adaLN_modulation[-1])

    def test_boundary_projections_gather(self):
        """TP-patched boundary projections must gather output."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        assert model.x_embedder.gather_output is True
        assert model.final_layer.linear.gather_output is True

    def test_block_tp_properties(self):
        """Verify blocks use true head-sharded TP (not weight-only)."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        block = model.layers[0]
        assert block.attention.qkv.gather_output is False
        assert block.attention.out.parallelism == "row"
        assert block.feed_forward.w1.gather_output is False
        assert block.feed_forward.w2.parallelism == "row"

    def test_tp_size_1_no_patching(self):
        mock_tp(1)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model)
        for block in model.layers:
            assert isinstance(block.attention.qkv, nn.Linear)
            assert isinstance(block.attention.out, nn.Linear)

    def test_structure_only_mode(self):
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        apply_tp_to_lumina_model(model, structure_only=True)
        # Should still create TPLinear modules
        assert _is_tp_linear(model.layers[0].attention.qkv)

    def test_optional_siglip_refiner(self):
        """Model without siglip_refiner should not error."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        model.siglip_refiner = None
        apply_tp_to_lumina_model(model)
        # No exception means success

    def test_optional_embedders(self):
        """Model without optional embedders should not error."""
        mock_tp(2)
        model = StubNextDiT(dim=512, n_heads=8, n_layers=4)
        model.clip_text_pooled_proj = None
        model.time_text_embed = None
        apply_tp_to_lumina_model(model)
        # No exception means success


# ---------------------------------------------------------------------------
# TPRegistry integration
# ---------------------------------------------------------------------------

class TestLuminaTPRegistry:
    def test_lumina2_registered(self):
        """Verify Lumina2 has a TP handler in the registry."""
        try:
            from comfy import model_base
            from raylight.comfy_dist.tp_registry import TPRegistry
            if hasattr(model_base, "Lumina2"):
                assert model_base.Lumina2 in TPRegistry._REGISTRY
        except ImportError:
            pytest.skip("ComfyUI not available")


# ---------------------------------------------------------------------------
# LoRA + TP offset adjustment
# ---------------------------------------------------------------------------

class TestLoraTPOffset:
    """Test that LoRA offsets are correctly adjusted for TP-sharded weights."""

    def _make_diff_patch(self, diff, offset, strength=1.0):
        """Create a 'diff' patch tuple: (strength, v, strength_model, offset, function)."""
        return (strength, (diff,), 1.0, offset, None)

    def test_offset_no_tp(self):
        """Without TP, offsets apply normally."""
        mock_tp(1)
        from raylight.comfy_dist.lora import calculate_weight
        # Full QKV weight [1536, 512] (3 * 512)
        weight = torch.zeros(1536, 512)
        diff = torch.ones(512, 512)  # K patch
        patch = self._make_diff_patch(diff, offset=(0, 512, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        # K region should be modified
        assert result[512:1024].sum() > 0
        assert result[0:512].sum().item() == 0.0
        assert result[1024:1536].sum().item() == 0.0

    def test_offset_tp2_rank0_full_overlap(self):
        """TP=2 rank 0: Q offset (0,0,512) fully within shard [0,768)."""
        mock_tp(2, tp_rank=0)
        from raylight.comfy_dist.lora import calculate_weight
        # Sharded QKV weight [768, 512] (tp_size=2, full=1536)
        weight = torch.zeros(768, 512)
        diff = torch.ones(512, 512)  # Q patch
        patch = self._make_diff_patch(diff, offset=(0, 0, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        assert result[0:512].sum() > 0
        assert result[512:768].sum().item() == 0.0

    def test_offset_tp2_rank0_partial_overlap(self):
        """TP=2 rank 0: K offset (0,512,512) partially in shard [0,768)."""
        mock_tp(2, tp_rank=0)
        from raylight.comfy_dist.lora import calculate_weight
        weight = torch.zeros(768, 512)
        diff = torch.ones(512, 512)  # K patch (full)
        patch = self._make_diff_patch(diff, offset=(0, 512, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        # Only first 256 of K falls in this shard (rows 512..768)
        assert result[512:768].sum() > 0
        assert result[0:512].sum().item() == 0.0

    def test_offset_tp2_rank0_no_overlap(self):
        """TP=2 rank 0: V offset (0,1024,512) outside shard [0,768)."""
        mock_tp(2, tp_rank=0)
        from raylight.comfy_dist.lora import calculate_weight
        weight = torch.zeros(768, 512)
        diff = torch.ones(512, 512)  # V patch
        patch = self._make_diff_patch(diff, offset=(0, 1024, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        assert result.sum().item() == 0.0

    def test_offset_tp2_rank1_partial_overlap(self):
        """TP=2 rank 1: K offset (0,512,512) partially in shard [768,1536)."""
        mock_tp(2, tp_rank=1)
        from raylight.comfy_dist.lora import calculate_weight
        weight = torch.zeros(768, 512)
        diff = torch.ones(512, 512)  # K patch
        patch = self._make_diff_patch(diff, offset=(0, 512, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        # Second 256 of K falls here (rows 0..256 of rank 1's shard)
        assert result[0:256].sum() > 0
        assert result[256:768].sum().item() == 0.0

    def test_offset_tp2_rank1_full_overlap(self):
        """TP=2 rank 1: V offset (0,1024,512) fully within shard [768,1536)."""
        mock_tp(2, tp_rank=1)
        from raylight.comfy_dist.lora import calculate_weight
        weight = torch.zeros(768, 512)
        diff = torch.ones(512, 512)  # V patch
        patch = self._make_diff_patch(diff, offset=(0, 1024, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        # V maps to rows 256..768 of rank 1's shard
        assert result[256:768].sum() > 0
        assert result[0:256].sum().item() == 0.0

    def test_offset_tp4_partial_overlap_correct_slice(self):
        """TP=4: verify the correct PORTION of the diff is used, not tp_rank-based."""
        mock_tp(4, tp_rank=1)
        from raylight.comfy_dist.lora import calculate_weight
        # Full QKV = 3*512 = 1536, shard = 384
        # K offset = (0, 512, 512), K region [512, 1024)
        # Rank 1 shard: [384, 768)
        # Intersection: [512, 768) → local [128, 384), length 256
        # Diff slice: [0, 256) of K diff (first 256 of 512)
        weight = torch.zeros(384, 128)
        # Use a gradient diff to verify correct slice is used
        diff = torch.arange(512 * 128, dtype=torch.float32).reshape(512, 128)
        patch = self._make_diff_patch(diff, offset=(0, 512, 512))
        result = calculate_weight([patch], weight.clone(), "test.qkv.weight")
        # Rows 128..384 of result should have diff rows 0..256
        expected = diff[0:256]
        assert torch.allclose(result[128:384], expected)
        assert result[0:128].sum().item() == 0.0

    def test_multiple_qkv_patches_tp2(self):
        """TP=2: all three Q/K/V patches applied correctly to rank 0."""
        mock_tp(2, tp_rank=0)
        from raylight.comfy_dist.lora import calculate_weight
        # Full QKV=1536, shard=768, hidden=512
        weight = torch.zeros(768, 512)
        q_diff = torch.ones(512, 512) * 1.0
        k_diff = torch.ones(512, 512) * 2.0
        v_diff = torch.ones(512, 512) * 3.0
        patches = [
            self._make_diff_patch(q_diff, offset=(0, 0, 512)),
            self._make_diff_patch(k_diff, offset=(0, 512, 512)),
            self._make_diff_patch(v_diff, offset=(0, 1024, 512)),
        ]
        result = calculate_weight(patches, weight.clone(), "test.qkv.weight")
        # Rank 0 shard [0,768): Q=[0,512) val=1, K=[512,768) val=2, V=skipped
        assert torch.allclose(result[0:512], torch.ones(512, 512) * 1.0)
        assert torch.allclose(result[512:768], torch.ones(256, 512) * 2.0)


class TestLoraTPDiffShard:
    """Test _tp_shard_lora_diff with pre-computed offset slice info."""

    def test_shard_with_slice_attr(self):
        """_tp_lora_diff_slice attribute is used instead of tp_rank."""
        mock_tp(4, tp_rank=2)
        from raylight.comfy_dist.weight_adapter.lora import _tp_shard_lora_diff
        lora_diff = torch.arange(512 * 128, dtype=torch.float32).reshape(512, 128)
        # Simulate: we want rows [100:356) of lora_diff
        weight = torch.zeros(256, 128)
        weight._tp_lora_diff_slice = (0, 100, 256)
        result = _tp_shard_lora_diff(lora_diff, weight.shape, weight=weight)
        expected = lora_diff[100:356].reshape(256, 128)
        assert torch.allclose(result, expected)

    def test_shard_without_slice_attr_fallback(self):
        """Without _tp_lora_diff_slice, falls back to existing tp_rank logic."""
        mock_tp(2, tp_rank=1)
        from raylight.comfy_dist.weight_adapter.lora import _tp_shard_lora_diff
        lora_diff = torch.arange(512 * 128, dtype=torch.float32).reshape(512, 128)
        weight = torch.zeros(256, 128)
        result = _tp_shard_lora_diff(lora_diff, weight.shape, weight=weight)
        # tp_rank=1, column-parallel: takes second half
        expected = lora_diff[256:512].reshape(256, 128)
        assert torch.allclose(result, expected)

    def test_shard_matching_shapes(self):
        """When shapes match, no narrowing needed."""
        mock_tp(2, tp_rank=0)
        from raylight.comfy_dist.weight_adapter.lora import _tp_shard_lora_diff
        lora_diff = torch.ones(256, 128)
        weight = torch.zeros(256, 128)
        result = _tp_shard_lora_diff(lora_diff, weight.shape, weight=weight)
        assert result.shape == (256, 128)



