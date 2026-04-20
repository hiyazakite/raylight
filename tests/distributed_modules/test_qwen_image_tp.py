"""Unit tests for Qwen Image TP patching — single-process, no real distributed backend.

Run standalone:
    export PYTHONPATH=$PYTHONPATH:/root/ComfyUI/custom_nodes/raylight/src
    python3 tests/distributed_modules/test_qwen_image_tp.py
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


def _make_mock_module(name, attrs=None):
    m = types.ModuleType(name)
    m.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m


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
import torch.nn.functional as F

from raylight.distributed_modules.tensor_parallel import TensorParallelState, TPLinear


def mock_tp(tp_size: int, tp_rank: int = 0) -> None:
    TensorParallelState._tp_size = tp_size
    TensorParallelState._tp_rank = tp_rank
    TensorParallelState._tp_group = None


# ---------------------------------------------------------------------------
# Minimal stub modules matching Qwen Image architecture
# ---------------------------------------------------------------------------

class StubGELU(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.gelu(self.proj(x), approximate="tanh")


class StubFeedForward(nn.Module):
    def __init__(self, dim=512, inner_dim=2048):
        super().__init__()
        self.net = nn.ModuleList([
            StubGELU(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim, bias=True),
        ])

    def forward(self, x):
        for m in self.net:
            x = m(x)
        return x


class StubAttention(nn.Module):
    def __init__(self, dim=512, heads=8, head_dim=64):
        super().__init__()
        self.heads = heads
        self.dim_head = head_dim
        inner_dim = heads * head_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.add_q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.add_v_proj = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(0.0),
        ])
        self.to_add_out = nn.Linear(inner_dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(head_dim)
        self.norm_k = nn.RMSNorm(head_dim)
        self.norm_added_q = nn.RMSNorm(head_dim)
        self.norm_added_k = nn.RMSNorm(head_dim)


class StubQwenBlock(nn.Module):
    def __init__(self, dim=512, heads=8, head_dim=64, inner_dim=2048):
        super().__init__()
        self.attn = StubAttention(dim=dim, heads=heads, head_dim=head_dim)
        self.img_mlp = StubFeedForward(dim=dim, inner_dim=inner_dim)
        self.txt_mlp = StubFeedForward(dim=dim, inner_dim=inner_dim)
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))


class StubQwenModel(nn.Module):
    def __init__(self, dim=512, heads=8, head_dim=64, n_layers=4, inner_dim=2048):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            StubQwenBlock(dim=dim, heads=heads, head_dim=head_dim, inner_dim=inner_dim)
            for _ in range(n_layers)
        ])


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from raylight.diffusion_models.qwen_image.tp import (
    apply_tp_to_qwen_attention,
    apply_tp_to_qwen_feedforward,
    apply_tp_to_qwen_block,
    apply_tp_to_qwen_image_model,
)


def _is_tp_linear(m):
    return isinstance(m, TPLinear)


@pytest.fixture(autouse=True)
def _reset_tp():
    TensorParallelState.reset()
    yield
    TensorParallelState.reset()


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

class TestQwenTPAttention:
    def test_img_qkv_column_no_gather(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        for attr in ("to_q", "to_k", "to_v"):
            lin = getattr(attn, attr)
            assert _is_tp_linear(lin), f"{attr} should be TPLinear"
            assert lin.parallelism == "column"
            assert lin.gather_output is False

    def test_txt_qkv_column_no_gather(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        for attr in ("add_q_proj", "add_k_proj", "add_v_proj"):
            lin = getattr(attn, attr)
            assert _is_tp_linear(lin), f"{attr} should be TPLinear"
            assert lin.parallelism == "column"
            assert lin.gather_output is False

    def test_img_out_row_parallel(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        assert _is_tp_linear(attn.to_out[0])
        assert attn.to_out[0].parallelism == "row"

    def test_txt_out_row_parallel(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        assert _is_tp_linear(attn.to_add_out)
        assert attn.to_add_out.parallelism == "row"

    def test_local_head_count(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        assert attn._tp_local_heads == 4  # 8 / 2

    def test_local_head_count_tp4(self):
        mock_tp(4)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        assert attn._tp_local_heads == 2  # 8 / 4

    def test_custom_forward_installed(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        original_forward = attn.forward
        apply_tp_to_qwen_attention(attn)
        assert attn.forward != original_forward

    def test_qk_norms_unchanged(self):
        mock_tp(2)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        # QK norms must NOT be replaced — they're per-head-dim
        assert isinstance(attn.norm_q, nn.RMSNorm)
        assert isinstance(attn.norm_k, nn.RMSNorm)
        assert isinstance(attn.norm_added_q, nn.RMSNorm)
        assert isinstance(attn.norm_added_k, nn.RMSNorm)

    def test_weight_sharded_for_heads(self):
        """Column-parallel: each rank holds out_features/tp_size rows."""
        dim = 512
        heads = 8
        head_dim = 64
        inner_dim = heads * head_dim  # 512

        for rank in range(2):
            mock_tp(2, tp_rank=rank)
            attn = StubAttention(dim=dim, heads=heads, head_dim=head_dim)
            # Fill to_q weight so rank 0 gets first half, rank 1 gets second half
            attn.to_q.weight.data.fill_(float(rank + 1))
            apply_tp_to_qwen_attention(attn)
            assert attn.to_q.weight.shape[0] == inner_dim // 2

    def test_tp_size_1_noop(self):
        mock_tp(1)
        attn = StubAttention(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_attention(attn)
        assert isinstance(attn.to_q, nn.Linear)


# ---------------------------------------------------------------------------
# FeedForward tests
# ---------------------------------------------------------------------------

class TestQwenTPFeedForward:
    def test_gate_column_no_gather(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, inner_dim=2048)
        apply_tp_to_qwen_feedforward(ff)
        assert _is_tp_linear(ff.net[0].proj)
        assert ff.net[0].proj.parallelism == "column"
        assert ff.net[0].proj.gather_output is False

    def test_output_row_parallel(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, inner_dim=2048)
        apply_tp_to_qwen_feedforward(ff)
        assert _is_tp_linear(ff.net[2])
        assert ff.net[2].parallelism == "row"

    def test_dropout_unchanged(self):
        mock_tp(2)
        ff = StubFeedForward(dim=512, inner_dim=2048)
        apply_tp_to_qwen_feedforward(ff)
        assert isinstance(ff.net[1], nn.Dropout)

    def test_intermediate_dim_sharded(self):
        """gate proj output dim = inner_dim / tp_size."""
        mock_tp(2)
        ff = StubFeedForward(dim=512, inner_dim=2048)
        apply_tp_to_qwen_feedforward(ff)
        assert ff.net[0].proj.weight.shape[0] == 2048 // 2

    def test_tp_size_1_noop(self):
        mock_tp(1)
        ff = StubFeedForward(dim=512, inner_dim=2048)
        apply_tp_to_qwen_feedforward(ff)
        assert isinstance(ff.net[0].proj, nn.Linear)
        assert isinstance(ff.net[2], nn.Linear)


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------

class TestQwenTPBlock:
    def test_attn_patched(self):
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        assert _is_tp_linear(block.attn.to_q)

    def test_img_mlp_patched(self):
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        assert _is_tp_linear(block.img_mlp.net[0].proj)
        assert _is_tp_linear(block.img_mlp.net[2])

    def test_txt_mlp_patched(self):
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        assert _is_tp_linear(block.txt_mlp.net[0].proj)
        assert _is_tp_linear(block.txt_mlp.net[2])

    def test_adaLN_img_mod_gather(self):
        """img_mod[1] must be column-parallel with gather_output=True."""
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        lin = block.img_mod[1]
        assert _is_tp_linear(lin)
        assert lin.parallelism == "column"
        assert lin.gather_output is True

    def test_adaLN_txt_mod_gather(self):
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        lin = block.txt_mod[1]
        assert _is_tp_linear(lin)
        assert lin.parallelism == "column"
        assert lin.gather_output is True

    def test_tp_patched_flag(self):
        mock_tp(2)
        block = StubQwenBlock()
        apply_tp_to_qwen_block(block)
        assert block._tp_patched is True

    def test_adaLN_output_dim_full(self):
        """gather_output=True: every rank has the full 6*dim adaLN output."""
        mock_tp(2)
        block = StubQwenBlock(dim=512, heads=8, head_dim=64)
        apply_tp_to_qwen_block(block)
        # out_features on a gather=True column-linear is the original full size
        assert block.img_mod[1].out_features == 6 * 512
        assert block.txt_mod[1].out_features == 6 * 512


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestQwenTPModel:
    def test_all_blocks_patched(self):
        mock_tp(2)
        model = StubQwenModel(dim=512, heads=8, head_dim=64, n_layers=4)
        apply_tp_to_qwen_image_model(model)
        for i, block in enumerate(model.transformer_blocks):
            assert _is_tp_linear(block.attn.to_q), f"block {i} attn.to_q not patched"
            assert _is_tp_linear(block.img_mlp.net[0].proj), f"block {i} img_mlp not patched"
            assert _is_tp_linear(block.txt_mlp.net[0].proj), f"block {i} txt_mlp not patched"

    def test_tp_size_1_noop(self):
        mock_tp(1)
        model = StubQwenModel(dim=512, heads=8, head_dim=64, n_layers=2)
        apply_tp_to_qwen_image_model(model)
        for block in model.transformer_blocks:
            assert isinstance(block.attn.to_q, nn.Linear)
            assert isinstance(block.img_mlp.net[0].proj, nn.Linear)

    def test_structure_only_skips_weight_copy(self):
        """structure_only=True: TPLinear on meta device, no weight data copied."""
        mock_tp(2)
        model = StubQwenModel(dim=512, heads=8, head_dim=64, n_layers=2)
        apply_tp_to_qwen_image_model(model, structure_only=True)
        # Weight device should be meta
        assert model.transformer_blocks[0].attn.to_q.weight.device.type == "meta"
