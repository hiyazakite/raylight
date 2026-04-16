"""Tests for kitchen_patch FP8 dispatch handlers."""
import os
import sys

import pytest
import torch
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock comfy_kitchen before importing the module under test
mock_kitchen = MagicMock()


class MockQuantizedTensor:
    def __init__(self, qdata, layout_cls, params):
        self._qdata = qdata
        self._layout_cls = layout_cls
        self._params = params


mock_kitchen.tensor.base.QuantizedTensor = MockQuantizedTensor
mock_kitchen.tensor.fp8.TensorCoreFP8Layout = MagicMock()


def mock_register_layout_op(op, layout_cls):
    def decorator(func):
        if not hasattr(mock_register_layout_op, "registry"):
            mock_register_layout_op.registry = {}
        mock_register_layout_op.registry[op] = func
        return func
    return decorator


mock_kitchen.tensor.base.register_layout_op = mock_register_layout_op
sys.modules["comfy_kitchen.tensor.base"] = mock_kitchen.tensor.base
sys.modules["comfy_kitchen.tensor.fp8"] = mock_kitchen.tensor.fp8

# Now import the module under test
from raylight.distributed_modules import kitchen_patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry():
    if hasattr(mock_register_layout_op, "registry"):
        mock_register_layout_op.registry = {}


@pytest.fixture
def mock_qt():
    qdata = torch.randn(4, 4)
    return MockQuantizedTensor(
        qdata, "layout",
        kitchen_patch.TensorCoreFP8Layout.Params(1.0, torch.float32, (4, 4)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_apply_patches_registration(mock_qt):
    with patch.object(torch.ops, "_c10d_functional", create=True) as mock_c10d:
        mock_c10d.all_gather_into_tensor.default = "all_gather_op"

        kitchen_patch.apply_patches()

        registry = mock_register_layout_op.registry

        assert torch.ops.aten.split.Tensor in registry
        assert torch.ops.aten.slice.Tensor in registry
        assert torch.ops.aten.cat.default in registry
        assert "all_gather_op" in registry


def test_handle_fp8_split(mock_qt):
    chunks = kitchen_patch._handle_fp8_split(None, [mock_qt, 2], {})
    assert len(chunks) == 2
    assert chunks[0]._qdata.shape == (2, 4)
    assert chunks[1]._qdata.shape == (2, 4)
    assert isinstance(chunks[0], MockQuantizedTensor)


def test_handle_fp8_slice(mock_qt):
    res = kitchen_patch._handle_fp8_slice_tensor(None, [mock_qt, 0, 0, 2], {})
    assert isinstance(res, MockQuantizedTensor)
    assert res._qdata.shape == (2, 4)


def test_handle_fp8_cat(mock_qt):
    qt2 = MockQuantizedTensor(
        torch.randn(4, 4), "layout",
        kitchen_patch.TensorCoreFP8Layout.Params(1.0, torch.float32, (4, 4)),
    )
    res = kitchen_patch._handle_fp8_cat(None, [[mock_qt, qt2]], {"dim": 0})
    assert isinstance(res, MockQuantizedTensor)
    assert res._qdata.shape == (8, 4)
