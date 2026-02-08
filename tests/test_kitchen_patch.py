import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Mock comfy_kitchen before importing the module under test
# We need to mock the entire structure that kitchen_patch expects
mock_kitchen = MagicMock()
mock_kitchen.tensor.base.QuantizedTensor = MagicMock()
mock_kitchen.tensor.fp8.TensorCoreFP8Layout = MagicMock()
# Mock the dispatch table
mock_params = MagicMock()
mock_params.orig_dtype = torch.float32
mock_params.scale = 1.0

# Mock QuantizedTensor instance/class
class MockQuantizedTensor:
    def __init__(self, qdata, layout_cls, params):
        self._qdata = qdata
        self._layout_cls = layout_cls
        self._params = params

mock_kitchen.tensor.base.QuantizedTensor = MockQuantizedTensor
mock_kitchen.tensor.fp8.TensorCoreFP8Layout = MagicMock()
# Mock the dispatch table
mock_params = MagicMock()
mock_params.orig_dtype = torch.float32
mock_params.scale = 1.0

def mock_register_layout_op(op, layout_cls):
    def decorator(func):
        # We can store the registration to verify it happened
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

class TestKitchenPatch(unittest.TestCase):
    def setUp(self):
        # Reset registry
        if hasattr(mock_register_layout_op, "registry"):
             mock_register_layout_op.registry = {}
        
        # Ensure we can use real torch ops for mocking
        self.mock_qdata = torch.randn(4, 4)
        self.mock_qt = MockQuantizedTensor(
            self.mock_qdata, 
            "layout", 
            kitchen_patch.TensorCoreFP8Layout.Params(1.0, torch.float32, (4, 4))
        )

    def test_apply_patches_registration(self):
        """Verify that apply_patches registers the expected operations."""
        # Mock c10d presence
        with patch.object(torch.ops, "_c10d_functional", create=True) as mock_c10d:
             mock_c10d.all_gather_into_tensor.default = "all_gather_op"
             
             kitchen_patch.apply_patches()
             
             registry = mock_register_layout_op.registry
             
             # Check basic Aten ops
             self.assertIn(torch.ops.aten.split.Tensor, registry)
             self.assertIn(torch.ops.aten.slice.Tensor, registry)
             self.assertIn(torch.ops.aten.cat.default, registry)
             
             # Check C10d ops (if mocked correctly)
             self.assertIn("all_gather_op", registry)

    def test_handle_fp8_split(self):
        """Test split handler logic."""
        # split into 2 chunks of size 2
        # We need to ensure torch.ops.aten.split.Tensor works on the _qdata (which is real tensor)
        # self.mock_qdata is real tensor so it should work fine.
        chunks = kitchen_patch._handle_fp8_split(None, [self.mock_qt, 2], {})
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]._qdata.shape, (2, 4))
        self.assertEqual(chunks[1]._qdata.shape, (2, 4))
        self.assertIsInstance(chunks[0], MockQuantizedTensor)

    def test_handle_fp8_slice(self):
        """Test slice handler logic."""
        # Slice first 2 rows
        res = kitchen_patch._handle_fp8_slice_tensor(None, [self.mock_qt, 0, 0, 2], {})
        
        self.assertIsInstance(res, MockQuantizedTensor)
        self.assertEqual(res._qdata.shape, (2, 4))

    def test_handle_fp8_cat(self):
        """Test cat handler logic."""
        # Cat two tensors
        qt2 = MockQuantizedTensor(
            torch.randn(4, 4), 
            "layout", 
            kitchen_patch.TensorCoreFP8Layout.Params(1.0, torch.float32, (4, 4))
        )
        
        res = kitchen_patch._handle_fp8_cat(None, [[self.mock_qt, qt2]], {"dim": 0})
        
        self.assertIsInstance(res, MockQuantizedTensor)
        # Check result shape
        self.assertEqual(res._qdata.shape, (8, 4))


if __name__ == '__main__':
    unittest.main()
