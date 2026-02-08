import torch
import logging
from dataclasses import replace

try:
    from comfy_kitchen.tensor.base import (
        QuantizedTensor,
        register_layout_op,
        _LAYOUT_DISPATCH_TABLE
    )
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout
except ImportError:
    # Dummy classes for static analysis and runtime safety when library is missing
    class QuantizedTensor: 
        class _params:
            scale = None
            orig_dtype = None
        _qdata = None
        _layout_cls = None

    class TensorCoreFP8Layout:
        class Params:
            def __init__(self, scale, orig_dtype, orig_shape):
                self.scale = scale
                self.orig_dtype = orig_dtype
                self.orig_shape = orig_shape
    
    def register_layout_op(*args):
        def decorator(f): return f
        return decorator
        
    _LAYOUT_DISPATCH_TABLE = {}

logger = logging.getLogger(__name__)

def _wrap_fp8_tensor(qtensor, qdata):
    # Helper to wrap a new qdata with same params as original qtensor
    new_params = TensorCoreFP8Layout.Params(
        scale=qtensor._params.scale,
        orig_dtype=qtensor._params.orig_dtype,
        orig_shape=tuple(qdata.shape),
    )
    return QuantizedTensor(qdata, qtensor._layout_cls, new_params)

# ==================== Distributed Operations ====================

def _handle_all_gather(qt, args, kwargs):
    # args: (tensor, ...) or (output_tensor, input_tensor, ...) depending on call
    # torch.ops._c10d_functional.all_gather_into_tensor.default(output, input, group_name, ...)
    
    # We need to find the input tensor which is QuantizedTensor
    input_tensor = None
    input_idx = None
    for i, arg in enumerate(args):
        if isinstance(arg, QuantizedTensor):
            input_tensor = arg
            input_idx = i

    assert input_tensor is not None, "all_gather handler called without QuantizedTensor input"

    qdata = input_tensor._qdata
    layout_cls = input_tensor._layout_cls
    params = input_tensor._params

    # View as uint8 (bytes) for transport
    qdata_bytes = qdata.contiguous().view(torch.uint8)

    new_args = list(args)
    new_args[input_idx] = qdata_bytes

    # Call underlying c10d op with bytes
    gathered_bytes = torch.ops._c10d_functional.all_gather_into_tensor.default(
        *new_args, **kwargs
    )

    # View back as original dtype (e.g. float8_e4m3fn)
    gathered_qdata = gathered_bytes.view(qdata.dtype)
    gathered_params = replace(params, orig_shape=tuple(gathered_qdata.shape))
    return QuantizedTensor(gathered_qdata, layout_cls, gathered_params)


def _handle_wait_tensor(qt, args, kwargs):
    qtensor = args[0]
    
    # Wait on the underlying bytes
    waited_bytes = torch.ops._c10d_functional.wait_tensor.default(
        qtensor._qdata.view(torch.uint8),
        *args[1:],
        **kwargs,
    )

    waited_qdata = waited_bytes.view(qtensor._qdata.dtype)
    waited_params = replace(qtensor._params, orig_shape=tuple(waited_qdata.shape))
    return QuantizedTensor(waited_qdata, qtensor._layout_cls, waited_params)


def _handle_broadcast(qt, args, kwargs):
    # args[0] is typically a list of tensors for broadcast
    tensor_list = args[0]

    input_tensor = None
    input_idx = None
    for idx, t in enumerate(tensor_list):
        if isinstance(t, QuantizedTensor):
            input_tensor = t
            input_idx = idx
            break

    if input_tensor is None:
        return torch.ops.c10d.broadcast_.default(*args, **kwargs)

    qdata = input_tensor._qdata.contiguous()
    qdata_bytes = qdata.view(torch.uint8)

    new_tensor_list = list(tensor_list)
    new_tensor_list[input_idx] = qdata_bytes

    new_args = list(args)
    new_args[0] = new_tensor_list

    broadcasted = torch.ops.c10d.broadcast_.default(
        *new_args,
        **kwargs,
    )

    if isinstance(broadcasted, tuple):
        tensor_list_out, work = broadcasted
    else:
        tensor_list_out = broadcasted
        work = None

    broadcasted_qdata = tensor_list_out[input_idx].view(qdata.dtype)

    new_out_list = list(tensor_list_out)
    new_out_list[input_idx] = _wrap_fp8_tensor(
        input_tensor,
        broadcasted_qdata,
    )

    if work is not None:
        return new_out_list, work
    else:
        return new_out_list


def _handle_scatter(qt, args, kwargs):
    output_tensors = args[0]
    input_tensors = args[1]

    quantized_outputs = []
    new_output_tensors = list(output_tensors)
    for idx, tensor in enumerate(output_tensors):
        if isinstance(tensor, QuantizedTensor):
            quantized_outputs.append((idx, tensor))
            new_output_tensors[idx] = tensor._qdata.contiguous().view(torch.uint8)

    has_quantized_input = False
    new_input_tensors = []

    def process_input_list(entry):
        nonlocal has_quantized_input
        processed = []
        for t in entry:
            if isinstance(t, QuantizedTensor):
                has_quantized_input = True
                processed.append(t._qdata.contiguous().view(torch.uint8))
            else:
                processed.append(t)
        return processed

    for entry in input_tensors:
        if isinstance(entry, (list, tuple)):
            new_input_tensors.append(process_input_list(entry))
        else:
            new_input_tensors.append(entry)

    if not quantized_outputs and not has_quantized_input:
        return torch.ops.c10d.scatter_.default(*args, **kwargs)

    new_args = [new_output_tensors, new_input_tensors, *args[2:]]
    result = torch.ops.c10d.scatter_.default(*new_args, **kwargs)

    if isinstance(result, tuple):
        output_list, work = result
    else:
        output_list = result
        work = None

    output_list = list(output_list)
    for idx, original in quantized_outputs:
        qdata = output_list[idx].view(original._qdata.dtype)
        output_list[idx] = _wrap_fp8_tensor(original, qdata)

    if work is not None:
        return output_list, work
    return output_list


# ==================== Aten Operations for Sharding ====================

def _handle_fp8_slice_tensor(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.slice.Tensor(*args, **kwargs)

    # Slice the underlying qdata
    sliced_qdata = torch.ops.aten.slice.Tensor(input_tensor._qdata, *args[1:], **kwargs)
    return _wrap_fp8_tensor(input_tensor, sliced_qdata)


def _handle_fp8_split(qt, args, kwargs):
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return torch.ops.aten.split.Tensor(*args, **kwargs)

    qdata_chunks = torch.ops.aten.split.Tensor(input_tensor._qdata, *args[1:], **kwargs)
    wrapped_chunks = tuple(
        _wrap_fp8_tensor(input_tensor, chunk)
        for chunk in qdata_chunks
    )
    return wrapped_chunks


def _handle_fp8_cat(qt, args, kwargs):
    tensors = args[0]
    if not isinstance(tensors, (list, tuple)) or not tensors:
        return torch.ops.aten.cat.default(*args, **kwargs)

    qdata_list = []
    first_qtensor = None
    for item in tensors:
        if not isinstance(item, QuantizedTensor):
            return torch.ops.aten.cat.default(*args, **kwargs)
        qdata_list.append(item._qdata)
        if first_qtensor is None:
            first_qtensor = item

    assert first_qtensor is not None
    concatenated_qdata = torch.ops.aten.cat.default(qdata_list, *args[1:], **kwargs)
    return _wrap_fp8_tensor(first_qtensor, concatenated_qdata)


def _make_fp8_shape_handler(aten_op):
    def handler(qt, args, kwargs):
        input_tensor = args[0]
        if not isinstance(input_tensor, QuantizedTensor):
            return aten_op(*args, **kwargs)

        # Apply aten op to quantized data
        # Note: args[1:] contains the shape args
        new_qdata = aten_op(input_tensor._qdata, *args[1:], **kwargs)

        new_params = TensorCoreFP8Layout.Params(
            scale=input_tensor._params.scale,
            orig_dtype=input_tensor._params.orig_dtype,
            orig_shape=tuple(new_qdata.shape),
        )
        return QuantizedTensor(new_qdata, "TensorCoreFP8Layout", new_params)
    return handler


def apply_patches():
    """
    Registers the distributed and shape handlers to comfy-kitchen's dispatch table.
    """
    if QuantizedTensor is None:
        logger.warning("comfy-kitchen not found, skipping kitchen patches")
        return

    logger.info("Applying ComfyUI-Kitchen distributed patches...")

    # C10D Ops
    if hasattr(torch.ops, "_c10d_functional") and hasattr(torch.ops._c10d_functional, "all_gather_into_tensor"):
        register_layout_op(torch.ops._c10d_functional.all_gather_into_tensor.default, TensorCoreFP8Layout)(_handle_all_gather)
        register_layout_op(torch.ops._c10d_functional.wait_tensor.default, TensorCoreFP8Layout)(_handle_wait_tensor)
        logger.info(" - Registered c10d all_gather/wait patches")
    
    if hasattr(torch.ops, "c10d"):
        if hasattr(torch.ops.c10d, "broadcast_"):
             register_layout_op(torch.ops.c10d.broadcast_.default, TensorCoreFP8Layout)(_handle_broadcast)
        if hasattr(torch.ops.c10d, "scatter_"):
             register_layout_op(torch.ops.c10d.scatter_.default, TensorCoreFP8Layout)(_handle_scatter)
        logger.info(" - Registered c10d broadcast/scatter patches")

    # Aten Ops
    register_layout_op(torch.ops.aten.slice.Tensor, TensorCoreFP8Layout)(_handle_fp8_slice_tensor)
    register_layout_op(torch.ops.aten.split.Tensor, TensorCoreFP8Layout)(_handle_fp8_split)
    register_layout_op(torch.ops.aten.cat.default, TensorCoreFP8Layout)(_handle_fp8_cat)
    
    # Shape Ops (view, reshape, etc.)
    shape_ops = [
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.t.default,
        torch.ops.aten.as_strided.default,
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.alias.default
    ]
    
    for op in shape_ops:
         register_layout_op(op, TensorCoreFP8Layout)(_make_fp8_shape_handler(op))
    
    logger.info(" - Registered aten shape/slice/split patches")
