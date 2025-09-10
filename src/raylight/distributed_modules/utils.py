import torch
from torch.distributed._tensor import DTensor


def detect_dtype_mismatch(module, ref_dtype):
    ignored_param = set()
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored_param.add(param)

    return ignored_param


def ensure_no_scalar(module):
    for name, param in module.named_parameters(recurse=False):
        if param.ndim == 0:
            new_param = param.unsqueeze(0)
            module._parameters[name] = torch.nn.Parameter(new_param)

    for name, buf in module.named_buffers(recurse=False):
        if buf.ndim == 0:
            new_buf = buf.unsqueeze(0)
            module._buffers[name] = new_buf
    return module


def inspect_tensor(t):
    if isinstance(t, DTensor):
        print("=== DTensor Info ===")
        print("Global shape:", t.shape)
        print("Local shape:", t.to_local().shape)
        print("Device mesh:", t.device_mesh)
        print("Placement:", t.placements)
        print("====================")
    else:
        print("Regular Tensor:", t.shape, t.device)
