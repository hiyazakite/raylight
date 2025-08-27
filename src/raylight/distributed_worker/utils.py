from torch.distributed._tensor import DTensor


def detect_dtype_mismatch(module, ref_dtype):
    ignored_param = set()
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored_param.add(param)

    return ignored_param


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
