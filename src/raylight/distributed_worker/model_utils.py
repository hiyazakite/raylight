def detect_dtype_mismatch(module, ref_dtype):
    ignored_param = set()
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored_param.add(param)

    return ignored_param


