import torch
from torch.distributed.tensor import DTensor


#  Honestly i just throw DTensor.from_local untill i dont get error
def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS,
                  generator=None, device_mesh=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    rand_local = torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )

    if isinstance(abs_x, DTensor):
        rand_local = DTensor.from_local(rand_local, abs_x.device_mesh, abs_x.placements)
        mantissa_scaled = mantissa_scaled + rand_local
    else:
        mantissa_scaled = mantissa_scaled + rand_local

    mantissa_scaled = mantissa_scaled.floor() / (2**MANTISSA_BITS)

    # Ensure return type mirrors abs_x
    if isinstance(abs_x, DTensor) and not isinstance(mantissa_scaled, DTensor):
        mantissa_scaled = DTensor.from_local(mantissa_scaled, abs_x.device_mesh, abs_x.placements)
    elif not isinstance(abs_x, DTensor) and isinstance(mantissa_scaled, DTensor):
        mantissa_scaled = mantissa_scaled.to_local()

    return mantissa_scaled


# Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype, generator=None, device_mesh=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    # Convert input precision
    x = x.half()

    # Sign + abs
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, torch.zeros_like(sign), sign)

    # Exponent
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )
    normal_mask = ~(exponent == 0)

    abs_x = calc_mantissa(abs_x, exponent, normal_mask,
                          MANTISSA_BITS, EXPONENT_BIAS,
                          generator=generator, device_mesh=device_mesh)

    sign = sign * torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    # Clamp to float8 range
    finfo = torch.finfo(dtype)
    torch.clamp(sign, min=finfo.min, max=finfo.max, out=sign)

    # Mirror input type
    if isinstance(x, DTensor) and not isinstance(sign, DTensor):
        sign = DTensor.from_local(sign, x.device_mesh, x.placements)
    elif not isinstance(x, DTensor) and isinstance(sign, DTensor):
        sign = sign.to_local()

    return sign


def stochastic_rounding(value, dtype, seed=0, device_mesh=None):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i:i+slice_size].copy_(manual_stochastic_round_to_float8(value[i:i+slice_size], dtype, generator=generator, device_mesh=device_mesh))
        return output

    return value.to(dtype=dtype)
