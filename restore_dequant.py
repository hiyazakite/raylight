import re

path = '/root/ComfyUI/custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/dequant.py'
with open(path, 'r') as f:
    code = f.read()

# 1. Add _SHIFT_TENSORS and get_shift if missing (already there but let's be sure)
if '_SHIFT_TENSORS = {}' not in code:
    code = code.replace('TORCH_COMPATIBLE_QTYPES =', '_SHIFT_TENSORS = {}\n\ndef get_shift(values, device, dtype, shape):\n    key = (values, str(device), dtype, shape)\n    if key not in _SHIFT_TENSORS:\n        _SHIFT_TENSORS[key] = torch.tensor(list(values), device=device, dtype=dtype).reshape(shape)\n    return _SHIFT_TENSORS[key]\n\nTORCH_COMPATIBLE_QTYPES =')

# 2. Replace the inline torch.tensor calls with get_shift
# Q6_K
code = code.replace('qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))',
                    'qh.reshape((n_blocks, -1, 1, 32)) >> get_shift((0, 2, 4, 6), d.device, torch.uint8, (1, 1, 4, 1))')

# Q5_K
code = code.replace('ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))',
                    'ql = qs.reshape((n_blocks, -1, 1, 32)) >> get_shift((0, 4), d.device, torch.uint8, (1, 1, 2, 1))')
code = code.replace('qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))',
                    'qh = qh.reshape((n_blocks, -1, 1, 32)) >> get_shift((0, 1, 2, 3, 4, 5, 6, 7), d.device, torch.uint8, (1, 1, 8, 1))')

# Q4_K
code = code.replace('qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))',
                    'qs = qs.reshape((n_blocks, -1, 1, 32)) >> get_shift((0, 4), d.device, torch.uint8, (1, 1, 2, 1))')

# Q3_K
code = code.replace('lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))',
                    'lscales.reshape((n_blocks, 1, 8)) >> get_shift((0, 4), d.device, torch.uint8, (1, 2, 1))')
code = code.replace('hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))',
                    'hscales.reshape((n_blocks, 1, 4)) >> get_shift((0, 2, 4, 6), d.device, torch.uint8, (1, 4, 1))')
code = code.replace('ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))',
                    'ql = qs.reshape((n_blocks, -1, 1, 32)) >> get_shift((0, 2, 4, 6), d.device, torch.uint8, (1, 1, 4, 1))')
code = code.replace('qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))',
                    'qh = hmask.reshape(n_blocks, -1, 1, 32) >> get_shift((0, 1, 2, 3, 4, 5, 6, 7), d.device, torch.uint8, (1, 1, 8, 1))')

# Q2_K
code = code.replace('shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))',
                    'shift = get_shift((0, 2, 4, 6), d.device, torch.uint8, (1, 1, 4, 1))')

# IQ4_NL
code = code.replace('qs.reshape((n_blocks, -1, 1, block_size//2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))',
                    'qs.reshape((n_blocks, -1, 1, block_size//2)) >> get_shift((0, 4), d.device, torch.uint8, (1, 1, 2, 1))')

# IQ4_XS
code = code.replace('shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))',
                    'shift_a = get_shift((0, 4), d.device, torch.uint8, (1, 1, 2))')
code = code.replace('shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))',
                    'shift_b = get_shift(tuple([2 * i for i in range(256 // 32)]), d.device, torch.uint8, (1, -1, 1))')

with open(path, 'w') as f:
    f.write(code)

print("dequant.py restored successfully!")
