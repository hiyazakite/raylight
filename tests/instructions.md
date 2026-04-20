# Running Tests

## Quick start

```bash
make test                   # runs all tests
make test-cuda              # runs GGUF CUDA kernel tests only
```

## Manual (inside development container)

```bash
export PYTHONPATH=$PYTHONPATH:/root/ComfyUI/custom_nodes/raylight/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvshmem/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib
export CUDA_VISIBLE_DEVICES=0

python3.13 -m pytest tests/ -v
```

> **Note:** pytest must be invoked from the raylight directory with `-c pyproject.toml`
> to avoid picking up `/root/ComfyUI/pytest.ini` as the rootdir.  Use absolute paths:

```bash
PYTHONPATH=/root/ComfyUI/custom_nodes/raylight/src:$PYTHONPATH \
  python3.13 -m pytest \
  -c /root/ComfyUI/custom_nodes/raylight/pyproject.toml \
  /root/ComfyUI/custom_nodes/raylight/tests/ -v
```

To run a single test file:

```bash
PYTHONPATH=/root/ComfyUI/custom_nodes/raylight/src:$PYTHONPATH \
  python3.13 -m pytest \
  -c /root/ComfyUI/custom_nodes/raylight/pyproject.toml \
  /root/ComfyUI/custom_nodes/raylight/tests/distributed_modules/test_qwen_image_tp.py -v
```

## Prerequisites

- Python 3.13 with torch 2.11+ (CUDA 13)
- Native extensions built: `make build`
- Dev dependencies installed: `pip install -e ".[dev]"`
