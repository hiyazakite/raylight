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

## Prerequisites

- Python 3.13 with torch 2.11+ (CUDA 13)
- Native extensions built: `make build`
- Dev dependencies installed: `pip install -e ".[dev]"`
