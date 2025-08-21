# Raylight

Raylight. Using Ray Worker to manage multi GPU sampler setup. With XDiT XFuser to implement sequence parallelism


## RTM, Known Issues
- Scroll further down for the installation guide.
- If NCCL communication fails before running (e.g., watchdog timeout), set the following environment variables:
  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_SHM_DISABLE=1
  ```
- Windows is not tested; I only have access to Linux cloud multi-GPU environments.
- If the initial model is larger than your VRAM, there is a high chance of an OOM error. This is currently the top priority fix.
- The tested model is the WanModel variant. The next model to be supported will be determined by usage popularity (Flux, Qwen, Hunyuan).
- Non-DiT models are not supported.
- Target hardware is 16GB vram and above. I will try to lower down to 12GB, but no promise for now
- Tested on PyTorch 2.7 - 2.8 CU128
- FLASH ATTENTION IS A MUST IF USING USP
- Example WF just open from your comfyui menu and browse templates

## Supported Models

| Model             | USP | FSDP |
|-------------------|-----|------|
| Wan 1.3B T2V      | ✅  | ✅   |
| Wan 14B T2V       | ✅  | ✅   |
| Wan 14B I2V       | ❓  | ❓   |
| Flux Dev          | ❌  | ✅   |
| Flux Konteks      | ❌  | ❓   |
| Flux ControlNet   | ❌  | ❌   |
| Qwen Image        | ❌  | ❓   |
| Hunyuan Video     | ❌  | ❓   |

**Legend:**
- ✅ = Supported
- ❌ = Not currently supported
- ❓ = Maybe work?


## Wan T2V 1.3B
<img width="1918" height="887" alt="image" src="https://github.com/user-attachments/assets/57b7cdf5-ebd5-4902-bccd-fa7bbfe9ef8b" />

https://github.com/user-attachments/assets/40deddd2-1a87-44de-98a5-d5fc3defbecd

## Wan T2V 14B on RTX 2000 ADA ≈ RTX 4060 TI 16GB
<img width="1117" height="716" alt="Screenshot 2025-08-20 125936" src="https://github.com/user-attachments/assets/b2ea1621-4be1-4925-8f4a-3ff9542c6415" />

## DEBUG Notes

### Wan T2V 1.3B (bf16) — 1×3090 24G

| Setup                        | VRAM (GB) |
|------------------------------|-----------|
| Non-Ulysses                  | 4.6       |
| Ulysses                      | 5.1       |
| XDiT Ring + Ulysses          | 9.2       |
| Ulysses + Sync Module FSDP1  | 9.7       |
| Ulysses + Unsync Module FSDP1| 5.2       |

---

### Wan T2V 14B (fp8) — 1×3090 24G

| Setup                        | VRAM (GB)        |
|------------------------------|------------------|
| Non-Ulysses                  | 15.5             |
| Ulysses                      | 15.9             |
| Ulysses + Sync Module FSDP1  | OOM (Pred. 47.3) |
| Ulysses + Unsync Module FSDP1| OOM (Pred. 31.8) |
| XDiT Ring + Ulysses          | OOM              |

---

### Wan T2V 14B (fp8) — 1×RTX 2000 ADA 16G
**Resolution:** 480×832 × 33F

| Setup   | VRAM (GB) | Speed            |
|---------|-----------|------------------|
| Normal  | OOM       | 22 it/s (before OOM) |

---

### Wan T2V 14B (fp8) — 2×RTX 2000 ADA 16G

| Setup           | VRAM (GB) / Device | Speed   |
|-----------------|--------------------|---------|
| Ulysses         | 15.8 (Near OOM)    | 11 it/s |
| FSDP2           | 12.8               | 19 it/s |
| Ulysses + FSDP2 | 10.25              | 12 it/s |

---

### Notes
- **FSDP OOM in single-device 14B model** caused by:
  - Sync Module FSDP: model is first loaded into CUDA, then sharded (not tested on dual GPU).
  - Lowest possible dtype for FSDP params is **bf16**, hence doubled size compared to fp8.
- **FSDP2** is now available and can do fp8 calculation, but needs scalar tensors converted into 1D tensors.


## Installation

1. Clone this repository under `ComfyUI/custom_nodes`.
2. `cd raylight`
3. Install dependencies:
   your_python_env - pip install -r requirements.txt
4. Install FlashAttention:
   - Option A (NOT recommended due to long build time):
     pip install flash-attn --no-build-isolation
   - Option B (recommended, use prebuilt wheel):
     For Torch 2.8:
       `wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl -O flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl`
       `pip install flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl`
     For other versions, check:
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/
5. Restart ComfyUI.
