# Raylight

Raylight. Using Ray Worker to manage multi GPU sampler setup. With XDiT-XFuser and FSDP to implement parallelism

## RTM, Known Issues
- Scroll further down for the installation guide.
- If NCCL communication fails before running (e.g., watchdog timeout), set the following environment variables:
  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_SHM_DISABLE=1
  ```
- Windows is not tested; I only have access to Linux cloud multi-GPU environments.
- The tested model is the WanModel variant. The next model to be supported will be determined by usage popularity (Flux, Qwen, Hunyuan).
- Non-DiT models are not supported.
- Example WF just open from your comfyui menu and browse templates

## GPU Architectures

### NVidia

1. **Ampere**: There is an issue with NCCL broadcast and reduction in FSDP on PyTorch 2.8.
   Please use the previous version instead. FSDP works successfully on Torch 2.7.1 CU128 for Ampere.
   Reference: https://github.com/pytorch/pytorch/issues/162057#issuecomment-3250217122

2. **Turing**: Not tested. Please use FlashAttn1 instead of FlashAttn2.

3. **Ada Lovelace**: There is also an issue with Torch 2.8 which when assigning
   `device_id` to `torch.dist_init_process_group()` cause OOM.
   In a mean time, you would see torch distributor complaining about device assigment, but other-
   than that it should be working fine.

4. **Blackwell**: Expected to work just like Ada Lovelace.

### AMD

1. **MI3XX** : User confirmed working on 8xMI300X using ROCm compiled PyTorch and Flash Attention


## Supported Models

**Wan**
| Model             | USP | FSDP |
|-------------------|-----|------|
| Wan2.1 14B T2V    | ✅  | ✅   |
| Wan2.1 14B I2V    | ✅  | ✅   |
| Wan2.2 14B I2V    | ✅  | ✅   |
| Wan2.2 14B I2V    | ✅  | ✅   |
| Wan2.1 1.3B T2V   | ✅  | ✅   |
| Wan2.2 5B TI2V    | ✅  | ✅   |
| Wan2.1 Vace       | ✅  | ❌   |


**Flux**
| Model             | USP | FSDP |
|-------------------|-----|------|
| Flux Dev          | ✅  | ✅   |
| Flux Konteks      | ✅  | ✅   |
| Flux Krea         | ✅  | ✅   |
| Flux ControlNet   | ❌  | ❌   |


**Qwen**
| Model             | USP | FSDP |
|-------------------|-----|------|
| Qwen Image        | ❌  | ❓   |
| Qwen Edit         | ❌  | ❓   |


**Hunyuan Video**
| Model             | USP | FSDP |
|-------------------|-----|------|
| Hunyuan Video     | ❌  | ❓   |

**Legend:**
- ✅ = Supported
- ❌ = Not currently supported
- ❓ = Maybe work?

**Notes:**
- Non standard Wan variant (Phantom, S2V, etc...) is not tested

## Scaled vs Non-Scaled Models

| Model       | USP | FSDP |
|-------------|-----|------|
| Non-Scaled  | ✅  | ✅   |
| Scaled      | ✅  | ⚠️    |

**Notes:**
- Scaled models use multiple dtypes inside their transformer blocks: typically **FP32** for scale, **FP16** for bias, and **FP8** for weights.
- Only Ada Lovelace and newer GPUs support **FP8 scaled FSDP2**.

## Attention

| Attention Variant    | Time (s) |
|----------------------|----------|
| sage_fp8             | 10.75    |
| sage_fp16_cuda       | 11.00    |
| sage_fp16_triton     | 11.17    |
| flash                | 11.24    |
| torch                | 11.36    |

**Notes:**
- Tested on Wan 2.1 T2V 14B 832x480 33 frame 2 RTX 2000 ADA



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


## Support
[PayPal](https://paypal.me/Komikndr)
Thanks for the support :) (I want to buy 2nd GPU (5060Ti) so i dont have to rent cloud GPU)
