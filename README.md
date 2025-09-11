# Raylight

Raylight. Using Ray Worker to manage multi GPU sampler setup. With XDiT-XFuser and FSDP to implement parallelism.

*"Why buy 5090 when you can buy 2x5070s"-Komikndr*

## UPDATE
- No need to install FlashAttn.
- SageAttn is now supported.
- Partial support for USP Flux.
- Full FSDP support for Qwen and Flux.
- Flux USP and Qwen USP is in partial testing, you can try but it will suck.
- Full LoRA support.
- FSDP CPU offload, analogous to block swap.

## RTM and Known Issues
- Scroll further down for the installation guide.
- **Rule of thumb**, if have enough VRAM just use USP, if not, FSDP, if it still not enough, use FSDP CPU Offload
- FSDP CPU Offload is for ultra low VRAM, there will be a performance hit of course
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

2. **Turing**: Not tested. Please use FlashAttn1 instead of FlashAttn2 or Torch Attn.

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
| Qwen Image/Edit   | ❓  | ❓   |
| ControlNet        | ❌  | ❌   |


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
- Raylight FSDP can work with scaled model, but it really does not like it. Since FSDP shards must have uniform dtype, if not it will not be sharded.

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
- **FSDP2** is now available and can do fp8 calculation, but needs scalar tensors converted into 1D tensors.


## Installation

**Manual**
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

**ComfyUI Manager**
1. Find raylight in the manager and install it.



## Support
[PayPal](https://paypal.me/Komikndr)
Thanks for the support :) (I want to buy 2nd GPU (5060Ti) so i dont have to rent cloud GPU)
