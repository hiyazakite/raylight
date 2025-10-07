# Raylight

Raylight. Using Ray Worker to manage multi GPU sampler setup. With XDiT-XFuser and FSDP to implement parallelism.

*"Why buy 5090 when you can buy 2x5070s"-Komikndr*

## UPDATE
- GGUF added thanks to [City96](https://github.com/city96/ComfyUI-GGUF), only in USP mode, not in FSDP
- Reworked the entire FSDP loader. Model loading should now be more stable and faster,
  as Raylight no longer kills active workers to reset the model state.
  Previously, this was necessary because Comfy could not remove FSDP models from VRAM, which caused memory leaks.
- No need to install FlashAttn.
- SageAttn is now supported.
- Full FSDP support for Qwen, Flux, and Hunyuan Video.
- Qwen USP can't do any square dimension, only 1280x1280 that's working, so pick any dim that's is not square
- Full LoRA support.
- FSDP CPU offload, analogous to block swap/DisTorch.

## What exactly is Raylight

Raylight is a parallelism node for ComfyUI, where the tensor of an image or video sequence
is split among GPU ranks. Raylight, as its partial namesake, uses [Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
to manage its GPU workers.

<img width="830" height="665" alt="image" src="https://github.com/user-attachments/assets/ea78f3d3-41cb-4620-b4b6-65f8229293b2" />

So how does it split among the ranks? It uses Unified Sequence Parallelism (USP), embedded inside
[XDiT](https://github.com/xdit-project/xDiT), a core library of Raylight that splits and allgathers tensors among GPU ranks.

Unfortunately, although it splits across GPUs, each GPU must still load the full model weight.
And let's be honest, most of us do not have a 4090 or 5090. In my opinion, buying a second 4070
is monetarily less painful than buying a 5090. This is where [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) comes in.
Its job is to split the model weights among GPUs.

**TLDR:** Raylight is multi-GPU nodes for Comfy, USP for splitting the work, and FSDP for splitting the model weights.

### Raylight vs MultiGPU vs ComfyUI Worksplit branch
- [MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
  Loads models selectively on specified GPUs without sharing workload.
  Includes CPU RAM offloading, which benefits single-GPU users.

- [ComfyUI Worksplit branch](https://github.com/comfyanonymous/ComfyUI/tree/worksplit-multigpu)
  Splits workload at the CFG level, not at the tensor level.
  Since most workflows use `CFG=1.0` like Wan with Lora, this approach provides limited use cases.

- **Raylight**
  Provides both tensor parallelism (USP) and model weight sharding (FSDP).
  Your GPUs will 100% being used at the same time.
  This enables efficient multi-GPU utilization and scales beyond single high-memory GPUs (e.g., RTX 4090/5090).


## RTM and Known Issues
- Scroll further down for the installation guide.
- If NCCL communication fails before running (e.g., watchdog timeout), set the following environment variables:
  ```bash
  export NCCL_P2P_DISABLE=1
  export NCCL_SHM_DISABLE=1
  ```
- **Windows** is in partial testing, switch to `dev` branch to test it. And scroll down below for more information
- The tested model is the WanModel variant. The next model to be supported will be determined by usage popularity (Flux, Qwen, Hunyuan).
- Non-DiT models are not supported.
- Example WF just open from your comfyui menu and browse templates

## Operation

### Mode

**Sequence Parallel**
This mode splits the sequence among GPUs, the full model will be loaded into each GPU.
Use the XFuser KSampler to increase the Ulysses degree according to the number of your GPUs,
while keeping the Ring degree at 1 for small systems.

<img width="834" height="437" alt="ValidUSP1" src="https://github.com/user-attachments/assets/c5430825-4db5-4b7d-aa44-b40d0ea0f516" />

---

**Data Parallel**
The full sequence will be processed independently on each GPU.
Use the Data Parallel KSampler. There are two options, enable FSDP, or disable all options in `Ray Init Actor`.
By disabling them, it will run in DP mode. Both FSDP and DP modes must have the Ulysses and Ring degrees set to
0.

FSDP will shard the weights, but each GPU will still work independently,
as the name suggests, Fully Sharded (Weight) Data Parallel.

<img width="892" height="638" alt="ValidDataParallel" src="https://github.com/user-attachments/assets/c9688e6b-7b0e-4b15-8279-2c94da46f78c" />

---

**Sequence + FSDP**
Activate FSDP, and set the Ulysses degree to the number of GPUs. Use the XFuser KSampler.

<img width="833" height="427" alt="ValidUSP" src="https://github.com/user-attachments/assets/9c5571ca-ae4c-4deb-97da-c3552ff43cea" />

---

### Side Notes
- **Rule of thumb**, if you have enough VRAM, just use USP, if not, enable the FSDP, and if that is still not enough,
  enable also the FSDP CPU Offload.
- FSDP CPU Offload is intended for systems with very low VRAM, though it will come with a performance hit work akin to
  DisTorch from MultiGPU.

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
| Qwen Image/Edit   | ✅  | ✅   |
| ControlNet        | ❌  | ❌   |


**Hunyuan Video**
| Model             | USP | FSDP |
|-------------------|-----|------|
| Hunyuan Video     | ✅  | ✅   |
| ControlNet        | ❌  | ❌   |

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

## Qwen Image 20B on RTX 2000 ADA ≈ RTX 4060 TI 16GB , 4x Playback speed up

https://github.com/user-attachments/assets/d5e262c7-16d5-4260-b847-27be2d809920


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
       ```bash
       wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl -O flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl
       ```
       ```bash
       pip install flash_attn-2.8.2+cu128torch2.7-cp311-cp311-linux_x86_64.whl`
       ```
     For other versions, check:
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/
5. Restart ComfyUI.

**ComfyUI Manager**
1. Find raylight in the manager and install it.

**Windows**
1. Only works for **PyTorch 2.7**, because of [pytorch/pytorch#150381](https://github.com/pytorch/pytorch/issues/150381)
2. POSIX and Win32 style paths can cause issues when importing **raylight**.
3. Recommended steps:
   - Manually clone the **Raylight** repo
   - Switch to the `dev` branch for now
   - Inside the top-most Raylight folder (where `pyproject.toml` exists), run:

   ```bash
   ..\..\..\python_embeded\python.exe -m pip install -r .\requirements.txt
   ..\..\..\python_embeded\python.exe -m pip install -e .
   ```
4. Highly experimental — please open an issue if you encounter errors.
5. Advisable to run in WSL since windows does not have NCCL support from PyTorch, raylight will run using GLOO,
   which is slower than NCCL



## Support
[PayPal](https://paypal.me/Komikndr)
Thanks for the support :) (I want to buy 2nd GPU (5060Ti) so i dont have to rent cloud GPU)
[RunPod](https://runpod.io?ref=yruu07gh)
