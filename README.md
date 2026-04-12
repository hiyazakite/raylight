# Raylight

## Fork Enhancements (hiyazakite)

This fork is a major overhaul of komikndr's Raylight (based on v0.16.0). It is **not compatible** with the upstream repository due to deep architectural changes. Tested on Linux (dockerized ComfyUI, 4 × 3090).

A full technical breakdown is in [improvements.md](improvements.md).

---

### Architecture — `distributed_worker` → `distributed_actor`

The runtime is rewritten around a `distributed_actor` package replacing the old `distributed_worker`. Key additions:
- `RayActor` with a `LoadedModel` slot and a polymorphic `model_context` system (FSDP, TP, GGUF, BnB, lazy-tensor, VAE).
- `ActorPool` for managing actor groups with fresh-actor detection.
- Dedicated managers: `SamplerManager`, `LoraManager`, `VaeManager`, `IDLoraManager`.
- Global monkey-patches (`LowVramPatch`, `calculate_weight`) applied exactly once per process.

---

### Memory — Pinned Caches & Host IPC

**RAM model copies** — the original repo loaded one full model into RAM per worker (N workers = N × model_size RAM). The fork eliminates this for data-parallel replicas: all actors on the same host share **one** `/dev/shm` pinned-RAM buffer (`SharedPinnedParamCache`), built cooperatively in parallel (round-robin param assignment, no locking). TP and FSDP workers keep private caches because their shards differ (`ContiguousPinnedCache` / `FSDPShardPinnedCache`). Model weights stay in pinned (page-locked) RAM as a DMA-ready staging buffer; VRAM is freed via `UntypedStorage.resize_(0)` and restored via `resize_(nbytes)` — all tensor views update automatically. **Hot-reload** after VRAM eviction hits pinned RAM rather than disk, restoring a multi-GB model in sub-second. Partial (watermark) eviction allows shedding only the model's tail blocks to make room for a VAE load without a full offload.

**Host IPC** (`ipc/`): file-mmap and POSIX-SHM backends with a lifecycle state machine (`PENDING→WRITTEN→READY→CONSUMED→RELEASED`), per-artifact metadata, memory stats collection, and stale-artifact cleanup at startup. Used for passing VAE decode outputs between processes without going through Ray's object store.

**Native allocator** (`lib/raylight_alloc.so`): C interceptor to redirect large host allocations to RAM-backed paths, reducing NUMA-crossing copies.

**`SafetensorMmapWrapper`** (`expansion/comfyui_lazytensors/`): streams mmap'd weights one tensor at a time to the GPU with `non_blocking=True`, eliminating the full-state-dict RAM spike.

---

### Tensor Parallelism

New `tensor_parallel.py` (794 lines) implements Megatron-LM–style TP:
- `TensorParallelState` — `initialize(tp_size)` creates contiguous-rank TP subgroups; module-level helpers `get_tp_group/rank/size`.
- `TPLinear` — column- and row-parallel linear with INT8 fast path, optional TP compression, LoRA/DoRA support.
- `TPAttention` — GQA-aware head sharding.
- `TPMLP` — SwiGLU MLP with column-parallel gate+up and row-parallel down.
- `gather_tensor_along_dim` — all-gather with padding for non-divisible sizes; normalises negative `dim` before calling PyTorch's `funcol_all_gather` (fix for PyTorch ≥2.11 `_maybe_view_chunk_cat` shape bug).
- `sequence_parallel.py` — Ulysses all-to-all scatter/gather over sequence dimension.

**TP communication compression** (`tp_compress.py`): TurboQuant — signed-permutation or WHT rotation + Lloyd-Max optimal group quantization (2/3/4 bit) + optional step-to-step residual error feedback. Triton kernel backend in `tp_compress_triton.py`.

**GGUF TP linear** (`tp_linear_factory.py`): `TPGGUFLinear` stores quantized GGML byte shards and dequantizes on-the-fly per forward pass, preserving GGML VRAM savings across TP ranks.

---

### FSDP

**`FSDPShardRegistry`** (`comfy_dist/fsdp_registry.py`): class-level registry dispatching `shard_model_fsdp2()` per ComfyUI model type. Registered: Flux, WAN21/22, Chroma, ChromaRadiance, QwenImage, HunyuanVideo, Lumina, Lightricks.

All per-model `diffusion_models/*/fsdp.py` files are fully implemented (reference only had stubs). They handle GGUF-safe, LoRA-streaming bottom-up FSDP2 sharding with correct `ignored_params` sets for quantization scales.

`fsdp_utils.py` trimmed from 603 to 161 lines — complex tree traversal moved into per-model files.

---

### LoRA / Weight Adapters

**TP-aware offset patching** (`comfy_dist/lora.py`): when a patch offset exceeds the local shard's size, the fork computes the intersection with the rank's column shard, skips ranks with no overlap, and narrows the diff tensor with `tp_diff_slice`.

**Fused multi-LoRA helpers** (`comfy_dist/model_patcher.py`): `_extract_lora_ab` batches LoRA patches into fused `(A, B, dora_scale)` triplets; `_compute_loha_delta` and `_compute_lokr_delta` compute full deltas for LoHa/LoKr adapters. `LowVramPatch` correctly handles non-standard dtypes via stochastic rounding.

**New adapter types**: `glora.py` (generalized LoRA), `oft.py` (Orthogonal Fine-Tuning), `boft.py` (Block OFT).

---

### INT8 Quantization (`comfy_extra_dist/int8/`)

W8A8 inference via `torch._int_mm` with tensorwise and axiswise quantization, stochastic rounding for LoRA deltas, optional Triton fast-path kernels, distributed ops for multi-rank inference, and ComfyUI nodes `RayINT8Loader` / `RayINT8LoRA`.

---

### Configuration (`config.py`)

Explicit frozen-dataclass hierarchy: `ExecutionStrategy` (holds all parallelism degrees, FSDP flags, attention backend, TP compression settings with `.is_tp/.is_fsdp/.is_xdit` properties), `RaylightAttnType` (TORCH, FLASH_ATTN, FLASH_ATTN_3, SAGE_*, AITER_ROCM), `CompactCompressType`, and sub-configs `ClusterConfig`, `DeviceConfig`, `DebugConfig`, `SystemConfig`.

---

### ComfyUI Nodes & Samplers

- `XFuserSamplerCustom` / `DPSamplerCustom` return two latents: `output` + `denoised_output`.
- Samplers call `_orig_unload_all_models()` (not the monkey-patched version) to free main-process VRAM while preserving actor pinned caches for hot-reload.
- Actors receive `reload_model_if_needed` and `reapply_loras_for_config` before each sampling run.
- On failure, `clear_ray_cluster()` cleans up actors to prevent stuck state.
- New nodes: `RayLTXFFNChunker`, block-cache nodes (CacheDiT), `RayINT8Loader`, `RayINT8LoRA`.
- `unload_all_models` monkey-patched to run stale IPC cleanup without disturbing actor pinned caches.

---

### ID-LoRA LTXAV Integration

`RayIDLoraKSampler` enables identity-consistent audio-video generation on LTXAV with a custom multi-guidance denoising loop:
- **Three independent CFG scales**: `video_guidance_scale`, `audio_guidance_scale`, `identity_guidance_scale` — applied to their respective conditioning streams inside `apply_model`.
- **Reference audio injection**: optional `reference_audio_latent` (speaker identity from `LTXVAudioVAEEncode`) is injected as tokens at negative temporal positions before the patchified audio sequence, then trimmed from output — no retraining of ComfyUI internals required.
- **STG (Skip-T Guidance)**: optional per-block output zeroing for structured guidance (`stg_scale`, `stg_blocks`).
- **AV bimodal guidance**: cross-modal coefficient for joint audio-video conditioning (`av_bimodal_scale`).
- **`idlora_patches.py`**: monkey-patches `LTXAVModel._process_input` at first call to add `ref_audio` support (replicating ComfyUI PR f02190a). Idempotent — skipped automatically if the upstream PR is already merged.
- Returns `(output, denoised_output)` as NestedTensors compatible with `LTXVSeparateAVLatent`.

---

### CompactFusion — Activation Delta Compression

`CompactAttentionBackend` reduces ring-attention all-gather bandwidth by exploiting the slow inter-step activation changes in diffusion models:
- **Delta transmission**: each rank caches the last reconstructed activation (`baseline`); only `delta = activation − baseline` is all-gathered and decompressed.
- **Error-feedback loop**: both sender and receiver update their baseline with the same reconstructed delta, preventing drift accumulation across steps.
- **Compression modes**: 1-bit binary (sign packed into uint8 + low-rank scale matrix), INT2/INT4 quantisation, explicit low-rank (`LOW_RANK` / `LOW_RANK_AWL`), or `WARMUP` (full activation for the first N steps while baselines stabilise).
- **Triton fastpath**: a fused kernel computes delta, 1-bit-quantises with U/V scale factors, and writes back the updated baseline in a single launch (requires `fastpath=True`, `residual=1`, `ef=True`).
- **Subspace iteration**: low-rank scale approximation uses `@torch.compile` power iteration with Q-matrix warm-start across steps.
- Selected via `RaylightAttnType.COMPACT` in `RaylightConfig`; tunable via `RaylightConfig.compact` (`warmup_steps`, `delta_compression`, `kv_cache_quant_bits`, etc.).

---

### Fixes

- `gather_tensor_along_dim`: negative `dim` normalised before `funcol_all_gather` — fixes `RuntimeError: shape [...] is invalid` on PyTorch ≥2.11.
- `DTensor` import path updated: `torch.distributed._tensor` → `torch.distributed.tensor._api`.
- `align_model_to_cuda()`: iterates params/buffers one-by-one with `non_blocking=True` to avoid VRAM spikes from bulk `.to("cuda")` on lazy tensors.
- GGUF `dequantize_tensor`: `@torch_compiler_disable` decorator + subclass strip via `as_subclass(torch.Tensor)` + compile-friendly `_const_like()` constants.
- CFG registry path fix: `diffusion_models.qwen` → `diffusion_models.qwen_image`.
- WAN21/HuMo USP injection rewritten to patch `audio_cross_attn` at class level rather than per-block.
- Arbitrary number of GPU support for Lumina models.
