# Raylight Fork — Comprehensive Technical Improvements

Reference: `reference/raylight-original`  
Fork: `src/raylight`  
Diff basis: full file-tree comparison + source inspection of changed modules.

**Stats**: reference had 100 files; fork has 350 files — 29 removed (superseded), 279 added, 40 common files with significant changes.

---

## 1. Architecture — `distributed_worker` → `distributed_actor`

The single biggest structural change. The original `distributed_worker/` package contained all runtime logic in two files (`ray_worker.py`, `parallel_manager.py`). This is replaced by a fully decomposed `distributed_actor/` package.

**Removed (reference only)**
- `distributed_worker/__init__.py`
- `distributed_worker/ray_worker.py`
- `distributed_worker/parallel_manager.py`
- `distributed_worker/managers/distribute_config_manager.py`
- `distributed_worker/utils.py`

**Replaced by (fork only)**
- `distributed_actor/actor.py` — `RayActor` class. Monkey-patches ComfyUI's `LowVramPatch` and `calculate_weight` exactly once per process (`_patch_once()`). Exposes a `model` property facade over a `LoadedModel` slot. Manages actor lifecycle integration with pinned caches.
- `distributed_actor/actor_pool.py` — `ActorPool` abstraction for managing groups of Ray actors with fresh-actor detection.
- `distributed_actor/actor_config.py` — `ActorConfig` dataclass encapsulating per-rank configuration (rank, world_size, TP degree, etc.).
- `distributed_actor/loaded_model.py` — `LoadedModel` slot wrapping a model patcher and its associated context.
- `distributed_actor/comm_test.py` — NCCL connectivity test helper (`ray_nccl_tester`).
- `distributed_actor/model_context/` — Polymorphic model context system with implementations for: `fsdp.py`, `tp.py`, `gguf.py`, `gguf_tp.py`, `bnb.py`, `lazy_tensor.py`, `vae.py`. Each handles loading, reloading, LoRA application and teardown for its specific backend.
- `distributed_actor/managers/` — Dedicated manager classes: `SamplerManager`, `LoraManager`, `VaeManager`, `IDLoraManager`.

---

## 2. Host IPC System (`src/raylight/ipc/`)

Entirely new. Zero counterpart in the reference.

**Purpose**: Non-GPU inter-process memory sharing between the main ComfyUI process and Ray actors. Used primarily to pass VAE decode outputs back to the host without serializing through Ray's object store.

**Backends**
- `backends/file_mmap.py` — `FileMmapHostIpcBackend`: mmap-backed files, portable on all Linux/Mac systems.
- `backends/posix_shm.py` — `PosixShmBackend`: `shm_open` / `mmap` using `/dev/shm`, faster but requires `POSIX_SHARED_MEMORY` support and sufficient `/dev/shm` size.
- `resolver.py` — `build_default_host_ipc_service()` auto-selects the best available backend. `resolve_file_mmap_root()` finds a RAM-backed path.

**Lifecycle management**
- `types.py` — `HostIpcLifecycleState` enum: `PENDING → WRITTEN → READY → CONSUMED → RELEASED`.
- `lifecycle.py` — `assert_valid_transition()` / `can_transition()` enforce allowed state machine transitions.
- `cleanup.py` — `cleanup_stale()` / `cleanup_all_stale()` remove orphaned artifacts from crashed sessions. Called once at import time in `nodes.py`.
- `memory_stats.py` — `collect_all_stats()`, `collect_ipc_artifact_stats()`, `collect_backend_stats()` for observability.

**VAE adapter**
- `vae_ipc.py` — `begin_vae_decode_job()` / `release_vae_decode_job()` lifecycle helpers. Called from `nodes.py` to track active decode slots and prevent resource leaks.

---

## 3. Native Allocator & Pinned-Cache System

### 3a. Native allocator (`src/raylight/lib/`)

New in fork. `raylight_alloc.so` is a small native shared library with a C allocator-interceptor (`src/raylight_alloc.c`) controlled by `alloc_interceptor.py`. Its purpose is to redirect large host-memory allocations to RAM-backed paths (e.g. `/dev/shm`) to give the OS better placement control and reduce NUMA-crossing copies on multi-socket machines.

### 3b. Pinned-cache system (`src/raylight/distributed_modules/pinned_cache/`)

Five distinct cache implementations sharing a common base interface:

```
cache.build(module)          # snapshot CUDA → pinned RAM (lazy)
cache.offload_to_cpu(module) # refresh snapshot + free VRAM storage
cache.reload_to_cuda(module) # re-allocate VRAM + copy pinned → CUDA
cache.built                  # bool
cache.param_count()
cache.pinned_ram_bytes()
```

The key primitive is `UntypedStorage.resize_(0)` to truly free the CUDA allocation while keeping the tensor's data pointer alive, and `resize_(nbytes)` to re-allocate it. All views referencing the storage automatically follow.

| Class | Description |
|---|---|
| `PinnedParamCache` | Single-rank; private pinned RAM (cudaHostAlloc) per process |
| `SharedPinnedParamCache` | Cross-process `/dev/shm` buffer + `cudaHostRegister`; useful for data-parallel replicas holding identical weights |
| `FSDPShardPinnedCache` | FSDP2 DTensor-aware; operates on per-rank local shards extracted via `.to_local()` |
| `ContiguousPinnedCache` | Packs all parameters into a single contiguous pinned buffer for bulk H2D transfer via a single `cudaMemcpy` |
| `DictPinnedCache` | Dictionary-keyed variant for selective caching |

The cache lifecycle is integrated into `nodes.py` via `release_all_pinned_caches()`, which gracefully handles dead/crashed actors and cleans up orphaned host-memory artifacts.

---

## 4. Configuration System (`src/raylight/config.py`)

Replaced an implicit, scattered config pattern with an explicit frozen-dataclass hierarchy. No counterpart in the reference.

**Key types**

- `RaylightAttnType` (Enum) — `TORCH`, `FLASH_ATTN`, `FLASH_ATTN_3`, `SAGE_AUTO_DETECT`, `SAGE_FP16_TRITON`, `SAGE_FP16_CUDA`, `SAGE_FP8_CUDA`, `SAGE_FP8_SM90`, `AITER_ROCM`.
- `CompactCompressType` (Enum) — `WARMUP`, `SPARSE`, `BINARY`, `INT2`, `INT2_MINMAX`, `INT4`, `IDENTITY`, `LOW_RANK`, `LOW_RANK_Q`, `LOW_RANK_AWL`.
- `ExecutionStrategy` (frozen dataclass) — Holds `ulysses_degree`, `ring_degree`, `cfg_degree`, `tensor_parallel_degree`, `fsdp_enabled`, `fsdp_cpu_offload`, `fsdp_parallel_load`, `attention_backend`, `attention_type`, `ring_impl`, plus TP compression settings (`tp_allreduce_compress`, `tp_compress_bits`, `tp_compress_group_size`, `tp_compress_residual`, `tp_compress_rotation`). Properties: `.is_fsdp`, `.is_tp`, `.is_xdit`, `.total_parallel_degree`.
- `CompactConfig` (frozen dataclass) — Activation-compression system settings.
- `ClusterConfig`, `DeviceConfig`, `DebugConfig`, `SystemConfig` — Structured sub-configs for the full `RaylightConfig`.

---

## 5. Tensor Parallelism (`src/raylight/distributed_modules/tensor_parallel.py`)

Entirely new. 794 lines. Megatron-LM–style TP with compile support.

**`TensorParallelState`** — class-level state machine managing TP process groups:
- `initialize(tp_size, pg=None)` — creates contiguous-rank TP subgroups (e.g. tp_size=2 on 8 GPUs → groups `[0,1],[2,3],[4,5],[6,7]`). Supports passing a pre-built `ProcessGroup` from a `DeviceMesh`.
- `get_group()`, `get_rank()`, `get_size()` — module-level convenience functions.

**`gather_tensor_along_dim(tensor, dim, num_splits, full_size)`** — All-gather across TP ranks with padding support for non-divisible sizes and automatic trim. This function includes the PyTorch ≥2.11 negative-dim fix (normalizes `dim < 0` to positive before calling `funcol_all_gather` to avoid `_maybe_view_chunk_cat` shape corruption).

**`TPLinear`** — Column-parallel and row-parallel linear layer:
- `parallelism="column"` — shards `weight[out_features/tp_size, in_features]`, optionally gathers output.
- `parallelism="row"` — shards `weight[out_features, in_features/tp_size]`, all-reduces results.
- LoRA support: fused `lora_down`/`lora_up` application, DoRA scale tracking, LoHa/LoKr full-delta path.
- INT8 fast path: uses `int8_forward_dynamic` / `int8_forward_dynamic_per_row` when `_HAS_INT8_KER`.
- Compression: optional `tp_compress.TPCompressor` to compress the partial sums before all-gather.
- Runtime TP size detection via `_tp_size_runtime`.

**`TPAttention`** — GQA-aware head sharding. Handles unequal Q/K/V head counts (GQA), splits heads across ranks, all-gathers outputs.

**`TPMLP`** — SwiGLU MLP sharding: splits gate and up projections column-parallel, down projection row-parallel.

**`TPRMSNormAcrossHeads`** — Distributed RMSNorm that operates on head-sharded tensors.

**`sequence_parallel.py`** — Ulysses-style sequence parallelism utilities (all-to-all gather/scatter over sequence dimension).

---

## 6. TP Communication Compression (`src/raylight/distributed_modules/tp_compress.py`)

New. TurboQuant-based compression of TP partial sums before all-gather to reduce NCCL bandwidth.

**`TPCompressConfig`** — `mode` (`none`/`fp8`/`turboquant`), `bits` (2/3/4), `group_size`, `use_residual`, `rotation` (`signperm`/`wht`).

**Rotation methods**
- `SignPermRotation` — applies a random sign-permutation matrix (fast, O(n)). Decorrelates activations to bring their distribution closer to Gaussian before quantization.
- `WHTRotation` — Walsh-Hadamard Transform rotation (O(n log n)), better decorrelation quality.

**Quantization**
- Lloyd-Max optimal centroids for 2-bit (4 levels) and 3-bit (8 levels) quantization of N(0,1) marginals stored as constants.
- Group quantization: splits the tensor into `group_size`-element groups each with an independent scale.
- Step-to-step residual caching (Phase 2): error-feedback loop to accumulate compression error across denoising steps.

Also `tp_compress_triton.py` — Triton kernel backend for the same operations.

---

## 7. TP Linear Factory for GGUF (`src/raylight/distributed_modules/tp_linear_factory.py`)

New. `TPGGUFLinear` — TP-parallel linear that stores quantized GGML byte shards and dequantizes on-the-fly during each forward pass, preserving the VRAM savings of GGML quantization across TP ranks.

Handles block-quantisation layout: GGML stores weights as 1-D raw bytes. `tp_linear_factory` reshapes them to `[out_features, blocks_per_row, type_size]` and then narrows per rank. Shard-boundary alignment is enforced to `block_size` multiples for the input-dim case.

---

## 8. FSDP Registry & Per-Model FSDP (`src/raylight/comfy_dist/fsdp_registry.py`)

**Reference**: had `comfy_dist/fsdp_utils.py` (603 lines) containing inline sharding logic, all `collect_*_ignored_params` helpers, `fully_shard_bottom_up`, `load_from_full_model_state_dict`, and `freeze_and_detect_qt` — all in one file. Also had a `fsdp_registry_unused.py` placeholder.

**Fork**: `fsdp_utils.py` is now 161 lines, only containing `prepare_fsdp_model_for_sampling()` (state-dict injection + LoRA baking coordination). All the complex tree-traversal sharding logic has moved into per-model `diffusion_models/*/fsdp.py` files.

`fsdp_registry.py` adds `FSDPShardRegistry` — a class-level registry mapping ComfyUI model base classes to their `shard_model_fsdp2()` functions. Subclasses must be registered before parents to ensure correct `isinstance()` dispatch. Currently registered: Flux, WAN21/WAN22, Chroma, ChromaRadiance, QwenImage, HunyuanVideo, Lumina, Lightricks (LTXV/LTXAV).

All per-model `fsdp.py` files (fork-only; reference only had `fsdp_unused.py` stubs) implement the actual GGUF-safe, LoRA-streaming, bottom-up FSDP2 sharding with proper `ignored_params` sets for quantization scales.

---

## 9. LoRA / Weight Adapter System

### 9a. TP-aware LoRA patching (`src/raylight/comfy_dist/lora.py`)

The `calculate_weight` function now handles TP-sharded weights:
- When a patch has an `offset` that exceeds the local weight's dimension size (indicating a TP column-parallel shard), it computes the intersection of the patch range with the rank's local shard range.
- `tp_diff_slice` is computed and attached to the weight object to allow the diff tensor to be narrowed to match exactly.
- Ranks with no overlap with the patch range skip the patch entirely (`continue`).
- The `"diff"` patch type now uses `tp_diff_slice` to narrow the diff tensor before the shape-mismatch check.

### 9b. Fused multi-LoRA merging (`src/raylight/comfy_dist/model_patcher.py`)

The fork's `model_patcher.py` grew from ~60 lines to ~500+ lines. Key additions:

**`_extract_lora_ab(patches)`** — Extracts all `(A, B, dora_scale)` triplets from a list of active LoRA patches. Handles: standard LoRA, DoRA (dora_scale), multi-LoRA batch merging. Bails out on: LoCon (mid tensor), offset patches, custom functions, mixed DoRA sets.

**`_compute_loha_delta(v, strength)`** — Computes LoHa delta: `strength * alpha * (w1a @ w1b) ⊙ (w2a @ w2b)` (Hadamard product). Bails on CP decomposition (4D t1/t2 required).

**`_compute_lokr_delta(v, strength)`** — Computes LoKr delta: `strength * alpha * kron(w1, w2)`. Handles `w1a/w1b` and `w2a/w2b` decomposed forms. Bails on 4D w2 (conv case).

**`LowVramPatch.__call__`** — Now correctly handles non-standard dtypes: if `intermediate_dtype` is not in `[float32, float16, bfloat16]`, it upcasts to float32, calculates weight, then stochastically rounds back to the original dtype.

### 9c. Expanded weight adapter suite

Reference had: `loha.py`, `lokr.py`, `lora.py`, `base.py`.  
Fork adds: `glora.py` (generalized LoRA), `oft.py` (Orthogonal Fine-Tuning), `boft.py` (Block OFT).

---

## 10. GGUF & Lazy Tensor Enhancements

### 10a. GGUF `dequant.py`

- Added `@torch_compiler_disable()` decorator to `dequantize_tensor` to prevent Dynamo tracing failures when `as_subclass(torch.Tensor)` is called during compilation.
- `dequantize_tensor` now strips the subclass before dequantization: `tensor.as_subclass(torch.Tensor)` bypasses `__torch_function__` overhead.
- Replaced `torch.arange(...)` / `torch.tensor([...])` constant constructions inside dequant kernels with `_const_like()` — a compile-friendly helper that uses `torch.arange` for arithmetic sequences and `torch.tensor` for irregular ones, avoiding graph breaks.

### 10b. Lazy Tensor Loader (`src/raylight/expansion/comfyui_lazytensors/`)

Entirely new module. `SafetensorMmapWrapper` wraps an mmap'd safetensor state dict and provides `stream_to_model(model, device)` — a streaming per-tensor transfer that avoids the RAM spike from cloning the full state dict. Parameters are transferred one by one with `non_blocking=True` and a single `torch.cuda.synchronize()` at the end.

---

## 11. INT8 Quantization System (`src/raylight/comfy_extra_dist/int8/`)

Entirely new. No counterpart in reference.

- `int8_quant.py` — `Int8TensorwiseOps` implementing W8A8 inference via `torch._int_mm`. Quantization functions: `quantize_int8_tensorwise`, `quantize_int8_axiswise`. LoRA support: `stochastic_round_int8_delta`. Optional Triton fast-path detected at import time.
- `int8_fused_kernel.py` — Triton kernels: `triton_int8_linear`, `triton_int8_linear_per_row`, `triton_quantize_rowwise`.
- `int8_distributed_ops.py` — Distributed int8 operations for multi-rank int8 inference.
- `nodes_int8_loader.py` — ComfyUI node to load models in INT8 mode.
- `nodes_int8_lora.py` — ComfyUI node handling LoRA application over INT8-quantized weights.

---

## 12. `nodes.py` — Actor pool, IPC integration, monkey-patching

The fork's `nodes.py` gained ~750 lines of new logic on top of the reference's ~250:

**Actor tracking + pinned cache lifecycle**: `_active_actors` list, `_register_actors()`, `release_all_pinned_caches()` with dead-actor handling. Called on explicit model unload.

**`unload_all_models` monkey-patch**: The fork replaces `comfy.model_management.unload_all_models` with `_patched_unload_all_models` that runs stale-shm cleanup but deliberately does NOT release pinned caches (hot-reload is the point). The original is saved as `_orig_unload_all_models` and used by sampler nodes to free main-process VRAM without disturbing actor caches.

**IPC service singleton**: `_get_host_ipc_service()` lazy-initializes the backend on first VAE decode. Stale IPC cleanup (`_ipc_cleanup_all_stale()`) runs at module import time.

**Actor pool integration**: `nodes.py` now imports from `distributed_actor.actor_pool.ActorPool` instead of `distributed_worker.ray_worker.make_ray_actor_fn`.

**new imports**: `RaylightConfig`, `ExecutionStrategy`, `CompactConfig`, `ClusterConfig`, `DeviceConfig`, `DebugConfig`, `SystemConfig`, `RaylightAttnType`, `CompactCompressType` — all from the new config module.

---

## 13. `comfy_dist/fsdp_utils.py` — FSDP model preparation

**Reference** (603 lines): contained the full bottom-up FSDP tree traversal (`collect_bottom_up_shard_order`, `_collect_leaf_parent_targets`, `_add_ancestors_to_root`), all `collect_*_ignored_params` helpers, `fully_shard_bottom_up`, `load_from_full_model_state_dict`, `freeze_and_detect_qt`, and `_should_materialize_unsharded_param`.

**Fork** (161 lines): only contains `prepare_fsdp_model_for_sampling(work_model, config, state_dict)` — a slim coordinator that injects a saved state dict if the patcher doesn't have one yet. The complex sharding logic moved to per-model `fsdp.py` files dispatched through `FSDPShardRegistry`. Uses `init_device_mesh` from `torch.distributed.device_mesh` for explicit mesh construction.

---

## 14. `distributed_modules/utils.py`

- Import path fix: `from torch.distributed._tensor import DTensor` → `from torch.distributed.tensor._api import DTensor` (API moved in newer PyTorch).
- New `align_model_to_cuda(model)` function — iterates parameters and buffers one-by-one with `non_blocking=True` to avoid VRAM spikes from bulk `.to("cuda")` on lazy tensors. Skips `FlatParameter` (already managed by FSDP). Single `cuda.synchronize()` at the end.

---

## 15. Custom Sampler Nodes (`comfy_extra_dist/nodes_custom_sampler.py`)

**Input type rename**: all `"ray_actors"` input slots renamed to `"actors"` — breaking change for saved workflows.

**`XFuserSamplerCustom`**:
- Now returns `(LATENT, LATENT)` — `output` + `denoised_output` — previously only returned `output`.
- Calls `_orig_unload_all_models()` (not the patched version) to free main-process VRAM while preserving actor pinned caches.
- Calls `actor.reload_model_if_needed.remote()` before sampling to trigger hot-reload from `/dev/shm` if VRAM was evicted.
- Calls `actor.reapply_loras_for_config.remote(lora_config_hash)` to sync LoRA state with actors.
- Wrapped in `try/except` that calls `clear_ray_cluster()` on failure to prevent actors from being stuck in a bad state.

**`DPSamplerCustom`**: `OUTPUT_IS_LIST` updated to `(True, True)` to match new two-output signature.

---

## 16. CFG and USP Parallel Registries (`cfg.py`, `usp.py`)

**`cfg.py`**: Fixed module path: `diffusion_models.qwen.xdit_cfg_parallel` → `diffusion_models.qwen_image.xdit_cfg_parallel`. Added LTXV/LTXAV combined registration. Removed noisy print on CFG initialization.

**`usp.py`**:
- Removed the `_patch_wan_attention_blocks` helper function — its logic is now inlined into each model's injection function to allow per-model customization.
- `WAN21_Vace` injection changed to `pass` (USP for this model variant disabled pending rework).
- `WAN21_HuMo` injection rewritten: instead of calling `_patch_wan_attention_blocks`, it patches `audio_cross_attn` at the class level (`model.wan_attn_block_class.audio_cross_attn`) rather than per-block.

---

## 17. New ComfyUI Nodes

| Node file | Contents |
|---|---|
| `comfy_extra_dist/nodes_cachedit.py` | CacheDiT-style block-caching nodes |
| `comfy_extra_dist/nodes_idlora.py` | `RayIDLoraKSampler` for ID-LoRA workflows |
| `comfy_extra_dist/nodes_ltx_ffn_chunker.py` | LTX FFN chunking for memory-efficient inference |
| `comfy_extra_dist/idlora_patches.py` | Monkey-patches for ID-LoRA integration |
| `comfy_extra_dist/int8/nodes_int8_loader.py` | Load model in INT8 |
| `comfy_extra_dist/int8/nodes_int8_lora.py` | Apply LoRA over INT8-quantized weights |

---

## 18. ID-LoRA LTXAV Integration

Entirely new feature. No counterpart in the reference.

**Purpose**: Identity-consistent audio-video generation using ID-LoRA on the LTXAV model. Allows injecting a reference speaker's audio latent at inference time to condition the generated audio on speaker identity, while also running separate video and audio guidance scales.

**`IDLoraDenoiseConfig`** (frozen dataclass, `distributed_actor/managers/idlora_manager.py`):
- `video_guidance_scale`, `audio_guidance_scale`, `identity_guidance_scale` — independent CFG scales for video, audio, and identity streams.
- `stg_scale`, `stg_block_idx` — Skip-T Guidance parameters (block-level skipping).
- `av_bimodal_scale` — cross-modal AV bimodal guidance coefficient.

**`IDLoraManager.idlora_denoise()`** — custom Euler denoising loop for LTXAV that calls `model.model.apply_model()` directly with kwargs that match what ComfyUI's `_calc_cond_batch` → `LTXAV.extra_conds` pipeline produces (`denoise_mask`, `audio_denoise_mask`, `frame_rate`, `latent_shapes`, `transformer_options`). Patchification, timestep embedding, and unpatchification all happen inside `apply_model` via ComfyUI's native LTXAV implementation. The loop handles:
- Combined AV latent splitting (video + audio streams processed together).
- Reference audio token injection at negative temporal positions (mapping the reference into negative time offsets so the model treats it as past context).
- Three concurrent CFG batches (video, audio, identity) with independent scale factors.
- STG (Skip-T Guidance): zero-out selected transformer block outputs for guidance.
- Post-loop blending with `v_clean` / `a_clean` for inpainting/extension tasks.
- Returns `((video_out, audio_out), (video_denoised, audio_denoised))` for both the final latent and the last-step denoised x₀ estimate.

**`idlora_patches.py`** — Runtime monkey-patches for `LTXAVModel` to support `ref_audio` injection inside `_process_input`. Replicates changes from ComfyUI PR f02190a. Applied exactly once at first ID-LoRA call via `ensure_patches_applied()`. Skipped automatically if the upstream PR has already been merged (detected by inspecting `_process_input` source for `"ref_audio"`). The patch:
- Extracts reference audio tokens from the input kwargs.
- Computes negative temporal positions (`time_offset` subtracted from the patchifier's audio time grid).
- Prepends reference tokens to the audio sequence with their coords before passing to the transformer.
- Marks reference token positions in the mask and trims them from the output after the forward pass.

**`RayIDLoraKSampler`** ComfyUI node (`comfy_extra_dist/nodes_idlora.py`):
- Inputs: `actors`, noise settings, positive/negative conditioning, `video_guidance_scale`, `audio_guidance_scale`, `identity_guidance_scale`, `sigmas`, `latent_image` (combined AV latent from `LTXVConcatAVLatent`).
- Optional: `reference_audio_latent` (speaker identity from `LTXVAudioVAEEncode`), `stg_scale`, `stg_blocks`, `av_bimodal_scale`.
- Returns `(LATENT, LATENT)` — `output` and `denoised_output`, both as NestedTensors compatible with `LTXVSeparateAVLatent`.
- Calls `_orig_unload_all_models()` before sampling to free main-process VRAM without destroying actor pinned caches.

---

## 19. Removed / Superseded

| Reference file | Status in fork |
|---|---|
| `comfy_dist/fsdp_registry_unused.py` | Replaced by `comfy_dist/fsdp_registry.py` |
| `comfy_dist/kitchen_distributed.py` | Removed; logic moved into per-model FSDP |
| `comfy_dist/kitchen_patches/fp8.py` etc. | Removed; replaced by `comfy_extra_dist/int8/` and `distributed_modules/quantize.py` |
| `diffusion_models/*/fsdp_unused.py` | All replaced by real `fsdp.py` implementations |
| `distributed_modules/attention.py` (monolithic) | Replaced by `distributed_modules/attention/` package with `backends/`, `dispatcher.py`, `layer.py`, `registry.py`, `interface.py` |
| `distributed_worker/*` | Replaced by `distributed_actor/*` |

---

## Breaking changes summary

| Area | Change |
|---|---|
| Worker API | `distributed_worker.ray_worker` → `distributed_actor.actor_pool.ActorPool` |
| Node input names | `"ray_actors"` → `"actors"` on all sampler nodes |
| Node outputs | `XFuserSamplerCustom`, `DPSamplerCustom` now return 2 latents |
| FSDP sharding | Must go through `FSDPShardRegistry`; old `fully_shard_bottom_up` removed |
| DTensor import | `torch.distributed._tensor.DTensor` → `torch.distributed.tensor._api.DTensor` |
| Config | All parallelism config must use `RaylightConfig` / `ExecutionStrategy` dataclasses |
