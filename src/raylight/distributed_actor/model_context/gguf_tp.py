"""GGUF + Tensor Parallel model context.

Loads a GGUF-quantised model via mmap, applies TP patching (replacing
``GGMLOps.Linear`` modules with ``TPGGMLLinear`` that store sharded
quantised bytes), then moves quantised shards to CUDA.

The quantised weights remain in their native GGML format on each GPU —
dequantisation happens per-forward, giving the same VRAM savings as
single-GPU GGUF with weight memory split across TP ranks.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from ._base import ModelContext, ModelState
from .gguf import GGUFContext

if TYPE_CHECKING:
    from raylight.raylight_types import ActorConfigLike, LoraManagerLike


class GGUFTPContext(GGUFContext):
    """GGUF model loading with Tensor Parallelism.

    Lifecycle
    ---------
    1. ``load()``  — GGUF mmap → model instantiation → TP patch → CUDA.
    2. ``offload()`` — move parameters to CPU.
    3. ``hot_load()`` (reload) — move parameters back to CUDA.
    """

    def __init__(self, use_mmap: bool = True, zero_ram: bool = False):
        super().__init__(use_mmap=use_mmap)
        self._zero_ram = zero_ram

    # ─── Load ────────────────────────────────────────────────

    def load(self, state: ModelState, config: "ActorConfigLike", state_cache: Any) -> Any:
        """Load GGUF model, apply TP patching, move to CUDA.

        Overrides ``GGUFContext.load()`` to skip mmap-pinning (unnecessary
        since sharded quantised bytes go directly to CUDA) and add TP
        patching after model instantiation.
        """
        from raylight.comfy_dist.tp_registry import TPRegistry
        from raylight.distributed_modules.tensor_parallel import TensorParallelState
        from raylight.distributed_modules.tp_compress import TPCompressConfig
        from raylight.comfy_dist.model_patcher import RaylightModelPatcher

        t0 = time.perf_counter()

        # 1. Load state dict (mmap) + instantiate model via GGUFContext
        #    Call grandparent (ModelContext.load) to avoid GGUFContext's
        #    pinning pass — we don't need pinned host copies for TP.
        model = ModelContext.load(self, state, config, state_cache)
        if model is None:
            raise RuntimeError(f"[GGUFTPContext] Could not load: {state.unet_path}")

        # 2. Apply TP patching  — replaces GGMLOps.Linear with TPGGMLLinear
        #    where the weight is GGML-quantised, and TPLinear otherwise.
        _strategy = config.raylight_config.strategy
        _compress_config = TPCompressConfig(
            mode=_strategy.tp_allreduce_compress,
            bits=_strategy.tp_compress_bits,
            group_size=_strategy.tp_compress_group_size,
            use_residual=_strategy.tp_compress_residual,
            rotation=_strategy.tp_compress_rotation,
            residual_bits=_strategy.tp_compress_residual_bits,
            residual_skip_threshold=_strategy.tp_compress_skip_threshold,
        )

        print("[GGUFTPContext] Applying TP patching to GGUF model...")
        TPRegistry.apply(
            model,
            tp_group=TensorParallelState.get_group(),
            compress_config=_compress_config,
            structure_only=False,  # weights already in mmap — shard in-place
        )

        # 3. Move TP-sharded model to CUDA
        target_device = config.device
        print(f"[GGUFTPContext] Moving quantised shards to {target_device}...")
        self._move_model_to_device(model, target_device)

        # 3b. Pre-size the shared dequant scratch buffer to the largest shard.
        # Must run after weights are on CUDA so the buffer is on the right device.
        from raylight.distributed_modules.tp_linear_factory import warmup_scratch_buffer
        warmup_scratch_buffer(model, device=target_device)

        # 4. Finalise
        model.__class__ = RaylightModelPatcher
        model.current_device = target_device
        model._zero_ram = self._zero_ram

        # Release mmap cache — the quantised bytes have been copied into
        # the per-rank TPGGMLLinear buffers (now on CUDA).
        if hasattr(model, "mmap_cache"):
            model.mmap_cache = None

        dt = (time.perf_counter() - t0) * 1000
        print(f"[GGUFTPContext] Load complete in {dt:.0f} ms")
        return model

    # ─── Device migration ────────────────────────────────────

    @staticmethod
    def _move_model_to_device(model: Any, device: torch.device) -> None:
        """Move all parameters and buffers to *device*."""
        from raylight.distributed_modules.tp_linear_factory import TPGGMLLinear
        from raylight.distributed_modules.tensor_parallel import TPLinear, TPRMSNormAcrossHeads

        inner = getattr(model, "model", None)
        if inner is None:
            return
        diff = getattr(inner, "diffusion_model", None)
        if diff is None:
            return

        moved = 0
        for name, module in diff.named_modules():
            if isinstance(module, (TPGGMLLinear, TPLinear, TPRMSNormAcrossHeads)):
                module.to(device)
                moved += 1
            else:
                # Move this module's OWN parameters and buffers (non-recursive)
                # regardless of whether it has TP children.  This covers e.g.
                # BasicAVTransformerBlock.scale_shift_table which is a direct
                # nn.Parameter on a parent that contains TP'd sub-modules.
                for p in module.parameters(recurse=False):
                    if p.device != device:
                        p.data = p.data.to(device)
                for b_name, buf in module.named_buffers(recurse=False):
                    if buf is not None and buf.device != device:
                        setattr(module, b_name, buf.to(device))
                moved += 1

        print(f"[GGUFTPContext] Moved {moved} modules to {device}")

    # ─── Offload ─────────────────────────────────────────────

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
    ) -> bool:
        """Soft-offload: move quantised shards and parameters to CPU."""
        if model is None:
            return False

        print(f"[GGUFTPContext {config.local_rank}] Offloading TP GGUF model to CPU...")

        if lora_manager:
            lora_manager.clear_gpu_refs(model, config)

        self._move_model_to_device(model, torch.device("cpu"))
        model.current_device = torch.device("cpu")

        print(f"[GGUFTPContext {config.local_rank}] Offload complete.")
        return False

    # ─── Hot load (reload) ───────────────────────────────────

    def hot_load(
        self, model: Any, device: torch.device,
        reload_params: Dict[str, Any],
    ) -> None:
        """Reload: move quantised shards back to CUDA."""
        print(f"[GGUFTPContext] Hot-loading to {device}...")
        self._move_model_to_device(model, device)
        model.current_device = device
        print(f"[GGUFTPContext] Hot-load complete.")
