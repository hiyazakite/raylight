"""Base model context: ABC, helpers, and shared dataclass.

All concrete contexts (GGUF, LazyTensor, FSDP, VAE, BNB) inherit from
``ModelContext`` defined here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import os

import torch

from raylight.utils.memory import MemoryPolicy, NULL_POLICY

if TYPE_CHECKING:
    from raylight.raylight_types import LoraManagerLike, ModelPatcherLike, StateCacheLike, ActorConfigLike


# ---------------------------------------------------------------------------
# VRAM budget helpers
# ---------------------------------------------------------------------------

def _compute_vram_budget(
    device: torch.device,
    model_size: int,
    min_inference_mem: Optional[int] = None,
    vram_limit_bytes: int = 0,
) -> int:
    """Decide how much VRAM to allocate for model weights.

    Uses the same heuristic as ComfyUI's ``load_models_gpu`` so the split
    between CUDA-resident and CPU-offloaded modules stays consistent.

    Parameters
    ----------
    vram_limit_bytes : int
        Optional hard cap on VRAM for weights (from user setting).
        0 = auto (use all available minus inference reserve).

    Returns
    -------
    0
        Model fits entirely in free VRAM → caller should do a full load.
    >0
        Partial budget in bytes → pass as ``lowvram_model_memory`` to
        ``model.load()`` so ComfyUI puts only this much on CUDA and
        streams the rest per-layer via ``comfy_cast_weights`` hooks.
    """
    if not torch.cuda.is_available():
        return 0

    # Mirror ComfyUI: minimum_inference_memory ≈ 0.8 GB + reserve_vram
    if min_inference_mem is None:
        try:
            import comfy.model_management as _mm
            min_inference_mem = int(_mm.minimum_inference_memory())
        except Exception:
            min_inference_mem = int(0.8 * (1024 ** 3))  # 0.8 GB fallback

    free = torch.cuda.mem_get_info(device)[0]

    # User-specified VRAM cap: treat as the effective "free" memory.
    if vram_limit_bytes > 0:
        effective_free = min(free, vram_limit_bytes)
    else:
        effective_free = free

    # Comfortable headroom → full load
    if model_size <= effective_free - min_inference_mem:
        return 0

    # Budget = effective free minus inference reserve (at least 1 byte to
    # avoid the ``lowvram_model_memory=0`` path which ComfyUI treats as
    # "load all as lowvram").
    budget = max(1, effective_free - min_inference_mem)
    return budget


# ---------------------------------------------------------------------------
# Custom ops selection (SafetensorOps vs Int8SafetensorOps)
# ---------------------------------------------------------------------------

def _ops_for_model_options(model_options: Dict[str, Any]):
    """Return the correct custom_operations class for *model_options*.

    When ``model_options`` contains the ``int8_quantize`` flag the INT8-aware
    hybrid ops (``Int8SafetensorOps``) are used.  Otherwise, the default
    zero-copy ``SafetensorOps`` are returned.
    """
    from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps

    if model_options.get("int8_quantize"):
        try:
            from raylight.comfy_extra_dist.int8.int8_distributed_ops import Int8SafetensorOps
            if Int8SafetensorOps is not None:
                # Propagate per-load configuration
                Int8SafetensorOps.dynamic_quantize = model_options.get("int8_dynamic", False)
                Int8SafetensorOps.excluded_names = model_options.get("int8_excluded_names", [])
                Int8SafetensorOps._is_prequantized = False
                return Int8SafetensorOps
        except ImportError:
            pass
    return SafetensorOps


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------

@dataclass
class ModelState:
    """Immutable snapshot of model loading parameters."""
    unet_path: str
    model_options: Dict[str, Any]
    dequant_dtype: Optional[str] = None
    patch_dtype: Optional[str] = None

    @property
    def cache_key(self) -> str:
        """Unique key for caching (file path)."""
        return self.unet_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage in reload_params."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Abstract base context
# ---------------------------------------------------------------------------

class ModelContext(ABC):
    """Abstract base for unified model lifecycle management.

    Handles load → offload → reload cycle with optional mmap caching.
    Each format (GGUF, Safetensors, FSDP) implements format-specific behavior.

    Offload follows the **Template Method** pattern: ``offload()`` is a
    concrete method that runs shared pre/post steps and delegates the
    format-specific part to ``_do_offload()``.
    """

    def __init__(self, use_mmap: bool = True):
        self.use_mmap = use_mmap

    # ─── Abstract Methods (format-specific) ──────────────────

    @abstractmethod
    def load_state_dict_mmap(self, state: ModelState, config: "ActorConfigLike") -> Any:
        """Load state dict with mmap (zero-copy where possible)."""
        ...

    @abstractmethod
    def load_state_dict_standard(self, state: ModelState, config: "ActorConfigLike") -> Any:
        """Load state dict without mmap (fallback)."""
        ...

    @abstractmethod
    def instantiate_model(self, sd: Dict[str, torch.Tensor], state: ModelState,
                          config: "ActorConfigLike", metadata: Any = None, **kwargs) -> Any:
        """Create ModelPatcher from state dict."""
        ...

    @abstractmethod
    def hot_load(self, model: Any, device: torch.device,
                 reload_params: Dict[str, Any]) -> None:
        """Fast VRAM transfer for soft-reloaded models."""
        ...

    @staticmethod
    def _apply_fp8_swap(model: Any, device: torch.device) -> None:
        """Walk model and swap ``nn.Linear`` FP8 weights → ``Fp8AmpereLinear``.

        Idempotent: guarded by ``_fp8_swap_done`` on the model patcher.
        Only runs when ``model_options["fp8_ampere"]`` is truthy, on Ampere
        (SM 8.0–8.8), and when the CUDA extension is available.

        Requires opt-in via ``model_options["fp8_ampere"] = True`` — the same
        flag used by the TP path — so non-TP and TP contexts behave identically.

        TP contexts run ``_apply_tp_fp8_ampere_swap`` during streaming load
        (which sets ``_fp8_swap_done``), so this is a no-op there.
        """
        if getattr(model, "_fp8_swap_done", False):
            return
        # Respect the same opt-in flag as the TP path.
        load_config = getattr(model, "load_config", None)
        model_options = getattr(load_config, "model_options", {}) if load_config is not None else {}
        if not model_options.get("fp8_ampere", False):
            model._fp8_swap_done = True
            return
        try:
            from raylight.distributed_modules.fp8_ampere.fp8_ampere_linear import (
                Fp8AmpereLinear,
                _is_ampere_only,
                _try_load_extension,
            )
        except ImportError:
            return
        if not _is_ampere_only(device) or not _try_load_extension():
            model._fp8_swap_done = True  # skip on future calls too
            return
        diff_model = getattr(model, "model", None)
        if diff_model is None:
            model._fp8_swap_done = True
            return
        swapped = 0
        for parent in diff_model.modules():
            for name, child in list(parent.named_children()):
                if (
                    isinstance(child, torch.nn.Linear)
                    and getattr(child, "weight", None) is not None
                    and child.weight.dtype == torch.float8_e4m3fn
                ):
                    src_device = child.weight.device  # may be CPU
                    new_mod = Fp8AmpereLinear.from_fp8_checkpoint(
                        child.weight.data,
                        child.bias.data if child.bias is not None else None,
                    )
                    # Pack on the source device (CPU or CUDA).
                    # model.load() / reload_to_cuda will move everything to
                    # CUDA; _activate_fp8_buffers will confirm placement.
                    new_mod._prepare_packed(src_device)
                    setattr(parent, name, new_mod)
                    swapped += 1
        model._fp8_swap_done = True
        if swapped:
            print(f"[ModelContext] FP8 Ampere: swapped {swapped} nn.Linear → Fp8AmpereLinear")

    @staticmethod
    def _activate_fp8_buffers(model: Any, device: torch.device) -> None:
        """Call ``on_device_activated`` on all ``Fp8KernelMixin`` instances.

        Called after every ``reload_to_cuda`` / ``model.load`` to ensure
        Marlin-packed weight buffers are on the correct device.  Safe to
        call multiple times — each ``on_device_activated`` is a no-op when
        buffers are already on *device*.
        """
        try:
            from raylight.distributed_modules.fp8_ampere.fp8_ampere_linear import Fp8KernelMixin
        except ImportError:
            return
        diff_model = getattr(model, "model", None)
        if diff_model is None:
            return
        for module in diff_model.modules():
            if isinstance(module, Fp8KernelMixin):
                module.on_device_activated(device)

    def activate(self, model: Any, device: torch.device,
                 memory: "MemoryPolicy" = NULL_POLICY) -> None:
        """Activate a freshly-loaded model on *device*.

        Called once after ``load()`` to move the model to CUDA and set
        internal state flags.  The default implementation calls
        ``model.load(device)`` which runs ComfyUI's patch/hook/callback
        pipeline.  Subclasses override for format-specific behaviour
        (e.g. zero-RAM models are already on CUDA after streaming).
        """
        if model is None:
            return
        if hasattr(model, "load"):
            model.load(device)
        self._apply_fp8_swap(model, device)
        self._activate_fp8_buffers(model, device)
        model.current_device = device

    # ─── Template Method: offload ────────────────────────────

    def offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
        memory: MemoryPolicy = NULL_POLICY,
    ) -> bool:
        """Offload model from VRAM (Template Method).

        Sequence:
        1. Delegate to ``_do_offload()`` (format-specific logic).
        2. Run ``_common_offload_cleanup()`` shared housekeeping.
        3. Let the *memory* policy handle sync / gc / cache-flush.

        Parameters
        ----------
        memory : MemoryPolicy
            Centralised cleanup controller.  Defaults to ``NULL_POLICY``
            (no-op) so callers that don't have a policy still work.

        Returns
        -------
        bool
            True  → model object was **destroyed** (Hard Offload).
            False → model persists on CPU (Soft Offload).
        """
        destroyed = self._do_offload(model, lora_manager, worker_mmap_cache, config)
        self._common_offload_cleanup(model, lora_manager, config, memory=memory)
        memory.after_offload()
        return destroyed

    def _do_offload(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        worker_mmap_cache: Any,
        config: "ActorConfigLike",
    ) -> bool:
        """Format-specific offload hook (override in subclasses).

        The default implementation simply prints a log line.  Concrete
        contexts override this to perform pointer swaps, pinned-cache
        offloads, shard caching, etc.

        Returns ``True`` if the model was destroyed (hard offload).
        """
        print(f"[{self.__class__.__name__}] Standard Offload: Releasing VRAM...")
        return False

    # ─── Shared helpers ──────────────────────────────────────

    @staticmethod
    def _clear_cached_pe(diffusion_model: Any) -> None:
        """Delete cached positional-encoding tensors from inner model."""
        inner = getattr(diffusion_model, "diffusion_model", None)
        for attr in ("_cached_pe", "cached_pe", "pe_cache", "_pe"):
            if hasattr(diffusion_model, attr):
                delattr(diffusion_model, attr)
            if inner is not None and hasattr(inner, attr):
                delattr(inner, attr)

    def _common_offload_cleanup(
        self,
        model: Any,
        lora_manager: Optional["LoraManagerLike"],
        config: "ActorConfigLike",
        *,
        memory: MemoryPolicy = NULL_POLICY,
    ) -> None:
        """Shared post-offload cleanup for all model formats.

        Heavy-weight side-effects (gc, CUDA cache flush, malloc_trim)
        are delegated to *memory* so callers get consistent, debounced
        behaviour without ad-hoc duplication.
        """
        if lora_manager:
            lora_manager.clear_tracking()
            print(f"[RayActor {config.local_rank}] LoRA tracking cleared.")

        if model is not None:
            diffusion_model = getattr(model, "model", None)
            if diffusion_model is not None:
                # Every concrete _do_offload() already moves params to CPU
                # (pinned-cache resize, pointer swap, unpatch_model, etc.)
                # and sets model.current_device = cpu.  A redundant
                # .to("cpu") here would be harmless for mmap/GGUF paths
                # but fatal for pinned-cache paths where storages have
                # been freed via resize_(0).  Skip it entirely.

                self._clear_cached_pe(diffusion_model)

        memory.log_vram(f"RayActor {config.local_rank} Post-Offload")

    def prepare_state_dict(
        self, sd: Dict[str, torch.Tensor], config: "ActorConfigLike"
    ) -> Dict[str, torch.Tensor]:
        """Optional pre-processing of state dict before instantiation."""
        return sd

    # ─── Fast-reload common (Point 2) ────────────────────────

    @staticmethod
    def _fast_reload_common(
        model: Any,
        inner: Any,
        device: torch.device,
        *,
        apply_patches_fn: Any = None,
        lowvram: bool = True,
    ) -> None:
        """Shared logic for bypassing ``model.load()`` during hot-reload.

        Both ``GGUFContext._fast_reload`` and ``LazyTensorContext._fast_reload_full``
        follow the same four-step pattern:

        1. (Re-)apply patches / hooks on modules.
        2. Set model state flags.
        3. Fire ``ON_LOAD`` callbacks.
        4. Apply forced hooks.

        The ``apply_patches_fn`` callback encapsulates format-specific
        differences (GGUF lowvram hooks vs SafeTensor weight patches) while
        this method handles the shared bookkeeping.
        """
        from comfy.patcher_extension import CallbacksMP

        with model.use_ejected():
            model.unpatch_hooks()

            patches = getattr(model, "patches", {})
            wrapper_patches = getattr(model, "weight_wrapper_patches", {})
            force_cast = getattr(model, "force_cast_weights", False)

            # (1) Format-specific patch/hook application
            if apply_patches_fn is not None:
                apply_patches_fn(
                    model=model,
                    inner=inner,
                    device=device,
                    patches=patches,
                    wrapper_patches=wrapper_patches,
                    force_cast=force_cast,
                )

            # (2) Set model state flags
            inner.model_lowvram = lowvram
            inner.device = device
            if lowvram:
                inner.model_loaded_weight_memory = 0
                inner.lowvram_patch_counter = len(patches) if patches else 0
            else:
                inner.model_loaded_weight_memory = sum(
                    p.nbytes for p in inner.parameters()
                )
                inner.lowvram_patch_counter = 0
            inner.model_offload_buffer_memory = 0
            inner.current_weight_patches_uuid = getattr(model, "patches_uuid", None)

            # (3) ON_LOAD callbacks
            for cb in model.get_all_callbacks(CallbacksMP.ON_LOAD):
                cb(model, device, 0, False, not lowvram)

            # (4) Forced hooks
            model.apply_hooks(
                getattr(model, "forced_hooks", None), force_apply=True
            )

        # If full-load path, re-inject after ejection context exits
        if not lowvram:
            model.inject_model()

    # ─── Shared Load Logic ───────────────────────────────────

    def load(self, state: ModelState, config: "ActorConfigLike", state_cache: Any) -> Any:
        """Unified model loading logic.

        1. Load state dict from disk (mmap or standard).
        2. Prepare state dict (e.g., strip prefixes).
        3. Instantiate model.
        4. Post-load setup.
        """
        sd = None
        metadata: Dict[str, Any] = {}

        # 1. Disk load (always — OS page cache handles re-read speed)
        if self.use_mmap:
            print(f"[ModelContext] Mmap Load: {os.path.basename(state.cache_key)}")
            res = self.load_state_dict_mmap(state, config)
            if isinstance(res, tuple) and len(res) == 2:
                sd, metadata = res
            else:
                sd = res
        else:
            print(f"[ModelContext] Standard Load: {os.path.basename(state.cache_key)}")
            res = self.load_state_dict_standard(state, config)
            if isinstance(res, tuple) and len(res) == 2:
                sd, metadata = res
            else:
                sd = res

        if sd is None:
            raise RuntimeError(f"Failed to load state dict for {state.unet_path}")
        if not isinstance(sd, dict):
            raise RuntimeError(f"State dict must be a dict, got {type(sd)} for {state.unet_path}")

        # 2. Prepare
        sd_to_use = self.prepare_state_dict(sd, config)

        # 3. Instantiate
        model = self.instantiate_model(sd_to_use, state, config, metadata=metadata)

        # 4. Post-load setup
        if model is not None:
            model.unet_path = state.unet_path
            model.load_config = state
            if self.use_mmap:
                model.mmap_cache = sd

        return model

    def reload_if_needed(
        self, model: Any, target_device: torch.device,
        config: "ActorConfigLike",
    ) -> Any:
        """Check device and trigger appropriate reload."""
        current = getattr(model, "current_device", None) if model else None

        if model is not None and str(current) == str(target_device):
            print(f"[ModelContext] Already on {target_device}, skipping reload.")
            return model

        if model is not None:
            print(f"[ModelContext] Hot-loading to {target_device}...")
            reload_params = model.load_config.to_dict() if hasattr(model, "load_config") else {}
            self.hot_load(model, target_device, reload_params)
            return model
        else:
            raise RuntimeError("Cannot reload model: Model is None (Lost context).")
