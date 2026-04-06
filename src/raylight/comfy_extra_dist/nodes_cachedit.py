"""CacheDiT: Inter-step Residual Caching for Diffusion Transformers.

Two-tier implementation:
  Tier 1 — Lightweight cache (always available, no extra deps):
      Replaces transformer.forward with a warmup + fixed-interval skip wrapper.
      Used directly for Flux / Qwen-Image / Z-Image (NextDiT) and as the
      fallback for every other model.

  Tier 2 — Full cache_dit library (optional, pip install cache-dit>=1.2.0):
      Uses BlockAdapter + DBCacheConfig (adaptive residual-diff threshold) +
      TaylorSeer (Taylor-series output extrapolation).  Attempted first for
      Wan, HunyuanVideo, LTX, Chroma, Kandinsky, Cosmos, HiDream, Lumina, etc.
      Falls back gracefully to Tier 1 if the library is absent or raises.

Two ComfyUI nodes are exported:
  RayCacheDiTAccelerator    — MODEL-in / MODEL-out.  Works on a single GPU
                              (no Ray required).  Auto-detects inference steps
                              from sigmas via the OUTER_SAMPLE hook.
  RayDistributedCacheDiT    — RAY_ACTORS-in / RAY_ACTORS-out.  Pushes the
                              lightweight cache config to every Ray actor.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import torch.distributed as dist

import comfy.model_patcher
import comfy.patcher_extension

logger = logging.getLogger(__name__)


# =============================================================================
# TP-aware sync helpers
# =============================================================================

def _get_tp_group():
    """Return the TP process group if TP is active, else None."""
    try:
        from ..distributed_modules.tensor_parallel import TensorParallelState
        if TensorParallelState.is_initialized():
            return TensorParallelState.get_group()
    except Exception:
        pass
    return None


def _tp_sync_bool(flag: bool, tp_group) -> bool:
    """Synchronise a boolean across TP ranks using all-reduce(MIN).

    All ranks must agree to skip (conservative: if any rank wants to compute,
    everyone computes).  Cost: one scalar all-reduce per step — negligible.
    """
    if tp_group is None:
        return flag
    try:
        device = torch.device("cuda", torch.cuda.current_device())
    except RuntimeError:
        device = torch.device("cpu")
    t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=tp_group)
    return bool(int(t.item()) != 0)


# =============================================================================
# Model Presets
# =============================================================================

@dataclass
class ModelPreset:
    name: str
    description: str
    # lightweight cache defaults
    warmup_steps: int           # how many steps to always compute before caching
    skip_interval: int          # after warmup: compute 1, skip (skip_interval-1)
    noise_scale: float          # noise added to replayed cached outputs (0 = off)
    # full cache_dit library settings (Tier 2)
    forward_pattern: str        # Pattern_0 … Pattern_5
    fn_blocks: int              # Fn_compute_blocks  (DBCacheConfig)
    bn_blocks: int              # Bn_compute_blocks  (DBCacheConfig)
    threshold: float            # residual_diff_threshold (DBCacheConfig)
    max_warmup_steps: int       # DBCacheConfig warmup
    enable_separate_cfg: Optional[bool]
    cfg_compute_first: bool = False
    strategy: str = "adaptive"  # adaptive / static / dynamic
    taylor_order: int = 1       # 0 = disabled


# Tier-1 bypass: these classes always use the lightweight cache
_LIGHTWEIGHT_ONLY_CLASSES = {"NextDiT", "QwenImage", "Flux"}


MODEL_PRESETS: Dict[str, ModelPreset] = {
    # -------------------------------------------------------------------------
    "Flux": ModelPreset(
        name="Flux",
        description="Flux.1 / Flux.2 (balanced caching)",
        warmup_steps=3, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_0",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=4,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Qwen-Image": ModelPreset(
        name="Qwen-Image",
        description="Qwen-Image edit models",
        warmup_steps=3, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_1",
        fn_blocks=1, bn_blocks=0, threshold=0.12, max_warmup_steps=8,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Z-Image": ModelPreset(
        name="Z-Image",
        description="Z-Image (NextDiT, 50 steps, cfg=4.0)",
        warmup_steps=12, skip_interval=3, noise_scale=0.0,
        forward_pattern="Pattern_1",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=25,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    "Z-Image-Turbo": ModelPreset(
        name="Z-Image-Turbo",
        description="Z-Image Turbo (distilled, 4–9 steps)",
        warmup_steps=2, skip_interval=3, noise_scale=0.002,
        forward_pattern="Pattern_1",
        fn_blocks=4, bn_blocks=0, threshold=0.15, max_warmup_steps=3,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="static", taylor_order=0,
    ),
    # -------------------------------------------------------------------------
    "Wan": ModelPreset(
        name="Wan",
        description="Wan T2V / I2V video model",
        warmup_steps=4, skip_interval=2, noise_scale=0.001,
        forward_pattern="Pattern_3",
        fn_blocks=6, bn_blocks=0, threshold=0.10, max_warmup_steps=6,
        enable_separate_cfg=False, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "HunyuanVideo": ModelPreset(
        name="HunyuanVideo",
        description="Hunyuan Video model",
        warmup_steps=4, skip_interval=2, noise_scale=0.001,
        forward_pattern="Pattern_3",
        fn_blocks=6, bn_blocks=0, threshold=0.10, max_warmup_steps=6,
        enable_separate_cfg=False, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "LTX": ModelPreset(
        name="LTX",
        description="LTX-Video (Lightricks)",
        warmup_steps=4, skip_interval=2, noise_scale=0.001,
        forward_pattern="Pattern_1",
        fn_blocks=4, bn_blocks=4, threshold=0.08, max_warmup_steps=6,
        enable_separate_cfg=False, cfg_compute_first=False,
        strategy="dynamic", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Chroma": ModelPreset(
        name="Chroma",
        description="Chroma / Chroma Radiance",
        warmup_steps=3, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_0",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=4,
        enable_separate_cfg=False, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Kandinsky": ModelPreset(
        name="Kandinsky",
        description="Kandinsky I2V",
        warmup_steps=4, skip_interval=2, noise_scale=0.001,
        forward_pattern="Pattern_1",
        fn_blocks=4, bn_blocks=0, threshold=0.10, max_warmup_steps=6,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Cosmos": ModelPreset(
        name="Cosmos",
        description="Cosmos video model",
        warmup_steps=4, skip_interval=2, noise_scale=0.001,
        forward_pattern="Pattern_3",
        fn_blocks=6, bn_blocks=0, threshold=0.10, max_warmup_steps=6,
        enable_separate_cfg=False, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "HiDream": ModelPreset(
        name="HiDream",
        description="HiDream image model",
        warmup_steps=4, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_1",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=6,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Lumina": ModelPreset(
        name="Lumina",
        description="Lumina-T2X",
        warmup_steps=4, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_1",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=6,
        enable_separate_cfg=True, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
    # -------------------------------------------------------------------------
    "Custom": ModelPreset(
        name="Custom",
        description="Custom / unknown model (safe defaults)",
        warmup_steps=4, skip_interval=2, noise_scale=0.0,
        forward_pattern="Pattern_1",
        fn_blocks=8, bn_blocks=0, threshold=0.12, max_warmup_steps=8,
        enable_separate_cfg=None, cfg_compute_first=False,
        strategy="adaptive", taylor_order=1,
    ),
}


def _auto_detect_model_type(transformer: torch.nn.Module) -> str:
    """Heuristically map a transformer class name to a preset key."""
    cls = transformer.__class__.__name__
    mod = transformer.__class__.__module__.lower()

    if "Flux" in cls or "FLUX" in cls:
        return "Flux"
    if "Qwen" in cls:
        return "Qwen-Image"
    if "NextDiT" in cls:
        return "Z-Image"
    if "wan" in mod or "Wan" in cls:
        return "Wan"
    if "hunyuan" in mod or "Hunyuan" in cls:
        return "HunyuanVideo"
    if "ltx" in mod or "LTX" in cls:
        return "LTX"
    if "chroma" in mod or "Chroma" in cls:
        return "Chroma"
    if "kandinsky" in mod or "Kandinsky" in cls:
        return "Kandinsky"
    if "cosmos" in mod or "Cosmos" in cls:
        return "Cosmos"
    if "hidream" in mod or "HiDream" in cls:
        return "HiDream"
    if "lumina" in mod or "Lumina" in cls:
        return "Lumina"
    return "Custom"


# =============================================================================
# CacheDiTConfig — runtime config object stored in transformer_options
# =============================================================================

class CacheDiTConfig:
    """Holds resolved CacheDiT configuration for one sampling run."""

    def __init__(
        self,
        model_type: str,
        preset: ModelPreset,
        *,
        warmup_steps_override: int = 0,
        skip_interval_override: int = 0,
        noise_scale_override: float = -1.0,
        print_summary: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model_type = model_type
        self.forward_pattern = preset.forward_pattern
        self.strategy = preset.strategy
        self.fn_blocks = preset.fn_blocks
        self.bn_blocks = preset.bn_blocks
        self.threshold = preset.threshold
        self.max_warmup_steps = preset.max_warmup_steps
        self.enable_separate_cfg = preset.enable_separate_cfg
        self.cfg_compute_first = preset.cfg_compute_first
        self.taylor_order = preset.taylor_order

        # Lightweight cache params (user can override preset defaults)
        self.warmup_steps: int = warmup_steps_override if warmup_steps_override > 0 else preset.warmup_steps
        self.skip_interval: int = skip_interval_override if skip_interval_override > 0 else preset.skip_interval
        self.noise_scale: float = noise_scale_override if noise_scale_override >= 0.0 else preset.noise_scale

        self.print_summary = print_summary
        self.verbose = verbose

        # Runtime state (reset each run)
        self.num_inference_steps: Optional[int] = None
        self.is_enabled: bool = False
        self.current_step: int = 0
        self.first_step_done: bool = False

    def clone(self) -> "CacheDiTConfig":
        c = copy.copy(self)
        return c

    def reset_runtime(self) -> None:
        self.num_inference_steps = None
        self.is_enabled = False
        self.current_step = 0
        self.first_step_done = False

    def to_worker_dict(self) -> dict:
        """Serialisable dict for shipping over Ray to actors."""
        return {
            "model_type": self.model_type,
            "warmup_steps": self.warmup_steps,
            "skip_interval": self.skip_interval,
            "noise_scale": self.noise_scale,
        }


# =============================================================================
# Lightweight cache — global state (one transformer at a time per process)
# =============================================================================

_lightweight_cache_state: dict = {
    "enabled": False,
    "transformer_id": None,
    "call_count": 0,
    "skip_count": 0,
    "compute_count": 0,
    "last_result": None,
    "compute_times": [],
}


def _cleanup_transformer_cache(transformer: torch.nn.Module) -> None:
    """Restore original forward and clear global cache state for this transformer."""
    global _lightweight_cache_state
    if hasattr(transformer, "_cachedit_original_forward"):
        try:
            transformer.forward = transformer._cachedit_original_forward
            delattr(transformer, "_cachedit_original_forward")
            logger.info("[CacheDiT] Restored original forward for transformer %d", id(transformer))
        except Exception as exc:
            logger.warning("[CacheDiT] Cleanup forward restore failed: %s", exc)

    if _lightweight_cache_state.get("transformer_id") == id(transformer):
        _lightweight_cache_state.update({
            "enabled": False,
            "transformer_id": None,
            "call_count": 0,
            "skip_count": 0,
            "compute_count": 0,
            "last_result": None,
            "compute_times": [],
        })


def _enable_lightweight_cache(
    transformer: torch.nn.Module,
    warmup_steps: int,
    skip_interval: int,
    noise_scale: float,
) -> None:
    """
    Replace transformer.forward with a warmup + skip-interval cache wrapper.

    - The first ``warmup_steps`` calls always compute.
    - After warmup: every ``skip_interval``-th call replays the last result
      (i.e. compute on steps 0, skip_interval, 2*skip_interval, …).
    - If ``noise_scale`` > 0, small Gaussian noise is added to replayed outputs
      to avoid "frozen" regions.
    """
    global _lightweight_cache_state

    current_id = id(transformer)
    _cleanup_transformer_cache(transformer)

    transformer._cachedit_original_forward = transformer.forward    # type: ignore[attr-defined]

    _lightweight_cache_state.update({
        "enabled": True,
        "transformer_id": current_id,
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "compute_times": [],
    })

    tp_group = _get_tp_group()

    def _cached_forward(*args, **kwargs):
        state = _lightweight_cache_state
        state["call_count"] += 1
        n = state["call_count"]

        # Warmup phase: always compute
        if n <= warmup_steps:
            t0 = time.perf_counter()
            result = transformer._cachedit_original_forward(*args, **kwargs)
            state["compute_times"].append(time.perf_counter() - t0)
            state["compute_count"] += 1
            state["last_result"] = _detach_result(result)
            return result

        # Post-warmup: skip every skip_interval-th call
        steps_after = n - warmup_steps
        should_skip = (skip_interval > 1) and ((steps_after % skip_interval) != 0)

        # TP safety: all ranks must agree to skip (conservative)
        should_skip = _tp_sync_bool(should_skip, tp_group)

        if should_skip and state["last_result"] is not None:
            state["skip_count"] += 1
            cached = state["last_result"]
            if noise_scale > 0.0:
                cached = _apply_noise(cached, noise_scale)
            return cached

        t0 = time.perf_counter()
        result = transformer._cachedit_original_forward(*args, **kwargs)
        state["compute_times"].append(time.perf_counter() - t0)
        state["compute_count"] += 1
        state["last_result"] = _detach_result(result)
        return result

    transformer.forward = _cached_forward

    logger.info(
        "[CacheDiT] Lightweight cache active: transformer=%d, warmup=%d, skip_interval=%d, noise=%.4f, tp_sync=%s",
        current_id, warmup_steps, skip_interval, noise_scale,
        tp_group is not None,
    )


def _detach_result(result):
    if isinstance(result, torch.Tensor):
        return result.detach()
    if isinstance(result, tuple):
        return tuple(r.detach() if isinstance(r, torch.Tensor) else r for r in result)
    return result


def _apply_noise(result, noise_scale: float):
    if isinstance(result, torch.Tensor):
        return result + torch.randn_like(result) * noise_scale
    if isinstance(result, tuple):
        return tuple(
            (r + torch.randn_like(r) * noise_scale) if isinstance(r, torch.Tensor) else r
            for r in result
        )
    return result


def _get_lightweight_cache_stats() -> Optional[dict]:
    state = _lightweight_cache_state
    if not state["enabled"] or state["call_count"] == 0:
        return None
    total = state["call_count"]
    cached = state["skip_count"]
    computed = state["compute_count"]
    avg_t = sum(state["compute_times"]) / max(len(state["compute_times"]), 1)
    return {
        "total_steps": total,
        "cached_steps": cached,
        "computed_steps": computed,
        "cache_hit_rate": (cached / total) * 100,
        "estimated_speedup": total / max(computed, 1),
        "avg_compute_time": avg_t,
    }


# =============================================================================
# Block extraction helper (needed for Tier-2 BlockAdapter)
# =============================================================================

def _manual_extract_blocks(
    transformer: torch.nn.Module,
) -> Optional[List[torch.nn.Module]]:
    """Extract the list of transformer blocks for cache_dit's BlockAdapter."""
    # Flux: double + single blocks
    if hasattr(transformer, "double_blocks") or hasattr(transformer, "single_blocks"):
        blocks: List[torch.nn.Module] = []
        for attr in ("double_blocks", "single_blocks"):
            m = getattr(transformer, attr, None)
            if isinstance(m, (list, torch.nn.ModuleList)):
                blocks.extend(list(m))
        if blocks:
            return blocks

    # NextDiT (Z-Image): .layers
    if hasattr(transformer, "layers"):
        layers = transformer.layers
        if isinstance(layers, (list, torch.nn.ModuleList)):
            return list(layers)
        if isinstance(layers, torch.nn.Sequential):
            return list(layers.children())

    # Standard DiT / LTX / Hunyuan: .blocks / .transformer_blocks / .dit_blocks
    for attr in ("blocks", "transformer_blocks", "dit_blocks"):
        m = getattr(transformer, attr, None)
        if isinstance(m, (list, torch.nn.ModuleList)):
            return list(m)
        if isinstance(m, torch.nn.Sequential):
            return list(m.children())

    # Deep search: first ModuleList/Sequential with ≥ 4 children
    for name, mod in transformer.named_children():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            children = list(mod)
            if len(children) >= 4:
                logger.info("[CacheDiT] Deep search found %d blocks in .%s", len(children), name)
                return children

    logger.warning("[CacheDiT] Block extraction failed for %s", transformer.__class__.__name__)
    return None


# =============================================================================
# Tier-2: full cache_dit library path
# =============================================================================

def _build_cache_config(config: CacheDiTConfig):
    """Build a DBCacheConfig from the resolved CacheDiTConfig."""
    from cache_dit import DBCacheConfig  # type: ignore[import]

    cfg = DBCacheConfig(
        Fn_compute_blocks=config.fn_blocks,
        Bn_compute_blocks=config.bn_blocks,
        residual_diff_threshold=config.threshold,
        max_warmup_steps=config.max_warmup_steps,
        num_inference_steps=config.num_inference_steps,
    )
    if config.enable_separate_cfg is not None:
        cfg.enable_separate_cfg = config.enable_separate_cfg
    cfg.cfg_compute_first = config.cfg_compute_first

    strategy = config.strategy
    n = config.num_inference_steps or 0
    if strategy == "static":
        cfg.max_cached_steps = int(n * 0.5) if n else -1
        cfg.max_continuous_cached_steps = -1
    elif strategy == "dynamic":
        cfg.max_cached_steps = int(n * 0.7) if n else -1
        cfg.max_continuous_cached_steps = 4
    else:  # adaptive
        cfg.max_cached_steps = -1
        cfg.max_continuous_cached_steps = -1

    return cfg


def _build_calibrator_config(taylor_order: int):
    if taylor_order <= 0:
        return None
    try:
        from cache_dit import TaylorSeerCalibratorConfig  # type: ignore[import]
        return TaylorSeerCalibratorConfig(
            enable_calibrator=True,
            enable_encoder_calibrator=True,
            taylorseer_order=taylor_order,
        )
    except Exception:
        return None


def _get_forward_pattern(pattern_name: str):
    import cache_dit  # type: ignore[import]
    mapping = {
        "Pattern_0": cache_dit.ForwardPattern.Pattern_0,
        "Pattern_1": cache_dit.ForwardPattern.Pattern_1,
        "Pattern_2": cache_dit.ForwardPattern.Pattern_2,
        "Pattern_3": cache_dit.ForwardPattern.Pattern_3,
        "Pattern_4": cache_dit.ForwardPattern.Pattern_4,
        "Pattern_5": cache_dit.ForwardPattern.Pattern_5,
    }
    return mapping.get(pattern_name, cache_dit.ForwardPattern.Pattern_1)


def _enable_cache_dit_full(transformer: torch.nn.Module, config: CacheDiTConfig) -> None:
    """
    Tier-2: attempt to enable cache_dit BlockAdapter + DBCache + TaylorSeer.
    Raises on failure so the caller can fall back to Tier 1.
    """
    import cache_dit  # type: ignore[import]
    from cache_dit import BlockAdapter  # type: ignore[import]

    blocks = _manual_extract_blocks(transformer)
    if not blocks:
        raise RuntimeError("Block extraction failed — cannot use cache_dit BlockAdapter")

    blocks_ml = blocks if isinstance(blocks, torch.nn.ModuleList) else torch.nn.ModuleList(blocks)

    if not hasattr(transformer, "blocks"):
        transformer.blocks = blocks_ml

    adapter = BlockAdapter(blocks=blocks_ml)
    if not getattr(adapter, "blocks", None):
        raise RuntimeError("BlockAdapter has no blocks")

    cache_config = _build_cache_config(config)
    calibrator_config = _build_calibrator_config(config.taylor_order)
    pattern = _get_forward_pattern(config.forward_pattern)

    enable_kwargs: dict = {"cache_config": cache_config, "forward_pattern": pattern}
    if calibrator_config is not None:
        enable_kwargs["calibrator_config"] = calibrator_config

    cache_dit.enable_cache(adapter, **enable_kwargs)
    logger.info(
        "[CacheDiT] Tier-2 BlockAdapter enabled: F%dB%d, threshold=%.3f, warmup=%d",
        config.fn_blocks, config.bn_blocks, config.threshold, config.max_warmup_steps,
    )


def _enable_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig) -> None:
    """
    Route to Tier-1 (lightweight) or attempt Tier-2 then fall back.
    """
    cls_name = transformer.__class__.__name__

    # Tier-1 bypass for known-incompatible classes
    if cls_name in _LIGHTWEIGHT_ONLY_CLASSES:
        logger.info("[CacheDiT] %s → Tier-1 (lightweight bypass)", cls_name)
        _enable_lightweight_cache(
            transformer,
            warmup_steps=config.warmup_steps,
            skip_interval=config.skip_interval,
            noise_scale=config.noise_scale,
        )
        return

    # Attempt Tier-2
    try:
        _enable_cache_dit_full(transformer, config)
        return
    except ImportError:
        logger.info("[CacheDiT] cache_dit library not installed → falling back to Tier-1")
    except Exception as exc:
        logger.warning("[CacheDiT] Tier-2 failed (%s) → falling back to Tier-1", exc)

    # Tier-1 fallback
    _enable_lightweight_cache(
        transformer,
        warmup_steps=config.warmup_steps,
        skip_interval=config.skip_interval,
        noise_scale=config.noise_scale,
    )


def _refresh_cache_dit(transformer: torch.nn.Module, config: CacheDiTConfig) -> None:
    """
    Reset per-run counters for an already-patched transformer.
    For lightweight cache this is a simple state reset.
    For cache_dit library it calls refresh_context.
    """
    global _lightweight_cache_state
    if _lightweight_cache_state.get("enabled") and \
            _lightweight_cache_state.get("transformer_id") == id(transformer):
        _lightweight_cache_state["call_count"] = 0
        _lightweight_cache_state["skip_count"] = 0
        _lightweight_cache_state["compute_count"] = 0
        _lightweight_cache_state["last_result"] = None
        _lightweight_cache_state["compute_times"] = []
        logger.info("[CacheDiT] Lightweight cache reset for new run (%d steps)", config.num_inference_steps or 0)
        return

    try:
        import cache_dit  # type: ignore[import]
        cache_config = _build_cache_config(config)
        calibrator_config = _build_calibrator_config(config.taylor_order)
        kwargs: dict = {"cache_config": cache_config, "verbose": config.verbose}
        if calibrator_config is not None:
            kwargs["calibrator_config"] = calibrator_config
        cache_dit.refresh_context(transformer, **kwargs)
        logger.info("[CacheDiT] cache_dit context refreshed (%d steps)", config.num_inference_steps or 0)
    except Exception as exc:
        logger.warning("[CacheDiT] refresh_context failed: %s", exc)


def _print_summary(transformer: torch.nn.Module, config: CacheDiTConfig) -> None:
    """Print a performance dashboard after sampling completes."""
    lightweight_stats = _get_lightweight_cache_stats()
    if lightweight_stats is not None:
        total = lightweight_stats["total_steps"]
        cached = lightweight_stats["cached_steps"]
        computed = lightweight_stats["computed_steps"]
        rate = lightweight_stats["cache_hit_rate"]
        speedup = lightweight_stats["estimated_speedup"]
        avg_t = lightweight_stats["avg_compute_time"]
        width = 54
        sep = "─" * width
        tp_active = _get_tp_group() is not None
        lines = [
            "╔" + "═" * width + "╗",
            "║" + " CacheDiT Performance Dashboard ".center(width) + "║",
            "╠" + sep + "╣",
            "║" + f"  Model : {config.model_type}".ljust(width) + "║",
            "║" + f"  Mode  : Lightweight (Tier-1, TP-sync={'on' if tp_active else 'off'})".ljust(width) + "║",
            "╠" + sep + "╣",
            "║" + f"  Total steps    : {total}".ljust(width) + "║",
            "║" + f"  Computed steps : {computed}".ljust(width) + "║",
            "║" + f"  Cached steps   : {cached}".ljust(width) + "║",
            "║" + f"  Cache hit rate : {rate:.1f}%".ljust(width) + "║",
            "║" + f"  Est. speedup   : {speedup:.2f}x".ljust(width) + "║",
            "║" + f"  Avg compute/step: {avg_t*1000:.1f} ms".ljust(width) + "║",
            "╚" + "═" * width + "╝",
        ]
        print("\n" + "\n".join(lines))
        return

    # Tier-2 stats via cache_dit.summary
    try:
        import cache_dit  # type: ignore[import]
        stats = cache_dit.summary(transformer)
        if stats is None:
            return
        total = getattr(stats, "total_steps", 0)
        cached = getattr(stats, "cached_steps", 0)
        computed = getattr(stats, "computed_steps", total - cached)
        speedup = getattr(stats, "speedup", total / max(computed, 1))
        avg_diff = getattr(stats, "avg_diff", 0.0)
        width = 54
        sep = "─" * width
        lines = [
            "╔" + "═" * width + "╗",
            "║" + " CacheDiT Performance Dashboard ".center(width) + "║",
            "╠" + sep + "╣",
            "║" + f"  Model   : {config.model_type}".ljust(width) + "║",
            "║" + f"  Mode    : BlockAdapter + DBCache (Tier-2)".ljust(width) + "║",
            "║" + f"  Pattern : {config.forward_pattern}".ljust(width) + "║",
            "╠" + sep + "╣",
            "║" + f"  Total steps     : {total}".ljust(width) + "║",
            "║" + f"  Computed steps  : {computed}".ljust(width) + "║",
            "║" + f"  Cached steps    : {cached}".ljust(width) + "║",
            "║" + f"  Est. speedup    : {speedup:.2f}x".ljust(width) + "║",
            "║" + f"  Avg residual Δ  : {avg_diff:.6f}".ljust(width) + "║",
            "║" + f"  Threshold       : {config.threshold:.4f}".ljust(width) + "║",
            "╚" + "═" * width + "╝",
        ]
        print("\n" + "\n".join(lines))
    except Exception as exc:
        logger.debug("[CacheDiT] Summary unavailable: %s", exc)


# =============================================================================
# ComfyUI patcher hooks (used by RayCacheDiTAccelerator / MODEL node)
# =============================================================================

def _cache_dit_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE hook: fires once per KSampler call.

    - Clones model_options for isolation.
    - Auto-detects num_inference_steps from sigmas.
    - Enables cache on first call; refreshes counters on subsequent calls.
    - Prints dashboard after sampling completes.
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config: Optional[CacheDiTConfig] = None

    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)

        config = guider.model_options.get("transformer_options", {}).get("cachedit_config")
        if config is None:
            return executor(*args, **kwargs)

        config = config.clone()
        config.reset_runtime()
        guider.model_options["transformer_options"]["cachedit_config"] = config

        # Auto-detect step count from sigmas (4th positional arg in ComfyUI)
        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            config.num_inference_steps = max(len(sigmas) - 1, 1)
            if config.verbose:
                logger.info("[CacheDiT] Auto-detected %d inference steps", config.num_inference_steps)

        # Get transformer reference
        mp = guider.model_patcher
        if hasattr(mp, "model") and hasattr(mp.model, "diffusion_model"):
            transformer = mp.model.diffusion_model

        if transformer is not None and config.num_inference_steps is not None:
            if not config.is_enabled or id(transformer) != _lightweight_cache_state.get("transformer_id"):
                _enable_cache_dit(transformer, config)
                config.is_enabled = True
            else:
                _refresh_cache_dit(transformer, config)

        result = executor(*args, **kwargs)

        if config.print_summary and transformer is not None:
            _print_summary(transformer, config)

        return result

    except Exception as exc:
        logger.error("[CacheDiT] outer_sample_wrapper error: %s", exc)
        import traceback; traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        guider.model_options = orig_model_options


def _cache_dit_diffusion_model_wrapper(executor, *args, **kwargs):
    """
    DIFFUSION_MODEL hook: fires once per denoising step.

    Increments current_step and applies noise injection on cached outputs.
    """
    try:
        transformer_options = (
            args[-1] if isinstance(args[-1], dict) else kwargs.get("transformer_options", {})
        )
        config: Optional[CacheDiTConfig] = transformer_options.get("cachedit_config")

        if config is not None:
            if not config.first_step_done:
                config.first_step_done = True
                config.current_step = 0
            else:
                config.current_step += 1

        output = executor(*args, **kwargs)

        if config is not None and config.noise_scale > 0.0:
            if config.current_step >= config.warmup_steps:
                output = _apply_noise(output, config.noise_scale)

        return output

    except Exception as exc:
        logger.error("[CacheDiT] diffusion_model_wrapper error: %s", exc)
        return executor(*args, **kwargs)


# =============================================================================
# Node: RayCacheDiTAccelerator  (MODEL-in / MODEL-out, no Ray needed)
# =============================================================================

class RayCacheDiTAccelerator:
    """
    Accelerate a diffusion model via CacheDiT inter-step caching.

    Works on a single GPU without Ray — just pipe a MODEL through this node
    before your KSampler.  The number of inference steps is auto-detected from
    the sigmas at runtime so no manual step-count wiring is required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = ["Auto"] + list(MODEL_PRESETS.keys())
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "model_type": (preset_names, {"default": "Auto"}),
                "warmup_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "Steps to always compute before caching starts (0 = use preset default)",
                }),
                "skip_interval": ("INT", {
                    "default": 0, "min": 0, "max": 20,
                    "tooltip": "Cache every N-th post-warmup step (0 = use preset default, 2 = skip every other)",
                }),
                "noise_scale": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 0.1, "step": 0.001,
                    "tooltip": "Noise added to replayed outputs to prevent static regions (-1 = use preset default)",
                }),
                "print_summary": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Raylight/extra"
    DESCRIPTION = (
        "CacheDiT: accelerate any DiT model by caching transformer outputs "
        "across denoising steps.  Auto-detects inference steps; no step count "
        "input required."
    )

    def apply(self, model, enable, model_type, warmup_steps, skip_interval, noise_scale, print_summary):
        if not enable:
            return self._disable(model)

        model = model.clone()

        # Resolve model type
        transformer = getattr(getattr(model, "model", None), "diffusion_model", None)
        if model_type == "Auto":
            model_type = _auto_detect_model_type(transformer) if transformer is not None else "Custom"
            logger.info("[CacheDiT] Auto-detected model type: %s", model_type)

        preset = MODEL_PRESETS.get(model_type, MODEL_PRESETS["Custom"])
        config = CacheDiTConfig(
            model_type=model_type,
            preset=preset,
            warmup_steps_override=warmup_steps,
            skip_interval_override=skip_interval,
            noise_scale_override=noise_scale,
            print_summary=print_summary,
        )

        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["cachedit_config"] = config

        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "cachedit",
            _cache_dit_outer_sample_wrapper,
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "cachedit",
            _cache_dit_diffusion_model_wrapper,
        )

        logger.info(
            "[CacheDiT] Accelerator attached: model_type=%s, warmup=%d, skip=%d, noise=%.4f",
            model_type,
            config.warmup_steps,
            config.skip_interval,
            config.noise_scale,
        )
        return (model,)

    def _disable(self, model):
        model = model.clone()
        to = model.model_options.get("transformer_options", {})
        to.pop("cachedit_config", None)
        for wt in (
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        ):
            model.wrappers.get(wt, {}).pop("cachedit", None)
        transformer = getattr(getattr(model, "model", None), "diffusion_model", None)
        if transformer is not None:
            _cleanup_transformer_cache(transformer)
            try:
                import cache_dit; cache_dit.disable_cache(transformer)  # type: ignore[import]
            except Exception:
                pass
        return (model,)


# =============================================================================
# Node: RayDistributedCacheDiT  (RAY_ACTORS-in / RAY_ACTORS-out)
# =============================================================================

class RayDistributedCacheDiT:
    """
    Push CacheDiT lightweight cache configuration to all Ray actor processes.

    Each actor patches its local transformer.forward directly — no ComfyUI
    model patcher hooks are needed on the actor side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = ["Auto"] + list(MODEL_PRESETS.keys())
        return {
            "required": {
                "actors": ("RAY_ACTORS",),
                "enable": ("BOOLEAN", {"default": True}),
                "model_type": (preset_names, {"default": "Auto"}),
                "warmup_steps": ("INT", {
                    "default": 0, "min": 0, "max": 100,
                    "tooltip": "0 = use preset default",
                }),
                "skip_interval": ("INT", {
                    "default": 0, "min": 0, "max": 20,
                    "tooltip": "0 = use preset default",
                }),
                "noise_scale": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 0.1, "step": 0.001,
                    "tooltip": "-1 = use preset default",
                }),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    FUNCTION = "apply"
    CATEGORY = "Raylight/extra"

    def apply(self, actors, enable, model_type, warmup_steps, skip_interval, noise_scale):
        import ray  # type: ignore[import]

        actor_list = actors["actors"]

        if not enable:
            ray.get([actor.apply_cachedit.remote({"enabled": False}) for actor in actor_list])
            return (actors,)

        # We don't have the transformer here to auto-detect, so actors will
        # do their own detection when model_type == "Auto".
        config_dict = {
            "enabled": True,
            "model_type": model_type,
            "warmup_steps": warmup_steps,
            "skip_interval": skip_interval,
            "noise_scale": noise_scale,
        }

        ray.get([actor.apply_cachedit.remote(config_dict) for actor in actor_list])
        return (actors,)


# =============================================================================
# Node registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "RayCacheDiTAccelerator": RayCacheDiTAccelerator,
    "RayDistributedCacheDiT": RayDistributedCacheDiT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayCacheDiTAccelerator": "CacheDiT Accelerator",
    "RayDistributedCacheDiT": "Ray CacheDiT (Distributed)",
}
