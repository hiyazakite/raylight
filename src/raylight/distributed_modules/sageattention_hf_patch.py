import logging
import inspect
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def _supports_argument(fn, name: str) -> bool:
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
        return name in params
    except (TypeError, ValueError):
        code = getattr(fn, "__code__", None)
        return bool(code and name in code.co_varnames)


def ensure_hf_sm90_kernel() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import sageattention
        from sageattention import core as official_core
    except ImportError:
        logger.warning("Unable to patch SageAttention SM90 kernel: package 'sageattention' not found.")
        return False

    hf_kernel = None

    try:
        from kernels import get_kernel

        try:
            hf_module = get_kernel("kernels-community/sage_attention")
            hf_kernel = getattr(hf_module, "sageattn", None)
            if hf_kernel is None:
                logger.warning("HuggingFace kernels SageAttention missing 'sageattn'; skipping patch.")
        except Exception as exc:
            logger.warning("Failed to load kernels-community/sage_attention: %s", exc)
    except ImportError:
        hf_kernel = None

    if hf_kernel is None:
        try:
            from sage_attention import sageattn_qk_int8_pv_fp8_cuda_sm90 as hf_kernel  # type: ignore
        except ImportError:
            logger.warning(
                "HuggingFace SageAttention SM90 kernel not available; continuing with original implementation."
            )
            return False

    def _hf_wrapper(
        q,
        k,
        v,
        *,
        tensor_layout: str = "HND",
        is_causal: bool = False,
        sm_scale=None,
        smooth_k: bool = True,
        return_lse: bool = False,
        **kwargs,
    ):
        call_kwargs = {
            "tensor_layout": tensor_layout,
            "is_causal": is_causal,
            "sm_scale": sm_scale,
            "return_lse": return_lse,
        }

        if _supports_argument(hf_kernel, "smooth_k"):
            call_kwargs["smooth_k"] = smooth_k

        call_kwargs.update(kwargs)

        result = hf_kernel(q, k, v, **call_kwargs)

        if return_lse:
            out, lse = result
            return out, lse

        return result

    official_core.sageattn_qk_int8_pv_fp8_cuda_sm90 = _hf_wrapper
    setattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda_sm90", _hf_wrapper)
    _PATCHED = True
    logger.info("Patched SageAttention SM90 kernel to use HuggingFace implementation.")
    return True
