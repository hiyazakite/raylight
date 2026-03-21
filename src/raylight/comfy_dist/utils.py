import torch
import itertools
import ray
import comfy.model_management
from typing import TypeVar, Union, Iterable, List, Any

T = TypeVar("T")


_RAY_CLUSTER_EPOCH = 0


def get_ray_cluster_epoch():
    return _RAY_CLUSTER_EPOCH


# Version-safe decorator factory for disabling torch.compile / torch._dynamo
def _get_torch_compiler_disable_decorator():
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
            return lambda: torch.compiler.disable
    except Exception:
        pass

    # No-op decorator factory for older PyTorch / absent compiler
    def _noop_factory():
        def _noop(fn):
            return fn

        return _noop

    return _noop_factory


_torch_compiler_disable = _get_torch_compiler_disable_decorator()


def clear_ray_cluster(ray_actors=None, reason=None):
    """Best-effort cleanup of Ray workers after a fatal workflow error."""
    global _RAY_CLUSTER_EPOCH
    if reason:
        print(f"[Raylight] Clearing Ray cluster after failure: {reason}")

    workers = []
    if isinstance(ray_actors, dict):
        workers = ray_actors.get("workers", []) or []

    for actor in workers:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            try:
                actor.kill.remote()
            except Exception:
                pass

    try:
        ray.shutdown()
    except Exception:
        pass

    _RAY_CLUSTER_EPOCH += 1


@torch.no_grad()
@_torch_compiler_disable()
def tiled_scale_multidim(
    samples,
    function,
    tile=(64, 64),
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    downscale=False,
    index_formulas=None,
    pbar=None
):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b + 1]

        # handle entire input fitting in a single tile
        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            # Cast explicitly to float32 so the output buffer (float32) is always correct
            # regardless of whether the decode_fn returns bf16 or fp32.
            output[b:b + 1] = function(s).to(dtype=torch.float32, device=output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        # #4: accumulate on GPU in native dtype (bf16 from VAE) — avoids N per-tile D2H copies.
        # out / out_div are lazily allocated on the first tile so we pick up ps.device / ps.dtype
        # automatically without a probe call.
        out = None
        out_div = None
        acc_shape = [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:])

        positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]

        # Cache feathered masks by tile output shape — all interior tiles share one shape
        # so the feathering computation and allocation happen only once per batch element.
        _mask_cache: dict = {}

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            # #5: keep ps on GPU in native dtype — no .float() and no .to(output_device) here.
            # The single cast to float32 + D2H happens once at the end of the batch element.
            ps = function(s_in)

            # Lazy init accumulators on the same device/dtype as ps (GPU, bf16).
            if out is None:
                out = torch.zeros(acc_shape, device=ps.device, dtype=ps.dtype)
                out_div = torch.zeros(acc_shape, device=ps.device, dtype=ps.dtype)

            # Reuse mask for tiles with identical output shape (all interior tiles).
            # Edge tiles with different shapes each get their own cache entry.
            ps_shape_key = tuple(ps.shape)
            if ps_shape_key not in _mask_cache:
                mask = torch.ones_like(ps)
                for d in range(2, dims + 2):
                    feather = round(get_scale(d - 2, overlap[d - 2]))
                    if feather >= mask.shape[d]:
                        continue
                    # Broadcastable 1D ramp: 2 kernel launches per dim instead of
                    # 2*feather scalar mul_ launches (e.g. 6 vs 48 for 3D, feather=8).
                    ramp_shape = [1] * len(mask.shape)
                    ramp_shape[d] = feather
                    ramp = (torch.arange(1, feather + 1, dtype=mask.dtype, device=mask.device) / feather).reshape(ramp_shape)
                    mask.narrow(d, 0, feather).mul_(ramp)
                    mask.narrow(d, mask.shape[d] - feather, feather).mul_(ramp.flip([d]))
                _mask_cache[ps_shape_key] = mask
            else:
                mask = _mask_cache[ps_shape_key]

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            # addcmul_ fuses multiply+add in-place, eliminating the temporary (ps * mask) allocation.
            o.addcmul_(ps, mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        # Single D2H transfer per batch element: divide on GPU, cast to float32, move to output_device.
        output[b:b + 1] = (out / out_div.clamp(min=1e-8)).to(dtype=torch.float32, device=output_device)
    return output


def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    output_device="cpu",
    pbar=None
):
    return tiled_scale_multidim(
        samples,
        function,
        (tile_y, tile_x),
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        pbar=pbar
    )


def cancellable_get(refs: Union[ray.ObjectRef, Iterable[ray.ObjectRef]], timeout: float = 0.1) -> Union[T, List[T]]:
    """
    A Ray.get() replacement that checks for ComfyUI cancellation.
    Returns a list of results if refs is a list, otherwise a single result.
    """
    # Handle single ref
    is_list = isinstance(refs, (list, tuple))
    remaining = list(refs) if is_list else [refs]
    
    # Store results in original order
    results: List[Any] = [None] * len(remaining)
    ref_to_idx = {ref: i for i, ref in enumerate(remaining)}
    
    # print(f"[Raylight] Awaiting {len(remaining)} Ray tasks with cancellation support...")
    
    try:
        while remaining:
            # Check ComfyUI cancel status
            if comfy.model_management.processing_interrupted():
                print("[Raylight] Cancellation detected! Force-canceling Ray tasks...")
                for ref in remaining:
                    try:
                        # Use force=True (SIGKILL) and recursive=True for immediate cleanup
                        ray.cancel(ref, force=True, recursive=True)
                    except Exception as e:
                        print(f"[Raylight] Error canceling task: {e}")
                # Raise exception to stop ComfyUI execution
                raise Exception("Raylight: Job canceled by user.")

            # Wait for some tasks to complete
            ready, remaining = ray.wait(remaining, timeout=timeout)
            for ref in ready:
                idx = ref_to_idx[ref]
                results[idx] = ray.get(ref)
    except Exception as e:
        # Rethrow if it's our cancellation exception, otherwise log and rethrow
        if "Job canceled by user" not in str(e):
            print(f"[Raylight] Error during cancellable_get: {e}")
        raise e
            
    return results if is_list else results[0]
