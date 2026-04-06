# TP + torch.compile Status

## Current State

**torch.compile is disabled for TP-patched blocks.** When TP is active on LTXAV,
`_discover_compile_keys` returns an empty list and compilation is skipped entirely.

Non-TP runs compile all transformer blocks individually as before.

## Why

Two upstream issues prevent stable compilation of TP-patched LTXAV blocks:

### 1. Resume function guard invalidation (fatal)

`BasicAVTransformerBlock.forward` in ComfyUI uses `del` statements between
attention calls (for VRAM management):

```python
attn1_out = self.attn1(norm_vx, ...)
del norm_vx
vx.addcmul_(attn1_out, vgate_msa)
del vgate_msa, attn1_out
```

When Dynamo hits a graph break at `self.attn1(...)` (due to `@torch.compiler.disable`
on the TP attention forward), it creates a resume function that captures local
variables as `___stack0`. The subsequent `del` deallocates them, invalidating the
Dynamo guard **on every denoising step**. This never stabilizes — it recompiles
every step until hitting the 64-recompile limit.

### 2. TPLinear per-instance recompilation (secondary)

Each unique `TPLinear` instance (48 blocks × ~4 per block = ~192 instances) triggers
a new Dynamo guard specialization on `self.parallelism`. Since they share one
`forward` function, the cache fills up to the 64-entry limit.

This is solvable (split into `TPColumnLinear`/`TPRowLinear` subclasses, or use
`torch._dynamo.allow_in_graph`), but problem #1 makes it moot for LTXAV.

## Infrastructure Ready

The funcol (functional collectives) infrastructure is in place for when these issues
are resolved:

- `tensor_parallel.py` imports `torch.distributed._functional_collectives` and uses
  `funcol_all_reduce` / `funcol_all_gather` in `all_reduce_tensor` and
  `gather_tensor_along_dim` (these are compile-safe, out-of-place ops)
- `@torch.compiler.disable` on `TPLinear.forward` and `TPRMSNormAcrossHeads.forward`
  can be removed once compilation is viable
- `_tp_patched` flag on blocks controls compile key filtering

## What Would Unblock

Any one of:

1. **PyTorch** fixes Dynamo resume-function guard handling for deallocated captures
2. **ComfyUI** removes `del` statements from `av_model.py` block forward
3. **Different model architecture** — models without `del` between graph breaks
   (e.g. Flux, SD) may already work with TP + compile (untested)

## Files Involved

| File | Role |
|------|------|
| `src/raylight/distributed_modules/tensor_parallel.py` | funcol imports, `@torch.compiler.disable` on forwards |
| `src/raylight/diffusion_models/lightricks/tp.py` | `block._tp_patched = True` marking |
| `src/raylight/nodes.py` | `_discover_compile_keys` filters `_tp_patched` blocks, `_apply_compile` skips when empty |
