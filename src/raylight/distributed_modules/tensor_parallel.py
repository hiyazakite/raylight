"""
Tensor Parallelism utilities for Raylight.

Implements Megatron-LM style TP with support for:
- TP process group management (TensorParallelState)
- Column-parallel and row-parallel linear layers (TPLinear)
- GQA-aware attention head sharding (TPAttention)
- SwiGLU MLP sharding (TPMLP)
- Distributed RMSNorm for QK-norm (TPRMSNormAcrossHeads)

References:
  - sglang: runtime/layers/linear.py, runtime/distributed/parallel_state.py
  - Megatron-LM: megatron/core/tensor_parallel/layers.py
"""

import math
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed._functional_collectives import (
        all_reduce as funcol_all_reduce,
        all_gather_tensor as funcol_all_gather,
    )
    _HAS_FUNCOL = True
except ImportError:
    _HAS_FUNCOL = False

try:
    from raylight.comfy_extra_dist.int8.int8_quant import (  # type: ignore
        int8_forward_dynamic,
        int8_forward_dynamic_per_row,
        dequantize,
    )
    _HAS_INT8_KER = True
except ImportError:
    _HAS_INT8_KER = False

# =============================================================================
# Type Aliases
# =============================================================================

TPLinearParallelism = Literal["column", "row"]

# =============================================================================
# TP Group Management
# =============================================================================


class TensorParallelState:
    """Manages tensor parallel group state.

    All ranks must call :meth:`initialize` with identical arguments (required by
    ``dist.new_group``).  After initialisation the module-level convenience
    functions (``get_tp_group``, ``get_tp_rank``, ``get_tp_size``) delegate here.
    """

    _tp_group: Optional[dist.ProcessGroup] = None
    _tp_rank: int = 0
    _tp_size: int = 1

    @classmethod
    def initialize(cls, tp_size: int, pg: Optional[dist.ProcessGroup] = None) -> None:
        """Initialize TP groups.

        Creates contiguous-rank TP subgroups.  For ``tp_size=2`` on 8 GPUs the
        groups are ``[0,1], [2,3], [4,5], [6,7]``.

        Args:
            tp_size: Number of ranks per TP group.
            pg: If provided, use this as the TP group directly (e.g. from a
                DeviceMesh ``mesh["tp"].get_group()``).  When *pg* is given the
                ``new_group`` creation loop is skipped.

        NOTE: sglang's GroupCoordinator also creates a parallel CPU (Gloo)
        process group alongside the NCCL one for non-tensor collectives (e.g.
        barrier, broadcasting Python scalars).  Add this if needed later:
            ``cls._tp_cpu_group = dist.new_group(ranks, backend="gloo")``
        """
        cls._tp_size = tp_size

        if pg is not None:
            # DeviceMesh path — group already constructed.
            cls._tp_group = pg
            cls._tp_rank = dist.get_rank(pg)
            return

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        cls._tp_rank = global_rank % tp_size

        if tp_size > 1:
            # All ranks must call new_group with the same rank lists.
            for group_id in range(world_size // tp_size):
                ranks = list(range(group_id * tp_size, (group_id + 1) * tp_size))
                grp = dist.new_group(ranks)
                if global_rank in ranks:
                    cls._tp_group = grp
        else:
            cls._tp_group = None

    @classmethod
    def get_group(cls) -> Optional[dist.ProcessGroup]:
        return cls._tp_group

    @classmethod
    def get_rank(cls) -> int:
        return cls._tp_rank

    @classmethod
    def get_size(cls) -> int:
        return cls._tp_size

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._tp_size > 1

    @classmethod
    def reset(cls) -> None:
        """Reset state (for testing).  Does NOT destroy process groups."""
        cls._tp_group = None
        cls._tp_rank = 0
        cls._tp_size = 1

    @classmethod
    def destroy(cls) -> None:
        """Destroy TP group and reset state."""
        if cls._tp_group is not None:
            dist.destroy_process_group(cls._tp_group)
        cls.reset()


# ---- Module-level convenience functions ------------------------------------


def get_tp_group() -> Optional[dist.ProcessGroup]:
    """Get current TP group."""
    return TensorParallelState.get_group()


def get_tp_rank() -> int:
    """Get current TP rank."""
    return TensorParallelState.get_rank()


def get_tp_size() -> int:
    """Get current TP size."""
    return TensorParallelState.get_size()


# =============================================================================
# Tensor helpers
# =============================================================================


def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int = 1,
    num_splits: Optional[int] = None,
) -> list[torch.Tensor]:
    """Split *tensor* into *num_splits* chunks along *dim*."""
    if num_splits is None:
        num_splits = get_tp_size()
    if num_splits == 1:
        return [tensor]
    return list(tensor.chunk(num_splits, dim=dim))


def gather_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int = -1,
    num_splits: Optional[int] = None,
    full_size: Optional[int] = None,
) -> torch.Tensor:
    """All-gather *tensor* from all TP ranks along *dim*.

    Args:
        tensor: Local shard on this rank.
        dim: Dimension to gather along.
        num_splits: Number of TP ranks (defaults to ``get_tp_size()``).
        full_size: Expected total size along *dim* after gathering
            (pass when the full size is not exactly divisible by tp_size so
            trailing padding is trimmed from the *result*).
    """
    if num_splits is None:
        num_splits = get_tp_size()
    if num_splits == 1:
        return tensor

    group = get_tp_group()
    if group is None:
        return tensor

    # Normalize negative dim — PyTorch ≥2.11 _maybe_view_chunk_cat breaks
    # on negative gather_dim values.
    if dim < 0:
        dim = tensor.ndim + dim

    local_size = tensor.shape[dim]
    expected_full = full_size or (local_size * num_splits)
    max_local = math.ceil(expected_full / num_splits)

    # Pad so every rank supplies equal-sized tensors.
    if local_size < max_local:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = max_local - local_size
        tensor = torch.cat(
            [tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)],
            dim=dim,
        )

    if _HAS_FUNCOL:
        result = funcol_all_gather(tensor.contiguous(), dim, group)
    else:
        tensor_list = [torch.empty_like(tensor) for _ in range(num_splits)]
        dist.all_gather(tensor_list, tensor.contiguous(), group=group)
        result = torch.cat(tensor_list, dim=dim)

    # Trim trailing padding from the concatenated result.
    if full_size is not None and result.shape[dim] != full_size:
        idx = [slice(None)] * result.ndim
        idx[dim] = slice(0, full_size)
        result = result[tuple(idx)]

    return result


def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce *tensor* across the TP group (compile-safe when possible)."""
    if get_tp_size() == 1:
        return tensor
    group = get_tp_group()
    if group is None:
        return tensor
    if _HAS_FUNCOL:
        return funcol_all_reduce(tensor, "sum", group)
    dist.all_reduce(tensor, group=group)
    return tensor


# =============================================================================
# TPLinear
# =============================================================================


class TPLinear(nn.Module):
    """Tensor-parallel linear layer (Megatron-LM style).

    Column-parallel (``parallelism="column"``):
        ``Y = X @ W_local^T + b_local``  — output features are sharded across
        TP ranks.  Set ``gather_output=True`` if the downstream consumer is *not*
        a TP-aware layer and needs the full output.

    Row-parallel (``parallelism="row"``):
        ``Y = X_local @ W_local^T``, then ``all-reduce(Y)`` — output is
        replicated.  ``input_is_parallel=True`` (default) means *x* is already
        the local shard from a preceding column-parallel layer.

    Pass *tp_group* explicitly to compose with DeviceMesh dimensions rather
    than relying on the global ``TensorParallelState`` singleton.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        parallelism: TPLinearParallelism = "column",
        gather_output: bool = False,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        compressor=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallelism = parallelism
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self._tp_group = tp_group
        self.compressor = compressor

        # Resolve tp_size at construction to compute weight shapes.
        _tp_size = (
            dist.get_world_size(tp_group) if tp_group is not None else get_tp_size()
        )

        if parallelism == "column":
            self.local_out_features = out_features // _tp_size
            self.local_in_features = in_features
        else:
            self.local_out_features = out_features
            self.local_in_features = in_features // _tp_size

        is_float = (dtype is None) or dtype.is_floating_point

        # Weight: [local_out, local_in] for column, [out, local_in] for row.
        self.weight = nn.Parameter(
            torch.empty(
                self.local_out_features, self.local_in_features,
                device=device, dtype=dtype,
            ),
            requires_grad=is_float
        )

        if bias:
            bias_dtype = dtype if is_float else torch.bfloat16
            if parallelism == "column":
                self.bias = nn.Parameter(
                    torch.empty(self.local_out_features, device=device, dtype=bias_dtype),
                    requires_grad=True
                )
            else:
                # Row-parallel bias is NOT sharded.  It is added only on rank 0
                # in forward() to prevent it from being summed tp_size times
                # during all-reduce.
                self.bias = nn.Parameter(
                    torch.empty(out_features, device=device, dtype=bias_dtype),
                    requires_grad=True
                )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # -- Properties for runtime group resolution -----------------------------

    @property
    def _resolved_group(self) -> Optional[dist.ProcessGroup]:
        return self._tp_group if self._tp_group is not None else get_tp_group()

    @property
    def _tp_rank(self) -> int:
        grp = self._resolved_group
        return dist.get_rank(grp) if grp is not None else 0

    @property
    def _tp_size_runtime(self) -> int:
        grp = self._resolved_group
        return dist.get_world_size(grp) if grp is not None else 1

    # -- Init ----------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Kaiming uniform init (matches ``nn.Linear``).

        Skipped for non-floating-point dtypes (e.g. INT8 quantized).
        """
        if not self.weight.is_floating_point():
            return
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.local_in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # -- Weight loading ------------------------------------------------------

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """Rank-aware checkpoint loading.

        Slices this rank's shard from the full checkpoint tensor using
        ``narrow()``, matching sglang's weight_loader contract.

        * Column-parallel: slice output dim (dim 0).
        * Row-parallel: slice input dim (dim 1 for weight, dim 0 for bias is
          handled by the bias-only-on-rank-0 rule in ``forward``).

        If the on-disk tensor is already pre-sharded (shape matches *param*),
        a direct copy is performed.
        """
        tp_rank = self._tp_rank
        param_data = param.data

        if self.parallelism == "column":
            if loaded_weight.shape[0] != param_data.shape[0]:
                shard_size = param_data.shape[0]
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)
        else:  # row — shard input dim
            if loaded_weight.ndim >= 2 and loaded_weight.shape[1] != param_data.shape[1]:
                shard_size = param_data.shape[1]
                loaded_weight = loaded_weight.narrow(1, tp_rank * shard_size, shard_size)

        assert param_data.shape == loaded_weight.shape, (
            f"TPLinear weight shape mismatch: param={param_data.shape}, "
            f"loaded_shard={loaded_weight.shape} (tp_rank={tp_rank})"
        )
        param_data.copy_(loaded_weight)

    # -- Forward -------------------------------------------------------------

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        
        # 1. Prepare input and bias for row-parallelism
        if self.parallelism == "row":
            if not self.input_is_parallel:
                # x is replicated; select this rank's input shard.
                x = x.chunk(self._tp_size_runtime, dim=-1)[self._tp_rank].contiguous()
            # Bias only on rank 0 — added before all-reduce so it is included once.
            bias = None if self._tp_rank > 0 else self.bias
        else:
            bias = self.bias

        # 2. Local matrix multiplication
        if not weight.is_floating_point():
            if _HAS_INT8_KER and hasattr(self, "weight_scale"):
                # Native INT8 fast-path (W8A8 dynamic)
                # Ensure weight is on the same device as input (handles
                # zero-RAM / LowVram mode where weights may be on CPU).
                if weight.device != x.device:
                    weight = weight.to(x.device, non_blocking=True)
                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor):
                    w_scale = w_scale.to(x.device, non_blocking=True)
                    
                is_per_row = False
                if isinstance(w_scale, torch.Tensor) and w_scale.dim() == 2 and w_scale.shape[1] == 1:
                    is_per_row = True
                    
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                if bias is not None and bias.device != x.device:
                    bias = bias.to(x.device, non_blocking=True)

                if x_2d.shape[0] > 16:
                    if is_per_row:
                        out = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        out = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                    out = out.reshape(*x_shape[:-1], out.shape[-1])
                else:
                    # Small batch fallback
                    w_float = dequantize(weight, w_scale).to(x.dtype)
                    bias_typed = bias.to(x.dtype) if bias is not None else None
                    out = torch.nn.functional.linear(x, w_float, bias_typed)
            else:
                # Naive W8A16 fallback
                w_float = weight.to(x.dtype)
                if hasattr(self, "weight_scale"):
                    w_scale = getattr(self, "weight_scale")
                    if isinstance(w_scale, torch.Tensor):
                        w_scale_t = w_scale.to(x.device, non_blocking=True)
                        w_float = w_float * w_scale_t.unsqueeze(-1).to(x.dtype)
                    else:
                        w_float = w_float * float(w_scale)  # type: ignore[arg-type]
                out = torch.nn.functional.linear(x, w_float, bias)
        else:
            # Cast weight/bias to match input dtype (e.g. float16 weight
            # from GGUF dequant fallback with bfloat16 activations).
            # Mirrors TPGGMLLinear.forward behaviour.
            # Also move to input device for zero-RAM mode where weights stay on CPU.
            if weight.device != x.device or weight.dtype != x.dtype:
                w = weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
            else:
                w = weight
            b = bias
            if b is not None and (b.device != x.device or b.dtype != x.dtype):
                b = b.to(device=x.device, dtype=x.dtype, non_blocking=True)
            out = torch.nn.functional.linear(x, w, b)

        # 3. Activation-space LoRA via fused Triton kernel
        lora_A = getattr(self, "lora_A", None)
        lora_B = getattr(self, "lora_B", None)
        if lora_A is not None and lora_B is not None:
            scale = getattr(self, "lora_scale", None)
            if scale is None:
                # Backward compat: lora_alpha from INT8's _apply_composition
                scale = getattr(self, "lora_alpha", 1.0)
            if scale is None:
                scale = 1.0
            from raylight.distributed_modules.lora_triton import lora_forward, apply_dora
            lora_delta = lora_forward(x, lora_A, lora_B, float(scale))
            dora_scale = getattr(self, "lora_dora_scale", None)
            if dora_scale is not None:
                out = apply_dora(
                    out, lora_delta, self.weight, dora_scale,
                    lora_a=lora_A, lora_b=lora_B,
                )
            else:
                out = out + lora_delta.to(out.dtype)
        else:
            # 3b. Full-delta path (LoHa / LoKr)
            ld = getattr(self, "lora_delta", None)
            if ld is not None:
                from raylight.distributed_modules.lora_triton import apply_dora_with_delta
                d = ld.to(device=x.device, dtype=x.dtype, non_blocking=True)
                act_delta = torch.nn.functional.linear(x, d)
                dora_scale = getattr(self, "lora_dora_scale", None)
                if dora_scale is not None:
                    out = apply_dora_with_delta(
                        out, act_delta, self.weight, dora_scale, lora_delta=d,
                    )
                else:
                    out = out + act_delta.to(out.dtype)

        # 4. Post-matmul communication
        if self.parallelism == "column":
            if self.gather_output and self._tp_size_runtime > 1:
                out = gather_tensor_along_dim(out, dim=-1, full_size=self.out_features)
        else:  # row
            if self.reduce_results and self._tp_size_runtime > 1:
                if self.compressor is not None:
                    out = self.compressor.compressed_all_reduce(out, group=self._resolved_group)
                else:
                    out = all_reduce_tensor(out)
                
        return out


# =============================================================================
# TPRMSNormAcrossHeads
# =============================================================================


class TPRMSNormAcrossHeads(nn.Module):
    """RMSNorm whose variance is computed across all TP ranks.

    Used for QK-norm inside TP-sharded attention: each rank holds a local shard
    of the full head dimension, so the global variance must be gathered via
    all-reduce before normalising.

    Args:
        full_hidden_size: The *unsharded* feature dimension (e.g. ``inner_dim``).
        local_hidden_size: Features on this rank (``full_hidden_size // tp_size``).
        eps: Layer norm epsilon.
        tp_group: Process group for the all-reduce.  Defaults to the global TP
            group from ``TensorParallelState``.
    """

    def __init__(
        self,
        full_hidden_size: int,
        local_hidden_size: int,
        eps: float = 1e-5,
        tp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.full_hidden_size = full_hidden_size
        self.local_hidden_size = local_hidden_size
        self.eps = eps
        self._tp_group = tp_group
        # Weight is sharded: shape [local_hidden_size].
        self.weight = nn.Parameter(torch.ones(local_hidden_size))

    @property
    def _resolved_group(self) -> Optional[dist.ProcessGroup]:
        return self._tp_group if self._tp_group is not None else get_tp_group()

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., local_hidden_size]
        local_sumsq = x.float().pow(2).sum(dim=-1, keepdim=True)
        if _HAS_FUNCOL:
            local_sumsq = funcol_all_reduce(local_sumsq, "sum", self._resolved_group)
        else:
            dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM, group=self._resolved_group)
        var = local_sumsq / float(self.full_hidden_size)
        w = self.weight if self.weight.device == x.device else self.weight.to(x.device, non_blocking=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * w

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load and shard weights for the norm layer."""
        tp_rank = TensorParallelState.get_rank()
        shard = loaded_weight.narrow(
            0, tp_rank * self.local_hidden_size, self.local_hidden_size
        )
        param.data.copy_(shard)


# =============================================================================
# TPAttention
# =============================================================================


class TPAttention(nn.Module):
    """Tensor-parallel multi-head attention.

    Follows sglang's ``QKVParallelLinear`` pattern for correct GQA handling:

    * When ``tp_size >= total_num_kv_heads`` (extreme TP), KV heads are
      *replicated* across TP ranks (``num_kv_head_replicas > 1``).
    * When ``tp_size < total_num_kv_heads``, each rank holds a distinct KV
      shard: ``num_kv_heads_per_rank = total_num_kv_heads // tp_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        output_bias: bool = True,
        tp_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads

        tp_size = (
            dist.get_world_size(tp_group) if tp_group is not None else get_tp_size()
        )

        # Q heads are always sharded.
        self.num_heads = total_num_heads // tp_size

        # GQA / MQA: when tp_size >= total_num_kv_heads, replicate KV heads.
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // self.total_num_kv_heads
        else:
            self.num_kv_heads = self.total_num_kv_heads // tp_size
            self.num_kv_head_replicas = 1

        q_proj_size = total_num_heads * head_size
        kv_proj_size = self.total_num_kv_heads * head_size
        local_kv_proj_size = self.num_kv_heads * head_size

        self.q_proj = TPLinear(
            hidden_size, q_proj_size,
            bias=qkv_bias, parallelism="column",
            tp_group=tp_group, device=device, dtype=dtype,
        )

        if self.num_kv_head_replicas == 1:
            # Normal case: column-parallel sharding aligns with KV head
            # boundaries (kv_proj_size / tp_size = num_kv_heads * head_size).
            self.k_proj = TPLinear(
                hidden_size, kv_proj_size,
                bias=qkv_bias, parallelism="column",
                tp_group=tp_group, device=device, dtype=dtype,
            )
            self.v_proj = TPLinear(
                hidden_size, kv_proj_size,
                bias=qkv_bias, parallelism="column",
                tp_group=tp_group, device=device, dtype=dtype,
            )
        else:
            # Replicated KV heads (tp_size > total_num_kv_heads): each rank
            # holds num_kv_heads full heads.  Column-parallel would split
            # across a head boundary, so use nn.Linear with a custom
            # weight_loader that maps multiple ranks to the same shard.
            self.k_proj = nn.Linear(
                hidden_size, local_kv_proj_size,
                bias=qkv_bias, device=device, dtype=dtype,
            )
            self.v_proj = nn.Linear(
                hidden_size, local_kv_proj_size,
                bias=qkv_bias, device=device, dtype=dtype,
            )

        # Attach GQA-aware weight_loaders for k/v.
        _num_kv_head_replicas = self.num_kv_head_replicas
        _num_kv_heads = self.num_kv_heads
        _head_size = head_size
        _tp_group = tp_group

        def _kv_weight_loader(
            param: nn.Parameter, loaded_weight: torch.Tensor
        ) -> None:
            tp_rank = (
                dist.get_rank(_tp_group) if _tp_group is not None else get_tp_rank()
            )
            shard_idx = tp_rank // _num_kv_head_replicas
            shard_size = _num_kv_heads * _head_size
            if loaded_weight.shape[0] != param.data.shape[0]:
                loaded_weight = loaded_weight.narrow(
                    0, shard_idx * shard_size, shard_size
                )
            assert param.data.shape == loaded_weight.shape, (
                f"KV weight shape mismatch: param={param.data.shape}, "
                f"loaded_shard={loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)

        self.k_proj.weight_loader = _kv_weight_loader  # type: ignore[assignment]
        self.v_proj.weight_loader = _kv_weight_loader  # type: ignore[assignment]

        # Output projection (row-parallel).
        self.o_proj = TPLinear(
            hidden_size, hidden_size,
            bias=output_bias, parallelism="row",
            tp_group=tp_group, device=device, dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)

        # GQA/MQA: expand KV heads to match Q heads for SDPA.
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_size)
        )
        return self.o_proj(attn_output)


# =============================================================================
# TPMLP
# =============================================================================


class TPMLP(nn.Module):
    """Tensor-parallel SwiGLU MLP.

    Layout::

        w1 (gate):  hidden → intermediate  (column-parallel)
        w2 (up):    hidden → intermediate  (column-parallel)
        w3 (down):  intermediate → hidden  (row-parallel, all-reduce)

    The intermediate dimension is sharded across TP ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        tp_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate/up: column-parallel (output dim sharded).
        self.w1 = TPLinear(
            hidden_size, intermediate_size,
            bias=False, parallelism="column",
            tp_group=tp_group, device=device, dtype=dtype,
        )
        self.w2 = TPLinear(
            hidden_size, intermediate_size,
            bias=False, parallelism="column",
            tp_group=tp_group, device=device, dtype=dtype,
        )
        # Down: row-parallel (input dim sharded, all-reduce on output).
        self.w3 = TPLinear(
            intermediate_size, hidden_size,
            bias=False, parallelism="row",
            tp_group=tp_group, device=device, dtype=dtype,
        )

        _activations = {
            "silu": torch.nn.functional.silu,
            "relu": torch.nn.functional.relu,
            "gelu": torch.nn.functional.gelu,
        }
        self.activation = _activations.get(activation, torch.nn.functional.silu)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU: activation(w1(x)) * w2(x)
        return self.w3(self.activation(self.w1(hidden_states)) * self.w2(hidden_states))
