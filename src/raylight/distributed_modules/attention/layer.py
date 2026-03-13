import torch
from torch import Tensor
import torch.distributed
from typing import Any, Optional
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")

from yunchang.comm.all_to_all import SeqAllToAll4D
import logging

from raylight.distributed_modules.attention.dispatcher import select_ring_attn_fn, prepare_ring_attn_kwargs

try:
    from raylight.distributed_modules.attention.backends.fusion.prof import Profiler
    get_profiler_scope = lambda x: Profiler.instance().scope(x)
except ImportError:
    from contextlib import nullcontext
    get_profiler_scope = lambda x: nullcontext()

logger = logging.getLogger(__name__)

ATTN_LAYER_IDX = 0

def reset_attn_layer_idx():
    global ATTN_LAYER_IDX
    ATTN_LAYER_IDX = 0

class RaylightAttention(LongContextAttention):
    """
    Unified attention layer for Raylight that supports both standard xFuser
    and CompactFusion backends via a centralized dispatcher.
    """
    ring_impl_type_supported_kv_cache = ["basic", "zigzag"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        use_compact_ring: bool = False,
    ) -> None:
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            use_sync=use_sync,
            attn_type=attn_type,
        )
        self.use_kv_cache = use_kv_cache
        self.use_compact_ring = use_compact_ring
        self.ring_impl_type = ring_impl_type
        
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )

    @torch.compiler.disable
    def forward(  # type: ignore[override]
        self,
        attn: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query: Optional[Tensor] = None,
        joint_tensor_key: Optional[Tensor] = None,
        joint_tensor_value: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        alibi_slopes: Optional[Tensor] = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
        joint_strategy: str = "none",
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                is_joint = True
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                is_joint = True
        elif (joint_tensor_query is None and 
            joint_tensor_key is None and 
            joint_tensor_value is None):
            pass
        else:
            raise ValueError(
                "joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )

        if is_joint:
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            ) if joint_tensor_key is not None else 0
            
            if joint_tensor_key is not None:
                joint_tensor_key = joint_tensor_key[
                    ...,
                    attn_heads_per_ulysses_rank
                    * ulysses_rank : attn_heads_per_ulysses_rank
                    * (ulysses_rank + 1),
                    :,
                ]
            if joint_tensor_value is not None:
                joint_tensor_value = joint_tensor_value[
                    ...,
                    attn_heads_per_ulysses_rank
                    * ulysses_rank : attn_heads_per_ulysses_rank
                    * (ulysses_rank + 1),
                    :,
                ]
        
        # Prof scope moved to top level for performance

        # Ulysses All2All
        with get_profiler_scope("ulysses.all2all"):
            if self.use_pack_qkv:
                qkv = torch.cat([query, key, value]).contiguous()
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
                assert qkv is not None
                qkv_chunks = torch.chunk(qkv, 3, dim=0)
                query_layer, key_layer, value_layer = qkv_chunks
            else:
                query_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, query, self.scatter_idx, self.gather_idx
                )
                key_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, key, self.scatter_idx, self.gather_idx
                )
                value_layer = SeqAllToAll4D.apply(
                    self.ulysses_pg, value, self.scatter_idx, self.gather_idx
                )

        # Use global counter for layer indexing. 
        # This supports "singleton" usage patterns where a single instance is reused.
        # CRITICAL (Distributed): In SPMD environments (like Context Parallel), all ranks 
        # MUST execute forward calls in the same order to stay synchronized on these indices.
        # Ensure reset_attn_layer_idx() is called at the start of each sample/step.
        global ATTN_LAYER_IDX
        call_idx = ATTN_LAYER_IDX
        ATTN_LAYER_IDX += 1

        # Centralized Dispatch
        ring_fn = select_ring_attn_fn(
            use_compact_ring=self.use_compact_ring,
            has_mask=(mask is not None),
            layer_idx=call_idx,
            ring_impl_type=self.ring_impl_type
        )
        
        # Assertions to satisfy type checker
        assert query_layer is not None
        assert key_layer is not None
        assert value_layer is not None

        attn_kwargs = prepare_ring_attn_kwargs(
            ring_fn=ring_fn,
            layer_idx=call_idx,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
            mask=mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_layer=attn if self.use_kv_cache else None,
            **kwargs
        )

        out = ring_fn(
            query_layer,
            key_layer,
            value_layer,
            **attn_kwargs
        )

        if isinstance(out, tuple):
            context_layer = out[0]
        else:
            context_layer = out

        # Inverse Ulysses All2All
        with get_profiler_scope("ulysses.all2all"):
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )

        assert output is not None
        return output
