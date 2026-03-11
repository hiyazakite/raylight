import torch
from torch import Tensor
import torch.distributed
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")

from yunchang.comm.all_to_all import SeqAllToAll4D

import logging

logger = logging.getLogger(__name__)

ATTN_LAYER_IDX = 0 # COMPACT
class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        use_compact_ring: bool = True,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            use_sync=use_sync,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        
        """
        COMPACT ATTN
        """
        # Ring backend is chosen by the attention backend that creates this object.
        # StandardAttentionBackend passes use_compact_ring=False → xfuser ring.
        # CompactFusion backend passes (or uses default) use_compact_ring=True → compact ring.
        # Regardless, if a mask is present at forward time and the xfuser ring is
        # active, we transparently divert to compact_fwd (see forward()).
        if use_compact_ring:
            from raylight.distributed_modules.compact.ring import compact_fwd
            self.ring_attn_fn = compact_fwd
        else:
            try:
                from xfuser.core.long_ctx_attention.ring import xdit_ring_flash_attn_func
                self.ring_attn_fn = xdit_ring_flash_attn_func
            except ImportError:
                from raylight.distributed_modules.compact.ring import compact_fwd
                logger.warning("[Raylight] Standard xfuser ring not found, falling back to compact ring")
                self.ring_attn_fn = compact_fwd
        
        self.idx = None # NOTE: assign idx in forward

    @torch.compiler.disable
    def forward(  # type: ignore[override]
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        mask=None,
        **kwargs,
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """
        is_joint = False
        if (joint_tensor_query is not None and 
            joint_tensor_key is not None and 
            joint_tensor_value is not None):
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
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
        from raylight.distributed_modules.compact.prof import Profiler
        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        with Profiler.instance().scope("ulysses.all2all"):
            if self.use_pack_qkv:
                # (3*bs, seq_len/N, head_cnt, head_size)
                qkv = torch.cat([query, key, value]).contiguous()
                # (3*bs, seq_len, head_cnt/N, head_size)
                qkv = SeqAllToAll4D.apply(
                    self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
                )
                qkv = torch.chunk(qkv, 3, dim=0)  # type: ignore
                query_layer, key_layer, value_layer = qkv

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

        """
        APPLY COMPACT ATTN
        """
        if self.idx is None:
            global ATTN_LAYER_IDX
            self.idx = ATTN_LAYER_IDX
            ATTN_LAYER_IDX += 1
        
        """
        Collector for Q, K, V, Layer/Step Specific
        """
        from raylight.distributed_modules.compact.context import compact_get_step
        # from raylight.distributed_modules.collector.collector import collect
        # collect(query_layer, "q", compact_get_step(), self.idx)
        # collect(key_layer, "k", compact_get_step(), self.idx)
        # collect(value_layer, "v", compact_get_step(), self.idx)
        
        # Build a common kwargs dict for the ring attention function
        from raylight.distributed_modules.compact.ring import compact_fwd, _VERBOSE_ATTN
        from raylight.distributed_modules.attention.xfuser_ring_patch import xdit_ring_flash_attn_func_patched
        
        is_compact_ring = self.ring_attn_fn == compact_fwd
        ring_fn = self.ring_attn_fn

        # If a mask is present and we are using the standard xfuser ring (which doesn't support masks),
        # transparently use our PATCHED xfuser ring for this call.
        if mask is not None and not is_compact_ring:
            if _VERBOSE_ATTN:
                print(f"[Raylight] ⚡ Mask present — using patched xfuser ring for mask support (Layer: {self.idx})")
            ring_fn = xdit_ring_flash_attn_func_patched

        attn_kwargs = {
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
            "return_attn_probs": return_attn_probs,
            "group": self.ring_pg,
            "attn_layer": attn if self.use_kv_cache else None,
            "joint_tensor_key": joint_tensor_key,
            "joint_tensor_value": joint_tensor_value,
            "joint_strategy": joint_strategy,
            # Note: mask is NOT included here — xfuser ring doesn't accept it.
            # When mask is non-None and xfuser ring was active, we already diverted
            # ring_fn to compact_fwd above (which does accept mask).
        }

        # Only pass compact-specific kwargs when using the compact ring backend
        if is_compact_ring:
            attn_kwargs["mod_idx"] = kwargs.get("mod_idx", self.idx)
            attn_kwargs["current_iter"] = kwargs.get("current_iter", compact_get_step())
            attn_kwargs["key_suffix"] = kwargs.get("key_suffix", "")
            attn_kwargs["mask"] = mask
        elif mask is not None:
            # Patched xfuser ring also needs the mask (already diverted to ring_fn)
            attn_kwargs["mask"] = mask

        out = ring_fn(
            query_layer,  # type: ignore
            key_layer,  # type: ignore
            value_layer,  # type: ignore
            **attn_kwargs
        )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        with Profiler.instance().scope("ulysses.all2all"):
            output = SeqAllToAll4D.apply(
                self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
            )

        # out e.g., [s/p::h]
        return output  # type: ignore
