from comfy import model_base
import types

def usp_inject_callback(
    model_patcher, device_to, lowvram_model_memory, force_patch_weights, full_load
):
    base_model = model_patcher.model

    if isinstance(base_model, model_base.WAN22_S2V):
        from ..wan.distributed.xdit_context_parallel import (
            usp_audio_dit_forward,
            usp_self_attn_forward,
            usp_t2v_cross_attn_forward,
            usp_audio_injector
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
            block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)

        model.audio_injector.forward = types.MethodType(usp_audio_injector, model.audio_injector)
        for inject in model.audio_injector.injector:
            inject.forward = types.MethodType(usp_t2v_cross_attn_forward, inject)

        model.forward_orig = types.MethodType(usp_audio_dit_forward, model)

    elif isinstance(base_model, model_base.WAN21):
        from ..wan.distributed.xdit_context_parallel import (
            usp_self_attn_forward,
            usp_dit_forward,
            usp_i2v_cross_attn_forward,
            usp_t2v_cross_attn_forward
        )
        from comfy.ldm.wan.model import WanT2VCrossAttention, WanI2VCrossAttention

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(usp_self_attn_forward, block.self_attn)
            if isinstance(block.cross_attn, WanT2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_t2v_cross_attn_forward, block.cross_attn)
            elif isinstance(block.cross_attn, WanI2VCrossAttention):
                block.cross_attn.forward = types.MethodType(usp_i2v_cross_attn_forward, block.cross_attn)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.Flux):
        from ..flux.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_single_stream_forward,
            usp_double_stream_forward
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.HunyuanVideo):
        from ..flux.distributed.xdit_context_parallel import (
            usp_single_stream_forward,
            usp_double_stream_forward
        )
        from ..hunyuan_video.distributed.xdit_context_paralllel import (
            usp_dit_forward
        )

        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.double_blocks:
            block.forward = types.MethodType(usp_double_stream_forward, block)
        for block in model.single_blocks:
            block.forward = types.MethodType(usp_single_stream_forward, block)
        model.forward_orig = types.MethodType(usp_dit_forward, model)

    elif isinstance(base_model, model_base.QwenImage):
        from ..qwen_image.distributed.xdit_context_parallel import (
            usp_dit_forward,
            usp_attn_forward,
        )
        model = base_model.diffusion_model
        print("Initializing USP")
        for block in model.transformer_blocks:
            block.attn.forward = types.MethodType(usp_attn_forward, block.attn)

        model._forward = types.MethodType(usp_dit_forward, model)

    else:
        raise ValueError(
            f"Model: {type(base_model).__name__}, is not yet supported for USP Parallelism"
        )
