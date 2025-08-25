from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FSDPModule
from raylight.distributed_worker.model_utils import detect_dtype_mismatch
import torch


def shard_model_fsdp2(model, device_to):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    if not isinstance(diffusion_model, FSDPModule):
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if not name.startswith("blocks."):
                ignored_params.add(param)

        # And also blocks is the most compute heavy part
        diffusion_model.blocks = diffusion_model.blocks.to("cpu")
        ref_dtype = diffusion_model.blocks[0].self_attn.v.weight.dtype
        for i, block in enumerate(diffusion_model.blocks):
            # This is for scaled model
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            diffusion_model.blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.bfloat16),
                reshard_after_forward=True,
                ignored_params=ignored_block_params
            )
        diffusion_model.blocks = diffusion_model.blocks.to(device_to)

        # Model root wrap with ignored params
        fully_shard(diffusion_model, ignored_params=ignored_params)
        model.diffusion_model = diffusion_model

        print("SHARD COMPLETE")
        return model
    else:
        print("FSDP Already applied, skipping")
        return model



from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FSDPModule
from raylight.distributed_worker.model_utils import detect_dtype_mismatch
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
import torch.distributed as dist
import torch


def shard_model_fsdp2(model, device_to, model_state_dict):
    diffusion_model = model.diffusion_model

    # Shard only the blocks, since other modules have different dtype
    # Collect params we want to ignore (everything except blocks)
    if not isinstance(diffusion_model, FSDPModule):
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if not name.startswith("blocks."):
                ignored_params.add(param)

        # And also blocks is the most compute heavy part

        diffusion_model.blocks = diffusion_model.blocks.to("meta")
        ref_dtype = diffusion_model.blocks[0].self_attn.v.weight.dtype
        for i, block in enumerate(diffusion_model.blocks):
            # This is for scaled model
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            diffusion_model.blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.bfloat16),
                reshard_after_forward=True,
                ignored_params=ignored_block_params
            )

        # Model root wrap with ignored params
        fully_shard(diffusion_model, ignored_params=ignored_params)
        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        model.diffusion_model = diffusion_model
        # diffusion_model.blocks = diffusion_model.blocks.to(device_to)

        print("SHARD COMPLETE")
        return model
    else:
        print("FSDP Already applied, skipping")
        return model

