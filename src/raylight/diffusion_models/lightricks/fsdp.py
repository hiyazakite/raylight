import os
import gc

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffload, FSDPModule

from raylight.distributed_modules.utils import align_model_to_cuda


def _iter_shard_targets(diffusion_model):
	targets = []

	if hasattr(diffusion_model, "transformer_blocks"):
		targets.append(("transformer_blocks", diffusion_model.transformer_blocks))

	return targets


def _iter_prebake_targets(diffusion_model):
	targets = []

	legacy_connector = getattr(diffusion_model, "embeddings_connector", None)
	if legacy_connector is not None and hasattr(legacy_connector, "transformer_1d_blocks"):
		targets.append(("embeddings_connector", legacy_connector))

	video_connector = getattr(diffusion_model, "video_embeddings_connector", None)
	if video_connector is not None and hasattr(video_connector, "transformer_1d_blocks"):
		targets.append(("video_embeddings_connector", video_connector))

	audio_connector = getattr(diffusion_model, "audio_embeddings_connector", None)
	if audio_connector is not None and hasattr(audio_connector, "transformer_1d_blocks"):
		targets.append(("audio_embeddings_connector", audio_connector))

	return targets


def _collect_ignored_params(diffusion_model, shard_prefixes):
	ignored_params = set()
	shard_prefixes = tuple(f"{prefix}." for prefix in shard_prefixes)

	for name, param in diffusion_model.named_parameters():
		if not name.startswith(shard_prefixes):
			ignored_params.add(param)

	return ignored_params


def _format_shard_target_summary(shard_targets):
	parts = []
	for prefix, module_list in shard_targets:
		parts.append(f"{prefix}={len(module_list)}")
	return ", ".join(parts)


def _format_prebake_target_summary(prebake_targets):
	parts = []
	for prefix, module in prebake_targets:
		param_count = sum(1 for _ in module.named_parameters())
		parts.append(f"{prefix}={param_count}p")
	return ", ".join(parts)


def _find_unmatched_ltx_patch_keys(patcher, handled_prefixes):
	if patcher is None or not getattr(patcher, "patches", None):
		return []

	handled_prefixes = tuple(f"diffusion_model.{prefix}." for prefix in handled_prefixes)
	relevant_markers = (
		"diffusion_model.transformer_blocks.",
		"diffusion_model.embeddings_connector.",
		"diffusion_model.video_embeddings_connector.",
		"diffusion_model.audio_embeddings_connector.",
	)

	unmatched = []
	for key in patcher.patches:
		if key.startswith(relevant_markers) and not key.startswith(handled_prefixes):
			unmatched.append(key)

	return sorted(unmatched)


def _warn_unmatched_ltx_patch_keys(patcher, handled_prefixes):
	unmatched = _find_unmatched_ltx_patch_keys(patcher, handled_prefixes)
	if not unmatched:
		return

	preview = ", ".join(unmatched[:5])
	if len(unmatched) > 5:
		preview = f"{preview}, ..."

	print(
		f"[Raylight][LightricksFSDP] Found {len(unmatched)} LTX-family LoRA patch keys outside handled shard prefixes: {preview}"
	)


def _bake_block(block, block_prefix, patcher):
	if patcher is None:
		return

	from raylight.distributed_modules.fsdp_utils import bake_lora_block

	bake_lora_block(block, block_prefix, patcher)


def _move_block_to_cuda(block):
	if not isinstance(block, FSDPModule):
		block.to("cuda")


def _prebake_module(module, module_prefix, patcher):
	if patcher is None or not getattr(patcher, "patches", None):
		return

	if isinstance(module, FSDPModule):
		return

	module.to("cuda")
	_bake_block(module, module_prefix, patcher)
	gc.collect()
	torch.cuda.empty_cache()


def _shard_module_list(module_list, block_prefix, enable_cpu_offload, patcher=None):
	if len(module_list) == 0:
		return

	_move_block_to_cuda(module_list[0])
	_bake_block(module_list[0], f"{block_prefix}.0", patcher)

	for i, block in enumerate(module_list):
		_move_block_to_cuda(block)

		if (i + 1) < len(module_list):
			next_block = module_list[i + 1]
			_move_block_to_cuda(next_block)
			_bake_block(next_block, f"{block_prefix}.{i + 1}", patcher)

		if isinstance(block, FSDPModule):
			continue

		try:
			module_list[i] = fully_shard(
				module=block,
				mp_policy=MixedPrecisionPolicy(),
				reshard_after_forward=True,
				offload_policy=CPUOffload(offload_params=enable_cpu_offload),
			)
		except AssertionError as e:
			if "fully_shard has already been applied" in str(e):
				print(f"{block_prefix}[{i}] already sharded, skipping...")
				continue
			raise e

		gc.collect()
		torch.cuda.empty_cache()


def shard_model_fsdp2(model, model_state_dict, enable_cpu_offload, patcher=None):
	diffusion_model = model.diffusion_model
	use_parallel_disk = os.environ.get("RAYLIGHT_FSDP_PARALLEL_LOAD", "1") == "1"

	shard_targets = _iter_shard_targets(diffusion_model)
	prebake_targets = _iter_prebake_targets(diffusion_model)
	if len(shard_targets) == 0:
		raise ValueError(f"No shard targets found for Lightricks model type: {type(diffusion_model).__name__}")

	shard_prefixes = [prefix for prefix, _ in shard_targets]
	prebake_prefixes = [prefix for prefix, _ in prebake_targets]
	handled_prefixes = shard_prefixes + prebake_prefixes
	ignored_params = _collect_ignored_params(diffusion_model, shard_prefixes)
	_warn_unmatched_ltx_patch_keys(patcher, handled_prefixes)

	print(
		"[Raylight][LightricksFSDP] Preparing shard plan: "
		f"{_format_shard_target_summary(shard_targets)} | "
		f"prebake={_format_prebake_target_summary(prebake_targets) or 'none'} | "
		f"cpu_offload={enable_cpu_offload} | parallel_disk={use_parallel_disk}"
	)

	for module_prefix, module in prebake_targets:
		_prebake_module(module, module_prefix, patcher)

	for block_prefix, module_list in shard_targets:
		_shard_module_list(module_list, block_prefix, enable_cpu_offload, patcher=patcher)

	if not isinstance(diffusion_model, FSDPModule):
		try:
			fully_shard(
				diffusion_model,
				ignored_params=ignored_params,
				reshard_after_forward=True,
				offload_policy=CPUOffload(offload_params=enable_cpu_offload),
			)
		except AssertionError as e:
			if "fully_shard has already been applied" in str(e):
				print("Root model already sharded, skipping...")
			else:
				raise e

	model.diffusion_model = diffusion_model

	if dist.is_initialized():
		dist.barrier()

	if not enable_cpu_offload:
		align_model_to_cuda(model)

	broadcast_from_rank0 = not use_parallel_disk

	if model_state_dict is not None:
		if dist.is_initialized() and dist.get_rank() > 0 and broadcast_from_rank0:
			model_state_dict.clear()

		set_model_state_dict(
			model=model,
			model_state_dict=model_state_dict,
			options=StateDictOptions(
				full_state_dict=True,
				broadcast_from_rank0=broadcast_from_rank0,
				cpu_offload=enable_cpu_offload,
			),
		)

	print("[Raylight][LightricksFSDP] Sharding complete and state dict load path configured.")

	return model
