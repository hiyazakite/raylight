!!! Exception during processing !!! ray::RayWorker.common_ksampler() (pid=19321, ip=10.128.5.2, actor_id=c68db7283dbe05a6756a650601000000, repr=<raylight.distributed_worker.ray_worker.RayWorker object at 0x754b3aabd490>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ray/session_2025-08-23_09-19-41_309720_17124/runtime_resources/py_modules_files/_ray_pkg_206d2eeef3d842d3/raylight/distributed_worker/ray_worker.py", line 302, in common_ksampler
    samples = comfy.sample.sample(
              ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sample.py", line 45, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1161, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1051, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1036, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 991, in outer_sample
    self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sampler_helpers.py", line 130, in prepare_sampling
    return executor.execute(model, noise_shape, conds, model_options=model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sampler_helpers.py", line 138, in _prepare_sampling
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required + inference_memory, minimum_memory_required=minimum_memory_required + inference_memory)
  File "/workspace/ComfyUI/comfy/model_management.py", line 663, in load_models_gpu
    loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
  File "/workspace/ComfyUI/comfy/model_management.py", line 474, in model_load
    self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
  File "/workspace/ComfyUI/comfy/model_management.py", line 503, in model_use_more_vram
    return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 864, in partially_load
    self.detach()
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 873, in detach
    self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 767, in unpatch_model
    self.model.to(device_to)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1355, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 942, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1348, in convert
    raise NotImplementedError(
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
Traceback (most recent call last):
  File "/workspace/ComfyUI/execution.py", line 496, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/execution.py", line 315, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, hidden_inputs=hidden_inputs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/execution.py", line 289, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/workspace/ComfyUI/execution.py", line 277, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "/workspace/ComfyUI/custom_nodes/raylight/src/raylight/nodes.py", line 338, in ray_sample
    results = ray.get(futures)
              ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/ray/_private/worker.py", line 2858, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/ray/_private/worker.py", line 958, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(NotImplementedError): ray::RayWorker.common_ksampler() (pid=19321, ip=10.128.5.2, actor_id=c68db7283dbe05a6756a650601000000, repr=<raylight.distributed_worker.ray_worker.RayWorker object at 0x754b3aabd490>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ray/session_2025-08-23_09-19-41_309720_17124/runtime_resources/py_modules_files/_ray_pkg_206d2eeef3d842d3/raylight/distributed_worker/ray_worker.py", line 302, in common_ksampler
    samples = comfy.sample.sample(
              ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sample.py", line 45, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1161, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1051, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 1036, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/samplers.py", line 991, in outer_sample
    self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sampler_helpers.py", line 130, in prepare_sampling
    return executor.execute(model, noise_shape, conds, model_options=model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/patcher_extension.py", line 112, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/sampler_helpers.py", line 138, in _prepare_sampling
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required + inference_memory, minimum_memory_required=minimum_memory_required + inference_memory)
  File "/workspace/ComfyUI/comfy/model_management.py", line 663, in load_models_gpu
    loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
  File "/workspace/ComfyUI/comfy/model_management.py", line 474, in model_load
    self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
  File "/workspace/ComfyUI/comfy/model_management.py", line 503, in model_use_more_vram
    return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 864, in partially_load
    self.detach()
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 873, in detach
    self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
  File "/workspace/ComfyUI/comfy/model_patcher.py", line 767, in unpatch_model
    self.model.to(device_to)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1355, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 942, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1348, in convert
    raise NotImplementedError(
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

Prompt executed in 51.33 seconds
(RayWorker pid=19320) [Rank 0] Applying FSDP to WanModel
(RayWorker pid=19321) diffusion_model is on meta
(RayWorker pid=19320) SHARD COMPLETE
^C
Stopped server
(RayWorker pid=19321) FSDP registered
(RayWorker pid=19321) self.model.model.device=device(type='cpu')
root@4db24e1c3451:/workspace/ComfyUI#
