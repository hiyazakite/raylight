from comfy_api.torch_helpers import set_torch_compile_wrapper
import ray


class RayTorchCompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "backend": (["inductor", "cudagraphs"],),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    def patch(self, ray_actors, backend):

        def _patch(model, backend):
            print(f"Compiler {backend} registered")
            m = model.clone()
            set_torch_compile_wrapper(model=m, backend=backend)
            return m

        gpu_workers = ray_actors["workers"]
        futures = []
        for actor in gpu_workers:
            futures.append(actor.model_function_runner.remote(_patch, backend))

        ray.get(futures)
        return (ray_actors,)


NODE_CLASS_MAPPINGS = {
    "RayTorchCompileModel": RayTorchCompileModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayTorchCompileModel": "Torch Compile Model (Ray)",
}
