import ray
from functools import wraps


# Decorator to make a patch function Ray-distributable.
# Handles wrapping into _patch and Ray actor execution,
def ray_patch(patch_func):
    @wraps(patch_func)
    def wrapper(self, actors, *args, **kwargs):
        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        actor_list = actors["actors"]
        futures = [actor.model_function_runner.remote(_patch, *args, **kwargs)
                   for actor in actor_list]

        ray.get(futures)
        return (actors,)
    return wrapper


# For nodes with return value, like produce float, int, or latent that still require model patcher.
def ray_patch_with_return(patch_func):
    @wraps(patch_func)
    def wrapper(self, actors, *args, **kwargs):
        def _patch(model, *inner_args, **inner_kwargs):
            # call the original patch on each model
            return patch_func(self, model, *inner_args, **inner_kwargs)

        # Just need rank 0
        actor = actors["actors"][0]
        value = ray.get(actor.model_function_runner_get_values.remote(_patch, *args, **kwargs))
        return value
    return wrapper
