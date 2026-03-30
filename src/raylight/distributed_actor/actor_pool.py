import ray
from typing import Any, Dict, List, Tuple, cast

from ray.exceptions import RayActorError

from raylight.config import RaylightConfig
from raylight.comfy_dist.utils import cancellable_get
from raylight.ipc.cleanup import cleanup_all_stale as _ipc_cleanup_all_stale


class ActorPool:
    """Manages creation and lifecycle of RayActor actors."""

    def __init__(self, world_size, parallel_dict, raylight_config=None):
        self.world_size = world_size
        self.parallel_dict = parallel_dict
        self.raylight_config = raylight_config

    def create(self) -> Dict[str, Any]:
        """Spawn fresh RayActor actors, killing stale ones first."""
        from raylight.distributed_actor.actor import RayActor

        gpu_actor = ray.remote(RayActor)

        # Kill any stale actors left over from a previous session (crash, Ctrl+C, etc.)
        stale_actors = []
        for local_rank in range(self.world_size):
            try:
                stale_actors.append(ray.get_actor(f"RayActor:{local_rank}"))
            except ValueError:
                pass  # No stale actor with this name — expected on clean start
        for stale in stale_actors:
            ray.kill(stale, no_restart=True)

        actors = []
        for local_rank in range(self.world_size):
            # Propagate system environment to Ray Workers
            init_runtime_env = {}
            if self.raylight_config:
                env_vars = {
                    "RAYLIGHT_ATTN_BACKEND": self.raylight_config.strategy.attention_backend,
                    "RAYLIGHT_ATTN_TYPE": self.raylight_config.strategy.attention_type.name,
                    "RAYLIGHT_RING_IMPL": self.raylight_config.strategy.ring_impl,
                    "RAYLIGHT_ENABLE_KITCHEN": "1" if self.raylight_config.debug.enable_kitchen else "0",
                    "RAYLIGHT_COMPACT_WARMUP_STEPS": str(self.raylight_config.compact.warmup_steps),
                    "RAYLIGHT_DELTA_COMPRESSION": self.raylight_config.compact.delta_compression.name,
                }
                if self.raylight_config.compact.kv_cache_quant_enable:
                    env_vars["RAYLIGHT_COMPACT_QUANTIZED_CACHE"] = "1"
                    env_vars["RAYLIGHT_COMPACT_CACHE_QUANT_BITS"] = str(self.raylight_config.compact.kv_cache_quant_bits)
                init_runtime_env["env_vars"] = env_vars

            actors.append(
                gpu_actor.options(
                    num_gpus=1,
                    name=f"RayActor:{local_rank}",
                    runtime_env=init_runtime_env
                ).remote(
                    local_rank=local_rank,
                    device_id=0,
                    parallel_dict=self.parallel_dict,
                    raylight_config=self.raylight_config
                )
            )

        actors_dict: Dict[str, Any] = {
            "actors": actors,
            "parallel_dict": self.parallel_dict,
        }

        # Parallel initialization check (faster startup)
        ready_refs = [actor.__ray_ready__.remote() for actor in actors]
        cancellable_get(ready_refs)

        return actors_dict

    @staticmethod
    def ensure_fresh(actors_init) -> Tuple[Dict[str, Any], List[Any], RaylightConfig]:
        """Ensure actors are healthy; restart via pool.create() if needed.

        Restarts if:
          1. A model is currently loaded (zero-state safety).
          2. The attention/parallelism configuration has changed.
          3. An actor has crashed or is unresponsive.
        """
        actors_dict, pool, new_config = actors_init
        actors = actors_dict["actors"]

        needs_restart = False
        try:
            # 1. Check if model is loaded
            is_loaded = cancellable_get(actors[0].get_is_model_loaded.remote())
            if is_loaded:
                print("[Raylight] Workers already have a model loaded. Restarting for fresh slate...")
                needs_restart = True

            # 2. Check for configuration change
            if not needs_restart and new_config is not None:
                old_config = cast(RaylightConfig, cancellable_get(actors[0].get_raylight_config.remote()))
                if old_config.strategy != new_config.strategy or old_config.device.world_size != new_config.device.world_size:
                    print(f"[Raylight] Configuration change detected ({old_config.strategy.attention_type.name} -> {new_config.strategy.attention_type.name}). Restarting actors...")
                    needs_restart = True
        except (RayActorError, IndexError, Exception):
            print("[Raylight] Workers in bad state or crashed. Restarting...")
            needs_restart = True

        if needs_restart:
            # Best-effort: release pinned caches before killing actors.
            release_refs = []
            for actor in actors:
                try:
                    release_refs.append(actor.release_pinned_cache.remote())
                except Exception:
                    pass
            if release_refs:
                try:
                    ray.get(release_refs, timeout=10)
                except Exception:
                    pass

            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass

            # Clean up orphaned host-memory artifacts.
            try:
                _ipc_cleanup_all_stale()
            except Exception:
                pass

            actors_dict = cast(Dict[str, Any], pool.create())
            actors = cast(List[Any], actors_dict["actors"])

        # Re-verify config from the (potentially new) actors
        config = cast(RaylightConfig, cancellable_get(actors[0].get_raylight_config.remote()))

        return actors_dict, actors, config
