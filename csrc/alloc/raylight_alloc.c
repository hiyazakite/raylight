/**
 * raylight_alloc.c — Thin CUDA allocator interceptor for pressure-driven eviction.
 *
 * On Linux, this .so overrides cudaMalloc/cudaMallocAsync/cudaFree/cudaFreeAsync
 * AND cuMemCreate (VMM) symbols via LD_PRELOAD.  When an allocation fails,
 * it calls a registered Python callback to free model weights (via
 * offload_by_pressure), then retries.
 *
 * cuMemCreate interception is critical for PyTorch >= 2.7 on Linux where
 * expandable_segments is the default allocator backend.  In this mode PyTorch
 * never calls cudaMalloc — it uses cuMemCreate to allocate physical memory
 * blocks and cuMemMap to map them into a reserved virtual range.
 *
 * This is a minimal port of comfy-aimdo's pyt-cu-plug-alloc-async.c — no VBAR,
 * no virtual memory pages, no priority linked list.  Just:
 *   alloc → fail → callback(size) → retry → succeed or OOM.
 *
 * Build:
 *   gcc -shared -fPIC -o raylight_alloc.so raylight_alloc.c \
 *       -I/usr/local/cuda/include -lcuda -ldl
 *
 * Runtime: loaded via LD_PRELOAD from Python (set before ray.init).
 * Only needs libcuda.so (CUDA driver) at runtime — present on all NVIDIA systems.
 */

#define _GNU_SOURCE  /* for RTLD_NEXT */
#include <dlfcn.h>
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>

/* ─── Types ─────────────────────────────────────────────────────────────── */

typedef int cudaError_t;
typedef struct CUstream_st *cudaStream_t;

/**
 * Pressure callback signature.  Called with the size (bytes) that failed
 * to allocate.  The Python side should free at least that much VRAM
 * (e.g. via offload_by_pressure) and return the number of bytes freed.
 * Return 0 if nothing could be freed.
 */
typedef uint64_t (*pressure_callback_t)(uint64_t needed_bytes);

/* ─── Globals ───────────────────────────────────────────────────────────── */

static pressure_callback_t g_pressure_callback = NULL;
static int g_log_level = 1;  /* 0=silent, 1=warnings, 2=info, 3=debug */

/** Extra headroom when forcing budget pressure (matches aimdo). */
#define ALLOC_HEADROOM (128ULL * 1024 * 1024)

/** Max pressure-retry rounds before giving up. */
#define MAX_RETRY_ROUNDS 3

/* Thread safety for callback registration */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* Re-entrancy guard: prevent callback → PyTorch alloc → callback loop */
static __thread int g_in_callback = 0;

/* ─── Exported API ──────────────────────────────────────────────────────── */

/**
 * Register the pressure callback.  Called once from Python at init time.
 * Pass NULL to disable interception (allocations pass through unchanged).
 */
__attribute__((visibility("default")))
void raylight_alloc_set_callback(pressure_callback_t cb) {
    pthread_mutex_lock(&g_lock);
    g_pressure_callback = cb;
    pthread_mutex_unlock(&g_lock);
}

/**
 * Set log verbosity: 0=silent, 1=warnings only, 2=info, 3=debug.
 */
__attribute__((visibility("default")))
void raylight_alloc_set_log_level(int level) {
    g_log_level = level;
}

/**
 * Query whether a callback is registered and interception is active.
 */
__attribute__((visibility("default")))
bool raylight_alloc_is_active(void) {
    return g_pressure_callback != NULL;
}

/* ─── Helpers ───────────────────────────────────────────────────────────── */

#define LOG_WARN(...)  do { if (g_log_level >= 1) { fprintf(stderr, "[raylight_alloc] " __VA_ARGS__); fflush(stderr); } } while(0)
#define LOG_INFO(...)  do { if (g_log_level >= 2) { fprintf(stderr, "[raylight_alloc] " __VA_ARGS__); fflush(stderr); } } while(0)
#define LOG_DEBUG(...) do { if (g_log_level >= 3) { fprintf(stderr, "[raylight_alloc] " __VA_ARGS__); fflush(stderr); } } while(0)

static inline CUcontext ensure_ctx(void) {
    CUcontext ctx = NULL;
    cuCtxGetCurrent(&ctx);
    return ctx;
}

/**
 * Try to relieve pressure by calling the registered callback.
 * Returns bytes freed, or 0 if no callback or re-entrant call.
 */
static inline uint64_t try_relieve(uint64_t needed) {
    pressure_callback_t cb;
    uint64_t freed;

    if (g_in_callback) {
        return 0;  /* re-entrant — don't recurse */
    }

    pthread_mutex_lock(&g_lock);
    cb = g_pressure_callback;
    pthread_mutex_unlock(&g_lock);

    if (!cb) {
        return 0;
    }

    g_in_callback = 1;
    freed = cb(needed);
    g_in_callback = 0;

    if (freed > 0) {
        LOG_INFO("Pressure callback freed %llu MB for %llu MB request\n",
                 (unsigned long long)(freed / (1024*1024)),
                 (unsigned long long)(needed / (1024*1024)));
    }

    return freed;
}

/* ─── cudaMalloc override ───────────────────────────────────────────────── */

__attribute__((visibility("default")))
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    CUdeviceptr dptr;
    CUresult res;

    if (!devPtr) return 1; /* cudaErrorInvalidValue */

    ensure_ctx();

    /* Attempt 1: normal allocation */
    res = cuMemAlloc_v2(&dptr, size);
    if (res == CUDA_SUCCESS) {
        *devPtr = (void *)dptr;
        return 0;
    }

    /* Attempt 2+: iterative pressure relief with retry loop.
     * Each round asks the Python callback to free more VRAM.
     * Multiple rounds handle:
     *   (a) PyTorch caching allocator needing empty_cache() between
     *       storage.resize_(0) and driver-visible free.
     *   (b) Multiple concurrent allocations during tiled VAE decode
     *       each needing incremental eviction.
     */
    if (res == CUDA_ERROR_OUT_OF_MEMORY) {
        for (int round = 0; round < MAX_RETRY_ROUNDS; round++) {
            LOG_DEBUG("cudaMalloc(%zu MB) failed (round %d), requesting pressure relief\n",
                      size / (1024*1024), round);

            uint64_t freed = try_relieve(size + ALLOC_HEADROOM);
            if (freed == 0 && round > 0) {
                /* Callback couldn't free anything — no point retrying */
                break;
            }

            res = cuMemAlloc_v2(&dptr, size);
            if (res == CUDA_SUCCESS) {
                *devPtr = (void *)dptr;
                return 0;
            }
        }

        LOG_WARN("cudaMalloc(%zu MB) OOM after %d pressure relief rounds\n",
                 size / (1024*1024), MAX_RETRY_ROUNDS);
    }

    *devPtr = NULL;
    return 2; /* cudaErrorMemoryAllocation */
}

/* ─── cudaFree override ─────────────────────────────────────────────────── */

__attribute__((visibility("default")))
cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return 0;
    ensure_ctx();
    CUresult res = cuMemFree_v2((CUdeviceptr)devPtr);
    return (res == CUDA_SUCCESS) ? 0 : (cudaError_t)res;
}

/* ─── cudaMallocAsync override ──────────────────────────────────────────── */

__attribute__((visibility("default")))
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    CUdeviceptr dptr;
    CUresult res;

    if (!devPtr) return 1;

    ensure_ctx();

    /* Attempt 1: normal async allocation */
    res = cuMemAllocAsync(&dptr, size, (CUstream)stream);
    if (res == CUDA_SUCCESS) {
        *devPtr = (void *)dptr;
        return 0;
    }

    /* Attempt 2+: iterative pressure relief with retry loop */
    if (res == CUDA_ERROR_OUT_OF_MEMORY) {
        for (int round = 0; round < MAX_RETRY_ROUNDS; round++) {
            LOG_DEBUG("cudaMallocAsync(%zu MB) failed (round %d), requesting pressure relief\n",
                      size / (1024*1024), round);

            uint64_t freed = try_relieve(size + ALLOC_HEADROOM);
            if (freed == 0 && round > 0) {
                break;
            }

            res = cuMemAllocAsync(&dptr, size, (CUstream)stream);
            if (res == CUDA_SUCCESS) {
                *devPtr = (void *)dptr;
                return 0;
            }
        }

        LOG_WARN("cudaMallocAsync(%zu MB) OOM after %d pressure relief rounds\n",
                 size / (1024*1024), MAX_RETRY_ROUNDS);
    }

    *devPtr = NULL;
    return 2;
}

/* ─── cudaFreeAsync override ────────────────────────────────────────────── */

__attribute__((visibility("default")))
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    if (!devPtr) return 0;
    ensure_ctx();
    CUresult res = cuMemFreeAsync((CUdeviceptr)devPtr, (CUstream)stream);
    return (res == CUDA_SUCCESS) ? 0 : (cudaError_t)res;
}

/* ─── cuMemCreate override (expandable_segments / VMM path) ─────────────
 *
 * PyTorch >= 2.7 on Linux defaults to expandable_segments which uses the
 * CUDA Virtual Memory Management API.  Physical memory is allocated via
 * cuMemCreate, not cudaMalloc.  We intercept this to trigger the same
 * pressure-relief callback on OOM.
 *
 * We use dlsym(RTLD_NEXT) to call the real cuMemCreate from the driver.
 * ────────────────────────────────────────────────────────────────────────── */

typedef CUresult (*real_cuMemCreate_fn)(CUmemGenericAllocationHandle *, size_t,
                                        const CUmemAllocationProp *,
                                        unsigned long long);

static real_cuMemCreate_fn g_real_cuMemCreate = NULL;
static pthread_once_t g_cuMemCreate_once = PTHREAD_ONCE_INIT;

static void resolve_cuMemCreate(void) {
    g_real_cuMemCreate = (real_cuMemCreate_fn)dlsym(RTLD_NEXT, "cuMemCreate");
    if (!g_real_cuMemCreate) {
        LOG_WARN("dlsym(RTLD_NEXT, \"cuMemCreate\") failed: %s\n", dlerror());
    }
}

__attribute__((visibility("default")))
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags) {
    CUresult res;

    pthread_once(&g_cuMemCreate_once, resolve_cuMemCreate);
    if (!g_real_cuMemCreate) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    /* Attempt 1: normal allocation */
    res = g_real_cuMemCreate(handle, size, prop, flags);
    if (res == CUDA_SUCCESS) {
        return CUDA_SUCCESS;
    }

    /* Attempt 2+: iterative pressure relief with retry loop */
    if (res == CUDA_ERROR_OUT_OF_MEMORY) {
        for (int round = 0; round < MAX_RETRY_ROUNDS; round++) {
            LOG_DEBUG("cuMemCreate(%zu MB) failed (round %d), requesting pressure relief\n",
                      size / (1024*1024), round);

            uint64_t freed = try_relieve(size + ALLOC_HEADROOM);
            if (freed == 0 && round > 0) {
                break;
            }

            res = g_real_cuMemCreate(handle, size, prop, flags);
            if (res == CUDA_SUCCESS) {
                return CUDA_SUCCESS;
            }
        }

        LOG_WARN("cuMemCreate(%zu MB) OOM after %d pressure relief rounds\n",
                 size / (1024*1024), MAX_RETRY_ROUNDS);
    }

    return res;
}
