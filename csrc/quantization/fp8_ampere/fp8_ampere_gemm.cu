// FP8-weight × BF16-activation GEMM for Ampere GPUs (SM 8.x).
// v5 — Optimized kernel:
//   (A) Inline PTX mma.sync.aligned.m16n8k16 for BF16 tensor cores
//   (B) 4-stage cp.async pipeline overlapping memory/compute
//   (C) FP8 E4M3 → BF16 register dequant (vLLM bitwise trick)
//   (D) Direct register → global epilogue (no smem_out staging)
//   (E) Per-channel BF16 scale fused into epilogue
//   (F) 128×128×32 tile with 8 warps (4×2 layout)
//   (G) Marlin-style offline permutation for B: single uint32 load per
//       mma fragment per thread (replaces 4 scalar byte loads)
//
// Weight: B stored in tiled-permuted format. Each [BLOCK_K, BLOCK_N] tile
// has bytes rearranged so thread T in a warp can load its 4 FP8 values
// for an mma.m16n8k16 B-fragment via a single aligned uint32 read.
// Use repack_fp8_for_mma() to prepare weights from [K, N] row-major.

#include <cstdint>
#include <algorithm>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Tile config — the kernel is templated on BLOCK_M_T (64 or 128).
// BLOCK_M=128 is the default large-tile path.
// BLOCK_M=64  is dispatched for small M to reduce wave-quantisation overhead.
constexpr int BLOCK_M  = 128;  // default tile height; also used by repack_fp8_for_mma
constexpr int BLOCK_N  = 128;
constexpr int BLOCK_K  = 32;
constexpr int STAGES   = 3;

// Warp layout: the N dimension always has 2 warp-columns (WARPS_N=2).
// The M dimension uses BLOCK_M_T/32 warp-rows (4 for M128, 2 for M64).
// WARP_M = BLOCK_M_T / WARPS_M_T = 32 for every tile variant.
constexpr int WARPS_N = 2;
constexpr int WARP_M  = 32;                      // always 32 for all tile variants
constexpr int WARP_N  = BLOCK_N / WARPS_N;       // 64

// MMA tile: m16n8k16
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int TM = WARP_M / MMA_M;  // 2  (same for all tile sizes — WARP_M is always 32)
constexpr int TN = WARP_N / MMA_N;  // 8
constexpr int TK = BLOCK_K / MMA_K; // 2

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Dispatch threshold: use BLOCK_M=64 kernel when M <= this value.
// Below the threshold the 128-row tile produces < 1 full wave on 82 SMs.
constexpr int M_SMALL_THRESHOLD = 256;

// Shared memory layout per stage:
//   A  : bf16 [BLOCK_M_T][A_STRIDE]  — size scales with BLOCK_M_T
//   B  : uint8 [BLOCK_K][BLOCK_N]    — fixed 4096 bytes regardless of BLOCK_M_T
// A_STRIDE padded by 8 bf16 (32→40) for 16-byte cp.async alignment and bank-conflict avoidance.
constexpr int A_STRIDE = BLOCK_K + 8;             // 40 bf16 words
constexpr int B_STAGE  = BLOCK_K * BLOCK_N;       // 4096 bytes
// BLOCK_M=128 smem totals (reference; used for static_assert):
constexpr int TOTAL_SMEM_128 = STAGES * (BLOCK_M * A_STRIDE * 2 + B_STAGE);  // 43008 bytes
// BLOCK_M=64 smem totals:
constexpr int TOTAL_SMEM_64  = STAGES * (64      * A_STRIDE * 2 + B_STAGE);  // 27648 bytes
static_assert(TOTAL_SMEM_128 <= 100 * 1024, "TOTAL_SMEM_128 exceeds 100 KB");
static_assert(TOTAL_SMEM_64  <= 100 * 1024, "TOTAL_SMEM_64 exceeds 100 KB");

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global [%1], [%2], 16;\n"
        "  @!p st.shared.v4.u32 [%1], {0, 0, 0, 0};\n"
        "}\n" :: "r"((int)pred), "r"(s), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// ldmatrix: warp-collective load of matrix fragments from shared memory.
// Loads 4 × 8×8 bf16 matrix tiles (4 registers per thread) in one instruction.
// Much faster than 4 individual shared memory reads.
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

// ---------------------------------------------------------------------------
// FP8 E4M3 → BF16 dequant: 4 uint8 values → 2 bf16x2 (4 bf16 values)
// Output order: out0 = {val0, val1} bf16x2, out1 = {val2, val3} bf16x2
// Input packing: packed = val0 | (val2 << 8) | (val1 << 16) | (val3 << 24)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void dequant_fp8x4(uint32_t packed,
                                               uint32_t& out0, uint32_t& out1) {
    constexpr uint32_t SIGN = 0x80008000u;
    constexpr uint32_t MASK = 0x7F007F00u;
    constexpr uint32_t BIAS = 0x7B807B80u;

    int q = static_cast<int>(packed);
    // hi extracts bytes at positions [15:8] and [31:24] = val2, val3
    int hi = (q & static_cast<int>(SIGN)) | ((q & static_cast<int>(MASK)) >> 4);
    // shift left 8 to bring val0,val1 into extractable positions
    q <<= 8;
    int lo = (q & static_cast<int>(SIGN)) | ((q & static_cast<int>(MASK)) >> 4);

    const nv_bfloat162 bias = *reinterpret_cast<const nv_bfloat162*>(&BIAS);
    nv_bfloat162 r_lo = __hmul2(*reinterpret_cast<nv_bfloat162*>(&lo), bias);
    nv_bfloat162 r_hi = __hmul2(*reinterpret_cast<nv_bfloat162*>(&hi), bias);

    out0 = *reinterpret_cast<uint32_t*>(&r_lo);  // {val0, val1}
    out1 = *reinterpret_cast<uint32_t*>(&r_hi);  // {val2, val3}
}

// ---------------------------------------------------------------------------
// Kernel — templated on BLOCK_M_T (64 or 128).
//
// TM, TN, TK, WARP_M, WARP_N are invariant across tile sizes:
//   WARP_M = BLOCK_M_T / (BLOCK_M_T/32) = 32 always
//   TM     = WARP_M / MMA_M = 2 always
// Only NTHREADS_T, A_STAGE_T, STAGE_SIZE_T, and LOAD_B_ITERS differ.
// ---------------------------------------------------------------------------

template <int BLOCK_M_T>
__global__ void __launch_bounds__(BLOCK_M_T * 2)
fp8_bf16_gemm_kernel_tmpl(
    const nv_bfloat16* __restrict__ A,
    const uint8_t*     __restrict__ B,  // tiled-permuted format
    nv_bfloat16*       __restrict__ C,
    const nv_bfloat16* __restrict__ s,
    int M, int N, int K)
{
    // Per-variant compile-time constants
    constexpr int NTHREADS_T   = BLOCK_M_T * 2;            // 256 (M128) or 128 (M64)
    constexpr int A_STAGE_T    = BLOCK_M_T * A_STRIDE * 2; // 10240 or 5120 bytes
    constexpr int STAGE_SIZE_T = A_STAGE_T + B_STAGE;      // 14336 or 9216 bytes
    // B tile = BLOCK_K*BLOCK_N bytes; each cp.async loads 16 bytes.
    // With NTHREADS_T threads: 1 iter for M128 (256 threads), 2 iter for M64 (128 threads).
    constexpr int LOAD_B_ITERS = BLOCK_K * BLOCK_N / 16 / NTHREADS_T;

    const int bm = blockIdx.y * BLOCK_M_T;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARPS_N;  // 0..BLOCK_M_T/32-1
    const int warp_col = warp_id % WARPS_N;  // 0..1

    extern __shared__ uint8_t smem_raw[];

    // Stage pointers
    auto sh_a = [&](int stage) -> nv_bfloat16* {
        return reinterpret_cast<nv_bfloat16*>(smem_raw + stage * STAGE_SIZE_T);
    };
    auto sh_b = [&](int stage) -> uint8_t* {
        return smem_raw + stage * STAGE_SIZE_T + A_STAGE_T;
    };

    // Accumulators: TM=2 × TN=8, each has 4 floats
    float fragC[TM][TN][4];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            fragC[i][j][0] = fragC[i][j][1] = fragC[i][j][2] = fragC[i][j][3] = 0.f;

    // ── Async copy: A tile [BLOCK_M_T × BLOCK_K] bf16 ─────────────
    // NTHREADS_T × 2 iters = BLOCK_M_T × (BLOCK_K/8) total copies — true for both variants.
    auto load_a = [&](int stage, int k_off) {
        nv_bfloat16* dst = sh_a(stage);
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * NTHREADS_T;
            int row = idx / (BLOCK_K / 8);     // 8 bf16 per 16-byte copy
            int col8 = idx % (BLOCK_K / 8);    // which group of 8
            int gm = bm + row;
            int gk = k_off + col8 * 8;
            bool pred = (gm < M) && (gk + 7 < K);
            cp_async_16(&dst[row * A_STRIDE + col8 * 8],
                        &A[gm * K + gk], pred);
        }
    };

    // ── Async copy: B tile [BLOCK_K × BLOCK_N] uint8 (tiled-permuted) ─
    // B is pre-permuted: each [BLOCK_K, BLOCK_N] tile is 4096 bytes stored
    // contiguously. LOAD_B_ITERS × NTHREADS_T × 16 = 4096 bytes total.
    const int N_tiles = N / BLOCK_N;
    auto load_b = [&](int stage, int k_off) {
        uint8_t* dst = sh_b(stage);
        int kt = k_off / BLOCK_K;
        int nt = bn / BLOCK_N;
        const uint8_t* tile_ptr = B + (kt * N_tiles + nt) * (BLOCK_K * BLOCK_N);
        bool pred = (k_off < K);
        #pragma unroll
        for (int i = 0; i < LOAD_B_ITERS; i++) {
            int offset = (tid + i * NTHREADS_T) * 16;
            cp_async_16(&dst[offset], &tile_ptr[offset], pred);
        }
    };

    // ── Load A fragment via ldmatrix ─────────────────────────────
    // ldmatrix.x4 is a warp-collective instruction that loads 4 × 8×8
    // matrix tiles from shared memory in a single op, distributing the
    // right bf16x2 pairs to each thread for mma.m16n8k16.
    //
    // For matrix_a of m16n8k16 (16 rows × 16 cols), we need 4 sub-matrices:
    //   sub0: rows [m..m+7],   cols [k..k+7]    → reg0
    //   sub1: rows [m+8..m+15],cols [k..k+7]    → reg1
    //   sub2: rows [m..m+7],   cols [k+8..k+15] → reg2
    //   sub3: rows [m+8..m+15],cols [k+8..k+15] → reg3
    //
    // Each thread in the warp provides the address of one row:
    //   thread t → row (t%8) of sub-matrix (t/8)
    // The address must point to 16 bytes (8 bf16) of contiguous data.

    auto load_a_frag = [&](uint32_t frag[4], const nv_bfloat16* sa,
                           int m_off, int k_off) {
        // Thread t provides address for row (t%8) of sub-matrix (t/8):
        //   sub 0: sa[(m_off + t%8)      * A_STRIDE + k_off]
        //   sub 1: sa[(m_off + 8 + t%8)  * A_STRIDE + k_off]
        //   sub 2: sa[(m_off + t%8)      * A_STRIDE + k_off + 8]
        //   sub 3: sa[(m_off + 8 + t%8)  * A_STRIDE + k_off + 8]
        int sub = lane / 8;       // 0..3
        int row_in_sub = lane % 8; // 0..7
        int row = m_off + (sub & 1) * 8 + row_in_sub;  // sub0,2 → m_off; sub1,3 → m_off+8
        int col = k_off + (sub >> 1) * 8;               // sub0,1 → k_off; sub2,3 → k_off+8
        const nv_bfloat16* addr = &sa[row * A_STRIDE + col];
        ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], addr);
    };

    // ── Load B fragment for mma.m16n8k16 and dequant ─────────────
    // B in shared memory is in tiled-permuted layout:
    //   32 mma sub-tiles (2 k-tiles × 16 n-tiles), each 128 bytes.
    //   Within each 128-byte sub-tile, thread T (lane) loads uint32 at
    //   offset T*4, which contains its 4 pre-arranged FP8 values:
    //     byte0 = B_orig[k0, col]     (v0)
    //     byte1 = B_orig[k0+8, col]   (v2)
    //     byte2 = B_orig[k0+1, col]   (v1)
    //     byte3 = B_orig[k0+9, col]   (v3)
    //   where k0 = (lane%4)*2, col = n_off + lane/4
    //
    // Single uint32 load → zero bank conflicts, 4× fewer instructions.
    constexpr int MMA_TILE_BYTES = 128;  // 32 threads × 4 bytes
    constexpr int N_MMA_TILES = BLOCK_N / MMA_N;  // 16

    auto load_b_frag = [&](uint32_t frag[2], const uint8_t* sb,
                           int n_off, int k_off) {
        int k_tile = k_off / MMA_K;              // 0 or 1
        int n_tile = n_off / MMA_N;              // 0..15
        int tile_idx = k_tile * N_MMA_TILES + n_tile;
        const uint32_t* tile_base = reinterpret_cast<const uint32_t*>(
            sb + tile_idx * MMA_TILE_BYTES);
        uint32_t packed = tile_base[lane];
        dequant_fp8x4(packed, frag[0], frag[1]);
    };

    // ── Prime pipeline ───────────────────────────────────────────
    int num_k = ceildiv(K, BLOCK_K);

    #pragma unroll
    for (int p = 0; p < STAGES - 1 && p < num_k; p++) {
        load_a(p, p * BLOCK_K);
        load_b(p, p * BLOCK_K);
        cp_async_fence();
    }

    // ── Main K-loop ──────────────────────────────────────────────
    for (int kt = 0; kt < num_k; kt++) {
        int stage = kt % STAGES;

        // Prefetch
        int pf = kt + STAGES - 1;
        if (pf < num_k) {
            load_a(pf % STAGES, pf * BLOCK_K);
            load_b(pf % STAGES, pf * BLOCK_K);
        }
        cp_async_fence();

        // Wait for current stage
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const nv_bfloat16* sa = sh_a(stage);
        const uint8_t* sb = sh_b(stage);

        // Compute — software-pipelined inner loop:
        // For each kk: bulk-load+dequant ALL B fragments first, then batch MMAs.
        // This separates ALU (dequant) from tensor core (MMA) work, letting the
        // warp scheduler overlap them across instruction issue slots.
        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            // Load all A fragments for this kk
            uint32_t fragA[TM][4];
            #pragma unroll
            for (int mi = 0; mi < TM; mi++) {
                int m_off = warp_row * WARP_M + mi * MMA_M;
                load_a_frag(fragA[mi], sa, m_off, kk * MMA_K);
            }

            // Bulk-load and dequant ALL B fragments for this kk
            uint32_t fragB[TN][2];
            #pragma unroll
            for (int ni = 0; ni < TN; ni++) {
                int n_off = warp_col * WARP_N + ni * MMA_N;
                load_b_frag(fragB[ni], sb, n_off, kk * MMA_K);
            }

            // Batch all MMAs — tensor core ops only, no ALU dependency stalls
            #pragma unroll
            for (int ni = 0; ni < TN; ni++) {
                #pragma unroll
                for (int mi = 0; mi < TM; mi++) {
                    mma_m16n8k16(
                        fragC[mi][ni][0], fragC[mi][ni][1],
                        fragC[mi][ni][2], fragC[mi][ni][3],
                        fragA[mi][0], fragA[mi][1],
                        fragA[mi][2], fragA[mi][3],
                        fragB[ni][0], fragB[ni][1]);
                }
            }
        }
        __syncthreads();
    }

    // ── Epilogue: scale + vectorized bf16x2 write from registers ─
    // mma.m16n8k16 accumulator layout:
    //   c[0],c[1] → (row=lane/4,       col=(lane%4)*2, col+1)  bf16x2 pair
    //   c[2],c[3] → (row=lane/4 + 8,   col=(lane%4)*2, col+1)  bf16x2 pair
    int group_id = lane / 4;
    int col_pair = (lane % 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < TM; mi++) {
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int base_row = bm + warp_row * WARP_M + mi * MMA_M;
            int base_col = bn + warp_col * WARP_N + ni * MMA_N + col_pair;

            float sc0 = __bfloat162float(s[base_col]);
            float sc1 = __bfloat162float(s[base_col + 1]);

            int r0 = base_row + group_id;
            if (r0 < M) {
                nv_bfloat162 v;
                v.x = __float2bfloat16(fragC[mi][ni][0] * sc0);
                v.y = __float2bfloat16(fragC[mi][ni][1] * sc1);
                *reinterpret_cast<nv_bfloat162*>(&C[r0 * N + base_col]) = v;
            }

            int r1 = base_row + group_id + 8;
            if (r1 < M) {
                nv_bfloat162 v;
                v.x = __float2bfloat16(fragC[mi][ni][2] * sc0);
                v.y = __float2bfloat16(fragC[mi][ni][3] * sc1);
                *reinterpret_cast<nv_bfloat162*>(&C[r1 * N + base_col]) = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher — dispatches BLOCK_M=64 for small M, BLOCK_M=128 otherwise.
//
// Both smem sizes are < 48 KB so no cudaFuncSetAttribute call is needed.
// ---------------------------------------------------------------------------

static void launch_fp8_bf16_gemm(
    const void* A, const void* B, void* C, const void* s,
    int M, int N, int K, cudaStream_t stream)
{
    if (M <= M_SMALL_THRESHOLD) {
        // BLOCK_M=64 kernel: 2× blocks vs M128 → better SM utilisation for small M.
        // smem = 3 × (64×40×2 + 4096) = 27648 bytes — well within default 48 KB limit.
        constexpr int BM   = 64;
        constexpr int SMEM = STAGES * (BM * A_STRIDE * 2 + B_STAGE);  // 27648
        dim3 grid(ceildiv(N, BLOCK_N), ceildiv(M, BM));
        dim3 block(BM * 2);  // 128 threads
        fp8_bf16_gemm_kernel_tmpl<BM><<<grid, block, SMEM, stream>>>(
            (const nv_bfloat16*)A, (const uint8_t*)B,
            (nv_bfloat16*)C, (const nv_bfloat16*)s,
            M, N, K);
    } else {
        // BLOCK_M=128 kernel: default large-batch path.
        // smem = 3 × (128×40×2 + 4096) = 43008 bytes — within default 48 KB limit.
        constexpr int BM   = 128;
        constexpr int SMEM = STAGES * (BM * A_STRIDE * 2 + B_STAGE);  // 43008
        dim3 grid(ceildiv(N, BLOCK_N), ceildiv(M, BM));
        dim3 block(BM * 2);  // 256 threads
        fp8_bf16_gemm_kernel_tmpl<BM><<<grid, block, SMEM, stream>>>(
            (const nv_bfloat16*)A, (const uint8_t*)B,
            (nv_bfloat16*)C, (const nv_bfloat16*)s,
            M, N, K);
    }
}

// ---------------------------------------------------------------------------
// PyTorch binding
// ---------------------------------------------------------------------------

torch::Tensor fp8_ampere_mm(
    const torch::Tensor& A,
    const torch::Tensor& W,
    const torch::Tensor& ws)
{
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && ws.is_cuda());
    TORCH_CHECK(A.is_contiguous() && W.is_contiguous() && ws.is_contiguous());
    TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(W.scalar_type() == at::kByte, "W must be uint8 (tiled-permuted)");
    TORCH_CHECK(ws.scalar_type() == at::kBFloat16, "ws must be bfloat16");

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)ws.size(0);

    TORCH_CHECK(K % BLOCK_K == 0, "K must be divisible by ", BLOCK_K);
    TORCH_CHECK(N % BLOCK_N == 0, "N must be divisible by ", BLOCK_N);
    TORCH_CHECK(W.numel() == K * N,
                "W must have K*N elements (tiled-permuted)");

    c10::cuda::CUDAGuard device_guard(A.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto C = torch::empty({M, N}, A.options().dtype(at::kBFloat16));

    launch_fp8_bf16_gemm(
        A.data_ptr(), W.data_ptr(), C.data_ptr(), ws.data_ptr(),
        M, N, K, stream);
    return C;
}

// ---------------------------------------------------------------------------
// Offline weight permutation: rearrange FP8 bytes for single-load mma frags.
//
// Input:  weight_fp8 [N, K] in FP8 E4M3 (standard PyTorch layout)
// Output: [K_pad*N_pad] uint8 flat in tiled-permuted format
//
// Single-pass: reads directly from [N, K] layout with bounds checking,
// eliminating the separate transpose + temp buffer allocation.
// ---------------------------------------------------------------------------

torch::Tensor repack_fp8_for_mma(const torch::Tensor& weight_fp8) {
    TORCH_CHECK(weight_fp8.is_contiguous());
    TORCH_CHECK(weight_fp8.dim() == 2);
    TORCH_CHECK(weight_fp8.scalar_type() == at::kByte ||
                weight_fp8.scalar_type() == at::kFloat8_e4m3fn);

    auto w = (weight_fp8.scalar_type() == at::kFloat8_e4m3fn)
           ? weight_fp8.view(at::kByte) : weight_fp8;

    const int N_orig = w.size(0);
    const int K_orig = w.size(1);

    const int K = ((K_orig + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
    const int N = ((N_orig + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;

    // Read directly from [N_orig, K_orig] — no transpose needed
    auto w_cpu = w.cpu();
    auto src_ptr = w_cpu.data_ptr<uint8_t>();

    auto out = torch::zeros({K * N}, w_cpu.options().dtype(at::kByte));
    auto dst_ptr = out.data_ptr<uint8_t>();

    // Bounds-checked read from [N, K] row-major layout
    auto rd = [&](int n, int k) -> uint8_t {
        return (n < N_orig && k < K_orig) ? src_ptr[n * K_orig + k] : 0;
    };

    const int K_tiles = K / BLOCK_K;
    const int N_tiles = N / BLOCK_N;
    constexpr int n_mma_n = BLOCK_N / MMA_N;  // 16

    for (int kt = 0; kt < K_tiles; kt++) {
        for (int nt = 0; nt < N_tiles; nt++) {
            int tile_byte_offset = (kt * N_tiles + nt) * (BLOCK_K * BLOCK_N);

            for (int k_sub = 0; k_sub < BLOCK_K / MMA_K; k_sub++) {
                int abs_k_base = kt * BLOCK_K + k_sub * MMA_K;

                for (int n_sub = 0; n_sub < n_mma_n; n_sub++) {
                    int sub_tile_offset = tile_byte_offset
                        + (k_sub * n_mma_n + n_sub) * 128;
                    int abs_n_base = nt * BLOCK_N + n_sub * MMA_N;

                    for (int T = 0; T < 32; T++) {
                        int abs_n = abs_n_base + T / 4;
                        int k0    = abs_k_base + (T % 4) * 2;

                        int d = sub_tile_offset + T * 4;
                        dst_ptr[d + 0] = rd(abs_n, k0);
                        dst_ptr[d + 1] = rd(abs_n, k0 + 8);
                        dst_ptr[d + 2] = rd(abs_n, k0 + 1);
                        dst_ptr[d + 3] = rd(abs_n, k0 + 9);
                    }
                }
            }
        }
    }

    return out.to(weight_fp8.device());
}

torch::Tensor compute_fp8_scales(const torch::Tensor& weight_fp8) {
    TORCH_CHECK(weight_fp8.dim() == 2);
    int N = weight_fp8.size(0);
    return torch::ones({N}, weight_fp8.options().dtype(at::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_ampere_mm", &fp8_ampere_mm,
          "FP8xBF16 GEMM via BF16 tensor cores (v5, permuted B).");
    m.def("repack_fp8_for_mma", &repack_fp8_for_mma,
          "Permute FP8 [N,K] -> tiled-permuted [K*N] for mma fragment loads.");
    m.def("compute_fp8_scales", &compute_fp8_scales,
          "Per-channel BF16 scales (1.0 placeholder).");
}
