// FP8-weight × BF16-activation GEMM for Ampere GPUs (SM 8.x).
// v4 — Optimized kernel:
//   (A) Inline PTX mma.sync.aligned.m16n8k16 for BF16 tensor cores
//   (B) 4-stage cp.async pipeline overlapping memory/compute
//   (C) FP8 E4M3 → BF16 register dequant (vLLM bitwise trick)
//   (D) Direct register → global epilogue (no smem_out staging)
//   (E) Per-channel BF16 scale fused into epilogue
//   (F) 128×128×32 tile with 8 warps (4×2 layout)
//
// Weight: B stored as [K, N] uint8 (transposed from [N, K]).

#include <cstdint>
#include <algorithm>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Tile config
constexpr int BLOCK_M  = 128;
constexpr int BLOCK_N  = 128;
constexpr int BLOCK_K  = 32;
constexpr int STAGES   = 4;
constexpr int NTHREADS = 256;  // 8 warps

// Warp layout: 4M × 2N
constexpr int WARPS_M = 4, WARPS_N = 2;
constexpr int WARP_M  = BLOCK_M / WARPS_M;  // 32
constexpr int WARP_N  = BLOCK_N / WARPS_N;  // 64

// MMA tile: m16n8k16
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int TM = WARP_M / MMA_M;  // 2
constexpr int TN = WARP_N / MMA_N;  // 8
constexpr int TK = BLOCK_K / MMA_K; // 2

static_assert(WARPS_M * WARPS_N == NTHREADS / 32);

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Shared memory: A is bf16 [BLOCK_M][BLOCK_K], B is uint8 [BLOCK_K][BLOCK_N]
constexpr int A_STAGE = BLOCK_M * BLOCK_K * 2;  // 8192 bytes
constexpr int B_STAGE = BLOCK_K * BLOCK_N;       // 4096 bytes
constexpr int STAGE_SIZE = A_STAGE + B_STAGE;    // 12288 bytes
constexpr int TOTAL_SMEM = STAGES * STAGE_SIZE;  // 49152 bytes
static_assert(TOTAL_SMEM <= 49152);

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
// Kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(NTHREADS)
fp8_bf16_gemm_kernel_v4(
    const nv_bfloat16* __restrict__ A,
    const uint8_t*     __restrict__ B,
    nv_bfloat16*       __restrict__ C,
    const nv_bfloat16* __restrict__ s,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id / WARPS_N;  // 0..3
    const int warp_col = warp_id % WARPS_N;  // 0..1

    extern __shared__ uint8_t smem_raw[];

    // Stage pointers
    auto sh_a = [&](int stage) -> nv_bfloat16* {
        return reinterpret_cast<nv_bfloat16*>(smem_raw + stage * STAGE_SIZE);
    };
    auto sh_b = [&](int stage) -> uint8_t* {
        return smem_raw + stage * STAGE_SIZE + A_STAGE;
    };

    // Accumulators: TM=2 × TN=8, each has 4 floats
    float fragC[TM][TN][4];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            fragC[i][j][0] = fragC[i][j][1] = fragC[i][j][2] = fragC[i][j][3] = 0.f;

    // ── Async copy: A tile [BLOCK_M × BLOCK_K] bf16 ─────────────
    // 8192 bytes = 512 × 16-byte. 256 threads → 2 copies/thread.
    auto load_a = [&](int stage, int k_off) {
        nv_bfloat16* dst = sh_a(stage);
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BLOCK_K / 8);     // 8 bf16 per 16-byte copy
            int col8 = idx % (BLOCK_K / 8);    // which group of 8
            int gm = bm + row;
            int gk = k_off + col8 * 8;
            bool pred = (gm < M) && (gk + 7 < K);
            cp_async_16(&dst[row * BLOCK_K + col8 * 8],
                        &A[gm * K + gk], pred);
        }
    };

    // ── Async copy: B tile [BLOCK_K × BLOCK_N] uint8 ────────────
    // 4096 bytes = 256 × 16-byte. 256 threads → 1 copy/thread.
    auto load_b = [&](int stage, int k_off) {
        uint8_t* dst = sh_b(stage);
        int row = tid / (BLOCK_N / 16);    // 0..31
        int col16 = tid % (BLOCK_N / 16);  // 0..7
        int gk = k_off + row;
        int gn = bn + col16 * 16;
        bool pred = (gk < K) && (gn + 15 < N);
        cp_async_16(&dst[row * BLOCK_N + col16 * 16],
                    &B[gk * N + gn], pred);
    };

    // ── Load A fragment for mma.m16n8k16 ─────────────────────────
    // A is row-major [M, K]. Fragment layout for matrix_a (row-major):
    //   a[0] = {A[row0, k0], A[row0, k1]} as bf16x2
    //   a[1] = {A[row1, k0], A[row1, k1]} as bf16x2  (row1 = row0 + 8)
    //   a[2] = {A[row0, k0+8], A[row0, k1+8]} as bf16x2  (NOT: a[2] and a[3] are for k+8)
    //   a[3] = {A[row1, k0+8], A[row1, k1+8]} as bf16x2
    // where row0 = lane/4, k0 = (lane%4)*2
    // Wait, that's for the 4-register m16n8k16 layout. Let me use the correct one:
    //
    // For mma.m16n8k16 matrix_a (4 uint32 regs):
    //   Thread (lane) contributes:
    //     reg0 = {A[row, k0], A[row, k0+1]}      (bf16x2)
    //     reg1 = {A[row+8, k0], A[row+8, k0+1]}  (bf16x2)
    //     reg2 = {A[row, k0+8], A[row, k0+9]}    (bf16x2)
    //     reg3 = {A[row+8, k0+8], A[row+8, k0+9]}(bf16x2)
    //   where row = lane / 4, k0 = (lane % 4) * 2

    auto load_a_frag = [&](uint32_t frag[4], const nv_bfloat16* sa,
                           int m_off, int k_off) {
        int row = m_off + lane / 4;
        int k0  = k_off + (lane % 4) * 2;
        const nv_bfloat16* base = &sa[row * BLOCK_K + k0];
        frag[0] = *reinterpret_cast<const uint32_t*>(base);
        frag[1] = *reinterpret_cast<const uint32_t*>(base + 8 * BLOCK_K);  // row+8
        frag[2] = *reinterpret_cast<const uint32_t*>(base + 8);            // k+8
        frag[3] = *reinterpret_cast<const uint32_t*>(base + 8 * BLOCK_K + 8); // row+8, k+8
    };

    // ── Load B fragment for mma.m16n8k16 and dequant ─────────────
    // For mma.m16n8k16 matrix_b (col-major, 2 uint32 regs):
    //   reg0 = {B[k0, col], B[k0+1, col]}        (bf16x2)
    //   reg1 = {B[k0+8, col], B[k0+8+1, col]}    (bf16x2)
    //   where k0 = (lane % 4) * 2, col = lane / 4

    auto load_b_frag = [&](uint32_t frag[2], const uint8_t* sb,
                           int n_off, int k_off) {
        int col = n_off + lane / 4;
        int k0  = k_off + (lane % 4) * 2;
        // Load 4 FP8 bytes
        uint8_t v0 = sb[k0       * BLOCK_N + col];
        uint8_t v1 = sb[(k0 + 1) * BLOCK_N + col];
        uint8_t v2 = sb[(k0 + 8) * BLOCK_N + col];
        uint8_t v3 = sb[(k0 + 9) * BLOCK_N + col];
        // Pack: byte0=v0, byte1=v2, byte2=v1, byte3=v3
        // dequant produces: out0={v0,v1}, out1={v2,v3}
        uint32_t packed = v0 | (v2 << 8) | (v1 << 16) | (v3 << 24);
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

        // Compute
        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            // Load A fragments
            uint32_t fragA[TM][4];
            #pragma unroll
            for (int mi = 0; mi < TM; mi++) {
                int m_off = warp_row * WARP_M + mi * MMA_M;
                load_a_frag(fragA[mi], sa, m_off, kk * MMA_K);
            }

            // Load B, dequant, and MMA
            #pragma unroll
            for (int ni = 0; ni < TN; ni++) {
                uint32_t fragB[2];
                int n_off = warp_col * WARP_N + ni * MMA_N;
                load_b_frag(fragB, sb, n_off, kk * MMA_K);

                #pragma unroll
                for (int mi = 0; mi < TM; mi++) {
                    mma_m16n8k16(
                        fragC[mi][ni][0], fragC[mi][ni][1],
                        fragC[mi][ni][2], fragC[mi][ni][3],
                        fragA[mi][0], fragA[mi][1],
                        fragA[mi][2], fragA[mi][3],
                        fragB[0], fragB[1]);
                }
            }
        }
        __syncthreads();
    }

    // ── Epilogue: scale + write from registers ───────────────────
    // mma.m16n8k16 accumulator layout:
    //   c[0] → (row=lane/4,       col=(lane%4)*2)     and col+1
    //   c[2] → (row=lane/4 + 8,   col=(lane%4)*2)     and col+1
    int group_id = lane / 4;
    int col_pair = (lane % 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < TM; mi++) {
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int base_row = bm + warp_row * WARP_M + mi * MMA_M;
            int base_col = bn + warp_col * WARP_N + ni * MMA_N + col_pair;

            float sc0 = (base_col < N) ? __bfloat162float(s[base_col]) : 0.f;
            float sc1 = (base_col + 1 < N) ? __bfloat162float(s[base_col + 1]) : 0.f;

            int r0 = base_row + group_id;
            if (r0 < M && base_col < N) {
                C[r0 * N + base_col] = __float2bfloat16(fragC[mi][ni][0] * sc0);
                if (base_col + 1 < N)
                    C[r0 * N + base_col + 1] = __float2bfloat16(fragC[mi][ni][1] * sc1);
            }

            int r1 = base_row + group_id + 8;
            if (r1 < M && base_col < N) {
                C[r1 * N + base_col] = __float2bfloat16(fragC[mi][ni][2] * sc0);
                if (base_col + 1 < N)
                    C[r1 * N + base_col + 1] = __float2bfloat16(fragC[mi][ni][3] * sc1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

static void launch_fp8_bf16_gemm(
    const void* A, const void* B, void* C, const void* s,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 grid(ceildiv(N, BLOCK_N), ceildiv(M, BLOCK_M));
    dim3 block(NTHREADS);

    fp8_bf16_gemm_kernel_v4<<<grid, block, TOTAL_SMEM, stream>>>(
        (const nv_bfloat16*)A, (const uint8_t*)B,
        (nv_bfloat16*)C, (const nv_bfloat16*)s,
        M, N, K);
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
    TORCH_CHECK(W.scalar_type() == at::kByte, "W must be uint8 [K,N]");
    TORCH_CHECK(ws.scalar_type() == at::kBFloat16, "ws must be bfloat16");

    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)ws.size(0);

    TORCH_CHECK(W.size(0) == K && W.size(1) == N,
                "W must be [K, N], got [", W.size(0), ", ", W.size(1), "]");

    c10::cuda::CUDAGuard device_guard(A.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto C = torch::empty({M, N}, A.options().dtype(at::kBFloat16));

    launch_fp8_bf16_gemm(
        A.data_ptr(), W.data_ptr(), C.data_ptr(), ws.data_ptr(),
        M, N, K, stream);
    return C;
}

torch::Tensor repack_fp8_for_marlin(const torch::Tensor& weight_fp8) {
    TORCH_CHECK(weight_fp8.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(weight_fp8.dim() == 2);
    TORCH_CHECK(weight_fp8.scalar_type() == at::kByte ||
                weight_fp8.scalar_type() == at::kFloat8_e4m3fn);
    auto w = (weight_fp8.scalar_type() == at::kFloat8_e4m3fn)
           ? weight_fp8.view(at::kByte) : weight_fp8;
    return w.t().contiguous();
}

torch::Tensor compute_marlin_scales(const torch::Tensor& weight_fp8) {
    TORCH_CHECK(weight_fp8.dim() == 2);
    int N = weight_fp8.size(0);
    return torch::ones({N}, weight_fp8.options().dtype(at::kBFloat16));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_ampere_mm", &fp8_ampere_mm,
          "FP8×BF16 GEMM via BF16 tensor cores (v4).");
    m.def("repack_fp8_for_marlin", &repack_fp8_for_marlin,
          "Transpose FP8 [N,K] → [K,N] uint8.");
    m.def("compute_marlin_scales", &compute_marlin_scales,
          "Per-channel BF16 scales (1.0).");
}
