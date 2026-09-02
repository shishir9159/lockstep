// MXFP4 GEMM on Hopper FP8 tensor cores, block-32 scaled.
//
//   C[M,N] = sum_kb  scaleA[kb,m] * scaleB[kb,n] * (A[m, 32kb:32kb+32] . B[n, ...])
//
// One mainloop iteration == one MXFP4 block of 32, so the FP32 promotion that
// applies the block scales is also the promotion that keeps every partial sum
// inside the ~14-bit Hopper FP8 accumulator. Worst case for a 32-deep FP4 dot
// product is 144*32 = 4608 quarter-units = 13 bits, so the mainloop is exact.
//
// Layouts (all K-contiguous, so both operands feed the MMA the same way):
//   A   [M, K]      e4m3 bytes on the E2M1 grid
//   Bt  [N, K]      e4m3 bytes            -- C = A * Bt^T
//   SA  [K/32, M]   fp32, scale-major so the per-stage load is coalesced
//   SB  [K/32, N]   fp32
//   C   [M, N]      fp32
//
// READ THIS BEFORE JUDGING THE THROUGHPUT NUMBER.
//
// This kernel is a readable *reference* for the block-scaled mainloop, not the
// fast path. Disassembling it for sm_90a shows why:
//
//     32 x HMMA.16816.F32          <- FP16 tensor core, k=16
//     48 x F2FP.F16.E4M3.UNPACK_B  <- FP8 -> FP16 conversion
//
// There are 16 mma.sync per k-step in this kernel, so ptxas turned each
// m16n8k32 e4m3 MMA into TWO FP16 m16n8k16 MMAs plus conversions. On Hopper the
// warp-level mma.sync FP8 path is emulated on the FP16 datapath; only the
// warp-group instruction (wgmma.mma_async ... .e4m3) reaches the native FP8
// tensor core and the 1979 TFLOP/s rate. That is why CUTLASS, DeepGEMM and
// Triton all use wgmma on sm_90.
//
// So: this file is for understanding and for correctness. For throughput use
// ../mxfp4_gemm.py (Triton emits wgmma) and run `python check_isa.py` to
// confirm which datapath your build actually picked. Porting this mainloop to
// wgmma means replacing the fragment loads with 64-bit SMEM matrix descriptors
// and a swizzled shared layout; the surrounding structure -- one MXFP4 block
// per iteration, scale-multiply into a separate FP32 accumulator -- is what is
// worth copying and does not change.
//
//   nvcc -O3 -arch=sm_90a -o mxfp4 mxfp4_mma_gemm.cu
//   ./mxfp4 --check          # correctness against a double-precision reference
//   ./mxfp4 4096 4096 4096   # timing

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "fp4_unpack.cuh"

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                 \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));       \
    exit(1); } } while (0)

constexpr int BM = 128, BN = 128, BK = 32;      // BK == the MXFP4 block size
constexpr int WARPS_M = 4, WARPS_N = 2;
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;   // 256
constexpr int WM = BM / WARPS_M;                   // 32
constexpr int WN = BN / WARPS_N;                   // 64
constexpr int MT = WM / 16;                        // 2 m-tiles per warp
constexpr int NT = WN / 8;                         // 8 n-tiles per warp
constexpr int STAGES = 4;

#define MMA_E4M3(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)                          \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "           \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"        \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                         \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ __launch_bounds__(NTHREADS) void mxfp4_gemm_kernel(
        const uint8_t *__restrict__ A, const uint8_t *__restrict__ Bt,
        const float *__restrict__ SA, const float *__restrict__ SB,
        float *__restrict__ C, int M, int N, int K) {
    __shared__ __align__(16) uint8_t sA[STAGES][BM * BK];
    __shared__ __align__(16) uint8_t sB[STAGES][BN * BK];
    __shared__ float sSA[STAGES][BM];
    __shared__ float sSB[STAGES][BN];

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int t = threadIdx.x;
    const int lane = t & 31, warp = t >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;

    // 256 threads x 16B == one 128x32 byte tile, exactly.
    const int ld_row = t >> 1, ld_col = (t & 1) * 16;
    const int nk = K / BK;

    auto load_stage = [&](int st, int kb) {
        const int k = kb * BK;
        cp_async16(smem_u32(&sA[st][ld_row * BK + ld_col]),
                   A + (size_t)(m0 + ld_row) * K + k + ld_col);
        cp_async16(smem_u32(&sB[st][ld_row * BK + ld_col]),
                   Bt + (size_t)(n0 + ld_row) * K + k + ld_col);
        if (t < BM) sSA[st][t] = SA[(size_t)kb * M + m0 + t];
        if (t < BN) sSB[st][t] = SB[(size_t)kb * N + n0 + t];
    };

    for (int s = 0; s < STAGES - 1; ++s) {
        load_stage(s, s);
        cp_async_commit();
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    for (int kb = 0; kb < nk; ++kb) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int st = kb % STAGES;

        // ---- fragments -------------------------------------------------------
        // m16n8k32 operand layout: each lane holds 4 contiguous bytes at
        // (row = lane/4, col = 4*(lane%4)), plus row+8 and col+16.
        uint32_t af[MT][4], bf[NT][2];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const uint8_t *p = &sA[st][(wm * WM + i * 16 + (lane >> 2)) * BK + (lane & 3) * 4];
            af[i][0] = *reinterpret_cast<const uint32_t *>(p);
            af[i][1] = *reinterpret_cast<const uint32_t *>(p + 8 * BK);
            af[i][2] = *reinterpret_cast<const uint32_t *>(p + 16);
            af[i][3] = *reinterpret_cast<const uint32_t *>(p + 8 * BK + 16);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const uint8_t *q = &sB[st][(wn * WN + j * 8 + (lane >> 2)) * BK + (lane & 3) * 4];
            bf[j][0] = *reinterpret_cast<const uint32_t *>(q);
            bf[j][1] = *reinterpret_cast<const uint32_t *>(q + 16);
        }
        // accumulator element (i,j,{0,1,2,3}) lives at
        //   row = wm*WM + i*16 + lane/4 (+8 for elements 2,3)
        //   col = wn*WN + j*8  + 2*(lane%4) (+1 for elements 1,3)
        float sal[MT], sah[MT], sb0[NT], sb1[NT];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            const int r = wm * WM + i * 16 + (lane >> 2);
            sal[i] = sSA[st][r];
            sah[i] = sSA[st][r + 8];
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int c = wn * WN + j * 8 + 2 * (lane & 3);
            sb0[j] = sSB[st][c];
            sb1[j] = sSB[st][c + 1];
        }

        // ---- prefetch the stage we will overwrite next -------------------------
        const int next = kb + STAGES - 1;
        if (next < nk) load_stage(next % STAGES, next);
        cp_async_commit();

        // ---- MMA, then promote to FP32 with the block scales -------------------
#pragma unroll
        for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                MMA_E4M3(d0, d1, d2, d3, af[i][0], af[i][1], af[i][2], af[i][3],
                         bf[j][0], bf[j][1]);
                acc[i][j][0] = fmaf(d0, sal[i] * sb0[j], acc[i][j][0]);
                acc[i][j][1] = fmaf(d1, sal[i] * sb1[j], acc[i][j][1]);
                acc[i][j][2] = fmaf(d2, sah[i] * sb0[j], acc[i][j][2]);
                acc[i][j][3] = fmaf(d3, sah[i] * sb1[j], acc[i][j][3]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r = m0 + wm * WM + i * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int c = n0 + wn * WN + j * 8 + 2 * (lane & 3);
            *reinterpret_cast<float2 *>(&C[(size_t)r * N + c]) =
                make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2 *>(&C[(size_t)(r + 8) * N + c]) =
                make_float2(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// Standalone FP4 -> FP8 expansion, to time the unpack path on its own.
__global__ void unpack_kernel(const uint32_t *__restrict__ in,
                              uint2 *__restrict__ out, size_t n32) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n32) {
        uint32_t lo, hi;
        fp4x8_to_e4m3x8(in[i], &lo, &hi);
        out[i] = make_uint2(lo, hi);
    }
}

// --------------------------------------------------------------------------- //
static void run(int M, int N, int K, bool check) {
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "M,N must be multiples of %d/%d and K of %d\n", BM, BN, BK);
        exit(1);
    }
    const int nkb = K / BK;
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> code(0, 15);
    std::uniform_int_distribution<int> sexp(-3, 3);

    std::vector<uint8_t> hA((size_t)M * K), hB((size_t)N * K), cA((size_t)M * K), cB((size_t)N * K);
    std::vector<float> hSA((size_t)nkb * M), hSB((size_t)nkb * N);
    for (size_t i = 0; i < hA.size(); ++i) { cA[i] = code(rng); hA[i] = fp4_code_to_e4m3_ref(cA[i]); }
    for (size_t i = 0; i < hB.size(); ++i) { cB[i] = code(rng); hB[i] = fp4_code_to_e4m3_ref(cB[i]); }
    for (auto &s : hSA) s = ldexpf(1.f, sexp(rng));
    for (auto &s : hSB) s = ldexpf(1.f, sexp(rng));

    uint8_t *dA, *dB; float *dSA, *dSB, *dC;
    CHECK(cudaMalloc(&dA, hA.size()));
    CHECK(cudaMalloc(&dB, hB.size()));
    CHECK(cudaMalloc(&dSA, hSA.size() * 4));
    CHECK(cudaMalloc(&dSB, hSB.size() * 4));
    CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
    CHECK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dSA, hSA.data(), hSA.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dSB, hSB.data(), hSB.size() * 4, cudaMemcpyHostToDevice));

    dim3 grid(N / BN, M / BM), blk(NTHREADS);
    mxfp4_gemm_kernel<<<grid, blk>>>(dA, dB, dSA, dSB, dC, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    if (check) {
        std::vector<float> hC((size_t)M * N);
        CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (int m = 0; m < M; m += 37) {
            for (int n = 0; n < N; n += 41) {
                double ref = 0.0;
                for (int kb = 0; kb < nkb; ++kb) {
                    double p = 0.0;
                    for (int k = kb * BK; k < (kb + 1) * BK; ++k)
                        p += (double)fp4_code_to_float(cA[(size_t)m * K + k]) *
                             (double)fp4_code_to_float(cB[(size_t)n * K + k]);
                    ref += p * hSA[(size_t)kb * M + m] * hSB[(size_t)kb * N + n];
                }
                double got = hC[(size_t)m * N + n];
                double d = fabs(got - ref) / (fabs(ref) + 1e-9);
                if (d > worst) worst = d;
            }
        }
        printf("  max rel err vs double reference: %.3e  %s\n", worst,
               worst < 1e-6 ? "OK" : "FAIL");
    }

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    const int iters = 50;
    for (int i = 0; i < 5; ++i) mxfp4_gemm_kernel<<<grid, blk>>>(dA, dB, dSA, dSB, dC, M, N, K);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) mxfp4_gemm_kernel<<<grid, blk>>>(dA, dB, dSA, dSB, dC, M, N, K);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= iters;
    printf("  %dx%dx%d  %.3f ms  %.1f TFLOP/s\n", M, N, K, ms,
           2.0 * M * N * K / (ms * 1e-3) / 1e12);

    // unpack throughput
    const size_t n32 = 64u << 20;
    uint32_t *up; uint2 *uo;
    CHECK(cudaMalloc(&up, n32 * 4));
    CHECK(cudaMalloc(&uo, n32 * 8));
    CHECK(cudaMemset(up, 0x5A, n32 * 4));
    unpack_kernel<<<(n32 + 255) / 256, 256>>>(up, uo, n32);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int i = 0; i < 10; ++i) unpack_kernel<<<(n32 + 255) / 256, 256>>>(up, uo, n32);
    CHECK(cudaEventRecord(e1));
    CHECK(cudaEventSynchronize(e1));
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= 10;
    printf("  fp4->fp8 expansion: %.3f ms for %zu values, %.0f GB/s r+w\n",
           ms, n32 * 8, (n32 * 12.0) / (ms * 1e-3) / 1e9);

    cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dC);
    cudaFree(up); cudaFree(uo);
}

int main(int argc, char **argv) {
    bool check = false;
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 2 && !strcmp(argv[1], "--check")) {
        check = true;
        M = N = 256;
        K = 256;
    } else if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("%s  sm_%d%d\n", p.name, p.major, p.minor);
    if (p.major != 9) printf("  WARNING: not Hopper; the 14-bit FP8 accumulator note is Hopper-specific\n");
    run(M, N, K, check);
    return 0;
}
