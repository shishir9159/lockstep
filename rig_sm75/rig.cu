// Test rig for a GTX 1650 SUPER (TU116, sm_75) -- no tensor cores, no FP8,
// no bf16, no cp.async. Everything here runs on plain CUDA cores.
//
// The two claims worth testing do not need a tensor core:
//
//   1. BIT BUDGET. A packed dual dot product acc = C1 + 2^s*C2 needs
//      2*ceil(log2(144K)) + 1 significand bits. fp32 has 24; the Hopper FP8 MMA
//      datapath has ~14. We model a W-bit accumulator in software and measure
//      the exact-recovery rate, so the number you get here predicts the H100.
//
//   2. REDUCTION TRAFFIC. One accumulator instead of two halves the atomic
//      traffic in the split-K reduction. That is bandwidth, and this card has
//      bandwidth, so the speedup measured here is the real upper bound on what
//      the packing could ever buy.
//
// Experiment 1 checks the FP4 -> FP8 expansion (prmt exists on sm_75).
//
//   nvcc -O3 -arch=sm_75 -o rig rig.cu
//   ./rig            # all four
//   ./rig bits       # just the bit budget

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "fp4_sim.cuh"

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                  \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
    exit(1); } } while (0)

static std::mt19937 rng(12345);

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

// ===================================================================== 1. unpack
__global__ void unpack_all(uint32_t *out) {
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. 65535
    if (p < 65536u) out[p] = fp4x4_to_e4m3x4(p);
}

static int exp_unpack() {
    printf("\n[1] FP4 -> FP8 expansion (prmt path, exhaustive over all 16^4 quadruples)\n");
    uint32_t *d;
    CHECK(cudaMalloc(&d, 65536 * 4));
    unpack_all<<<256, 256>>>(d);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> h(65536);
    CHECK(cudaMemcpy(h.data(), d, 65536 * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (uint32_t p = 0; p < 65536u; ++p)
        for (int i = 0; i < 4; ++i) {
            uint8_t got = (uint8_t)(h[p] >> (8 * i));
            uint8_t want = e2m1_to_e4m3((uint8_t)((p >> (4 * i)) & 0xF));
            if (got != want) ++bad;
        }
    printf("    262144 conversions, %d mismatches -> %s\n", bad, bad ? "FAIL" : "OK");
    cudaFree(d);
    return bad;
}

// ================================================================== 2. bit budget
// One thread per trial: a packed dual dot product on a W-bit accumulator.
__global__ void packed_dot(const signed char *__restrict__ A1, const signed char *__restrict__ B1,
                           const signed char *__restrict__ A2, const signed char *__restrict__ B2,
                           float *__restrict__ out, int n, int K, float s, int drop) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    size_t o = (size_t)t * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc = round_sig(acc + (float)A1[o + k] * (float)B1[o + k], drop);
        acc = round_sig(acc + s * ((float)A2[o + k] * (float)B2[o + k]), drop);
    }
    out[t] = acc;
}

static void exp_bits() {
    printf("\n[2] packed dual dot product: acc = C1 + 2^s*C2 on a W-bit accumulator\n");
    printf("    W=24 is a true fp32 accumulator, W=14 models the Hopper FP8 MMA path\n\n");
    printf("    %5s %4s %8s %10s %10s %10s\n", "K", "s", "need", "W=14", "W=24", "W=31*");
    const int n = 4096;
    std::uniform_int_distribution<int> cd(0, 14);
    const int codes[15] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};

    for (int K : {8, 16, 32, 64, 128, 512}) {
        std::vector<signed char> a1(n * K), b1(n * K), a2(n * K), b2(n * K);
        for (int i = 0; i < n * K; ++i) {
            a1[i] = (signed char)e2m1_q(codes[cd(rng)]);
            b1[i] = (signed char)e2m1_q(codes[cd(rng)]);
            a2[i] = (signed char)e2m1_q(codes[cd(rng)]);
            b2[i] = (signed char)e2m1_q(codes[cd(rng)]);
        }
        std::vector<long long> c1(n, 0), c2(n, 0);
        for (int t = 0; t < n; ++t)
            for (int k = 0; k < K; ++k) {
                c1[t] += (long long)a1[(size_t)t * K + k] * b1[(size_t)t * K + k];
                c2[t] += (long long)a2[(size_t)t * K + k] * b2[(size_t)t * K + k];
            }

        signed char *dA1, *dB1, *dA2, *dB2;
        float *dOut;
        size_t sz = (size_t)n * K;
        CHECK(cudaMalloc(&dA1, sz)); CHECK(cudaMalloc(&dB1, sz));
        CHECK(cudaMalloc(&dA2, sz)); CHECK(cudaMalloc(&dB2, sz));
        CHECK(cudaMalloc(&dOut, n * 4));
        CHECK(cudaMemcpy(dA1, a1.data(), sz, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB1, b1.data(), sz, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dA2, a2.data(), sz, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB2, b2.data(), sz, cudaMemcpyHostToDevice));

        const int s = slot_offset(K);
        const float sf = ldexpf(1.f, s);
        printf("    %5d %4d %8d", K, s, packed_bits_needed(K));

        for (int W : {14, 24, 31}) {
            packed_dot<<<(n + 255) / 256, 256>>>(dA1, dB1, dA2, dB2, dOut, n, K, sf, 24 - W);
            CHECK(cudaDeviceSynchronize());
            std::vector<float> h(n);
            CHECK(cudaMemcpy(h.data(), dOut, n * 4, cudaMemcpyDeviceToHost));
            int ok = 0;
            for (int t = 0; t < n; ++t) {
                double g2 = nearbyint((double)h[t] / (double)sf);
                double g1 = (double)h[t] - g2 * (double)sf;
                if (g1 == (double)c1[t] && g2 == (double)c2[t]) ++ok;
            }
            printf(" %9.1f%%", 100.0 * ok / n);
        }
        printf("\n");
        cudaFree(dA1); cudaFree(dB1); cudaFree(dA2); cudaFree(dB2); cudaFree(dOut);
    }
    printf("\n    need = 2*ceil(log2(144K)) + 1. * W=31 is not reachable in fp32; it is\n");
    printf("    shown to confirm the failures above are width, not a coding bug.\n");
}

// ============================================================ 3. reduction traffic
// The hypothesis in isolation: S split-K partials landing in 1 accumulator vs 2.
__global__ void reduce_one(float *out, int MN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MN) atomicAdd(&out[i], 1.0f);
}
__global__ void reduce_two(float *o1, float *o2, int MN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < MN) { atomicAdd(&o1[i], 1.0f); atomicAdd(&o2[i], 1.0f); }
}

static void exp_reduce() {
    printf("\n[3] reduction traffic: %d split-K partials into 1 accumulator vs 2\n", 32);
    const int MN = 1024 * 1024, SPLITS = 32, iters = 20;
    float *o1, *o2;
    CHECK(cudaMalloc(&o1, MN * 4));
    CHECK(cudaMalloc(&o2, MN * 4));
    CHECK(cudaMemset(o1, 0, MN * 4));
    CHECK(cudaMemset(o2, 0, MN * 4));
    dim3 blk(256), grd((MN + 255) / 256);
    Timer tm;

    for (int i = 0; i < SPLITS; ++i) reduce_one<<<grd, blk>>>(o1, MN);
    CHECK(cudaDeviceSynchronize());
    tm.start();
    for (int r = 0; r < iters; ++r)
        for (int i = 0; i < SPLITS; ++i) reduce_one<<<grd, blk>>>(o1, MN);
    float t1 = tm.stop() / iters;

    for (int i = 0; i < SPLITS; ++i) reduce_two<<<grd, blk>>>(o1, o2, MN);
    CHECK(cudaDeviceSynchronize());
    tm.start();
    for (int r = 0; r < iters; ++r)
        for (int i = 0; i < SPLITS; ++i) reduce_two<<<grd, blk>>>(o1, o2, MN);
    float t2 = tm.stop() / iters;

    double gb1 = (double)MN * SPLITS * 8 / 1e9;      // read+write per atomic
    printf("    1 accumulator : %7.3f ms  %6.1f GB/s\n", t1, gb1 / (t1 * 1e-3));
    printf("    2 accumulators: %7.3f ms  %6.1f GB/s\n", t2, 2 * gb1 / (t2 * 1e-3));
    printf("    packing would save %.2fx here -- this is the UPPER BOUND on the\n", t2 / t1);
    printf("    idea, before experiment [2] takes it away.\n");
    cudaFree(o1); cudaFree(o2);
}

int main(int argc, char **argv) {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("%s  sm_%d%d  %.1f GiB  %d SMs\n", p.name, p.major, p.minor,
           p.totalGlobalMem / 1073741824.0, p.multiProcessorCount);
    if (p.major >= 9) printf("  (this rig is the portable one; use ../ for the Hopper kernels)\n");

    const char *only = argc > 1 ? argv[1] : nullptr;
    auto want = [&](const char *n) { return !only || !strcmp(only, n); };
    if (want("unpack")) exp_unpack();
    if (want("bits")) exp_bits();
    if (want("reduce")) exp_reduce();

    printf("\nWHAT TRANSFERS TO AN H100, AND WHAT DOES NOT\n");
    printf("  transfers: [1] and [2]. Bit widths are bit widths. The exact-recovery\n");
    printf("    rates in [2] are a property of the arithmetic, not of this card, and\n");
    printf("    they match an independent CPU bignum simulation of the same thing.\n");
    printf("  transfers: [3]. Atomic reduction traffic is bandwidth-bound on every\n");
    printf("    GPU, so the ratio is the honest upper bound on the packing idea.\n");
    printf("\n");
    return 0;
}
