// rig -- does packing two FP4 results into one accumulator work? [1]-[5],
// plain CUDA cores, sm_75 and up (no tensor cores, FP8, bf16 or cp.async).
//
//   [1] unpack   FP4 -> FP8 via prmt, exhaustive over 16^4 inputs
//   [2] bits     packed dual dot product on a modelled W-bit accumulator
//   [3] reduce   split-K traffic, 1 accumulator vs 2 (upper bound on packing)
//   [4] gemm     fwd + wgrad: 2 launches / concat / fused 2-acc / packed 1-acc
//   [5] splitk   split-K with packed int16 partials as a transport format
//
//   nvcc -O3 -std=c++17 -arch=sm_75 -o rig rig.cu
//   ./rig [name]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
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
        CHECK(cudaGetLastError());            // a failed launch must not time as zero
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

// Host model of a W-bit significand (W <= 53), round half away from zero.
static double round_sig_d(double x, int W) {
    if (x == 0.0) return 0.0;
    int e;
    double m = frexp(x, &e);
    return ldexp(round(ldexp(m, W)), e - W);
}

static void exp_bits() {
    printf("\n[2] packed dual dot product: acc = C1 + 2^s*C2 on a W-bit accumulator\n");
    printf("    W=24 is fp32, W=14 models the Hopper FP8 MMA path (both on the GPU);\n");
    printf("    W=31 is modelled on the host in fp64, as a control that enough bits suffice\n\n");
    printf("    %5s %4s %8s %10s %10s %10s\n", "K", "s", "need", "W=14", "W=24", "W=31");
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

        auto recovered = [&](const std::vector<double> &h) {
            int ok = 0;
            for (int t = 0; t < n; ++t) {
                double g2 = nearbyint(h[t] / (double)sf);
                double g1 = h[t] - g2 * (double)sf;
                if (g1 == (double)c1[t] && g2 == (double)c2[t]) ++ok;
            }
            return 100.0 * ok / n;
        };
        for (int W : {14, 24}) {
            packed_dot<<<(n + 255) / 256, 256>>>(dA1, dB1, dA2, dB2, dOut, n, K, sf, 24 - W);
            CHECK(cudaDeviceSynchronize());
            std::vector<float> h(n);
            CHECK(cudaMemcpy(h.data(), dOut, n * 4, cudaMemcpyDeviceToHost));
            printf(" %9.1f%%", recovered(std::vector<double>(h.begin(), h.end())));
        }
        std::vector<double> h31(n);
        for (int t = 0; t < n; ++t) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                size_t o = (size_t)t * K + k;
                acc = round_sig_d(acc + (double)a1[o] * b1[o], 31);
                acc = round_sig_d(acc + (double)sf * ((double)a2[o] * b2[o]), 31);
            }
            h31[t] = acc;
        }
        printf(" %9.1f%%\n", recovered(h31));
        cudaFree(dA1); cudaFree(dB1); cudaFree(dA2); cudaFree(dB2); cudaFree(dOut);
    }
    printf("\n    need = 2*ceil(log2(144K)) + 1\n");
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

// ================================================== 4. minimal fwd + wgrad on CUDA cores
#define TS 16

__global__ void gemm_tiled(const float *__restrict__ A, const float *__restrict__ B,
                           float *__restrict__ C, int M, int N, int K) {
    __shared__ float sA[TS][TS], sB[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TS) {
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t + (int)threadIdx.x < K) ? A[(size_t)row * K + t + threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t + (int)threadIdx.y < K && col < N) ? B[(size_t)(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// C[M,N] = A^T @ B, with A stored [K,M]. This is the wgrad shape: dW = dY^T @ X
// with dY stored [batch, Cout]. Both wgrad variants use it, so the strided read
// of A is paid equally and only the op count differs.
__global__ void gemm_tiled_at(const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C, int M, int N, int K) {
    __shared__ float sA[TS][TS], sB[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TS) {
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t + (int)threadIdx.x < K) ? A[(size_t)(t + threadIdx.x) * M + row] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t + (int)threadIdx.y < K && col < N) ? B[(size_t)(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

__global__ void add_into(float *dst, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

// THE CONTROL. Two accumulators, but B is loaded once and reused by both
// microbatches -- exactly like the packed kernel, minus the bit trick. If this
// matches `packed`, then the speedup is operand reuse and the packing adds
// nothing.
__global__ void gemm_fused2(const float *__restrict__ A1, const float *__restrict__ A2,
                            const float *__restrict__ B, float *__restrict__ C1,
                            float *__restrict__ C2, int M, int N, int K) {
    __shared__ float sA1[TS][TS], sA2[TS][TS], sB[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc1 = 0.f, acc2 = 0.f;
    for (int t = 0; t < K; t += TS) {
        bool ra = row < M && t + (int)threadIdx.x < K;
        sA1[threadIdx.y][threadIdx.x] = ra ? A1[(size_t)row * K + t + threadIdx.x] : 0.f;
        sA2[threadIdx.y][threadIdx.x] = ra ? A2[(size_t)row * K + t + threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t + (int)threadIdx.y < K && col < N) ? B[(size_t)(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k) {
            float b = sB[k][threadIdx.x];
            acc1 += sA1[threadIdx.y][k] * b;
            acc2 += sA2[threadIdx.y][k] * b;
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C1[(size_t)row * N + col] = acc1;
        C2[(size_t)row * N + col] = acc2;
    }
}

// Packed: one accumulator carries both microbatches. B is shared between them,
// which is exactly the forward/dgrad case.
__global__ void gemm_packed(const float *__restrict__ A1, const float *__restrict__ A2,
                            const float *__restrict__ B, float *__restrict__ C,
                            int M, int N, int K, float s, int drop) {
    __shared__ float sA1[TS][TS], sA2[TS][TS], sB[TS][TS];
    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TS) {
        bool ra = row < M && t + (int)threadIdx.x < K;
        sA1[threadIdx.y][threadIdx.x] = ra ? A1[(size_t)row * K + t + threadIdx.x] : 0.f;
        sA2[threadIdx.y][threadIdx.x] = ra ? A2[(size_t)row * K + t + threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t + (int)threadIdx.y < K && col < N) ? B[(size_t)(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TS; ++k) {
            float b = sB[k][threadIdx.x];
            acc = round_sig(acc + sA1[threadIdx.y][k] * b, drop);
            acc = round_sig(acc + s * (sA2[threadIdx.y][k] * b), drop);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

static void exp_gemm() {
    const int B = 512, CIN = 1024, COUT = 1024;
    printf("\n[4] minimal fwd + wgrad, CUDA-core GEMMs (B=%d Cin=%d Cout=%d)\n",
           B, CIN, COUT);

    std::uniform_int_distribution<int> cd(0, 14);
    const int codes[15] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    auto fill = [&](std::vector<float> &v) {
        for (auto &x : v) x = (float)e2m1_q(codes[cd(rng)]);
    };
    std::vector<float> hX((size_t)2 * B * CIN), hW((size_t)CIN * COUT), hdY((size_t)2 * B * COUT);
    fill(hX); fill(hW); fill(hdY);

    float *X, *W, *dY, *C1, *C2, *Ccat, *dW;
    CHECK(cudaMalloc(&X, hX.size() * 4));
    CHECK(cudaMalloc(&W, hW.size() * 4));
    CHECK(cudaMalloc(&dY, hdY.size() * 4));
    CHECK(cudaMalloc(&C1, (size_t)B * COUT * 4));
    CHECK(cudaMalloc(&C2, (size_t)B * COUT * 4));
    CHECK(cudaMalloc(&Ccat, (size_t)2 * B * COUT * 4));
    CHECK(cudaMalloc(&dW, (size_t)COUT * CIN * 4));
    CHECK(cudaMemcpy(X, hX.data(), hX.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(W, hW.data(), hW.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dY, hdY.data(), hdY.size() * 4, cudaMemcpyHostToDevice));
    float *X2 = X + (size_t)B * CIN;

    const int s = slot_offset(CIN);
    const float sf = ldexpf(1.f, s);
    dim3 blk(TS, TS);
    Timer tm;
    const int iters = 20;

    // Best-of-5. A desktop card boosts and throttles, and the mean of a single
    // burst moves by 40% between runs; the minimum is far steadier.
    auto time_it = [&](const char *name, auto fn, double flops) {
        fn(); CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            tm.start();
            for (int i = 0; i < iters; ++i) fn();
            float ms = tm.stop() / iters;
            if (ms < best) best = ms;
        }
        printf("    %-30s %8.3f ms  %6.1f GFLOP/s\n", name, best,
               flops / (best * 1e-3) / 1e9);
        return best;
    };

    dim3 g1((COUT + TS - 1) / TS, (B + TS - 1) / TS);
    dim3 gc((COUT + TS - 1) / TS, (2 * B + TS - 1) / TS);
    double fl = 2.0 * (2 * B) * CIN * COUT;

    printf("  forward (both microbatches share W):\n");
    float tv = time_it("vanilla: 2 GEMM launches", [&] {
        gemm_tiled<<<g1, blk>>>(X, W, C1, B, COUT, CIN);
        gemm_tiled<<<g1, blk>>>(X2, W, C2, B, COUT, CIN);
    }, fl);
    float tc = time_it("concat: 1 GEMM, taller M", [&] {
        gemm_tiled<<<gc, blk>>>(X, W, Ccat, 2 * B, COUT, CIN);
    }, fl);
    float tf = time_it("fused: 2 acc, B loaded once", [&] {
        gemm_fused2<<<g1, blk>>>(X, X2, W, C1, C2, B, COUT, CIN);
    }, fl);
    float tp = time_it("packed: 1 acc, B loaded once", [&] {
        gemm_packed<<<g1, blk>>>(X, X2, W, C1, B, COUT, CIN, sf, 0);
    }, fl);
    printf("    vs vanilla: concat %.2fx, fused %.2fx, packed %.2fx\n",
           tv / tc, tv / tf, tv / tp);
    printf("    attribution: operand reuse (fused) %.2fx, packing on top of it %.2fx\n",
           tv / tf, tf / tp);

    // Is the packed forward result actually recoverable? Spot-check exactly.
    std::vector<float> hc(( size_t)B * COUT);
    CHECK(cudaMemcpy(hc.data(), C1, hc.size() * 4, cudaMemcpyDeviceToHost));
    int ok = 0, tried = 0;
    for (int m = 0; m < B; m += 37)
        for (int n = 0; n < COUT; n += 41) {
            long long r1 = 0, r2 = 0;
            for (int k = 0; k < CIN; ++k) {
                r1 += (long long)hX[(size_t)m * CIN + k] * hW[(size_t)k * COUT + n];
                r2 += (long long)hX[(size_t)(B + m) * CIN + k] * hW[(size_t)k * COUT + n];
            }
            double v = hc[(size_t)m * COUT + n];
            double g2 = nearbyint(v / (double)sf), g1v = v - g2 * (double)sf;
            ++tried;
            if (g1v == (double)r1 && g2 == (double)r2) ++ok;
        }
    printf("    packed forward exactly recoverable: %d/%d  (needs %d bits, fp32 has 24)\n",
           ok, tried, packed_bits_needed(CIN));

    printf("  wgrad = dY^T @ X (the two microbatches are SUMMED -- the reduction):\n");
    dim3 gw((CIN + TS - 1) / TS, (COUT + TS - 1) / TS);
    float *T1, *T2;
    CHECK(cudaMalloc(&T1, (size_t)COUT * CIN * 4));
    CHECK(cudaMalloc(&T2, (size_t)COUT * CIN * 4));
    float *dY2 = dY + (size_t)B * COUT;
    size_t wn = (size_t)COUT * CIN;

    float wv = time_it("vanilla: 2 GEMM + add", [&] {
        gemm_tiled_at<<<gw, blk>>>(dY, X, T1, COUT, CIN, B);
        gemm_tiled_at<<<gw, blk>>>(dY2, X2, T2, COUT, CIN, B);
        add_into<<<(unsigned)((wn + 255) / 256), 256>>>(dW, T1, T2, wn);
    }, 2.0 * COUT * CIN * (2 * B));
    float wc = time_it("concat-K: 1 GEMM, 1 accumulator", [&] {
        gemm_tiled_at<<<gw, blk>>>(dY, X, dW, COUT, CIN, 2 * B);
    }, 2.0 * COUT * CIN * (2 * B));
    printf("    concat-K %.2fx vs vanilla (CUDA-core GEMM; the add is relatively larger on\n"
           "    tensor cores, so this ratio does not transfer)\n", wv / wc);

    // Both wgrad paths must agree exactly: same integers, same order per output.
    {
        gemm_tiled_at<<<gw, blk>>>(dY, X, T1, COUT, CIN, B);
        gemm_tiled_at<<<gw, blk>>>(dY2, X2, T2, COUT, CIN, B);
        add_into<<<(unsigned)((wn + 255) / 256), 256>>>(T1, T1, T2, wn);
        gemm_tiled_at<<<gw, blk>>>(dY, X, dW, COUT, CIN, 2 * B);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> hv(wn), hc2(wn);
        CHECK(cudaMemcpy(hv.data(), T1, wn * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hc2.data(), dW, wn * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < wn; ++i) if (hv[i] != hc2[i]) ++diff;
        printf("    vanilla vs concat-K: %zu of %zu elements differ -> %s\n",
               diff, wn, diff ? "MISMATCH" : "identical");
    }

    cudaFree(X); cudaFree(W); cudaFree(dY);
    cudaFree(C1); cudaFree(C2); cudaFree(Ccat); cudaFree(dW);
    cudaFree(T1); cudaFree(T2);
}

// ============================== 5. split-K with packed int16 partials =========
// Packing as a TRANSPORT format: two int16 partials share one int32 word and
// are unpacked before they are summed, so nothing is multiplied or accumulated
// while packed. Partials go to memory for a second kernel instead of atomics.
// Exact while the split depth d <= 227, since |partial| <= 144*d.
#define SK_TS 16
#define SK_TK 16
#define SK_PAD 4

template <int MODE>   // 0 = atomics, 1 = two int32 partials, 2 = packed int16 pair
__global__ __launch_bounds__(SK_TS * SK_TS) void splitk_fused(
        const signed char *__restrict__ A1, const signed char *__restrict__ A2,
        const signed char *__restrict__ Bt, int *__restrict__ out,
        int M, int N, int K, int S) {
    __shared__ __align__(16) signed char sA1[SK_TS][SK_TK + SK_PAD];
    __shared__ __align__(16) signed char sA2[SK_TS][SK_TK + SK_PAD];
    __shared__ __align__(16) signed char sB[SK_TS][SK_TK + SK_PAD];

    const int split = blockIdx.z;
    const int d = K / S;
    const int k0 = split * d;
    const int row = blockIdx.y * SK_TS + threadIdx.y;
    const int col = blockIdx.x * SK_TS + threadIdx.x;
    const int lk = threadIdx.x, lr = threadIdx.y;

    int acc1 = 0, acc2 = 0;
    for (int t = 0; t < d; t += SK_TK) {
        sA1[lr][lk] = A1[(size_t)(blockIdx.y * SK_TS + lr) * K + k0 + t + lk];
        sA2[lr][lk] = A2[(size_t)(blockIdx.y * SK_TS + lr) * K + k0 + t + lk];
        sB[lr][lk] = Bt[(size_t)(blockIdx.x * SK_TS + lr) * K + k0 + t + lk];
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < SK_TK; kk += 4) {
            int a1 = *(const int *)&sA1[threadIdx.y][kk];
            int a2 = *(const int *)&sA2[threadIdx.y][kk];
            int b = *(const int *)&sB[threadIdx.x][kk];
            acc1 = __dp4a(a1, b, acc1);
            acc2 = __dp4a(a2, b, acc2);       // Bt reused by both microbatches
        }
        __syncthreads();
    }

    const size_t MN = (size_t)M * N;
    const size_t i = (size_t)row * N + col;
    if (MODE == 0) {
        atomicAdd(&out[i], acc1);
        atomicAdd(&out[MN + i], acc2);
    } else if (MODE == 1) {
        out[((size_t)split * 2) * MN + i] = acc1;          // 8 bytes per output
        out[((size_t)split * 2 + 1) * MN + i] = acc2;
    } else {
        unsigned p = ((unsigned)(acc2 & 0xFFFF) << 16) | (unsigned)(acc1 & 0xFFFF);
        out[(size_t)split * MN + i] = (int)p;              // 4 bytes per output
    }
}

__global__ void reduce_store2(const int *__restrict__ part, int *c1, int *c2,
                              size_t MN, int S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int a = 0, b = 0;
    for (int s = 0; s < S; ++s) {
        a += part[((size_t)s * 2) * MN + i];
        b += part[((size_t)s * 2 + 1) * MN + i];
    }
    c1[i] = a; c2[i] = b;
}

__global__ void reduce_packed(const int *__restrict__ part, int *c1, int *c2,
                              size_t MN, int S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int a = 0, b = 0;
    for (int s = 0; s < S; ++s) {
        unsigned p = (unsigned)part[(size_t)s * MN + i];
        a += (int)(short)(p & 0xFFFF);        // unpack, then sum in full width
        b += (int)(short)(p >> 16);
    }
    c1[i] = a; c2[i] = b;
}

static void exp_splitk() {
    const int M = 256, N = 256, K = 8192;
    const size_t MN = (size_t)M * N;
    printf("\n[5] split-K with packed int16 partials, M=%d N=%d K=%d, dp4a GEMM\n", M, N, K);

    std::uniform_int_distribution<int> cd(0, 14);
    const int codes[15] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    std::vector<signed char> hA1((size_t)M * K), hA2((size_t)M * K), hBt((size_t)N * K);
    for (auto &v : hA1) v = (signed char)e2m1_q(codes[cd(rng)]);
    for (auto &v : hA2) v = (signed char)e2m1_q(codes[cd(rng)]);
    for (auto &v : hBt) v = (signed char)e2m1_q(codes[cd(rng)]);

    signed char *A1, *A2, *Bt;
    CHECK(cudaMalloc(&A1, hA1.size())); CHECK(cudaMalloc(&A2, hA2.size()));
    CHECK(cudaMalloc(&Bt, hBt.size()));
    CHECK(cudaMemcpy(A1, hA1.data(), hA1.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(A2, hA2.data(), hA2.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Bt, hBt.data(), hBt.size(), cudaMemcpyHostToDevice));

    // exact reference on a sample of outputs
    std::vector<std::pair<int, int>> sample;
    for (int m = 0; m < M; m += 29)
        for (int n = 0; n < N; n += 31) sample.push_back({m, n});
    std::vector<long long> r1(sample.size()), r2(sample.size());
    for (size_t q = 0; q < sample.size(); ++q) {
        long long a = 0, b = 0;
        int m = sample[q].first, n = sample[q].second;
        for (int k = 0; k < K; ++k) {
            a += (long long)hA1[(size_t)m * K + k] * hBt[(size_t)n * K + k];
            b += (long long)hA2[(size_t)m * K + k] * hBt[(size_t)n * K + k];
        }
        r1[q] = a; r2[q] = b;
    }

    int *c1, *c2, *atom, *part;
    CHECK(cudaMalloc(&c1, MN * 4)); CHECK(cudaMalloc(&c2, MN * 4));
    CHECK(cudaMalloc(&atom, MN * 8));
    CHECK(cudaMalloc(&part, MN * 4 * 512));      // room for S=256 unpacked

    dim3 blk(SK_TS, SK_TS);
    Timer tm;
    const int iters = 20;
    auto best_of = [&](auto fn) {
        fn(); CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            tm.start();
            for (int i = 0; i < iters; ++i) fn();
            best = std::min(best, tm.stop() / iters);
        }
        return best;
    };
    auto verdict = [&]() {
        std::vector<int> h1(MN), h2(MN);
        CHECK(cudaMemcpy(h1.data(), c1, MN * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h2.data(), c2, MN * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t q = 0; q < sample.size(); ++q) {
            size_t i = (size_t)sample[q].first * N + sample[q].second;
            if ((long long)h1[i] != r1[q] || (long long)h2[i] != r2[q]) ++bad;
        }
        return bad ? "WRONG" : "exact";
    };

    printf("\n    %4s %5s %7s %10s %10s %10s  %-8s %s\n",
           "S", "depth", "int16?", "gemm ms", "reduce ms", "total ms", "path", "check");

    const double gemm_ops = 2.0 * 2 * M * N * K;
    float packed_ref = 0.f, store2_ref = 0.f, gemm_ref = 0.f;
    double packed_bytes = 0, store2_bytes = 0;
    const unsigned rg = (unsigned)((MN + 255) / 256);

    for (int S : {32, 64, 128, 256}) {
        const int d = K / S;
        if (d % SK_TK) continue;
        dim3 grd(N / SK_TS, M / SK_TS, S);
        const bool safe = (long long)FP4_PMAX * d <= 32767;

        float g1 = best_of([&] { splitk_fused<1><<<grd, blk>>>(A1, A2, Bt, part, M, N, K, S); });
        float rd1 = best_of([&] { reduce_store2<<<rg, 256>>>(part, c1, c2, MN, S); });
        splitk_fused<1><<<grd, blk>>>(A1, A2, Bt, part, M, N, K, S);
        reduce_store2<<<rg, 256>>>(part, c1, c2, MN, S);
        CHECK(cudaDeviceSynchronize());
        printf("    %4d %5d %7s %10.3f %10.3f %10.3f  %-8s %s\n", S, d, safe ? "safe" : "OVERFL",
               g1, rd1, g1 + rd1, "int32x2", verdict());

        float g2 = best_of([&] { splitk_fused<2><<<grd, blk>>>(A1, A2, Bt, part, M, N, K, S); });
        float rd2 = best_of([&] { reduce_packed<<<rg, 256>>>(part, c1, c2, MN, S); });
        splitk_fused<2><<<grd, blk>>>(A1, A2, Bt, part, M, N, K, S);
        reduce_packed<<<rg, 256>>>(part, c1, c2, MN, S);
        CHECK(cudaDeviceSynchronize());
        printf("    %4s %5s %7s %10.3f %10.3f %10.3f  %-8s %s  %.2fx\n", "", "", "", g2, rd2,
               g2 + rd2, "packed", verdict(), (g1 + rd1) / (g2 + rd2));

        float g3 = best_of([&] {
            CHECK(cudaMemsetAsync(atom, 0, MN * 8));
            splitk_fused<0><<<grd, blk>>>(A1, A2, Bt, atom, M, N, K, S);
        });
        printf("    %4s %5s %7s %10.3f %10s %10.3f  atomics\n\n", "", "", "", g3, "-", g3);

        if (S == 64) {
            packed_ref = g2 + rd2;
            store2_ref = g1 + rd1;
            gemm_ref = g2;
            packed_bytes = (double)S * MN * 4 * 2 + MN * 8;    // write + read + out
            store2_bytes = (double)S * MN * 8 * 2 + MN * 8;
        }
    }
    printf("    OVERFL = 144*depth > 32767 (worst case); random data still verifies.\n");
    printf("    Atomics stay in L2 here (%.0f KB output); two-phase paths write %.0f MB.\n",
           MN * 8 / 1024.0, (double)64 * MN * 8 / 1e6);

    if (store2_ref > 0.f) {
        // Peak DRAM bandwidth from the device, achieved GEMM rate from the timing
        // above; H100 figures are achieved-rate assumptions, not measurements.
        int clk_khz = 0, bus_bits = 0;
        CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, 0));
        CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
        const double card_bw = 2.0 * clk_khz * 1e3 * bus_bits / 8.0;
        const double card_ops = gemm_ops / (gemm_ref * 1e-3);
        const double h100_ops = 1400e12, h100_bw = 3350e9;
        const double gemm_h = gemm_ops / h100_ops * 1e3;
        const double red_hu = store2_bytes / h100_bw * 1e3, red_hp = packed_bytes / h100_bw * 1e3;
        printf("\n    roofline at S=64 (partials %.1f MB unpacked -> %.1f MB packed):\n",
               store2_bytes / 1e6, packed_bytes / 1e6);
        printf("      this card: %.3f -> %.3f ms measured, %.2fx; traffic is %.0f%% of runtime\n",
               store2_ref, packed_ref, store2_ref / packed_ref,
               100.0 * (store2_bytes / card_bw * 1e3) / store2_ref);
        printf("      H100 est.: %.4f -> %.4f ms, %.2fx; traffic is %.0f%% of runtime\n",
               gemm_h + red_hu, gemm_h + red_hp, (gemm_h + red_hu) / (gemm_h + red_hp),
               100.0 * red_hu / (gemm_h + red_hu));
        printf("      (H100 does %.0fx more ops per byte than this card: %.0f GOP/s, %.0f GB/s)\n",
               (h100_ops / h100_bw) / (card_ops / card_bw), card_ops / 1e9, card_bw / 1e9);
    }

    cudaFree(A1); cudaFree(A2); cudaFree(Bt);
    cudaFree(c1); cudaFree(c2); cudaFree(atom); cudaFree(part);
}

// Fails fast when this binary has no kernel image for the GPU it runs on.
__global__ void probe() {}

int main(int argc, char **argv) {
    struct { const char *name; void (*fn)(); } exps[] = {
        {"unpack", [] { exp_unpack(); }}, {"bits", exp_bits}, {"reduce", exp_reduce},
        {"gemm", exp_gemm}, {"splitk", exp_splitk}};
    const char *only = argc > 1 ? argv[1] : nullptr;
    bool known = !only;
    for (auto &e : exps) known = known || !strcmp(only, e.name);
    if (!known) {
        fprintf(stderr, "unknown experiment '%s'; one of:", only);
        for (auto &e : exps) fprintf(stderr, " %s", e.name);
        fprintf(stderr, "\n");
        return 1;
    }
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("%s  sm_%d%d  %.1f GiB  %d SMs\n", p.name, p.major, p.minor,
           p.totalGlobalMem / 1073741824.0, p.multiProcessorCount);
    probe<<<1, 1>>>();
    CHECK(cudaGetLastError());
    for (auto &e : exps)
        if (!only || !strcmp(only, e.name)) {
            rng.seed(12345);        // each experiment reproduces standalone
            e.fn();
        }
    printf("\n");
    return 0;
}
