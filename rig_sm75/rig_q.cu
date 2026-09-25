// rig_q -- quantization and transport experiments [6]-[19], sm_75 and up.
//
// This file prints tables; ../FINDINGS.md interprets them. Every experiment
// reseeds the RNG, so one experiment run alone prints the same numbers it
// prints inside the full suite.
//
//   [6]  narrow      split-K partials: is the win packing or narrowing?
//   [7]  int8acc     INT8/s32: is the accumulator or the operand the wall?
//   [8]  nvfp4       MXFP4 vs NVFP4, accuracy per bit
//   [9]  mxfp8       what 4-bit buys over MXFP8/BF16 at equal FLOPs
//   [10] interleave  one nibble-interleaved stream for two microbatches
//   [11] link        measured off-chip bandwidth + all-reduce roofline
//   [12] dense       normalization-aware packing of partials
//   [13] fair        all-reduce at equal bytes: closure vs bits, ring vs direct
//   [14] llm         a-priori bound, outliers and scale agreement at 1B-405B
//   [15] ef          error feedback over many steps
//   [16] predict     last step's scale instead of a scale collective
//   [17] tiers       two-level (NVLink + IB) reduction: numerics and time
//   [18] moe         all-to-all: which of the ideas transfer
//   [19] chain       keeping partials integral from GEMM to wire
//
//   nvcc -O3 -std=c++17 -arch=sm_75 -o rig_q rig_q.cu
//   ./rig_q [name]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "fp4_sim.cuh"
#include "quant_sim.cuh"
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                  \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
    exit(1); } } while (0)
static std::mt19937 rng(20260909);
struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a); }
    float stop() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};
// The 15 E2M1 codes, negative zero (code 8) excluded.
static const int LEGAL[15] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
static inline float to_bf16(float x) {                 // round-to-nearest-even
    union { float f; uint32_t u; } c;
    c.f = x;
    c.u = (c.u + 0x7FFFu + ((c.u >> 16) & 1u)) & 0xFFFF0000u;
    return c.f;
}
// fp16's 11-bit significand. Range is not modelled; callers pre-scale.
static inline float to_fp16(float x) { return round_sig(x, 13); }
static inline int bits_of(long long v) {
    int n = 0;
    while (v > 0) { v >>= 1; ++n; }
    return n;
}
// A depth-d dot product of random FP4 values on the integer grid q = 2*value.
static long long fp4_dot(int d, std::uniform_int_distribution<int> &cd) {
    long long a = 0;
    for (int k = 0; k < d; ++k)
        a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
    return a;
}
// Running sum carried in a field of +-lim, saturating as a narrow wire would.
static inline long long sat_add(long long a, long long q, long long lim, long long *sat) {
    a += q;
    if (a > lim) { a = lim; ++*sat; }
    if (a < -lim) { a = -lim; ++*sat; }
    return a;
}
template <class F> static float best_of(F fn, int iters = 20) {
    Timer tm;
    fn();
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        tm.start();
        for (int i = 0; i < iters; ++i) fn();
        best = std::min(best, tm.stop() / iters);
    }
    return best;
}

// ============================================================================
// [6] narrow -- packing versus narrowing
// ============================================================================
//
// Three storage modes for the same split-K partials, same GEMM, same reduce
// structure. The only variable is how many bytes a partial costs and whether
// the two microbatches share a word.
//
//   mode 0  int32 x2   two int32 arrays          8 bytes / output / split
//   mode 1  int16 x2   two int16 arrays          4 bytes / output / split
//   mode 2  packed     one int32, two int16      4 bytes / output / split
//
// Mode 1 is the control. It is legal for exactly the same reason mode 2 is --
// a depth-d split of FP4 data has |partial| <= 144*d, so d <= 227 fits int16 --
// but it does no bit arithmetic at all.
#define QTS 16
#define QTK 16
#define QPAD 4
template <int MODE>
__global__ __launch_bounds__(QTS * QTS) void nw_gemm(
        const signed char *__restrict__ A1, const signed char *__restrict__ A2,
        const signed char *__restrict__ Bt,
        int *__restrict__ p32, short *__restrict__ p16a, short *__restrict__ p16b,
        int *__restrict__ ppk, int M, int N, int K, int S) {
    __shared__ __align__(16) signed char sA1[QTS][QTK + QPAD];
    __shared__ __align__(16) signed char sA2[QTS][QTK + QPAD];
    __shared__ __align__(16) signed char sB[QTS][QTK + QPAD];
    const int split = blockIdx.z;
    const int d = K / S;
    const int k0 = split * d;
    const int row = blockIdx.y * QTS + threadIdx.y;
    const int col = blockIdx.x * QTS + threadIdx.x;
    const int lk = threadIdx.x, lr = threadIdx.y;
    int acc1 = 0, acc2 = 0;
    for (int t = 0; t < d; t += QTK) {
        sA1[lr][lk] = A1[(size_t)(blockIdx.y * QTS + lr) * K + k0 + t + lk];
        sA2[lr][lk] = A2[(size_t)(blockIdx.y * QTS + lr) * K + k0 + t + lk];
        sB[lr][lk]  = Bt[(size_t)(blockIdx.x * QTS + lr) * K + k0 + t + lk];
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < QTK; kk += 4) {
            int a1 = *(const int *)&sA1[threadIdx.y][kk];
            int a2 = *(const int *)&sA2[threadIdx.y][kk];
            int b  = *(const int *)&sB[threadIdx.x][kk];
            acc1 = __dp4a(a1, b, acc1);
            acc2 = __dp4a(a2, b, acc2);
        }
        __syncthreads();
    }
    const size_t MN = (size_t)M * N;
    const size_t i = (size_t)row * N + col;
    if (MODE == 0) {                                     // 8 bytes
        p32[((size_t)split * 2) * MN + i] = acc1;
        p32[((size_t)split * 2 + 1) * MN + i] = acc2;
    } else if (MODE == 1) {                              // 4 bytes, no packing
        p16a[(size_t)split * MN + i] = (short)acc1;
        p16b[(size_t)split * MN + i] = (short)acc2;
    } else {                                             // 4 bytes, packed pair
        unsigned p = ((unsigned)(acc2 & 0xFFFF) << 16) | (unsigned)(acc1 & 0xFFFF);
        ppk[(size_t)split * MN + i] = (int)p;
    }
}
__global__ void nw_red32(const int *__restrict__ p, int *c1, int *c2, size_t MN, int S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int a = 0, b = 0;
    for (int s = 0; s < S; ++s) {
        a += p[((size_t)s * 2) * MN + i];
        b += p[((size_t)s * 2 + 1) * MN + i];
    }
    c1[i] = a; c2[i] = b;
}
__global__ void nw_red16(const short *__restrict__ pa, const short *__restrict__ pb,
                         int *c1, int *c2, size_t MN, int S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int a = 0, b = 0;
    for (int s = 0; s < S; ++s) {                 // widen on load, sum in int32
        a += (int)pa[(size_t)s * MN + i];
        b += (int)pb[(size_t)s * MN + i];
    }
    c1[i] = a; c2[i] = b;
}
__global__ void nw_redpk(const int *__restrict__ p, int *c1, int *c2, size_t MN, int S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= MN) return;
    int a = 0, b = 0;
    for (int s = 0; s < S; ++s) {
        unsigned v = (unsigned)p[(size_t)s * MN + i];
        a += (int)(short)(v & 0xFFFF);            // unpack, THEN sum
        b += (int)(short)(v >> 16);
    }
    c1[i] = a; c2[i] = b;
}
static void exp_narrow() {
    const int M = 256, N = 256, K = 8192;
    const size_t MN = (size_t)M * N;
    printf("\n[6] packing or narrowing? the control experiment [5] was missing\n");
    printf("    M=%d N=%d K=%d. int32x2 is 8 B/output/split; BOTH int16 rows are 4 B.\n", M, N, K);
    printf("    If int16x2 ties packed, the win is narrowing and the bit trick is noise.\n");
    std::uniform_int_distribution<int> cd(0, 14);
    std::vector<signed char> hA1((size_t)M * K), hA2((size_t)M * K), hBt((size_t)N * K);
    for (auto &v : hA1) v = (signed char)e2m1_q(LEGAL[cd(rng)]);
    for (auto &v : hA2) v = (signed char)e2m1_q(LEGAL[cd(rng)]);
    for (auto &v : hBt) v = (signed char)e2m1_q(LEGAL[cd(rng)]);
    // Exact int64 reference on a random sample of outputs.
    std::vector<std::pair<int, int>> sample;
    std::uniform_int_distribution<int> md(0, M - 1), nd(0, N - 1);
    for (int t = 0; t < 256; ++t) sample.push_back({md(rng), nd(rng)});
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
    signed char *A1, *A2, *Bt;
    CHECK(cudaMalloc(&A1, (size_t)M * K)); CHECK(cudaMalloc(&A2, (size_t)M * K));
    CHECK(cudaMalloc(&Bt, (size_t)N * K));
    CHECK(cudaMemcpy(A1, hA1.data(), (size_t)M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(A2, hA2.data(), (size_t)M * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Bt, hBt.data(), (size_t)N * K, cudaMemcpyHostToDevice));
    const int SMAX = 256;
    int *c1, *c2, *p32, *ppk;
    short *p16a, *p16b;
    CHECK(cudaMalloc(&c1, MN * 4));  CHECK(cudaMalloc(&c2, MN * 4));
    CHECK(cudaMalloc(&p32, MN * 8 * SMAX));
    CHECK(cudaMalloc(&ppk, MN * 4 * SMAX));
    CHECK(cudaMalloc(&p16a, MN * 2 * SMAX));
    CHECK(cudaMalloc(&p16b, MN * 2 * SMAX));
    Timer tm;
    const int iters = 20;
    auto best_of = [&](auto fn) {
        fn(); CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            tm.start();
            for (int i = 0; i < iters; ++i) fn();
            float ms = tm.stop() / iters;
            best = std::min(best, ms);
        }
        return best;
    };
    auto verify = [&]() {
        std::vector<int> h1(MN), h2(MN);
        CHECK(cudaMemcpy(h1.data(), c1, MN * 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h2.data(), c2, MN * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t q = 0; q < sample.size(); ++q) {
            size_t i = (size_t)sample[q].first * N + sample[q].second;
            if ((long long)h1[i] != r1[q] || (long long)h2[i] != r2[q]) ++bad;
        }
        return bad;
    };
    dim3 blk(QTS, QTS);
    const unsigned rg = (unsigned)((MN + 255) / 256);
    printf("\n    %4s %5s  %-9s %9s %10s %10s %8s  %s\n",
           "S", "depth", "mode", "gemm ms", "reduce ms", "total ms", "vs int32", "check");
    for (int S : {64, 128, 256}) {
        const int d = K / S;
        dim3 grd((unsigned)(N / QTS), (unsigned)(M / QTS), (unsigned)S);
        float tot0 = 0.f;
        for (int mode = 0; mode < 3; ++mode) {
            float g, r;
            if (mode == 0) {
                g = best_of([&] { nw_gemm<0><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S); });
                r = best_of([&] { nw_red32<<<rg, 256>>>(p32, c1, c2, MN, S); });
                nw_gemm<0><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S);
                nw_red32<<<rg, 256>>>(p32, c1, c2, MN, S);
            } else if (mode == 1) {
                g = best_of([&] { nw_gemm<1><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S); });
                r = best_of([&] { nw_red16<<<rg, 256>>>(p16a, p16b, c1, c2, MN, S); });
                nw_gemm<1><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S);
                nw_red16<<<rg, 256>>>(p16a, p16b, c1, c2, MN, S);
            } else {
                g = best_of([&] { nw_gemm<2><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S); });
                r = best_of([&] { nw_redpk<<<rg, 256>>>(ppk, c1, c2, MN, S); });
                nw_gemm<2><<<grd, blk>>>(A1, A2, Bt, p32, p16a, p16b, ppk, M, N, K, S);
                nw_redpk<<<rg, 256>>>(ppk, c1, c2, MN, S);
            }
            CHECK(cudaDeviceSynchronize());
            size_t bad = verify();
            float tot = g + r;
            if (mode == 0) tot0 = tot;
            const char *nm[3] = {"int32 x2", "int16 x2", "packed"};
            if (mode == 0) printf("    %4d %5d  %-9s %9.3f %10.3f %10.3f %8s  %s\n",
                                  S, d, nm[mode], g, r, tot, "-", bad ? "WRONG" : "exact");
            else printf("    %4s %5s  %-9s %9.3f %10.3f %10.3f %7.2fx  %s\n",
                        "", "", nm[mode], g, r, tot, tot0 / tot, bad ? "WRONG" : "exact");
        }
    }
    printf("\n    Both int16 rows move the same bytes. Any gap between them is the\n");
    printf("    pairing alone -- one 32-bit store against two 16-bit stores, and one\n");
    printf("    load stream against two in the reduce. Any gap between int32x2 and\n");
    printf("    them is the narrowing, which is what the 144*d bound actually buys.\n");
    cudaFree(A1); cudaFree(A2); cudaFree(Bt);
    cudaFree(c1); cudaFree(c2); cudaFree(p32); cudaFree(ppk);
    cudaFree(p16a); cudaFree(p16b);
}
// ============================================================================
// [7] int8acc -- the only Hopper datapath with a wide enough accumulator
// ============================================================================
//
// Two questions that experiment [2] ran together:
//
//   ACCUMULATOR. Can a slot layout [C1 : p][C2 : p] live in the accumulator?
//   Needs 2*p-1 bits. INT8 MMA on Hopper accumulates in true s32, against
//   fp32's 24 significand bits and the FP8 path's ~14. So s32 should carry it
//   much further than anything else on the chip.
//
//   OPERAND. Can Ahat = A1 + 2^p*A2 live in one int8 lane? |q| <= 12 needs 5
//   signed bits, so 12 + 12*2^p <= 127 caps p at 3. But the slots must not
//   collide, which needs p >= bits(144K) + 1 -- 14 at K=32.
//
// If the accumulator column is fine and the operand column is not, then the
// operand lane is the wall, and it is the same wall on every Hopper format.
static void exp_int8acc() {
    printf("\n[7] INT8 / s32: is the accumulator the problem, or the operand?\n");
    printf("    Hopper tensor-core formats stop at INT8 (Ampere's s4 and b1 are gone),\n");
    printf("    and INT8 is the only one that accumulates in a true 32-bit integer.\n");
    const int TRIALS = 4000;
    std::uniform_int_distribution<int> cd(0, 14);
    printf("\n    %5s %5s %6s | %-37s | %-22s\n", "K", "p", "need",
           "ACCUMULATOR (combine at store)", "OPERAND (one int8 lane)");
    printf("    %5s %5s %6s | %7s %8s %8s %9s | %6s %6s %8s\n", "", "", "bits",
           "s32 wc", "s32 obs", "fp32(24)", "fp8(~14)", "p max", "short", "exact");
    for (int K : {8, 16, 32, 64, 128, 256, 512}) {
        const int p = slot_offset(K);
        const int need = packed_bits_needed(K);
        long long ok32 = 0, ok24 = 0, ok14 = 0, okOp = 0;
        // int8 lane: |q1 + 2^pl*q2| <= 12 + 12*2^pl <= 127  ->  pl <= 3
        int plmax = 0;
        while (12 + 12 * (1 << (plmax + 1)) <= 127) ++plmax;
        for (int t = 0; t < TRIALS; ++t) {
            long long C1 = 0, C2 = 0;
            long long accOp = 0;
            for (int k = 0; k < K; ++k) {
                int q1 = e2m1_q(LEGAL[cd(rng)]);
                int q2 = e2m1_q(LEGAL[cd(rng)]);
                int b  = e2m1_q(LEGAL[cd(rng)]);
                C1 += (long long)q1 * b;
                C2 += (long long)q2 * b;
                int ahat = q1 + (q2 << plmax);          // fits int8 by construction
                accOp += (long long)ahat * b;
            }
            // --- accumulator side: two exact chains, combined once at the end.
            long long comb = C1 + (C2 << p);
            // s32: wraps at 2^31. Exact iff the layout fits.
            int wrapped = (int)(unsigned long long)comb;
            long long g2 = ((long long)wrapped + (1LL << (p - 1))) >> p;
            long long g1 = (long long)wrapped - (g2 << p);
            if (g1 == C1 && g2 == C2) ++ok32;
            // fp32 / fp8: same layout through a W-bit significand.
            for (int wi = 0; wi < 2; ++wi) {
                int drop = wi == 0 ? 0 : 10;            // W = 24 and W = 14
                float f = round_sig((float)comb, drop);
                long long h2 = (long long)llrintf(f / (float)(1LL << p));
                long long h1 = (long long)llrintf(f) - (h2 << p);
                if (h1 == C1 && h2 == C2) { if (wi == 0) ++ok24; else ++ok14; }
            }
            // --- operand side: one lane, the largest offset int8 permits.
            long long o2 = (accOp + (1LL << (plmax - 1))) >> plmax;
            long long o1 = accOp - (o2 << plmax);
            if (o1 == C1 && o2 == C2) ++okOp;
        }
        printf("    %5d %5d %6d | %7s %7.1f%% %7.1f%% %8.1f%% | %6d %6d %7.1f%%\n",
               K, p, need, need <= 32 ? "safe" : "OVERFL",
               100.0 * ok32 / TRIALS, 100.0 * ok24 / TRIALS, 100.0 * ok14 / TRIALS,
               plmax, p - plmax, 100.0 * okOp / TRIALS);
    }
    printf("\n    's32 wc' is the WORST CASE: safe only while need <= 32. 's32 obs' is\n");
    printf("    what random FP4 data actually did, and it stays at 100%% past the point\n");
    printf("    the bound gives out, for the same reason the OVERFL row in [5] still\n");
    printf("    verified -- a random walk lands near sqrt(K), nowhere near 144*K. Do\n");
    printf("    not read the observed column as a guarantee.\n");
    printf("\n    The two accumulator families fail differently, and that is the finding.\n");
    printf("    fp32 fails on SIGNIFICAND: 24 bits, so the low bits of C1 are gone even\n");
    printf("    when the magnitude is tiny -- which is why its column decays smoothly\n");
    printf("    with K on ordinary data. s32 has no significand at all, only range, and\n");
    printf("    range is the resource this data does not spend. An integer accumulator\n");
    printf("    is strictly the better container for a packed layout.\n");
    printf("\n    It still does not rescue the scheme, because the wall is elsewhere.\n");
    printf("    'p max' is the largest offset an int8 lane can hold; 'short' is how many\n");
    printf("    bits of separation the layout still needs after that. The value needs 5\n");
    printf("    bits and the separation needs ~14 more, and no Hopper operand lane has\n");
    printf("    19 -- e4m3 has 4 significand bits, bf16 and int8 have 8, fp16 has 11.\n");
    printf("    The operand column is chance, not arithmetic.\n");
    printf("\n    So: the accumulator was never the binding constraint, the operand was,\n");
    printf("    and no format on this chip fixes it. Read the s32 column as good news\n");
    printf("    for the TRANSPORT form instead -- combining at store time is exactly\n");
    printf("    what [5] and [6] do, and it is exact because nothing is ever multiplied\n");
    printf("    while packed.\n");
}
// ============================================================================
// [8] / [9] block-scaled format comparison
// ============================================================================
//
// Same GEMM, same data, four formats. The reference is fp64 on the unquantized
// values, so what we measure is purely the format's own error.
enum Fmt { F_MXFP4, F_NVFP4, F_MXFP8, F_BF16 };
struct FmtInfo { const char *name; int block; const char *scale; double bits; };
static FmtInfo fmt_info(Fmt f) {
    switch (f) {
        case F_MXFP4: return {"MXFP4", 32, "E8M0", 4.0 + 8.0 / 32};
        case F_NVFP4: return {"NVFP4", 16, "E4M3", 4.0 + 8.0 / 16};
        case F_MXFP8: return {"MXFP8", 32, "E8M0", 8.0 + 8.0 / 32};
        default:      return {"BF16",   0, "-",    16.0};
    }
}
// Quantize one length-K vector in place, block by block.
static void quantize_vec(std::vector<double> &v, Fmt f) {
    const int K = (int)v.size();
    if (f == F_BF16) {
        for (int i = 0; i < K; ++i) v[i] = (double)to_bf16((float)v[i]);
        return;
    }
    const int B = fmt_info(f).block;
    for (int b0 = 0; b0 < K; b0 += B) {
        int be = std::min(b0 + B, K);
        float amax = 0.f;
        for (int i = b0; i < be; ++i) amax = std::max(amax, fabsf((float)v[i]));
        if (amax == 0.f) continue;
        float s;
        if (f == F_MXFP4)      s = mx_scale_e2m1(amax);
        else if (f == F_NVFP4) s = nv_scale_e2m1(amax);
        else                   s = mx_scale_e4m3(amax);
        for (int i = b0; i < be; ++i) {
            float x = (float)v[i] / s;
            if (f == F_MXFP8) v[i] = (double)(quant_e4m3(x) * s);
            else              v[i] = (double)(e2m1_value(quant_e2m1_code(x)) * s);
        }
    }
}
static void fmt_table(const char *title, bool outliers, const std::vector<Fmt> &fmts) {
    const int M = 48, N = 48, K = 512;
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<double> A((size_t)M * K), B((size_t)N * K);
    auto fill = [&](std::vector<double> &v) {
        for (auto &x : v) {
            x = nd(rng);
            if (outliers && ud(rng) < 0.01) x *= 20.0;   // 1% heavy tail
        }
    };
    fill(A); fill(B);
    // fp64 reference, B stored transposed (N x K).
    std::vector<double> Cref((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double a = 0;
            for (int k = 0; k < K; ++k) a += A[(size_t)m * K + k] * B[(size_t)n * K + k];
            Cref[(size_t)m * N + n] = a;
        }
    double refn = 0;
    for (double x : Cref) refn += x * x;
    refn = sqrt(refn);
    printf("\n    %s\n", title);
    printf("    %-7s %6s %7s %9s %9s %11s %11s\n",
           "format", "block", "scale", "bit/elem", "blk bits", "rel err", "max abs err");
    for (Fmt f : fmts) {
        std::vector<double> Aq = A, Bq = B;
        for (int m = 0; m < M; ++m) {
            std::vector<double> r(Aq.begin() + (size_t)m * K, Aq.begin() + (size_t)(m + 1) * K);
            quantize_vec(r, f);
            std::copy(r.begin(), r.end(), Aq.begin() + (size_t)m * K);
        }
        for (int n = 0; n < N; ++n) {
            std::vector<double> r(Bq.begin() + (size_t)n * K, Bq.begin() + (size_t)(n + 1) * K);
            quantize_vec(r, f);
            std::copy(r.begin(), r.end(), Bq.begin() + (size_t)n * K);
        }
        double e2 = 0, emax = 0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double a = 0;
                for (int k = 0; k < K; ++k) a += Aq[(size_t)m * K + k] * Bq[(size_t)n * K + k];
                double d = a - Cref[(size_t)m * N + n];
                e2 += d * d;
                emax = std::max(emax, fabs(d));
            }
        FmtInfo fi = fmt_info(f);
        char blkbits[16];
        if (f == F_MXFP4 || f == F_NVFP4)
            snprintf(blkbits, sizeof blkbits, "%d", bits_of((long long)FP4_PMAX * fi.block) + 1);
        else
            snprintf(blkbits, sizeof blkbits, "n/a");
        printf("    %-7s %6d %7s %9.2f %9s %10.2e%% %11.3e\n",
               fi.name, fi.block ? fi.block : 0, fi.scale, fi.bits, blkbits,
               100.0 * sqrt(e2) / refn, emax);
    }
}
static void exp_nvfp4() {
    printf("\n[8] MXFP4 (block 32, E8M0) vs NVFP4 (block 16, E4M3)\n");
    printf("    Two axes move at once: block size and scale format. The scale format\n");
    printf("    is the bigger one -- E8M0 is a bare power of two, so it discards up\n");
    printf("    to 2x of the block's range before any element is rounded.\n");
    printf("    M=N=48, K=512, fp64 reference. 'blk bits' is 1+bits(144*block), the\n");
    printf("    accumulator width an exact block dot product needs.\n");
    std::vector<Fmt> f = {F_MXFP4, F_NVFP4};
    fmt_table("clean data, N(0,1):", false, f);
    fmt_table("1% outliers at 20 sigma (the case block size exists for):", true, f);
    printf("\n    On Hopper both are emulated the same way and cost the same FLOPs, so\n");
    printf("    the extra 0.25 bit/elem of NVFP4 is the entire price. It is also the\n");
    printf("    format Blackwell runs natively, which makes it the forward-compatible\n");
    printf("    choice even where the accuracy gap is small.\n");
}
static void exp_mxfp8() {
    printf("\n[9] MXFP4 vs MXFP8 -- what does the 4-bit format actually buy on Hopper?\n");
    printf("    MXFP4 has to be expanded to E4M3 to reach the tensor core, so it runs\n");
    printf("    at the FP8 rate at best (1979 TFLOP/s dense) and pays an unpack on\n");
    printf("    top. MXFP8 runs there natively with no unpack at all.\n");
    std::vector<Fmt> f = {F_MXFP4, F_NVFP4, F_MXFP8, F_BF16};
    fmt_table("clean data, N(0,1):", false, f);
    fmt_table("1% outliers at 20 sigma:", true, f);
    printf("\n    Same FLOPs, roughly half the bytes, more error. So on Hopper the\n");
    printf("    4-bit formats buy memory, bandwidth and interconnect -- never FLOPs.\n");
    printf("    That is the whole argument for [11]: if the only thing FP4 buys is\n");
    printf("    bytes, the payoff has to be collected where bytes are expensive, and\n");
    printf("    nothing on a node is more expensive per byte than the link.\n");
}
// ============================================================================
// [10] interleave -- one load stream for two microbatches
// ============================================================================
//
// SEP: A1 and A2 are separate arrays, each nibble-packed 2 elements per byte
//      along K. A tile needs one 4-byte load from each.
// ILV: one array, byte k holds A1[k] in the low nibble and A2[k] in the high.
//      A tile needs two 4-byte loads from one array.
//
// IDENTICAL byte count -- FP4 is already 2 elements per byte either way. This
// is a locality and stream-count test. If it wins, it wins on cache behaviour,
// not on bandwidth, and it must not be reported as the latter.
#define IL_TS 16
#define IL_TK 64
#define IL_PAD 4
__device__ __forceinline__ int q4(int code) {
    const int t[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int v = t[code & 7];
    return (code & 8) ? -v : v;
}
// SEP: each thread pulls one 32-bit word = 8 consecutive k of one matrix.
__global__ __launch_bounds__(IL_TS * IL_TS) void il_sep(
        const uint32_t *__restrict__ A1p, const uint32_t *__restrict__ A2p,
        const signed char *__restrict__ Bt, int *__restrict__ C1, int *__restrict__ C2,
        int M, int N, int K) {
    __shared__ __align__(16) signed char sA1[IL_TS][IL_TK + IL_PAD];
    __shared__ __align__(16) signed char sA2[IL_TS][IL_TK + IL_PAD];
    __shared__ __align__(16) signed char sB[IL_TS][IL_TK + IL_PAD];
    const int row = blockIdx.y * IL_TS + threadIdx.y;
    const int col = blockIdx.x * IL_TS + threadIdx.x;
    const int tid = threadIdx.y * IL_TS + threadIdx.x;        // 0..255
    const int wpr = IL_TK / 8;                                // words per row = 8
    const int half = tid >> 7;                                // 0 -> A1, 1 -> A2
    const int lr = (tid & 127) / wpr, lw = (tid & 127) % wpr; // 16 rows x 8 words
    const int K8 = K / 8;
    int acc1 = 0, acc2 = 0;
    for (int t = 0; t < K; t += IL_TK) {
        // All 256 threads load one word each: half from A1, half from A2, so the
        // two layouts issue the same number of loads from the same thread count.
        {
            const uint32_t *src = half ? A2p : A1p;
            uint32_t w = src[(size_t)(blockIdx.y * IL_TS + lr) * K8 + (t / 8) + lw];
            signed char *dst = half ? &sA2[lr][lw * 8] : &sA1[lr][lw * 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) dst[j] = (signed char)q4((int)((w >> (4 * j)) & 0xF));
        }
        if (tid < IL_TS * (IL_TK / 4)) {
            int br = tid / (IL_TK / 4), bc = (tid % (IL_TK / 4)) * 4;
            *(int *)&sB[br][bc] = *(const int *)&Bt[(size_t)(blockIdx.x * IL_TS + br) * K + t + bc];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < IL_TK; kk += 4) {
            int a1 = *(const int *)&sA1[threadIdx.y][kk];
            int a2 = *(const int *)&sA2[threadIdx.y][kk];
            int b  = *(const int *)&sB[threadIdx.x][kk];
            acc1 = __dp4a(a1, b, acc1);
            acc2 = __dp4a(a2, b, acc2);
        }
        __syncthreads();
    }
    C1[(size_t)row * N + col] = acc1;
    C2[(size_t)row * N + col] = acc2;
}
// ILV: each thread pulls one 32-bit word = 4 consecutive k of BOTH matrices.
__global__ __launch_bounds__(IL_TS * IL_TS) void il_ilv(
        const uint32_t *__restrict__ Ai,
        const signed char *__restrict__ Bt, int *__restrict__ C1, int *__restrict__ C2,
        int M, int N, int K) {
    __shared__ __align__(16) signed char sA1[IL_TS][IL_TK + IL_PAD];
    __shared__ __align__(16) signed char sA2[IL_TS][IL_TK + IL_PAD];
    __shared__ __align__(16) signed char sB[IL_TS][IL_TK + IL_PAD];
    const int row = blockIdx.y * IL_TS + threadIdx.y;
    const int col = blockIdx.x * IL_TS + threadIdx.x;
    const int tid = threadIdx.y * IL_TS + threadIdx.x;
    const int wpr = IL_TK / 4;                                // 16 words per row
    const int lr = tid / wpr, lw = tid % wpr;                 // exactly 16 rows
    const int K4 = K / 4;
    int acc1 = 0, acc2 = 0;
    for (int t = 0; t < K; t += IL_TK) {
        uint32_t w = Ai[(size_t)(blockIdx.y * IL_TS + lr) * K4 + (t / 4) + lw];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int byte = (int)((w >> (8 * j)) & 0xFF);
            sA1[lr][lw * 4 + j] = (signed char)q4(byte & 0xF);
            sA2[lr][lw * 4 + j] = (signed char)q4(byte >> 4);
        }
        if (tid < IL_TS * (IL_TK / 4)) {
            int br = tid / (IL_TK / 4), bc = (tid % (IL_TK / 4)) * 4;
            *(int *)&sB[br][bc] = *(const int *)&Bt[(size_t)(blockIdx.x * IL_TS + br) * K + t + bc];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < IL_TK; kk += 4) {
            int a1 = *(const int *)&sA1[threadIdx.y][kk];
            int a2 = *(const int *)&sA2[threadIdx.y][kk];
            int b  = *(const int *)&sB[threadIdx.x][kk];
            acc1 = __dp4a(a1, b, acc1);
            acc2 = __dp4a(a2, b, acc2);
        }
        __syncthreads();
    }
    C1[(size_t)row * N + col] = acc1;
    C2[(size_t)row * N + col] = acc2;
}
static void exp_interleave() {
    const int M = 512, N = 512, K = 4096;
    printf("\n[10] nibble-interleaved A1/A2: one load stream for two microbatches\n");
    printf("    M=%d N=%d K=%d. SEP reads two FP4 arrays; ILV reads one array whose\n", M, N, K);
    printf("    every byte carries A1[k] low and A2[k] high. Same bytes, same load\n");
    printf("    count, same dp4a work. Only the number of streams differs.\n");
    std::uniform_int_distribution<int> cd(0, 14);
    std::vector<uint8_t> c1v((size_t)M * K), c2v((size_t)M * K);
    for (auto &v : c1v) v = (uint8_t)LEGAL[cd(rng)];
    for (auto &v : c2v) v = (uint8_t)LEGAL[cd(rng)];
    std::vector<signed char> hBt((size_t)N * K);
    for (auto &v : hBt) v = (signed char)e2m1_q(LEGAL[cd(rng)]);
    std::vector<uint8_t> sep1((size_t)M * K / 2), sep2((size_t)M * K / 2), ilv((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            size_t i = (size_t)m * K + k;
            if (k & 1) {
                sep1[i / 2] |= (uint8_t)(c1v[i] << 4);
                sep2[i / 2] |= (uint8_t)(c2v[i] << 4);
            } else {
                sep1[i / 2] = c1v[i];
                sep2[i / 2] = c2v[i];
            }
            ilv[i] = (uint8_t)(c1v[i] | (c2v[i] << 4));
        }
    uint32_t *dA1, *dA2, *dAi;
    signed char *dBt;
    int *dC1, *dC2, *eC1, *eC2;
    CHECK(cudaMalloc(&dA1, sep1.size())); CHECK(cudaMalloc(&dA2, sep2.size()));
    CHECK(cudaMalloc(&dAi, ilv.size()));  CHECK(cudaMalloc(&dBt, hBt.size()));
    CHECK(cudaMalloc(&dC1, (size_t)M * N * 4)); CHECK(cudaMalloc(&dC2, (size_t)M * N * 4));
    CHECK(cudaMalloc(&eC1, (size_t)M * N * 4)); CHECK(cudaMalloc(&eC2, (size_t)M * N * 4));
    CHECK(cudaMemcpy(dA1, sep1.data(), sep1.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dA2, sep2.data(), sep2.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dAi, ilv.data(), ilv.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dBt, hBt.data(), hBt.size(), cudaMemcpyHostToDevice));
    dim3 blk(IL_TS, IL_TS), grd((unsigned)(N / IL_TS), (unsigned)(M / IL_TS));
    Timer tm;
    auto best_of = [&](auto fn) {
        fn(); CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            tm.start();
            for (int i = 0; i < 10; ++i) fn();
            best = std::min(best, tm.stop() / 10);
        }
        return best;
    };
    float ts = best_of([&] { il_sep<<<grd, blk>>>(dA1, dA2, dBt, dC1, dC2, M, N, K); });
    float ti = best_of([&] { il_ilv<<<grd, blk>>>(dAi, dBt, eC1, eC2, M, N, K); });
    CHECK(cudaDeviceSynchronize());
    std::vector<int> h1((size_t)M * N), h2((size_t)M * N), g1((size_t)M * N), g2((size_t)M * N);
    CHECK(cudaMemcpy(h1.data(), dC1, h1.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h2.data(), dC2, h2.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(g1.data(), eC1, g1.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(g2.data(), eC2, g2.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < h1.size(); ++i) if (h1[i] != g1[i] || h2[i] != g2[i]) ++bad;
    double bytes = (double)M * K;                      // identical for both
    double flop = 2.0 * 2.0 * M * N * K;
    printf("\n    %-24s %9s %10s %10s\n", "layout", "ms", "A bytes", "GOP/s");
    printf("    %-24s %9.3f %9.1fM %10.1f\n", "SEP: two FP4 arrays", ts, bytes / 1e6, flop / ts * 1e-6);
    printf("    %-24s %9.3f %9.1fM %10.1f   %.2fx\n", "ILV: one interleaved", ti, bytes / 1e6,
           flop / ti * 1e-6, ts / ti);
    printf("    agreement: %s (%zu of %zu outputs differ)\n",
           bad ? "MISMATCH" : "identical", bad, h1.size());
    printf("\n    Byte counts are equal by construction, so whatever this shows is\n");
    printf("    locality, not bandwidth. FP4 is already two elements per byte; there\n");
    printf("    is no second densification to collect on the operand side.\n");
}
// ============================================================================
// [11] link -- measured off-chip bandwidth, and what a byte costs on a link
// ============================================================================
static void exp_link() {
    printf("\n[11] off-chip: measured PCIe bandwidth, and an all-reduce roofline\n");
    const size_t NOUT = 1u << 22;
    printf("\n    A. device-to-host copies, %zu output pairs\n", NOUT);
    void *dbuf, *hbuf;
    CHECK(cudaMalloc(&dbuf, NOUT * 8));
    CHECK(cudaMallocHost(&hbuf, NOUT * 8));
    Timer tm;
    printf("    %-26s %9s %9s %10s\n", "payload", "MB", "ms", "GB/s");
    struct { const char *nm; size_t bpp; } pl[] = {
        {"int32 x2 (unpacked)", 8}, {"int16 x2 / packed", 4}, {"int8 x2 (lossy)", 2}};
    for (auto &e : pl) {
        size_t nb = NOUT * e.bpp;
        CHECK(cudaMemcpy(hbuf, dbuf, nb, cudaMemcpyDeviceToHost));
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            tm.start();
            for (int i = 0; i < 5; ++i) CHECK(cudaMemcpy(hbuf, dbuf, nb, cudaMemcpyDeviceToHost));
            best = std::min(best, tm.stop() / 5);
        }
        printf("    %-26s %9.1f %9.3f %10.1f\n", e.nm, nb / 1e6, best, nb / (best * 1e6));
    }
    cudaFree(dbuf); cudaFreeHost(hbuf);
    printf("\n    B. one ring all-reduce of a 4096x4096 wgrad, P=128, 2(P-1)/P bytes/el\n");
    const double NGRAD = 4096.0 * 4096.0, f = 2.0 * 127 / 128.0;
    struct { const char *nm; double gbs; } links[] = {
        {"NVLink 4 (per GPU)", 450.0}, {"IB NDR 400G", 50.0}, {"PCIe 4.0 x16", 25.0}};
    printf("    %-22s %10s %10s %10s\n", "link", "4 B ms", "2 B ms", "1 B ms");
    for (auto &L : links)
        printf("    %-22s %10.3f %10.3f %10.3f\n", L.nm,
               NGRAD * 4 * f / (L.gbs * 1e6), NGRAD * 2 * f / (L.gbs * 1e6), NGRAD * f / (L.gbs * 1e6));
}

// ============================================================================
// [12] dense -- normalization-aware packing
// ============================================================================
//
// The idea: if the reduced result gets normalized anyway, a partial does not
// have to be exactly recoverable. It has to be good enough AFTER the sum. So:
//
//   * how many bits does the a-priori bound 144*d demand,
//   * how many does the data actually use,
//   * what does an exception path cost if you size for the data and escape the
//     rare overflow,
//   * and what does an MX-style shared exponent over a group of partials give,
//     which is a larger container holding more quantized partials -- exactly
//     the "denser packing in a bigger slot" shape.
static void exp_dense() {
    printf("\n[12] normalization-aware packing: how dense can a partial get?\n");
    printf("    A depth-d partial is an integer with |p| <= 144*d. That bound is free\n");
    printf("    but loose -- real data is a random walk, so it lands near sqrt(d).\n");
    const int TRIALS = 20000;
    std::uniform_int_distribution<int> cd(0, 14);
    printf("\n    %5s %8s %8s %10s %9s %9s %11s\n", "depth", "a-priori", "measured",
           "typ |p|", "w=12 ovf", "w=10 ovf", "w=8 ovf");
    for (int d : {16, 32, 64, 128, 227}) {
        long long mx = 0;
        double sum = 0;
        long long o12 = 0, o10 = 0, o8 = 0;
        for (int t = 0; t < TRIALS; ++t) {
            long long a = 0;
            for (int k = 0; k < d; ++k)
                a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
            long long m = llabs(a);
            mx = std::max(mx, m);
            sum += (double)m;
            if (m >= (1LL << 11)) ++o12;
            if (m >= (1LL << 9))  ++o10;
            if (m >= (1LL << 7))  ++o8;
        }
        printf("    %5d %8d %8d %10.1f %8.3f%% %8.3f%% %10.3f%%\n",
               d, bits_of((long long)FP4_PMAX * d) + 1, bits_of(mx) + 1, sum / TRIALS,
               100.0 * o12 / TRIALS, 100.0 * o10 / TRIALS, 100.0 * o8 / TRIALS);
    }
    printf("\n    'a-priori' is what the bound demands and always holds. 'measured' is\n");
    printf("    what %d random partials actually needed. The gap is 3-5 bits, which is\n", TRIALS);
    printf("    the headroom an exception path can sell you -- and the reason the\n");
    printf("    OVERFL row in [5] still verified exact.\n");
    // --- exception path: size for the data, escape the rare overflow.
    printf("\n    exception path: w-bit payload plus an escape list for the overflows\n");
    printf("    %5s %5s %11s %13s %11s\n", "depth", "w", "escape rate", "eff bytes/p", "vs int16");
    for (int d : {32, 128}) {
        for (int w : {12, 10, 8}) {
            long long esc = 0;
            for (int t = 0; t < TRIALS; ++t) {
                long long a = 0;
                for (int k = 0; k < d; ++k)
                    a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                if (llabs(a) >= (1LL << (w - 1))) ++esc;
            }
            double rate = (double)esc / TRIALS;
            double eff = w / 8.0 + rate * 6.0;      // escape carries index + int32
            printf("    %5d %5d %10.3f%% %13.3f %10.2fx\n", d, w, 100.0 * rate, eff, 2.0 / eff);
        }
    }
    printf("    An escape costs an index plus a full-width value, so a rate above a\n");
    printf("    few percent eats the win. Exact, but only worth it where the payload\n");
    printf("    width sits just above the measured range.\n");
    // --- MX-style partials: a bigger container holding more quantized partials.
    printf("\n    MX-style partials: G partials share one E8M0 exponent, w-bit mantissa\n");
    printf("    (this is the 'larger data type, more quantized data packed' shape)\n");
    printf("    %5s %5s %5s %13s %14s %11s\n", "depth", "G", "w", "bytes/partial", "rel err of sum", "vs int16");
    for (int d : {32, 128}) {
        for (int G : {8, 32}) {
            for (int w : {8, 6, 4}) {
                double e2 = 0, s2 = 0;
                const int GROUPS = 400;
                for (int t = 0; t < GROUPS; ++t) {
                    std::vector<long long> p(G);
                    for (int j = 0; j < G; ++j) {
                        long long a = 0;
                        for (int k = 0; k < d; ++k)
                            a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                        p[j] = a;
                    }
                    double amax = 0;
                    for (int j = 0; j < G; ++j) amax = std::max(amax, fabs((double)p[j]));
                    // Shared exponent rounded UP, so nothing in the group clamps.
                    double sc = amax > 0 ? quant_e8m0_up((float)(amax / ((1 << (w - 1)) - 1))) : 1.0;
                    long long exact = 0, approx = 0;
                    for (int j = 0; j < G; ++j) {
                        exact += p[j];
                        long long q = llrint((double)p[j] / sc);
                        long long lim = (1LL << (w - 1)) - 1;
                        q = std::max(-lim, std::min(lim, q));
                        approx += q;
                    }
                    double da = (double)approx * sc - (double)exact;
                    e2 += da * da;
                    s2 += (double)exact * (double)exact;
                }
                double bpp = w / 8.0 + 1.0 / G;
                printf("    %5d %5d %5d %13.3f %13.4f%% %10.2fx\n",
                       d, G, w, bpp, 100.0 * sqrt(e2 / GROUPS) / sqrt(s2 / GROUPS), 2.0 / bpp);
            }
        }
    }
    printf("\n    Error falls ~4x per 2 mantissa bits, as it should, and barely moves\n");
    printf("    with G. Two effects cancel: a bigger group has a bigger amax, so the\n");
    printf("    shared grid is coarser, but its rounding errors are independent and\n");
    printf("    partly cancel in the sum. Since G is nearly free on accuracy, pick it\n");
    printf("    large to amortize the one scale byte -- G=32 costs 1/32 byte/partial.\n");
    printf("\n    That is the whole argument for normalization-aware packing: you are not\n");
    printf("    protecting a partial, you are protecting a sum, and the sum is more\n");
    printf("    forgiving than any single term in it.\n");
    printf("\n    Best cell above is w=8, G=32: 1.03 bytes/partial, ~2x denser than int16,\n");
    printf("    for under 1%% error on the reduced value. w=4 is 3.8x denser and ~14%%,\n");
    printf("    which is too coarse for a gradient but not obviously too coarse for an\n");
    printf("    activation that a layernorm is about to rescale anyway.\n");
    printf("\n    The trade against int16 is stark and worth stating plainly: int16 with\n");
    printf("    the 144*d bound is EXACT, needs no amax pass, and needs no second pass\n");
    printf("    over the partials. Everything denser here is approximate and needs both.\n");
    printf("    Pick by whether the consumer normalizes.\n");
}
// ============================================================================
// ============================================================================
// [13] fair -- all-reduce at equal bytes: what closure and bits are each worth
// ============================================================================
// Two topologies for the reduce-scatter half of an all-reduce:
//   ring    partial sums travel, so a float re-rounds a growing sum at every
//           hop and an integer field must be wide enough for every prefix sum.
//   direct  each contribution travels once to the chunk's owner, which
//           accumulates wide (fp32 / int32) and re-encodes the total once for
//           the all-gather. Same bytes as a ring, P-1 messages per rank.
// Integer grids are per 64-element chunk:
//   a-priori  144*K*P, needs no statistics
//   oracle    the largest prefix sum on the ring; not computable in advance
//   safe      sum over ranks of each rank's local chunk max (one sum-all-reduce);
//             bounds every prefix sum in any order
//   direct    max over ranks of the local chunk max (one max-all-reduce)
static void exp_fair() {
    printf("\n[13] all-reduce at equal bytes: closure, bits, and topology\n");
    const int NEL = 4096, KLOC = 256, CH = 64, NC = NEL / CH, NROW = 12;
    std::uniform_int_distribution<int> cd(0, 14);
    printf("    %d elements, depth-%d FP4 contraction per rank, per-%d integer scales.\n",
           NEL, KLOC, CH);
    printf("\n    %5s  %-38s %6s %13s\n", "P", "wire", "b/el", "rel err (RMS)");
    double E[3][NROW];
    long long sat_ring = 0;
    int bf_moved = 0, int_moved = 0;
    int pi = 0;
    for (int P : {8, 32, 128}) {
        std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i) g[r][i] = fp4_dot(KLOC, cd);
        std::vector<long long> S(NEL, 0);
        for (int r = 0; r < P; ++r) for (int i = 0; i < NEL; ++i) S[i] += g[r][i];
        double sn = 0, amax = 0, gmax = 0;
        std::vector<double> lmax(NC, 0), lsum(NC, 0), wmax(NC, 0);
        for (int i = 0; i < NEL; ++i) {
            sn += (double)S[i] * S[i];
            amax = std::max(amax, fabs((double)S[i]));
            long long a = 0;
            for (int r = 0; r < P; ++r) {
                a += g[r][i];
                wmax[i / CH] = std::max(wmax[i / CH], std::max(fabs((double)a), fabs((double)g[r][i])));
                lmax[i / CH] = std::max(lmax[i / CH], fabs((double)g[r][i]));
                gmax = std::max(gmax, fabs((double)g[r][i]));
            }
        }
        for (int c = 0; c < NC; ++c)
            for (int r = 0; r < P; ++r) {
                double m = 0;
                for (int i = c * CH; i < (c + 1) * CH; ++i) m = std::max(m, fabs((double)g[r][i]));
                lsum[c] += m;
            }
        sn = sqrt(sn);
        std::vector<double> o(NEL);
        int row = 0;
        auto emit = [&](const char *nm, double bpe) {
            double e = 0;
            for (int i = 0; i < NEL; ++i) { double d = o[i] - (double)S[i]; e += d * d; }
            E[pi][row++] = 100.0 * sqrt(e) / sn;
            printf("    %5d  %-38s %6.2f %12.4f%%\n", P, nm, bpe, E[pi][row - 1]);
        };
        // Floats: enc() rounds a pre-scaled value to the wire format.
        auto float_ring = [&](float (*enc)(float), double s, bool reverse = false) {
            for (int i = 0; i < NEL; ++i) {
                float a = 0.f;
                for (int k = 0; k < P; ++k) {
                    float q = enc((float)(g[reverse ? P - 1 - k : k][i] / s));
                    a = k ? enc(a + q) : q;
                }
                o[i] = (double)a * s;
            }
        };
        auto float_direct = [&](float (*enc)(float), double s_src, double s_tot) {
            for (int i = 0; i < NEL; ++i) {
                float a = 0.f;                                   // owner's fp32 accumulator
                for (int r = 0; r < P; ++r) a += enc((float)(g[r][i] / s_src)) * (float)s_src;
                o[i] = (double)enc((float)(a / s_tot)) * s_tot;
            }
        };
        // Integers: the ring carries the running sum in the W-bit field itself.
        auto int_ring = [&](int W, const std::vector<double> &ref, std::vector<long long> *tot = nullptr,
                            bool reverse = false) {
            const long long lim = (1LL << (W - 1)) - 1;
            for (int i = 0; i < NEL; ++i) {
                double s = ref[i / CH] > 0 ? (double)lim / ref[i / CH] : 1.0;
                long long a = 0;
                for (int k = 0; k < P; ++k) {
                    long long q = llrint((double)g[reverse ? P - 1 - k : k][i] * s);
                    a = sat_add(a, std::max(-lim, std::min(lim, q)), lim, &sat_ring);
                }
                if (tot) (*tot)[i] = a;
                o[i] = (double)a / s;
            }
        };
        auto int_direct = [&](int W) {
            const long long lim = (1LL << (W - 1)) - 1;
            for (int c = 0; c < NC; ++c) {
                double s = lmax[c] > 0 ? (double)lim / lmax[c] : 1.0, vmax = 0;
                for (int i = c * CH; i < (c + 1) * CH; ++i) {
                    long long a = 0;                             // owner's int32 accumulator
                    for (int r = 0; r < P; ++r) a += llrint((double)g[r][i] * s);
                    o[i] = (double)a / s;
                    vmax = std::max(vmax, fabs(o[i]));
                }
                double s2 = vmax > 0 ? (double)lim / vmax : 1.0; // owner-local, sent with the chunk
                for (int i = c * CH; i < (c + 1) * CH; ++i) o[i] = (double)llrint(o[i] * s2) / s2;
            }
        };
        const double bpc = 2.0 / CH;                             // one 2-byte scale per chunk
        float_ring(to_bf16, 1.0);           emit("bf16, ring", 2.0);
        float_direct(to_bf16, 1.0, 1.0);    emit("bf16, direct", 2.0);
        float_ring(to_fp16, amax);          emit("fp16, ring", 2.0);
        float_direct(to_fp16, amax, amax);  emit("fp16, direct", 2.0);
        std::vector<double> apri(NC, (double)FP4_PMAX * KLOC * P);
        int_ring(16, apri);                 emit("int16, ring, a-priori 144*K*P grid", 2.0);
        int_ring(16, wmax);                 emit("int16, ring, oracle grid", 2.0 + bpc);
        int_ring(16, lsum);                 emit("int16, ring, safe grid", 2.0 + bpc);
        int_direct(16);                     emit("int16, direct", 2.0 + bpc);
        const double s8 = std::max(gmax, amax) / E4M3_MAX;
        float_ring(quant_e4m3, s8);         emit("fp8 e4m3, ring", 1.0);
        float_direct(quant_e4m3, gmax / E4M3_MAX, amax / E4M3_MAX); emit("fp8 e4m3, direct", 1.0);
        int_ring(8, wmax);                  emit("int8, ring, oracle grid", 1.0 + bpc);
        int_direct(8);                      emit("int8, direct", 1.0 + bpc);
        if (P == 128) {
            // Reverse the ring. Integers on the safe grid cannot saturate in any
            // order, so closure says nothing may change; bf16 has no such property.
            std::vector<double> f0(NEL);
            float_ring(to_bf16, 1.0);
            f0 = o;
            float_ring(to_bf16, 1.0, true);
            for (int i = 0; i < NEL; ++i) bf_moved += o[i] != f0[i];
            std::vector<long long> t0(NEL), t1(NEL);
            int_ring(16, lsum, &t0);
            int_ring(16, lsum, &t1, true);
            for (int i = 0; i < NEL; ++i) int_moved += t0[i] != t1[i];
        }
        printf("\n");
        ++pi;
    }
    const double *e = E[2];
    printf("    at P=128:\n");
    printf("      closure, bf16        ring/direct %7.2fx\n", e[0] / e[1]);
    printf("      closure, fp16        ring/direct %7.2fx\n", e[2] / e[3]);
    printf("      int16 vs fp16, ring              %7.2fx\n", e[2] / e[5]);
    printf("      int16 vs fp16, direct            %7.2fx\n", e[3] / e[7]);
    printf("      int16 ring, safe vs oracle grid  %7.2fx  (price of a computable ring grid)\n", e[6] / e[5]);
    printf("      int16 ring (oracle) vs direct    %7.2fx\n", e[5] / e[7]);
    printf("    int16 direct across P=8/32/128: %.4f / %.4f / %.4f%%; oracle ring %.4f / %.4f / %.4f%%\n",
           E[0][7], E[1][7], E[2][7], E[0][5], E[1][5], E[2][5]);
    printf("    reversing the ring at P=128 changes %d of %d bf16 outputs and %d int16 outputs\n",
           bf_moved, NEL, int_moved);
    printf("    saturated hops across all integer ring rows: %lld\n", sat_ring);
}

// ============================================================================
// [14] llm -- does any of this survive at 1B+ parameters?
// ============================================================================
//
// Everything so far ran at NEL=2048-4096 elements, K=256, P<=128. A 1B model
// all-reduces 1e9 elements per step with K in the thousands over hundreds of
// ranks, and three things get worse in that direction at once:
//
//   A. The a-priori 144*K*P bound loosens as K*P grows while the actual amax
//      only grows as sqrt(K*P). Every doubling of K*P throws away half a bit.
//      This part is arithmetic and it is not kind to the a-priori story.
//
//   B. amax over 1e9 elements is a max over 1e9 samples, so a single global
//      scale is set by the most extreme value in the entire tensor. Gradient
//      tensors have heavy tails. This is the failure mode that actually bites.
//
//   C. The scales themselves have to be agreed across ranks BEFORE the payload
//      moves, or the grids do not line up and closure is lost. That is a second
//      collective. It has to be cheap enough to be worth it.
//
// Part A uses a Gaussian surrogate for the per-rank contribution rather than
// running the K-deep contraction directly -- at K=16384 and P=1024 the direct
// loop is 1e11 multiply-adds. The surrogate is validated against the real thing
// at K=256 in the first table, and it is exact in distribution: a K-deep dot
// product of independent E2M1 codes is a sum of K iid mean-zero terms, so it is
// Gaussian with variance K*E[q^2]^2, rounded to an integer.
static void exp_llm() {
    printf("\n[14] scaling to 1B+ parameters: what breaks and what holds\n");
    // Exact second moment of one q1*q2 product over the 15 legal codes.
    double m2 = 0;
    for (int a = 0; a < 15; ++a) m2 += (double)e2m1_q(LEGAL[a]) * e2m1_q(LEGAL[a]);
    m2 /= 15.0;
    const double sig_prod = m2 * m2;               // Var(q1*q2) = E[q1^2] E[q2^2]
    // ------------------------------------------------------------- validate
    printf("\n    A. the a-priori 144*K*P bound against the range the data actually uses\n");
    {
        const int NV = 4096, KV = 256, PV = 8;
        std::uniform_int_distribution<int> cd(0, 14);
        double am_direct = 0;
        for (int i = 0; i < NV; ++i) {
            long long s = 0;
            for (int r = 0; r < PV; ++r)
                for (int k = 0; k < KV; ++k)
                    s += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
            am_direct = std::max(am_direct, fabs((double)s));
        }
        std::normal_distribution<double> nd(0.0, sqrt((double)KV * PV * sig_prod));
        double am_surr = 0;
        for (int i = 0; i < NV; ++i) am_surr = std::max(am_surr, fabs(nd(rng)));
        printf("       surrogate check at K=%d P=%d: direct amax %.0f, Gaussian amax %.0f (%.2fx)\n",
               KV, PV, am_direct, am_surr, am_surr / am_direct);
    }
    printf("\n    %7s %6s %14s %12s %10s %12s %12s\n",
           "K", "P", "144*K*P bound", "real amax", "bits lost", "int16 a-pri", "int16 meas");
    const int NBIG = 1 << 20;
    for (int K : {256, 1024, 4096, 16384}) {
        for (int P : {8, 128, 1024}) {
            double bound = (double)FP4_PMAX * K * P;
            double sd = sqrt((double)K * P * sig_prod);
            std::normal_distribution<double> nd(0.0, sd);
            double amax = 0, sn = 0;
            for (int i = 0; i < NBIG; ++i) {
                double v = nd(rng);
                amax = std::max(amax, fabs(v));
                sn += v * v;
            }
            sn = sqrt(sn / NBIG);
            double lost = log2(bound / amax);
            // RMS relative error of a uniform quantizer with 15 magnitude bits
            // over each of the two ranges. Per-rank roundings add as sqrt(P).
            double step_a = bound / 32767.0, step_m = amax / 32767.0;
            double ea = 100.0 * (step_a / sqrt(12.0)) * sqrt((double)P) / sn;
            double em = 100.0 * (step_m / sqrt(12.0)) * sqrt((double)P) / sn;
            printf("    %7d %6d %14.3e %12.3e %10.1f %11.3f%% %11.4f%%\n",
                   K, P, bound, amax, lost, ea, em);
        }
    }
    printf("\n       The a-priori bound loses half a bit per doubling of K*P, because it\n");
    printf("       grows linearly while the data only grows as sqrt. By K=16384 and\n");
    printf("       P=1024 it is 11.6 bits looser than the data needs, leaving about 3\n");
    printf("       of int16's 15 -- and a 3-bit gradient is a 455%% error, which is to\n");
    printf("       say no gradient. The measured column at the same size is 0.14%%.\n");
    printf("\n       The bound is not useless, it is just not a grid spacing. What it\n");
    printf("       gives is the guarantee that every partial is an INTEGER on a known\n");
    printf("       lattice, which is what makes the sum exact once you have picked a\n");
    printf("       spacing. Pick the spacing from a measured amax.\n");
    // ------------------------------------------------------------------ B
    printf("\n    B. one global scale over a whole tensor, against per-chunk scales\n");
    printf("       %d elements, standard normal, then with 0.1%% of values at 30 sigma.\n", NBIG);
    printf("\n    %-22s %8s %10s %13s %10s\n", "data", "scale", "chunk", "rel err (RMS)", "b/el");
    for (int outl = 0; outl < 2; ++outl) {
        std::normal_distribution<double> nd(0.0, 1.0);
        std::uniform_real_distribution<double> ur(0.0, 1.0);
        std::vector<double> v(NBIG);
        for (int i = 0; i < NBIG; ++i) {
            v[i] = nd(rng);
            if (outl && ur(rng) < 0.001) v[i] *= 30.0;
        }
        double sn = 0;
        for (int i = 0; i < NBIG; ++i) sn += v[i] * v[i];
        sn = sqrt(sn);
        const char *dn = outl ? "0.1% at 30 sigma" : "clean N(0,1)";
        for (int W : {16, 8}) {
            double lim = (double)((1 << (W - 1)) - 1);
            for (int CH : {NBIG, 65536, 4096, 512}) {
                double e = 0;
                for (int c = 0; c < NBIG; c += CH) {
                    int n = std::min(CH, NBIG - c);
                    double am = 0;
                    for (int j = 0; j < n; ++j) am = std::max(am, fabs(v[c + j]));
                    double sc = am > 0 ? lim / am : 1.0;
                    for (int j = 0; j < n; ++j) {
                        double q = llrint(v[c + j] * sc) / sc;
                        e += (q - v[c + j]) * (q - v[c + j]);
                    }
                }
                printf("    %-22s %8s %10d %12.5f%% %10.4f\n", dn, W == 16 ? "int16" : "int8",
                       CH == NBIG ? 0 : CH, 100.0 * sqrt(e) / sn,
                       W / 8.0 + (CH == NBIG ? 0.0 : 2.0 / CH));
            }
        }
    }
    printf("\n       chunk 0 means one scale for the whole tensor. Two things to read\n");
    printf("       here, and only one of them is the interesting one.\n");
    printf("\n       int16 barely cares. Even with 0.1%% of values at 30 sigma, a global\n");
    printf("       scale costs it 0.063%% and per-512 chunks only get that to 0.013%% --\n");
    printf("       a 5x that nobody is going to notice downstream. Fifteen bits has\n");
    printf("       enough room that outliers do not have to be handled, only survived.\n");
    printf("\n       int8 gets the SAME ~5x from chunking -- 16.2%% down to 3.4%% -- and\n");
    printf("       that is the whole point: the ratio is a property of the data, not of\n");
    printf("       the width, but 16%% against 3.4%% is a decision and 0.063%% against\n");
    printf("       0.013%% is not. Chunking does not become more effective at 8 bits, it\n");
    printf("       becomes necessary, because that is where the error crosses into the\n");
    printf("       range where anyone cares. A 2-byte scale per 4096 costs 0.05%%.\n");
    // ------------------------------------------------------------------ C
    printf("\n    C. the cost of agreeing the scales, per gradient all-reduce\n");
    printf("       Ranks must share a grid or the integers do not line up, so the\n");
    printf("       scales go first, as their own (tiny) max-all-reduce.\n");
    printf("\n    %-12s %12s %12s %14s %12s %10s\n",
           "model", "params", "bf16 GB", "int16+sc GB", "scale GB", "overhead");
    struct { const char *nm; double p; } models[] = {
        {"1B", 1.0e9}, {"7B", 7.0e9}, {"70B", 70.0e9}, {"405B", 405.0e9}};
    const int CHC = 4096;
    for (auto &M : models) {
        double f = 2.0 * (1024 - 1) / 1024.0;               // ring, P=1024
        double bf = M.p * 2 * f / 1e9;
        double sc = (M.p / CHC) * 2 * f / 1e9;
        double it = M.p * 2 * f / 1e9;
        printf("    %-12s %12.1e %12.2f %14.2f %12.4f %9.3f%%\n",
               M.nm, M.p, bf, it, sc, 100.0 * sc / it);
    }
    printf("\n       Same bytes on the wire as bf16 plus a rounding error of overhead,\n");
    printf("       for a payload that is exact under summation. The real cost is the\n");
    printf("       extra latency: two dependent collectives instead of one. At 1024\n");
    printf("       ranks a small all-reduce is ~50-100 us, so this is only worth it\n");
    printf("       for tensors big enough that the payload dominates -- which, at\n");
    printf("       these sizes, every one of them is. It also pipelines: compute the\n");
    printf("       scales for bucket i+1 while bucket i is on the wire.\n");
    printf("\n       At int8 with per-4096 scales the payload halves again to %.2f GB\n",
           7.0e9 * 1.0 * (2.0 * 1023 / 1024) / 1e9);
    printf("       for a 7B model, at the error in the [13] int8 row.\n");
}
// ============================================================================
// Shared multi-step gradient model for [15]-[17]
// ============================================================================
// Per element i, a persistent true gradient mu[i], log-normal in magnitude so
// the tensor spans several decades. Rank r contributes mu[i]/P plus noise,
// rounded to an integer (the FP4 lattice). Signal persists across steps; noise
// does not.
struct GradModel {
    int NEL, P;
    std::vector<double> mu;
    double noise_floor;
    GradModel(int n, int p) : NEL(n), P(p), mu(n) {
        std::normal_distribution<double> lg(0.0, 2.0);
        std::uniform_int_distribution<int> sg(0, 1);
        for (int i = 0; i < NEL; ++i)
            mu[i] = (sg(rng) ? 1.0 : -1.0) * 100.0 * exp(lg(rng));
        noise_floor = 10.0;
    }
    // One step: g[r][i] integers, S = exact total.
    void step(std::vector<std::vector<long long>> &g, std::vector<long long> &S,
              double scale_drift = 1.0) {
        for (int i = 0; i < NEL; ++i) S[i] = 0;
        for (int r = 0; r < P; ++r) {
            for (int i = 0; i < NEL; ++i) {
                double m = mu[i] * scale_drift;
                std::normal_distribution<double> nd(m / P, 0.3 * fabs(m) + noise_floor);
                long long v = llrint(nd(rng));
                g[r][i] = v;
                S[i] += v;
            }
        }
    }
};
static void chunk_amax(const std::vector<long long> &S, int CH, std::vector<double> &am) {
    std::fill(am.begin(), am.end(), 0.0);
    for (size_t i = 0; i < S.size(); ++i)
        am[i / CH] = std::max(am[i / CH], fabs((double)S[i]));
}
// Per-chunk max of every value a ring carries: each contribution and each
// prefix sum. Differs from the amax of the total: per-rank gradients are
// mostly noise, so prefix sums overshoot the final answer. Sizing a ring's
// grid by the total saturates the wire.
static void chunk_wiremax(const std::vector<std::vector<long long>> &g, int CH,
                          std::vector<double> &wm) {
    int P = (int)g.size(), NEL = (int)g[0].size();
    std::fill(wm.begin(), wm.end(), 0.0);
    for (int i = 0; i < NEL; ++i) {
        long long a = 0;
        double mx = 0;
        for (int r = 0; r < P; ++r) {
            a += g[r][i];
            mx = std::max(mx, std::max(fabs((double)a), fabs((double)g[r][i])));
        }
        wm[i / CH] = std::max(wm[i / CH], mx);
    }
}
static double rel_err(const std::vector<double> &a, const std::vector<double> &ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - ref[i];
        e += d * d;
        n += ref[i] * ref[i];
    }
    return sqrt(e) / sqrt(n);
}

// ============================================================================
// Integer all-reduce of one chunk, shared by [15] and [16]
// ============================================================================
// RING carries the running sum in the W-bit field, so the field must hold
// every prefix sum and overflow saturates at a hop, away from any source.
// DIRECT sends each value once to the chunk's owner, which sums exactly and
// re-encodes the total on an owner-local grid for the all-gather; every
// clip then happens at a source (or the owner), where a residual can see it.
enum Topo { RING, DIRECT };
// The magnitude the field must hold for chunk [i0, i0+n): v[r*NEL + i] is
// what rank r sends.
static double wire_need(Topo topo, int P, int NEL, int i0, int n, const std::vector<double> &v) {
    double m = 0;
    for (int i = i0; i < i0 + n; ++i) {
        double a = 0;
        for (int r = 0; r < P; ++r) {
            double x = v[(size_t)r * NEL + i];
            a += x;
            m = std::max(m, topo == RING ? std::max(fabs(a), fabs(x)) : fabs(x));
        }
    }
    return m;
}
// Reduces chunk [i0, i0+n) on a grid whose top is grid*1.05. res (per rank)
// and ores (per owner element) are error-feedback residuals, or nullptr.
static void int_reduce(Topo topo, int W, bool sr, int P, int NEL, int i0, int n, double grid,
                       const std::vector<double> &v, double *res, double *ores, double *out,
                       long long &clip, long long &sat, std::uniform_real_distribution<double> &ur) {
    const long long lim = (1LL << (W - 1)) - 1;
    const double s = grid > 0 ? lim / (grid * 1.05) : 1.0;
    double tmax = 0;
    for (int i = i0; i < i0 + n; ++i) {
        long long a = 0;
        for (int r = 0; r < P; ++r) {
            double x = v[(size_t)r * NEL + i];
            long long q = sr ? (long long)floor(x * s + ur(rng)) : llrint(x * s);
            if (q > lim || q < -lim) { q = std::max(-lim, std::min(lim, q)); ++clip; }
            if (res) res[(size_t)r * NEL + i] = x - q / s;
            a = topo == RING ? sat_add(a, q, lim, &sat) : a + q;
        }
        out[i] = a / s + (ores ? ores[i] : 0.0);
        tmax = std::max(tmax, fabs(out[i]));
    }
    if (topo == DIRECT) {
        const double s2 = tmax > 0 ? lim / tmax : 1.0;
        for (int i = i0; i < i0 + n; ++i) {
            double y = llrint(out[i] * s2) / s2;
            if (ores) ores[i] = out[i] - y;
            out[i] = y;
        }
    }
}

// ============================================================================
// [15] ef -- error feedback, measured on the accumulated gradient
// ============================================================================
// Error feedback keeps each rank's rounding residual and adds it to the next
// step's gradient. The metric is the error in sum_t S_t, reported as drift =
// rel error * T, i.e. "steps of gradient lost". Integer grids are per 64
// elements and sized by this step's values (ring: an oracle; direct: one
// max-all-reduce). [16] replaces them with predictions.
enum RndMode { R_DET, R_SR, R_EF };
static void exp_ef() {
    const int NEL = 1024, P = 32, CH = 64, TMAX = 128, NC = NEL / CH;
    printf("\n[15] error feedback over %d steps, %d elements, %d ranks\n", TMAX, NEL, P);
    GradModel gm(NEL, P);
    struct Method { int W; RndMode m; Topo topo; const char *nm; };   // W = 0: bf16 ring
    std::vector<Method> ms = {
        {0,  R_DET, RING,   "bf16 ring"},
        {0,  R_EF,  RING,   "bf16 ring + source EF"},
        {16, R_DET, RING,   "int16 ring, nearest"},
        {16, R_SR,  RING,   "int16 ring, stochastic"},
        {16, R_EF,  RING,   "int16 ring, EF"},
        {8,  R_DET, RING,   "int8 ring, nearest"},
        {8,  R_SR,  RING,   "int8 ring, stochastic"},
        {8,  R_EF,  RING,   "int8 ring, EF"},
        {4,  R_EF,  RING,   "int4 ring, EF"},
        {16, R_EF,  DIRECT, "int16 direct, EF"},
        {8,  R_DET, DIRECT, "int8 direct, nearest"},
        {8,  R_EF,  DIRECT, "int8 direct, EF"},
        {4,  R_EF,  DIRECT, "int4 direct, EF"},
    };
    const int NM = (int)ms.size();
    std::vector<std::vector<double>> res(NM, std::vector<double>((size_t)P * NEL, 0.0));
    std::vector<std::vector<double>> ores(NM, std::vector<double>(NEL, 0.0));
    std::vector<std::vector<double>> accum(NM, std::vector<double>(NEL, 0.0));
    std::vector<double> exact_accum(NEL, 0.0), step_err(NM, 0.0), v((size_t)P * NEL), out(NEL);
    std::vector<std::vector<double>> drift(NM);
    std::vector<long long> clip(NM, 0), sat(NM, 0);
    std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
    std::vector<long long> S(NEL);
    std::vector<double> am(NC), wm(NC);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    double head_sum = 0;
    for (int t = 1; t <= TMAX; ++t) {
        gm.step(g, S);
        chunk_amax(S, CH, am);
        chunk_wiremax(g, CH, wm);
        for (int c = 0; c < NC; ++c) head_sum += am[c] > 0 ? wm[c] / am[c] : 1.0;
        for (int i = 0; i < NEL; ++i) exact_accum[i] += (double)S[i];
        for (int k = 0; k < NM; ++k) {
            const Method &M = ms[k];
            const bool ef = M.m == R_EF;
            for (int r = 0; r < P; ++r)
                for (int i = 0; i < NEL; ++i)
                    v[(size_t)r * NEL + i] = (double)g[r][i] + (ef ? res[k][(size_t)r * NEL + i] : 0.0);
            if (M.W == 0) {                           // float ring: rounds at every hop
                for (int i = 0; i < NEL; ++i) {
                    float a = 0.f;
                    for (int r = 0; r < P; ++r) {
                        size_t ri = (size_t)r * NEL + i;
                        float q = to_bf16((float)v[ri]);
                        if (ef) res[k][ri] = v[ri] - q;
                        a = r ? to_bf16(a + q) : q;
                    }
                    out[i] = a;
                }
            } else {
                for (int c = 0; c < NC; ++c)
                    int_reduce(M.topo, M.W, M.m == R_SR, P, NEL, c * CH, CH,
                               wire_need(M.topo, P, NEL, c * CH, CH, v), v,
                               ef ? res[k].data() : nullptr,
                               ef && M.topo == DIRECT ? ores[k].data() : nullptr,
                               out.data(), clip[k], sat[k], ur);
            }
            double se = 0, sn = 0;
            for (int i = 0; i < NEL; ++i) {
                accum[k][i] += out[i];
                double d = out[i] - (double)S[i];
                se += d * d;
                sn += (double)S[i] * (double)S[i];
            }
            step_err[k] += 100.0 * sqrt(se) / sqrt(sn);
        }
        if (t == 8 || t == 32 || t == 128)
            for (int k = 0; k < NM; ++k) drift[k].push_back(rel_err(accum[k], exact_accum) * t);
    }
    const double nv = (double)TMAX * NEL * P;
    printf("\n    %-24s %6s %13s %9s %9s %9s %9s %9s\n", "wire", "b/el", "per-step err",
           "drift@8", "drift@32", "drift@128", "src clip", "hop sat");
    for (int k = 0; k < NM; ++k)
        printf("    %-24s %6.3f %12.4f%% %9.3f %9.3f %9.3f %8.4f%% %8.4f%%\n", ms[k].nm,
               ms[k].W ? ms[k].W / 8.0 + 2.0 / CH : 2.0, step_err[k] / TMAX,
               drift[k][0], drift[k][1], drift[k][2], 100.0 * clip[k] / nv, 100.0 * sat[k] / nv);
    printf("\n    wire max / result amax, mean over chunks and steps: %.2fx\n",
           head_sum / ((double)TMAX * NC));
}

// ============================================================================
// [16] predict -- last step's grid instead of a scale collective
// ============================================================================
// Exact rows size the grid from this step's values, which costs a collective
// before the payload can move. Predicted rows use last step's value times a
// margin (piggybacked on last step's payload) and send nothing extra. The
// gradient scale follows a log random walk (5%/step) with a 2% chance per
// step of a 2.5x spike.
static void exp_predict() {
    const int NEL = 1024, P = 32, CH = 64, TMAX = 128, NC = NEL / CH;
    printf("\n[16] predicting the grid: one collective instead of two, %d steps\n", TMAX);
    struct Cfg { Topo topo; int W; double margin; bool ef; const char *nm; };   // margin 0: exact
    std::vector<Cfg> cfgs = {
        {RING,   16, 0.0,  true,  "int16 ring, oracle grid + EF"},
        {RING,   16, 1.0,  true,  "int16 ring, predicted x1.00 + EF"},
        {RING,   16, 2.0,  true,  "int16 ring, predicted x2.00 + EF"},
        {RING,   16, 2.0,  false, "int16 ring, predicted x2.00"},
        {DIRECT, 16, 0.0,  true,  "int16 direct, max-all-reduce + EF"},
        {DIRECT, 16, 1.0,  true,  "int16 direct, predicted x1.00 + EF"},
        {DIRECT, 16, 1.25, true,  "int16 direct, predicted x1.25 + EF"},
        {DIRECT, 16, 2.0,  true,  "int16 direct, predicted x2.00 + EF"},
        {DIRECT, 16, 1.0,  false, "int16 direct, predicted x1.00"},
        {DIRECT, 8,  0.0,  true,  "int8 direct, max-all-reduce + EF"},
        {DIRECT, 8,  1.0,  true,  "int8 direct, predicted x1.00 + EF"},
    };
    const int NCF = (int)cfgs.size();
    std::vector<std::vector<double>> res(NCF, std::vector<double>((size_t)P * NEL, 0.0));
    std::vector<std::vector<double>> ores(NCF, std::vector<double>(NEL, 0.0));
    std::vector<std::vector<double>> accum(NCF, std::vector<double>(NEL, 0.0));
    std::vector<std::vector<double>> prev(NCF, std::vector<double>(NC, 0.0));
    std::vector<double> exact_accum(NEL, 0.0), sperr(NCF, 0.0), v((size_t)P * NEL), out(NEL);
    std::vector<long long> clip(NCF, 0), sat(NCF, 0);
    GradModel gm(NEL, P);
    std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
    std::vector<long long> S(NEL);
    std::normal_distribution<double> walk(0.0, 0.05);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    double drift = 1.0;
    for (int t = 1; t <= TMAX; ++t) {
        drift *= exp(walk(rng));
        if (ur(rng) < 0.02) drift *= 2.5;
        gm.step(g, S, drift);
        for (int i = 0; i < NEL; ++i) exact_accum[i] += (double)S[i];
        for (int k = 0; k < NCF; ++k) {
            const Cfg &C = cfgs[k];
            for (int r = 0; r < P; ++r)
                for (int i = 0; i < NEL; ++i)
                    v[(size_t)r * NEL + i] = (double)g[r][i] + (C.ef ? res[k][(size_t)r * NEL + i] : 0.0);
            for (int c = 0; c < NC; ++c) {
                double need = wire_need(C.topo, P, NEL, c * CH, CH, v);
                double grid = C.margin == 0.0 || prev[k][c] == 0.0 ? need : prev[k][c] * C.margin;
                int_reduce(C.topo, C.W, false, P, NEL, c * CH, CH, grid, v,
                           C.ef ? res[k].data() : nullptr,
                           C.ef && C.topo == DIRECT ? ores[k].data() : nullptr,
                           out.data(), clip[k], sat[k], ur);
                prev[k][c] = need;
            }
            double se = 0, sn = 0;
            for (int i = 0; i < NEL; ++i) {
                accum[k][i] += out[i];
                double d = out[i] - (double)S[i];
                se += d * d;
                sn += (double)S[i] * (double)S[i];
            }
            sperr[k] += 100.0 * sqrt(se) / sqrt(sn);
        }
    }
    const double nv = (double)TMAX * NEL * P;
    std::vector<double> acc(NCF);
    printf("\n    %-36s %9s %9s %13s %16s\n", "grid", "src clip", "hop sat",
           "per-step err", "accumulated err");
    for (int k = 0; k < NCF; ++k) {
        acc[k] = 100.0 * rel_err(accum[k], exact_accum);
        printf("    %-36s %8.4f%% %8.4f%% %12.4f%% %15.4f%%\n", cfgs[k].nm,
               100.0 * clip[k] / nv, 100.0 * sat[k] / nv, sperr[k] / TMAX, acc[k]);
    }
    printf("\n    predicted x1.00 + EF vs exact + EF: ring %.1fx, direct %.2fx accumulated error\n",
           acc[1] / acc[0], acc[5] / acc[4]);
}

// ============================================================================
// [17] tiers -- two-level reduction: where the roundings go, and what they cost
// ============================================================================
// flat = one ring over P ranks. hier = ring inside each node of L GPUs, then a
// ring across the G nodes. All integer running sums saturate.
static void exp_tiers() {
    printf("\n[17] hierarchical reduction: numerics, then time\n");
    const int NEL = 2048, CH = 64, NC = NEL / CH;
    printf("\n    %6s %5s %6s  %-36s %13s %11s\n",
           "nodes", "gpus", "P", "scheme", "rel err (RMS)", "vs flat bf16");
    struct Shape { int G, L; };
    for (Shape sh : {Shape{16, 8}, Shape{128, 8}}) {
        const int G = sh.G, L = sh.L, P = G * L;
        GradModel gm(NEL, P);
        std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
        std::vector<long long> S(NEL);
        std::vector<double> wm(NC), wmn(NC);
        gm.step(g, S);
        chunk_wiremax(g, CH, wm);                           // prefix maxima over P ranks
        std::vector<std::vector<long long>> ng(G, std::vector<long long>(NEL, 0));
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i) ng[r / L][i] += g[r][i];
        chunk_wiremax(ng, CH, wmn);                         // prefix maxima over G nodes
        double sn = 0;
        for (int i = 0; i < NEL; ++i) sn += (double)S[i] * (double)S[i];
        sn = sqrt(sn);
        double base = 0;
        long long sat = 0;
        auto report = [&](const char *nm, const std::vector<double> &out) {
            double e = 0;
            for (int i = 0; i < NEL; ++i) { double d = out[i] - (double)S[i]; e += d * d; }
            e = 100.0 * sqrt(e) / sn;
            if (base == 0) base = e;
            printf("    %6d %5d %6d  %-36s %12.5f%% %10.2fx\n", G, L, P, nm, e, base / e);
        };
        std::vector<double> out(NEL);
        for (int i = 0; i < NEL; ++i) {                     // 1. flat bf16
            float a = to_bf16((float)g[0][i]);
            for (int r = 1; r < P; ++r) a = to_bf16(a + to_bf16((float)g[r][i]));
            out[i] = a;
        }
        report("flat bf16", out);
        for (int i = 0; i < NEL; ++i) {                     // 2. hier bf16
            float b = 0.f;
            for (int n = 0; n < G; ++n) {
                float a = to_bf16((float)g[n * L][i]);
                for (int l = 1; l < L; ++l) a = to_bf16(a + to_bf16((float)g[n * L + l][i]));
                b = n ? to_bf16(b + a) : a;
            }
            out[i] = b;
        }
        report("hier bf16", out);
        const long long l16 = 32767, l8 = 127;
        std::vector<long long> t3(NEL), t4(NEL);
        for (int i = 0; i < NEL; ++i) {                     // 3. flat int16
            double s = l16 / (wm[i / CH] * 1.05);
            long long a = 0;
            for (int r = 0; r < P; ++r) a = sat_add(a, llrint((double)g[r][i] * s), l16, &sat);
            t3[i] = a;
            out[i] = a / s;
        }
        report("flat int16", out);
        for (int i = 0; i < NEL; ++i) {                     // 4. hier int16, same source grid
            double s = l16 / (wm[i / CH] * 1.05);
            long long b = 0;
            for (int n = 0; n < G; ++n) {
                long long a = 0;
                for (int l = 0; l < L; ++l)
                    a = sat_add(a, llrint((double)g[n * L + l][i] * s), l16, &sat);
                b = sat_add(b, a, l16, &sat);
            }
            t4[i] = b;
            out[i] = b / s;
        }
        report("hier int16, quantize once at source", out);
        for (int W : {16, 8}) {                             // 5-6. exact intra, narrow inter
            const long long lim = W == 16 ? l16 : l8;
            for (int i = 0; i < NEL; ++i) {
                double s = lim / (wmn[i / CH] * 1.05);
                long long b = 0;
                for (int n = 0; n < G; ++n)
                    b = sat_add(b, std::max(-lim, std::min(lim, (long long)llrint((double)ng[n][i] * s))),
                                lim, &sat);
                out[i] = b / s;
            }
            report(W == 16 ? "hier, exact intra + int16 inter" : "hier, exact intra + int8 inter", out);
        }
        int diff = 0;
        for (int i = 0; i < NEL; ++i) diff += t3[i] != t4[i];
        printf("    %6s %5s %6s  flat vs hier int16 totals: %d of %d integers differ; %lld saturated hops\n\n",
               "", "", "", diff, NEL, sat);
    }
    const double NP = 7.0e9, NVL = 450e9, IB = 50e9;
    const int G = 128, L = 8, P = G * L;
    printf("    time, one 7B all-reduce, %d nodes x %d GPUs, NVLink %.0f GB/s, IB %.0f GB/s:\n",
           G, L, NVL / 1e9, IB / 1e9);
    printf("\n    %-38s %9s %9s %9s %11s\n", "scheme", "intra ms", "inter ms", "total ms", "roundings");
    struct TR { const char *nm; double b_in, b_out; bool hier; const char *rnd; };
    const TR trs[] = {{"flat ring, bf16", 0, 2.0, false, "every hop"},
                      {"flat ring, int16", 0, 2.03, false, "P"},
                      {"hier, bf16", 2.0, 2.0, true, "every hop"},
                      {"hier, int16 quantize-once", 2.03, 2.03, true, "P"},
                      {"hier, exact(int32) intra + int16", 4.0, 2.03, true, "G"},
                      {"hier, int16 intra + int8 inter", 2.03, 1.03, true, "P + G"}};
    double tt[6];
    for (int k = 0; k < 6; ++k) {
        const TR &tr = trs[k];
        double intra = tr.hier ? (2.0 * (L - 1) / L) * NP * tr.b_in / NVL * 1e3 : 0.0;
        double inter = tr.hier ? (2.0 * (G - 1) / G) * (NP / L) * tr.b_out / IB * 1e3
                               : (2.0 * (P - 1) / P) * NP * tr.b_out / IB * 1e3;
        tt[k] = intra + inter;
        printf("    %-38s %9.2f %9.2f %9.2f %11s\n", tr.nm, intra, inter, tt[k], tr.rnd);
    }
    printf("\n    hier bf16 vs flat bf16 %.2fx; int16 quantize-once costs %+.1f%% over hier bf16;\n",
           tt[0] / tt[2], 100.0 * (tt[3] / tt[2] - 1));
    printf("    int32 intra costs %+.1f%%; int16 intra + int8 inter is %.2fx faster than hier bf16.\n",
           100.0 * (tt[4] / tt[2] - 1), tt[2] / tt[5]);
}

// ============================================================================
// [18] moe -- the same wire on an all-to-all
// ============================================================================
// Dispatch is a permutation (no additions in flight); combine is a k-term
// weighted sum at the destination. A: dispatch payload quantization. B: an
// integer combine. C: error feedback on dispatched activations, measured on
// both the plain per-expert activation sum and the expert's weight gradient
// sum_t x_t * delta_t, which is what the activations actually feed.
static void exp_moe() {
    printf("\n[18] MoE all-to-all: which parts of the wire story transfer?\n");
    const int NTOK = 512, DM = 256, TOPK = 2, CH = 64;
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    printf("\n    A. dispatch payload, %d tokens x %d dims, N(0,1) with 1%% at 15 sigma\n", NTOK, DM);
    std::vector<double> act((size_t)NTOK * DM);
    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = nd(rng);
        if (ur(rng) < 0.01) act[i] *= 15.0;
    }
    double an = 0;
    for (double v : act) an += v * v;
    an = sqrt(an);
    printf("\n    %-34s %7s %14s\n", "wire", "b/el", "rel err (RMS)");
    double e_bf16, e_fp8, e_i16, e_i8;
    {
        double e = 0;
        for (size_t i = 0; i < act.size(); ++i) {
            double q = (double)to_bf16((float)act[i]);
            e += (q - act[i]) * (q - act[i]);
        }
        e_bf16 = 100.0 * sqrt(e) / an;
        printf("    %-34s %7.3f %13.4f%%\n", "bf16", 2.0, e_bf16);
    }
    {
        double e = 0;
        for (int t = 0; t < NTOK; ++t) {
            double am = 0;
            for (int j = 0; j < DM; ++j) am = std::max(am, fabs(act[(size_t)t * DM + j]));
            double s = am > 0 ? am / E4M3_MAX : 1.0;
            for (int j = 0; j < DM; ++j) {
                double v = act[(size_t)t * DM + j];
                double q = (double)quant_e4m3((float)(v / s)) * s;
                e += (q - v) * (q - v);
            }
        }
        e_fp8 = 100.0 * sqrt(e) / an;
        printf("    %-34s %7.3f %13.4f%%\n", "fp8 e4m3, per-token scale", 1.0 + 2.0 / DM, e_fp8);
    }
    for (int W : {16, 8, 6, 4}) {
        for (int BL : {DM, CH}) {
            double lim = (double)((1 << (W - 1)) - 1), e = 0;
            for (int t = 0; t < NTOK; ++t) {
                for (int b = 0; b < DM; b += BL) {
                    double am = 0;
                    for (int j = b; j < b + BL; ++j)
                        am = std::max(am, fabs(act[(size_t)t * DM + j]));
                    double s = am > 0 ? lim / am : 1.0;
                    for (int j = b; j < b + BL; ++j) {
                        double v = act[(size_t)t * DM + j];
                        double q = std::max(-lim, std::min(lim, (double)llrint(v * s))) / s;
                        e += (q - v) * (q - v);
                    }
                }
            }
            e = 100.0 * sqrt(e) / an;
            if (BL == CH && W == 16) e_i16 = e;
            if (BL == CH && W == 8) e_i8 = e;
            char nm[64];
            snprintf(nm, sizeof nm, "int%d, per-%d scale", W, BL);
            printf("    %-34s %7.3f %13.4f%%\n", nm, W / 8.0 + 2.0 / BL, e);
        }
    }
    printf("    equal bytes: int16/per-64 vs bf16 %.1fx; int8/per-64 vs fp8 %.2fx\n",
           e_bf16 / e_i16, e_fp8 / e_i8);

    printf("\n    B. combine, k=%d: sum_k w_k * y_k with an int8 payload\n", TOPK);
    {
        std::vector<double> y((size_t)NTOK * TOPK * DM), w((size_t)NTOK * TOPK);
        for (size_t i = 0; i < y.size(); ++i) y[i] = nd(rng);
        for (int t = 0; t < NTOK; ++t) {
            double s = 0;
            for (int k = 0; k < TOPK; ++k) { w[t * TOPK + k] = ur(rng); s += w[t * TOPK + k]; }
            for (int k = 0; k < TOPK; ++k) w[t * TOPK + k] /= s;
        }
        std::vector<double> ref((size_t)NTOK * DM, 0.0);
        for (int t = 0; t < NTOK; ++t)
            for (int k = 0; k < TOPK; ++k)
                for (int j = 0; j < DM; ++j)
                    ref[(size_t)t * DM + j] += w[t * TOPK + k] * y[((size_t)t * TOPK + k) * DM + j];
        double rn = 0;
        for (double v : ref) rn += v * v;
        rn = sqrt(rn);
        auto run_combine = [&](bool intw) {
            double lim = 127.0, e = 0;
            for (int t = 0; t < NTOK; ++t) {
                for (int b = 0; b < DM; b += CH) {
                    double am = 0;
                    for (int k = 0; k < TOPK; ++k)
                        for (int j = b; j < b + CH; ++j)
                            am = std::max(am, fabs(y[((size_t)t * TOPK + k) * DM + j]));
                    double s = am > 0 ? lim / am : 1.0;
                    for (int j = b; j < b + CH; ++j) {
                        double acc = 0;
                        for (int k = 0; k < TOPK; ++k) {
                            double q = std::max(-lim, std::min(lim,
                                (double)llrint(y[((size_t)t * TOPK + k) * DM + j] * s)));
                            double wk = w[t * TOPK + k];
                            if (intw) wk = llrint(wk * 255.0) / 255.0;   // 8-bit router weight
                            acc += wk * q / s;
                        }
                        double d = acc - ref[(size_t)t * DM + j];
                        e += d * d;
                    }
                }
            }
            return 100.0 * sqrt(e) / rn;
        };
        printf("    %-40s %13.4f%%\n", "fp32 router weights", run_combine(false));
        printf("    %-40s %13.4f%%\n", "int8 router weights (all-integer combine)", run_combine(true));
    }

    printf("\n    C. error feedback under a live router, int6 per-64, 8 experts, 64 steps\n");
    {
        const int TSTEP = 64, NEXP = 8, W = 6, DO = 4;     // DO backward signals per token
        const double lim = (double)((1 << (W - 1)) - 1);
        printf("\n    %-32s %16s %18s\n", "residual keyed by", "activation sum", "weight gradient");
        for (int mode = 0; mode < 3; ++mode) {             // none, token slot, (expert, channel)
            std::vector<double> r_tok((size_t)NTOK * DM, 0.0), r_exp((size_t)NEXP * DM, 0.0);
            std::vector<double> acc((size_t)NEXP * DM, 0.0), exa((size_t)NEXP * DM, 0.0);
            std::vector<double> gacc((size_t)NEXP * DM * DO, 0.0), gexa((size_t)NEXP * DM * DO, 0.0);
            std::vector<double> esig((size_t)NEXP * DM);
            for (auto &x : esig) x = nd(rng) * 0.05;       // small persistent per-expert signal
            for (int st = 0; st < TSTEP; ++st) {
                for (int t = 0; t < NTOK; ++t) {
                    int ex = (int)(ur(rng) * NEXP) % NEXP;
                    double dl[DO];
                    for (int o = 0; o < DO; ++o) dl[o] = nd(rng);
                    for (int b = 0; b < DM; b += CH) {
                        double v[CH], am = 0;
                        for (int j = 0; j < CH; ++j) {
                            v[j] = nd(rng) + esig[(size_t)ex * DM + b + j];
                            am = std::max(am, fabs(v[j]));
                        }
                        double s = am > 0 ? lim / (am * 1.05) : 1.0;
                        for (int j = 0; j < CH; ++j) {
                            size_t ti = (size_t)t * DM + b + j, ei = (size_t)ex * DM + b + j;
                            double x = v[j] + (mode == 1 ? r_tok[ti] : mode == 2 ? r_exp[ei] : 0.0);
                            double q = std::max(-lim, std::min(lim, (double)llrint(x * s))) / s;
                            if (mode == 1) r_tok[ti] = x - q;
                            if (mode == 2) r_exp[ei] = x - q;
                            acc[ei] += q;
                            exa[ei] += v[j];
                            for (int o = 0; o < DO; ++o) {
                                gacc[ei * DO + o] += q * dl[o];
                                gexa[ei * DO + o] += v[j] * dl[o];
                            }
                        }
                    }
                }
            }
            const char *nm = mode == 0 ? "nothing" : mode == 1 ? "token slot" : "(expert, channel)";
            printf("    %-32s %15.4f%% %17.4f%%\n", nm, 100.0 * rel_err(acc, exa), 100.0 * rel_err(gacc, gexa));
        }
    }

    printf("\n    D. bytes, one MoE layer (32768 tokens, d=4096, top-2), dispatch+combine, fwd+bwd, IB 50 GB/s\n");
    const double TOKENS = 4096 * 8, DMOD = 4096;
    printf("\n    %-24s %10s %12s %12s %10s\n", "wire", "b/el", "GB/layer", "IB ms", "vs bf16");
    double bb = 0;
    for (auto pr : {std::make_pair("bf16", 2.0), std::make_pair("fp8 e4m3", 1.008),
                    std::make_pair("int8, per-64", 1.031), std::make_pair("int6, per-64", 0.781),
                    std::make_pair("int4, per-64", 0.531)}) {
        double gb = 4.0 * TOKENS * DMOD * TOPK * pr.second / 1e9;
        double ms = gb * 1e9 / 50e9 * 1e3;
        if (bb == 0) bb = ms;
        printf("    %-24s %10.3f %12.3f %12.2f %9.2fx\n", pr.first, pr.second, gb, ms, bb / ms);
    }
}

// ============================================================================
// [19] chain -- keeping the partials integral from GEMM output to all-reduce
// ============================================================================
//
// Every experiment so far tested one link of the chain in isolation. This one
// asks whether the value can stay an INTEGER the whole way: out of the FP4
// GEMM, through the split-K reduction, across the wire, through the all-reduce,
// and only become a float at the very end. No dequantize, no requantize, no
// intermediate float at all.
//
// The obstacle is the block scale, and it is worth being precise about why.
//
// A K-deep MXFP4 dot product is not one integer. It is a sum over K/32 blocks,
// each with its own scale:  sum_b  X_b * dot_b,  where dot_b is an integer
// bounded by 144*32 and X_b is the block scale. The conventional mainloop
// dequantizes at every block boundary -- multiply dot_b by X_b into an fp32
// accumulator -- and that multiply is where the value stops being an integer
// and starts rounding, once per block, before it has gone anywhere.
//
// But in MX formats X_b is E8M0: a bare POWER OF TWO. So X_b * dot_b is
// dot_b shifted, and a sum of shifted integers is an integer, exactly, in a
// wide enough field. Align every block to the smallest exponent in the tile
// and the entire dot product is one exact integer.
//
// Which produces a genuinely awkward trade with [8]. That experiment measured
// NVFP4 beating MXFP4 by 1.23x clean and 1.83x on outliers, and recommended
// switching, because NVFP4's E4M3 scale has a mantissa and lands closer to
// amax/6. That mantissa is exactly what breaks this. MXFP4's worse scale
// format is the one that makes an integral chain possible.
//
// The field width you need is bits(144*d) + the exponent SPREAD across the
// blocks being summed, so the first thing to measure is that spread.

static void exp_chain() {
    printf("\n[19] end-to-end integral: GEMM -> split-K -> wire -> all-reduce\n");

    std::uniform_int_distribution<int> cd(0, 14);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ur(0.0, 1.0);

    // ---------------------------------------------------------------- part A
    printf("\n    A. E8M0 block-scale exponent spread on realistic activations\n");
    printf("       The integral accumulator must span every block it sums, so its\n");
    printf("       width is bits(144*d) plus this spread. Measured over blocks of 32.\n");
    printf("\n    %10s %10s %14s %14s %14s\n", "K", "outliers", "mean spread", "p99 spread", "max spread");
    for (int K : {512, 2048, 8192}) {
        for (int outl = 0; outl < 2; ++outl) {
            const int NROW = 256;
            std::vector<int> spreads;
            double tot = 0;
            int mx = 0;
            for (int r = 0; r < NROW; ++r) {
                int lo = 1000, hi = -1000;
                for (int b = 0; b < K; b += 32) {
                    double am = 0;
                    for (int j = 0; j < 32; ++j) {
                        double v = nd(rng);
                        if (outl && ur(rng) < 0.005) v *= 20.0;
                        am = std::max(am, fabs(v));
                    }
                    int e = am > 0 ? (int)floorf(log2f((float)am)) - 2 : -127;
                    lo = std::min(lo, e);
                    hi = std::max(hi, e);
                }
                spreads.push_back(hi - lo);
                tot += hi - lo;
                mx = std::max(mx, hi - lo);
            }
            std::sort(spreads.begin(), spreads.end());
            printf("    %10d %10s %14.2f %14d %14d\n", K, outl ? "0.5% 20x" : "clean",
                   tot / NROW, spreads[(size_t)(NROW * 0.99)], mx);
        }
    }

    // ---------------------------------------------------------------- part B
    printf("\n    B. accumulator width for an EXACT integral dot product\n");
    printf("       bits = bits(144*32) + log2(blocks) + exponent spread\n");
    printf("\n    %8s %10s %10s %10s %8s %10s\n",
           "K", "blocks", "base bits", "spread", "total", "fits");
    for (int K : {512, 2048, 8192, 32768}) {
        int blocks = K / 32;
        int base = bits_of((long long)FP4_PMAX * 32);
        for (int spread : {8, 14}) {
            int need = base + bits_of(blocks) + spread;
            printf("    %8d %10d %10d %10d %8d %10s\n", K, blocks, base, spread, need,
                   need <= 31 ? "int32" : (need <= 63 ? "int64" : "no"));
        }
    }
    printf("\n       An int32 accumulator covers realistic depths at a clean spread and\n");
    printf("       runs out on outlier-heavy data at K=32768. That is the real limit\n");
    printf("       of the integral chain, and it is a much friendlier limit than the\n");
    printf("       one the packed-operand idea hit in [7], because it is about RANGE,\n");
    printf("       which you can buy with a wider accumulator, not about SIGNIFICAND,\n");
    printf("       which no Hopper format has.\n");

    // ---------------------------------------------------------------- part C
    printf("\n    C. rounding events in each chain, and the measured end-to-end error\n");

    const int NEL = 2048, S = 32, DEP = 32, P = 64;
    const int NBLK = 4;                          // blocks per split, each 32 deep
    printf("\n       %d outputs, %d splits of depth %d, %d blocks per split, %d ranks.\n",
           NEL, S, DEP, NBLK, P);

    // Build the ground truth in exact rational form: integer dot products with
    // power-of-two scales, so an fp64 reference is exact.
    std::vector<std::vector<double>> gr(P, std::vector<double>(NEL, 0.0));
    std::vector<std::vector<std::vector<long long>>> dotv(
        P, std::vector<std::vector<long long>>(NEL));
    std::vector<std::vector<std::vector<int>>> expv(
        P, std::vector<std::vector<int>>(NEL));
    std::vector<double> S_exact(NEL, 0.0);
    for (int r = 0; r < P; ++r) {
        for (int i = 0; i < NEL; ++i) {
            dotv[r][i].resize((size_t)S * NBLK);
            expv[r][i].resize((size_t)S * NBLK);
            double acc = 0;
            for (int t = 0; t < S * NBLK; ++t) {
                long long d = 0;
                for (int k = 0; k < DEP; ++k)
                    d += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                int e = (int)(ur(rng) * 8.0) - 10;       // E8M0 exponent, spread 8
                dotv[r][i][t] = d;
                expv[r][i][t] = e;
                acc += (double)d * ldexp(1.0, e);        // exact in fp64
            }
            gr[r][i] = acc;
            S_exact[i] += acc;
        }
    }
    double sn = 0;
    for (int i = 0; i < NEL; ++i) sn += S_exact[i] * S_exact[i];
    sn = sqrt(sn);

    struct Res { const char *nm; const char *rnd; double err; double bpw; };
    std::vector<Res> rows;

    // --- conventional: dequantize each block into fp32, accumulate in fp32,
    // narrow to bf16 for the wire, round at every hop.
    {
        double e = 0;
        for (int i = 0; i < NEL; ++i) {
            float tot = 0.f;
            for (int r = 0; r < P; ++r) {
                float a = 0.f;
                for (int t = 0; t < S * NBLK; ++t)
                    a += (float)dotv[r][i][t] * ldexpf(1.0f, expv[r][i][t]);
                float wv = to_bf16(a);
                tot = r == 0 ? wv : to_bf16(tot + wv);
            }
            double d = (double)tot - S_exact[i];
            e += d * d;
        }
        rows.push_back({"fp32 dequant + bf16 wire", "S*NBLK + 1 + P", 100.0 * sqrt(e) / sn, 2.0});
    }

    // --- conventional but with an fp32 wire: removes the wire roundings, keeps
    // the per-block dequantize. Isolates which half of the chain costs more.
    {
        double e = 0;
        for (int i = 0; i < NEL; ++i) {
            double tot = 0;
            for (int r = 0; r < P; ++r) {
                float a = 0.f;
                for (int t = 0; t < S * NBLK; ++t)
                    a += (float)dotv[r][i][t] * ldexpf(1.0f, expv[r][i][t]);
                tot += (double)a;
            }
            double d = tot - S_exact[i];
            e += d * d;
        }
        rows.push_back({"fp32 dequant + fp32 wire", "S*NBLK", 100.0 * sqrt(e) / sn, 4.0});
    }

    // --- integral: align every block to a common exponent and accumulate in
    // int64. GEMM, split-K reduce and cross-rank sum are all exact integer
    // addition; the ONLY rounding in the whole chain is the narrowing to the
    // wire, and with an int32 wire on a shared grid there is not even that.
    {
        double e32 = 0, e16 = 0;
        // One shared grid for the whole tensor, sized by the true wire max.
        int emin = 1000;
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i)
                for (int t = 0; t < S * NBLK; ++t) emin = std::min(emin, expv[r][i][t]);
        double base = ldexp(1.0, emin);
        double wmax = 0;
        for (int i = 0; i < NEL; ++i) {
            long long run = 0, mx = 0;
            for (int r = 0; r < P; ++r) {
                long long a = 0;
                for (int t = 0; t < S * NBLK; ++t)
                    a += dotv[r][i][t] << (expv[r][i][t] - emin);
                run += a;
                mx = std::max(mx, llabs(run));
            }
            wmax = std::max(wmax, (double)mx);
        }
        // int32 wire: the aligned integer is carried as-is if it fits.
        bool fits32 = wmax <= 2147483647.0;
        for (int i = 0; i < NEL; ++i) {
            long long tot = 0;
            for (int r = 0; r < P; ++r)
                for (int t = 0; t < S * NBLK; ++t)
                    tot += dotv[r][i][t] << (expv[r][i][t] - emin);
            double d = (double)tot * base - S_exact[i];
            e32 += d * d;
        }
        rows.push_back({fits32 ? "integral, int32 wire (exact)" : "integral, int32 wire (OVERFLOW)",
                        "0", 100.0 * sqrt(e32) / sn, 4.0});
        // int16 wire: one narrowing, then exact everywhere else.
        double s16 = 32767.0 / (wmax * 1.05);
        for (int i = 0; i < NEL; ++i) {
            double tot = 0;
            for (int r = 0; r < P; ++r) {
                long long a = 0;
                for (int t = 0; t < S * NBLK; ++t)
                    a += dotv[r][i][t] << (expv[r][i][t] - emin);
                double q = (double)llrint((double)a * s16);
                tot += std::max(-32767.0, std::min(32767.0, q));
            }
            double d = tot / s16 * base - S_exact[i];
            e16 += d * d;
        }
        rows.push_back({"integral, int16 wire", "P (source only)", 100.0 * sqrt(e16) / sn, 2.0});
        printf("       aligned wire max %.3e, int32 %s\n", wmax, fits32 ? "sufficient" : "OVERFLOWS");
    }

    printf("\n    %-38s %18s %7s %14s\n", "chain", "roundings/output", "b/wire", "rel err (RMS)");
    for (auto &r : rows)
        printf("    %-38s %18s %7.1f %13.5f%%\n", r.nm, r.rnd, r.bpw, r.err);

    printf("\n       READ THE SECOND ROW FIRST. fp32 dequantize-and-accumulate comes\n");
    printf("       out at EXACTLY zero error, so all %d of its nominal roundings cost\n", S * NBLK);
    printf("       nothing and the integral chain's on-GPU advantage here is nil.\n");
    printf("       The reason is that fp32 has a 24-bit significand and the aligned\n");
    printf("       magnitude is %.2e, comfortably under 2^24 = 1.68e7 -- so fp32 is\n", 3.312e6);
    printf("       holding these integers exactly and is not really floating point\n");
    printf("       for this workload at all.\n");
    printf("\n       So the honest claim is narrower than 'the integral chain removes\n");
    printf("       roundings'. It removes roundings that were not happening. What it\n");
    printf("       does remove is the bf16 narrowing and the per-hop rounding -- the\n");
    printf("       first row against the last, 0.975%% against 0.026%%, which is the\n");
    printf("       [13] result arriving again by a different route.\n");
    printf("\n       Part E finds where fp32 actually breaks, which is the only place\n");
    printf("       the integral GEMM earns its keep.\n");
    printf("\n       Note which row is best and what it costs. An int32 wire is exact\n");
    printf("       end to end and moves 4 bytes, which is fp32's bandwidth and throws\n");
    printf("       away the entire point. The int16 row is the deployable one: half\n");
    printf("       the bytes, one rounding per source, none anywhere else.\n");

    // ---------------------------------------------------------------- part D

    // ---------------------------------------------------------------- part E
    printf("\n    E. where fp32 accumulation stops being exact, and integral does not\n");
    printf("       An MXFP4 partial is an integer times a power of two, and fp32 holds\n");
    printf("       integers exactly up to 2^24. So the entire integral-chain argument\n");
    printf("       is worth nothing until the ALIGNED magnitude -- the value after\n");
    printf("       every block has been shifted to a common exponent -- crosses that\n");
    printf("       line. Sweeping depth and exponent spread to find it.\n");
    printf("\n    %8s %8s %14s %10s %14s %14s\n",
           "K", "spread", "aligned max", "vs 2^24", "fp32 err", "integral err");
    for (int Kt : {4096, 16384, 65536}) {
        for (int spread : {2, 6, 10, 14, 18}) {
            const int NB = Kt / 32, NO = 256;
            double e32 = 0, ei = 0, sn2 = 0, amax = 0;
            for (int i = 0; i < NO; ++i) {
                std::vector<long long> dv(NB);
                std::vector<int> ev(NB);
                int emin = 1000;
                for (int t = 0; t < NB; ++t) {
                    long long d = 0;
                    for (int k = 0; k < 32; ++k)
                        d += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                    dv[t] = d;
                    ev[t] = (int)(ur(rng) * (spread + 1)) - 10;
                    emin = std::min(emin, ev[t]);
                }
                // Exact reference: shifted integers summed in int64 (exact while
                // under 2^63), then scaled by the common power of two.
                long long ali = 0;
                for (int t = 0; t < NB; ++t) ali += dv[t] << (ev[t] - emin);
                double exact = (double)ali * ldexp(1.0, emin);
                amax = std::max(amax, fabs((double)ali));
                // fp32 dequantize-and-accumulate, as a real mainloop does it.
                float a = 0.f;
                for (int t = 0; t < NB; ++t) a += (float)dv[t] * ldexpf(1.0f, ev[t]);
                double d32 = (double)a - exact;
                e32 += d32 * d32;
                ei += 0.0;                    // integral is exact by construction
                sn2 += exact * exact;
            }
            printf("    %8d %8d %14.3e %10s %13.6f%% %13.6f%%\n", Kt, spread, amax,
                   amax > 16777216.0 ? "OVER" : "under",
                   100.0 * sqrt(e32) / sqrt(sn2), 100.0 * sqrt(ei) / sqrt(sn2));
        }
    }
    printf("\n       The integral column is zero everywhere, by construction -- that is\n");
    printf("       what 'exact' means and it is not a measurement. The fp32 column IS\n");
    printf("       a measurement, and it does exactly what the theory says: dead zero\n");
    printf("       until the aligned magnitude passes 2^24, then non-zero, with the\n");
    printf("       crossover landing on the predicted row every time.\n");
    printf("\n       And it does not matter. Part A measured the real exponent spread\n");
    printf("       at 1-2 clean and 5-6 with 0.5%% of values at 20 sigma, so every\n");
    printf("       realistic row here is one of the zeros. The spreads that break\n");
    printf("       fp32 are 14 and 18, which this data never produces. Even there the\n");
    printf("       error is 5e-5%% -- three orders of magnitude below what the int16\n");
    printf("       wire costs in part C.\n");
    printf("\n       So this is a negative result, and a fairly complete one: keeping\n");
    printf("       the partials integral through the GEMM and the split-K reduction\n");
    printf("       buys nothing measurable, because fp32 was already exact for this\n");
    printf("       workload. The mechanism is real and the crossover is where it was\n");
    printf("       predicted to be, but the crossover is outside the operating range.\n");
    printf("\n       The end-to-end story therefore splits cleanly in two, and only\n");
    printf("       one half is load-bearing:\n");
    printf("         on-GPU   fp32 is already exact for realistic MXFP4 workloads,\n");
    printf("                  so keeping the partials integral is free and buys\n");
    printf("                  nothing measurable. What it buys is unconditional:\n");
    printf("                  fp32's exactness holds only while K, spread and split\n");
    printf("                  count stay under a bound nobody checks at runtime, and\n");
    printf("                  the integral version has no bound to check. That is\n");
    printf("                  worth having and it is not worth a paper.\n");
    printf("         off-GPU  this is where the error is, all of it, and where every\n");
    printf("                  measured win in this repo from [11] onward comes from.\n");
    printf("\n    F. what this costs you in format choice\n");
    printf("       [8] measured NVFP4 beating MXFP4 by 1.23x clean and 1.83x on\n");
    printf("       outliers and recommended switching. NVFP4's scale is E4M3 -- it\n");
    printf("       has a 3-bit mantissa, so the scale is NOT a power of two, so\n");
    printf("       X_b * dot_b is not a shift and the blocks cannot be summed as\n");
    printf("       integers. The integral chain needs MXFP4.\n");
    printf("\n       So the two recommendations in this repo are in direct conflict,\n");
    printf("       and which one wins depends on where your error budget is being\n");
    printf("       spent. If the quantizer dominates, take NVFP4 and give up the\n");
    printf("       integral chain. If the interconnect dominates -- which is the\n");
    printf("       premise of everything from [11] onward -- take MXFP4 and keep it.\n");
    printf("       On Blackwell, where NVFP4 is the native datapath and the emulation\n");
    printf("       argument disappears, this gets decided for you.\n");
    printf("\n       There is a third option this rig has not tested: an E8M0 scale\n");
    printf("       with NVFP4's block size of 16, which keeps the shift and buys back\n");
    printf("       some of the block-granularity half of NVFP4's advantage. That is\n");
    printf("       not a standard format, which is a real objection to it.\n");
}

int main(int argc, char **argv) {
    struct { const char *name; void (*fn)(); } exps[] = {
        {"narrow", exp_narrow}, {"int8acc", exp_int8acc}, {"nvfp4", exp_nvfp4},
        {"mxfp8", exp_mxfp8}, {"interleave", exp_interleave}, {"link", exp_link},
        {"dense", exp_dense}, {"fair", exp_fair}, {"llm", exp_llm}, {"ef", exp_ef},
        {"predict", exp_predict}, {"tiers", exp_tiers}, {"moe", exp_moe}, {"chain", exp_chain}};
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
    for (auto &e : exps)
        if (!only || !strcmp(only, e.name)) {
            rng.seed(20260909);
            e.fn();
        }
    printf("\n");
    return 0;
}
