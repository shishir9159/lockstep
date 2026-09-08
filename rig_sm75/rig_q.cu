// rig_q -- quantization and transport experiments, sm_75 and up.
//
// rig.cu asked whether the bit-packing idea works. The answer was: not as an
// accumulation format, yes as a transport format. This file asks the follow-on
// questions, and every one of them exists because a previous claim in this repo
// did not survive its own control.
//
//   [6]  narrow      Is the split-K win PACKING or just NARROWING? Experiment
//                    [5] compared int32x2 against a packed int16 pair, which is
//                    8 bytes against 4. Two separate int16 arrays are also 4.
//                    If that third row ties the packed one, the result is "FP4
//                    lets you store 16-bit partials exactly" and the pairing is
//                    a rounding error. This is the control [5] was missing.
//
//   [7]  int8acc     INT8/s32 is the only Hopper datapath whose accumulator is
//                    wide enough for a packed dual dot product (32 bits against
//                    fp32's 24 and the FP8 path's ~14). Does the scheme survive
//                    there? Separates the accumulator question from the operand
//                    question so we learn which one actually kills it.
//
//   nvcc -O3 -arch=sm_75 -o rig_q rig_q.cu
//   ./rig_q              # all of them
//   ./rig_q narrow       # just the control
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
static std::mt19937 rng(20260909);
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
// The 15 legal E2M1 codes (code 8 is negative zero, skipped).
static const int LEGAL[15] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
static inline float to_bf16(float x) {
    union { float f; uint32_t u; } c;
    c.f = x;
    c.u = (c.u + 0x7FFFu + ((c.u >> 16) & 1u)) & 0xFFFF0000u;
    return c.f;
}
static inline int bits_of(long long v) {
    int n = 0;
    while (v > 0) { v >>= 1; ++n; }
    return n;
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
int main(int argc, char **argv) {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("%s  sm_%d%d  %.1f GiB  %d SMs\n", p.name, p.major, p.minor,
           p.totalGlobalMem / 1073741824.0, p.multiProcessorCount);
    const char *only = argc > 1 ? argv[1] : nullptr;
    // Reseed before every experiment. Without this, an experiment's numbers
    // depend on which experiments ran BEFORE it, so `./rig_q tiers` and
    // `./rig_q` disagree and neither is quotable. Same seed each time, so a
    // single experiment reproduces standalone and in the suite.
    auto want = [&](const char *n) {
        bool w = !only || !strcmp(only, n);
        if (w) rng.seed(20260909);
        return w;
    };
    if (want("narrow")) exp_narrow();
    if (want("int8acc")) exp_int8acc();
    printf("\n");
    return 0;
}
