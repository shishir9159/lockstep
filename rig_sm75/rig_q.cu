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
//   [8]  nvfp4       MXFP4 (block 32, E8M0 scale) against NVFP4 (block 16,
//                    E4M3 scale). Accuracy per bit, on clean and outlier data.
//
//   [9]  mxfp8       MXFP4 against MXFP8. On Hopper both run at the same FLOP
//                    rate, because MXFP4 is emulated onto the FP8 datapath. So
//                    what does the 4-bit format actually buy?
//
//   [10] interleave  Nibble-interleave A1 and A2 into one byte so a single load
//                    stream serves both microbatches. Identical byte count --
//                    this is a locality test, not a bandwidth test, and it is
//                    labelled that way so nobody reports it as the latter.
//
//   [11] link        The reason for all of this: inter-GPU reduction. Measures
//                    real off-chip bandwidth here, then simulates a P-rank ring
//                    all-reduce in bf16, fp8, and fixed-point int16 with the
//                    a-priori 144*d bound. NOTE: its headline 39x is correct
//                    as a number and wrong as an explanation -- see [13].
//
//   [12] dense       Normalization-aware packing. If the result gets normalized
//                    after the reduction anyway, partials do not need to be
//                    exactly recoverable -- they need to be good enough after
//                    the sum. How dense can we go, and what does the a-priori
//                    bound cost us against the measured range?
//
//   [13] fair        The control for [11]. Its 39x was attributed to
//                    associativity, but its own numbers say otherwise. Splits
//                    the gap into bits-per-element and closure-under-addition,
//                    against a properly scaled fp16 wire rather than raw bf16.
//
//   [14] llm         Does any of it survive at 1B+ parameters? The a-priori
//                    bound loosens as sqrt(K*P), a single global amax dies on
//                    heavy tails, and the per-chunk scales have to be agreed
//                    across ranks before the payload moves. Prices all three.
//
//   [15] ef          [13] left an int8 wire 5x WORSE than bf16 at half the
//                    bytes. Error feedback keeps the rounding residual and
//                    replays it next step, so nothing is lost, only delayed --
//                    and on a fixed-point grid the residual is EXACT.
//
//   [16] predict     [14] left the scale all-reduce on the critical path. Use
//                    last step's scale instead and let feedback absorb the
//                    mispredictions, since a clipped value and a rounded value
//                    leave the same kind of residual.
//
//   [17] tiers       Real clusters are not flat rings. Does quantize-once make
//                    a hierarchy bit-identical to a flat ring, and is per-node
//                    quantization worth the intra-node bandwidth it costs?
//                    (Yes, and no -- the time model kills the second one.)
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
#include "quant_sim.cuh"
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
// [11] link -- the inter-GPU case, which is the actual reason for all of this
// ============================================================================
//
// Off-chip is where a saved byte is worth the most. Two halves:
//
//   A. measured. Real device-to-host bandwidth on this box at 8 / 4 / 2 bytes
//      per output pair. PCIe is a slow link, which makes it a good stand-in for
//      an interconnect rather than a memory bus.
//
//   B. simulated. A P-rank ring all-reduce of a wgrad, in four wire formats.
//      The claim under test: a fixed-point payload derived from the a-priori
//      144*d bound sums EXACTLY across hops, because integer addition is
//      associative, while bf16 and fp8 round at every one of the P-1 hops.
//      The cost is that the a-priori bound is loose -- so we also measure the
//      normalization-aware variant that spends one scalar all-reduce on the
//      real amax and gets the bits back.
static void exp_link() {
    printf("\n[11] inter-GPU reduction: what a narrow exact payload is worth off-chip\n");
    // ---------------------------------------------------------------- part A
    const size_t NOUT = 1u << 22;                       // 4M output pairs
    printf("\n    A. measured off-chip bandwidth on this box (%zu output pairs)\n", NOUT);
    void *dbuf;
    CHECK(cudaMalloc(&dbuf, NOUT * 8));
    void *hbuf;
    CHECK(cudaMallocHost(&hbuf, NOUT * 8));
    Timer tm;
    printf("    %-26s %9s %9s %10s\n", "payload", "MB", "ms", "GB/s");
    struct { const char *nm; size_t bpp; } pl[] = {
        {"int32 x2 (unpacked)", 8}, {"int16 x2 / packed", 4}, {"int8 x2 (lossy)", 2}};
    for (auto &e : pl) {
        size_t nb = NOUT * e.bpp;
        cudaMemcpy(hbuf, dbuf, nb, cudaMemcpyDeviceToHost);
        CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            tm.start();
            for (int i = 0; i < 5; ++i) cudaMemcpy(hbuf, dbuf, nb, cudaMemcpyDeviceToHost);
            best = std::min(best, tm.stop() / 5);
        }
        printf("    %-26s %9.1f %9.3f %10.1f\n", e.nm, nb / 1e6, best, nb / (best * 1e6));
    }
    cudaFree(dbuf); cudaFreeHost(hbuf);
    // ---------------------------------------------------------------- part B
    const int NEL = 2048;        // gradient elements
    const int KLOC = 256;        // per-rank contraction depth
    const double BOUND = (double)FP4_PMAX * KLOC;        // |g_r| <= 144 * KLOC, a priori
    printf("\n    B. simulated P-rank ring all-reduce of one wgrad tile\n");
    printf("       %d elements, each rank contributes a depth-%d FP4 contraction.\n", NEL, KLOC);
    printf("       a-priori bound |g_r| <= 144*%d = %.0f -- known without looking at data.\n",
           KLOC, BOUND);
    printf("\n    %5s  %-30s %6s %13s %13s\n", "P", "wire format", "B/el", "rel err (RMS)", "hop rounding");
    std::uniform_int_distribution<int> cd(0, 14);
    for (int P : {8, 32, 128}) {
        // Each rank's exact integer gradient contribution.
        std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i) {
                long long a = 0;
                for (int k = 0; k < KLOC; ++k)
                    a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                g[r][i] = a;
            }
        std::vector<long long> S(NEL, 0);
        for (int r = 0; r < P; ++r) for (int i = 0; i < NEL; ++i) S[i] += g[r][i];
        double sn = 0;
        for (int i = 0; i < NEL; ++i) sn += (double)S[i] * S[i];
        sn = sqrt(sn);
        double amax_tot = 0;
        for (int i = 0; i < NEL; ++i) amax_tot = std::max(amax_tot, fabs((double)S[i]));
        auto rms = [&](const std::vector<double> &out) {
            double e = 0;
            for (int i = 0; i < NEL; ++i) { double d = out[i] - (double)S[i]; e += d * d; }
            return 100.0 * sqrt(e) / sn;
        };
        // bf16 wire: round at every hop.
        std::vector<double> o(NEL);
        for (int i = 0; i < NEL; ++i) {
            float a = to_bf16((float)g[0][i]);
            for (int r = 1; r < P; ++r) a = to_bf16(a + to_bf16((float)g[r][i]));
            o[i] = a;
        }
        printf("    %5d  %-30s %6.1f %12.4f%% %13s\n", P, "bf16", 2.0, rms(o), "every hop");
        // fp8 e4m3 wire with a per-tensor scale: round at every hop, harder.
        double gmax = 0;
        for (int r = 0; r < P; ++r) for (int i = 0; i < NEL; ++i)
            gmax = std::max(gmax, fabs((double)g[r][i]));
        double s8 = std::max(gmax, amax_tot) / E4M3_MAX;
        for (int i = 0; i < NEL; ++i) {
            float a = quant_e4m3((float)(g[0][i] / s8));
            for (int r = 1; r < P; ++r) a = quant_e4m3(a + quant_e4m3((float)(g[r][i] / s8)));
            o[i] = (double)a * s8;
        }
        printf("    %5d  %-30s %6.1f %12.4f%% %13s\n", P, "fp8 e4m3, per-tensor scale", 1.0, rms(o), "every hop");
        // int16 fixed point, a-priori bound sized for the worst-case total.
        {
            double sc = fixed_scale(BOUND * P, 16);
            for (int i = 0; i < NEL; ++i) {
                long long a = 0;
                for (int r = 0; r < P; ++r) a += (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;                   // integer adds: exact
            }
            printf("    %5d  %-30s %6.1f %12.4f%% %13s\n", P,
                   "int16 fixed, a-priori bound", 2.0, rms(o), "none");
        }
        // int16 fixed point, normalization aware: one scalar all-reduce buys the
        // real amax of the TOTAL, so the grid is sized to the answer, not the bound.
        {
            double sc = fixed_scale(amax_tot, 16);
            for (int i = 0; i < NEL; ++i) {
                long long a = 0;
                for (int r = 0; r < P; ++r) a += (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;
            }
            printf("    %5d  %-30s %6.1f %12.4f%% %13s\n", P,
                   "int16 fixed, measured amax", 2.0, rms(o), "none");
        }
        printf("\n");
    }
    printf("    CORRECTED BY [13]. The 'hop rounding: none' column is true, but the\n");
    printf("    conclusion originally drawn from it was not. Fixed point does not stay\n");
    printf("    flat in P: read the int16 column above and it grows as sqrt(P), the\n");
    printf("    same rate as bf16, because every scheme quantizes each rank's own\n");
    printf("    contribution before it reaches the wire. The ratio between the two\n");
    printf("    columns NARROWS with P rather than widening. Run `just fair`.\n");
    printf("\n    What the gap actually is: bf16 spends 8 of its 15 magnitude bits on\n");
    printf("    an exponent this data does not use. Against a properly scaled fp16\n");
    printf("    wire at the same 2 bytes the advantage is ~7x, not ~39x. Closure buys\n");
    printf("    a guarantee -- reduction-order independence -- not an error factor.\n");
    printf("\n    The gap between the two int16 rows is the price of the a-priori\n");
    printf("    bound: 144*K*P is a worst case that random data never approaches, so\n");
    printf("    sizing the grid to it throws away real bits. One scalar all-reduce of\n");
    printf("    the true amax gets them back, which is the normalization-aware form.\n");
    // ---------------------------------------------------------------- part C
    printf("\n    C. roofline: one all-reduce of a 4096x4096 wgrad, ring, 2(P-1)/P bytes\n");
    const double NGRAD = 4096.0 * 4096.0;
    struct { const char *nm; double gbs; } links[] = {
        {"NVLink 4 (per GPU)", 450.0}, {"IB NDR 400G", 50.0}, {"PCIe 4.0 x16", 25.0}};
    printf("    %-22s %10s %10s %10s %10s\n", "link", "fp32 ms", "bf16 ms", "int16 ms", "saved");
    for (auto &L : links) {
        double f = 2.0 * (128 - 1) / 128.0;
        double t32 = NGRAD * 4 * f / (L.gbs * 1e9) * 1e3;
        double t16 = NGRAD * 2 * f / (L.gbs * 1e9) * 1e3;
        printf("    %-22s %10.3f %10.3f %10.3f %9.2fx\n", L.nm, t32, t16, t16, t32 / t16);
    }
    printf("    bf16 and int16 move the same bytes -- the difference between them is\n");
    printf("    not time, it is that one of them is exact under summation.\n");
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
// [13] fair -- where does the 39x actually come from?
// ============================================================================
//
// [11] reported int16 fixed point beating bf16 by 39x at 128 ranks and put it
// down to associativity: integer addition rounds once, floating point rounds at
// every hop. Reading the table in [11] back, that attribution does not survive.
// The int16 error goes 0.0080 -> 0.0177 -> 0.0329% across P = 8 -> 32 -> 128,
// which is sqrt(P), not flat. bf16 goes 0.4234 -> 0.7259 -> 1.2951, very nearly
// the same rate. The ratio between them NARROWS with P (53x, 41x, 39x) instead
// of widening. Whatever produces the 39x, it is not a difference in how the
// error accumulates with rank count.
//
// The reason both grow as sqrt(P) is that every scheme quantizes each rank's
// own contribution before it ever reaches the wire. P independent roundings go
// in, so P independent roundings come out, associativity or no associativity.
//
// So decompose it. Two independent claims are tangled together:
//
//   1. BITS. A 16-bit fixed-point field spends all 15 magnitude bits on the
//      mantissa. bf16 spends 8 of its 15 on an exponent that gradient data,
//      already scaled to a known range, does not use. That is a static ~7 bit
//      advantage available to ANY integer wire format, with or without FP4,
//      and it is not news -- it is why gradient-compression work going back to
//      1-bit SGD scales into a fixed grid. The honest 2-byte float control is
//      fp16 (11 mantissa bits), not bf16.
//
//   2. CLOSURE. A fixed-point grid is closed under addition: the encoding of a
//      sum IS the sum of the encodings, exactly, as long as the total stays
//      inside the field. So re-encoding the running sum at each hop costs
//      nothing. THIS is the part the FP4 144*d bound underwrites, and the way
//      to see it is to run one format both ways rather than two formats once.
//
// Every row below is re-encoded at every hop, as the payload of a real ring
// reduce-scatter is. That is the comparison [11] did not run.
static void exp_fair() {
    printf("\n[13] decomposing the int16-vs-bf16 gap: bits, or closure?\n");
    const int NEL = 4096;
    const int KLOC = 256;
    const int CH = 64;                     // block-scale chunk
    std::uniform_int_distribution<int> cd(0, 14);
    printf("\n    %d elements, depth-%d FP4 contraction per rank, ring reduce-scatter.\n", NEL, KLOC);
    printf("    Every format is re-encoded at every hop, as it would be on a real ring.\n");
    printf("\n    %5s  %-34s %6s %13s %11s\n", "P", "wire format", "b/el", "rel err (RMS)", "vs bf16");
    double keep[3][8];
    int pi = 0;
    for (int P : {8, 32, 128}) {
        std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i) {
                long long a = 0;
                for (int k = 0; k < KLOC; ++k)
                    a += (long long)e2m1_q(LEGAL[cd(rng)]) * e2m1_q(LEGAL[cd(rng)]);
                g[r][i] = a;
            }
        std::vector<long long> S(NEL, 0);
        for (int r = 0; r < P; ++r) for (int i = 0; i < NEL; ++i) S[i] += g[r][i];
        double sn = 0, amax_tot = 0;
        for (int i = 0; i < NEL; ++i) {
            sn += (double)S[i] * S[i];
            amax_tot = std::max(amax_tot, fabs((double)S[i]));
        }
        sn = sqrt(sn);
        std::vector<double> o(NEL);
        auto rms = [&](const std::vector<double> &v) {
            double e = 0;
            for (int i = 0; i < NEL; ++i) { double d = v[i] - (double)S[i]; e += d * d; }
            return 100.0 * sqrt(e) / sn;
        };
        // Per-chunk amax of the TOTAL. Reachable in practice with one small
        // max-all-reduce over NEL/CH scalars before the payload moves -- [14]
        // measures what that costs.
        std::vector<double> camax(NEL / CH, 0.0);
        for (int i = 0; i < NEL; ++i)
            camax[i / CH] = std::max(camax[i / CH], fabs((double)S[i]));
        int row = 0;
        auto emit = [&](const char *nm, double bpe, double err) {
            keep[pi][row++] = err;
            printf("    %5d  %-34s %6.2f %12.4f%% %10.1fx\n", P, nm, bpe, err,
                   err > 0 ? keep[pi][0] / err : 0.0);
        };
        // --- row 0: bf16, raw. The [11] baseline, reproduced.
        for (int i = 0; i < NEL; ++i) {
            float a = to_bf16((float)g[0][i]);
            for (int r = 1; r < P; ++r) a = to_bf16(a + to_bf16((float)g[r][i]));
            o[i] = a;
        }
        emit("bf16, raw (the [11] baseline)", 2.0, rms(o));
        // --- row 1: bf16 with a per-tensor scale. Control: a float is scale
        // invariant, so this MUST match row 0. If it does not, row 0 was a
        // scaling mistake rather than a format result.
        for (int i = 0; i < NEL; ++i) {
            double s = amax_tot;
            float a = to_bf16((float)(g[0][i] / s));
            for (int r = 1; r < P; ++r) a = to_bf16(a + to_bf16((float)(g[r][i] / s)));
            o[i] = (double)a * s;
        }
        emit("bf16, per-tensor scale (control)", 2.0, rms(o));
        // --- row 2: fp16, per-tensor scale. THE honest 2-byte float: same
        // bytes, 11 mantissa bits instead of 8, and the scale keeps every value
        // inside its narrower exponent range so the range never binds.
        for (int i = 0; i < NEL; ++i) {
            double s = amax_tot;
            float a = round_sig((float)(g[0][i] / s), 13);
            for (int r = 1; r < P; ++r)
                a = round_sig(a + round_sig((float)(g[r][i] / s), 13), 13);
            o[i] = (double)a * s;
        }
        emit("fp16, per-tensor scale", 2.0, rms(o));
        // --- row 3: int16 fixed, global scale, RE-ENCODED AT EVERY HOP.
        {
            double sc = fixed_scale(amax_tot, 16);
            for (int i = 0; i < NEL; ++i) {
                long long a = llrint((double)g[0][i] * sc);
                for (int r = 1; r < P; ++r) a = a + (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;
            }
            emit("int16 fixed, re-encode every hop", 2.0, rms(o));
        }
        // --- row 4: same grid, encoded ONCE at the source. If closure holds,
        // this is identical to row 3 -- that is the whole claim, and it is a
        // claim about the format, not about the data.
        {
            double sc = fixed_scale(amax_tot, 16);
            for (int i = 0; i < NEL; ++i) {
                long long a = 0;
                for (int r = 0; r < P; ++r) a += (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;
            }
            emit("int16 fixed, encode once", 2.0, rms(o));
        }
        // --- row 5: int16 on per-chunk grids. Still exact under summation,
        // because every rank uses the same grid for the same chunk.
        {
            for (int i = 0; i < NEL; ++i) {
                double sc = fixed_scale(std::max(camax[i / CH], 1.0), 16);
                long long a = 0;
                for (int r = 0; r < P; ++r) a += (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;
            }
            emit("int16, per-64 scale, exact", 2.0 + 2.0 / CH, rms(o));
        }
        // --- row 6: half the bytes, same construction.
        {
            for (int i = 0; i < NEL; ++i) {
                double sc = fixed_scale(std::max(camax[i / CH], 1.0), 8);
                long long a = 0;
                for (int r = 0; r < P; ++r) a += (long long)llrint((double)g[r][i] * sc);
                o[i] = (double)a / sc;
            }
            emit("int8, per-64 scale, exact", 1.0 + 2.0 / CH, rms(o));
        }
        printf("\n");
        ++pi;
    }
    printf("    Attribution at P=128, rows 0-5 all at ~2 bytes per element:\n");
    printf("      bf16 -> fp16          %7.1fx  3 more mantissa bits\n", keep[2][0] / keep[2][2]);
    printf("      fp16 -> int16 fixed   %7.1fx  4 more, by dropping the exponent field\n",
           keep[2][2] / keep[2][3]);
    printf("      re-encode every hop   %7.3fx  closure: rows 3 and 4 are the same number\n",
           keep[2][3] / keep[2][4]);
    printf("      global -> per-64      %7.1fx  fitting the grid to a smaller range\n",
           keep[2][4] / keep[2][5]);
    printf("\n    So the 39x is a BITS result, not a closure result: it is what you get\n");
    printf("    for spending all 15 magnitude bits on mantissa instead of 8. Against a\n");
    printf("    properly scaled fp16 wire -- the control [11] never ran -- the fixed\n");
    printf("    point advantage is the second line, and that is the number to quote.\n");
    printf("\n    Closure is still worth having, but it buys a GUARANTEE, not a factor:\n");
    printf("    re-encoding a fixed-point running sum at every hop changes nothing at\n");
    printf("    all, so the error is independent of topology, rank count and reduction\n");
    printf("    order. Two runs on differently shaped clusters return bitwise identical\n");
    printf("    gradients. No float wire offers that at any width.\n");
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
// A shared multi-step gradient model for [15]-[17]
// ============================================================================
//
// Experiments [6]-[14] all measured ONE reduction. Training does millions, and
// three of the ideas worth testing only make sense across steps: error feedback
// carries state from step to step, scale prediction reads the previous step,
// and the hierarchy question is about where in the topology the roundings
// happen. So they share a gradient model.
//
// Data-parallel SGD, per element i:
//
//   mu[i]        a persistent true gradient, log-normal in magnitude so the
//                tensor spans several decades -- this is the part that matters,
//                because it is the SMALL elements that a coarse grid rounds to
//                zero every single step, and they stay lost forever unless
//                something remembers them.
//   g[r][i]      rank r's contribution: mu[i]/P plus gradient noise, rounded to
//                an integer. The integer is not decoration: it is the FP4
//                lattice from [11], and it is what makes summation exact.
//
// The signal is persistent across steps and the noise is not, which is the
// whole reason SGD works and also the reason error feedback works.

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
    // One step. Fills g[r][i] with integers; returns the exact total in S.
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

// Per-chunk amax of a vector.
static void chunk_amax(const std::vector<long long> &S, int CH, std::vector<double> &am) {
    int NC = (int)am.size();
    for (int c = 0; c < NC; ++c) am[c] = 0.0;
    for (size_t i = 0; i < S.size(); ++i)
        am[i / CH] = std::max(am[i / CH], fabs((double)S[i]));
}

// Per-chunk max of anything that ever appears ON THE WIRE.
//
// This is not the same as the amax of the reduced total, and getting the two
// confused is a real bug that a first version of [15] shipped: the integer rows
// came out width-independent (int16 4.92%, int8 7.30% -- 1.5x apart when 8 bits
// should be 256x apart), which is the signature of a clipping floor rather than
// a quantization floor.
//
// The reason they differ is cancellation. In data-parallel SGD each rank's
// gradient is mostly NOISE around a much smaller mean, so |g_r| is routinely
// larger than |sum_r g_r|, and the running partial sum random-walks past the
// final answer on its way there. A ring carries prefix sums, so the field has
// to hold the largest prefix sum, not the largest result. Sizing the grid by
// the answer silently saturates the wire.
static void chunk_wiremax(const std::vector<std::vector<long long>> &g, int CH,
                          std::vector<double> &wm) {
    int P = (int)g.size(), NEL = (int)g[0].size();
    for (size_t c = 0; c < wm.size(); ++c) wm[c] = 0.0;
    for (int i = 0; i < NEL; ++i) {
        long long a = 0;
        double mx = 0;
        for (int r = 0; r < P; ++r) {
            a += g[r][i];
            mx = std::max(mx, fabs((double)a));
        }
        wm[i / CH] = std::max(wm[i / CH], mx);
    }
}

// ============================================================================
// [15] ef -- error feedback on a wire that is exact under summation
// ============================================================================
//
// [13] left a negative on the table: an int8 wire is FIVE TIMES WORSE than bf16
// at half the bytes, because per-rank quantization noise piles up over sqrt(P)
// ranks. That is the thing standing between this idea and a 1-byte gradient.
//
// Error feedback is the standard fix and it goes back to 1-bit SGD: do not
// throw the rounding residual away, keep it in a local buffer and add it to
// next step's gradient before quantizing. Anything too small to send this step
// accumulates until it IS big enough, so no gradient component is permanently
// lost -- only delayed.
//
// The reason it belongs in THIS repo rather than being a citation is that the
// residual here is EXACT. On a fixed-point grid, r = v - q/s is computed with
// no error of its own, because both terms are integers on the same lattice.
// On a float wire the residual is itself a rounded quantity, so the correction
// you carry forward is already wrong. Closure is what makes the bookkeeping
// exact, and error feedback is what turns that from a guarantee into a factor.
//
// The metric has to be the ACCUMULATED error, not the per-step error. SGD
// integrates gradients; what hurts is a bias that lands the same way every
// step, not noise that averages out. So we measure the error in sum_t S_t.
//
// Stochastic rounding is in the table because it is the cheap competitor -- it
// also removes the bias, with no residual buffer at all -- and a reviewer will
// ask.

enum RndMode { R_DET, R_SR, R_EF };

static void exp_ef() {
    printf("\n[15] error feedback: making a 1-byte gradient wire usable\n");

    const int NEL = 1024, P = 32, CH = 64, TMAX = 128;
    const int NC = NEL / CH;
    GradModel gm(NEL, P);

    printf("\n    %d elements, %d ranks, per-chunk scales over %d chunks of %d.\n",
           NEL, P, NC, CH);
    printf("    Metric is the error in the ACCUMULATED gradient sum_t S_t, which is\n");
    printf("    what the optimizer actually integrates. Per-step error is reported\n");
    printf("    beside it so the difference between the two is visible.\n");

    struct Method { int W; RndMode m; const char *nm; };
    std::vector<Method> ms = {
        {16, R_DET, "bf16 (float baseline)"},        // W ignored, flagged below
        {16, R_DET, "int16, round-to-nearest"},
        {16, R_SR,  "int16, stochastic rounding"},
        {16, R_EF,  "int16, error feedback"},
        {8,  R_DET, "int8,  round-to-nearest"},
        {8,  R_SR,  "int8,  stochastic rounding"},
        {8,  R_EF,  "int8,  error feedback"},
        {4,  R_DET, "int4,  round-to-nearest"},
        {4,  R_SR,  "int4,  stochastic rounding"},
        {4,  R_EF,  "int4,  error feedback"},
    };
    const int NM = (int)ms.size();

    // Per-method persistent state: the residual buffer, one per rank per element.
    std::vector<std::vector<double>> resid(NM, std::vector<double>((size_t)P * NEL, 0.0));
    std::vector<std::vector<double>> accum(NM, std::vector<double>(NEL, 0.0));
    std::vector<double> exact_accum(NEL, 0.0);
    std::vector<double> step_err(NM, 0.0);
    int step_n = 0;

    std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
    std::vector<long long> S(NEL);
    std::vector<double> am(NC), wm(NC);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    double head_sum = 0;

    printf("\n    %-30s %6s %14s %16s %11s\n", "wire format", "b/el", "per-step err",
           "accumulated err", "drift/step");

    for (int t = 1; t <= TMAX; ++t) {
        gm.step(g, S);
        chunk_amax(S, CH, am);
        chunk_wiremax(g, CH, wm);
        for (int c = 0; c < NC; ++c) head_sum += am[c] > 0 ? wm[c] / am[c] : 1.0;
        for (int i = 0; i < NEL; ++i) exact_accum[i] += (double)S[i];
        ++step_n;

        for (int k = 0; k < NM; ++k) {
            const Method &M = ms[k];
            double se = 0, sn = 0;
            for (int c = 0; c < NC; ++c) {
                // Sized by the largest PREFIX SUM in the chunk, not by the
                // largest result, plus 5% so error feedback's residual cannot
                // push the last hop over the edge. See chunk_wiremax.
                double lim = (double)((1 << (M.W - 1)) - 1);
                double s = wm[c] > 0 ? lim / (wm[c] * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int r = 0; r < P; ++r) {
                        if (k == 0) {                       // bf16 float wire
                            tot += (double)to_bf16((float)g[r][i]);
                            continue;
                        }
                        size_t ri = (size_t)r * NEL + i;
                        double v = (double)g[r][i] + (M.m == R_EF ? resid[k][ri] : 0.0);
                        double x = v * s;
                        double q;
                        if (M.m == R_SR) q = floor(x + ur(rng));
                        else q = (double)llrint(x);
                        if (q > lim) q = lim;
                        if (q < -lim) q = -lim;
                        if (M.m == R_EF) resid[k][ri] = v - q / s;
                        tot += q / s;
                    }
                    accum[k][i] += tot;
                    double d = tot - (double)S[i];
                    se += d * d;
                    sn += (double)S[i] * (double)S[i];
                }
            }
            step_err[k] += 100.0 * sqrt(se) / sqrt(sn);
        }

        if (t == 8 || t == 32 || t == 128) {
            printf("\n    after %d steps:\n", t);
            for (int k = 0; k < NM; ++k) {
                double e = 0, n = 0;
                for (int i = 0; i < NEL; ++i) {
                    double d = accum[k][i] - exact_accum[i];
                    e += d * d;
                    n += exact_accum[i] * exact_accum[i];
                }
                double bpe = k == 0 ? 2.0 : ms[k].W / 8.0 + 2.0 / CH;
                double rel = sqrt(e) / sqrt(n);
                // Accumulated error expressed in units of ONE average step's
                // contribution: "how many steps of gradient have been lost".
                // The plain relative column shrinks with T for every method
                // simply because the denominator grows linearly, which hides
                // the thing being measured.
                printf("    %-30s %6.3f %13.4f%% %15.4f%% %11.3f\n", ms[k].nm, bpe,
                       step_err[k] / step_n, 100.0 * rel, rel * t);
            }
        }
    }
    printf("\n    Wire headroom actually needed: %.2fx the amax of the reduced\n",
           head_sum / (step_n * NC));
    printf("    result, because prefix sums overshoot the answer they converge to.\n");

    printf("\n    The drift column is the one that matters, and it is the only one\n");
    printf("    that is not misleading. Plain relative accumulated error SHRINKS with\n");
    printf("    T for every method, because the denominator grows linearly while the\n");
    printf("    error does not -- so it hides exactly what it is meant to show. Drift\n");
    printf("    divides that out and reads as steps of gradient lost.\n");
    printf("\n    Across 8 -> 32 -> 128 steps:\n");
    printf("      int16 round-to-nearest    0.002 -> 0.034 -> 0.122     growing\n");
    printf("      int16 stochastic          0.003 -> 0.035 -> 0.122     growing\n");
    printf("      int16 error feedback      0.001 -> 0.001 -> 0.001     FLAT\n");
    printf("      int8  round-to-nearest    0.257 -> 0.867 -> 2.976     growing fast\n");
    printf("      int8  error feedback      0.095 -> 0.110 -> 0.135     nearly flat\n");
    printf("      bf16                      0.006 -> 0.030 -> 0.039     growing\n");
    printf("\n    Error feedback does not make any single step more accurate -- look at\n");
    printf("    the per-step column, where it is WORSE, because last step's residual\n");
    printf("    is injected as extra noise. What it does is stop the errors adding\n");
    printf("    up. Round-to-nearest is biased: a small persistent gradient component\n");
    printf("    rounds the same way every step and never arrives at all. Stochastic\n");
    printf("    rounding removes the bias but random-walks as sqrt(T). Feedback\n");
    printf("    bounds the total by whatever is sitting in the residual buffers,\n");
    printf("    which does not grow with T.\n");
    printf("\n    Honest reading of the int8 row: [13] found int8 five times WORSE\n");
    printf("    than bf16 at half the bytes, and that was the wall. Feedback moves it\n");
    printf("    to within ~3.5x of bf16 on accumulated error at 1.03 bytes -- viable,\n");
    printf("    not superior. The claim that survives is at EQUAL bytes: int16+EF at\n");
    printf("    2.03 B/el drifts 0.001 against bf16's 0.039, roughly 39x, and stays\n");
    printf("    there while bf16 keeps climbing.\n");
    printf("\n    (128 steps is not 100k. The trends -- one flat, one sqrt(T), one\n");
    printf("    linear -- are visible and have the right shapes, but extrapolating\n");
    printf("    them to a real run is extrapolation, and a convergence run is the\n");
    printf("    only thing that settles it.)\n");
    printf("\n    Why this belongs here and is not just a citation: on a fixed-point\n");
    printf("    grid the residual v - q/s is EXACT, both terms being integers on the\n");
    printf("    same lattice. On a float wire the correction you carry forward has\n");
    printf("    itself been rounded. Closure is what makes the bookkeeping exact.\n");
}

// ============================================================================
// [16] predict -- removing the scale all-reduce from the critical path
// ============================================================================
//
// [14] priced the block-scaled design and found the bytes negligible and the
// LATENCY not: the scales have to be agreed before the payload can be encoded,
// so it is two dependent collectives where bf16 needs one. At 1024 ranks that
// is tens of microseconds on the critical path of every step.
//
// The way out is that gradient statistics are not adversarial. The amax of a
// given chunk moves slowly from step to step, so last step's amax times a
// safety margin is a usable grid for this step, and it needs no communication
// at all. The cost is the margin -- log2(margin) bits straight off the
// mantissa -- and the risk is that a step where the gradient jumps saturates
// the field.
//
// The interesting part is what happens on saturation. A clipped value leaves a
// residual, and [15] built the machine that carries residuals forward. So the
// exception path is not an exception path: prediction overflow and quantization
// error are the same quantity, and error feedback already handles it. That is
// the thing worth testing -- whether the two ideas compose or whether the
// clipped mass is too big for the buffer to absorb.
//
// The drift model: sigma follows a log random walk with occasional spikes,
// which is a caricature of a real loss curve, but the caricature is the point --
// it is deliberately harder than a smooth run.

static void exp_predict() {
    printf("\n[16] predicting the scale: one collective instead of two\n");

    const int NEL = 1024, P = 32, CH = 64, TMAX = 128, W = 16;
    const int NC = NEL / CH;
    const double lim = (double)((1 << (W - 1)) - 1);

    printf("\n    %d elements, %d ranks, int%d wire, %d chunks of %d, %d steps.\n",
           NEL, P, W, NC, CH, TMAX);
    printf("    Gradient scale drifts as a log random walk, sigma 5%% per step,\n");
    printf("    with a 2%% chance per step of a 2.5x spike.\n");

    struct Cfg { double margin; bool ef; const char *nm; };
    std::vector<Cfg> cfgs = {
        {0.0,  false, "oracle amax (2 collectives)"},
        {0.0,  true,  "oracle amax + EF (2 collectives)"},
        {1.0,  false, "predicted, margin 1.00"},
        {1.25, false, "predicted, margin 1.25"},
        {2.0,  false, "predicted, margin 2.00"},
        {1.0,  true,  "predicted, margin 1.00 + EF"},
        {1.25, true,  "predicted, margin 1.25 + EF"},
        {2.0,  true,  "predicted, margin 2.00 + EF"},
    };
    const int NCF = (int)cfgs.size();

    std::vector<std::vector<double>> resid(NCF, std::vector<double>((size_t)P * NEL, 0.0));
    std::vector<std::vector<double>> accum(NCF, std::vector<double>(NEL, 0.0));
    std::vector<std::vector<double>> prev(NCF, std::vector<double>(NC, 0.0));
    std::vector<double> exact_accum(NEL, 0.0);
    std::vector<double> sperr(NCF, 0.0);
    std::vector<double> clipped(NCF, 0.0);
    double nvals = 0;

    GradModel gm(NEL, P);
    std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
    std::vector<long long> S(NEL);
    std::vector<double> am(NC), wm(NC);
    std::normal_distribution<double> walk(0.0, 0.05);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    double drift = 1.0;

    for (int t = 1; t <= TMAX; ++t) {
        drift *= exp(walk(rng));
        if (ur(rng) < 0.02) drift *= 2.5;
        gm.step(g, S, drift);
        chunk_amax(S, CH, am);
        chunk_wiremax(g, CH, wm);
        for (int i = 0; i < NEL; ++i) exact_accum[i] += (double)S[i];
        nvals += (double)NEL * P;

        for (int k = 0; k < NCF; ++k) {
            const Cfg &C = cfgs[k];
            double se = 0, sn = 0;
            for (int c = 0; c < NC; ++c) {
                // The oracle needs this step's amax, which costs a collective.
                // Everything else uses last step's, which costs nothing.
                double use = C.margin == 0.0 ? wm[c]
                           : (prev[k][c] > 0 ? prev[k][c] * C.margin : wm[c]);
                double s = use > 0 ? lim / (use * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int r = 0; r < P; ++r) {
                        size_t ri = (size_t)r * NEL + i;
                        double v = (double)g[r][i] + (C.ef ? resid[k][ri] : 0.0);
                        double q = (double)llrint(v * s);
                        if (q > lim) { q = lim; clipped[k] += 1.0; }
                        if (q < -lim) { q = -lim; clipped[k] += 1.0; }
                        if (C.ef) resid[k][ri] = v - q / s;
                        tot += q / s;
                    }
                    accum[k][i] += tot;
                    double d = tot - (double)S[i];
                    se += d * d;
                    sn += (double)S[i] * (double)S[i];
                }
                prev[k][c] = wm[c];              // measured locally after the fact
            }
            sperr[k] += 100.0 * sqrt(se) / sqrt(sn);
        }
    }

    printf("\n    %-32s %10s %13s %16s\n", "scale source", "clip rate", "per-step err", "accumulated err");
    for (int k = 0; k < NCF; ++k) {
        double e = 0, n = 0;
        for (int i = 0; i < NEL; ++i) {
            double d = accum[k][i] - exact_accum[i];
            e += d * d;
            n += exact_accum[i] * exact_accum[i];
        }
        printf("    %-32s %9.4f%% %12.4f%% %15.4f%%\n", cfgs[k].nm,
               100.0 * clipped[k] / nvals, sperr[k] / TMAX, 100.0 * sqrt(e) / sqrt(n));
    }

    printf("\n    The oracle rows are what [14] priced: correct scale, two dependent\n");
    printf("    collectives, full latency. The predicted rows use the PREVIOUS step's\n");
    printf("    wire max and send nothing extra at all.\n");
    printf("\n    The comparison that matters is oracle+EF against predicted+EF,\n");
    printf("    because EF alone is worth more than the scale is. Oracle+EF 0.0064%%\n");
    printf("    against predicted margin 1.00 +EF 0.0112%%: dropping an entire\n");
    printf("    collective off the critical path of every step costs 1.75x on\n");
    printf("    accumulated error. That is the whole trade, and it looks worth it.\n");
    printf("\n    Without feedback the margin is a real dilemma -- too small and the\n");
    printf("    spikes clip (margin 1.00 is 4.3x worse than the oracle), too large\n");
    printf("    and you have spent bits on headroom the data never uses (margin 2.00\n");
    printf("    is one whole bit). With feedback the dilemma goes away: 0.0112,\n");
    printf("    0.0130, 0.0189 across the three margins, and the TIGHTEST one wins.\n");
    printf("    Do not buy headroom; let the residual buffer absorb the overflow.\n");
    printf("\n    That is the point of this experiment. A clipped value leaves a\n");
    printf("    residual, and the residual buffer already exists to carry residuals\n");
    printf("    forward. Overflow and rounding are the same quantity, so the\n");
    printf("    exception path IS error feedback -- there is no second mechanism to\n");
    printf("    build, and a mispredicted scale costs a one-step delay rather than a\n");
    printf("    lost gradient.\n");
    printf("\n    One thing worth noticing in the no-EF rows: the ORACLE has a worse\n");
    printf("    accumulated error than margin 2.00 (0.1243%% against 0.0385%%) despite\n");
    printf("    a strictly better grid and a better per-step error. A tight grid\n");
    printf("    clips occasionally, and clipping is BIASED -- always toward zero --\n");
    printf("    so it accumulates linearly while rounding error does not. Picking\n");
    printf("    the scale to minimise per-step error is the wrong objective.\n");
}

// ============================================================================
// [17] tiers -- put the roundings where the bandwidth is
// ============================================================================
//
// Every experiment so far has treated the P ranks as one flat ring. Real
// clusters are not flat: 8 GPUs inside a node talk over NVLink at ~450 GB/s,
// and nodes talk to each other over InfiniBand at ~50 GB/s. Nearly 10x.
//
// Hierarchical all-reduce is standard practice for the bandwidth reason alone
// (reduce-scatter in the node, all-reduce across nodes, all-gather in the
// node). What is specific to a fixed-point wire is the NUMERICS of doing it
// that way. Intra-node the payload is small and the link is fast, so the
// intra-node stage can stay in int32 and be EXACTLY zero-error. Only the
// inter-node stage needs a narrow wire.
//
// That drops the number of quantization events from P to G, the node count.
// Since per-rank rounding errors add in quadrature, the error should fall by
// sqrt(L) where L is GPUs per node -- a free 2.8x at L=8 that costs nothing
// and composes with everything in [15] and [16].
//
// A float wire cannot do this. Its intra-node stage rounds too.

static void exp_tiers() {
    printf("\n[17] hierarchical reduction: where the roundings go, and what they cost\n");

    const int NEL = 2048, CH = 64;
    const int NC = NEL / CH;

    printf("\n    %d elements, per-chunk scales, error against an exact integer sum.\n", NEL);
    printf("    flat = one ring over all P ranks. hier = ring inside the node, then\n");
    printf("    ring across nodes, then broadcast back -- standard practice.\n");
    printf("\n    %6s %5s %6s  %-36s %13s %11s\n",
           "nodes", "gpus", "P", "scheme", "rel err (RMS)", "vs flat");

    struct Shape { int G, L; };
    for (Shape sh : {Shape{16, 8}, Shape{128, 8}}) {
        int G = sh.G, L = sh.L, P = G * L;
        GradModel gm(NEL, P);
        std::vector<std::vector<long long>> g(P, std::vector<long long>(NEL));
        std::vector<long long> S(NEL);
        std::vector<double> am(NC), wm(NC), wmn(NC);
        gm.step(g, S);
        chunk_amax(S, CH, am);
        chunk_wiremax(g, CH, wm);            // prefix maxima over P ranks
        double sn = 0;
        for (int i = 0; i < NEL; ++i) sn += (double)S[i] * (double)S[i];
        sn = sqrt(sn);

        // Node-level exact partial sums: pure integer addition, no error.
        std::vector<std::vector<long long>> ng(G, std::vector<long long>(NEL, 0));
        for (int r = 0; r < P; ++r)
            for (int i = 0; i < NEL; ++i) ng[r / L][i] += g[r][i];
        chunk_wiremax(ng, CH, wmn);          // prefix maxima over G nodes

        double base_f = 0, once_err = -1, flat_err = -1;
        double maxdiff = 0;

        auto report = [&](const char *nm, double err, double ref) {
            printf("    %6d %5d %6d  %-36s %12.5f%% %10.2fx\n",
                   G, L, P, nm, err, ref > 0 ? ref / err : 1.0);
        };

        // --- 1. flat bf16: a ring over P ranks, rounding at every hop.
        {
            double e = 0;
            for (int i = 0; i < NEL; ++i) {
                float a = to_bf16((float)g[0][i]);
                for (int r = 1; r < P; ++r) a = to_bf16(a + to_bf16((float)g[r][i]));
                double d = (double)a - (double)S[i];
                e += d * d;
            }
            base_f = 100.0 * sqrt(e) / sn;
            report("flat bf16", base_f, 0);
        }

        // --- 2. hier bf16: fewer hops, but it still rounds at every one of
        // them, INCLUDING when a node result crosses to the inter-node ring.
        {
            double e = 0;
            for (int i = 0; i < NEL; ++i) {
                std::vector<float> nv(G);
                for (int n = 0; n < G; ++n) {
                    float a = to_bf16((float)g[n * L][i]);
                    for (int l = 1; l < L; ++l)
                        a = to_bf16(a + to_bf16((float)g[n * L + l][i]));
                    nv[n] = a;
                }
                float b = nv[0];
                for (int n = 1; n < G; ++n) b = to_bf16(b + nv[n]);
                double d = (double)b - (double)S[i];
                e += d * d;
            }
            report("hier bf16", 100.0 * sqrt(e) / sn, base_f);
        }

        // --- 3. flat int16: every rank quantizes onto a shared grid, then the
        // ring adds integers.
        {
            double lim = 32767.0, e = 0;
            for (int c = 0; c < NC; ++c) {
                double s = wm[c] > 0 ? lim / (wm[c] * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int r = 0; r < P; ++r) {
                        double q = (double)llrint((double)g[r][i] * s);
                        tot += std::max(-lim, std::min(lim, q)) / s;
                    }
                    double d = tot - (double)S[i];
                    e += d * d;
                }
            }
            flat_err = 100.0 * sqrt(e) / sn;
            report("flat int16", flat_err, base_f);
        }

        // --- 4. hier int16, QUANTIZED ONCE AT THE SOURCE. The intra-node ring
        // and the inter-node ring both add integers on the same grid, so
        // neither one rounds. This is the design closure actually buys, and it
        // must come out bit-identical to row 3 -- the topology has stopped
        // being a numerical parameter at all.
        {
            double lim = 32767.0, e = 0;
            for (int c = 0; c < NC; ++c) {
                double s = wm[c] > 0 ? lim / (wm[c] * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int n = 0; n < G; ++n) {
                        double nodesum = 0;
                        for (int l = 0; l < L; ++l) {
                            double q = (double)llrint((double)g[n * L + l][i] * s);
                            nodesum += std::max(-lim, std::min(lim, q));
                        }
                        tot += nodesum / s;
                    }
                    double d = tot - (double)S[i];
                    e += d * d;
                    maxdiff = std::max(maxdiff, fabs(tot - (double)S[i]));
                }
            }
            once_err = 100.0 * sqrt(e) / sn;
            report("hier int16, quantize once at source", once_err, base_f);
        }

        // --- 5. hier with a REQUANTIZATION at the node boundary: the intra
        // stage carries exact integers, then each node result is re-encoded
        // once for the slow link. G roundings instead of P.
        {
            double lim = 32767.0, e = 0;
            for (int c = 0; c < NC; ++c) {
                double s = wmn[c] > 0 ? lim / (wmn[c] * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int n = 0; n < G; ++n) {
                        double q = (double)llrint((double)ng[n][i] * s);
                        tot += std::max(-lim, std::min(lim, q)) / s;
                    }
                    double d = tot - (double)S[i];
                    e += d * d;
                }
            }
            report("hier, exact intra + int16 inter", 100.0 * sqrt(e) / sn, base_f);
        }

        // --- 6. the same thing with a 1-byte slow link.
        {
            double lim = 127.0, e = 0;
            for (int c = 0; c < NC; ++c) {
                double s = wmn[c] > 0 ? lim / (wmn[c] * 1.05) : 1.0;
                for (int j = 0; j < CH; ++j) {
                    int i = c * CH + j;
                    double tot = 0;
                    for (int n = 0; n < G; ++n) {
                        double q = (double)llrint((double)ng[n][i] * s);
                        tot += std::max(-lim, std::min(lim, q)) / s;
                    }
                    double d = tot - (double)S[i];
                    e += d * d;
                }
            }
            report("hier, exact intra + int8 inter", 100.0 * sqrt(e) / sn, base_f);
        }

        printf("    %6s %5s %6s  rows 3 and 4 differ by %.3e -- %s\n", "", "", "",
               fabs(flat_err - once_err),
               fabs(flat_err - once_err) < 1e-12 ? "IDENTICAL, as closure requires"
                                                : "MISMATCH");
        printf("\n");
    }

    printf("    Row 4 is the result. Quantize once at the source and every sum\n");
    printf("    afterwards is exact integer addition, so a two-level hierarchy\n");
    printf("    gives BIT-IDENTICAL output to a flat ring -- the same number, not a\n");
    printf("    close one. Topology stops being a numerical parameter. Compare row 2:\n");
    printf("    a bf16 hierarchy has to round when a node result crosses to the\n");
    printf("    inter-node ring, so changing the cluster shape changes the answer.\n");
    printf("\n    Rows 5 and 6 buy accuracy instead of reproducibility. Reducing the\n");
    printf("    node exactly and quantizing once PER NODE means G roundings instead\n");
    printf("    of P, and errors add in quadrature, so the gain is sqrt(L) = %.2f at\n", sqrt(8.0));
    printf("    8 GPUs per node. int16 delivers it. int8 gets only ~1.95x, because\n");
    printf("    at 7 bits the per-chunk dynamic range starts binding before the\n");
    printf("    rounding count does -- the [14] effect showing up again.\n");
    printf("\n    But this is NOT free, and the time model below is what says so.\n");

    // ------------------------------------------------------------------ time
    const double NP = 7.0e9, NVL = 450e9, IB = 50e9;
    const int G = 128, L = 8, P = G * L;
    printf("\n    Time model, one 7B gradient all-reduce, %d nodes x %d GPUs,\n", G, L);
    printf("    NVLink %.0f GB/s intra, IB %.0f GB/s inter:\n", NVL / 1e9, IB / 1e9);
    printf("\n    %-38s %9s %9s %9s %11s\n",
           "scheme", "intra ms", "inter ms", "total ms", "roundings");
    struct TR { const char *nm; double b_in, b_out; bool hier; const char *rnd; };
    for (TR tr : {TR{"flat ring, bf16", 0, 2.0, false, "every hop"},
                  TR{"flat ring, int16", 0, 2.0, false, "P"},
                  TR{"hier, bf16", 2.0, 2.0, true, "every hop"},
                  TR{"hier, int16 quantize-once", 2.03, 2.03, true, "P"},
                  TR{"hier, exact(int32) intra + int16", 4.0, 2.03, true, "G"},
                  TR{"hier, int16 intra + int8 inter+EF", 2.03, 1.03, true, "P + G"}}) {
        double intra = 0, inter = 0;
        if (tr.hier) {
            intra = (2.0 * (L - 1) / L) * NP * tr.b_in / NVL * 1e3;
            inter = (2.0 * (G - 1) / G) * (NP / L) * tr.b_out / IB * 1e3;
        } else {
            inter = (2.0 * (P - 1) / P) * NP * tr.b_out / IB * 1e3;
        }
        printf("    %-38s %9.2f %9.2f %9.2f %11s\n",
               tr.nm, intra, inter, intra + inter, tr.rnd);
    }

    printf("\n    Read this against the error table and the ranking changes.\n");
    printf("\n    The hierarchy itself is worth 4.5x and has nothing to do with this\n");
    printf("    repo -- it divides the inter-node payload by L before it reaches the\n");
    printf("    slow link, which is why everyone already does it. What IS ours is\n");
    printf("    that row 4 costs 1.5%% more bytes than the bf16 hierarchy -- 126 ms\n");
    printf("    against 124, the difference being the scale bytes -- and returns\n");
    printf("    bit-identical results to a flat ring while doing it.\n");
    printf("\n    The sqrt(L) accuracy row is a trap and the time model catches it:\n");
    printf("    carrying int32 intra-node doubles the fast-link traffic and lands at\n");
    printf("    179 ms against the bf16 hierarchy's 125 ms. Paying 43%% more wall\n");
    printf("    clock for 2.8x accuracy you did not ask for is a bad trade. The\n");
    printf("    accuracy was never free -- it was on the NVLink bill.\n");
    printf("\n    The last row is where the stack pays off: int16 on the fast link,\n");
    printf("    int8 plus [15]'s error feedback on the slow one, 91 ms against the\n");
    printf("    bf16 hierarchy's 124 ms at comparable accuracy on the accumulated\n");
    printf("    metric. 1.36x on the collective, and 6.1x against a flat bf16 ring.\n");
    printf("\n    Caveat on that last row: [15] measured int8+EF over 128 steps at 32\n");
    printf("    ranks, not over a real run at 1024. It is the row most worth\n");
    printf("    building for real and the one least entitled to be quoted yet.\n");
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
    if (want("nvfp4")) exp_nvfp4();
    if (want("mxfp8")) exp_mxfp8();
    if (want("interleave")) exp_interleave();
    if (want("link")) exp_link();
    if (want("dense")) exp_dense();
    if (want("fair")) exp_fair();
    if (want("llm")) exp_llm();
    if (want("ef")) exp_ef();
    if (want("predict")) exp_predict();
    if (want("tiers")) exp_tiers();
    printf("\n");
    return 0;
}
