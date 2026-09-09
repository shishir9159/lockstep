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
    printf("\n");
    return 0;
}
