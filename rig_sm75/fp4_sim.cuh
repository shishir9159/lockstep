// FP4 (E2M1) grid, FP4 -> FP8 expansion and a narrow-accumulator model.
// Plain CUDA-core code, sm_75 and up.
#pragma once
#include <stdint.h>

// --------------------------------------------------------------- E2M1 grid
// FP4 values {0, +-0.5, 1, 1.5, 2, 3, 4, 6} times two, so every value is an
// exact small integer and every dot product is an exact integer.
//   q in {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12},  |q1*q2| <= 144
#define FP4_QMAX 12
#define FP4_PMAX 144

__host__ __device__ inline int e2m1_q(int code) {
    const int t[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    int v = t[code & 7];
    return (code & 8) ? -v : v;
}

__host__ __device__ inline float e2m1_value(int code) {
    const float m[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = m[code & 7];
    return (code & 8) ? -v : v;
}

// E2M1 code -> E4M3 byte. Lossless: every E2M1 value is exactly an E4M3 value.
__host__ __device__ inline uint8_t e2m1_to_e4m3(uint8_t code) {
    const uint8_t t[8] = {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C};
    return (uint8_t)(t[code & 7] | ((code & 8) ? 0x80 : 0x00));
}

// PRMT-based 4-at-a-time expansion. prmt.b32 exists on sm_75, so this is the
// same instruction sequence the Hopper kernel uses.
#define FP4_LUT_LO 0x3C383000u
#define FP4_LUT_HI 0x4C484440u
#define FP4_SGN_LUT 0x00008000u

__device__ inline uint32_t fp4_prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}

__device__ inline uint32_t fp4x4_to_e4m3x4(uint32_t p) {
    uint32_t mag = fp4_prmt(FP4_LUT_LO, FP4_LUT_HI, p & 0x7777u);
    uint32_t sgn = fp4_prmt(FP4_SGN_LUT, 0u, (p >> 3) & 0x1111u);
    return mag | sgn;
}

// ------------------------------------------------- narrow accumulator model
// Round x to W significand bits. fp32 has 24, so pass drop = 24 - W.
//
//   W = 24  true fp32 accumulator            (drop 0)
//   W = 14  the Hopper FP8 MMA datapath      (drop 10)
//
// Adding `half` to the raw bit pattern and masking is round-half-away-from-zero
// in sign-magnitude, and a mantissa carry correctly increments the exponent.
__host__ __device__ inline float round_sig(float x, int drop) {
    if (drop <= 0) return x;
    union { float f; uint32_t u; } c;
    c.f = x;
    uint32_t half = 1u << (drop - 1);
    uint32_t mask = ~((1u << drop) - 1u);
    c.u = (c.u + half) & mask;
    return c.f;
}

// Minimum slot offset for a K-deep FP4 dot product: bits(144K) + 1 guard.
__host__ __device__ inline int slot_offset(int K) {
    long long b = (long long)FP4_PMAX * K;
    int n = 0;
    while (b > 0) { b >>= 1; ++n; }
    return n + 1;
}

// Bits an accumulator needs to hold [C1 : s][C2 : s] for a K-deep problem.
__host__ __device__ inline int packed_bits_needed(int K) {
    return 2 * slot_offset(K) - 1;
}
