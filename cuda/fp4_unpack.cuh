// FP4 (E2M1) -> FP8 (E4M3) expansion.
//
// The cast is lossless: every E2M1 value {0, .5, 1, 1.5, 2, 3, 4, 6} is exactly
// representable in E4M3, so this is a 16-entry table lookup, not a rounding step.
//
//   code 0..7  ->  0x00 0x30 0x38 0x3C 0x40 0x44 0x48 0x4C   (bit 3 = sign)
//
// Both halves of the table fit in two 32-bit registers, so PRMT does four
// lookups in one instruction. Signs come from a second PRMT against a two-entry
// {0x00, 0x80} table. Six instructions per four values.
#pragma once
#include <stdint.h>

#define FP4_LUT_LO 0x3C383000u   // bytes 0..3 -> codes 0,1,2,3
#define FP4_LUT_HI 0x4C484440u   // bytes 4..7 -> codes 4,5,6,7
#define FP4_SGN_LUT 0x00008000u  // byte 0 = 0x00, byte 1 = 0x80

#if defined(__CUDACC__)
__device__ __forceinline__ uint32_t fp4_prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
    return d;
}
#define FP4_HD __device__ __forceinline__
#define FP4_BOTH __host__ __device__ __forceinline__
#else
// Host/CPU model of PTX prmt (default mode) so the logic can be unit tested.
static inline uint32_t fp4_prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint8_t src[8] = {(uint8_t)(a), (uint8_t)(a >> 8), (uint8_t)(a >> 16), (uint8_t)(a >> 24),
                      (uint8_t)(b), (uint8_t)(b >> 8), (uint8_t)(b >> 16), (uint8_t)(b >> 24)};
    uint32_t d = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned sel = (s >> (4 * i)) & 0xF;
        uint8_t v = src[sel & 7];
        if (sel & 8) v = (v & 0x80) ? 0xFF : 0x00;   // sign-replicate mode
        d |= (uint32_t)v << (8 * i);
    }
    return d;
}
#define FP4_HD static inline
#define FP4_BOTH static inline
#endif

// 4 packed FP4 codes (low 16 bits, low nibble first) -> 4 E4M3 bytes.
FP4_HD uint32_t fp4x4_to_e4m3x4(uint32_t p) {
    uint32_t mag = fp4_prmt(FP4_LUT_LO, FP4_LUT_HI, p & 0x7777u);
    uint32_t sgn = fp4_prmt(FP4_SGN_LUT, 0u, (p >> 3) & 0x1111u);
    return mag | sgn;
}

// 8 packed FP4 codes (one uint32 of packed bytes) -> 8 E4M3 bytes.
FP4_HD void fp4x8_to_e4m3x8(uint32_t p, uint32_t *lo, uint32_t *hi) {
    *lo = fp4x4_to_e4m3x4(p & 0xFFFFu);
    *hi = fp4x4_to_e4m3x4(p >> 16);
}

// Reference table, for tests and for host-side packing.
FP4_BOTH uint8_t fp4_code_to_e4m3_ref(uint8_t code) {
    const uint8_t t[8] = {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C};
    return (uint8_t)(t[code & 7] | ((code & 8) ? 0x80 : 0x00));
}

// E2M1 code -> value, and value -> 2*value on the exact integer grid.
FP4_BOTH float fp4_code_to_float(uint8_t code) {
    const float m[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = m[code & 7];
    return (code & 8) ? -v : v;
}
