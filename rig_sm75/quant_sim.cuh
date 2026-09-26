// Block-scaled quantization formats in software, host + device, sm_75 and up.
//
//   MXFP4   block 32, E8M0 scale (power of two), E2M1 elements   4.25 bit/elem
//   NVFP4   block 16, E4M3 scale (has a mantissa), E2M1 elements  4.50 bit/elem
//   MXFP8   block 32, E8M0 scale, E4M3 elements                   8.25 bit/elem
//
// E8M0 can only be a power of two, so it can waste up to 2x of a block's range;
// E4M3 lands close to amax/6.
#pragma once
#include <stdint.h>
#include <math.h>
#include "fp4_sim.cuh"

// ------------------------------------------------------------------- E4M3
// OCP E4M3 (the "FN" variant torch calls float8_e4m3fn): 1s 4e 3m, bias 7,
// no infinities, max finite 448, min subnormal 2^-9.
#define E4M3_MAX 448.0f
#define E4M3_MIN_SUB 0.001953125f          // 2^-9

__host__ __device__ inline float quant_e4m3(float x) {
    if (!(x != 0.f)) return 0.f;                       // 0 and NaN -> 0
    float s = x < 0.f ? -1.f : 1.f;
    float a = fabsf(x);
    if (a >= E4M3_MAX) return s * E4M3_MAX;            // saturating
    if (a < E4M3_MIN_SUB * 0.5f) return 0.f;
    int e = (int)floorf(log2f(a));
    if (e < -6) e = -6;                                // subnormal binade
    float step = ldexpf(1.0f, e - 3);                  // 3 mantissa bits
    float r = rintf(a / step) * step;                  // round-half-to-even
    if (r > E4M3_MAX) r = E4M3_MAX;
    return s * r;
}

// ------------------------------------------------------------------- E8M0
// A bare power of two. This is the MX shared scale: no mantissa at all.
__host__ __device__ inline float quant_e8m0(float x) {
    if (!(x > 0.f)) return 1.0f;
    int e = (int)floorf(log2f(x));
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return ldexpf(1.0f, e);
}

// A power of two >= x, so x/scale never exceeds 1. Rounding a shared exponent
// down instead clamps the top of every block: an error floor that does not
// improve with mantissa width.
__host__ __device__ inline float quant_e8m0_up(float x) {
    if (!(x > 0.f)) return 1.0f;
    int e = (int)ceilf(log2f(x));
    if (ldexpf(1.0f, e) < x) ++e;                      // guard log2 rounding
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return ldexpf(1.0f, e);
}

// ------------------------------------------------------------------- E2M1
// Nearest E2M1 code for a value already divided by the block scale. Ties go
// to the smaller magnitude (OCP specifies ties-to-even); fp4.py matches.
__host__ __device__ inline int quant_e2m1_code(float v) {
    const float m[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float a = fabsf(v);
    int best = 0;
    float bd = fabsf(a - m[0]);
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(a - m[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best | (v < 0.f ? 8 : 0);
}

// ------------------------------------------------------- block scale choice
// OCP MX: X = 2^(floor(log2(amax)) - emax_elem). For E2M1 the largest value is
// 6 = 1.5 * 2^2, so emax_elem = 2. For E4M3 it is 448 = 1.75 * 2^8, so 8.
__host__ __device__ inline float mx_scale_e2m1(float amax) {
    if (!(amax > 0.f)) return 1.0f;
    int e = (int)floorf(log2f(amax)) - 2;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return ldexpf(1.0f, e);
}

__host__ __device__ inline float mx_scale_e4m3(float amax) {
    if (!(amax > 0.f)) return 1.0f;
    int e = (int)floorf(log2f(amax)) - 8;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return ldexpf(1.0f, e);
}

// NVFP4: the scale itself is an E4M3 number, chosen so amax maps onto 6, the
// top of the E2M1 grid. Nothing is wasted rounding the scale down to a power
// of two.
__host__ __device__ inline float nv_scale_e2m1(float amax) {
    if (!(amax > 0.f)) return 1.0f;
    float s = quant_e4m3(amax / 6.0f);
    return s > 0.f ? s : E4M3_MIN_SUB;
}

// Scale that maps `bound` onto the largest magnitude of a signed `bits` field.
__host__ __device__ inline double fixed_scale(double bound, int bits) {
    return (double)((1LL << (bits - 1)) - 1) / bound;
}
