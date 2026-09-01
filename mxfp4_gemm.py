"""MXFP4 GEMM emulated on Hopper FP8 tensor cores.

The mainloop advances K in sub-blocks of 32 -- the MXFP4 block size -- and
promotes to an FP32 CUDA-core accumulator at every block boundary. That serves
two purposes at once:

  1. It is where the per-block scales get applied (H100 has no block-scaled MMA).
  2. It makes the arithmetic exact. The Hopper FP8 MMA accumulates at ~14 bits
     (DeepSeek-V3 sec 3.5.2); an FP4 dot product over 32 terms is bounded by
     144*32 = 4608 quarter-units = 13 bits. Promoting every 32 keeps every
     partial sum inside that budget, so there is no accumulation error at all.

Promotion costs ~3 FFMA per 32 MACs, i.e. ~3% of the tensor-core work.
"""

import torch
import triton
import triton.language as tl


def _configs():
    # Every config uses BLOCK_K = SUB*32 <= 128, so K must be a multiple of 128
    # for the unrolled inner loop to stay in bounds (asserted in the wrappers).
    cfgs = []
    for bm, bn, w in [(128, 128, 8), (128, 256, 8), (256, 128, 8),
                      (128, 64, 4), (64, 128, 4), (64, 64, 4)]:
        for sub in [2, 4]:                       # BLOCK_K = sub * 32
            for s in [3, 4]:
                cfgs.append(triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "SUB": sub, "GROUP_M": 8},
                    num_warps=w, num_stages=s))
    return cfgs


# --------------------------------------------------------------------------- #
#  A [M,K] fp8e4m3 (K-contig), B [K,N] fp8e4m3 (N-contig), values on E2M1 grid
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp4_gemm_fp8_kernel(
        A, B, C, SA, SB,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        stride_sam, stride_sak, stride_sbk, stride_sbn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        SUB: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, 32)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, SUB * 32)):
        for j in tl.static_range(SUB):
            a = tl.load(a_ptrs + j * 32 * stride_ak)          # [BM, 32] fp8
            b = tl.load(b_ptrs + j * 32 * stride_bk)          # [32, BN] fp8
            p = tl.dot(a, b)                                  # exact: 13 bits
            kb = k * SUB + j
            sa = tl.load(SA + offs_m * stride_sam + kb * stride_sak)
            sb = tl.load(SB + kb * stride_sbk + offs_n * stride_sbn)
            acc += p * (sa[:, None] * sb[None, :])            # promote to FP32
        a_ptrs += SUB * 32 * stride_ak
        b_ptrs += SUB * 32 * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def mxfp4_gemm_fp8(a_fp8, b_fp8, a_scale, b_scale, out_dtype=torch.bfloat16):
    """a_fp8 [M,K] K-contig, b_fp8 [K,N] N-contig, a_scale [M,K//32], b_scale [K//32,N]."""
    M, K = a_fp8.shape
    K2, N = b_fp8.shape
    assert K == K2 and K % 128 == 0, "K must be a multiple of 128 (max BLOCK_K)"
    assert a_scale.shape == (M, K // 32) and b_scale.shape == (K // 32, N)
    c = torch.empty((M, N), device=a_fp8.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _mxfp4_gemm_fp8_kernel[grid](
        a_fp8, b_fp8, c, a_scale, b_scale, M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), b_fp8.stride(0), b_fp8.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1), b_scale.stride(0), b_scale.stride(1))
    return c


# --------------------------------------------------------------------------- #
#  Packed variant: 2 FP4 per byte in HBM (the layout that actually saves memory)
#  A [M, K//2] uint8, B_T [N, K//2] uint8, both K-contiguous.  C = A @ B_T.T
# --------------------------------------------------------------------------- #
@triton.jit
def _unpack_to_e4m3(byts, BM: tl.constexpr, HALF: tl.constexpr):
    """[BM, HALF] packed uint8 -> [BM, 2*HALF] fp8e4nv, low nibble first.

    E2M1 code n maps to the E4M3 byte  sign<<7 | (exp+6)<<3 | mant<<2, which
    collapses to ((n & 8) << 4) | ((n & 7) << 2) | 0x30 for magnitude codes >= 2.
    Codes 0 (+-0.0) and 1 (+-0.5) are special-cased. Table:
        code 0..7 -> 0x00 0x30 0x38 0x3C 0x40 0x44 0x48 0x4C
    """
    b = byts.to(tl.int32)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    n = tl.reshape(tl.join(lo, hi), (BM, 2 * HALF))
    mag = n & 7
    out = (mag << 2) + 0x30
    out = tl.where(mag == 1, 0x30, out)
    out = tl.where(mag == 0, 0, out)
    out = out | ((n & 8) << 4)
    return out.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp4_gemm_packed_kernel(
        A, B, C, SA, SB,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        stride_sam, stride_sak, stride_sbn, stride_sbk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        SUB: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_b = tl.arange(0, 16)                      # 16 bytes == 32 FP4 values

    a_ptrs = A + offs_m[:, None] * stride_am + offs_b[None, :] * stride_ak
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_b[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, SUB * 32)):
        for j in tl.static_range(SUB):
            a = _unpack_to_e4m3(tl.load(a_ptrs + j * 16 * stride_ak), BLOCK_M, 16)
            b = _unpack_to_e4m3(tl.load(b_ptrs + j * 16 * stride_bk), BLOCK_N, 16)
            p = tl.dot(a, tl.trans(b))
            kb = k * SUB + j
            sa = tl.load(SA + offs_m * stride_sam + kb * stride_sak)
            sb = tl.load(SB + offs_n * stride_sbn + kb * stride_sbk)
            acc += p * (sa[:, None] * sb[None, :])
        a_ptrs += SUB * 16 * stride_ak
        b_ptrs += SUB * 16 * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def mxfp4_gemm_packed(a_pk, bt_pk, a_scale, b_scale, out_dtype=torch.bfloat16):
    """a_pk [M,K//2] uint8, bt_pk [N,K//2] uint8, a_scale [M,K//32], b_scale [N,K//32]."""
    M, Kh = a_pk.shape
    N, Kh2 = bt_pk.shape
    K = Kh * 2
    assert Kh == Kh2 and K % 128 == 0, "K must be a multiple of 128 (max BLOCK_K)"
    c = torch.empty((M, N), device=a_pk.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _mxfp4_gemm_packed_kernel[grid](
        a_pk, bt_pk, c, a_scale, b_scale, M, N, K,
        a_pk.stride(0), a_pk.stride(1), bt_pk.stride(0), bt_pk.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1), b_scale.stride(0), b_scale.stride(1))
    return c
