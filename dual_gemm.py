"""The packed dual-batch GEMM experiment.

Hypothesis under test: two independent FP4 GEMMs C1 = A1 B1 and C2 = A2 B2 can
share ONE fp32 accumulator, because FP4 dot products are integers and can be
stacked in disjoint bit fields:

        acc = sum(A1 B1) + 2^s * sum(A2 B2)

If that holds, the mainloop needs half the accumulator registers and split-K
needs half the partial-sum traffic -- the reduction win.

Two things decide whether it holds, and this file measures both on real silicon:

  encoding     the 2^s offset has to live somewhere. In bf16 it is free (it is
               just the exponent field). In fp8 e4m3 the operand saturates at
               448, so at most 2^5 can be folded per side, i.e. s <= 10 --
               already short of the s = 14 that K = 32 needs.

  accumulator  the slot layout is [C1 : s bits][C2 : s bits], so the accumulator
               needs 2*ceil(log2(144*K)) + 1 bits: 27 at K=32, 31 at K=128.
               True fp32 gives 24. The Hopper fp8 MMA datapath gives ~14.

Values are carried on the integer grid q = 2*value, so C1 and C2 are exact
integers and the split is well defined. q is in {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12},
so |q1*q2| <= 144 and |dot| <= 144*K.

Kernels here always leave the accumulator PACKED (one fp32 tile). Splitting is
a host-side op -- keeping it out of the kernel is the point: the packed kernel
must write half as many bytes as the separate one, or there is no win to have.
"""

import math
import torch
import triton
import triton.language as tl


def slot_offset(K: int) -> int:
    """Minimum 2^s separation for a K-deep FP4 dot product (+1 guard for sign)."""
    return int(math.floor(math.log2(144 * K))) + 2


def unpack_dual(acc: torch.Tensor, s: int):
    """Split acc = C1 + 2^s * C2 back into the two results (host side)."""
    w = float(2 ** s)
    c2 = torch.round(acc / w)
    return acc - c2 * w, c2


def _cfgs():
    out = []
    for bm, bn in [(128, 128), (128, 64), (64, 128), (64, 64)]:
        for bk in [32, 64]:
            for w, st in [(4, 3), (8, 3), (8, 4)]:
                out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                         num_warps=w, num_stages=st))
    return out


# --------------------------------------------------------------------------- #
#  PACKED: one accumulator carries both problems, one tile stored
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
@triton.jit
def _dual_packed_kernel(
        A1, A2S, B1, B2, OUT,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """A2S already carries the 2^s offset (pre-scaled on the host)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a1p = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a2p = A2S + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1p = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2p = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(a1p), tl.load(b1p), acc)     # low slot
        acc = tl.dot(tl.load(a2p), tl.load(b2p), acc)     # high slot, pre-offset
        a1p += BLOCK_K * stride_ak
        a2p += BLOCK_K * stride_ak
        b1p += BLOCK_K * stride_bk
        b2p += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(OUT + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


# --------------------------------------------------------------------------- #
#  SEPARATE: two accumulators in the same fused kernel (the fair baseline)
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
@triton.jit
def _dual_separate_kernel(
        A1, A2, B1, B2, OUT1, OUT2,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a1p = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a2p = A2 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1p = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2p = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc1 = tl.dot(tl.load(a1p), tl.load(b1p), acc1)
        acc2 = tl.dot(tl.load(a2p), tl.load(b2p), acc2)
        a1p += BLOCK_K * stride_ak
        a2p += BLOCK_K * stride_ak
        b1p += BLOCK_K * stride_bk
        b2p += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    base = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(OUT1 + base, acc1, mask=mask)
    tl.store(OUT2 + base, acc2, mask=mask)


# --------------------------------------------------------------------------- #
#  split-K: where the reduction traffic actually lives.
#  Requires K % (SPLITS * BLOCK_K) == 0 (asserted in the wrappers).
# --------------------------------------------------------------------------- #
@triton.jit
def _splitk_packed_kernel(
        A1, A2S, B1, B2, OUT,
        M, N, K, KPER,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = pid_k * KPER * BLOCK_K + tl.arange(0, BLOCK_K)

    a1p = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a2p = A2S + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1p = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2p = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, KPER):
        acc = tl.dot(tl.load(a1p), tl.load(b1p), acc)
        acc = tl.dot(tl.load(a2p), tl.load(b2p), acc)
        a1p += BLOCK_K * stride_ak
        a2p += BLOCK_K * stride_ak
        b1p += BLOCK_K * stride_bk
        b2p += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(OUT + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc,
                  mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def _splitk_separate_kernel(
        A1, A2, B1, B2, OUT1, OUT2,
        M, N, K, KPER,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = pid_k * KPER * BLOCK_K + tl.arange(0, BLOCK_K)

    a1p = A1 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a2p = A2 + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1p = B1 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2p = B2 + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, KPER):
        acc1 = tl.dot(tl.load(a1p), tl.load(b1p), acc1)
        acc2 = tl.dot(tl.load(a2p), tl.load(b2p), acc2)
        a1p += BLOCK_K * stride_ak
        a2p += BLOCK_K * stride_ak
        b1p += BLOCK_K * stride_bk
        b2p += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    base = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.atomic_add(OUT1 + base, acc1, mask=mask)
    tl.atomic_add(OUT2 + base, acc2, mask=mask)


# --------------------------------------------------------------------------- #
#  Plain single GEMM with an fp32 output, so every variant in train_step.py can
#  be compared against the same kernel family. Comparing a packed Triton kernel
#  against cuBLAS would measure Triton-vs-cuBLAS, not the technique; and a bf16
#  output would round the accumulator away before the slots can be split.
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
@triton.jit
def _gemm_kernel(
        A, B, OUT,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    ap = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    bp = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BLOCK_K * stride_ak
        bp += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(OUT + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def gemm(a, b):
    """Single GEMM, one fp32 accumulator, fp32 out. a [M,K], b [K,N]."""
    M, K = a.shape
    N = b.shape[1]
    assert K % 64 == 0, "K must be a multiple of 64 (max BLOCK_K); zero-pad if shorter"
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _gemm_kernel[grid](
        a, b, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1))
    return out


# ------------------------------------------------------------------- wrappers
def dual_packed(a1, a2s, b1, b2):
    """Returns the PACKED fp32 tile. Call unpack_dual(acc, s) to split it."""
    M, K = a1.shape
    N = b1.shape[1]
    assert K % 64 == 0, "K must be a multiple of 64 (max BLOCK_K); zero-pad if shorter"
    out = torch.empty((M, N), device=a1.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _dual_packed_kernel[grid](
        a1, a2s, b1, b2, out, M, N, K,
        a1.stride(0), a1.stride(1), b1.stride(0), b1.stride(1),
        out.stride(0), out.stride(1))
    return out


def dual_separate(a1, a2, b1, b2):
    M, K = a1.shape
    N = b1.shape[1]
    assert K % 64 == 0, "K must be a multiple of 64 (max BLOCK_K); zero-pad if shorter"
    c1 = torch.empty((M, N), device=a1.device, dtype=torch.float32)
    c2 = torch.empty_like(c1)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _dual_separate_kernel[grid](
        a1, a2, b1, b2, c1, c2, M, N, K,
        a1.stride(0), a1.stride(1), b1.stride(0), b1.stride(1),
        c1.stride(0), c1.stride(1))
    return c1, c2


_SK = dict(BM=128, BN=128, BK=64, warps=8, stages=3)


def splitk_packed(a1, a2s, b1, b2, splits=8, **kw):
    p = {**_SK, **kw}
    M, K = a1.shape
    N = b1.shape[1]
    assert K % (splits * p["BK"]) == 0, "K must divide splits*BK"
    out = torch.zeros((M, N), device=a1.device, dtype=torch.float32)
    _splitk_packed_kernel[(triton.cdiv(M, p["BM"]), triton.cdiv(N, p["BN"]), splits)](
        a1, a2s, b1, b2, out, M, N, K, K // p["BK"] // splits,
        a1.stride(0), a1.stride(1), b1.stride(0), b1.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=p["BM"], BLOCK_N=p["BN"], BLOCK_K=p["BK"],
        num_warps=p["warps"], num_stages=p["stages"])
    return out


def splitk_separate(a1, a2, b1, b2, splits=8, **kw):
    p = {**_SK, **kw}
    M, K = a1.shape
    N = b1.shape[1]
    assert K % (splits * p["BK"]) == 0, "K must divide splits*BK"
    o1 = torch.zeros((M, N), device=a1.device, dtype=torch.float32)
    o2 = torch.zeros_like(o1)
    _splitk_separate_kernel[(triton.cdiv(M, p["BM"]), triton.cdiv(N, p["BN"]), splits)](
        a1, a2, b1, b2, o1, o2, M, N, K, K // p["BK"] // splits,
        a1.stride(0), a1.stride(1), b1.stride(0), b1.stride(1),
        o1.stride(0), o1.stride(1),
        BLOCK_M=p["BM"], BLOCK_N=p["BN"], BLOCK_K=p["BK"],
        num_warps=p["warps"], num_stages=p["stages"])
    return o1, o2
