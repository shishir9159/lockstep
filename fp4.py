"""FP4 (E2M1) / MXFP4 primitives: quantization, packing, exact references.

Layout conventions used everywhere in this repo:
    A      : [M, K]      K-contiguous
    B      : [K, N]      N-contiguous   (for the fp8 path,  C = A @ B)
    B_T    : [N, K]      K-contiguous   (for the packed path, C = A @ B_T.T)
    A scale: [M, K//32]
    B scale: [K//32, N]  (fp8 path)  or  [N, K//32]  (packed path)

Why E2M1 -> E4M3 is lossless: the 16 E2M1 values are {0, +-0.5, 1, 1.5, 2, 3, 4, 6}
and every one of them is an exactly representable E4M3 value (0.5 = 2^-1 is a
normal in E4M3; 6 << 448).  So the cast is a 16-entry table lookup, not a
rounding step, and an FP8 tensor-core GEMM over FP4-valued operands reproduces
FP4 arithmetic exactly.
"""

import torch

# ---------------------------------------------------------------- E2M1 tables
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]          # code & 7 -> magnitude
FP4_MAX = 6.0
FP4_EMAX = 2                                                   # max exponent of E2M1

E2M1_VALUES = torch.tensor(_E2M1_MAG + [-v for v in _E2M1_MAG], dtype=torch.float32)

# code -> E4M3 byte.  byte = sign<<7 | (exp+6)<<3 | mant<<2, with 0 and 0.5 special.
E2M1_TO_E4M3 = torch.tensor(
    [0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
     0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC], dtype=torch.uint8)

# code -> 2*value as a signed integer, i.e. the exact integer grid the FP4 values
# live on.  Used by the packed dual-GEMM experiment, which needs integer results.
E2M1_Q = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                      dtype=torch.int32)


def _round_to_e2m1(q: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest onto the E2M1 grid.  Returns 4-bit codes as uint8.

    Ties round away from zero (not to even) -- fine for a research harness, but
    note it if you are chasing bit-parity with a specific vendor kernel.
    """
    mags = torch.tensor(_E2M1_MAG, device=q.device, dtype=torch.float32)
    mid = (mags[1:] + mags[:-1]) * 0.5                          # 7 midpoints
    a = q.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(a, mid)                               # 0..7
    return (idx.to(torch.uint8) | (torch.signbit(q).to(torch.uint8) << 3))


def quantize_mxfp4(x: torch.Tensor, block: int = 32):
    """MXFP4 quantize along the last dim.

    Returns (codes uint8 [..., K], scale fp32 [..., K//block]).
    Shared scale follows OCP MX: 2^(floor(log2(amax)) - emax_elem), emax_elem = 2.
    """
    x = x.float()
    *lead, K = x.shape
    assert K % block == 0, f"K={K} must be a multiple of block={block}"
    xb = x.reshape(*lead, K // block, block)
    amax = xb.abs().amax(dim=-1)
    e = torch.floor(torch.log2(amax.clamp(min=1e-30))) - FP4_EMAX
    scale = torch.exp2(e.clamp(-127, 127))
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    codes = _round_to_e2m1(xb / scale.unsqueeze(-1)).reshape(*lead, K)
    return codes, scale


def dequantize_mxfp4(codes: torch.Tensor, scale: torch.Tensor, block: int = 32):
    v = E2M1_VALUES.to(codes.device)[codes.long()]
    *lead, K = v.shape
    return (v.reshape(*lead, K // block, block) * scale.unsqueeze(-1)).reshape(*lead, K)


def codes_to_e4m3(codes: torch.Tensor) -> torch.Tensor:
    """Lossless FP4 -> FP8(E4M3) cast via table lookup."""
    return E2M1_TO_E4M3.to(codes.device)[codes.long()].view(torch.float8_e4m3fn)


def codes_to_q(codes: torch.Tensor) -> torch.Tensor:
    """FP4 code -> 2*value as int32 (the exact integer grid)."""
    return E2M1_Q.to(codes.device)[codes.long()]


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] uint8 codes -> [..., K//2] bytes; even index in the low nibble."""
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0x0F, packed >> 4
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


# ------------------------------------------------------------------ references
def reference_mxfp4_gemm(a_codes, a_scale, b_codes, b_scale, block=32):
    """float64 reference for C = A @ B with A [M,K], B [K,N].

    b_codes is [K, N] and b_scale is [K//block, N], so B is quantized along K.
    """
    A = dequantize_mxfp4(a_codes, a_scale, block).double()                  # [M,K]
    Bv = E2M1_VALUES.to(b_codes.device)[b_codes.long()].double()            # [K,N]
    K, N = Bv.shape
    B = (Bv.reshape(K // block, block, N) * b_scale.double().unsqueeze(1)).reshape(K, N)
    return (A @ B).float()


def blockwise_exact_dot(qa: torch.Tensor, qb: torch.Tensor, block: int = 32):
    """Exact integer dot products, per block, from the int grid (qa,qb = 2*value).

    Returns int64 [.., K//block] of sum(qa*qb) per block.  The true FP4 partial
    sum is this / 4.  Bound: |sum| <= 144*block quarter-units, so block<=64 fits
    in 14 bits -- which is exactly why Hopper's ~14-bit FP8 accumulator is
    lossless for MXFP4 (block 32) and NVFP4 (block 16).
    """
    p = (qa.long() * qb.long())
    *lead, K = p.shape
    return p.reshape(*lead, K // block, block).sum(-1)
