"""H100 driver: correctness + throughput for both experiments.

    python bench.py                # everything
    python bench.py --only mxfp4
    python bench.py --only dual
    python bench.py --shape 8192 8192 8192
"""

import argparse
import torch
import triton

import fp4
from mxfp4_gemm import mxfp4_gemm_fp8, mxfp4_gemm_packed
import dual_gemm as dg


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def tflops(ms, M, N, K, gemms=1):
    return 2.0 * M * N * K * gemms / (ms * 1e-3) / 1e12


def bench(fn, **kw):
    return triton.testing.do_bench(fn, warmup=25, rep=100, **kw)


def random_fp4_codes(shape, device):
    """Uniform over the 15 distinct E2M1 codes (skips the -0 duplicate)."""
    pick = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
                        device=device, dtype=torch.uint8)
    idx = torch.randint(0, 15, shape, device=device)
    return pick[idx]


# --------------------------------------------------------------------------- #
def run_mxfp4(M, N, K):
    section(f"1. MXFP4 GEMM emulated on FP8 tensor cores   (M={M} N={N} K={K})")
    dev = "cuda"

    # ---- exactness: a 128-deep GEMM with unit scales must be BIT exact -------
    # 4 blocks x 4608 quarter-units = 18432 -> 15 bits, still exact in fp32 once
    # each 32-block has been promoted out of the 14-bit MMA accumulator.
    m0, n0, k0 = 256, 256, 128
    ca = random_fp4_codes((m0, k0), dev)
    cb = random_fp4_codes((k0, n0), dev)
    sa = torch.ones((m0, k0 // 32), device=dev, dtype=torch.float32)
    sb = torch.ones((k0 // 32, n0), device=dev, dtype=torch.float32)
    got = mxfp4_gemm_fp8(fp4.codes_to_e4m3(ca), fp4.codes_to_e4m3(cb), sa, sb,
                         out_dtype=torch.float32)
    qa = fp4.codes_to_q(ca).double()          # q = 2*value, so C = (qa@qb)/4
    qb = fp4.codes_to_q(cb).double()
    exact = (qa @ qb) / 4.0
    nbad = (got.double() != exact).sum().item()
    print(f"  bit-exactness (K=64, unit scales): {nbad} of {m0*n0} elements differ "
          f"-> {'EXACT' if nbad == 0 else 'LOSSY'}")
    print("  (13-bit worst case per 32-block vs the ~14-bit Hopper FP8 accumulator)")

    # ---- accuracy on real data ----------------------------------------------
    a = torch.randn((M, K), device=dev)
    b = torch.randn((K, N), device=dev)
    a_codes, a_scale = fp4.quantize_mxfp4(a, 32)
    bt_codes, bt_scale = fp4.quantize_mxfp4(b.t().contiguous(), 32)   # [N,K],[N,K/32]
    b_codes = bt_codes.t().contiguous()                              # [K,N]
    b_scale = bt_scale.t().contiguous()                              # [K/32,N]

    a_fp8 = fp4.codes_to_e4m3(a_codes)
    b_fp8 = fp4.codes_to_e4m3(b_codes)
    c = mxfp4_gemm_fp8(a_fp8, b_fp8, a_scale, b_scale, out_dtype=torch.float32)
    ref = fp4.reference_mxfp4_gemm(a_codes, a_scale, b_codes, b_scale, 32)
    rel = ((c - ref).abs().max() / ref.abs().max()).item()
    print(f"  max rel err vs float64 reference : {rel:.3e}   "
          f"({'ok' if rel < 1e-5 else 'CHECK'})")

    # ---- packed (2 FP4 per byte in HBM) --------------------------------------
    a_pk = fp4.pack_fp4(a_codes)
    bt_pk = fp4.pack_fp4(bt_codes)
    cp = mxfp4_gemm_packed(a_pk, bt_pk, a_scale, bt_scale, out_dtype=torch.float32)
    relp = ((cp - ref).abs().max() / ref.abs().max()).item()
    print(f"  packed-input kernel rel err      : {relp:.3e}   "
          f"({'ok' if relp < 1e-5 else 'CHECK'})")

    # ---- throughput ----------------------------------------------------------
    ab = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    bb = torch.randn((K, N), device=dev, dtype=torch.bfloat16)
    rows = [("torch bf16 matmul", bench(lambda: ab @ bb))]

    one = torch.tensor(1.0, device=dev)
    b_col = b_fp8.t().contiguous().t()
    try:
        torch._scaled_mm(a_fp8, b_col, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        rows.append(("torch _scaled_mm fp8 (per-tensor ceiling)",
                     bench(lambda: torch._scaled_mm(a_fp8, b_col, scale_a=one,
                                                    scale_b=one,
                                                    out_dtype=torch.bfloat16))))
    except Exception as e:                                    # noqa: BLE001
        print(f"  [skip] _scaled_mm unavailable: {type(e).__name__}: {e}")

    rows.append(("mxfp4 -> fp8, block-32 scaled",
                 bench(lambda: mxfp4_gemm_fp8(a_fp8, b_fp8, a_scale, b_scale))))
    rows.append(("mxfp4 packed (2/byte) -> fp8, block-32",
                 bench(lambda: mxfp4_gemm_packed(a_pk, bt_pk, a_scale, bt_scale))))

    print(f"\n  {'kernel':<44}{'ms':>10}{'TFLOP/s':>12}")
    for name, ms in rows:
        print(f"  {name:<44}{ms:>10.3f}{tflops(ms, M, N, K):>12.1f}")
    print("\n  H100 SXM dense peaks: bf16 989 TFLOP/s, fp8 1979 TFLOP/s.")
    print("  The fp8 line is the ceiling for FP4 on Hopper -- nothing beats it.")


# --------------------------------------------------------------------------- #
def run_dual(M, N, K):
    section(f"2. Packed dual-batch GEMM   (M={M} N={N} K={K})")
    dev = "cuda"

    print("\n  a) numerics: can one accumulator carry both results?\n")
    print(f"  {'K':>6}{'s':>5}{'bits needed':>13}{'bf16 exact':>13}"
          f"{'fp8 encodable':>15}{'fp8 exact':>12}")

    def exact_rate(dt, q, s, ref1, ref2):
        a1, b1, b2 = q[0].to(dt), q[2].to(dt), q[3].to(dt)
        a2s = (q[1] * (2.0 ** s)).to(dt)
        acc = dg.dual_packed(a1, a2s, b1, b2)
        g1, g2 = dg.unpack_dual(acc, s)
        return ((g1.double() == ref1) & (g2.double() == ref2)).float().mean().item()

    for k in [16, 32, 64, 128, 512]:
        s = dg.slot_offset(k)
        need = 2 * (144 * k).bit_length() + 1
        m = n = 256
        # The kernels step K in blocks of up to 64, so zero-pad short K. Zeros
        # contribute nothing to the dot product, and both the kernel and the
        # reference see the same padded operands.
        kp = max(64, (k + 63) // 64 * 64)
        q = []
        for sh, axis in (((m, k), 1), ((m, k), 1), ((k, n), 0), ((k, n), 0)):
            v = fp4.codes_to_q(random_fp4_codes(sh, dev)).float()
            if kp != k:
                pad = torch.zeros((m, kp - k) if axis == 1 else (kp - k, n), device=dev)
                v = torch.cat([v, pad], dim=axis)
            q.append(v.contiguous())
        ref1 = q[0].double() @ q[2].double()
        ref2 = q[1].double() @ q[3].double()

        bf = exact_rate(torch.bfloat16, q, s, ref1, ref2)

        # can the 2^s offset even be written into an e4m3 operand? max is 448.
        a2s = q[1] * (2.0 ** s)
        rt = a2s.to(torch.float8_e4m3fn).float()
        enc = bool(torch.isfinite(rt).all().item() and (rt == a2s).all().item())
        f8 = f"{100*exact_rate(torch.float8_e4m3fn, q, s, ref1, ref2):>11.1f}%" \
            if enc else f"{'--':>12}"
        print(f"  {k:>6}{s:>5}{need:>13}{100*bf:>12.1f}%"
              f"{('yes' if enc else 'NO (>448)'):>15}{f8}")

    print("\n  bits needed = 2*ceil(log2(144K))+1. fp32 mantissa = 24; Hopper fp8 MMA ~14.")

    # ---- throughput ----------------------------------------------------------
    print("\n  b) throughput: does one accumulator buy anything?\n")
    a1 = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    a2 = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    b1 = torch.randn((K, N), device=dev, dtype=torch.bfloat16)
    b2 = torch.randn((K, N), device=dev, dtype=torch.bfloat16)

    rows = [
        ("2x torch.matmul (two launches)",
         bench(lambda: (a1 @ b1, a2 @ b2))),
        ("fused, TWO accumulators",
         bench(lambda: dg.dual_separate(a1, a2, b1, b2))),
        ("fused, ONE packed accumulator",
         bench(lambda: dg.dual_packed(a1, a2, b1, b2))),
    ]
    for splits in (4, 8):
        if K % (splits * 64) == 0:
            rows.append((f"split-K={splits}, TWO accumulators + 2 atomics",
                         bench(lambda s=splits: dg.splitk_separate(a1, a2, b1, b2, s))))
            rows.append((f"split-K={splits}, ONE accumulator + 1 atomic",
                         bench(lambda s=splits: dg.splitk_packed(a1, a2, b1, b2, s))))

    print(f"  {'variant':<44}{'ms':>10}{'TFLOP/s':>12}")
    for name, ms in rows:
        print(f"  {name:<44}{ms:>10.3f}{tflops(ms, M, N, K, gemms=2):>12.1f}")
    print("\n  Same MAC count in every row -- the only difference is accumulator")
    print("  registers and store/atomic traffic. That is the whole hypothesis.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["mxfp4", "dual"])
    p.add_argument("--shape", nargs=3, type=int, default=[4096, 4096, 4096],
                   metavar=("M", "N", "K"))
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  sm_{props.major}{props.minor}  "
          f"{props.total_memory/2**30:.0f} GiB  torch {torch.__version__}  "
          f"triton {triton.__version__}")
    if props.major != 9:
        print("WARNING: not Hopper. The 14-bit FP8 accumulator claim is Hopper-specific.")

    M, N, K = args.shape
    if args.only in (None, "mxfp4"):
        run_mxfp4(M, N, K)
    if args.only in (None, "dual"):
        run_dual(M, N, K)


if __name__ == "__main__":
    main()
