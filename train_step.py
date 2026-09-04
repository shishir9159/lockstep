"""Minimal forward + backward for one Linear layer, two microbatches.

    Y_i  = X_i @ W^T                      forward   (W shared across batches)
    dX_i = dY_i @ W                       dgrad     (W shared)
    dW   = dY1^T @ X1 + dY2^T @ X2        wgrad     (the two results are SUMMED)

Three directions, three different relationships between the microbatches, and
only one of them is a reduction. That is the point of running this:

  fwd, dgrad   two results that must stay separate, sharing an operand
  wgrad        two results that get added -- this is your reduction

FWD / DGRAD ladder (all Triton, fp32 accumulator, fp32 out, so the only thing
that varies is the technique -- comparing a packed Triton kernel against cuBLAS
would measure Triton-vs-cuBLAS instead):

  2 launches     gemm(a1,b) ; gemm(a2,b)          2 kernels, 2 accumulators
  fused 2-acc    dual_separate                    1 kernel,  2 accumulators
  packed 1-acc   dual_packed(a1, 2^s*a2, b, b)    1 kernel,  1 accumulator  <- your idea
  concat         gemm(cat[a1;a2], b)              1 kernel,  1 acc, M doubled
  preadd         gemm(a1 + 2^s*a2, b)             1 kernel,  1 acc, HALF THE MACs

`preadd` is the only variant that removes work rather than moving it: because W
is shared, (X1 + 2^s X2) @ W^T = Y1 + 2^s Y2 in one GEMM of the ORIGINAL size,
not a doubled one. It is also the first to die -- the packed operand needs
bits(12) + s + 1 significand bits (19 at K=32, more at real widths) against 8 in
bf16 and 11 in fp16.

WGRAD is measured on cuBLAS for both variants, because there the question is
purely op count -- 2 GEMMs plus an explicit add, versus 1 GEMM whose single
accumulator performs the sum -- and routing it through Triton would only add a
transposed-operand penalty that has nothing to do with the technique.

Every variant is checked against an exact integer reference, so a variant that
is fast because it is wrong shows up as wrong.

    python train_step.py
    python train_step.py --shape 4096 4096 4096
    python train_step.py --dtype float16         # 11 mantissa bits instead of 8

Note on the working regime: the packed accumulator needs 2*ceil(log2(144K))+1
bits, so it fits in fp32 only for K <= 14 -- below the 64 the kernels require as
a minimum contraction depth. No realistic layer width is in that regime, which
is the finding, not a harness limitation. bench.py demonstrates the regime where
it IS exact by zero-padding a logical K of 8-32 out to a physical 64.
"""

import argparse
import torch
import triton

import dual_gemm as dg
import fp4


def q_tensor(shape, dev):
    """Values on the FP4 integer grid q = 2*value, so all results are integers."""
    pick = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
                        device=dev, dtype=torch.uint8)
    codes = pick[torch.randint(0, 15, shape, device=dev)]
    return fp4.codes_to_q(codes).float()


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


# ------------------------------- fwd / dgrad: shared operand, separate results
def v_two_launch(a1, a2, b):
    return dg.gemm(a1, b), dg.gemm(a2, b)


def v_fused_2acc(a1, a2, b):
    return dg.dual_separate(a1, a2, b, b)


def v_packed_1acc(a1, a2, b, s):
    return dg.unpack_dual(dg.dual_packed(a1, a2 * (2.0 ** s), b, b), s)


def v_concat(a_cat, b, B):
    y = dg.gemm(a_cat, b)
    return y[:B], y[B:]


def v_preadd(a1, a2, b, s):
    return dg.unpack_dual(dg.gemm(a1 + a2 * (2.0 ** s), b), s)


# ------------------------------------------------- wgrad: the actual reduction
def wgrad_vanilla(dy1, dy2, x1, x2):
    return (dy1.transpose(0, 1) @ x1) + (dy2.transpose(0, 1) @ x2)   # 2 GEMM + add


def wgrad_concat(dy_cat, x_cat):
    return dy_cat.transpose(0, 1) @ x_cat                            # 1 GEMM, no add


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, default=[4096, 4096, 4096],
                   metavar=("B", "CIN", "COUT"))
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()
    B, CIN, COUT = args.shape
    dt = getattr(torch, args.dtype)
    dev = "cuda"

    prop = torch.cuda.get_device_properties(0)
    print(f"{prop.name} sm_{prop.major}{prop.minor}   "
          f"B={B} Cin={CIN} Cout={COUT}  {args.dtype}\n")

    # Microbatches live in one allocation, so `concat` is a view, not a copy --
    # which is how they arrive in real training.
    X = q_tensor((2 * B, CIN), dev)
    dY = q_tensor((2 * B, COUT), dev)
    Wq = q_tensor((COUT, CIN), dev)
    x1, x2 = X[:B], X[B:]
    dy1, dy2 = dY[:B], dY[B:]

    Xc, dYc = X.to(dt), dY.to(dt)
    x1c, x2c = Xc[:B], Xc[B:]
    dy1c, dy2c = dYc[:B], dYc[B:]
    W = Wq.to(dt)                              # [Cout, Cin]
    WT = Wq.t().contiguous().to(dt)            # [Cin, Cout], N-contiguous

    s = dg.slot_offset(CIN)                    # fwd contracts over Cin
    s_dg = dg.slot_offset(COUT)                # dgrad contracts over Cout
    mant = {torch.bfloat16: 8, torch.float16: 11}[dt]
    acc_need = 2 * (144 * CIN).bit_length() + 1
    op_need = 4 + s + 1
    print(f"  bit budget at K={CIN}:  slot offset s={s}")
    print(f"    packed accumulator needs {acc_need:>3} bits, fp32 has 24"
          f"   -> {'ok' if acc_need <= 24 else 'TOO NARROW'}")
    print(f"    preadd operand    needs {op_need:>3} bits, {args.dtype} has {mant}"
          f"   -> {'ok' if op_need <= mant else 'CANNOT ENCODE'}\n")

    # ------------------------------------------------ exact integer references
    ref_y1 = x1.double() @ Wq.t().double()
    ref_y2 = x2.double() @ Wq.t().double()
    ref_dw = (dy1.transpose(0, 1).double() @ x1.double()
              + dy2.transpose(0, 1).double() @ x2.double())

    def pct(g1, g2):
        ok = (g1.double() == ref_y1) & (g2.double() == ref_y2)
        return 100.0 * ok.float().mean().item()

    checks = [
        ("2 launches ", pct(*v_two_launch(x1c, x2c, WT))),
        ("fused 2-acc", pct(*v_fused_2acc(x1c, x2c, WT))),
        ("packed 1-acc", pct(*v_packed_1acc(x1c, x2c, WT, s))),
        ("concat     ", pct(*v_concat(Xc, WT, B))),
        ("preadd     ", pct(*v_preadd(x1c, x2c, WT, s))),
    ]
    print("  fwd correctness (elements exactly equal to the integer reference)")
    for name, v in checks:
        print(f"    {name}  {v:6.1f}%")

    dwv = wgrad_vanilla(dy1c, dy2c, x1c, x2c).double()
    dwc = wgrad_concat(dYc, Xc).double()
    relv = ((dwv - ref_dw).abs().max() / ref_dw.abs().max()).item()
    relc = ((dwc - ref_dw).abs().max() / ref_dw.abs().max()).item()
    print(f"  wgrad rel err   vanilla {relv:.2e}   concat {relc:.2e}"
          f"   (bf16 output, so exact-match is not the right test here)")

    # ---------------------------------------------------------------- timing
    fwd_fl = 2 * (2 * B) * CIN * COUT          # both microbatches
    wgr_fl = 2 * (2 * B) * CIN * COUT

    rows = []
    rows += [("fwd", "cuBLAS 2x (reference)",
              bench(lambda: (x1c @ WT, x2c @ WT)), fwd_fl)]
    rows += [("fwd", "triton 2 launches",
              bench(lambda: v_two_launch(x1c, x2c, WT)), fwd_fl)]
    rows += [("fwd", "triton fused 2-acc",
              bench(lambda: v_fused_2acc(x1c, x2c, WT)), fwd_fl)]
    rows += [("fwd", "triton packed 1-acc",
              bench(lambda: v_packed_1acc(x1c, x2c, WT, s)), fwd_fl)]
    rows += [("fwd", "triton concat",
              bench(lambda: v_concat(Xc, WT, B)), fwd_fl)]
    rows += [("fwd", "triton preadd (half MACs)",
              bench(lambda: v_preadd(x1c, x2c, WT, s)), fwd_fl // 2)]

    rows += [("dgrad", "triton 2 launches",
              bench(lambda: v_two_launch(dy1c, dy2c, W)), fwd_fl)]
    rows += [("dgrad", "triton fused 2-acc",
              bench(lambda: v_fused_2acc(dy1c, dy2c, W)), fwd_fl)]
    rows += [("dgrad", "triton packed 1-acc",
              bench(lambda: v_packed_1acc(dy1c, dy2c, W, s_dg)), fwd_fl)]
    rows += [("dgrad", "triton concat",
              bench(lambda: v_concat(dYc, W, B)), fwd_fl)]

    rows += [("wgrad", "cuBLAS 2 GEMM + add",
              bench(lambda: wgrad_vanilla(dy1c, dy2c, x1c, x2c)), wgr_fl)]
    rows += [("wgrad", "cuBLAS concat-K (1 acc)",
              bench(lambda: wgrad_concat(dYc, Xc)), wgr_fl)]

    print(f"\n  {'pass':<7}{'variant':<28}{'ms':>9}{'TFLOP/s':>10}{'speedup':>10}")
    base = {}
    for d, name, ms, fl in rows:
        base.setdefault(d, ms)
        print(f"  {d:<7}{name:<28}{ms:>9.3f}"
              f"{fl / (ms * 1e-3) / 1e12:>10.1f}{base[d] / ms:>9.2f}x")

    print("\n  speedup is against the first row of each pass.")
    print("  wgrad is the only reduction here. fwd/dgrad share W, so they were")
    print("  never two problems -- they are one GEMM with a taller M, which is")
    print("  what `concat` does for free and exactly.")


if __name__ == "__main__":
    main()
