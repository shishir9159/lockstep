"""Which tensor-core datapath did the compiler actually pick?

On sm_90 this is not cosmetic. Warp-level `mma.sync` with e4m3 operands is
emulated: ptxas converts to FP16 and issues HMMA, so you get FP16 rate minus the
conversions. Only `wgmma.mma_async` reaches the native FP8 tensor core. Run this
before believing any FP4-on-Hopper throughput number.

    uv run check_isa.py
"""

import re
import torch
from triton.runtime.jit import JITFunction

import fp4
from h100 import require_h100
from mxfp4_gemm import _mxfp4_gemm_fp8_kernel, mxfp4_gemm_fp8
from dual_gemm import _dual_packed_kernel, dual_packed


def _kernels(fn):
    """Every compiled variant of a Triton kernel, including autotuned ones.

    @autotune wraps the JITFunction (its own .cache maps tuning keys to Configs,
    not kernels), and the JIT cache moved from .cache to .device_caches in 3.2.
    """
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    caches = getattr(fn, "device_caches", None)
    if caches is not None:
        return [k for entry in caches.values() for k in entry[0].values()]
    return [k for per_device in fn.cache.values() for k in per_device.values()]


def report(name, jit_fn):
    """Print the MMA instructions found across every compiled variant."""
    ks = _kernels(jit_fn)
    if not ks:
        print(f"  {name}: nothing compiled")
        return
    ptx = "\n".join(k.asm.get("ptx", "") for k in ks)
    mma = sorted(set(re.findall(r"\b(?:wgmma\.mma_async|mma)\.[a-z0-9_.]+", ptx)))
    mma = [m for m in mma if "fence" not in m and "commit" not in m and "wait" not in m]
    print(f"  {name}")
    for m in mma[:6]:
        native = "wgmma" in m
        print(f"      {m:<62}{'NATIVE fp8/bf16 datapath' if native else 'warp-level'}")
    if not mma:
        print("      (no mma found -- kernel may have been dead-code eliminated)")


def main():
    p = require_h100()
    print(f"{p.name} sm_{p.major}{p.minor}\n")

    M = N = K = 1024
    dev = "cuda"
    ca = torch.randint(0, 16, (M, K), device=dev, dtype=torch.uint8)
    cb = torch.randint(0, 16, (K, N), device=dev, dtype=torch.uint8)
    sa = torch.ones((M, K // 32), device=dev)
    sb = torch.ones((K // 32, N), device=dev)
    mxfp4_gemm_fp8(fp4.codes_to_e4m3(ca), fp4.codes_to_e4m3(cb), sa, sb)

    a = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=dev, dtype=torch.bfloat16)
    dual_packed(a, a, b, b)

    print("PTX instruction selection:")
    report("mxfp4 -> fp8 block-scaled GEMM", _mxfp4_gemm_fp8_kernel)
    report("dual packed accumulator GEMM", _dual_packed_kernel)
    print("\nOn sm_90 you want to see wgmma.mma_async.*.e4m3 for the FP4 kernel.")
    print("If you see mma.sync.*.e4m3 instead, you are running on the FP16")
    print("datapath at roughly half rate. Compare with cuda/mxfp4_mma_gemm.cu,")
    print("which disassembles to HMMA.16816 + F2FP.F16.E4M3 for exactly that reason.")


if __name__ == "__main__":
    main()
