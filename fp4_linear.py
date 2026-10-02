"""MXFP4 linear layer: FP4 forward, straight-through backward at the input's precision.

On an H100 the forward runs the Triton FP8 kernel (mxfp4_gemm.mxfp4_gemm_packed).
Anywhere else it multiplies the dequantized MXFP4 values, which is the same math.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import fp4


@functools.cache
def kernel():
    """The Triton MXFP4 GEMM on an H100, else None."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        return None
    try:
        from mxfp4_gemm import mxfp4_gemm_packed
    except ImportError:
        return None
    return mxfp4_gemm_packed


def path():
    return "triton-fp8" if kernel() else "fake-quant"


class _MXFP4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        xc, xs = fp4.quantize_mxfp4(x)
        wc, ws = fp4.quantize_mxfp4(w)
        if (k := kernel()) is not None:
            return k(fp4.pack_fp4(xc), fp4.pack_fp4(wc), xs, ws, out_dtype=x.dtype)
        return F.linear(fp4.dequantize_mxfp4(xc, xs), fp4.dequantize_mxfp4(wc, ws)).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        return (gy @ w.to(gy.dtype)).to(x.dtype), (gy.t() @ x.to(gy.dtype)).to(w.dtype)


class MXFP4Linear(nn.Linear):
    """nn.Linear whose forward GEMM runs in MXFP4 (bias unsupported)."""

    def __init__(self, fan_in, fan_out, bias=False):
        assert not bias, "MXFP4Linear has no bias"
        super().__init__(fan_in, fan_out, bias=False)

    def forward(self, x):
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        with torch.autocast("cuda", enabled=False):
            y = _MXFP4Matmul.apply(x.reshape(-1, x.shape[-1]).to(dt), self.weight)
        return y.view(*x.shape[:-1], -1)
