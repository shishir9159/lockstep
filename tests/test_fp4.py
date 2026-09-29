"""CPU tests for fp4.py, and its agreement with the CUDA tables. No GPU needed."""

import re
from pathlib import Path

import pytest
import torch

import fp4

ROOT = Path(__file__).resolve().parents[1]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_e4m3_table_decodes_to_e2m1_values():
    codes = torch.arange(16, dtype=torch.uint8)
    got = fp4.codes_to_e4m3(codes).float()
    want = torch.tensor(E2M1 + [-v for v in E2M1])
    assert torch.equal(got, want)


def test_integer_grid_is_twice_the_value():
    codes = torch.arange(16, dtype=torch.uint8)
    assert torch.equal(fp4.codes_to_q(codes).float(), 2 * fp4.E2M1_VALUES)


def test_cuda_prmt_tables_match_python_table():
    src = (ROOT / "cuda" / "fp4_unpack.cuh").read_text()
    lut = {k: int(v, 16) for k, v in re.findall(r"#define (FP4_LUT_\w+)\s+(0x[0-9A-Fa-f]+)", src)}
    table = (lut["FP4_LUT_LO"] | lut["FP4_LUT_HI"] << 32).to_bytes(8, "little")
    assert list(table) == fp4.E2M1_TO_E4M3[:8].tolist()


def test_pack_unpack_roundtrip():
    codes = fp4.random_codes((7, 64))
    packed = fp4.pack_fp4(codes)
    assert packed.shape == (7, 32)
    assert torch.equal(fp4.unpack_fp4(packed), codes)
    assert torch.equal(packed[0, 0] & 0x0F, codes[0, 0])        # even index in low nibble


def test_random_codes_skip_negative_zero():
    codes = fp4.random_codes((20000,))
    assert not (codes == 8).any()
    assert set(codes.unique().tolist()) == {c for c in range(16) if c != 8}


def test_ties_go_to_smaller_magnitude():
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    got = fp4.E2M1_VALUES[fp4._round_to_e2m1(mids).long()]
    assert torch.equal(got, torch.tensor(E2M1[:7]))


@pytest.mark.parametrize("block", [16, 32])
def test_quantize_mxfp4_properties(block):
    torch.manual_seed(0)
    x = torch.randn(64, 256) * torch.logspace(-3, 3, 64).unsqueeze(1)
    codes, scale = fp4.quantize_mxfp4(x, block)
    # E8M0: every scale is an exact power of two.
    assert torch.equal(torch.exp2(torch.log2(scale).round()), scale)
    # amax lands in [4, 8) grid units, so nothing larger than 8x the scale.
    xb = x.reshape(64, -1, block)
    amax = xb.abs().amax(-1)
    assert ((amax / scale >= 4 * (1 - 1e-6)) & (amax / scale < 8)).all()
    # Round-to-nearest with clipping at 6: error per element is bounded.
    y = fp4.dequantize_mxfp4(codes, scale, block).reshape(64, -1, block)
    err = (y - xb).abs() / scale.unsqueeze(-1)
    assert (err <= 2.0 + 1e-6).all()             # worst case: 8 clipped to 6


def test_zero_block_is_exact():
    codes, scale = fp4.quantize_mxfp4(torch.zeros(2, 32))
    assert torch.equal(scale, torch.ones(2, 1))
    assert torch.equal(fp4.dequantize_mxfp4(codes, scale), torch.zeros(2, 32))


def test_blockwise_exact_dot_bound_and_value():
    qa = fp4.codes_to_q(fp4.random_codes((4, 128)))
    qb = fp4.codes_to_q(fp4.random_codes((4, 128)))
    blocks = fp4.blockwise_exact_dot(qa, qb, 32)
    assert blocks.abs().max() <= 144 * 32
    assert torch.equal(blocks.sum(-1), (qa.long() * qb.long()).sum(-1))


def test_reference_gemm_matches_dequantized_matmul():
    torch.manual_seed(1)
    a_codes, a_scale = fp4.quantize_mxfp4(torch.randn(8, 64))
    bt_codes, bt_scale = fp4.quantize_mxfp4(torch.randn(16, 64))
    ref = fp4.reference_mxfp4_gemm(a_codes, a_scale, bt_codes.t(), bt_scale.t())
    a = fp4.dequantize_mxfp4(a_codes, a_scale).double()
    b = fp4.dequantize_mxfp4(bt_codes, bt_scale).double()
    assert torch.allclose(ref.double(), a @ b.t(), rtol=1e-6)


def test_dual_slot_split():
    dg = pytest.importorskip("dual_gemm")       # imports triton: Linux only
    for k in (8, 32, 128):
        s = dg.slot_offset(k)
        c1 = torch.tensor([144.0 * k, -144.0 * k, 3.0], dtype=torch.float64)
        c2 = torch.tensor([-144.0 * k, 144.0 * k, -7.0], dtype=torch.float64)
        g1, g2 = dg.unpack_dual(c1 + c2 * 2.0**s, s)
        assert torch.equal(g1, c1) and torch.equal(g2, c2)
