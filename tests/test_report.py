import report

RIG = """
[5] split-K with packed int16 partials, M=256 N=256 K=8192, dp4a GEMM

      32   256  OVERFL      1.512      0.097      1.609  int32x2  exact
                            1.534      0.052      1.586  packed   exact  1.01x
                            1.575          -      1.575  atomics
"""

RIGQ = """
[13] all-reduce at equal bytes: closure, bits, and topology
      128  int16, direct                            2.03       0.0042%
      128  fp16, direct                             2.00       0.0290%
[14] scaling to 1B+ parameters
[15] error feedback over 128 steps, 1024 elements, 32 ranks
    int8 direct, EF           1.031       4.8496%     0.096     0.101     0.072   0.0000%   0.0000%
[16] predicting the grid: one collective instead of two, 128 steps
    int16 direct, predicted x2.00 + EF     0.0038%   0.0000%       1.1689%          0.0050%
[17] hierarchical reduction: numerics, then time
"""


def test_sections_split_on_step_markers():
    s = report.sections("\n== env\nNVIDIA H100 80GB HBM3, 9.0\n\n== rig\nx\n")
    assert s["env"].strip() == "NVIDIA H100 80GB HBM3, 9.0" and s["rig"].strip() == "x"


def test_rig_tables_parse():
    assert report.parse_splitk(RIG) == {
        32: {"int32 x2": 1.609, "packed int16": 1.586, "atomics": 1.575}}
    assert report.parse_fair(RIGQ) == {"int16, direct": {128: 0.0042}, "fp16, direct": {128: 0.029}}
    assert report.parse_ef(RIGQ) == {"int8 direct, EF": [0.096, 0.101, 0.072]}
    assert report.parse_predict(RIGQ) == {"int16 direct, predicted x2.00 + EF": 0.005}
