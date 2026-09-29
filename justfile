# Each recipe's comment is the question it answers. Results: FINDINGS.md and
# rig_sm75/README.md. On Windows set CCBIN to MSVC's Hostx64/x64 first.

set windows-shell := ["bash", "-c"]

ccbin := env_var_or_default("CCBIN", "")
nvcc_flags := "-O3 -std=c++17" + if ccbin != "" { " -ccbin \"" + ccbin + "\"" } else { "" }
arch := "sm_75"          # override: just arch=sm_90a rig

_default:
    @just --list --unsorted

# Build both portable rigs
build:
    nvcc {{nvcc_flags}} -arch={{arch}} -o rig_sm75/rig rig_sm75/rig.cu
    nvcc {{nvcc_flags}} -arch={{arch}} -o rig_sm75/rig_q rig_sm75/rig_q.cu

# [1]-[5], or one of: unpack bits reduce gemm splitk
rig name="": build
    ./rig_sm75/rig {{name}}

# [6]-[19], or one experiment by name
rig-q name="": build
    ./rig_sm75/rig_q {{name}}

# [1] Is FP4 -> FP8 via prmt lossless? (exhaustive)
unpack: build
    ./rig_sm75/rig unpack

# [2] How many accumulator bits does a packed dual dot product need?
bits: build
    ./rig_sm75/rig bits

# [3] Upper bound: what does halving split-K accumulators buy?
reduce: build
    ./rig_sm75/rig reduce

# [4] Is the fwd speedup packing, or loading the shared operand once?
gemm: build
    ./rig_sm75/rig gemm

# [5] Do packed int16 partials work as a split-K transport format?
splitk: build
    ./rig_sm75/rig splitk

# [6] Is [5]'s win the pairing, or just 16-bit partials?
narrow: build
    ./rig_sm75/rig_q narrow

# [7] Is the accumulator or the operand lane the wall for packing?
int8acc: build
    ./rig_sm75/rig_q int8acc

# [8] MXFP4 vs NVFP4: accuracy per bit, clean and with outliers
nvfp4: build
    ./rig_sm75/rig_q nvfp4

# [9] What does 4-bit buy over MXFP8/BF16 at equal FLOPs?
mxfp8: build
    ./rig_sm75/rig_q mxfp8

# [10] Does one nibble-interleaved stream beat two FP4 arrays?
interleave: build
    ./rig_sm75/rig_q interleave

# [11] Does off-chip time scale with payload bytes?
link: build
    ./rig_sm75/rig_q link

# [12] How dense can partials get if the result is normalized?
dense: build
    ./rig_sm75/rig_q dense

# [13] At equal bytes, what are closure, bits and topology each worth?
fair: build
    ./rig_sm75/rig_q fair

# [14] Does the integer wire survive 1B-405B parameters?
llm: build
    ./rig_sm75/rig_q llm

# [15] Does error feedback make a 1-byte wire usable over many steps?
ef: build
    ./rig_sm75/rig_q ef

# [16] Can last step's grid replace the scale collective?
predict: build
    ./rig_sm75/rig_q predict

# [17] Two-level NVLink + IB reduction: numerics and time
tiers: build
    ./rig_sm75/rig_q tiers

# [18] Which ideas transfer to an MoE all-to-all?
moe: build
    ./rig_sm75/rig_q moe

# [19] Does keeping partials integral inside the GEMM buy anything?
chain: build
    ./rig_sm75/rig_q chain

# CPU tests (pytest) and the exhaustive FP4 table check; no GPU needed
test:
    uv run pytest
    cc -O2 -I cuda -o cuda/test_unpack cuda/test_unpack.c
    ./cuda/test_unpack > /dev/null && echo "test_unpack: 0 mismatches"

lint:
    uv run ruff check .

# Hopper block-scaled GEMM: correctness, then 4096^3 throughput (sm_90a)
h100:
    nvcc {{nvcc_flags}} -arch=sm_90a -o cuda/mxfp4 cuda/mxfp4_mma_gemm.cu
    ./cuda/mxfp4 --check
    ./cuda/mxfp4 4096 4096 4096

# Which tensor-core path did ptxas pick for the CUDA kernel?
sass:
    nvcc {{nvcc_flags}} -arch=sm_90a --resource-usage -cubin -o cuda/mxfp4.cubin cuda/mxfp4_mma_gemm.cu
    cuobjdump -sass cuda/mxfp4.cubin | grep -oE "HMMA[^ ]*|QGMMA[^ ]*|HGMMA[^ ]*|F2FP[^ ]*|LDGSTS[^ ]*" | sort | uniq -c

# Triton: both H100 experiments at 4096^3
bench:
    uv run bench.py

# Triton: one Linear layer fwd + bwd, vanilla vs packed
train:
    uv run train_step.py

# Triton: did the FP4 kernel get wgmma?
isa:
    uv run check_isa.py

clean:
    rm -f rig_sm75/rig rig_sm75/rig_q rig_sm75/*.exe rig_sm75/*.exp rig_sm75/*.lib rig_sm75/*.obj
    rm -f cuda/mxfp4 cuda/mxfp4.exe cuda/mxfp4.cubin cuda/*.obj cuda/test_unpack cuda/test_unpack.exe
    rm -rf __pycache__ tests/__pycache__ .ruff_cache .pytest_cache
