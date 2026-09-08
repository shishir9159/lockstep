# intra-gpu-reduction
#
#   just              list recipes
#   just rig          experiments [1]-[5]: does the packing idea work at all?
#   just rig-q        experiments [6]-[19]: quantization formats and transport
#   just h100         the Hopper experiments (needs an H100)
#
# Every recipe says what QUESTION its experiment answers. Most of them exist to
# control a claim an earlier version of this repo got wrong, so the goal line
# matters as much as the code.

set windows-shell := ["bash", "-c"]

# MSVC is only needed on Windows; on Linux leave CCBIN empty.
ccbin := env_var_or_default("CCBIN", "")
nvcc_flags := "-O3 -std=c++17" + if ccbin != "" { " -ccbin \"" + ccbin + "\"" } else { "" }

_default:
    @just --list

# Builds for sm_75 by default. Override: just arch=sm_90a rig
arch := "sm_75"

# Build both portable binaries (rig, rig_q)
build:
    nvcc {{nvcc_flags}} -arch={{arch}} -o rig_sm75/rig rig_sm75/rig.cu
    nvcc {{nvcc_flags}} -arch={{arch}} -o rig_sm75/rig_q rig_sm75/rig_q.cu

# ------------------------------------------------- [1]-[5]  does the idea work?

# [1]-[5]: unpack, bit budget, reduction traffic, fwd/bwd attribution, split-K
rig: build
    ./rig_sm75/rig

# GOAL: prove FP4 -> FP8 costs no accuracy, only two PRMT. Exhaustive over 16^4.
# [1] is the FP4 -> FP8 expansion lossless?
unpack: build
    ./rig_sm75/rig unpack

# GOAL: kill or confirm accumulator-packing with arithmetic instead of opinion.
# Answer: 27 bits at K=32, against fp32's 24 and the Hopper FP8 path's ~14.
# [2] how many significand bits does a packed dual dot product need?
bits: build
    ./rig_sm75/rig bits

# GOAL: measure the honest UPPER BOUND on what halving partials can ever buy,
# with nothing else in the way.
# [3] split-K reduction traffic, 1 accumulator vs 2
reduce: build
    ./rig_sm75/rig reduce

# GOAL: attribution. An earlier version reported 1.85x for packing; adding the
# fused control showed the win was entirely operand reuse. Never drop a control.
# [4] minimal fwd + wgrad: 2 launches / concat / fused-2-acc / packed-1-acc
gemm: build
    ./rig_sm75/rig gemm

# GOAL: the repaired idea -- packing as TRANSPORT, never as accumulation, with
# partials going to memory for a second kernel instead of to atomics.
# [5] split-K storing packed int16 partials
splitk: build
    ./rig_sm75/rig splitk

# --------------------------------------------- [6]-[14]  formats and transport

# [6]-[19]: formats, controls, the interconnect, scale, feedback, MoE, the full chain
rig-q: build
    ./rig_sm75/rig_q

# GOAL: the control [5] was missing. Both int16 rows move IDENTICAL bytes, so a
# gap between them is the pairing alone and a gap to int32x2 is the narrowing.
# Decides whether the write-up says "packing" or "narrowing".
# [6] int32x2 vs int16x2 vs packed-int16 -- packing or narrowing?
narrow: build
    ./rig_sm75/rig_q narrow

# GOAL: separate the ACCUMULATOR question from the OPERAND question, so we learn
# which one actually kills packed accumulation. INT8 is the only Hopper datapath
# with a true 32-bit integer accumulator (fp32 has 24 significand bits, FP8 ~14).
# Prints a worst-case column beside the observed one so random-data success
# cannot be mistaken for a guarantee.
# [7] INT8/s32 -- is the accumulator the problem, or the operand?
int8acc: build
    ./rig_sm75/rig_q int8acc

# Host-only exhaustive check of the FP4 table, no GPU needed
test-unpack:
    cc -O2 -I cuda -o cuda/test_unpack cuda/test_unpack.c
    ./cuda/test_unpack

# ------------------------------------------------------------------ H100 side

h100-build:
    nvcc {{nvcc_flags}} -arch=sm_90a -o cuda/mxfp4 cuda/mxfp4_mma_gemm.cu

# The Hopper block-scaled GEMM: correctness then throughput (needs an H100)
h100: h100-build
    ./cuda/mxfp4 --check
    ./cuda/mxfp4 4096 4096 4096

# Which tensor-core datapath did ptxas actually pick? (wgmma vs emulated mma.sync)
sass:
    nvcc {{nvcc_flags}} -arch=sm_90a --resource-usage -cubin -o cuda/mxfp4.cubin cuda/mxfp4_mma_gemm.cu
    cuobjdump -sass cuda/mxfp4.cubin | grep -oE "HMMA[^ ]*|QGMMA[^ ]*|HGMMA[^ ]*|F2FP[^ ]*|LDGSTS[^ ]*" | sort | uniq -c

# One Linear layer, fwd + bwd, vanilla vs the technique (Triton, needs an H100)
train:
    uv run train_step.py

# Both Hopper experiments at 4096^3 (Triton, needs an H100)
bench:
    uv run bench.py

# Confirm which tensor-core path Triton selected (needs an H100)
isa:
    uv run check_isa.py

lint:
    uvx ruff check .

clean:
    rm -f rig_sm75/rig rig_sm75/rig.exe rig_sm75/rig_q rig_sm75/rig_q.exe
    rm -f cuda/mxfp4 cuda/mxfp4.exe cuda/mxfp4.cubin
    rm -f cuda/test_unpack cuda/test_unpack.exe cuda/*.obj rig_sm75/*.obj
    rm -rf __pycache__ .ruff_cache
