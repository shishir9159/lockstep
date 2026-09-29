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

# ------------------------------------------------ H100 only: each recipe checks GPU 0

# Exit unless GPU 0 is an H100 (sm_90)
[private]
h100-guard:
    #!/usr/bin/env bash
    gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i 0 2>/dev/null)
    case "$gpu" in
        *H100*", 9.0") echo "$gpu" ;;
        *) echo "needs an H100; found: ${gpu:-no NVIDIA GPU}" >&2; exit 1 ;;
    esac

# Driver, nvcc, torch and triton versions
h100-env: h100-guard
    nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader -i 0
    nvcc --version | tail -1
    uv run python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)"

# CUDA block-scaled GEMM: bit-exactness, then 4096^3 throughput
h100: h100-guard
    nvcc {{nvcc_flags}} -arch=sm_90a -o cuda/mxfp4 cuda/mxfp4_mma_gemm.cu
    ./cuda/mxfp4 --check
    ./cuda/mxfp4 4096 4096 4096

# Which tensor-core instructions ptxas emitted (HMMA = FP16 path, QGMMA = FP8 wgmma)
sass:
    nvcc {{nvcc_flags}} -arch=sm_90a --resource-usage -cubin -o cuda/mxfp4.cubin cuda/mxfp4_mma_gemm.cu
    cuobjdump -sass cuda/mxfp4.cubin | grep -oE "HMMA[^ ]*|QGMMA[^ ]*|HGMMA[^ ]*|F2FP[^ ]*|LDGSTS[^ ]*" | sort | uniq -c

# Triton: MXFP4 exactness and throughput, then the packed dual GEMM
bench: h100-guard
    uv run bench.py

# Triton: one Linear layer fwd + bwd, vanilla vs packed
train: h100-guard
    uv run train_step.py

# Triton: did the FP4 kernel get wgmma?
isa: h100-guard
    uv run check_isa.py

# The portable rigs, rebuilt for sm_90a
h100-rig: h100-guard
    {{just_executable()}} arch=sm_90a rig
    {{just_executable()}} arch=sm_90a rig-q

# Everything above into results/<time>/: log, JSON, charts, report.html, tarball
h100-all: h100-guard
    #!/usr/bin/env bash
    set -uo pipefail
    out="results/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"
    j="{{just_executable()}}"
    step() { local name=$1; shift; printf '\n== %s\n' "$name"; "$@" || echo "$name" >> "$out/failed.txt"; }
    {
        step env   "$j" h100-env
        step test  "$j" test
        step cuda  "$j" h100
        step sass  "$j" sass
        step isa   uv run check_isa.py
        step bench uv run bench.py --json "$out/bench.json"
        step train uv run train_step.py --json "$out/train.json"
        step build "$j" arch=sm_90a build
        step rig   ./rig_sm75/rig
        step rig_q ./rig_sm75/rig_q
    } 2>&1 | tee "$out/run.log"
    echo
    uv run report.py "$out"
    tar -czf "$out.tar.gz" -C results "$(basename "$out")"
    host="$(whoami)@$(hostname -f 2>/dev/null || hostname)"
    echo "fetch: scp $host:$PWD/$out.tar.gz ."
    echo "view:  just h100-serve   (then on your machine: ssh -N -L 8000:localhost:8000 $host)"

# Serve results/ on localhost; open it from your machine through an ssh tunnel
h100-serve port="8000":
    @echo "on your machine: ssh -N -L {{port}}:localhost:{{port}} $(whoami)@$(hostname -f 2>/dev/null || hostname)"
    @echo "then open http://localhost:{{port}} and click into the run folder"
    uv run python -m http.server {{port}} --bind 127.0.0.1 --directory results

clean:
    rm -f rig_sm75/rig rig_sm75/rig_q rig_sm75/*.exe rig_sm75/*.exp rig_sm75/*.lib rig_sm75/*.obj
    rm -f cuda/mxfp4 cuda/mxfp4.exe cuda/mxfp4.cubin cuda/*.obj cuda/test_unpack cuda/test_unpack.exe
    rm -rf __pycache__ tests/__pycache__ .ruff_cache .pytest_cache
