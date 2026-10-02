# Step runner for the justfile recipes (sourced, not run). Every step is timed and
# recorded in $out/steps.csv with the GPUs it keeps busy and idle; a failed step is
# recorded and the run carries on.

GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
NVCC=(nvcc -O3 -std=c++17)
[ -n "${CCBIN:-}" ] && NVCC+=(-ccbin "$CCBIN")
METHODS="fp32:dense bf16:dense int16:dense int8ef:dense bf16:mxfp4 int8ef:mxfp4"

begin() {                       # begin <recipe>: creates results/<recipe>-<time> as $out
    out="results/$1-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"
    echo "step,gpus_busy,gpus_idle,seconds,status" > "$out/steps.csv"
}

step() {                        # step <name> <GPUs busy> <command...>
    local name=$1 busy=$2 t0 rc
    shift 2
    local idle=$(( GPUS > busy ? GPUS - busy : 0 ))
    printf '\n== %s  [%s of %s GPUs busy, %s idle]\n' "$name" "$busy" "$GPUS" "$idle"
    t0=$(date +%s.%N)
    "$@"
    rc=$?
    printf '%s,%s,%s,%s,%s\n' "$name" "$busy" "$idle" \
        "$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')" \
        "$([ "$rc" -eq 0 ] && echo ok || echo FAILED)" >> "$out/steps.csv"
}

finish() {
    uv run report.py "$out"
}

env_info() {
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
    nvcc --version | tail -1
    uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
}

build() {                       # build <arch>: both rigs
    "${NVCC[@]}" -arch="$1" -o rig_sm75/rig rig_sm75/rig.cu &&
        "${NVCC[@]}" -arch="$1" -o rig_sm75/rig_q rig_sm75/rig_q.cu
}

rig_q_gpu() {                   # the rig_q experiments timed on the GPU; the rest are CPU-only
    local e
    for e in narrow interleave link; do ./rig_sm75/rig_q "$e" || return 1; done
}

cuda_kernel() {                 # Hopper block-scaled GEMM: exactness, then 4096^3 timing
    "${NVCC[@]}" -arch=sm_90a -o cuda/mxfp4 cuda/mxfp4_mma_gemm.cu &&
        ./cuda/mxfp4 --check && ./cuda/mxfp4 4096 4096 4096
}

sass() {                        # which tensor-core instructions ptxas emitted
    "${NVCC[@]}" -arch=sm_90a --resource-usage -cubin -o cuda/mxfp4.cubin cuda/mxfp4_mma_gemm.cu &&
        cuobjdump -sass cuda/mxfp4.cubin |
        grep -oE "HMMA[^ ]*|QGMMA[^ ]*|HGMMA[^ ]*|F2FP[^ ]*|LDGSTS[^ ]*" | sort | uniq -c
}

train_all() {                   # train_all <preset> <tokens> <GPUs busy> <launcher...>
    local preset=$1 tokens=$2 busy=$3 m
    shift 3
    for m in $METHODS; do
        step "train-${m%:*}-${m#*:}" "$busy" "$@" train.py --preset "$preset" --tokens "$tokens" \
            --wire "${m%:*}" --gemm "${m#*:}" --out "$out"
    done
}

h100_kernels() {                # h100_kernels <quick|full>: the single-GPU H100 steps
    step cuda-kernel 1 cuda_kernel
    step sass 0 sass
    step isa 1 uv run check_isa.py
    step bench 1 uv run bench.py --json "$out/bench.json"
    step train-step 1 uv run train_step.py --json "$out/train_step.json"
    step build 0 build sm_90a
    if [ "$1" = quick ]; then           # the packing-as-transport check only
        step rig-splitk 1 ./rig_sm75/rig splitk
        step rig_q-narrow 1 ./rig_sm75/rig_q narrow
    else
        step rig 1 ./rig_sm75/rig
        step rig_q 1 rig_q_gpu
    fi
}
