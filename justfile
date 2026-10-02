# One recipe per machine, GPU steps only. Each writes results/<recipe>-<time>/:
# matrix.md (methods x metrics, with time to finish), report.html, CSVs, run.log.
# The -10m versions check that the hypotheses hold before paying for a full run.
#
#   just turing          GTX 1650/1660 SUPER (sm_75)        ~10 min
#   just h100-10m        one H100, quick check              ~10 min (est.)
#   just h100            one H100, full                     ~30 min (est.)
#   just h100-node-10m   every GPU of an H100 node, quick   ~10 min (est.)
#   just h100-node       every GPU of an H100 node, full    ~30 min on 8 GPUs (est.)
#
# Fresh machine: curl -LsSf https://astral.sh/uv/install.sh | sh && uv tool install rust-just
# Windows: export CCBIN="<MSVC>/bin/Hostx64/x64" first.

set windows-shell := ["bash", "-c"]

_default:
    @just --list --unsorted

# GTX 1650/1660 SUPER: GPU-timed rig experiments, tiny-GPT method matrix (1M tokens)
turing: (_guard "turing")
    #!/usr/bin/env bash
    set -uo pipefail
    source run.sh
    begin turing
    {
        step env 0 env_info
        step build 0 build sm_75
        step rig 1 ./rig_sm75/rig
        step rig_q 1 rig_q_gpu
        step data 0 uv run data.py --shards 1
        train_all tiny 1000000 1 uv run
    } 2>&1 | tee "$out/run.log"
    finish

# One H100, quick: kernels, packing check, GPT-2 method matrix at 3M tokens
h100-10m: (_guard "h100")
    #!/usr/bin/env bash
    set -uo pipefail
    source run.sh
    export CUDA_VISIBLE_DEVICES=0                      # on a multi-GPU box the others idle
    begin h100-10m
    {
        step env 0 env_info
        h100_kernels quick
        step data 0 uv run data.py --shards 1
        train_all gpt2 3000000 1 uv run
    } 2>&1 | tee "$out/run.log"
    finish

# One H100, full: kernels, every GPU rig experiment, GPT-2 method matrix at 20M tokens
h100: (_guard "h100")
    #!/usr/bin/env bash
    set -uo pipefail
    source run.sh
    export CUDA_VISIBLE_DEVICES=0                      # on a multi-GPU box the others idle
    begin h100
    {
        step env 0 env_info
        h100_kernels full
        step data 0 uv run data.py --shards 1
        train_all gpt2 20000000 1 uv run
    } 2>&1 | tee "$out/run.log"
    finish

# H100 node, quick: wire microbench, GPT-2 methods at 20M tokens on all GPUs, kernels
h100-node-10m: (_guard "node")
    #!/usr/bin/env bash
    set -uo pipefail
    source run.sh
    begin h100-node-10m
    tr=(uv run torchrun --standalone --nproc-per-node=gpu)   # ranks from torchrun's env
    {
        step env 0 env_info
        step data 0 uv run data.py --shards 1                # all GPUs idle
        step wire "$GPUS" "${tr[@]}" wire.py --out "$out"     # all GPUs busy
        train_all gpt2 20000000 "$GPUS" "${tr[@]}"           # all GPUs busy
        (export CUDA_VISIBLE_DEVICES=0; h100_kernels quick)  # 1 GPU busy, the rest idle
    } 2>&1 | tee "$out/run.log"
    finish

# For several nodes, launch train.py and wire.py with torchrun or srun yourself: they
# read RANK/WORLD_SIZE/LOCAL_RANK or SLURM_* from the environment.
# H100 node, full: wire microbench, GPT-2 methods at 200M tokens on all GPUs, kernels
h100-node: (_guard "node")
    #!/usr/bin/env bash
    set -uo pipefail
    source run.sh
    begin h100-node
    tr=(uv run torchrun --standalone --nproc-per-node=gpu)   # ranks from torchrun's env
    {
        step env 0 env_info
        step data 0 uv run data.py --shards 2                # all GPUs idle
        step wire "$GPUS" "${tr[@]}" wire.py --out "$out"     # all GPUs busy
        train_all gpt2 200000000 "$GPUS" "${tr[@]}"          # all GPUs busy
        (export CUDA_VISIBLE_DEVICES=0; h100_kernels full)   # 1 GPU busy, the rest idle
    } 2>&1 | tee "$out/run.log"
    finish

# Fails unless the GPUs match the recipe
_guard kind:
    #!/usr/bin/env bash
    gpus=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)
    n=$(printf '%s\n' "$gpus" | grep -c .)
    h100=$(printf '%s\n' "$gpus" | grep -c 'H100.*, 9\.0$')
    first=$(printf '%s\n' "$gpus" | sed -n 1p)
    case "{{kind}}" in
        turing) [[ "$first" == *", 7.5" ]] ;;
        h100)   [[ "$first" == *H100*", 9.0" ]] ;;
        node)   [ "$n" -ge 2 ] && [ "$h100" -eq "$n" ] ;;
    esac || { echo "'{{kind}}' does not match this machine: ${gpus:-no NVIDIA GPU}" >&2; exit 1; }

# One rig experiment by name, e.g. `just rig-q fair` (any sm_75+ GPU)
[private]
rig-q name="" arch="sm_75":
    bash -c 'source run.sh && build {{arch}}' && ./rig_sm75/rig_q {{name}}

[private]
rig name="" arch="sm_75":
    bash -c 'source run.sh && build {{arch}}' && ./rig_sm75/rig {{name}}
