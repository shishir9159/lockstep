#!/usr/bin/env bash
# Every H100 benchmark on a freshly rented machine, unattended (~10-15 min).
#
#   bash h100.sh
#
# Checks the GPU before installing anything, installs uv and just if missing,
# runs `just h100-all` and leaves results/latest.tar.gz. From your own machine,
# `just h100-remote user@host` does the upload, the run and the download.
set -euo pipefail
cd "$(dirname "$0")"
start=$(date +%s)

gpu=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader -i 0 2>/dev/null || true)
case "$gpu" in
    *H100*", 9.0") echo "GPU 0: $gpu" ;;
    *) echo "needs an H100; found: ${gpu:-no NVIDIA GPU}" >&2; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES=0

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v just >/dev/null || uv tool install rust-just
if ! command -v nvcc >/dev/null; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
    done
fi
command -v nvcc >/dev/null || echo "no nvcc: CUDA kernel and rig steps will fail" >&2

uv sync
just h100-all
echo "done in $(( ($(date +%s) - start) / 60 )) min: fetch results/latest.tar.gz, then stop the instance"
