#!/usr/bin/env bash
# Run a disk-backed KV-cache model under a hard memory budget:
#   VRAM  <= the GPUs (model weights + GPU compute)
#   RAM   <= $RAM_MAX (warm tier: mmap page cache, MADV_SEQUENTIAL, evicts to disk)
#   DISK  =  the full KV cache cold tier on $KV_DISK
#
# Attention runs on the CPU over the host/disk-backed KV; the model weights and
# all other layers run on the GPU. See the "GPU attention" note in the project
# notes for the remaining (custom paged-attention) work.
#
# Usage: ./run-nemo-disk.sh [n_ctx] [extra llama-completion args...]
set -euo pipefail

MODEL=${MODEL:-models/nemo/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf}
KV_DISK=${KV_DISK:-/media/moe/FastDisk}
RAM_MAX=${RAM_MAX:-48G}
PAGE_TOKENS=${PAGE_TOKENS:-8192}
NGL=${NGL:-99}
BIN=${BIN:-./build-cuda/bin/llama-completion}
N_CTX=${1:-1000000}
shift || true

echo "model=$MODEL  n_ctx=$N_CTX  RAM_MAX=$RAM_MAX  KV_DISK=$KV_DISK  ngl=$NGL"

# A memory cgroup bounds the warm tier (mmap page cache) to RAM_MAX. The cache is
# reclaimable, so reaching the limit evicts clean KV pages back to disk rather
# than OOM-killing (unlike CUDA unified memory, which is unevictable).
exec systemd-run --user --scope -p MemoryMax="$RAM_MAX" -p MemorySwapMax=0 \
    bash -c 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} "$0" \
        -m "$1" -c "$2" -ngl "$3" --kv-disk-path "$4" --kv-page-tokens "$5" \
        -fa off -no-cnv "${@:6}"' \
    "$BIN" "$MODEL" "$N_CTX" "$NGL" "$KV_DISK" "$PAGE_TOKENS" "$@"
