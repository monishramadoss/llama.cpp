#!/usr/bin/env bash
# Serve NVIDIA-Nemotron-3-Super-120B-A12B (hybrid Mamba2 + attention MoE) at a
# 1,000,000-token context on a small box (2x GTX 1080 Ti, 62 GB RAM).
#
# Memory plan for the 120B / Q4_K_M (~82 GB) weights:
#   GPU  (-ngl 99)     : the DENSE path of every block -- attention (+ its KV),
#                        Mamba2 SSM, norms, router, shared expert. ~10-22 GB.
#   RAM  (--cpu-moe)   : the 512 routed-expert FFNs (the bulk of the 82 GB),
#                        mmap'd from the GGUF on $MODEL_DIR so the OS page cache
#                        keeps the hot experts resident and pages the rest.
#
# Native context is 1,048,576 -- NO rope/YaRN scaling needed (unlike Nemo).
# The model is a hybrid: most of the 88 blocks are Mamba2 (constant-size
# recurrent state), so the attention KV cache stays small even at 1M tokens.
# That is why this needs no disk-backed KV by default; flip KV_DISK=1 to use it.
#
# Usage: ./run-nemotron-super.sh [n_ctx] [extra llama-completion args...]
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/media/moe/FastDisk/models/nemotron3-super}
MODEL=${MODEL:-$MODEL_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf}
BIN=${BIN:-./build-cuda/bin/llama-completion}
NGL=${NGL:-99}
N_CTX=${1:-1000000}
shift || true

# Routed experts -> CPU/RAM, everything else -> GPU. This is the knob that makes
# a 120B MoE fit two 11 GB GPUs. If VRAM still OOMs at load, lower NGL (e.g. 40)
# so some blocks' dense path also falls back to CPU.
CPU_MOE_ARGS=(--cpu-moe)

# Optional disk-backed KV cold tier (your paged-attention path). Off by default
# because the hybrid KV is small; set KV_DISK=1 to exercise it.
KV_ARGS=()
if [[ "${KV_DISK:-0}" == "1" ]]; then
    KV_DISK_PATH=${KV_DISK_PATH:-/media/moe/FastDisk}
    KV_PAGE_TOKENS=${KV_PAGE_TOKENS:-8192}
    KV_ARGS=(--kv-offload-disk --kv-disk-path "$KV_DISK_PATH" --kv-page-tokens "$KV_PAGE_TOKENS")
fi

echo "model=$MODEL"
echo "n_ctx=$N_CTX  ngl=$NGL  cpu_moe=on  kv_disk=${KV_DISK:-0}  gpus=${CUDA_VISIBLE_DEVICES:-0,1}"

# mmap stays ON (do NOT pass --no-mmap): the 82 GB of weights exceed RAM, so the
# experts must be paged from the GGUF rather than fully resident.
exec env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
    "$BIN" \
        -m "$MODEL" \
        -c "$N_CTX" \
        -ngl "$NGL" \
        "${CPU_MOE_ARGS[@]}" \
        "${KV_ARGS[@]}" \
        -fa off \
        -no-cnv \
        "$@"
