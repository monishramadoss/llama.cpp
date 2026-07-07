#pragma once

#include "common.cuh"

// Build a GPU paged-attention node: attention over a host/disk-resident KV cache
// (K/V stay in host memory; chunks are staged into a bounded VRAM scratch). The
// returned node is a custom op that runs on the CPU backend but performs its
// compute on the GPU, so the KV cache can far exceed VRAM.
//
//   q    : [d_k, n_q,  n_head,    n_stream]   (compute backend tensor)
//   k    : [d_k, n_kv, n_head_kv, n_stream]   (host/disk KV, F16 or F32)
//   v    : [n_kv, d_v, n_head_kv, n_stream]   (host/disk KV, F16 or F32)
//   mask : [n_kv, n_q(.. padded)] additive, or nullptr
//   returns kqv [d_v, n_q, n_head, n_stream]
//
// Exposed via ggml_backend_reg_get_proc_address("ggml_cuda_paged_attn").
typedef struct ggml_tensor * (*ggml_cuda_paged_attn_t)(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * mask,
    float                 scale,
    int                   chunk);

extern "C" struct ggml_tensor * ggml_cuda_paged_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    struct ggml_tensor  * mask,
    float                 scale,
    int                   chunk);

// true if op is our paged-attention custom op (so the CUDA backend claims it)
bool ggml_cuda_is_paged_attn(const struct ggml_tensor * op);

// run the paged-attention op inline on the CUDA backend (q/mask/out on device,
// K/V streamed from host into a bounded device scratch on ctx's stream)
void ggml_cuda_paged_attn_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);
