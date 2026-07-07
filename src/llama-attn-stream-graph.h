#pragma once

// Streaming / online-softmax attention, as a ggml graph.
//
// This is the graph-builder counterpart to the CPU primitive in
// llama-attn-stream.{h,cpp}. It constructs a ggml subgraph that computes scaled
// dot-product attention while processing the keys/values in fixed-size *chunks*
// (pages) along the KV dimension and accumulating the partial softmax numerator
// and denominator across chunks (the chunked / online-softmax pattern).
//
// The key property is that the largest score intermediate ever materialised is
// [n_kv_chunk, n_query, ...] instead of the full [n_kv, n_query, ...] matrix, so
// the attention working set stays bounded regardless of context length. This is
// what makes the disk-backed tiered KV cache (llama-kv-pager.{h,cpp}) usable in
// the live compute path: one page of K/V is folded in and can then be evicted
// before the next page is touched.
//
// Masked entries (additive -inf bias) contribute exactly zero, so causal and
// sparse masks are handled without NaNs; a fully-masked query row yields zeros.
// The running-max stabilised form of online softmax lives in the CPU primitive
// (llama-attn-stream.{h,cpp}); the in-graph accumulation here relies on the
// bounded attention logits of the supported (non soft-capped, non-ALiBi) archs,
// and anything outside that set falls back to the dense path.
//
// The builder uses only public ggml ops so it runs on any backend and can be
// unit-tested on the CPU backend (see tests/test-attn-stream-graph.cpp). It is
// mathematically equivalent (up to floating-point rounding) to the dense
// soft_max path in llm_graph_context::build_attn_mha.

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

// Build the chunked online-softmax attention subgraph and return the attention
// output tensor "kqv" with shape [d_v, n_query, n_head, n_stream] — i.e. exactly
// what the dense path produces from ggml_mul_mat(v, ggml_soft_max_ext(kq, ...)).
//
// Tensor layout (matching llm_graph_context::build_attn_mha after its permutes):
//   q    : [d_k,  n_query, n_head,    n_stream]
//   k    : [d_k,  n_kv,    n_head_kv, n_stream]   (n_head % n_head_kv == 0)
//   v    : [n_kv, d_v,     n_head_kv, n_stream]   (already transposed: dim0 = n_kv)
//   mask : [n_kv, n_query, 1, 1] additive F32 bias, or nullptr for no masking
//
//   scale        : softmax scale applied to the raw dot products
//   chunk_tokens : number of KV tokens to fold in per chunk (> 0). Splitting the
//                  KV into different chunk sizes does not change the result.
//
// When n_kv <= chunk_tokens this degenerates to a single chunk, i.e. the plain
// dense computation.
ggml_tensor * llama_attn_stream_build_graph(
        ggml_context * ctx,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * mask,
        float          scale,
        int64_t        chunk_tokens);
