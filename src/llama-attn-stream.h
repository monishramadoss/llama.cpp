#pragma once

// Streaming / online-softmax (FlashAttention-style) attention.
//
// This module implements numerically-stable scaled dot-product attention that
// consumes the keys/values *chunk by chunk* instead of materialising the full
// attention matrix. It is the compute-side counterpart to the disk-backed
// tiered KV pager (src/llama-kv-pager.{h,cpp}): the pager makes one page of K/V
// resident at a time, and this module folds that page into a running softmax so
// the working set stays bounded regardless of context length.
//
// The classic softmax over a row of scores s_j is
//
//     out = (sum_j exp(s_j) * v_j) / (sum_j exp(s_j))
//
// which requires every score at once. The online ("streaming") formulation
// keeps three running quantities per query and updates them as each chunk of
// keys arrives, rescaling by the change in the running maximum for stability:
//
//     m  = running max of the scores seen so far          (init -inf)
//     l  = running sum of exp(s_j - m)                     (init 0)
//     acc= running sum of exp(s_j - m) * v_j               (init 0)
//
// On a new chunk with local max m', the previous accumulators are rescaled by
// exp(m - max(m, m')) before the chunk's contributions are added. The final
// output is acc / l. This yields exactly the same result as a one-shot softmax
// (up to floating-point rounding) while only ever touching one chunk of K/V at
// a time.
//
// Like the pager, the module is intentionally decoupled from the ggml attention
// graph so it can be developed and unit-tested in isolation on CPU.

#include <cstdint>
#include <vector>

// Geometry of a streaming-attention problem.
//
// Queries are laid out row-major as [n_query, head_dim]. Keys are fed in
// chunks as [n_keys, head_dim] and values as [n_keys, v_dim]. The output is
// [n_query, v_dim].
struct llama_attn_stream_params {
    int64_t n_query  = 0;   // number of query rows
    int64_t head_dim = 0;   // dimension of each query/key vector (must be > 0)
    int64_t v_dim    = 0;   // dimension of each value vector (must be > 0)

    // softmax scale applied to the raw dot products. When <= 0 the standard
    // 1/sqrt(head_dim) is used.
    float   scale    = 0.0f;
};

// Numerically-stable online-softmax attention accumulator.
//
// Usage:
//     llama_attn_stream attn(params);
//     attn.begin(q);
//     for (each chunk of keys/values) {
//         attn.update(k_chunk, v_chunk, n_keys, mask /* optional */);
//     }
//     attn.finish(out);
//
// The same instance can be reused for another problem of identical geometry by
// calling begin() again. Chunks may have any number of keys (>= 0); splitting
// the keys into different chunk sizes does not change the result.
class llama_attn_stream {
public:
    explicit llama_attn_stream(const llama_attn_stream_params & params);

    const llama_attn_stream_params & params() const { return params_; }

    // reset all running accumulators and bind the query matrix for the sequence
    // of chunks that follows. q points to [n_query, head_dim] row-major data and
    // must remain valid until finish() is called.
    void begin(const float * q);

    // fold one chunk of keys/values into the running softmax.
    //
    //   k_chunk : [n_keys, head_dim] row-major
    //   v_chunk : [n_keys, v_dim]    row-major
    //   n_keys  : number of keys in this chunk (>= 0; 0 is a no-op)
    //   mask    : optional [n_query, n_keys] row-major additive bias added to
    //             the scores before softmax (e.g. 0 to keep, -INFINITY to drop
    //             a key for a given query). nullptr => no masking.
    //
    // Keys are global only in the sense that each call appends more keys to the
    // softmax; the caller is responsible for feeding them in a consistent order
    // and for supplying a mask consistent with that order.
    void update(const float * k_chunk,
                const float * v_chunk,
                int64_t       n_keys,
                const float * mask = nullptr);

    // write the final attention output to out ([n_query, v_dim] row-major).
    // Query rows that never received an unmasked key produce all-zeros.
    void finish(float * out) const;

private:
    llama_attn_stream_params params_;

    const float * q_ = nullptr; // [n_query, head_dim], bound by begin()

    // per-query running state
    std::vector<float> m_;   // [n_query]            running max
    std::vector<float> l_;   // [n_query]            running denominator
    std::vector<float> acc_; // [n_query * v_dim]    running weighted value sum
};

// Convenience one-shot reference implementation: computes the full attention in
// a single call. Used as a correctness oracle for the streaming path and handy
// when the whole K/V already fits in memory.
//
//   q   : [n_query, head_dim]
//   k   : [n_keys,  head_dim]
//   v   : [n_keys,  v_dim]
//   out : [n_query, v_dim]
//   mask: optional [n_query, n_keys] additive bias, or nullptr.
void llama_attn_reference(const llama_attn_stream_params & params,
                          const float * q,
                          const float * k,
                          const float * v,
                          int64_t       n_keys,
                          float       * out,
                          const float * mask = nullptr);
