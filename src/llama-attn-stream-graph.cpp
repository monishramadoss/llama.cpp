// Streaming / online-softmax attention, as a ggml graph.
// See llama-attn-stream-graph.h for the contract and tensor layout.

#include "llama-attn-stream-graph.h"

#include "ggml.h"

#include <cassert>
#include <cfloat>

// Small floor applied to the softmax denominator so that a query row whose keys
// are *all* masked out yields a finite (zero) result instead of 0/0 = NaN,
// matching the behaviour of the dense soft_max path on padded rows.
static const float LLAMA_ATTN_STREAM_DENOM_EPS = 1e-30f;

// Lower bound for the running row maximum. Its only job is to neutralise the
// sentinel that max-pooling produces for an all-masked row (-FLT_MAX): without
// a bound, combining that sentinel with a real maximum in the element-wise max
// below would overflow / lose all precision. It is chosen far below any logit a
// real model produces (attention scores are |scale * K.Q|, well within this) so
// the true per-row maximum is always preserved, while staying small enough in
// magnitude that the element-wise max remains numerically accurate.
static const float LLAMA_ATTN_STREAM_MIN_MAX = -1e6f;

// element-wise max of two tensors of identical shape, built from primitive ops
// (ggml has no binary max): max(a,b) = (a + b + |a - b|) / 2.
static ggml_tensor * llama_attn_stream_emax(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * sum = ggml_add(ctx, ggml_add(ctx, a, b), ggml_abs(ctx, ggml_sub(ctx, a, b)));
    return ggml_scale(ctx, sum, 0.5f);
}

ggml_tensor * llama_attn_stream_build_graph(
        ggml_context * ctx,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * mask,
        float          scale,
        int64_t        chunk_tokens) {
    GGML_ASSERT(q && k && v);
    GGML_ASSERT(chunk_tokens > 0);

    const int64_t n_kv = k->ne[1];
    GGML_ASSERT(v->ne[0] == n_kv && "v must be transposed so that dim0 == n_kv");

    // Online (FlashAttention-style) softmax over chunks. For each chunk c we
    // maintain a running per-query row maximum m and accumulate the numerator
    // and denominator of the softmax *relative to that maximum*:
    //
    //   s_c   = scale * (K_c . Q) + mask_c       (masked entries -> -inf)
    //   m     = max(m, rowmax(s_c))              (running per-row max)
    //   corr  = exp(m_old - m_new)               (rescale prior accumulators)
    //   e_c   = exp(s_c - m_new)                 (in (0, 1], never overflows)
    //   den   = den*corr + sum_j e_cj
    //   num   = num*corr + sum_j e_cj * v_j
    //
    // The final output is num / den. Subtracting the running maximum is what
    // makes this numerically stable: without it exp(scale * K.Q) overflows to
    // +inf on real-magnitude logits, giving inf/inf = NaN. Because only one
    // chunk's scores are materialised at a time, the score-matrix working set is
    // bounded by the chunk size regardless of n_kv.
    //
    // It is important to subtract the *true* per-row maximum (not max(rowmax, 0)
    // or any other shallow bound): when a row's true maximum is negative,
    // shifting by 0 instead leaves every exp() far below 1, and the subsequent
    // (F16) value matmul then loses relative precision - empirically ~10% error
    // on a real model, even though it is exact in infinite precision. The only
    // clamp applied is a very negative floor (LLAMA_ATTN_STREAM_MIN_MAX) whose
    // sole purpose is to neutralise the all-masked-row pooling sentinel.
    ggml_tensor * num = nullptr; // [d_v, n_query, n_head, n_stream]
    ggml_tensor * den = nullptr; // [1,   n_query, n_head, n_stream]
    ggml_tensor * m   = nullptr; // [1,   n_query, n_head, n_stream] running row max

    for (int64_t off = 0; off < n_kv; off += chunk_tokens) {
        const int64_t n = (off + chunk_tokens <= n_kv) ? chunk_tokens : (n_kv - off);

        // slice this chunk of keys/values (and the matching mask columns).
        // ggml_cont keeps the per-chunk operands contiguous for mul_mat and
        // bounds the working set to a single chunk.
        ggml_tensor * k_c = ggml_cont(ctx,
                ggml_view_4d(ctx, k, k->ne[0], n, k->ne[2], k->ne[3],
                        k->nb[1], k->nb[2], k->nb[3], off * k->nb[1]));

        ggml_tensor * v_c = ggml_cont(ctx,
                ggml_view_4d(ctx, v, n, v->ne[1], v->ne[2], v->ne[3],
                        v->nb[1], v->nb[2], v->nb[3], off * v->nb[0]));

        // scores for this chunk: [n, n_query, n_head, n_stream]
        ggml_tensor * kq = ggml_mul_mat(ctx, k_c, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        // s = scale * kq (+ additive mask). The mask broadcasts over the head
        // and stream dimensions (its dims 2/3 are 1).
        ggml_tensor * s = ggml_scale(ctx, kq, scale);
        if (mask) {
            ggml_tensor * mask_c = ggml_view_4d(ctx, mask, n, mask->ne[1], mask->ne[2], mask->ne[3],
                    mask->nb[1], mask->nb[2], mask->nb[3], off * mask->nb[0]);
            s = ggml_add(ctx, s, mask_c);
        }

        // per-row max of this chunk over the key dimension (ne0). Max-pooling
        // with a window of the full chunk width reduces [n, n_query, h, s] to
        // [1, n_query, h, s]. An all-masked row pools to -FLT_MAX; floor it (see
        // note above) so the element-wise max stays finite and accurate.
        ggml_tensor * m_c = ggml_pool_2d(ctx, s, GGML_OP_POOL_MAX, (int) n, 1, (int) n, 1, 0, 0);
        m_c = ggml_clamp(ctx, m_c, LLAMA_ATTN_STREAM_MIN_MAX, FLT_MAX);

        ggml_tensor * m_new = m ? llama_attn_stream_emax(ctx, m, m_c) : m_c;

        // e = exp(s - m_new), broadcasting the per-row max over the key dim.
        ggml_tensor * e = ggml_exp(ctx, ggml_add(ctx, s, ggml_scale(ctx, m_new, -1.0f)));

        ggml_tensor * l = ggml_sum_rows(ctx, e);         // [1,   n_query, h, s]
        ggml_tensor * o = ggml_mul_mat(ctx, v_c, e);     // [d_v, n_query, h, s]

        if (num) {
            // rescale the running accumulators from the old max to the new max
            ggml_tensor * corr = ggml_exp(ctx, ggml_add(ctx, m, ggml_scale(ctx, m_new, -1.0f)));
            num = ggml_add(ctx, ggml_mul(ctx, num, corr), o);
            den = ggml_add(ctx, ggml_mul(ctx, den, corr), l);
        } else {
            num = o;
            den = l;
        }
        m = m_new;
    }

    GGML_ASSERT(num && den);

    // floor the denominator so fully-masked rows give 0 instead of 0/0 = NaN
    den = ggml_clamp(ctx, den, LLAMA_ATTN_STREAM_DENOM_EPS, FLT_MAX);

    return ggml_div(ctx, num, den); // [d_v, n_query, n_head, n_stream]
}
