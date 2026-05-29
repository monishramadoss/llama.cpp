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

    // Online softmax over chunks. For each chunk c we accumulate the
    // *unnormalised* numerator and denominator of the softmax:
    //
    //   e_c = exp(scale * (K_c . Q) + mask_c)   (masked entries -> exp(-inf) = 0)
    //   den += sum_j e_cj                        (partial denominator)
    //   num += sum_j e_cj * v_j                  (partial weighted values)
    //
    // The final output is num / den. Because only one chunk's scores are
    // materialised at a time, the score-matrix working set is bounded by the
    // chunk size regardless of n_kv. Masked entries contribute exactly zero, so
    // additive -inf masks (causal / sparse) are handled without any NaNs.
    ggml_tensor * num = nullptr; // [d_v, n_query, n_head, n_stream]
    ggml_tensor * den = nullptr; // [1,   n_query, n_head, n_stream]

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

        ggml_tensor * e = ggml_exp(ctx, s);              // [n,   n_query, h, s], masked -> 0

        ggml_tensor * l = ggml_sum_rows(ctx, e);         // [1,   n_query, h, s]
        ggml_tensor * o = ggml_mul_mat(ctx, v_c, e);     // [d_v, n_query, h, s]

        num = num ? ggml_add(ctx, num, o) : o;
        den = den ? ggml_add(ctx, den, l) : l;
    }

    GGML_ASSERT(num && den);

    // floor the denominator so fully-masked rows give 0 instead of 0/0 = NaN
    den = ggml_clamp(ctx, den, LLAMA_ATTN_STREAM_DENOM_EPS, FLT_MAX);

    return ggml_div(ctx, num, den); // [d_v, n_query, n_head, n_stream]
}
