// Unit test for the in-graph chunked online-softmax attention builder
// (src/llama-attn-stream-graph.*).
//
// Builds the chunked attention subgraph on the CPU backend and checks that, for
// several chunk sizes and with/without a causal mask, it matches a dense
// reference attention computed with the same ggml ops. This validates that
// folding the KV in pages and combining with the online-softmax recurrence is
// numerically equivalent to materialising the full attention matrix.

#include "ggml.h"
#include "ggml-cpu.h"

#include "llama-attn-stream-graph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

// Dense reference attention built from the same primitive ggml ops the live
// non-flash path uses: out = mul_mat(v, soft_max(scale*kq + mask)).
//   q : [d_k, n_q, n_head, 1]
//   k : [d_k, n_kv, n_head, 1]
//   v : [n_kv, d_v, n_head, 1]
static ggml_tensor * build_dense(ggml_context * ctx,
                                 ggml_tensor * q,
                                 ggml_tensor * k,
                                 ggml_tensor * v,
                                 ggml_tensor * mask,
                                 float scale) {
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * p = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    return ggml_mul_mat(ctx, v, p);
}

static void fill_random(ggml_tensor * t, std::mt19937 & rng, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = dist(rng);
    }
}

static std::vector<float> compute(ggml_tensor * out, ggml_context * ctx) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> r((size_t) ggml_nelements(out));
    memcpy(r.data(), out->data, r.size() * sizeof(float));
    return r;
}

static bool close_enough(const std::vector<float> & a, const std::vector<float> & b, float eps) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            fprintf(stderr, "  mismatch at %zu: %g vs %g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static int run_case(int64_t d_k, int64_t d_v, int64_t n_q, int64_t n_kv,
                    int64_t n_head, bool causal, unsigned seed) {
    struct ggml_init_params ip = {
        /* .mem_size   = */ (size_t) 512 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(ip);

    const float scale = 1.0f / std::sqrt((float) d_k);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k,  n_q,  n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k,  n_kv, n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, d_v,  n_head, 1);

    std::mt19937 rng(seed);
    fill_random(q, rng);
    fill_random(k, rng);
    fill_random(v, rng);

    ggml_tensor * mask = nullptr;
    if (causal) {
        // [n_kv, n_q] additive mask; query i (global pos n_kv-n_q+i) attends to
        // keys j <= its position. Use a simple lower-triangular layout with the
        // queries aligned to the end of the KV range.
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_q, 1, 1);
        float * md = (float *) mask->data;
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t pos = n_kv - n_q + i;
            for (int64_t j = 0; j < n_kv; ++j) {
                md[i * n_kv + j] = (j <= pos) ? 0.0f : -INFINITY;
            }
        }
    }

    std::vector<float> ref = compute(build_dense(ctx, q, k, v, mask, scale), ctx);

    int rc = 0;
    for (int64_t chunk : {(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 5, n_kv, n_kv + 7}) {
        ggml_tensor * out = llama_attn_stream_build_graph(ctx, q, k, v, mask, scale, chunk);
        const auto got = compute(out, ctx);
        if (!close_enough(got, ref, 2e-3f)) {
            fprintf(stderr, "FAIL chunk=%lld (d_k=%lld d_v=%lld n_q=%lld n_kv=%lld n_head=%lld causal=%d)\n",
                    (long long) chunk, (long long) d_k, (long long) d_v,
                    (long long) n_q, (long long) n_kv, (long long) n_head, causal);
            rc = 1;
        }
    }

    ggml_free(ctx);
    return rc;
}

int main() {
    int rc = 0;

    // square self-attention, no mask
    rc |= run_case(/*d_k*/ 8, /*d_v*/ 8, /*n_q*/ 6, /*n_kv*/ 6,  /*n_head*/ 1, /*causal*/ false, 1);
    // longer KV than queries (prefill-style), no mask, multi-head
    rc |= run_case(16, 12, 4, 23, 3, false, 2);
    // causal masking across chunk boundaries
    rc |= run_case(8, 8, 6, 6, 1, true, 3);
    rc |= run_case(16, 16, 5, 17, 2, true, 4);
    // single query (decode step) over a long context
    rc |= run_case(32, 32, 1, 40, 4, false, 5);

    if (rc == 0) {
        printf("test-attn-stream-graph: all tests passed\n");
    }
    return rc;
}
