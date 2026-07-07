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
    // a large KV split into many chunks produces many nodes; size the graph
    // generously so the chunked builder always fits.
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1 << 18, false);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> r((size_t) ggml_nelements(out));
    memcpy(r.data(), out->data, r.size() * sizeof(float));
    return r;
}

static double nmse_of(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d * d; den += (double) a[i] * a[i];
    }
    return den > 0 ? num / den : num;
}

static bool close_enough(const std::vector<float> & a, const std::vector<float> & b, float eps) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            fprintf(stderr, "  mismatch at %zu: %g vs %g (nmse=%.3e)\n", i, a[i], b[i], nmse_of(a, b));
            return false;
        }
    }
    return true;
}

// n_head_kv < n_head exercises grouped-query attention (GQA), where the KV
// heads are broadcast across query heads (as in Qwen3, Llama-2/3, Mistral, ...).
// qk_mag scales the magnitude of q/k entries: large values drive the raw scores
// scale*(k.q) well past the range where exp() overflows, which is exactly the
// regime real models operate in and where a non-stabilised softmax produces NaN.
static int run_case(int64_t d_k, int64_t d_v, int64_t n_q, int64_t n_kv,
                    int64_t n_head, bool causal, unsigned seed,
                    int64_t n_head_kv = 0, float qk_mag = 1.0f, bool kv_f16 = false,
                    bool prefill = false, bool k_noncontig = false) {
    if (n_head_kv == 0) {
        n_head_kv = n_head;
    }
    struct ggml_init_params ip = {
        /* .mem_size   = */ (size_t) 512 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(ip);

    const float scale = 1.0f / std::sqrt((float) d_k);

    std::mt19937 rng(seed);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k,  n_q,  n_head,    1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, d_v,  n_head_kv, 1);

    // K layout. The live KV cache presents K with logical shape
    // [d_k, n_kv, n_head_kv] but stores it physically as [d_k, n_head_kv, n_kv]
    // (so it is *non-contiguous*: nb[1] > nb[2]). Reproduce that here when
    // requested by building the physical tensor and permuting.
    ggml_tensor * k;
    if (k_noncontig) {
        ggml_tensor * kphys = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k, n_head_kv, n_kv, 1);
        fill_random(kphys, rng, -qk_mag, qk_mag);
        k = ggml_permute(ctx, kphys, 0, 2, 1, 3); // logical [d_k, n_kv, n_head_kv, 1], non-contiguous
    } else {
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k, n_kv, n_head_kv, 1);
        fill_random(k, rng, -qk_mag, qk_mag);
    }

    fill_random(q, rng, -qk_mag, qk_mag);
    fill_random(v, rng);

    // the live KV cache stores K and V as F16; exercise that here so the test
    // matches the precision the real attention paths actually run at.
    if (kv_f16) {
        k = ggml_cast(ctx, k, GGML_TYPE_F16);
        v = ggml_cast(ctx, v, GGML_TYPE_F16);
    }

    ggml_tensor * mask = nullptr;
    if (causal) {
        // [n_kv, n_q] additive mask; query i attends to keys j <= its position.
        // prefill=false: queries aligned to the END of the KV range (most keys
        // visible). prefill=true: queries at the START (positions 0..n_q), as
        // when a short prompt is decoded into a long KV cache - then the vast
        // majority of the n_kv keys are masked out (the real decode regime).
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_q, 1, 1);
        float * md = (float *) mask->data;
        for (int64_t i = 0; i < n_q; ++i) {
            const int64_t pos = prefill ? i : (n_kv - n_q + i);
            for (int64_t j = 0; j < n_kv; ++j) {
                md[i * n_kv + j] = (j <= pos) ? 0.0f : -INFINITY;
            }
        }
    }

    std::vector<float> ref = compute(build_dense(ctx, q, k, v, mask, scale), ctx);

    int rc = 0;
    for (int64_t chunk : {(int64_t) 1, (int64_t) 2, (int64_t) 3, (int64_t) 5,
                          n_kv / 8, n_kv / 3, n_kv, n_kv + 7}) {
        if (chunk <= 0) {
            continue;
        }
        // keep the graph (and the recursive graph walk) bounded: skip chunk
        // sizes that would split a large n_kv into an unreasonable number of
        // chunks. The remaining sizes still cover partial and fully-masked
        // chunks for the decode regime.
        if (n_kv / chunk > 64) {
            continue;
        }
        ggml_tensor * out = llama_attn_stream_build_graph(ctx, q, k, v, mask, scale, chunk);
        const auto got = compute(out, ctx);
        if (chunk == n_kv) {
            fprintf(stderr, "  [seed %u] single-chunk nmse vs dense = %.3e\n", seed, nmse_of(ref, got));
        }
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

    // grouped-query attention: fewer KV heads than query heads (Qwen3-style)
    rc |= run_case(32, 32, 4, 40, 8, false, 6, /*n_head_kv*/ 4);
    rc |= run_case(16, 16, 5, 17, 8, true,  7, /*n_head_kv*/ 2);

    // large logit magnitudes: scale*(k.q) far exceeds exp()'s overflow range, so
    // a softmax that does not subtract the running row max produces inf/inf=NaN.
    // This is the regime every real model operates in.
    rc |= run_case(32, 32, 4, 40, 4, false, 8,  /*n_head_kv*/ 0,  /*qk_mag*/ 30.0f);
    rc |= run_case(32, 32, 6, 20, 4, true,  9,  /*n_head_kv*/ 0,  /*qk_mag*/ 30.0f);
    // and both together (closest to a real GQA decode with realistic activations)
    rc |= run_case(64, 64, 4, 48, 8, true,  10, /*n_head_kv*/ 4,  /*qk_mag*/ 20.0f);

    // F16 K/V cache (as the live attention paths actually run): Qwen3-like shape
    rc |= run_case(128, 128, 5, 40, 16, true, 11, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ true);
    rc |= run_case(128, 128, 1, 64, 16, true, 12, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ true);

    // the real decode regime: a short prompt decoded into a LARGE KV cache, so
    // the vast majority of keys are masked out (only ~n_q of n_kv are visible).
    // This is exactly how Qwen3 runs and the case the earlier tests missed.
    rc |= run_case(128, 128, 6, 2048, 16, true, 13, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ true, /*prefill*/ true);
    rc |= run_case(128, 128, 1, 2048, 16, true, 14, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ true, /*prefill*/ true);

    // non-contiguous K (logical [d_k, n_kv, n_head_kv], physically permuted),
    // exactly as the live KV cache presents it. The kernel must respect K's
    // real strides when slicing it into chunks.
    rc |= run_case(128, 128, 6, 64,  16, true, 15, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ false, /*prefill*/ false, /*k_noncontig*/ true);
    rc |= run_case(128, 128, 5, 256, 16, true, 16, /*n_head_kv*/ 8, /*qk_mag*/ 1.0f, /*kv_f16*/ false, /*prefill*/ true,  /*k_noncontig*/ true);

    if (rc == 0) {
        printf("test-attn-stream-graph: all tests passed\n");
    }
    return rc;
}
