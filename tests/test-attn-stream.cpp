// Unit tests for streaming / online-softmax attention (src/llama-attn-stream.*).
//
// The streaming path folds keys/values into a running softmax chunk by chunk.
// These tests check that:
//   - it matches an independent one-shot softmax oracle,
//   - the result is invariant to how the keys are split into chunks,
//   - it stays numerically stable for very large scores,
//   - additive masks (incl. causal masking and fully-masked rows) work,
//   - degenerate geometries (single key, empty stream) behave sensibly.

#include "llama-attn-stream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

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

static std::vector<float> rand_vec(std::mt19937 & rng, size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

// Stream the K/V in chunks of the given size and return the output.
static std::vector<float> run_streamed(const llama_attn_stream_params & p,
                                       const std::vector<float> & q,
                                       const std::vector<float> & k,
                                       const std::vector<float> & v,
                                       int64_t n_keys,
                                       int64_t chunk,
                                       const std::vector<float> * mask = nullptr) {
    llama_attn_stream attn(p);
    attn.begin(q.data());

    for (int64_t off = 0; off < n_keys; off += chunk) {
        const int64_t n = std::min(chunk, n_keys - off);

        std::vector<float> mchunk;
        const float * mptr = nullptr;
        if (mask) {
            // slice the [n_query, n_keys] mask down to this key range
            mchunk.resize((size_t) p.n_query * (size_t) n);
            for (int64_t i = 0; i < p.n_query; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    mchunk[i * n + j] = (*mask)[i * n_keys + (off + j)];
                }
            }
            mptr = mchunk.data();
        }

        attn.update(k.data() + off * p.head_dim,
                    v.data() + off * p.v_dim,
                    n,
                    mptr);
    }

    std::vector<float> out((size_t) p.n_query * (size_t) p.v_dim);
    attn.finish(out.data());
    return out;
}

// Streaming over several chunk sizes must match the one-shot reference.
static int test_parity_and_chunk_invariance() {
    std::mt19937 rng(1234);

    llama_attn_stream_params p;
    p.n_query  = 5;
    p.head_dim = 8;
    p.v_dim    = 6;
    p.scale    = 0.0f; // default 1/sqrt(head_dim)

    const int64_t n_keys = 37;

    const auto q = rand_vec(rng, (size_t) p.n_query * p.head_dim);
    const auto k = rand_vec(rng, (size_t) n_keys * p.head_dim);
    const auto v = rand_vec(rng, (size_t) n_keys * p.v_dim);

    std::vector<float> ref((size_t) p.n_query * p.v_dim);
    llama_attn_reference(p, q.data(), k.data(), v.data(), n_keys, ref.data());

    for (int64_t chunk : {1, 2, 4, 8, 16, 37, 64}) {
        const auto out = run_streamed(p, q, k, v, n_keys, chunk);
        CHECK(close_enough(out, ref, 1e-4f));
    }
    return 0;
}

// With huge logits a naive softmax overflows; the running-max formulation must
// stay finite and still match a (carefully computed) reference.
static int test_numerical_stability() {
    llama_attn_stream_params p;
    p.n_query  = 1;
    p.head_dim = 4;
    p.v_dim    = 3;
    p.scale    = 1.0f;

    const int64_t n_keys = 3;

    // q . k for key j is roughly 100 * j -> scores 0, 100, 200
    std::vector<float> q = { 100.0f, 0.0f, 0.0f, 0.0f };
    std::vector<float> k = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 0.0f,
    };
    std::vector<float> v = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const auto out = run_streamed(p, q, k, v, n_keys, 1);

    // result must be finite
    for (float x : out) {
        CHECK(std::isfinite(x));
    }
    // the last key dominates overwhelmingly -> output ~ v[2] = (0,0,1)
    CHECK(std::fabs(out[0] - 0.0f) < 1e-5f);
    CHECK(std::fabs(out[1] - 0.0f) < 1e-5f);
    CHECK(std::fabs(out[2] - 1.0f) < 1e-5f);

    std::vector<float> ref((size_t) p.n_query * p.v_dim);
    llama_attn_reference(p, q.data(), k.data(), v.data(), n_keys, ref.data());
    CHECK(close_enough(out, ref, 1e-5f));
    return 0;
}

// Causal masking (query i may only attend to keys j <= i) fed across chunks.
static int test_causal_mask() {
    std::mt19937 rng(99);

    llama_attn_stream_params p;
    p.n_query  = 6;
    p.head_dim = 4;
    p.v_dim    = 4;

    const int64_t n_keys = 6; // square, query i attends to keys 0..i

    const auto q = rand_vec(rng, (size_t) p.n_query * p.head_dim);
    const auto k = rand_vec(rng, (size_t) n_keys * p.head_dim);
    const auto v = rand_vec(rng, (size_t) n_keys * p.v_dim);

    std::vector<float> mask((size_t) p.n_query * n_keys);
    for (int64_t i = 0; i < p.n_query; ++i) {
        for (int64_t j = 0; j < n_keys; ++j) {
            mask[i * n_keys + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }

    std::vector<float> ref((size_t) p.n_query * p.v_dim);
    llama_attn_reference(p, q.data(), k.data(), v.data(), n_keys, ref.data(), mask.data());

    for (int64_t chunk : {1, 2, 3, 6}) {
        const auto out = run_streamed(p, q, k, v, n_keys, chunk, &mask);
        CHECK(close_enough(out, ref, 1e-4f));
    }

    // query 0 attends to a single key -> its output equals v[0]
    for (int64_t t = 0; t < p.v_dim; ++t) {
        CHECK(std::fabs(ref[t] - v[t]) < 1e-4f);
    }
    return 0;
}

// A query whose keys are all masked out must produce zeros (no NaNs).
static int test_fully_masked_row() {
    llama_attn_stream_params p;
    p.n_query  = 2;
    p.head_dim = 3;
    p.v_dim    = 3;

    const int64_t n_keys = 4;

    std::mt19937 rng(7);
    const auto q = rand_vec(rng, (size_t) p.n_query * p.head_dim);
    const auto k = rand_vec(rng, (size_t) n_keys * p.head_dim);
    const auto v = rand_vec(rng, (size_t) n_keys * p.v_dim);

    std::vector<float> mask((size_t) p.n_query * n_keys, 0.0f);
    // mask out every key for query 1
    for (int64_t j = 0; j < n_keys; ++j) {
        mask[1 * n_keys + j] = -INFINITY;
    }

    const auto out = run_streamed(p, q, k, v, n_keys, 2, &mask);
    for (int64_t t = 0; t < p.v_dim; ++t) {
        CHECK(out[1 * p.v_dim + t] == 0.0f); // query 1: all-zero, not NaN
    }
    // query 0 still produced something finite
    for (int64_t t = 0; t < p.v_dim; ++t) {
        CHECK(std::isfinite(out[t]));
    }
    return 0;
}

// An empty stream (no keys at all) yields zeros, and update() with n_keys==0 is
// a no-op.
static int test_empty_stream() {
    llama_attn_stream_params p;
    p.n_query  = 3;
    p.head_dim = 4;
    p.v_dim    = 5;

    std::mt19937 rng(0);
    const auto q = rand_vec(rng, (size_t) p.n_query * p.head_dim);

    llama_attn_stream attn(p);
    attn.begin(q.data());
    attn.update(nullptr, nullptr, 0); // no-op, must not dereference null

    std::vector<float> out((size_t) p.n_query * p.v_dim, 1.0f);
    attn.finish(out.data());
    for (float x : out) {
        CHECK(x == 0.0f);
    }
    return 0;
}

// Reusing one instance via begin() for a second problem must not leak state
// from the first.
static int test_reuse_instance() {
    std::mt19937 rng(555);

    llama_attn_stream_params p;
    p.n_query  = 4;
    p.head_dim = 6;
    p.v_dim    = 6;

    const int64_t n_keys = 20;

    llama_attn_stream attn(p);

    for (int iter = 0; iter < 3; ++iter) {
        const auto q = rand_vec(rng, (size_t) p.n_query * p.head_dim);
        const auto k = rand_vec(rng, (size_t) n_keys * p.head_dim);
        const auto v = rand_vec(rng, (size_t) n_keys * p.v_dim);

        std::vector<float> ref((size_t) p.n_query * p.v_dim);
        llama_attn_reference(p, q.data(), k.data(), v.data(), n_keys, ref.data());

        attn.begin(q.data());
        attn.update(k.data(), v.data(), n_keys);
        std::vector<float> out((size_t) p.n_query * p.v_dim);
        attn.finish(out.data());

        CHECK(close_enough(out, ref, 1e-4f));
    }
    return 0;
}

// Invalid geometry must be rejected.
static int test_invalid_config() {
    bool threw = false;
    try {
        llama_attn_stream_params p;
        p.n_query  = 1;
        p.head_dim = 0; // invalid
        p.v_dim    = 4;
        llama_attn_stream attn(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        llama_attn_stream_params p;
        p.n_query  = 1;
        p.head_dim = 4;
        p.v_dim    = 0; // invalid
        llama_attn_stream attn(p);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_parity_and_chunk_invariance();
    rc |= test_numerical_stability();
    rc |= test_causal_mask();
    rc |= test_fully_masked_row();
    rc |= test_empty_stream();
    rc |= test_reuse_instance();
    rc |= test_invalid_config();

    if (rc == 0) {
        printf("test-attn-stream: all tests passed\n");
    }
    return rc;
}
