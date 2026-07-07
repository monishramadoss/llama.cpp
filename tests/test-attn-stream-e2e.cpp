// End-to-end equivalence test for the streaming / chunked online-softmax
// attention path wired into the live graph builder (llm_graph_context::
// build_attn_mha).
//
// Builds a tiny random LLAMA model in memory and decodes the same tokens twice:
// once with the default dense attention and once with the streaming attention
// path enabled (and a deliberately tiny chunk size so the KV is folded in many
// chunks). The resulting logits must match closely, proving that the in-graph
// chunked path is behavior-preserving on a real model.

#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type");
    }
}

static bool silent_progress(float, void *) { return true; }

// minimal LLAMA gguf, enough to instantiate a small dense model
static gguf_context_ptr make_llama_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());

    const uint32_t n_ctx   = 128;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 128;
    const uint32_t n_head  = 4;
    const uint32_t n_ff    = 256;
    const uint32_t n_layer = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,             n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,       false);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,        n_embd_head);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");

    return ret;
}

static llama_context_ptr make_ctx(llama_model * model, bool streaming, uint32_t chunk_tokens) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 0;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.n_ubatch        = 64;
    // compare against the dense (non-flash) softmax path so the streaming path
    // is exercised and the comparison is apples-to-apples in F32.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // streaming attention is enabled alongside the experimental disk-offload
    // config; the page size in tokens is reused as the attention chunk size.
    cp.kv_offload_disk = streaming;
    cp.kv_page_tokens  = chunk_tokens;

    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("failed to create context");
    }
    return ctx;
}

static std::vector<float> decode_logits(llama_model * model, llama_context * ctx,
                                        const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        batch.token[pos]     = tokens[pos];
        batch.pos[pos]       = (llama_pos) pos;
        batch.n_seq_id[pos]  = 1;
        batch.seq_id[pos][0] = 0;
        batch.logits[pos]    = 1;
    }
    batch.n_tokens = (int32_t) tokens.size();

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("decode failed");
    }

    std::vector<float> out;
    out.reserve(tokens.size() * n_vocab);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        for (uint32_t j = 0; j < n_vocab; ++j) {
            out.push_back(l[j]);
        }
    }
    llama_batch_free(batch);
    return out;
}

// normalized mean squared error
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return 1e9;
    }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) a[i] * (double) a[i];
    }
    return den > 0.0 ? num / den : num;
}

int main() {
    llama_backend_init();

    int rc = 0;
    try {
        const size_t seed = 42;

        gguf_context_ptr gguf = make_llama_gguf();
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        size_t tmp = seed;
        llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &tmp, mp));
        if (!model) {
            throw std::runtime_error("failed to create model");
        }

        // a batch long enough to span several chunks at chunk size 4
        std::vector<llama_token> tokens;
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dist(0, (int) n_vocab - 1);
        for (int i = 0; i < 24; ++i) {
            tokens.push_back(dist(rng));
        }

        std::vector<float> ref;
        {
            auto ctx = make_ctx(model.get(), /*streaming*/ false, 0);
            ref = decode_logits(model.get(), ctx.get(), tokens);
        }

        for (uint32_t chunk : {8u, 16u, 64u}) {
            auto ctx = make_ctx(model.get(), /*streaming*/ true, chunk);
            const auto got = decode_logits(model.get(), ctx.get(), tokens);
            const double e = nmse(ref, got);
            printf("chunk=%u nmse=%.3e\n", chunk, e);
            if (!(e < 1e-6)) {
                fprintf(stderr, "FAIL: streaming logits diverge from dense (chunk=%u, nmse=%.3e)\n", chunk, e);
                rc = 1;
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "exception: %s\n", e.what());
        rc = 1;
    }

    llama_backend_free();

    if (rc == 0) {
        printf("test-attn-stream-e2e: all tests passed\n");
    }
    return rc;
}
