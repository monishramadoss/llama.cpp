// Streaming / online-softmax (FlashAttention-style) attention.
// See llama-attn-stream.h for the algorithm and API contract.

#include "llama-attn-stream.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

llama_attn_stream::llama_attn_stream(const llama_attn_stream_params & params) : params_(params) {
    if (params_.n_query < 0) {
        throw std::invalid_argument("llama_attn_stream: n_query must be >= 0");
    }
    if (params_.head_dim <= 0) {
        throw std::invalid_argument("llama_attn_stream: head_dim must be > 0");
    }
    if (params_.v_dim <= 0) {
        throw std::invalid_argument("llama_attn_stream: v_dim must be > 0");
    }

    if (params_.scale <= 0.0f) {
        params_.scale = 1.0f / std::sqrt((float) params_.head_dim);
    }

    m_.resize((size_t) params_.n_query);
    l_.resize((size_t) params_.n_query);
    acc_.resize((size_t) params_.n_query * (size_t) params_.v_dim);
}

void llama_attn_stream::begin(const float * q) {
    q_ = q;

    const size_t nq = (size_t) params_.n_query;
    for (size_t i = 0; i < nq; ++i) {
        m_[i] = -INFINITY;
        l_[i] = 0.0f;
    }
    std::fill(acc_.begin(), acc_.end(), 0.0f);
}

void llama_attn_stream::update(const float * k_chunk,
                               const float * v_chunk,
                               int64_t       n_keys,
                               const float * mask) {
    if (n_keys < 0) {
        throw std::invalid_argument("llama_attn_stream::update: n_keys must be >= 0");
    }
    if (n_keys == 0) {
        return; // nothing to fold in
    }
    if (q_ == nullptr) {
        throw std::logic_error("llama_attn_stream::update: begin() must be called first");
    }
    if (k_chunk == nullptr || v_chunk == nullptr) {
        throw std::invalid_argument("llama_attn_stream::update: k_chunk/v_chunk must not be null");
    }

    const int64_t d     = params_.head_dim;
    const int64_t dv    = params_.v_dim;
    const float   scale = params_.scale;

    for (int64_t i = 0; i < params_.n_query; ++i) {
        const float * q = q_ + i * d;
        float * acc     = acc_.data() + (size_t) i * (size_t) dv;

        float m = m_[i];
        float l = l_[i];

        for (int64_t j = 0; j < n_keys; ++j) {
            const float * k = k_chunk + j * d;

            float s = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                s += q[t] * k[t];
            }
            s *= scale;
            if (mask) {
                s += mask[i * n_keys + j];
            }

            // a masked-out (-inf) score contributes nothing and must not move
            // the running max, otherwise exp(m - new_m) would be exp(-inf)=0
            // and wipe out the accumulators.
            if (s == -INFINITY) {
                continue;
            }

            if (s > m) {
                // rescale existing accumulators to the new maximum
                const float corr = (m == -INFINITY) ? 0.0f : std::exp(m - s);
                l *= corr;
                for (int64_t t = 0; t < dv; ++t) {
                    acc[t] *= corr;
                }
                m = s;
            }

            const float p = std::exp(s - m);
            l += p;

            const float * v = v_chunk + j * dv;
            for (int64_t t = 0; t < dv; ++t) {
                acc[t] += p * v[t];
            }
        }

        m_[i] = m;
        l_[i] = l;
    }
}

void llama_attn_stream::finish(float * out) const {
    const int64_t dv = params_.v_dim;

    for (int64_t i = 0; i < params_.n_query; ++i) {
        const float * acc = acc_.data() + (size_t) i * (size_t) dv;
        float * o         = out + i * dv;

        const float l = l_[i];
        if (l > 0.0f) {
            const float inv = 1.0f / l;
            for (int64_t t = 0; t < dv; ++t) {
                o[t] = acc[t] * inv;
            }
        } else {
            // query saw no unmasked keys: define the output as zero
            for (int64_t t = 0; t < dv; ++t) {
                o[t] = 0.0f;
            }
        }
    }
}

void llama_attn_reference(const llama_attn_stream_params & params,
                          const float * q,
                          const float * k,
                          const float * v,
                          int64_t       n_keys,
                          float       * out,
                          const float * mask) {
    // Independent, non-streaming oracle: materialise the full score row per
    // query and apply a plain max-shifted softmax. Intentionally does NOT share
    // code with llama_attn_stream so tests can use it to validate the streaming
    // path.
    int64_t d  = params.head_dim;
    int64_t dv = params.v_dim;

    float scale = params.scale;
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt((float) d);
    }

    std::vector<float> scores((size_t) n_keys);

    for (int64_t i = 0; i < params.n_query; ++i) {
        const float * qi = q + i * d;
        float * o        = out + i * dv;

        float m = -INFINITY;
        for (int64_t j = 0; j < n_keys; ++j) {
            const float * kj = k + j * d;
            float s = 0.0f;
            for (int64_t t = 0; t < d; ++t) {
                s += qi[t] * kj[t];
            }
            s *= scale;
            if (mask) {
                s += mask[i * n_keys + j];
            }
            scores[(size_t) j] = s;
            if (s > m) {
                m = s;
            }
        }

        for (int64_t t = 0; t < dv; ++t) {
            o[t] = 0.0f;
        }

        if (m == -INFINITY) {
            continue; // no unmasked keys
        }

        float l = 0.0f;
        for (int64_t j = 0; j < n_keys; ++j) {
            const float s = scores[(size_t) j];
            if (s == -INFINITY) {
                continue;
            }
            const float p = std::exp(s - m);
            l += p;
            const float * vj = v + j * dv;
            for (int64_t t = 0; t < dv; ++t) {
                o[t] += p * vj[t];
            }
        }

        if (l > 0.0f) {
            const float inv = 1.0f / l;
            for (int64_t t = 0; t < dv; ++t) {
                o[t] *= inv;
            }
        }
    }
}
