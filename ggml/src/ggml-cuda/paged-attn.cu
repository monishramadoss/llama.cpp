// GPU paged attention over a host/disk-resident KV cache.
//
// K/V live in host (mmap'd) memory and are streamed into a BOUNDED device
// scratch one chunk at a time; attention is computed on the GPU with the
// online-softmax recurrence. VRAM use depends on the chunk size, not on n_kv,
// so a KV cache far larger than VRAM can be served while the model weights stay
// resident on the device.
//
// This is wrapped in a ggml custom op (runs on the CPU backend, but the callback
// performs its work on the GPU). The K/V host pointers reach the callback as
// ordinary CPU tensors; q (and mask) are copied to host by the scheduler. The
// graph therefore contains a single node per attention layer instead of the
// O(n_kv/chunk) nodes the in-graph streaming path would generate.

#include "paged-attn.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <vector>

template<typename T> static __device__ float pa_to_f(T x);
template<> __device__ float pa_to_f<float>(float x){ return x; }
template<> __device__ float pa_to_f<__half>(__half x){ return __half2float(x); }

template<typename T>
static __global__ void pa_scores(const T* Kc, const float* q, const float* maskc, float* scores,
                                 int d_k, int n, int ldn, int n_q, int g, float scale) {
    int ik = blockIdx.x*blockDim.x + threadIdx.x, iq = blockIdx.y, h = blockIdx.z;
    if (ik >= n) return;
    int hk = h / g;
    const T*     kp = Kc + (size_t)hk*d_k*ldn + (size_t)ik*d_k;
    const float* qp = q  + (size_t)h *d_k*n_q + (size_t)iq*d_k;
    float dot = 0.f;
    for (int d=0; d<d_k; ++d) dot += pa_to_f(kp[d])*qp[d];
    float s = scale*dot;
    if (maskc) s += maskc[ik + (size_t)iq*ldn];
    scores[ik + (size_t)iq*ldn + (size_t)h*ldn*n_q] = s;
}

template<typename T>
static __global__ void pa_merge(const float* scores, const T* Vc, float* m, float* l, float* acc,
                                int n, int ldn, int d_v, int n_q, int g) {
    int iq = blockIdx.x*blockDim.x + threadIdx.x, h = blockIdx.y;
    if (iq >= n_q) return;
    int hk = h / g;
    const float* sp = scores + (size_t)iq*ldn + (size_t)h*ldn*n_q;
    float* accp = acc + (size_t)iq*d_v + (size_t)h*d_v*n_q;
    int mi = iq + h*n_q;
    float cmax = -INFINITY;
    for (int ik=0; ik<n; ++ik) cmax = fmaxf(cmax, sp[ik]);
    if (cmax == -INFINITY) return;
    float m_old = m[mi], m_new = fmaxf(m_old, cmax);
    float corr  = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
    float l_new = l[mi]*corr;
    for (int d=0; d<d_v; ++d) accp[d] *= corr;
    for (int ik=0; ik<n; ++ik) {
        float s = sp[ik];
        float p = (s == -INFINITY) ? 0.f : expf(s - m_new);
        l_new += p;
        const T* vp = Vc + (size_t)hk*ldn*d_v + (size_t)ik;
        for (int d=0; d<d_v; ++d) accp[d] += p * pa_to_f(vp[(size_t)d*ldn]);
    }
    m[mi] = m_new; l[mi] = l_new;
}

static __global__ void pa_finalize(const float* acc, const float* l, float* out, int d_v, int n_q) {
    int d = blockIdx.x*blockDim.x + threadIdx.x, iq = blockIdx.y, h = blockIdx.z;
    if (d >= d_v) return;
    float ll = l[iq + h*n_q]; if (!(ll > 1e-30f)) ll = 1e-30f;
    size_t idx = (size_t)d + (size_t)iq*d_v + (size_t)h*d_v*n_q;
    out[idx] = acc[idx] / ll;
}

// one stream: q [d_k,n_q,n_head] (contiguous, scale already folded in),
// K/V host with byte strides, out [d_v,n_q,n_head] host.
static void pa_one_stream(
    const float* q, const void* k, const void* v, const float* mask, float* out,
    int d_k, int d_v, int n_q, int n_kv, int n_head, int n_head_kv, int chunk, bool kv_f16,
    size_t k_nb1, size_t k_nb2, size_t v_nb1, size_t v_nb2, size_t mask_nb1) {
    const int g = n_head / n_head_kv;
    if (chunk > n_kv) chunk = n_kv;
    if (chunk < 1) chunk = 1;
    const int ldn = chunk;
    const size_t es = kv_f16 ? 2 : 4;

    // The callback runs on the CPU backend thread with no CUDA device current;
    // pin paged attention (and its persistent streams/events, which are bound to
    // the device current at creation) to device 0, then restore.
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    cudaSetDevice(0);
    struct dev_restore { int d; ~dev_restore(){ cudaSetDevice(d); } } _dr{prev_dev};

    // Persistent, reused-across-calls scratch + double-buffered staging. Paged
    // attention runs once per attention layer per token, so per-call cudaMalloc/
    // Free + full-device sync would dominate. K/V/mask are double-buffered across
    // two streams so the H2D copy of chunk i+1 overlaps the (serial online-softmax)
    // compute of chunk i.
    struct pa_ctx {
        cudaStream_t copyS=nullptr, compS=nullptr;
        cudaEvent_t evcopy[2]={nullptr,nullptr}, evcomp[2]={nullptr,nullptr};
        char *Kc[2]={nullptr,nullptr}, *Vc[2]={nullptr,nullptr}; float *maskc[2]={nullptr,nullptr};
        float *q=nullptr,*scores=nullptr,*m=nullptr,*l=nullptr,*acc=nullptr,*out=nullptr;
        size_t cKc=0,cVc=0,cMsk=0,cQ=0,cScr=0,cM=0,cL=0,cAcc=0,cOut=0; bool init=false;
    };
    static pa_ctx c;
    auto ens =[&](void**p,size_t*cap,size_t need){ if(*cap<need){ cudaFree(*p); cudaMalloc(p,need); *cap=need; } };
    auto ens2=[&](char**a,char**b,size_t*cap,size_t need){ if(*cap<need){ cudaFree(*a);cudaFree(*b); cudaMalloc((void**)a,need);cudaMalloc((void**)b,need); *cap=need; } };
    if (!c.init){ cudaStreamCreate(&c.copyS); cudaStreamCreate(&c.compS);
        for(int s=0;s<2;++s){ cudaEventCreateWithFlags(&c.evcopy[s],cudaEventDisableTiming); cudaEventCreateWithFlags(&c.evcomp[s],cudaEventDisableTiming);} c.init=true; }

    size_t sz_q=(size_t)d_k*n_q*n_head*4, sz_Kc=(size_t)d_k*ldn*n_head_kv*es, sz_Vc=(size_t)ldn*d_v*n_head_kv*es;
    size_t sz_msk=mask?(size_t)ldn*n_q*4:0, sz_scr=(size_t)ldn*n_q*n_head*4, sz_ml=(size_t)n_q*n_head*4, sz_acc=(size_t)d_v*n_q*n_head*4;
    ens2(&c.Kc[0],&c.Kc[1],&c.cKc,sz_Kc); ens2(&c.Vc[0],&c.Vc[1],&c.cVc,sz_Vc);
    if(mask) ens2((char**)&c.maskc[0],(char**)&c.maskc[1],&c.cMsk,sz_msk);
    ens((void**)&c.q,&c.cQ,sz_q); ens((void**)&c.scores,&c.cScr,sz_scr);
    ens((void**)&c.m,&c.cM,sz_ml); ens((void**)&c.l,&c.cL,sz_ml); ens((void**)&c.acc,&c.cAcc,sz_acc); ens((void**)&c.out,&c.cOut,sz_acc);
    if (!c.Kc[0]||!c.Vc[0]||!c.q||!c.scores||!c.acc||!c.out) return; // alloc failed; caller zeroed out

    cudaMemcpyAsync(c.q,q,sz_q,cudaMemcpyHostToDevice,c.compS);
    { std::vector<float> hm((size_t)n_q*n_head,-INFINITY);
      cudaMemcpyAsync(c.m,hm.data(),sz_ml,cudaMemcpyHostToDevice,c.compS);
      cudaMemsetAsync(c.l,0,sz_ml,c.compS); cudaMemsetAsync(c.acc,0,sz_acc,c.compS); }

    auto stage=[&](int off,int slot){
        int n=(off+chunk<=n_kv)?chunk:(n_kv-off);
        for(int hk=0;hk<n_head_kv;++hk)
            cudaMemcpy2DAsync(c.Kc[slot]+(size_t)hk*d_k*ldn*es,(size_t)d_k*es,
                              (const char*)k+hk*k_nb2+(size_t)off*k_nb1,k_nb1,(size_t)d_k*es,(size_t)n,cudaMemcpyHostToDevice,c.copyS);
        for(int hk=0;hk<n_head_kv;++hk)
            cudaMemcpy2DAsync(c.Vc[slot]+(size_t)hk*ldn*d_v*es,(size_t)ldn*es,
                              (const char*)v+hk*v_nb2+(size_t)off*es,v_nb1,(size_t)n*es,(size_t)d_v,cudaMemcpyHostToDevice,c.copyS);
        if(mask)
            cudaMemcpy2DAsync(c.maskc[slot],(size_t)ldn*4,(const char*)mask+(size_t)off*4,mask_nb1,(size_t)n*4,(size_t)n_q,cudaMemcpyHostToDevice,c.copyS);
    };

    const int N = (n_kv+chunk-1)/chunk;
    stage(0,0); cudaEventRecord(c.evcopy[0],c.copyS);
    for (int i=0;i<N;++i){
        int s=i&1, ns=(i+1)&1, off=i*chunk, n=(off+chunk<=n_kv)?chunk:(n_kv-off);
        if (i+1<N){ cudaStreamWaitEvent(c.copyS,c.evcomp[ns],0); stage((i+1)*chunk,ns); cudaEventRecord(c.evcopy[ns],c.copyS); }
        cudaStreamWaitEvent(c.compS,c.evcopy[s],0);
        dim3 gs((n+63)/64,n_q,n_head), gm((n_q+63)/64,n_head,1), b(64,1,1);
        if (kv_f16){
            pa_scores<<<gs,b,0,c.compS>>>((const __half*)c.Kc[s],c.q,mask?c.maskc[s]:nullptr,c.scores,d_k,n,ldn,n_q,g,1.0f);
            pa_merge <<<gm,b,0,c.compS>>>(c.scores,(const __half*)c.Vc[s],c.m,c.l,c.acc,n,ldn,d_v,n_q,g);
        } else {
            pa_scores<<<gs,b,0,c.compS>>>((const float*)c.Kc[s],c.q,mask?c.maskc[s]:nullptr,c.scores,d_k,n,ldn,n_q,g,1.0f);
            pa_merge <<<gm,b,0,c.compS>>>(c.scores,(const float*)c.Vc[s],c.m,c.l,c.acc,n,ldn,d_v,n_q,g);
        }
        cudaEventRecord(c.evcomp[s],c.compS);
    }
    pa_finalize<<<dim3((d_v+63)/64,n_q,n_head),dim3(64,1,1),0,c.compS>>>(c.acc,c.l,c.out,d_v,n_q);
    cudaMemcpyAsync(out,c.out,sz_acc,cudaMemcpyDeviceToHost,c.compS);
    cudaStreamSynchronize(c.compS);
}

// custom-op callback (runs on CPU backend, computes on GPU). userdata = chunk.
static void pa_custom_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) return; (void) nth;
    const int chunk = (int)(intptr_t) userdata;

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3]; // may be null

    const int d_k       = (int) q->ne[0];
    const int n_q       = (int) q->ne[1];
    const int n_head    = (int) q->ne[2];
    const int n_stream  = (int) q->ne[3];
    const int n_kv      = (int) k->ne[1];
    const int n_head_kv = (int) k->ne[2];
    const int d_v       = (int) v->ne[1];
    const bool kv_f16   = (k->type == GGML_TYPE_F16);

    const char * qb = (const char *) q->data;
    const char * kb = (const char *) k->data;
    const char * vb = (const char *) v->data;
    const char * mb = mask ? (const char *) mask->data : nullptr;
    char * ob = (char *) dst->data;

    for (int s = 0; s < n_stream; ++s) {
        pa_one_stream(
            (const float*)(qb + (size_t)s*q->nb[3]),
            (const void *)(kb + (size_t)s*k->nb[3]),
            (const void *)(vb + (size_t)s*v->nb[3]),
            mb ? (const float*)(mb + (size_t)(mask->ne[3]>1 ? s*mask->nb[3] : 0)) : nullptr,
            (float*)(ob + (size_t)s*dst->nb[3]),
            d_k, d_v, n_q, n_kv, n_head, n_head_kv, chunk, kv_f16,
            k->nb[1], k->nb[2], v->nb[1], v->nb[2], mask ? mask->nb[1] : 0);
    }
}

extern "C" struct ggml_tensor * ggml_cuda_paged_attn(
    struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k,
    struct ggml_tensor * v, struct ggml_tensor * mask, float scale, int chunk) {
    // fold scale into q so the kernel uses scale = 1. q is a permuted view, so
    // make it contiguous first (ggml_scale requires a padded/contiguous tensor),
    // which also gives the callback a clean [d_k, n_q, n_head, n_stream] layout.
    struct ggml_tensor * qs = ggml_scale(ctx, ggml_cont(ctx, q), scale);
    struct ggml_tensor * args[4] = { qs, k, v, mask };
    const int n_args = mask ? 4 : 3;
    if (chunk <= 0) chunk = 512;
    struct ggml_tensor * out = ggml_custom_4d(
        ctx, GGML_TYPE_F32, v->ne[1], q->ne[1], q->ne[2], q->ne[3],
        args, n_args, pa_custom_op, /*n_tasks*/ 1, (void*)(intptr_t) chunk);
    return out;
}
