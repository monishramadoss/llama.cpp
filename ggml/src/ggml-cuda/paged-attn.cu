// GPU paged attention over a host/disk-resident KV cache, dispatched INLINE by
// the CUDA backend (no CPU<->GPU round trip per layer).
//
// The op is a GGML_OP_CUSTOM whose K/V are passed out-of-band (a pool indexed by
// the op's userdata) rather than as src operands, so the scheduler never copies
// the (host/disk) KV to VRAM. q and mask are device src operands. The CUDA
// backend identifies our op by its callback pointer, claims it in supports_op,
// and runs ggml_cuda_paged_attn_forward on its own stream: q/mask/out stay on
// the device; only K/V chunks are staged host->VRAM into a bounded scratch.
//
// If the op ever lands on the CPU backend instead, pa_host_fallback computes it
// on the host (correct, slow) so the graph is always valid.

#include "paged-attn.cuh"
#include "ggml-impl.h"

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

// per-device persistent scratch (paged attention runs once per attention layer
// per token; avoid malloc/free in the hot path).
struct pa_scratch {
    char *Kc=nullptr,*Vc=nullptr; float *maskc=nullptr,*scores=nullptr,*m=nullptr,*l=nullptr,*acc=nullptr;
    size_t cKc=0,cVc=0,cMsk=0,cScr=0,cM=0,cL=0,cAcc=0;
};
static pa_scratch g_scr[GGML_CUDA_MAX_DEVICES];

static void pa_ensure(void** p, size_t* cap, size_t need){ if(*cap<need){ cudaFree(*p); cudaMalloc(p,need); *cap=need; } }

// q_dev, mask_dev (may be null), out_dev are DEVICE pointers for ONE stream;
// k_host/v_host are HOST pointers. Enqueues all work on `stream`, no sync.
static void pa_core(int device, cudaStream_t stream,
    const float* q_dev, const void* k_host, const void* v_host, const float* mask_dev, float* out_dev,
    int d_k, int d_v, int n_q, int n_kv, int n_head, int n_head_kv, float scale, int chunk, bool kv_f16,
    size_t k_nb1, size_t k_nb2, size_t v_nb1, size_t v_nb2, size_t mask_nb1) {
    const int g = n_head / n_head_kv;
    if (chunk > n_kv) chunk = n_kv;
    if (chunk < 1) chunk = 1;
    const int ldn = chunk;
    const size_t es = kv_f16 ? 2 : 4;
    pa_scratch& c = g_scr[device];
    size_t sz_Kc=(size_t)d_k*ldn*n_head_kv*es, sz_Vc=(size_t)ldn*d_v*n_head_kv*es;
    size_t sz_msk=mask_dev?(size_t)ldn*n_q*4:0, sz_scr=(size_t)ldn*n_q*n_head*4, sz_ml=(size_t)n_q*n_head*4, sz_acc=(size_t)d_v*n_q*n_head*4;
    pa_ensure((void**)&c.Kc,&c.cKc,sz_Kc); pa_ensure((void**)&c.Vc,&c.cVc,sz_Vc);
    if(mask_dev) pa_ensure((void**)&c.maskc,&c.cMsk,sz_msk);
    pa_ensure((void**)&c.scores,&c.cScr,sz_scr); pa_ensure((void**)&c.m,&c.cM,sz_ml);
    pa_ensure((void**)&c.l,&c.cL,sz_ml); pa_ensure((void**)&c.acc,&c.cAcc,sz_acc);
    if(!c.Kc||!c.Vc||!c.scores||!c.m||!c.l||!c.acc) return;

    { static std::vector<float> hm; if((int)hm.size()<n_q*n_head) hm.assign((size_t)n_q*n_head,-INFINITY);
      cudaMemcpyAsync(c.m,hm.data(),sz_ml,cudaMemcpyHostToDevice,stream);
      cudaMemsetAsync(c.l,0,sz_ml,stream); cudaMemsetAsync(c.acc,0,sz_acc,stream); }

    for (int off=0; off<n_kv; off+=chunk) {
        int n=(off+chunk<=n_kv)?chunk:(n_kv-off);
        for(int hk=0;hk<n_head_kv;++hk)
            cudaMemcpy2DAsync(c.Kc+(size_t)hk*d_k*ldn*es,(size_t)d_k*es,
                              (const char*)k_host+hk*k_nb2+(size_t)off*k_nb1,k_nb1,(size_t)d_k*es,(size_t)n,cudaMemcpyHostToDevice,stream);
        for(int hk=0;hk<n_head_kv;++hk)
            cudaMemcpy2DAsync(c.Vc+(size_t)hk*ldn*d_v*es,(size_t)ldn*es,
                              (const char*)v_host+hk*v_nb2+(size_t)off*es,v_nb1,(size_t)n*es,(size_t)d_v,cudaMemcpyHostToDevice,stream);
        if(mask_dev)
            cudaMemcpy2DAsync(c.maskc,(size_t)ldn*4,(const char*)mask_dev+(size_t)off*4,mask_nb1,(size_t)n*4,(size_t)n_q,cudaMemcpyDeviceToDevice,stream);
        dim3 gs((n+63)/64,n_q,n_head), gm((n_q+63)/64,n_head,1), b(64,1,1);
        if (kv_f16){
            pa_scores<<<gs,b,0,stream>>>((const __half*)c.Kc,q_dev,mask_dev?c.maskc:nullptr,c.scores,d_k,n,ldn,n_q,g,scale);
            pa_merge <<<gm,b,0,stream>>>(c.scores,(const __half*)c.Vc,c.m,c.l,c.acc,n,ldn,d_v,n_q,g);
        } else {
            pa_scores<<<gs,b,0,stream>>>((const float*)c.Kc,q_dev,mask_dev?c.maskc:nullptr,c.scores,d_k,n,ldn,n_q,g,scale);
            pa_merge <<<gm,b,0,stream>>>(c.scores,(const float*)c.Vc,c.m,c.l,c.acc,n,ldn,d_v,n_q,g);
        }
    }
    pa_finalize<<<dim3((d_v+63)/64,n_q,n_head),dim3(64,1,1),0,stream>>>(c.acc,c.l,out_dev,d_v,n_q);
}

// -------- the pool carrying host K/V (not src operands) + scale/chunk --------
struct pa_pool_entry { const ggml_tensor* k=nullptr; const ggml_tensor* v=nullptr; float scale=1.f; int chunk=512; };
static const int PA_POOL = 4096;
static pa_pool_entry g_pool[PA_POOL];
static int g_pool_next = 0;

// CPU fallback (host compute) — used only if the op is not placed on CUDA.
static void pa_host_fallback(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    if (ith != 0) return; (void) nth;
    if (getenv("PA_DBG")) { static bool o=false; if(!o){o=true; fprintf(stderr,"[paged-attn] running CPU HOST FALLBACK\n");} }
    const int idx = (int)(intptr_t) userdata;
    const pa_pool_entry& e = g_pool[idx];
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* mask = dst->src[1];
    const ggml_tensor* k = e.k; const ggml_tensor* v = e.v;
    const int d_k=(int)q->ne[0], n_q=(int)q->ne[1], n_head=(int)q->ne[2], n_stream=(int)q->ne[3];
    const int n_kv=(int)k->ne[1], n_head_kv=(int)k->ne[2], d_v=(int)v->ne[1], g=n_head/n_head_kv;
    const bool kf16 = (k->type==GGML_TYPE_F16);
    auto kf=[&](const char* base,size_t nb1,size_t nb2,int d,int ik,int hk)->float{
        const char* p=base+hk*nb2+(size_t)ik*nb1+(size_t)d*(kf16?2:4);
        return kf16?__half2float(*(const __half*)p):*(const float*)p; };
    for (int s=0;s<n_stream;++s){
        const float* qb=(const float*)((const char*)q->data+(size_t)s*q->nb[3]);
        const char* kb=(const char*)k->data+(size_t)s*k->nb[3];
        const char* vb=(const char*)v->data+(size_t)s*v->nb[3];
        const char* mb=mask?(const char*)mask->data+(size_t)(mask->ne[3]>1?s*mask->nb[3]:0):nullptr;
        float* ob=(float*)((char*)dst->data+(size_t)s*dst->nb[3]);
        std::vector<float> sc(n_kv);
        for(int h=0;h<n_head;++h){int hk=h/g; for(int iq=0;iq<n_q;++iq){
            float mx=-INFINITY;
            for(int ik=0;ik<n_kv;++ik){double dot=0; for(int d=0;d<d_k;++d) dot+=(double)qb[d+iq*d_k+h*d_k*n_q]*kf(kb,k->nb[1],k->nb[2],d,ik,hk);
                float sv=e.scale*(float)dot; if(mb) sv+=*(const float*)(mb+(size_t)ik*mask->nb[0]+(size_t)iq*mask->nb[1]); sc[ik]=sv; if(sv>mx)mx=sv;}
            float l=0; for(int ik=0;ik<n_kv;++ik){sc[ik]=expf(sc[ik]-mx); l+=sc[ik];} if(!(l>0))l=1e-30f;
            for(int d=0;d<d_v;++d){double a=0; for(int ik=0;ik<n_kv;++ik) a+=(double)sc[ik]*kf(vb,v->nb[0],v->nb[2],ik,d,hk);
                ob[d+iq*d_v+h*d_v*n_q]=(float)(a/l);}}}
    }
}

bool ggml_cuda_is_paged_attn(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) return false;
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) op->op_params;
    return p->fun == pa_host_fallback;
}

void ggml_cuda_paged_attn_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    if (getenv("PA_DBG")) { static bool o=false; if(!o){o=true; fprintf(stderr,"[paged-attn] running INLINE on CUDA (device %d)\n", ctx.device);} }
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) dst->op_params;
    const int idx = (int)(intptr_t) p->userdata;
    const pa_pool_entry& e = g_pool[idx];
    const ggml_tensor* q = dst->src[0];
    const ggml_tensor* mask = dst->src[1];
    const ggml_tensor* k = e.k; const ggml_tensor* v = e.v;
    const int d_k=(int)q->ne[0], n_q=(int)q->ne[1], n_head=(int)q->ne[2], n_stream=(int)q->ne[3];
    const int n_kv=(int)k->ne[1], n_head_kv=(int)k->ne[2], d_v=(int)v->ne[1];
    const bool kf16 = (k->type==GGML_TYPE_F16);
    cudaStream_t stream = ctx.stream();
    for (int s=0;s<n_stream;++s){
        const float* qd=(const float*)((const char*)q->data+(size_t)s*q->nb[3]);
        const void*  kh=(const char*)k->data+(size_t)s*k->nb[3];
        const void*  vh=(const char*)v->data+(size_t)s*v->nb[3];
        const float* md=mask?(const float*)((const char*)mask->data+(size_t)(mask->ne[3]>1?s*mask->nb[3]:0)):nullptr;
        float* od=(float*)((char*)dst->data+(size_t)s*dst->nb[3]);
        pa_core(ctx.device, stream, qd, kh, vh, md, od,
                d_k,d_v,n_q,n_kv,n_head,n_head_kv,e.scale,e.chunk,kf16,
                k->nb[1],k->nb[2],v->nb[1],v->nb[2], mask?mask->nb[1]:0);
    }
}

extern "C" struct ggml_tensor * ggml_cuda_paged_attn(
    struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k,
    struct ggml_tensor * v, struct ggml_tensor * mask, float scale, int chunk) {
    struct ggml_tensor * qc = ggml_cont(ctx, q); // contiguous [d_k, n_q, n_head, n_stream]
    const int idx = g_pool_next; g_pool_next = (g_pool_next + 1) % PA_POOL;
    g_pool[idx].k = k; g_pool[idx].v = v; g_pool[idx].scale = scale; g_pool[idx].chunk = chunk>0?chunk:512;
    struct ggml_tensor * args[2] = { qc, mask };
    const int n_args = mask ? 2 : 1;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[1], q->ne[1], q->ne[2], q->ne[3],
                          args, n_args, pa_host_fallback, /*n_tasks*/ 1, (void*)(intptr_t) idx);
}
