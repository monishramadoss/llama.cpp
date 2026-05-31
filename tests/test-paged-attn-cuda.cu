// Standalone, self-verifying prototype of GPU paged attention for a disk/host
// backed KV cache. Reads K/V from HOST memory (any dtype F16/F32, arbitrary
// key/head strides) and stages them into a BOUNDED device scratch one chunk at
// a time, computing online-softmax attention on the GPU. Intended to be wired
// into build_attn_mha via a ggml custom op (model weights stay in VRAM; KV
// streams disk -> RAM -> VRAM scratch).
//
// build & run:
//   nvcc -O2 -arch=sm_61 -diag-suppress 225 tests/test-paged-attn-cuda.cu -o /tmp/tpa -lcublas && /tmp/tpa
//
// Layout (matches the ggml streaming path; strides are in BYTES):
//   q[d_k,n_q,n_head] F32 contiguous   k[d_k,n_kv,n_head_kv] (d_k contiguous)
//   v[n_kv,d_v,n_head_kv] (n_kv contiguous)   mask[n_kv,n_q] F32 (n_kv contiguous) or null
//   out[d_v,n_q,n_head] F32 contiguous

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <vector>
#include <random>

size_t g_paged_attn_peak_scratch = 0;

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); return; } } while(0)

template<typename T> __device__ float to_f(T x);
template<> __device__ float to_f<float>(float x){ return x; }
template<> __device__ float to_f<__half>(__half x){ return __half2float(x); }

template<typename T>
__global__ void k_scores(const T* Kc, const float* q, const float* maskc, float* scores,
                         int d_k, int n, int ldn, int n_q, int n_head, int g, float scale) {
    int ik = blockIdx.x*blockDim.x + threadIdx.x, iq = blockIdx.y, h = blockIdx.z;
    if (ik >= n) return;
    int hk = h / g;
    const T*     kp = Kc + (size_t)hk*d_k*ldn + (size_t)ik*d_k;
    const float* qp = q  + (size_t)h *d_k*n_q + (size_t)iq*d_k;
    float dot = 0.f;
    for (int d=0; d<d_k; ++d) dot += to_f(kp[d])*qp[d];
    float s = scale*dot;
    if (maskc) s += maskc[ik + (size_t)iq*ldn];
    scores[ik + (size_t)iq*ldn + (size_t)h*ldn*n_q] = s;
}

template<typename T>
__global__ void k_merge(const float* scores, const T* Vc, float* m, float* l, float* acc,
                        int n, int ldn, int d_v, int n_q, int n_head, int g) {
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
        for (int d=0; d<d_v; ++d) accp[d] += p * to_f(vp[(size_t)d*ldn]);
    }
    m[mi] = m_new; l[mi] = l_new;
}

__global__ void k_finalize(const float* acc, const float* l, float* out, int d_v, int n_q, int n_head) {
    int d = blockIdx.x*blockDim.x + threadIdx.x, iq = blockIdx.y, h = blockIdx.z;
    if (d >= d_v) return;
    float ll = l[iq + h*n_q]; if (!(ll > 1e-30f)) ll = 1e-30f;
    size_t idx = (size_t)d + (size_t)iq*d_v + (size_t)h*d_v*n_q;
    out[idx] = acc[idx] / ll;
}

// kv_f16: K/V are __half (else float). k_nb1/k_nb2, v_nb1/v_nb2 are BYTE strides
// of the key axis / head axis; d_k (K) and n_kv (V) are assumed contiguous.
void paged_attn_cuda(
    const float* q, const void* k, const void* v, const float* mask, float* out,
    int d_k, int d_v, int n_q, int n_kv, int n_head, int n_head_kv,
    float scale, int chunk, bool kv_f16,
    size_t k_nb1, size_t k_nb2, size_t v_nb1, size_t v_nb2, size_t mask_nb1) {
    const int g = n_head / n_head_kv;
    if (chunk > n_kv) chunk = n_kv;
    const int ldn = chunk;
    const size_t es = kv_f16 ? 2 : 4; // KV element size

    char *d_Kc=0,*d_Vc=0; float *d_q=0,*d_maskc=0,*d_scores=0,*d_m=0,*d_l=0,*d_acc=0,*d_out=0;
    size_t sz_q=(size_t)d_k*n_q*n_head*4, sz_Kc=(size_t)d_k*ldn*n_head_kv*es, sz_Vc=(size_t)ldn*d_v*n_head_kv*es;
    size_t sz_msk=mask?(size_t)ldn*n_q*4:0, sz_scr=(size_t)ldn*n_q*n_head*4, sz_ml=(size_t)n_q*n_head*4, sz_acc=(size_t)d_v*n_q*n_head*4;
    g_paged_attn_peak_scratch = sz_q+sz_Kc+sz_Vc+sz_msk+sz_scr+2*sz_ml+2*sz_acc;
    CUDA_OK(cudaMalloc(&d_q,sz_q)); CUDA_OK(cudaMalloc(&d_Kc,sz_Kc)); CUDA_OK(cudaMalloc(&d_Vc,sz_Vc));
    if(mask) CUDA_OK(cudaMalloc(&d_maskc,sz_msk));
    CUDA_OK(cudaMalloc(&d_scores,sz_scr)); CUDA_OK(cudaMalloc(&d_m,sz_ml)); CUDA_OK(cudaMalloc(&d_l,sz_ml));
    CUDA_OK(cudaMalloc(&d_acc,sz_acc)); CUDA_OK(cudaMalloc(&d_out,sz_acc));
    CUDA_OK(cudaMemcpy(d_q,q,sz_q,cudaMemcpyHostToDevice));
    { size_t mn=(size_t)n_q*n_head; float* hm=(float*)malloc(mn*4); for(size_t i=0;i<mn;++i) hm[i]=-INFINITY;
      CUDA_OK(cudaMemcpy(d_m,hm,sz_ml,cudaMemcpyHostToDevice)); free(hm);
      CUDA_OK(cudaMemset(d_l,0,sz_ml)); CUDA_OK(cudaMemset(d_acc,0,sz_acc)); }
    for (int off=0; off<n_kv; off+=chunk) {
        int n = (off+chunk<=n_kv)?chunk:(n_kv-off);
        for (int hk=0; hk<n_head_kv; ++hk) // K: gather n keys (each d_k contiguous) at byte stride k_nb1
            CUDA_OK(cudaMemcpy2D(d_Kc+(size_t)hk*d_k*ldn*es, (size_t)d_k*es,
                                 (const char*)k+hk*k_nb2+(size_t)off*k_nb1, k_nb1,
                                 (size_t)d_k*es, (size_t)n, cudaMemcpyHostToDevice));
        for (int hk=0; hk<n_head_kv; ++hk) // V: n keys contiguous, d_v cols at byte stride v_nb1
            CUDA_OK(cudaMemcpy2D(d_Vc+(size_t)hk*ldn*d_v*es, (size_t)ldn*es,
                                 (const char*)v+hk*v_nb2+(size_t)off*es, v_nb1,
                                 (size_t)n*es, (size_t)d_v, cudaMemcpyHostToDevice));
        if (mask)
            CUDA_OK(cudaMemcpy2D(d_maskc,(size_t)ldn*4, (const char*)mask+(size_t)off*4, mask_nb1,
                                 (size_t)n*4,(size_t)n_q,cudaMemcpyHostToDevice));
        dim3 gs((n+63)/64,n_q,n_head), gm((n_q+63)/64,n_head,1), b(64,1,1);
        if (kv_f16) {
            k_scores<<<gs,b>>>((const __half*)d_Kc,d_q,mask?d_maskc:0,d_scores,d_k,n,ldn,n_q,n_head,g,scale);
            k_merge<<<gm,b>>>(d_scores,(const __half*)d_Vc,d_m,d_l,d_acc,n,ldn,d_v,n_q,n_head,g);
        } else {
            k_scores<<<gs,b>>>((const float*)d_Kc,d_q,mask?d_maskc:0,d_scores,d_k,n,ldn,n_q,n_head,g,scale);
            k_merge<<<gm,b>>>(d_scores,(const float*)d_Vc,d_m,d_l,d_acc,n,ldn,d_v,n_q,n_head,g);
        }
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    }
    k_finalize<<<dim3((d_v+63)/64,n_q,n_head),dim3(64,1,1)>>>(d_acc,d_l,d_out,d_v,n_q,n_head);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaMemcpy(out,d_out,sz_acc,cudaMemcpyDeviceToHost));
    cudaFree(d_q);cudaFree(d_Kc);cudaFree(d_Vc);if(d_maskc)cudaFree(d_maskc);
    cudaFree(d_scores);cudaFree(d_m);cudaFree(d_l);cudaFree(d_acc);cudaFree(d_out);
}

// ---------------- test ----------------
static void attn_reference(const float* q,const float* k,const float* v,const float* mask,float* out,
                           int d_k,int d_v,int n_q,int n_kv,int n_head,int n_head_kv,float scale){
    const int g=n_head/n_head_kv; std::vector<float> s(n_kv);
    for(int h=0;h<n_head;++h){int hk=h/g; for(int iq=0;iq<n_q;++iq){
        float m=-INFINITY;
        for(int ik=0;ik<n_kv;++ik){double dot=0; for(int d=0;d<d_k;++d) dot+=(double)q[d+iq*d_k+h*d_k*n_q]*k[d+ik*d_k+hk*d_k*n_kv];
            float sv=scale*(float)dot; if(mask)sv+=mask[ik+iq*n_kv]; s[ik]=sv; if(sv>m)m=sv;}
        float l=0; for(int ik=0;ik<n_kv;++ik){s[ik]=expf(s[ik]-m); l+=s[ik];} if(!(l>0))l=1e-30f;
        for(int d=0;d<d_v;++d){double acc=0; for(int ik=0;ik<n_kv;++ik) acc+=(double)s[ik]*v[ik+d*n_kv+hk*n_kv*d_v];
            out[d+iq*d_v+h*d_v*n_q]=(float)(acc/l);}}}
}
static double nmse(const std::vector<float>&a,const std::vector<float>&b){double num=0,den=0;for(size_t i=0;i<a.size();++i){double d=(double)a[i]-b[i];num+=d*d;den+=(double)a[i]*a[i];}return den>0?num/den:num;}

// kv_f16: store K/V as half. strided: lay K out physically as [d_k, n_head_kv, n_kv]
// (Nemo's real layout) and pass the matching byte strides, instead of contiguous.
static int run_case(int d_k,int d_v,int n_q,int n_kv,int n_head,int n_head_kv,int chunk,bool causal,
                    bool kv_f16,bool strided,unsigned seed){
    std::mt19937 rng(seed); std::uniform_real_distribution<float> dist(-1,1);
    std::vector<float> q(d_k*n_q*n_head),kf(d_k*n_kv*n_head_kv),vf(n_kv*d_v*n_head_kv);
    for(auto&x:q)x=dist(rng);for(auto&x:kf)x=dist(rng);for(auto&x:vf)x=dist(rng);
    std::vector<float> mask; const float* mp=0;
    if(causal){mask.assign((size_t)n_kv*n_q,0); for(int iq=0;iq<n_q;++iq){int pos=n_kv-n_q+iq;for(int ik=0;ik<n_kv;++ik)mask[ik+iq*n_kv]=(ik<=pos)?0.f:-INFINITY;} mp=mask.data();}
    float scale=1.f/sqrtf((float)d_k);
    std::vector<float> ref(d_v*n_q*n_head),got(d_v*n_q*n_head,0);
    attn_reference(q.data(),kf.data(),vf.data(),mp,ref.data(),d_k,d_v,n_q,n_kv,n_head,n_head_kv,scale);

    // build the K/V host buffers in the requested dtype & layout, plus byte strides.
    // logical k[d, ik, hk]; logical v[ik, d, hk].
    size_t k_nb0,k_nb1,k_nb2, v_nb0,v_nb1,v_nb2;
    size_t es = kv_f16?2:4;
    std::vector<__half> kh, vh; std::vector<float> kc, vc; const void *kp,*vp;
    auto idxK=[&](int d,int ik,int hk){ // element index into the chosen physical layout
        return strided ? ((size_t)d + (size_t)hk*d_k + (size_t)ik*d_k*n_head_kv)   // [d_k, n_head_kv, n_kv]
                       : ((size_t)d + (size_t)ik*d_k + (size_t)hk*d_k*n_kv); };     // [d_k, n_kv, n_head_kv]
    auto idxV=[&](int ik,int d,int hk){
        return strided ? ((size_t)ik + (size_t)hk*n_kv + (size_t)d*n_kv*n_head_kv) // [n_kv, n_head_kv, d_v]
                       : ((size_t)ik + (size_t)d*n_kv + (size_t)hk*n_kv*d_v); };    // [n_kv, d_v, n_head_kv]
    size_t kN=(size_t)d_k*n_kv*n_head_kv, vN=(size_t)n_kv*d_v*n_head_kv;
    if(kv_f16){kh.resize(kN);vh.resize(vN);} else {kc.resize(kN);vc.resize(vN);}
    for(int hk=0;hk<n_head_kv;++hk)for(int ik=0;ik<n_kv;++ik)for(int d=0;d<d_k;++d){
        float val=kf[d+ik*d_k+hk*d_k*n_kv]; size_t id=idxK(d,ik,hk);
        if(kv_f16) kh[id]=__float2half(val); else kc[id]=val; }
    for(int hk=0;hk<n_head_kv;++hk)for(int d=0;d<d_v;++d)for(int ik=0;ik<n_kv;++ik){
        float val=vf[ik+d*n_kv+hk*n_kv*d_v]; size_t id=idxV(ik,d,hk);
        if(kv_f16) vh[id]=__float2half(val); else vc[id]=val; }
    kp = kv_f16?(const void*)kh.data():(const void*)kc.data();
    vp = kv_f16?(const void*)vh.data():(const void*)vc.data();
    // byte strides for the chosen layout
    k_nb0=es; k_nb1=strided?(size_t)d_k*n_head_kv*es:(size_t)d_k*es; k_nb2=strided?(size_t)d_k*es:(size_t)d_k*n_kv*es;
    v_nb0=es; v_nb1=strided?(size_t)n_kv*n_head_kv*es:(size_t)n_kv*es; v_nb2=strided?(size_t)n_kv*es:(size_t)n_kv*d_v*es;
    (void)k_nb0;(void)v_nb0;
    size_t mask_nb1=(size_t)n_kv*4;

    paged_attn_cuda(q.data(),kp,vp,mp,got.data(),d_k,d_v,n_q,n_kv,n_head,n_head_kv,scale,chunk,kv_f16,
                    k_nb1,k_nb2,v_nb1,v_nb2,mask_nb1);
    double e=nmse(ref,got);
    double tol = kv_f16?2e-3:1e-4;
    printf("  d_k=%d n_q=%d n_kv=%d heads=%d/%d chunk=%d causal=%d f16=%d strided=%d nmse=%.3e scratch=%zuKB %s\n",
        d_k,n_q,n_kv,n_head,n_head_kv,chunk,(int)causal,(int)kv_f16,(int)strided,e,g_paged_attn_peak_scratch/1024,e<tol?"ok":"FAIL");
    return e<tol?0:1;
}
int main(){
    int rc=0;
    // F32 contiguous (baseline)
    rc|=run_case(64,64,4,40,8,8,8,true,false,false,1);
    rc|=run_case(128,128,1,200,16,8,32,false,false,false,2);
    // F16 contiguous
    rc|=run_case(128,128,1,200,16,8,32,false,true,false,3);
    rc|=run_case(128,128,6,300,16,8,64,true,true,false,4);
    // F16 + Nemo-like strided layout (the real cache)
    rc|=run_case(128,128,1,512,16,8,64,false,true,true,5);
    rc|=run_case(128,128,6,512,16,8,64,true,true,true,6);
    rc|=run_case(128,128,1,2000,32,8,128,true,true,true,7);
    // bounded scratch
    run_case(128,128,1,1000,16,8,64,false,true,true,8); size_t a=g_paged_attn_peak_scratch;
    run_case(128,128,1,8000,16,8,64,false,true,true,9); size_t b=g_paged_attn_peak_scratch;
    printf("scratch bounded: n_kv=1000->%zuKB n_kv=8000->%zuKB %s\n",a/1024,b/1024,(b<=a+1024)?"ok":"FAIL");
    if(b>a+1024)rc=1;
    if(rc==0)printf("test-paged-attn-cuda: all tests passed\n");
    return rc;
}
