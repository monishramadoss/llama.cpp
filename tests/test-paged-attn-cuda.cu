// Standalone, self-verifying prototype of GPU paged attention for a disk/host
// backed KV cache. Reads K/V from HOST memory and stages them into a BOUNDED
// device scratch one chunk at a time, computing online-softmax attention on the
// GPU. This is the compute core intended to be wired into build_attn_mha via a
// ggml custom op so attention runs on the GPU over a KV cache far larger than
// VRAM (the model weights stay in VRAM; KV streams disk -> RAM -> VRAM scratch).
//
// Status: verified standalone (parity vs CPU reference; VRAM bounded by `chunk`,
// independent of n_kv). Graph integration (F16 K/V, real cache strides, custom
// op, CMake wiring) is the follow-up.
//
// build & run:
//   nvcc -O2 -arch=sm_61 -diag-suppress 225 tests/test-paged-attn-cuda.cu -o /tmp/tpa -lcublas && /tmp/tpa
//
// Layout (matches the ggml streaming path):
//   q[d_k,n_q,n_head]  k[d_k,n_kv,n_head_kv]  v[n_kv,d_v,n_head_kv]  mask[n_kv,n_q]  out[d_v,n_q,n_head]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <vector>
#include <random>

size_t g_paged_attn_peak_scratch = 0;

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); return; } } while(0)

__global__ void k_scores(const float* Kc, const float* q, const float* maskc, float* scores,
                         int d_k, int n, int ldn, int n_q, int n_head, int g, float scale) {
    int ik = blockIdx.x*blockDim.x + threadIdx.x, iq = blockIdx.y, h = blockIdx.z;
    if (ik >= n) return;
    int hk = h / g;
    const float* kp = Kc + (size_t)hk*d_k*ldn + (size_t)ik*d_k;
    const float* qp = q  + (size_t)h *d_k*n_q + (size_t)iq*d_k;
    float dot = 0.f;
    for (int d=0; d<d_k; ++d) dot += kp[d]*qp[d];
    float s = scale*dot;
    if (maskc) s += maskc[ik + (size_t)iq*ldn];
    scores[ik + (size_t)iq*ldn + (size_t)h*ldn*n_q] = s;
}

__global__ void k_merge(const float* scores, const float* Vc, float* m, float* l, float* acc,
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
    float m_old = m[mi];
    float m_new = fmaxf(m_old, cmax);
    float corr  = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
    float l_new = l[mi]*corr;
    for (int d=0; d<d_v; ++d) accp[d] *= corr;
    for (int ik=0; ik<n; ++ik) {
        float s = sp[ik];
        float p = (s == -INFINITY) ? 0.f : expf(s - m_new);
        l_new += p;
        const float* vp = Vc + (size_t)hk*ldn*d_v + (size_t)ik;
        for (int d=0; d<d_v; ++d) accp[d] += p * vp[(size_t)d*ldn];
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

void paged_attn_cuda(const float* q, const float* k, const float* v, const float* mask, float* out,
                     int d_k, int d_v, int n_q, int n_kv, int n_head, int n_head_kv, float scale, int chunk) {
    const int g = n_head / n_head_kv;
    if (chunk > n_kv) chunk = n_kv;
    const int ldn = chunk;
    float *d_q=0,*d_Kc=0,*d_Vc=0,*d_maskc=0,*d_scores=0,*d_m=0,*d_l=0,*d_acc=0,*d_out=0;
    size_t sz_q=(size_t)d_k*n_q*n_head*4, sz_Kc=(size_t)d_k*ldn*n_head_kv*4, sz_Vc=(size_t)ldn*d_v*n_head_kv*4;
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
        for (int hk=0; hk<n_head_kv; ++hk)
            CUDA_OK(cudaMemcpy(d_Kc+(size_t)hk*d_k*ldn, k+(size_t)hk*d_k*n_kv+(size_t)off*d_k, (size_t)n*d_k*4, cudaMemcpyHostToDevice));
        for (int hk=0; hk<n_head_kv; ++hk)
            CUDA_OK(cudaMemcpy2D(d_Vc+(size_t)hk*ldn*d_v, (size_t)ldn*4, v+(size_t)hk*n_kv*d_v+(size_t)off, (size_t)n_kv*4, (size_t)n*4, (size_t)d_v, cudaMemcpyHostToDevice));
        if (mask) CUDA_OK(cudaMemcpy2D(d_maskc,(size_t)ldn*4, mask+(size_t)off,(size_t)n_kv*4,(size_t)n*4,(size_t)n_q,cudaMemcpyHostToDevice));
        k_scores<<<dim3((n+63)/64,n_q,n_head),dim3(64,1,1)>>>(d_Kc,d_q,mask?d_maskc:0,d_scores,d_k,n,ldn,n_q,n_head,g,scale);
        k_merge<<<dim3((n_q+63)/64,n_head,1),dim3(64,1,1)>>>(d_scores,d_Vc,d_m,d_l,d_acc,n,ldn,d_v,n_q,n_head,g);
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
static int run_case(int d_k,int d_v,int n_q,int n_kv,int n_head,int n_head_kv,int chunk,bool causal,unsigned seed){
    std::mt19937 rng(seed); std::uniform_real_distribution<float> dist(-1,1);
    std::vector<float> q(d_k*n_q*n_head),k(d_k*n_kv*n_head_kv),v(n_kv*d_v*n_head_kv);
    for(auto&x:q)x=dist(rng);for(auto&x:k)x=dist(rng);for(auto&x:v)x=dist(rng);
    std::vector<float> mask; const float* mp=0;
    if(causal){mask.assign((size_t)n_kv*n_q,0); for(int iq=0;iq<n_q;++iq){int pos=n_kv-n_q+iq;for(int ik=0;ik<n_kv;++ik)mask[ik+iq*n_kv]=(ik<=pos)?0.f:-INFINITY;} mp=mask.data();}
    float scale=1.f/sqrtf((float)d_k);
    std::vector<float> ref(d_v*n_q*n_head),got(d_v*n_q*n_head,0);
    attn_reference(q.data(),k.data(),v.data(),mp,ref.data(),d_k,d_v,n_q,n_kv,n_head,n_head_kv,scale);
    paged_attn_cuda(q.data(),k.data(),v.data(),mp,got.data(),d_k,d_v,n_q,n_kv,n_head,n_head_kv,scale,chunk);
    double e=nmse(ref,got);
    printf("  d_k=%d d_v=%d n_q=%d n_kv=%d heads=%d/%d chunk=%d causal=%d nmse=%.3e scratch=%zuKB %s\n",
        d_k,d_v,n_q,n_kv,n_head,n_head_kv,chunk,(int)causal,e,g_paged_attn_peak_scratch/1024,e<1e-4?"ok":"FAIL");
    return e<1e-4?0:1;
}
int main(){
    int rc=0;
    rc|=run_case(64,64,4,40,8,8,8,false,1);
    rc|=run_case(64,64,4,40,8,8,8,true,2);
    rc|=run_case(128,128,1,200,16,8,32,false,3);
    rc|=run_case(128,128,6,300,16,8,64,true,4);
    run_case(128,128,1,1000,16,8,64,false,5); size_t a=g_paged_attn_peak_scratch;
    run_case(128,128,1,8000,16,8,64,false,6); size_t b=g_paged_attn_peak_scratch;
    printf("scratch bounded: n_kv=1000->%zuKB n_kv=8000->%zuKB %s\n",a/1024,b/1024,(b<=a+1024)?"ok":"FAIL");
    if(b>a+1024)rc=1;
    if(rc==0)printf("test-paged-attn-cuda: all tests passed\n");
    return rc;
}
