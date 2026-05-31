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

    // Persistent, reused-across-calls scratch + streams/events (paged attention is
    // invoked once per attention layer per token, so per-call cudaMalloc/Free and
    // cudaDeviceSynchronize would dominate). Double-buffered K/V/mask so the H2D
    // copy of chunk i+1 overlaps the (serial) online-softmax compute of chunk i.
    struct ctx_t {
        cudaStream_t copyS=0, compS=0;
        cudaEvent_t evcopy[2]={0,0}, evcomp[2]={0,0};
        char *Kc[2]={0,0}, *Vc[2]={0,0}; float *maskc[2]={0,0};
        float *q=0,*scores=0,*m=0,*l=0,*acc=0,*out=0;
        size_t capKc=0,capVc=0,capMsk=0,capQ=0,capScr=0,capM=0,capL=0,capAcc=0,capOut=0; bool init=false;
    };
    static ctx_t c;
    auto ensure=[&](void** p,size_t* cap,size_t need){ if(*cap<need){ cudaFree(*p); cudaMalloc(p,need); *cap=need; } };
    auto ensure2=[&](char** p0,char** p1,size_t* cap,size_t need){ if(*cap<need){ cudaFree(*p0);cudaFree(*p1); cudaMalloc((void**)p0,need);cudaMalloc((void**)p1,need); *cap=need; } };
    if (!c.init){ cudaStreamCreate(&c.copyS); cudaStreamCreate(&c.compS);
        for(int s=0;s<2;++s){cudaEventCreateWithFlags(&c.evcopy[s],cudaEventDisableTiming); cudaEventCreateWithFlags(&c.evcomp[s],cudaEventDisableTiming);} c.init=true; }

    size_t sz_q=(size_t)d_k*n_q*n_head*4, sz_Kc=(size_t)d_k*ldn*n_head_kv*es, sz_Vc=(size_t)ldn*d_v*n_head_kv*es;
    size_t sz_msk=mask?(size_t)ldn*n_q*4:0, sz_scr=(size_t)ldn*n_q*n_head*4, sz_ml=(size_t)n_q*n_head*4, sz_acc=(size_t)d_v*n_q*n_head*4;
    g_paged_attn_peak_scratch = sz_q+2*sz_Kc+2*sz_Vc+2*sz_msk+sz_scr+2*sz_ml+sz_acc;
    ensure2(&c.Kc[0],&c.Kc[1],&c.capKc,sz_Kc); ensure2(&c.Vc[0],&c.Vc[1],&c.capVc,sz_Vc);
    if(mask) ensure2((char**)&c.maskc[0],(char**)&c.maskc[1],&c.capMsk,sz_msk);
    ensure((void**)&c.q,&c.capQ,sz_q); ensure((void**)&c.scores,&c.capScr,sz_scr);
    ensure((void**)&c.m,&c.capM,sz_ml); ensure((void**)&c.l,&c.capL,sz_ml); ensure((void**)&c.acc,&c.capAcc,sz_acc); ensure((void**)&c.out,&c.capOut,sz_acc);

    // q copy + state init on compute stream (orders before the first compute)
    CUDA_OK(cudaMemcpyAsync(c.q,q,sz_q,cudaMemcpyHostToDevice,c.compS));
    { std::vector<float> hm((size_t)n_q*n_head,-INFINITY);
      CUDA_OK(cudaMemcpyAsync(c.m,hm.data(),sz_ml,cudaMemcpyHostToDevice,c.compS));
      CUDA_OK(cudaMemsetAsync(c.l,0,sz_ml,c.compS)); CUDA_OK(cudaMemsetAsync(c.acc,0,sz_acc,c.compS)); }

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
            k_scores<<<gs,b,0,c.compS>>>((const __half*)c.Kc[s],c.q,mask?c.maskc[s]:0,c.scores,d_k,n,ldn,n_q,n_head,g,scale);
            k_merge<<<gm,b,0,c.compS>>>(c.scores,(const __half*)c.Vc[s],c.m,c.l,c.acc,n,ldn,d_v,n_q,n_head,g);
        } else {
            k_scores<<<gs,b,0,c.compS>>>((const float*)c.Kc[s],c.q,mask?c.maskc[s]:0,c.scores,d_k,n,ldn,n_q,n_head,g,scale);
            k_merge<<<gm,b,0,c.compS>>>(c.scores,(const float*)c.Vc[s],c.m,c.l,c.acc,n,ldn,d_v,n_q,n_head,g);
        }
        cudaEventRecord(c.evcomp[s],c.compS);
    }
    k_finalize<<<dim3((d_v+63)/64,n_q,n_head),dim3(64,1,1),0,c.compS>>>(c.acc,c.l,c.out,d_v,n_q,n_head);
    CUDA_OK(cudaMemcpyAsync(out,c.out,sz_acc,cudaMemcpyDeviceToHost,c.compS));
    CUDA_OK(cudaStreamSynchronize(c.compS));
    CUDA_OK(cudaGetLastError());
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
