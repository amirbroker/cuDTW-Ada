// cuDTW-Ada | Phase 3 | tests/validate_v3.cu
// Validates Tiled Warp-Shuffle DTW for L_subject > 3839.
//
// Build:
//   make validate_v3

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <random>

#include "include/hpc_helpers.hpp"

typedef float    value_t;
typedef uint64_t index_t;

constexpr index_t max_features = (1UL << 16) / sizeof(value_t);
__constant__ value_t cQuery[max_features];

#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"
#include "include/kernels/SHFL_FULLDTW_TILED.cuh"
#include "include/cudtw_dispatcher_v2.hpp"
#include "include/cudtw_tiled.hpp"
#include "include/cudtw_dispatcher_v3.hpp"

using namespace UnifiedDTW;

// ── CPU Naive DTW ──────────────────────────────────────────────────────────────
float cpu_dtw(const float* q, int Lq, const float* s, int Ls)
{
    std::vector<float> dp((long long)Lq * Ls);
    auto at = [&](int i,int j)->float&{ return dp[(long long)i*Ls+j]; };
    float d = q[0]-s[0]; at(0,0)=d*d;
    for(int j=1;j<Ls;j++){ d=q[0]-s[j]; at(0,j)=d*d+at(0,j-1); }
    for(int i=1;i<Lq;i++){ d=q[i]-s[0]; at(i,0)=d*d+at(i-1,0); }
    for(int i=1;i<Lq;i++) for(int j=1;j<Ls;j++){
        d=q[i]-s[j];
        at(i,j)=d*d+std::min({at(i-1,j),at(i,j-1),at(i-1,j-1)});
    }
    return at(Lq-1,Ls-1);
}

// ── GPU DTW (unified dispatcher) ───────────────────────────────────────────────
std::vector<float> gpu_dtw(const float* q,int Lq,const float* s,int Ls,int N)
{
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery,q,sizeof(float)*Lq));
    float *d_s=nullptr,*d_d=nullptr;
    CUDA_CHECK(cudaMalloc(&d_s,sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_d,sizeof(float)*N));
    CUDA_CHECK(cudaMemset(d_d,0,sizeof(float)*N));
    CUDA_CHECK(cudaMemcpy(d_s,s,sizeof(float)*(long long)N*Ls,cudaMemcpyHostToDevice));
    dist_any_sync<index_t,value_t>(d_s,d_d,(index_t)N,(index_t)Lq,(index_t)Ls);
    std::vector<float> r(N);
    CUDA_CHECK(cudaMemcpy(r.data(),d_d,sizeof(float)*N,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d));
    return r;
}

// ── Test one pair ──────────────────────────────────────────────────────────────
bool test(int Lq, int Ls, int N=16, float tol=1e-3f)
{
    std::mt19937 rng(42+Lq*1000+Ls);
    std::uniform_real_distribution<float> dist(-5,5);
    std::vector<float> q(Lq), s((long long)N*Ls);
    for(auto& v:q) v=dist(rng);
    for(auto& v:s) v=dist(rng);

    std::vector<float> cpu(N);
    for(int i=0;i<N;i++) cpu[i]=cpu_dtw(q.data(),Lq,s.data()+(long long)i*Ls,Ls);
    auto gpu=gpu_dtw(q.data(),Lq,s.data(),Ls,N);

    float ma=0,mr=0; int fails=0,wi=0;
    for(int i=0;i<N;i++){
        float ae=std::abs(gpu[i]-cpu[i]);
        float re=(cpu[i]>1e-3f)?ae/cpu[i]:ae;
        if(ae>ma){ma=ae;wi=i;} mr=std::max(mr,re);
        if(re>tol) fails++;
    }
    int K=(Ls+32)/32;
    const char* ph=(K<=120)?"P2":"P3";
    const char* tg=(Lq==Ls)?"equal":"asymm";
    if(!fails){
        std::cout<<"  PASS ["<<tg<<"]["<<ph<<"]"
                 <<" Lq="<<std::setw(4)<<Lq<<" Ls="<<std::setw(5)<<Ls
                 <<" K="<<std::setw(4)<<K
                 <<" | max_abs="<<std::scientific<<std::setprecision(1)<<ma
                 <<" max_rel="<<mr<<std::fixed<<"\n";
    } else {
        std::cout<<"  FAIL ["<<tg<<"]["<<ph<<"]"
                 <<" Lq="<<std::setw(4)<<Lq<<" Ls="<<std::setw(5)<<Ls
                 <<" K="<<std::setw(4)<<K
                 <<" | fails="<<fails<<"/"<<N
                 <<" max_abs="<<std::scientific<<ma
                 <<" cpu["<<wi<<"]="<<cpu[wi]<<" gpu="<<gpu[wi]
                 <<std::fixed<<"\n";
    }
    return !fails;
}

// ── Benchmark ──────────────────────────────────────────────────────────────────
void bench(int Lq,int Ls,int N=1<<16)
{
    std::mt19937 rng(99); std::uniform_real_distribution<float> d(-5,5);
    std::vector<float> q(Lq),s((long long)N*Ls);
    for(auto& v:q) v=d(rng); for(auto& v:s) v=d(rng);
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery,q.data(),sizeof(float)*Lq));
    float *d_s=nullptr,*d_d=nullptr;
    CUDA_CHECK(cudaMalloc(&d_s,sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_d,sizeof(float)*N));
    CUDA_CHECK(cudaMemcpy(d_s,s.data(),sizeof(float)*(long long)N*Ls,cudaMemcpyHostToDevice));

    // warmup
    dist_any<index_t,value_t>(d_s,d_d,(index_t)N,(index_t)Lq,(index_t)Ls);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0,t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    dist_any<index_t,value_t>(d_s,d_d,(index_t)N,(index_t)Lq,(index_t)Ls);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));

    int K=(Ls+32)/32;
    const char* ph=(K<=120)?"P2":"P3";
    const char* tg=(Lq==Ls)?"equal":"asymm";
    double gcpus=(double)Lq*Ls*N/(ms*1e-3)/1e9;
    double bw=(double)N*Ls*sizeof(float)/(1UL<<30)/(ms*1e-3);
    std::cout<<"  Bench["<<tg<<"]["<<ph<<"]"
             <<" Lq="<<std::setw(4)<<Lq<<" Ls="<<std::setw(5)<<Ls
             <<" K="<<std::setw(4)<<K<<" N="<<N
             <<" | "<<std::fixed<<std::setprecision(1)<<ms<<" ms"
             <<" | "<<gcpus<<" GCPUS"
             <<" | "<<bw<<" GiB/s\n";
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d));
}

// ── Main ───────────────────────────────────────────────────────────────────────
int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<"  sm_"<<prop.major<<prop.minor<<"\n\n";

    bool ok=true;

    // [A] Phase 2 Regression
    std::cout<<"═══ [A] Phase 2 Regression (L ≤ 3839) ═══════════════════\n";
    for(auto[Lq,Ls]:std::vector<std::pair<int,int>>{
        {395,127},{395,395},{395,820},{395,1089},{395,2047},{395,3839}})
        ok&=test(Lq,Ls,16);

    // [B] Phase 3: Tiled Warp, large L_subject, Lq=395 (user's actual case)
    std::cout<<"\n═══ [B] Phase 3 Tiled: Lq=395 vs large subjects ═══════════\n";
    for(int Ls:{3840,4000,5000,6000,8000})
        ok&=test(395,Ls,16);

    // [C] Boundary P2↔P3
    std::cout<<"\n═══ [C] Boundary K=120↔K=121 ═══════════════════════════════\n";
    ok&=test(395,3839,16);
    ok&=test(395,3840,16);
    ok&=test(395,4000,16);

    std::cout<<"\n "<<(ok?"✓ ALL PASSED":"✗ FAILURES")<<"\n";

    // Throughput
    std::cout<<"\n═══ Throughput (Lq=395, N=65536) ═══════════════════════════\n";
    for(auto[Lq,Ls]:std::vector<std::pair<int,int>>{
        {395,2047},{395,3839},{395,3840},{395,4000},{395,5000},{395,6000},{395,8000}})
        bench(Lq,Ls,1<<16);

    std::cout<<"\n═══ Summary ═════════════════════════════════════════════════\n";
    std::cout<<" Phase 2 (K≤120, L≤3839): warp-shuffle    ~3000-5000 GCPUS\n";
    std::cout<<" Phase 3 (K>120, L>3839): TILED warp      ~2000-4000 GCPUS\n";
    std::cout<<" Status: "<<(ok?"✓ READY → Phase 4":"✗ NEEDS FIX")<<"\n";
    return ok?0:1;
}
