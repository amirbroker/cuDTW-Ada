// cuDTW-Ada | Phase 7 | tests/validate_opt.cu
// Ada Lovelace Tuning: compare OPT kernel vs GENERIC on all lengths.
//
// Build: make validate_opt
// Run:   ./validate_opt

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

#include "include/hpc_helpers.hpp"

typedef float    value_t;
typedef uint64_t index_t;

constexpr index_t MAX_Q = 8192;
__constant__ value_t cQuery[MAX_Q];

// Include BOTH kernel sets
#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"   // Phase 2 kernels
#include "include/kernels/SHFL_FULLDTW_OPT.cuh"        // Phase 7 optimized

#include "include/cudtw_dispatcher_v2.hpp"

// ── OPT dispatcher wrapper ────────────────────────────────────────────────────
namespace OptDTW {
    inline int K(int L){ return (L+32)/32; }

    template<typename index_t, typename value_t>
    void dist_opt_sync(
        const value_t* d_subj_packed,
        value_t*       d_dist,
        index_t N, index_t Lq, index_t Ls)
    {
        int k = K((int)Ls);
        int stride = k*32;
        value_t* d_pad=nullptr;
        CUDA_CHECK(cudaMalloc(&d_pad, sizeof(value_t)*(long long)N*stride));

        // Pad (reuse Phase 2 padding kernel)
        GenericDTW::pad_sequences_gpu<value_t>(
            d_subj_packed, d_pad, (int)N, (int)Ls, stride, 0);

        // Launch OPT kernel
        dispatch_opt<index_t,value_t>(
            d_pad, d_dist, N, Lq, Ls, (index_t)stride, k, 0);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_pad));
        CUERR
    }
}

// ── Benchmark helper ──────────────────────────────────────────────────────────
struct Result { double gcpus; };

Result bench_generic(int Lq, int Ls, int N)
{
    int K=(Ls+32)/32, stride=K*32;
    std::mt19937 rng(42); std::uniform_real_distribution<float> d(-5,5);
    std::vector<float> q(Lq),s((long long)N*Ls);
    for(auto& v:q) v=d(rng); for(auto& v:s) v=d(rng);
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery,q.data(),sizeof(float)*Lq));

    float *d_s=nullptr,*d_d=nullptr,*d_p=nullptr;
    CUDA_CHECK(cudaMalloc(&d_s,sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_d,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_p,sizeof(float)*(long long)N*stride));
    CUDA_CHECK(cudaMemcpy(d_s,s.data(),sizeof(float)*(long long)N*Ls,cudaMemcpyHostToDevice));

    // warmup
    GenericDTW::dist_v2<index_t,float>(d_s,d_d,(index_t)N,(index_t)Lq,(index_t)Ls,0,d_p,(index_t)stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0,t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    GenericDTW::dist_v2<index_t,float>(d_s,d_d,(index_t)N,(index_t)Lq,(index_t)Ls,0,d_p,(index_t)stride);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));

    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d)); CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return {(double)Lq*Ls*N/(ms*1e-3)/1e9};
}

Result bench_opt(int Lq, int Ls, int N)
{
    int K=(Ls+32)/32, stride=K*32;
    std::mt19937 rng(42); std::uniform_real_distribution<float> d(-5,5);
    std::vector<float> q(Lq),s((long long)N*Ls);
    for(auto& v:q) v=d(rng); for(auto& v:s) v=d(rng);
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery,q.data(),sizeof(float)*Lq));

    float *d_s=nullptr,*d_d=nullptr,*d_p=nullptr;
    CUDA_CHECK(cudaMalloc(&d_s,sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_d,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_p,sizeof(float)*(long long)N*stride));
    CUDA_CHECK(cudaMemcpy(d_s,s.data(),sizeof(float)*(long long)N*Ls,cudaMemcpyHostToDevice));

    // warmup
    GenericDTW::pad_sequences_gpu<float>(d_s,d_p,(int)N,(int)Ls,stride,0);
    dispatch_opt<index_t,float>(d_p,d_d,(index_t)N,(index_t)Lq,(index_t)Ls,(index_t)stride,K,0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0,t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    GenericDTW::pad_sequences_gpu<float>(d_s,d_p,(int)N,(int)Ls,stride,0);
    CUDA_CHECK(cudaEventRecord(t0));
    dispatch_opt<index_t,float>(d_p,d_d,(index_t)N,(index_t)Lq,(index_t)Ls,(index_t)stride,K,0);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));

    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d)); CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return {(double)Lq*Ls*N/(ms*1e-3)/1e9};
}

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"GPU: "<<prop.name<<"  sm_"<<prop.major<<prop.minor<<"\n\n";

    std::cout<<"═══ Phase 7: OPT vs GENERIC Throughput (Lq=395, N=131072) ═══\n";
    std::cout<<"  K≤64: subject in registers  |  K>64: subject via __ldg\n\n";
    std::cout<<"  Ls     K   GENERIC   OPT      speedup\n";
    std::cout<<"  ─────────────────────────────────────\n";

    for(int Ls:{127,255,395,511,820,1023,1089,2047,2469,3000,3839})
    {
        int K=(Ls+32)/32;
        int N=(Ls<=1000)?1<<17:1<<16;
        auto g=bench_generic(395,Ls,N);
        auto o=bench_opt(395,Ls,N);
        std::cout<<"  Ls="<<std::setw(5)<<Ls
                 <<" K="<<std::setw(4)<<K
                 <<" | "<<std::setw(7)<<std::fixed<<std::setprecision(0)<<g.gcpus
                 <<" | "<<std::setw(7)<<o.gcpus<<" GCPUS"
                 <<" | "<<std::setprecision(2)<<o.gcpus/g.gcpus<<"×"
                 <<(K<=64?" (regs)":" (__ldg)")
                 <<"\n";
    }

    std::cout<<"\n═══ Summary ════════════════════════════════════════════════\n";
    std::cout<<" Goal: K=120 (L=3839) from ~390 GCPUS → >1000 GCPUS\n";
    std::cout<<" __launch_bounds__ + __ldg should reduce register spill\n";
    std::cout<<"═══════════════════════════════════════════════════════════\n\n";
    return 0;
}
