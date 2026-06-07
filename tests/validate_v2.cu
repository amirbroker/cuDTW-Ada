// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 2 | tests/validate_v2.cu
//
// Validates generic kernel for:
//   (a) Equal-length cases that WERE NOT in original kernels (e.g. L=300, L=820)
//   (b) Asymmetric DTW: L_query ≠ L_subject (core use case)
//   (c) Boundary cases: L=128 (first non-original), L=3839 (max K=120)
//
// Build (add to Makefile or run directly):
//   nvcc -O3 -std=c++20 -arch=sm_89 \
//        -Xcompiler='-fopenmp -march=native -O3 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0' \
//        -I src -I src/include tests/validate_v2.cu -o validate_v2
//
// Run: ./validate_v2
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <random>
#include <cassert>
#include <chrono>

#include "include/hpc_helpers.hpp"

typedef float    value_t;
typedef uint64_t index_t;

// cQuery must be declared before including any kernel header
constexpr index_t max_features = (1UL << 16) / sizeof(value_t);
__constant__ value_t cQuery[max_features];

#include "include/cudtw_dispatcher_v2.hpp"
using namespace GenericDTW;

// ─────────────────────────────────────────────────────────────────────────────
// CPU Naive Full DTW  (L_query × L_subject DP matrix)
// ─────────────────────────────────────────────────────────────────────────────
float cpu_dtw(const float* q, int Lq, const float* s, int Ls)
{
    std::vector<float> dp(Lq * Ls, 0.0f);
    auto at = [&](int i, int j) -> float& { return dp[i * Ls + j]; };

    float d = q[0] - s[0];
    at(0, 0) = d * d;
    for (int j = 1; j < Ls; j++) { d = q[0]-s[j]; at(0,j) = d*d + at(0,j-1); }
    for (int i = 1; i < Lq; i++) { d = q[i]-s[0]; at(i,0) = d*d + at(i-1,0); }
    for (int i = 1; i < Lq; i++)
        for (int j = 1; j < Ls; j++) {
            d = q[i] - s[j];
            at(i,j) = d*d + std::min({at(i-1,j), at(i,j-1), at(i-1,j-1)});
        }
    return at(Lq-1, Ls-1);
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU DTW batch via generic kernel
// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> gpu_dtw_v2(
    const float* query, int Lq,
    const float* subjects, int Ls,
    int N)
{
    // Upload query to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, query, sizeof(float) * Lq));

    // Alloc GPU memory for subjects (packed) and results
    float* d_subj = nullptr;
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subj, sizeof(float) * (long long)N * Ls));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * N));
    CUDA_CHECK(cudaMemset(d_dist, 0, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_subj, subjects,
                          sizeof(float) * (long long)N * Ls,
                          cudaMemcpyHostToDevice));

    // Call dispatcher (allocates pad buffer internally)
    dist_v2_simple<index_t, value_t>(
        d_subj, d_dist,
        (index_t)N, (index_t)Lq, (index_t)Ls);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUERR

    std::vector<float> results(N);
    CUDA_CHECK(cudaMemcpy(results.data(), d_dist,
                          sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_subj));
    CUDA_CHECK(cudaFree(d_dist));
    return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test one (Lq, Ls) pair
// ─────────────────────────────────────────────────────────────────────────────
bool test_pair(int Lq, int Ls, int N = 128, float tol_rel = 1e-3f) {
    std::mt19937 rng(42 + Lq * 1000 + Ls);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> query(Lq), subjects((long long)N * Ls);
    for (auto& v : query)    v = dist(rng);
    for (auto& v : subjects) v = dist(rng);

    // CPU reference
    std::vector<float> cpu_res(N);
    for (int i = 0; i < N; i++)
        cpu_res[i] = cpu_dtw(query.data(), Lq,
                             subjects.data() + (long long)i * Ls, Ls);

    // GPU generic kernel
    auto gpu_res = gpu_dtw_v2(query.data(), Lq, subjects.data(), Ls, N);

    // Compare
    float max_abs = 0, max_rel = 0;
    int fails = 0;
    int worst = 0;
    for (int i = 0; i < N; i++) {
        float ae = std::abs(gpu_res[i] - cpu_res[i]);
        float re = (cpu_res[i] > 1e-3f) ? ae / cpu_res[i] : ae;
        if (ae > max_abs) { max_abs = ae; worst = i; }
        max_rel = std::max(max_rel, re);
        if (re > tol_rel) fails++;
    }

    int  K       = compute_K(Ls);
    bool passed  = (fails == 0);
    auto tag     = (Lq == Ls) ? "equal" : "asymm";

    if (passed) {
        std::cout << "  PASS [" << tag << "] Lq=" << std::setw(4) << Lq
                  << " Ls=" << std::setw(4) << Ls
                  << " K=" << std::setw(3) << K
                  << " | max_abs=" << std::scientific << std::setprecision(1) << max_abs
                  << " max_rel=" << max_rel << std::fixed << "\n";
    } else {
        std::cout << "  FAIL [" << tag << "] Lq=" << std::setw(4) << Lq
                  << " Ls=" << std::setw(4) << Ls
                  << " K=" << std::setw(3) << K
                  << " | fails=" << fails << "/" << N
                  << " max_abs=" << std::scientific << max_abs
                  << " | cpu[" << worst << "]=" << cpu_res[worst]
                  << " gpu=" << gpu_res[worst]
                  << std::fixed << "\n";
    }
    return passed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Throughput benchmark
// ─────────────────────────────────────────────────────────────────────────────
void benchmark(int Lq, int Ls, int N = 1 << 17) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    std::vector<float> query(Lq), subjects((long long)N * Ls);
    for (auto& v : query)    v = d(rng);
    for (auto& v : subjects) v = d(rng);

    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, query.data(), sizeof(float)*Lq));

    int    K      = compute_K(Ls);
    int    stride = K * 32;
    float* d_subj = nullptr;
    float* d_pad  = nullptr;
    float* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_subj, sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_pad,  sizeof(float)*(long long)N*stride));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float)*N));
    CUDA_CHECK(cudaMemcpy(d_subj, subjects.data(),
                          sizeof(float)*(long long)N*Ls, cudaMemcpyHostToDevice));

    // Warmup
    dist_v2<index_t,value_t>(d_subj, d_dist,
                              (index_t)N,(index_t)Lq,(index_t)Ls,
                              (cudaStream_t)0, d_pad, (index_t)stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t t0, t1; float ms;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    dist_v2<index_t,value_t>(d_subj, d_dist,
                              (index_t)N,(index_t)Lq,(index_t)Ls,
                              (cudaStream_t)0, d_pad, (index_t)stride);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // GCPUS = (Lq × Ls × N) / time_seconds / 1e9
    double gcpus = (double)Lq * Ls * N / (ms*1e-3) / 1e9;
    double bw    = (double)N * Ls * sizeof(float) / (1UL<<30) / (ms*1e-3);

    auto tag = (Lq==Ls) ? "equal" : "asymm";
    std::cout << "  Bench [" << tag << "] Lq=" << std::setw(4) << Lq
              << " Ls=" << std::setw(4) << Ls
              << " K=" << std::setw(3) << K
              << " N=" << N
              << " | " << std::fixed << std::setprecision(1)
              << ms << " ms"
              << " | " << std::setprecision(1) << gcpus << " GCPUS"
              << " | " << std::setprecision(1) << bw << " GiB/s\n";

    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_subj)); CUDA_CHECK(cudaFree(d_pad));
    CUDA_CHECK(cudaFree(d_dist));
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "  sm_" << prop.major << prop.minor << "\n\n";

    // ── Correctness tests ─────────────────────────────────────────────────────
    std::cout << "═══ Phase 2 Correctness Validation ══════════════════════════\n";
    std::cout << " tolerance: rel < 0.1%\n\n";

    bool all_ok = true;

    // (A) Equal-length, NEW lengths not in original kernels
    std::cout << " [A] Equal-length, non-standard (new kernel needed):\n";
    for (int L : {128, 200, 300, 395, 512, 820, 1000, 1024, 1089, 1500, 2000, 2047, 2048, 2469, 3000, 3839})
        all_ok &= test_pair(L, L);

    // (B) Asymmetric: target=395 vs various subject lengths (the real use case!)
    std::cout << "\n [B] Asymmetric DTW — target=395 vs subjects (CORE USE CASE):\n";
    for (int Ls : {127, 200, 395, 512, 820, 1089, 1500, 2047, 2469, 3000, 3839})
        all_ok &= test_pair(395, Ls);

    // (C) Asymmetric: short query, long subject
    std::cout << "\n [C] Asymmetric — short query:\n";
    all_ok &= test_pair(100, 500);
    all_ok &= test_pair(100, 1000);
    all_ok &= test_pair(50,  2000);

    // (D) Boundary checks
    std::cout << "\n [D] Boundary cases:\n";
    all_ok &= test_pair(395, 127);   // Ls=127 → K=4 (original size but generic kernel)
    all_ok &= test_pair(395, 2047);  // K=64
    all_ok &= test_pair(395, 3839);  // K=120 (maximum for Phase 2)

    std::cout << "\n " << (all_ok ? "✓ ALL PASSED" : "✗ FAILURES DETECTED") << "\n";

    // ── Throughput benchmark ──────────────────────────────────────────────────
    std::cout << "\n═══ Phase 2 Throughput Benchmark (N=131072) ═════════════════\n";
    std::cout << " Comparing new generic kernel vs Phase 1 baseline:\n\n";

    // Key lengths from user's dataset
    for (auto [Lq, Ls] : std::vector<std::pair<int,int>>{
            {395,  127},
            {395,  395},
            {395,  820},
            {395, 1089},
            {395, 2047},
            {395, 2469},
            {395, 3000},
            {395, 3839}})
    {
        benchmark(Lq, Ls, 1 << 17);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n═══ Phase 2 Summary ═══════════════════════════════════════════\n";
    std::cout << " Supported subject lengths: 1 .. 3839 (K=1..120)\n";
    std::cout << " Asymmetric DTW: YES (L_query ≠ L_subject)\n";
    std::cout << " Status: " << (all_ok ? "✓ READY → Phase 3" : "✗ NEEDS FIX") << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    return all_ok ? 0 : 1;
}
