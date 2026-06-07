// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | tests/validate.cu
//
// Correctness validation against a CPU reference (FullDTW). Covers:
//   • generic path  (K ≤ 120, L_subject ≤ 3839)
//   • tiled path     (K > 120) — the path ALL real user data uses
//   • exact tile-boundary lengths (2046..2050, 4093..4096, 6140..6142)
//   • real user lengths (4437, 5788) with the real query length (395)
//   • asymmetric L_query ≠ L_subject
//
// The kernels compute STANDARD (constrained) FullDTW: dp[0][0]=c(0,0),
// first row/col cumulative, min over (up,left,diag). The reference below
// matches that exactly. (Data is assumed pre-normalized; no z-norm here.)
//
// Build:  make validate
// Run:    ./validate
// Exit code: 0 if all PASS, 1 if any FAIL (CI-friendly).
// ─────────────────────────────────────────────────────────────────────────────
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <random>

#include "include/hpc_helpers.hpp"
#include "include/cudtw_constants.hpp"

typedef float    value_t;
typedef uint64_t index_t;

__constant__ value_t cQuery[CUDTW_QUERY_STORAGE];

#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"
#include "include/kernels/SHFL_FULLDTW_TILED.cuh"
#include "include/cudtw_scratch.hpp"
#include "include/cudtw_dispatcher_v2.hpp"
#include "include/cudtw_tiled.hpp"
#include "include/cudtw_dispatcher_v3.hpp"

using namespace UnifiedDTW;

// ── CPU reference: constrained FullDTW (matches the kernels) ────────────────
static double cpu_dtw(const float* q, int Lq, const float* s, int Ls)
{
    std::vector<double> prev(Ls), cur(Ls);
    auto sq = [](double x){ return x*x; };
    prev[0] = sq(q[0]-s[0]);
    for (int j = 1; j < Ls; ++j) prev[j] = sq(q[0]-s[j]) + prev[j-1];
    for (int i = 1; i < Lq; ++i) {
        cur[0] = sq(q[i]-s[0]) + prev[0];
        for (int j = 1; j < Ls; ++j)
            cur[j] = sq(q[i]-s[j]) + std::min({prev[j], cur[j-1], prev[j-1]});
        std::swap(prev, cur);
    }
    return prev[Ls-1];
}

// ── GPU via unified dispatcher (self-allocating sync path) ──────────────────
static std::vector<float> gpu_dtw(const float* q, int Lq,
                                  const float* s, int Ls, int N)
{
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, q, sizeof(float)*Lq));
    float *d_s=nullptr, *d_d=nullptr;
    CUDA_CHECK(cudaMalloc(&d_s, sizeof(float)*(long long)N*Ls));
    CUDA_CHECK(cudaMalloc(&d_d, sizeof(float)*N));
    CUDA_CHECK(cudaMemset(d_d, 0, sizeof(float)*N));
    CUDA_CHECK(cudaMemcpy(d_s, s, sizeof(float)*(long long)N*Ls,
                          cudaMemcpyHostToDevice));
    dist_any_sync<index_t,value_t>(d_s, d_d, (index_t)N, (index_t)Lq, (index_t)Ls);
    std::vector<float> r(N);
    CUDA_CHECK(cudaMemcpy(r.data(), d_d, sizeof(float)*N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d));
    return r;
}

static int g_pass = 0, g_fail = 0;

static void run(int Lq, int Ls, int N=8, float tol=1e-4f, unsigned seed=1234)
{
    std::mt19937 rng(seed + Lq*131u + Ls*977u);
    std::uniform_real_distribution<float> d(-3.f, 3.f);
    std::vector<float> q(Lq), s((long long)N*Ls);
    for (auto& v : q) v = d(rng);
    for (auto& v : s) v = d(rng);

    auto gpu = gpu_dtw(q.data(), Lq, s.data(), Ls, N);

    float worst = 0.f; int bad = 0, wi = 0; double wc = 0; float wg = 0;
    for (int i = 0; i < N; ++i) {
        double c = cpu_dtw(q.data(), Lq, s.data() + (long long)i*Ls, Ls);
        double ae = std::abs((double)gpu[i] - c);
        double re = (c > 1e-3) ? ae / c : ae;
        if (re > worst) { worst = (float)re; wi = i; wc = c; wg = gpu[i]; }
        if (re > tol) ++bad;
    }
    int K = (Ls + 32) / 32;
    const char* ph = (K <= 120) ? "generic" : "tiled  ";
    int n_tiles = 1;
    if (K > 120) {
        TiledDTW::TileSpan sched[256];
        n_tiles = TiledDTW::build_tile_schedule(Ls, sched, 256);
    }
    if (bad == 0) {
        ++g_pass;
        std::cout << "  PASS [" << ph << "] Lq=" << std::setw(4) << Lq
                  << " Ls=" << std::setw(5) << Ls
                  << " tiles=" << n_tiles
                  << " | max_rel=" << std::scientific << std::setprecision(2)
                  << worst << std::fixed << "\n";
    } else {
        ++g_fail;
        std::cout << "  FAIL [" << ph << "] Lq=" << std::setw(4) << Lq
                  << " Ls=" << std::setw(5) << Ls
                  << " tiles=" << n_tiles
                  << " | bad=" << bad << "/" << N
                  << " max_rel=" << std::scientific << worst
                  << " cpu[" << wi << "]=" << wc << " gpu=" << wg
                  << std::fixed << "\n";
    }
}

int main()
{
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::cout << "cuDTW-Ada validate | " << p.name
              << " sm_" << p.major << p.minor << "\n";

    std::cout << "\n── Generic path (K ≤ 120) ──────────────────────────────\n";
    for (int Ls : {32, 48, 63, 64, 80, 95, 96, 127, 200, 500, 1000, 2047, 3000, 3839})
        run(395, Ls);
    run(395, 395);   // symmetric
    run(64, 500);    // short query

    std::cout << "\n── Tiled path: tile-boundary stress (K > 120) ──────────\n";
    for (int Ls : {3840, 3841, 4093, 4094, 4095, 4096,
                   6140, 6141, 6142, 8188, 8189})
        run(395, Ls);

    std::cout << "\n── Tiled path: REAL user lengths ───────────────────────\n";
    run(395, 4437);
    run(395, 5788);
    run(100, 4437);   // asymmetric
    run(100, 5788);

    std::cout << "\n────────────────────────────────────────────────────────\n";
    std::cout << "  RESULT: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
