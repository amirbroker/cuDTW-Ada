// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | tests/bench.cu
//
// Phase 2 measurement harness: kernel-only GCUPS + theoretical occupancy.
// Times ONLY the compute kernels (no disk I/O, no H2D/D2H) so the GCUPS number
// is comparable across configs (item #12). Reports occupancy from the CUDA
// occupancy API for the actual launch config.
// NOTE: this kernel-only window DIFFERS from the drivers (main_v4/main_v5),
// whose GCUPS also includes H2D+D2H transfers. Compare bench-to-bench and
// driver-to-driver, not bench-to-driver.
//
// Build:  make bench
// Run:    ./bench                 # default sizes
//         ./bench 395 4437 65536  # Lq Ls N
//
// To A/B test occupancy knobs, rebuild with e.g.:
//   make bench GENWPB=8 TILEWPB=8 GENMINB=1 TILEMINB=1
// (see Makefile bench target).
// ─────────────────────────────────────────────────────────────────────────────
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <random>
#include <string>

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

// Report theoretical occupancy for a representative kernel at the active config.
template<typename Kern>
static void report_occupancy(const char* tag, Kern kern, int block_threads)
{
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, kern, block_threads, 0));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int warps_per_block = block_threads / 32;
    int active_warps    = max_blocks * warps_per_block;
    int max_warps_sm    = p.maxThreadsPerMultiProcessor / 32;
    double occ = (double)active_warps / max_warps_sm;

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));

    std::cout << "  [" << tag << "] block=" << block_threads
              << " regs/thread=" << attr.numRegs
              << " blocks/SM=" << max_blocks
              << " warps/SM=" << active_warps << "/" << max_warps_sm
              << " → occupancy=" << std::fixed << std::setprecision(0)
              << occ * 100 << "%\n";
}

int main(int argc, char** argv)
{
    int Lq = (argc > 1) ? atoi(argv[1]) : 395;
    int Ls = (argc > 2) ? atoi(argv[2]) : 4437;
    int N  = (argc > 3) ? atoi(argv[3]) : (1 << 16);

    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::cout << "cuDTW-Ada bench | " << p.name << " sm_" << p.major << p.minor << "\n";
    std::cout << "  Lq=" << Lq << " Ls=" << Ls << " N=" << N
              << " | GEN_WPB=" << CUDTW_GENERIC_WARPS_PER_BLOCK
              << " TILE_WPB=" << CUDTW_TILED_WARPS_PER_BLOCK << "\n\n";

    // Random normalized-ish data (values don't affect timing).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-3.f, 3.f);
    std::vector<float> q(Lq), s((long long)N * Ls);
    for (auto& v : q) v = d(rng);
    for (auto& v : s) v = d(rng);

    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, q.data(), sizeof(float) * Lq));

    float *d_s = nullptr, *d_d = nullptr;
    CUDA_CHECK(cudaMalloc(&d_s, sizeof(float) * (long long)N * Ls));
    CUDA_CHECK(cudaMalloc(&d_d, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_s, s.data(), sizeof(float) * (long long)N * Ls,
                          cudaMemcpyHostToDevice));

    // Pre-allocated scratch (no malloc in timed region).
    DtwScratch<value_t> scr;
    const int K_full = (Ls + 32) / 32;
    const long long pad_stride_max = (K_full <= 120)
        ? (long long)K_full * 32 : SCRATCH_TILE_SIZE;
    scr.ensure((long long)N, (long long)Lq, pad_stride_max);

    // Occupancy report for a representative kernel on this path.
    const char* path = (K_full <= 120) ? "generic" : "tiled";
    std::cout << "Occupancy (theoretical, active config):\n";
    if (K_full <= 120) {
        report_occupancy("generic K64", shfl_FullDTW_K64<index_t,value_t>,
                         32 * CUDTW_GENERIC_WARPS_PER_BLOCK);
    } else {
        report_occupancy("tiled K64 RW", shfl_FullDTW_Tiled_K64_RW_impl<index_t,value_t>,
                         32 * CUDTW_TILED_WARPS_PER_BLOCK);
        report_occupancy("tiled K64 last", shfl_FullDTW_Tiled_K64<index_t,value_t>,
                         32 * CUDTW_TILED_WARPS_PER_BLOCK);
    }
    std::cout << "\n";

    // Warmup.
    dist_any<index_t,value_t>(scr, d_s, d_d, (index_t)N, (index_t)Lq, (index_t)Ls, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed: kernel only, averaged over several iterations.
    const int iters = 20;
    cudaEvent_t t0, t1; float ms = 0;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i)
        dist_any<index_t,value_t>(scr, d_s, d_d, (index_t)N, (index_t)Lq, (index_t)Ls, 0);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    double cells = (double)Lq * Ls * N;
    double gcups = cells / (ms * 1e-3) / 1e9;
    std::cout << "Throughput (" << path << " path, kernel-only):\n";
    std::cout << "  " << std::fixed << std::setprecision(3) << ms << " ms/iter"
              << " | " << std::setprecision(1) << gcups << " GCUPS\n";

    scr.free();
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_d));
    return 0;
}
