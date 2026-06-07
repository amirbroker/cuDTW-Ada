// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 5 | src/main_v5.cu
//
// Streaming pipeline with:
//   1. Pinned (page-locked) host memory → async H2D/D2H transfers
//   2. Double-buffer with N_STREAMS streams → overlap transfer + compute
//   3. cudaAccessPolicyWindow → lock cQuery in L2 cache
//
// Why these matter for target=395 vs sequences 127..5000+:
//   Phase 4: sequential (memcpy → compute → memcpy → ...) — GPU idle during transfer
//   Phase 5: pipeline (stream0 computes while stream1 transfers) — no idle time
//
// Expected improvement:
//   Small N batches (N<1000): up to 3-4× faster
//   Large N batches (N>50000): 5-15% faster (already compute-bound)
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <numeric>

#include "include/hpc_helpers.hpp"
#include "include/cudtw_constants.hpp"

namespace fs = std::filesystem;

typedef float    value_t;
typedef uint64_t index_t;

constexpr index_t MAX_QUERY = CUDTW_MAX_QUERY;
__constant__ value_t cQuery[CUDTW_QUERY_STORAGE];

#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"
#include "include/kernels/SHFL_FULLDTW_TILED.cuh"
#include "include/cudtw_scratch.hpp"
#include "include/cudtw_dispatcher_v2.hpp"
#include "include/cudtw_tiled.hpp"
#include "include/cudtw_dispatcher_v3.hpp"

using namespace UnifiedDTW;

// ── Config ─────────────────────────────────────────────────────────────────
static constexpr int  N_STREAMS   = 4;         // number of concurrent streams
static constexpr long GPU_HEADROOM = 2L<<30;   // keep 2GB free

// ─────────────────────────────────────────────────────────────────────────────
// I/O helpers
// ─────────────────────────────────────────────────────────────────────────────
std::vector<value_t> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t nb = f.tellg(); f.seekg(0);
    std::vector<value_t> v(nb / sizeof(value_t));
    f.read(reinterpret_cast<char*>(v.data()), nb);
    return v;
}
void write_bin(const std::string& p, const std::vector<value_t>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(value_t));
}
index_t name_to_len(const std::string& nm) {
    auto d = nm.rfind('.'); if (d==std::string::npos) return 0;
    try { return std::stoull(nm.substr(0,d)); } catch(...) { return 0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// NOTE on L2 persistence (REMOVED — item #3):
//   The previous version persisted d_subj[0] in L2. That buffer changes every
//   batch and each element is read exactly once by the kernel, so persisting it
//   gives no reuse benefit and needlessly reserves a carve-out of the 72MB L2,
//   evicting genuinely reused data. The truly hot data (the query) lives in
//   __constant__ memory and is served from the constant cache, not L2, so an
//   access-policy window cannot help it. Persistence is therefore not used.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Streaming processor for one sequences file
// Uses N_STREAMS double-buffered async pipeline:
//   while stream[s%N] computes batch[s], stream[(s+1)%N] transfers batch[s+1]
// ─────────────────────────────────────────────────────────────────────────────
bool process_file_streamed(
    const std::string& seq_path,
    const std::string& res_path,
    index_t L_query,
    index_t L_subject,
    bool verbose)
{
    auto seqs = read_bin(seq_path);
    if (seqs.empty()) {
        std::cerr << "[ERROR] " << seq_path
                  << ": empty or unreadable file — skipped.\n";
        return false;
    }

    const index_t N_total = seqs.size() / L_subject;
    if (N_total == 0 || seqs.size() % L_subject != 0) {
        std::cerr << "[ERROR] " << seq_path << ": file holds "
                  << seqs.size() << " floats, which is not a whole multiple of "
                  << "the sequence length L_subject=" << L_subject
                  << " (remainder " << (seqs.size() % L_subject) << "). "
                  << "Check that the filename encodes the correct length. Skipped.\n";
        return false;
    }

    std::vector<value_t> results(N_total);

    // ── Compute per-stream batch size ──────────────────────────────────────
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    size_t usable = (free_b > (size_t)GPU_HEADROOM)
                  ? free_b - GPU_HEADROOM : free_b / 2;

    // Each stream needs, per sequence:
    //   • d_subj : L_subject floats
    //   • d_dist : 1 float
    //   • scratch pad   : up to max(padded_stride, TILE_SIZE) floats
    //   • scratch bnd×2 : 2 × L_query floats
    //   • pinned host subj+dist : (L_subject + 1) floats
    const int    K_full_     = (int)((L_subject + 32) / 32);
    const size_t pad_floats_ = (K_full_ <= 120)
                             ? (size_t)K_full_ * 32
                             : (size_t)SCRATCH_TILE_SIZE;
    size_t bytes_per_seq = ( (size_t)L_subject      // d_subj
                           + 1                        // d_dist
                           + pad_floats_              // scratch pad
                           + 2 * (size_t)L_query      // scratch boundaries
                           + (size_t)L_subject + 1    // pinned host staging
                           ) * sizeof(value_t);
    index_t batch_n = std::max((index_t)1,
                               (index_t)(usable / N_STREAMS / bytes_per_seq));
    batch_n = std::min(batch_n, N_total);

    // ── Allocate N_STREAMS × (pinned host + device) buffers ──────────────
    cudaStream_t streams[N_STREAMS];
    value_t* h_subj[N_STREAMS];   // pinned host subject
    value_t* h_dist[N_STREAMS];   // pinned host dist
    value_t* d_subj[N_STREAMS];   // device subject
    value_t* d_dist[N_STREAMS];   // device dist

    for (int s = 0; s < N_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMallocHost(&h_subj[s], batch_n * L_subject * sizeof(value_t)));
        CUDA_CHECK(cudaMallocHost(&h_dist[s], batch_n * sizeof(value_t)));
        CUDA_CHECK(cudaMalloc(&d_subj[s],     batch_n * L_subject * sizeof(value_t)));
        CUDA_CHECK(cudaMalloc(&d_dist[s],     batch_n * sizeof(value_t)));
    }

    // ── Per-stream scratch (allocated ONCE here, reused for every batch) ─────
    // This removes ALL cudaMalloc/cudaFree from the batch loop (item #1).
    // padded_stride upper bound: generic path = compute_K(L)*32 ≤ TILE_SIZE for
    // L≤2047; for larger L the tiled path caps the pad buffer at TILE_SIZE.
    const int K_full = (int)((L_subject + 32) / 32);
    const long long pad_stride_max =
        (K_full <= 120) ? (long long)K_full * 32 : SCRATCH_TILE_SIZE;
    DtwScratch<value_t> scr[N_STREAMS];
    for (int s = 0; s < N_STREAMS; s++)
        scr[s].ensure((long long)batch_n, (long long)L_query, pad_stride_max);

    // ── Timing convention (item #12) ─────────────────────────────────────────
    // The timer brackets exactly: host→pinned memcpy + H2D + kernel + D2H +
    // pinned→host memcpy, for ALL batches. It EXCLUDES: disk read_bin (done
    // above), disk write_bin (done below), and all cudaMalloc/scratch.ensure
    // (done above). main_v4.cu uses the identical window, so GCUPS is directly
    // comparable across the sequential and streaming pipelines.
    auto t_start = std::chrono::steady_clock::now();

    // ── Pipeline: schedule all batches across N_STREAMS ───────────────────
    // Batches 0..K-1 are assigned to streams 0..N_STREAMS-1 cyclically.
    // Stream s+1 is transferring while stream s is computing.

    const index_t n_batches = (N_total + batch_n - 1) / batch_n;

    // Phase 1: issue all async H2D + kernel launches
    for (index_t b = 0; b < n_batches; b++) {
        const int    s       = (int)(b % N_STREAMS);
        const index_t offset = b * batch_n;
        const index_t bn     = std::min(batch_n, N_total - offset);

        // Copy to pinned host (CPU)
        memcpy(h_subj[s],
               seqs.data() + offset * L_subject,
               bn * L_subject * sizeof(value_t));

        // Async H2D
        CUDA_CHECK(cudaMemcpyAsync(d_subj[s], h_subj[s],
            bn * L_subject * sizeof(value_t),
            cudaMemcpyHostToDevice, streams[s]));

        // Kernel (async on stream s) — uses pre-allocated scratch, no malloc
        dist_any<index_t,value_t>(
            scr[s],
            d_subj[s], d_dist[s],
            bn, L_query, L_subject,
            streams[s]);

        // Async D2H
        CUDA_CHECK(cudaMemcpyAsync(h_dist[s], d_dist[s],
            bn * sizeof(value_t),
            cudaMemcpyDeviceToHost, streams[s]));
    }

    // Phase 2: sync and collect results
    for (index_t b = 0; b < n_batches; b++) {
        const int    s      = (int)(b % N_STREAMS);
        const index_t offset = b * batch_n;
        const index_t bn    = std::min(batch_n, N_total - offset);

        // Wait only for this stream's batch
        // (already overlapped with other streams)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        memcpy(results.data() + offset, h_dist[s],
               bn * sizeof(value_t));
    }

    auto t_end = std::chrono::steady_clock::now();
    double ms  = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    double gcpus = (double)L_query * L_subject * N_total / (ms*1e-3) / 1e9;

    // ── Cleanup ────────────────────────────────────────────────────────────
    for (int s = 0; s < N_STREAMS; s++) {
        scr[s].free();
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
        CUDA_CHECK(cudaFreeHost(h_subj[s]));
        CUDA_CHECK(cudaFreeHost(h_dist[s]));
        CUDA_CHECK(cudaFree(d_subj[s]));
        CUDA_CHECK(cudaFree(d_dist[s]));
    }

    write_bin(res_path, results);

    if (verbose) {
        int K = (int)(L_subject + 32) / 32;
        std::cout << "  Lq="   << std::setw(4) << L_query
                  << " Ls="   << std::setw(5) << L_subject
                  << " K="    << std::setw(4) << K
                  << " N="    << std::setw(7) << N_total
                  << " str="  << N_STREAMS
                  << " | "    << std::fixed << std::setprecision(1)
                  << ms       << " ms"
                  << " | "    << gcpus << " GCUPS"
                  << " → "    << fs::path(res_path).filename().string()
                  << "\n";
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <target_dir> <sequences_dir> <result_dir>\n";
        return 1;
    }
    const std::string tgt_dir = argv[1];
    const std::string seq_dir = argv[2];
    const std::string res_dir = argv[3];

    // GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " cuDTW-Ada | Phase 5 — Streaming Pipeline\n";
    std::cout << " GPU:     " << prop.name
              << "  sm_" << prop.major << prop.minor << "\n";
    std::cout << " VRAM:    " << prop.totalGlobalMem/(1UL<<30) << " GB\n";
    std::cout << " L2:      " << prop.l2CacheSize/(1<<20) << " MB\n";
    std::cout << " Streams: " << N_STREAMS << "\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // Find target
    index_t L_query = 0;
    std::vector<value_t> query;
    for (auto& p : fs::directory_iterator(tgt_dir)) {
        if (p.path().extension() != ".bin") continue;
        index_t L = name_to_len(p.path().filename().string());
        if (!L) continue;
        auto v = read_bin(p.path().string());
        if (v.size() != L) continue;
        L_query = L; query = std::move(v); break;
    }
    if (!L_query) { std::cerr << "No target found\n"; return 1; }
    if (L_query > MAX_QUERY) {
        std::cerr << "[ERROR] L_query=" << L_query
                  << " exceeds CUDTW_MAX_QUERY=" << MAX_QUERY
                  << ". Rebuild with a larger -DCUDTW_MAX_QUERY (≤16384) "
                     "or move the query out of constant memory.\n";
        return 1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, query.data(),
                                  sizeof(value_t)*L_query));
    std::cout << " Target: L_query = " << L_query << " floats\n\n";

    // Enumerate sequences
    std::vector<std::pair<index_t,std::string>> files;
    for (auto& p : fs::directory_iterator(seq_dir)) {
        if (p.path().extension() != ".bin") continue;
        index_t L = name_to_len(p.path().filename().string());
        if (L) files.emplace_back(L, p.path().string());
    }
    std::sort(files.begin(), files.end());
    std::cout << " Found " << files.size() << " files\n\n";

    fs::create_directories(res_dir);

    std::cout << "═══ Processing ═════════════════════════════════════════════\n";

    auto wall0 = std::chrono::steady_clock::now();
    size_t total_seqs = 0; double total_cells = 0;

    for (auto& [Ls, sp] : files) {
        size_t fsz = fs::file_size(sp);
        index_t N  = fsz / (Ls * sizeof(value_t));
        std::string rp = (fs::path(res_dir)/fs::path(sp).filename()).string();
        if (process_file_streamed(sp, rp, L_query, Ls, true)) {
            total_seqs  += N;
            total_cells += (double)L_query * Ls * N;
        }
    }

    auto wall1 = std::chrono::steady_clock::now();
    double wms = std::chrono::duration<double,std::milli>(wall1-wall0).count();

    std::cout << "\n═══ Summary ════════════════════════════════════════════════\n";
    std::cout << " Total sequences: " << total_seqs << "\n";
    std::cout << " Total compute:   "
              << std::fixed << std::setprecision(3)
              << total_cells/1e12 << " TCPU\n";
    std::cout << " Wall time:       " << wms/1000.0 << " s\n";
    if (wms > 0) {
        std::cout << " Avg throughput:  "
                  << total_cells/(wms*1e-3)/1e9 << " GCUPS\n";
        std::cout << " Seq/s:           "
                  << (double)total_seqs/(wms/1000.0) << "\n";
    }
    std::cout << " Results → " << res_dir << "\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    return 0;
}
