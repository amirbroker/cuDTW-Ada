// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 4 | src/main_v4.cu
//
// Full pipeline: target/ × sequences/ → result/
//
// Binary format (all raw float32, little-endian, no header):
//   target/L.bin       → L floats  (one query sequence of length L)
//   sequences/L.bin    → N × L floats  (N sequences of length L)
//   result/L.bin       → N floats  (DTW distances, same order as input)
//
// Usage:
//   ./dtw_main <target_dir> <sequences_dir> <result_dir>
//
//   Example:
//   ./dtw_main target/ sequences/ result/
//
// Features:
//   • Auto-detects query length from target/*.bin filename
//   • Processes every sequences/L.bin file (any subject length)
//   • Batches sequences to fit GPU memory (default: 8 GB per batch)
//   • Phase 2 (L≤3839) + Phase 3 tiled (L>3839) dispatched automatically
//   • Reports throughput (GCUPS, sequences/sec, ETA)
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <chrono>

#include "include/hpc_helpers.hpp"

namespace fs = std::filesystem;

typedef float    value_t;
typedef uint64_t index_t;

// cQuery in constant memory (up to 64k floats = 256KB)
constexpr index_t MAX_QUERY = (1UL << 16) / sizeof(value_t);
__constant__ value_t cQuery[MAX_QUERY];

#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"
#include "include/kernels/SHFL_FULLDTW_TILED.cuh"
#include "include/cudtw_dispatcher_v2.hpp"
#include "include/cudtw_tiled.hpp"
#include "include/cudtw_dispatcher_v3.hpp"

using namespace UnifiedDTW;

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

// Read raw float32 file into a vector
std::vector<value_t> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; return {}; }
    size_t bytes = f.tellg();
    f.seekg(0);
    std::vector<value_t> v(bytes / sizeof(value_t));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// Write raw float32 vector to file
void write_bin(const std::string& path, const std::vector<value_t>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot write: " << path << "\n"; return; }
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(value_t));
}

// Extract sequence length from filename "L.bin" → L
index_t filename_to_length(const std::string& name) {
    size_t dot = name.rfind('.');
    if (dot == std::string::npos) return 0;
    try { return std::stoull(name.substr(0, dot)); }
    catch (...) { return 0; }
}

// GPU memory available for subject data (leave 2GB headroom)
size_t gpu_batch_bytes() {
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return (free_bytes > 2UL<<30) ? free_bytes - (2UL<<30) : free_bytes / 2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Process one sequences file
// ─────────────────────────────────────────────────────────────────────────────
bool process_file(
    const std::string& seq_path,
    const std::string& res_path,
    index_t L_query,
    index_t L_subject,
    bool verbose)
{
    // ── Load sequences ─────────────────────────────────────────────────────
    auto seqs = read_bin(seq_path);
    if (seqs.empty()) return false;

    const index_t N_total = seqs.size() / L_subject;
    if (N_total == 0 || seqs.size() % L_subject != 0) {
        std::cerr << "  [WARN] " << seq_path
                  << " size " << seqs.size()
                  << " not divisible by L=" << L_subject << "\n";
        return false;
    }

    std::vector<value_t> results(N_total);

    // ── Batch by available GPU memory ─────────────────────────────────────
    const size_t avail      = gpu_batch_bytes();
    const size_t bytes_per  = L_subject * sizeof(value_t);
    const index_t batch_max = std::max((size_t)1, avail / bytes_per);

    value_t* d_subj = nullptr;
    value_t* d_dist = nullptr;

    // Allocate for the maximum needed batch size
    const index_t alloc_N = std::min(batch_max, N_total);
    CUDA_CHECK(cudaMalloc(&d_subj, sizeof(value_t) * alloc_N * L_subject));
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(value_t) * alloc_N));

    // ── Timing convention (item #12) ─────────────────────────────────────────
    // Brackets exactly: H2D + kernel + D2H for ALL batches. EXCLUDES disk
    // read_bin (above), write_bin (below), and cudaMalloc (above). main_v5.cu
    // uses the identical window so GCUPS is directly comparable.
    auto t_start = std::chrono::steady_clock::now();
        const index_t batch_n = std::min(batch_max, N_total - offset);

        CUDA_CHECK(cudaMemcpy(d_subj,
            seqs.data() + offset * L_subject,
            sizeof(value_t) * batch_n * L_subject,
            cudaMemcpyHostToDevice));

        dist_any_sync<index_t, value_t>(
            d_subj, d_dist, batch_n, L_query, L_subject);

        CUDA_CHECK(cudaMemcpy(
            results.data() + offset, d_dist,
            sizeof(value_t) * batch_n,
            cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_subj));
    CUDA_CHECK(cudaFree(d_dist));

    auto t_end = std::chrono::steady_clock::now();
    double ms  = std::chrono::duration<double,std::milli>(t_end - t_start).count();
    double gcpus = (double)L_query * L_subject * N_total / (ms * 1e-3) / 1e9;

    // ── Save results ───────────────────────────────────────────────────────
    write_bin(res_path, results);

    if (verbose) {
        int K = (int)(L_subject + 32) / 32;
        std::cout << "  Lq=" << std::setw(4) << L_query
                  << " Ls=" << std::setw(5) << L_subject
                  << " K=" << std::setw(4) << K
                  << " N=" << std::setw(7) << N_total
                  << " | " << std::fixed << std::setprecision(1)
                  << ms << " ms"
                  << " | " << gcpus << " GCUPS"
                  << " → " << fs::path(res_path).filename().string()
                  << "\n";
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <target_dir> <sequences_dir> <result_dir>\n"
                  << "Example: ./dtw_main target/ sequences/ result/\n";
        return 1;
    }

    const std::string target_dir = argv[1];
    const std::string seqs_dir   = argv[2];
    const std::string result_dir = argv[3];

    // ── GPU info ──────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << " cuDTW-Ada | Phase 4 Pipeline\n";
    std::cout << " GPU: " << prop.name << "  sm_"
              << prop.major << prop.minor << "\n";
    std::cout << " VRAM: " << prop.totalGlobalMem / (1UL<<30) << " GB\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    // ── Find target sequence ──────────────────────────────────────────────
    index_t L_query = 0;
    std::vector<value_t> query;

    for (auto& p : fs::directory_iterator(target_dir)) {
        if (p.path().extension() != ".bin") continue;
        index_t L = filename_to_length(p.path().filename().string());
        if (L == 0) continue;
        auto v = read_bin(p.path().string());
        if (v.size() != L) {
            std::cerr << "[WARN] " << p.path() << " size mismatch\n";
            continue;
        }
        L_query = L;
        query   = std::move(v);
        break;
    }

    if (L_query == 0 || query.empty()) {
        std::cerr << "[ERROR] No valid target .bin found in " << target_dir << "\n";
        return 1;
    }

    // Upload query to GPU constant memory
    assert(L_query <= MAX_QUERY);
    CUDA_CHECK(cudaMemcpyToSymbol(cQuery, query.data(),
                                  sizeof(value_t) * L_query));

    std::cout << " Target: L_query = " << L_query << " floats\n\n";

    // ── Create result directory ───────────────────────────────────────────
    fs::create_directories(result_dir);

    // ── Enumerate and sort sequences files ───────────────────────────────
    std::vector<std::pair<index_t, std::string>> files;
    for (auto& p : fs::directory_iterator(seqs_dir)) {
        if (p.path().extension() != ".bin") continue;
        index_t L = filename_to_length(p.path().filename().string());
        if (L == 0) continue;
        files.emplace_back(L, p.path().string());
    }
    std::sort(files.begin(), files.end());  // sort by L_subject

    std::cout << " Found " << files.size() << " sequence files in "
              << seqs_dir << "\n\n";

    // ── Process each file ─────────────────────────────────────────────────
    std::cout << "═══ Processing ═════════════════════════════════════════════\n";

    auto wall_start = std::chrono::steady_clock::now();
    size_t total_seqs = 0, processed_files = 0;
    double total_cells = 0;

    for (auto& [L_sub, seq_path] : files) {
        const std::string fname  = fs::path(seq_path).filename().string();
        const std::string res_path = (fs::path(result_dir) / fname).string();

        // Count sequences in this file
        size_t file_bytes = fs::file_size(seq_path);
        index_t N_seqs    = file_bytes / (L_sub * sizeof(value_t));

        bool ok = process_file(seq_path, res_path, L_query, L_sub, true);
        if (ok) {
            total_seqs    += N_seqs;
            total_cells   += (double)L_query * L_sub * N_seqs;
            processed_files++;
        }
    }

    auto wall_end = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double,std::milli>(
                         wall_end - wall_start).count();

    // ── Summary ───────────────────────────────────────────────────────────
    std::cout << "\n═══ Summary ════════════════════════════════════════════════\n";
    std::cout << " Files processed:  " << processed_files
              << " / " << files.size() << "\n";
    std::cout << " Total sequences:  " << total_seqs << "\n";
    std::cout << " Total compute:    "
              << std::fixed << std::setprecision(2)
              << total_cells / 1e12 << " TCPU\n";
    std::cout << " Wall time:        "
              << wall_ms / 1000.0 << " s\n";
    if (wall_ms > 0) {
        double avg_gcpus = total_cells / (wall_ms * 1e-3) / 1e9;
        std::cout << " Avg throughput:   " << avg_gcpus << " GCUPS\n";
        std::cout << " Seq/s:            "
                  << (double)total_seqs / (wall_ms / 1000.0) << "\n";
    }
    std::cout << " Results saved to: " << result_dir << "\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";

    return 0;
}
