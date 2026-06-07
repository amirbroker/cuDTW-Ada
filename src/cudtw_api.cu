// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 8 | src/cudtw_api.cu
//
// C-compatible API for Python ctypes binding.
// Exports:
//   dtw_init()              — initialize CUDA device
//   dtw_set_query()         — upload query to GPU constant memory
//   dtw_compute()           — compute DTW for a batch of subjects
//   dtw_compute_file()      — process one binary file
//   dtw_process_folder()    — process entire sequences/ folder
//   dtw_info()              — GPU info string
//
// Build as shared library:
//   make libcudtw.so
//
// Python usage:
//   import cudtw
//   results = cudtw.compute(query, subjects)
// ─────────────────────────────────────────────────────────────────────────────

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "include/hpc_helpers.hpp"
#include "include/cudtw_constants.hpp"

namespace fs = std::filesystem;

typedef float    value_t;
typedef uint64_t index_t;

constexpr index_t MAX_Q = CUDTW_MAX_QUERY;
__constant__ value_t cQuery[CUDTW_QUERY_STORAGE];

#include "include/kernels/SHFL_FULLDTW_GENERIC.cuh"
#include "include/kernels/SHFL_FULLDTW_OPT.cuh"
#include "include/kernels/SHFL_FULLDTW_TILED.cuh"
#include "include/cudtw_dispatcher_v2.hpp"
#include "include/cudtw_tiled.hpp"
#include "include/cudtw_dispatcher_v3.hpp"

using namespace UnifiedDTW;

// ── Internal state ────────────────────────────────────────────────────────────
static int  g_device      = -1;
static int  g_L_query     = 0;
static char g_error[1024] = {0};

static void set_error(const char* msg) {
    strncpy(g_error, msg, sizeof(g_error)-1);
}

// ─────────────────────────────────────────────────────────────────────────────
// C API — exported with extern "C" for ctypes
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

// Initialize CUDA (call once at startup)
// Returns: 0 on success, -1 on error
int dtw_init(int device_id)
{
    if (cudaSetDevice(device_id) != cudaSuccess) {
        set_error("cudaSetDevice failed"); return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    g_device = device_id;
    snprintf(g_error, sizeof(g_error),
             "OK: %s sm_%d%d", prop.name, prop.major, prop.minor);
    return 0;
}

// Get last error/info string
const char* dtw_last_error() { return g_error; }

// GPU info string (name + VRAM)
void dtw_info(char* buf, int buf_len)
{
    if (g_device < 0) { strncpy(buf, "Not initialized", buf_len); return; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, g_device);
    snprintf(buf, buf_len, "%s | sm_%d%d | %llu GB VRAM",
             p.name, p.major, p.minor,
             (unsigned long long)(p.totalGlobalMem >> 30));
}

// Upload query sequence to GPU constant memory
// query: float array of length L_query
// Returns: 0 on success, -1 on error
int dtw_set_query(const float* query, int L_query)
{
    if (L_query <= 0 || L_query > (int)MAX_Q) {
        snprintf(g_error, sizeof(g_error),
                 "L_query=%d out of range [1, %d] (CUDTW_MAX_QUERY)",
                 L_query, (int)MAX_Q);
        return -1;
    }
    if (cudaMemcpyToSymbol(cQuery, query, sizeof(float)*L_query) != cudaSuccess) {
        set_error("cudaMemcpyToSymbol failed"); return -1;
    }
    g_L_query = L_query;
    return 0;
}

// Compute DTW distances: subjects vs current query
// subjects: float array [N × L_subject] row-major
// results:  float array [N] output (caller-allocated)
// Returns: 0 on success, -1 on error
int dtw_compute(
    const float* subjects, int N, int L_subject,
    float* results)
{
    if (g_L_query <= 0) { set_error("Call dtw_set_query first"); return -1; }
    if (N <= 0 || L_subject <= 0) { set_error("Invalid dimensions"); return -1; }
    if (L_subject < UnifiedDTW::MIN_SUBJECT_LEN) {
        snprintf(g_error, sizeof(g_error),
                 "L_subject=%d is below the minimum supported length of %d "
                 "(kernels exist only for K>=4). Pad short subjects to >= %d.",
                 L_subject, UnifiedDTW::MIN_SUBJECT_LEN, UnifiedDTW::MIN_SUBJECT_LEN);
        return -1;
    }

    float *d_subj=nullptr, *d_dist=nullptr;
    if (cudaMalloc(&d_subj, sizeof(float)*(long long)N*L_subject) != cudaSuccess ||
        cudaMalloc(&d_dist, sizeof(float)*N) != cudaSuccess) {
        set_error("cudaMalloc failed"); return -1;
    }

    cudaMemcpy(d_subj, subjects, sizeof(float)*(long long)N*L_subject,
               cudaMemcpyHostToDevice);

    dist_any_sync<index_t,value_t>(
        d_subj, d_dist,
        (index_t)N, (index_t)g_L_query, (index_t)L_subject);

    cudaMemcpy(results, d_dist, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cudaFree(d_subj); cudaFree(d_dist);
    return 0;
}

// Process one binary sequences file → write results to binary file
// Returns N (number of sequences) on success, -1 on error
long long dtw_compute_file(
    const char* sequences_path,
    int L_subject,
    const char* result_path)
{
    if (g_L_query <= 0) { set_error("Call dtw_set_query first"); return -1; }

    // Read sequences
    std::ifstream f(sequences_path, std::ios::binary | std::ios::ate);
    if (!f) { set_error("Cannot open sequences file"); return -1; }
    size_t nb = f.tellg(); f.seekg(0);
    int N = (int)(nb / (L_subject * sizeof(float)));
    if (N <= 0) { set_error("Empty or wrong-size file"); return -1; }

    std::vector<float> seqs(nb/sizeof(float));
    f.read(reinterpret_cast<char*>(seqs.data()), nb);
    f.close();

    std::vector<float> results(N);
    if (dtw_compute(seqs.data(), N, L_subject, results.data()) < 0) return -1;

    // Write results
    std::ofstream out(result_path, std::ios::binary);
    if (!out) { set_error("Cannot write result file"); return -1; }
    out.write(reinterpret_cast<char*>(results.data()), N*sizeof(float));
    return N;
}

// Process entire sequences folder
// sequences_dir: path like "sequences/"
// result_dir:    path like "result/"
// Returns: number of files processed, -1 on error
int dtw_process_folder(
    const char* sequences_dir,
    const char* result_dir,
    void (*progress_cb)(const char* filename, int N, float gcpus))  // optional callback
{
    if (g_L_query <= 0) { set_error("Call dtw_set_query first"); return -1; }

    try {
        fs::create_directories(result_dir);
        int count = 0;

        // Collect files
        std::vector<std::pair<int,std::string>> files;
        for (auto& p : fs::directory_iterator(sequences_dir)) {
            if (p.path().extension() != ".bin") continue;
            std::string stem = p.path().stem().string();
            try {
                int L = std::stoi(stem);
                files.emplace_back(L, p.path().string());
            } catch(...) {}
        }
        std::sort(files.begin(), files.end());

        for (auto& [L_sub, sp] : files) {
            std::string fname = fs::path(sp).filename().string();
            std::string rp = (fs::path(result_dir) / fname).string();

            size_t fsz = fs::file_size(sp);
            int N = (int)(fsz / (L_sub * sizeof(float)));

            auto t0 = std::chrono::steady_clock::now();
            long long ret = dtw_compute_file(sp.c_str(), L_sub, rp.c_str());
            auto t1 = std::chrono::steady_clock::now();

            if (ret > 0) {
                if (progress_cb) {
                    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
                    double gcpus = (double)g_L_query*L_sub*N/(ms*1e-3)/1e9;
                    progress_cb(fname.c_str(), N, (float)gcpus);
                }
                count++;
            }
        }
        return count;
    } catch (std::exception& e) {
        set_error(e.what()); return -1;
    }
}

}  // extern "C"
