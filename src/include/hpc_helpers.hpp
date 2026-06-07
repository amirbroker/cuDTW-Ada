#ifndef HPC_HELPERS_HPP
#define HPC_HELPERS_HPP

// ─── cuDTW-Ada | Phase 1 | hpc_helpers.hpp ───────────────────────────────────
// Changes vs original:
//   + Added CUDA_CHECK macro (explicit per-call error checking)
//   + CUERR unchanged (backward compatible)
//   + SDIV unchanged

#include <iostream>
#include <cstdint>
#include <cstdio>

#ifndef __CUDACC__
    #include <chrono>
#endif

// ─── CPU Timer ───────────────────────────────────────────────────────────────
#ifndef __CUDACC__
    #define TIMERSTART(label)                                                   \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label;  \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                   \
        cudaSetDevice(0);                                                       \
        cudaEvent_t start##label, stop##label;                                  \
        float time##label;                                                      \
        cudaEventCreate(&start##label);                                         \
        cudaEventCreate(&stop##label);                                          \
        cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                    \
        b##label = std::chrono::system_clock::now();                            \
        std::chrono::duration<double> delta##label = b##label - a##label;       \
        std::cout << "# elapsed time (" << #label << "): "                      \
                  << delta##label.count() << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                    \
        cudaSetDevice(0);                                                       \
        cudaEventRecord(stop##label, 0);                                        \
        cudaEventSynchronize(stop##label);                                      \
        cudaEventElapsedTime(&time##label, start##label, stop##label);          \
        std::cout << "# elapsed time (" << #label << "): "                      \
                  << time##label / 1000.0f << "s" << std::endl;
#endif

// ─── CUDA GPU Timer (unchanged from original) ────────────────────────────────
#define TIMERSTART_CUDA(label)                                                  \
        cudaSetDevice(0);                                                       \
        cudaEvent_t start##label, stop##label;                                  \
        float time##label;                                                      \
        cudaEventCreate(&start##label);                                         \
        cudaEventCreate(&stop##label);                                          \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP_CUDA(label)                                                   \
        cudaSetDevice(0);                                                       \
        cudaEventRecord(stop##label, 0);                                        \
        cudaEventSynchronize(stop##label);                                      \
        cudaEventElapsedTime(&time##label, start##label, stop##label);          \
        std::cout << "TIMING: " << time##label << " ms" << std::endl;

// ─── CUERR: checks last CUDA error (original, backward compatible) ────────────
#ifdef __CUDACC__
    #define CUERR {                                                             \
        cudaError_t err;                                                        \
        if ((err = cudaGetLastError()) != cudaSuccess) {                        \
            std::cout << "CUDA error: " << cudaGetErrorString(err)              \
                      << " : " << __FILE__ << ", line " << __LINE__             \
                      << std::endl;                                             \
            exit(1);                                                            \
        }                                                                       \
    }
#endif

// ─── CUDA_CHECK: explicit per-call error checking (NEW) ──────────────────────
// Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr,                                                     \
                    "[CUDA ERROR] %s:%d — %s\n  Call: %s\n",                   \
                    __FILE__, __LINE__,                                         \
                    cudaGetErrorString(_err), #call);                           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Safe integer ceiling division ───────────────────────────────────────────
#define SDIV(x, y) (((x) + (y) - 1) / (y))

#endif // HPC_HELPERS_HPP
