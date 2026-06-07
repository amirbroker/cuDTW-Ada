// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | src/include/cudtw_scratch.hpp
//
// Pre-allocated per-stream scratch buffers, so the hot path performs NO
// cudaMalloc / cudaFree (each of which is an implicit full-device sync that
// destroys multi-stream overlap).
//
// Usage (per stream, allocated ONCE outside the batch loop):
//   DtwScratch<value_t> scr;
//   scr.ensure(max_N, L_query, L_subject_max);   // grow-only allocation
//   ...
//   dist_any(scr, d_subj, d_dist, N, L_query, L_subject, stream);  // no malloc
//   ...
//   scr.free();                                   // once, at shutdown
// ─────────────────────────────────────────────────────────────────────────────
#ifndef CUDTW_SCRATCH_HPP
#define CUDTW_SCRATCH_HPP

#include "hpc_helpers.hpp"
#include <algorithm>

namespace UnifiedDTW {

// Mirror of the tiled constants so we can size the pad buffer without including
// the tiled header here (avoids a circular include).
static constexpr int SCRATCH_TILE_K    = 64;
static constexpr int SCRATCH_TILE_SIZE = SCRATCH_TILE_K * 32;   // 2048

template<typename value_t>
struct DtwScratch {
    value_t* d_pad   = nullptr;   // padded subject buffer (generic OR tile)
    value_t* d_bnd_a = nullptr;   // boundary ping
    value_t* d_bnd_b = nullptr;   // boundary pong
    long long pad_floats = 0;     // capacity of d_pad,   in floats
    long long bnd_floats = 0;     // capacity of each bnd, in floats

    // Grow-only: (re)allocate buffers if current capacity is insufficient.
    // padded_stride_max is the largest K*32 stride that will be used; for the
    // generic path that is compute_K(L)*32, for the tiled path it is TILE_SIZE
    // (intermediate) or compute_K(tile_len)*32 (last) — TILE_SIZE bounds both
    // because the last tile's stride ≤ TILE_SIZE.
    void ensure(long long N, long long L_query, long long padded_stride_max) {
        const long long want_pad = N * std::max<long long>(padded_stride_max,
                                                           SCRATCH_TILE_SIZE);
        const long long want_bnd = N * L_query;
        if (want_pad > pad_floats) {
            if (d_pad) CUDA_CHECK(cudaFree(d_pad));
            CUDA_CHECK(cudaMalloc(&d_pad, sizeof(value_t) * want_pad));
            pad_floats = want_pad;
        }
        if (want_bnd > bnd_floats) {
            if (d_bnd_a) CUDA_CHECK(cudaFree(d_bnd_a));
            if (d_bnd_b) CUDA_CHECK(cudaFree(d_bnd_b));
            CUDA_CHECK(cudaMalloc(&d_bnd_a, sizeof(value_t) * want_bnd));
            CUDA_CHECK(cudaMalloc(&d_bnd_b, sizeof(value_t) * want_bnd));
            bnd_floats = want_bnd;
        }
    }

    void free() {
        if (d_pad)   { CUDA_CHECK(cudaFree(d_pad));   d_pad   = nullptr; }
        if (d_bnd_a) { CUDA_CHECK(cudaFree(d_bnd_a)); d_bnd_a = nullptr; }
        if (d_bnd_b) { CUDA_CHECK(cudaFree(d_bnd_b)); d_bnd_b = nullptr; }
        pad_floats = bnd_floats = 0;
    }
};

}  // namespace UnifiedDTW

#endif  // CUDTW_SCRATCH_HPP
