// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 3 | src/include/cudtw_tiled.hpp
//
// Tiled Warp-Shuffle DTW — handles L_subject > 3839 correctly and fast.
//
// Algorithm:
//   Split L_subject into tiles of 2047 (K=64, max safe register count).
//   Each tile runs the same warp-shuffle kernel as Phase 2.
//   A boundary column of L_query floats is passed between tiles.
//
// Performance (vs Anti-Diagonal Phase 3):
//   Measured ~4150 GCUPS (RTX 4090, K=64, kernel-only; see RESULTS.md).
//   Register-resident design; recurrence-latency-bound, not occupancy-bound.
//   → ~18× faster!
//
// Usage:
//   TiledDTW::dist_tiled(d_subj, d_dist, N, L_query, L_subject, stream);
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CUDTW_TILED_HPP
#define CUDTW_TILED_HPP

#include "hpc_helpers.hpp"
#include "cudtw_scratch.hpp"
#include "kernels/SHFL_FULLDTW_TILED.cuh"
#include "kernels/SHFL_FULLDTW_GENERIC.cuh"   // Phase 2 kernels for last tile
#include "cudtw_dispatcher_v2.hpp"             // pad + dispatch helper

namespace TiledDTW {

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int TILE_K    = 64;           // K for intermediate tiles
static constexpr int TILE_SIZE = TILE_K * 32;  // = 2048 padded stride
static constexpr int TILE_LEN  = TILE_SIZE - 1; // = 2047 actual elements per tile

// Minimum tile length that keeps K within [4, 120].  K = (L+32)/32 ≥ 4 ⇒ L ≥ 96.
static constexpr int MIN_TILE_LEN = 96;

// ── K calculator ──────────────────────────────────────────────────────────────
inline int compute_K(int L) { return (L + 32) / 32; }

// ── Tile schedule ──────────────────────────────────────────────────────────────
// Splits [0, Ls) into tiles.  INTERMEDIATE tiles (all but the last) MUST have a
// "natural" width of exactly 32*K-1 elements, because the warp-shuffle kernel
// writes its seam from a fixed register lane that only equals the tile's last
// real column when tile_len == 32*K-1.  The LAST tile may have any length ≥
// MIN_TILE_LEN (its kernel reads a seam but writes no seam, so its width is
// unconstrained beyond K ∈ [4,120]).
//
// Normal case: (n-1) tiles of TILE_LEN (=2047=32*64-1) + one last tile.
// If the natural last tile would be < MIN_TILE_LEN, the final ~2 tiles are
// rebalanced: the penultimate (intermediate) tile shrinks to the largest valid
// 32*K-1 width that leaves the last tile ≥ MIN_TILE_LEN.
struct TileSpan { int start; int len; };

// Largest intermediate width (of form 32*K-1, K∈[4,64]) ≤ R-MIN_TILE_LEN.
inline int largest_inter_width(int R) {
    // want 32*K-1 ≤ R - MIN_TILE_LEN  ⇒  K ≤ (R-MIN_TILE_LEN+1)/32
    int K = (R - MIN_TILE_LEN + 1) / 32;
    if (K > 64) K = 64;
    if (K < 4)  K = 4;          // floor; caller guarantees feasibility
    return 32 * K - 1;
}

inline int build_tile_schedule(int Ls, TileSpan* out, int max_tiles) {
    int n = (Ls + TILE_LEN - 1) / TILE_LEN;
    if (n <= 1) { out[0] = {0, Ls}; return 1; }

    int last = Ls - (n - 1) * TILE_LEN;
    int cnt = 0, s = 0;
    if (last < MIN_TILE_LEN) {
        // Rebalance final 1.x tiles. R spans the last two natural tiles.
        int R = Ls - (n - 2) * TILE_LEN;          // R ∈ (TILE_LEN, 2*TILE_LEN]
        int a = largest_inter_width(R);           // valid 32*K-1 intermediate
        int b = R - a;                            // last tile, ≥ MIN_TILE_LEN
        for (int t = 0; t < n - 2 && cnt < max_tiles; ++t) { out[cnt++] = {s, TILE_LEN}; s += TILE_LEN; }
        if (cnt < max_tiles) { out[cnt++] = {s, a}; s += a; }
        if (cnt < max_tiles)   out[cnt++] = {s, b};
    } else {
        for (int t = 0; t < n - 1 && cnt < max_tiles; ++t) { out[cnt++] = {s, TILE_LEN}; s += TILE_LEN; }
        if (cnt < max_tiles) out[cnt++] = {s, last};
    }
    return cnt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pad subject tile into pre-allocated GPU buffer
// src: [N × L_subject_full] — full packed subject
// dst: [N × padded_stride]  — padded tile
// tile_start: column offset in src
// tile_len:   actual columns to copy (rest padded with last value)
// ─────────────────────────────────────────────────────────────────────────────
template<typename value_t>
__global__ void pad_tile_kernel(
    const value_t* __restrict__ src,
    value_t*       __restrict__ dst,
    int N, int L_full, int tile_start, int tile_len, int padded_stride)
{
    const int seq = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (seq >= N || col >= padded_stride) return;
    const value_t* s = src + (long long)seq * L_full + tile_start;
    dst[(long long)seq * padded_stride + col] =
        (col < tile_len) ? s[col] : s[tile_len - 1];
}

template<typename value_t>
void pad_tile_gpu(const value_t* d_src, value_t* d_dst,
                  int N, int L_full, int tile_start, int tile_len,
                  int padded_stride, cudaStream_t stream)
{
    const int BLK = 128;
    dim3 grid(N, (padded_stride + BLK - 1) / BLK);
    pad_tile_kernel<value_t><<<grid, BLK, 0, stream>>>(
        d_src, d_dst, N, L_full, tile_start, tile_len, padded_stride);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main tiled dispatcher
// Handles L_subject > 3839 by splitting into 2047-element tiles.
// ─────────────────────────────────────────────────────────────────────────────
template<typename index_t, typename value_t>
void dist_tiled(
    const value_t* d_subject_packed,   // [N × L_subject] — NOT padded
    value_t*       d_dist,             // [N] output
    index_t        N,
    index_t        L_query,
    index_t        L_subject,
    cudaStream_t   stream = 0)
{
    const int Lq   = (int)L_query;
    const int Ls   = (int)L_subject;
    TileSpan sched[256];
    const int n_tiles = build_tile_schedule(Ls, sched, 256);

    // Allocate: padded tile buffer + two boundary columns (ping-pong)
    value_t* d_tile  = nullptr;
    value_t* d_bnd_a = nullptr;   // boundary after tile i
    value_t* d_bnd_b = nullptr;   // boundary after tile i-1

    CUDA_CHECK(cudaMalloc(&d_tile,  sizeof(value_t) * (long long)N * TILE_SIZE));
    CUDA_CHECK(cudaMalloc(&d_bnd_a, sizeof(value_t) * (long long)N * Lq));
    CUDA_CHECK(cudaMalloc(&d_bnd_b, sizeof(value_t) * (long long)N * Lq));

    const dim3 grid_warp(N), block_warp(32);

    for (int t = 0; t < n_tiles; t++) {
        const int tile_start = sched[t].start;
        const int tile_len   = sched[t].len;
        const bool first     = (t == 0);
        const bool last      = (t == n_tiles - 1);
        const int  K_t       = compute_K(tile_len);
        const int  ps_t      = K_t * 32;

        const value_t* left_bnd  = first ? nullptr : d_bnd_a;
        value_t*       right_bnd = last  ? nullptr : d_bnd_b;

        pad_tile_gpu<value_t>(d_subject_packed, d_tile,
                              (int)N, Ls, tile_start, tile_len, ps_t, stream);

        if (!last) {
            // Intermediate tile: read left bnd (if any), write right bnd.
            dispatch_tiled_inter<index_t,value_t>(
                d_tile, d_dist, N, L_query, (index_t)tile_len, (index_t)ps_t,
                left_bnd, right_bnd, K_t, stream);
        } else {
            // Last tile: read left bnd, output Dist.
            dispatch_tiled_last<index_t,value_t>(
                d_tile, d_dist, N, L_query, (index_t)tile_len, (index_t)ps_t,
                left_bnd, K_t, stream);
        }

        // Swap boundary ping-pong buffers
        value_t* tmp = d_bnd_a; d_bnd_a = d_bnd_b; d_bnd_b = tmp;
    }

    CUDA_CHECK(cudaFree(d_tile));
    CUDA_CHECK(cudaFree(d_bnd_a));
    CUDA_CHECK(cudaFree(d_bnd_b));
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch-based tiled dispatcher (NO cudaMalloc/cudaFree in hot path).
// Buffers come from a caller-owned DtwScratch (allocated once per stream).
// ─────────────────────────────────────────────────────────────────────────────
template<typename index_t, typename value_t>
void dist_tiled(
    UnifiedDTW::DtwScratch<value_t>& scr,
    const value_t* d_subject_packed,
    value_t*       d_dist,
    index_t        N,
    index_t        L_query,
    index_t        L_subject,
    cudaStream_t   stream)
{
    const int Lq   = (int)L_query; (void)Lq;
    const int Ls   = (int)L_subject;
    TileSpan sched[256];
    const int n_tiles = build_tile_schedule(Ls, sched, 256);

    value_t* d_tile  = scr.d_pad;
    value_t* d_bnd_a = scr.d_bnd_a;
    value_t* d_bnd_b = scr.d_bnd_b;

    // The first tile passes left_bnd=nullptr (no read), and every intermediate
    // tile writes all Lq seam rows before the next tile reads them, so the
    // boundary buffers need no pre-zeroing.

    const dim3 grid_warp(N), block_warp(32);

    for (int t = 0; t < n_tiles; t++) {
        const int tile_start = sched[t].start;
        const int tile_len   = sched[t].len;
        const bool first     = (t == 0);
        const bool last      = (t == n_tiles - 1);
        const int  K_t       = compute_K(tile_len);
        const int  ps_t      = K_t * 32;

        const value_t* left_bnd  = first ? nullptr : d_bnd_a;
        value_t*       right_bnd = last  ? nullptr : d_bnd_b;

        pad_tile_gpu<value_t>(d_subject_packed, d_tile,
                              (int)N, Ls, tile_start, tile_len, ps_t, stream);

        if (!last) {
            dispatch_tiled_inter<index_t,value_t>(
                d_tile, d_dist, N, L_query, (index_t)tile_len, (index_t)ps_t,
                left_bnd, right_bnd, K_t, stream);
        } else {
            dispatch_tiled_last<index_t,value_t>(
                d_tile, d_dist, N, L_query, (index_t)tile_len, (index_t)ps_t,
                left_bnd, K_t, stream);
        }
        value_t* tmp = d_bnd_a; d_bnd_a = d_bnd_b; d_bnd_b = tmp;
    }
}

// Synchronous wrapper
template<typename index_t, typename value_t>
void dist_tiled_sync(
    const value_t* d_subject_packed, value_t* d_dist,
    index_t N, index_t L_query, index_t L_subject)
{
    dist_tiled<index_t,value_t>(d_subject_packed, d_dist, N, L_query, L_subject);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUERR
}

}  // namespace TiledDTW

#endif  // CUDTW_TILED_HPP
