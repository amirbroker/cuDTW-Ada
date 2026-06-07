// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 2 | src/include/cudtw_dispatcher_v2.hpp
//
// dist_v2(): asymmetric DTW dispatcher
//   • Supports ANY subject length ≤ 3839 (Phase 2)
//   • Supports L_query ≠ L_subject
//   • Pads subject in-place on GPU before kernel launch
//   • Falls back to error for L_subject > 3839 (Phase 3 will handle)
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CUDTW_DISPATCHER_V2_HPP
#define CUDTW_DISPATCHER_V2_HPP

#include "hpc_helpers.hpp"
#include "kernels/SHFL_FULLDTW_GENERIC.cuh"

#include <cmath>     // std::ceil
#include <cstring>   // memset
#include <stdexcept>

namespace GenericDTW {

// ── K calculator ──────────────────────────────────────────────────────────────
// Returns the minimum K such that K*32 - 1 ≥ L_subject
// i.e., K = ceil((L_subject + 1) / 32)
inline int compute_K(int L_subject) {
    return (L_subject + 32) / 32;   // = ceil((L+1)/32)
}

// Returns the padded stride (K*32) for a given L_subject
inline int padded_stride(int L_subject) {
    return compute_K(L_subject) * 32;
}

// ── GPU padding kernel ────────────────────────────────────────────────────────
// Pads each sequence from L_actual to stride elements by repeating last value.
// Input:  src[N * L_actual]  (N sequences, tightly packed)
// Output: dst[N * stride]    (N sequences, padded)
template<typename value_t>
__global__ void pad_sequences_kernel(
    const value_t* __restrict__ src,
    value_t* __restrict__ dst,
    int N, int L_actual, int stride)
{
    const int seq_id  = blockIdx.x;
    const int col     = blockIdx.y * blockDim.x + threadIdx.x;
    if (seq_id >= N || col >= stride) return;

    const value_t* s = src + (long long)seq_id * L_actual;
    value_t*       d = dst + (long long)seq_id * stride;

    // Copy real element or pad with last value
    d[col] = (col < L_actual) ? s[col] : s[L_actual - 1];
}

template<typename value_t>
void pad_sequences_gpu(
    const value_t* d_src,   // device: N × L_actual (packed)
    value_t*       d_dst,   // device: N × stride   (padded)
    int N, int L_actual, int stride,
    cudaStream_t stream = 0)
{
    // Grid: (N sequences) × (stride/128 column tiles)
    const int BLOCK = 128;
    dim3 grid(N, (stride + BLOCK - 1) / BLOCK);
    dim3 block(BLOCK);
    pad_sequences_kernel<value_t><<<grid, block, 0, stream>>>(
        d_src, d_dst, N, L_actual, stride);
}

// ── Main dispatcher ───────────────────────────────────────────────────────────
//
// Parameters:
//   d_subject  — device array [N × L_subject] packed (NOT padded)
//   d_dist     — device array [N]  output distances
//   N          — number of subject sequences
//   L_query    — actual query length (cQuery must be set before calling)
//   L_subject  — actual subject length (arbitrary, ≤ 3839 for Phase 2)
//   stream     — CUDA stream
//
// Returns true if handled, false if L_subject > 3839 (needs Phase 3 tiling)
template<typename index_t, typename value_t>
bool dist_v2(
    const value_t* d_subject_packed,  // [N × L_subject]
    value_t*       d_dist,            // [N]
    index_t        N,
    index_t        L_query,
    index_t        L_subject,
    cudaStream_t   stream,
    value_t*       d_pad_buffer,      // pre-allocated [N × padded_stride]
    index_t        pad_buf_stride)    // must equal padded_stride(L_subject)
{
    const int K = compute_K((int)L_subject);

    // Phase 2 covers up to K=120 (L≤3839)
    if (K > 120) return false;

    const int stride = K * 32;

    // 1. Pad subject sequences on GPU
    pad_sequences_gpu<value_t>(
        d_subject_packed, d_pad_buffer,
        (int)N, (int)L_subject, stride, stream);

    // 2. Dispatch to the right generic kernel
    dispatch_generic<index_t, value_t>(
        d_pad_buffer, d_dist,
        N, L_query, L_subject,
        (index_t)stride, K, stream);

    return true;
}

// ── Convenience wrapper: allocates pad buffer internally ──────────────────────
// Use this for single calls where you don't want to manage the buffer.
// For high-throughput batch processing, use dist_v2() with a pre-allocated buffer.
template<typename index_t, typename value_t>
void dist_v2_simple(
    const value_t* d_subject_packed,
    value_t*       d_dist,
    index_t        N,
    index_t        L_query,
    index_t        L_subject,
    cudaStream_t   stream = 0)
{
    const int K      = compute_K((int)L_subject);
    const int stride = K * 32;

    value_t* d_pad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pad, sizeof(value_t) * N * stride));

    bool ok = dist_v2<index_t, value_t>(
        d_subject_packed, d_dist,
        N, L_query, L_subject,
        stream, d_pad, (index_t)stride);

    CUDA_CHECK(cudaFree(d_pad));

    if (!ok) {
        // L_subject > 3839: needs tiling (Phase 3)
        // For now, report error — Phase 3 will handle this
        fprintf(stderr,
            "[dist_v2] L_subject=%llu > 3839: needs 1D tiling (Phase 3)\n",
            (unsigned long long)L_subject);
    }
}

}  // namespace GenericDTW

#endif  // CUDTW_DISPATCHER_V2_HPP
