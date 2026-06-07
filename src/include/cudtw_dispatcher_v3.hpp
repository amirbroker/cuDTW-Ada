// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | Phase 3 | src/include/cudtw_dispatcher_v3.hpp  (FINAL)
//
// Unified dispatcher for ALL subject lengths:
//   L_subject ≤ 3839  → Phase 2 (generic warp-shuffle)
//   L_subject > 3839  → Phase 3 (TILED warp-shuffle)
//   Measured ~4150 GCUPS (RTX 4090, K=64, kernel-only). See RESULTS.md.
//
// Phase 3 uses the Tiled Warp-Shuffle approach (NOT anti-diagonal):
//   • Splits subject into 2047-element tiles
//   • Each tile uses the same warp-shuffle register kernel as Phase 2
//   • Passes L_query-float boundary column between tiles
//   • ~18× faster than anti-diagonal approach
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CUDTW_DISPATCHER_V3_HPP
#define CUDTW_DISPATCHER_V3_HPP

#include "hpc_helpers.hpp"
#include "cudtw_scratch.hpp"
#include "cudtw_dispatcher_v2.hpp"   // Phase 2 (L ≤ 3839)
#include "cudtw_tiled.hpp"           // Phase 3 (L > 3839, tiled warp)

namespace UnifiedDTW {

inline int compute_K(int L) { return (L + 32) / 32; }

// Smallest subject length for which a kernel exists. Kernels are generated for
// K = ceil((L+1)/32) in [2,120]; K>=2 requires L_subject >= 32. (K=1, i.e.
// L<32, is not generated — those tiny subjects are rare and the warp would be
// mostly idle.)
static constexpr int MIN_SUBJECT_LEN = 32;

// Returns true if L_subject is in the supported range. On failure prints a
// clear message (host side) so the caller can abort instead of computing
// garbage. Tiled path has no upper length limit (bounded only by VRAM).
inline bool subject_len_supported(long long L_subject, const char* where) {
    if (L_subject < MIN_SUBJECT_LEN) {
        fprintf(stderr,
            "[cuDTW-Ada] ERROR in %s: L_subject=%lld is below the minimum "
            "supported length of %d (kernels exist only for K=ceil((L+1)/32) "
            ">= 4). Pad short subjects to at least %d, or extend the kernel "
            "generators to cover K<4.\n",
            where, L_subject, MIN_SUBJECT_LEN, MIN_SUBJECT_LEN);
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch-based dist_any: NO cudaMalloc/cudaFree, NO implicit device sync.
// Use this in the streaming hot loop. Allocate one DtwScratch per stream once.
// ─────────────────────────────────────────────────────────────────────────────
template<typename index_t, typename value_t>
void dist_any(
    DtwScratch<value_t>& scr,
    const value_t* d_subject_packed,
    value_t*       d_dist,
    index_t        N,
    index_t        L_query,
    index_t        L_subject,
    cudaStream_t   stream)
{
    if (!subject_len_supported((long long)L_subject, "dist_any(scratch)")) return;
    const int K = compute_K((int)L_subject);
    if (K <= 120) {
        const int stride = K * 32;
        GenericDTW::dist_v2<index_t, value_t>(
            d_subject_packed, d_dist,
            N, L_query, L_subject,
            stream, scr.d_pad, (index_t)stride);
    } else {
        TiledDTW::dist_tiled<index_t, value_t>(
            scr, d_subject_packed, d_dist,
            N, L_query, L_subject, stream);
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// dist_any: unified entry point for any L_subject
// ─────────────────────────────────────────────────────────────────────────────
template<typename index_t, typename value_t>
void dist_any(
    const value_t* d_subject_packed,   // [N × L_subject] — NOT padded
    value_t*       d_dist,             // [N] output distances
    index_t        N,
    index_t        L_query,            // cQuery[] must be set before calling
    index_t        L_subject,
    cudaStream_t   stream = 0)
{
    if (!subject_len_supported((long long)L_subject, "dist_any")) return;
    const int K = compute_K((int)L_subject);

    if (K <= 120) {
        // ── Phase 2: single warp-shuffle kernel, no tiling ────────────────────
        const int stride = K * 32;
        value_t* d_pad = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pad, sizeof(value_t) * (long long)N * stride));

        GenericDTW::dist_v2<index_t, value_t>(
            d_subject_packed, d_dist,
            N, L_query, L_subject,
            stream, d_pad, (index_t)stride);

        CUDA_CHECK(cudaFree(d_pad));
    } else {
        // ── Phase 3: tiled warp-shuffle (L_subject > 3839) ───────────────────
        TiledDTW::dist_tiled<index_t, value_t>(
            d_subject_packed, d_dist,
            N, L_query, L_subject, stream);
    }
}

// Synchronous convenience wrapper
template<typename index_t, typename value_t>
void dist_any_sync(
    const value_t* d_subject_packed,
    value_t*       d_dist,
    index_t        N,
    index_t        L_query,
    index_t        L_subject)
{
    dist_any<index_t, value_t>(
        d_subject_packed, d_dist, N, L_query, L_subject, (cudaStream_t)0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUERR
}

}  // namespace UnifiedDTW

#endif  // CUDTW_DISPATCHER_V3_HPP
