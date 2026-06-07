// ─────────────────────────────────────────────────────────────────────────────
// cuDTW-Ada | src/include/cudtw_constants.hpp
//
// Single source of truth for the query-capacity constant. Previously each TU
// defined its own size: cudtw_api.cu used 8192 while main_v5.cu used 16384.
// Because every .cu declares its OWN __constant__ cQuery[], a single linker
// symbol can't be shared — but the SIZE and the bounds-check threshold must be
// identical everywhere, which is what this header guarantees.
//
// Each .cu must declare its constant array as:
//     __constant__ value_t cQuery[CUDTW_MAX_QUERY];
// and validate L_query <= CUDTW_MAX_QUERY before cudaMemcpyToSymbol().
// ─────────────────────────────────────────────────────────────────────────────
#ifndef CUDTW_CONSTANTS_HPP
#define CUDTW_CONSTANTS_HPP

// Max query length (in elements). 8192 floats = 32 KB of constant memory,
// comfortably within the 64 KB __constant__ budget on sm_89 while leaving room
// for other constant-bank usage. Realistic queries (e.g. 395) are far smaller.
// Do not raise toward 16384 (64 KB) without verifying constant-bank headroom,
// and never beyond it without moving the query out of constant memory.
#ifndef CUDTW_MAX_QUERY
#define CUDTW_MAX_QUERY 8192
#endif

// The kernels read cQuery at index up to (L_query + 62) because of the warp's
// anti-diagonal lane offset (idx = k + thid - 1, with k ≤ L_query+tgt_t+1 and
// thid ≤ 31, tgt_t ≤ 31). The backing array must therefore have margin past
// the usable length, or those lanes read out of bounds (a latent bug in the
// original code). Always declare the constant array with CUDTW_QUERY_STORAGE.
#define CUDTW_QUERY_STORAGE (CUDTW_MAX_QUERY + 64)

#endif  // CUDTW_CONSTANTS_HPP
