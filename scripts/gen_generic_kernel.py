#!/usr/bin/env python3
"""
cuDTW-Ada | Phase 2 | scripts/gen_generic_kernel.py  (v3 - FINAL FIX)

FIXES vs v2:
  FIX A (v2): pre_loop r=1 save skipped → penalty_temp1=INFINITY preserved ✓
  FIX B (v3): loop bound changed + epilogue removed

Root cause of FIX B:
  Thread tgt_t processes its last valid row at k = L_query + tgt_t + 1.
  Old loop:  k < L_query + 32  → runs (31-tgt_t) garbage iterations after result ← corrupt
  Old epilogue: one more garbage step if result was at last main loop k         ← corrupt
  Fix: k < L_query + tgt_t + 2  → last iteration IS the result computation.
  The epilogue step is now included as the last main loop iteration. No separate epilogue.
"""
import sys

def save_var(r):
    return "penalty_temp0" if r % 2 == 0 else "penalty_temp1"

def diag_var(r):
    if r == 0: return "penalty_diag"
    return "penalty_temp0" if r % 2 == 1 else "penalty_temp1"

def gen_kernel(K):
    L = []; a = L.append

    a(f"// K={K:3d} | L≤{K*32-1}")
    a(f"template<typename index_t, typename value_t> __global__")
    a(f"__launch_bounds__(32*CUDTW_GENERIC_WARPS_PER_BLOCK, CUDTW_GENERIC_MIN_BLOCKS)")
    a(f"void shfl_FullDTW_K{K}(")
    a(f"    const value_t* __restrict__ Subject,")
    a(f"    value_t* __restrict__ Dist,")
    a(f"    index_t N, index_t L_query, index_t L_subject, index_t padded_stride)")
    a(f"{{")
    a(f"    // Multi-warp blocks (item #6): each warp handles one independent")
    a(f"    // sequence. lane = intra-warp 0..31; warp_in_block selects the warp.")
    a(f"    // All __shfl_*_sync below use width=32, so they stay warp-local.")
    a(f"    const index_t lane = threadIdx.x & 31u;")
    a(f"    const index_t warp_in_block = threadIdx.x >> 5;")
    a(f"    const index_t blid = (index_t)blockIdx.x * (blockDim.x >> 5) + warp_in_block;")
    a(f"    if (blid >= N) return;")
    a(f"    const index_t thid = lane;")
    a(f"    const index_t l = thid;")
    a(f"    const index_t WARP_SIZE = 32;")
    a(f"    const index_t base = blid * padded_stride;")
    a(f"")
    for r in range(K): a(f"    value_t penalty_here{r} = INFINITY;")
    a(f"    value_t penalty_left = INFINITY;")
    a(f"    value_t penalty_diag = 0;")
    a(f"    value_t penalty_temp0, penalty_temp1;")
    a(f"")
    a(f"    if (thid == 0) {{")
    a(f"        penalty_diag = INFINITY;")
    for r in range(K): a(f"        penalty_here{r} = INFINITY;")
    a(f"    }}")
    a(f"")
    a(f"#if CUDTW_GENERIC_LDG_SUBJECT")
    a(f"    #define SV0  ((l==0)?(value_t)0:__ldg(&Subject[base+{K}*l-1]))")
    for r in range(1, K+1):
        a(f"    #define SV{r}  (__ldg(&Subject[base+{K}*l+{r-1}]))")
    a(f"#else")
    a(f"    const value_t subject_value0 = (l==0)?(value_t)0:Subject[base+{K}*l-1];")
    for r in range(1, K+1):
        a(f"    const value_t subject_value{r} = Subject[base+{K}*l+{r-1}];")
    a(f"    #define SV0  subject_value0")
    for r in range(1, K+1):
        a(f"    #define SV{r}  subject_value{r}")
    a(f"#endif")
    a(f"")
    # FIX B: compute tgt_t here for loop bound
    a(f"    // Result position (needed for loop bound — FIX B)")
    a(f"    const int tgt_t = (int)(L_subject / {K});")
    a(f"    const int tgt_r = (int)(L_subject % {K});")
    a(f"    // Thread tgt_t processes row L_query-1 at k = L_query+tgt_t+1.")
    a(f"    // Loop runs until that step. No epilogue needed.")
    a(f"    const index_t k_stop = L_query + (index_t)tgt_t + 2;")
    a(f"")
    a(f"    index_t counter = 1;")
    a(f"    value_t query_value = INFINITY;")
    a(f"    value_t new_query_value = cQuery[thid];")
    a(f"    if (thid == 0) query_value = new_query_value;")
    if K >= 2:
        a(f"    if (thid == 0) penalty_here1 = (value_t)0;  // free start")
    a(f"    new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);")
    a(f"")

    def inner_row(pre_loop):
        a(f"        penalty_temp0 = penalty_here0;")
        a(f"        penalty_here0 = (query_value-SV0)*(query_value-SV0)"
          f"+min(penalty_left,min(penalty_here0,penalty_diag));")
        if pre_loop:
            a(f"        penalty_temp1 = INFINITY;")
        for r in range(1, K):
            sv = save_var(r); dv = diag_var(r); is_last = (r == K-1)
            skip_save = (pre_loop and r == 1)   # FIX A: keep INFINITY for r=2 diagonal
            if not is_last and not skip_save:
                a(f"        {sv} = penalty_here{r};")
            a(f"        penalty_here{r} = (query_value-SV{r})"
              f"*(query_value-SV{r})"
              f"+min(penalty_here{r-1},min(penalty_here{r},{dv}));")

    # PRE-LOOP
    a(f"    // Pre-loop: first anti-diagonal")
    inner_row(pre_loop=True)
    a(f"    query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);")
    a(f"    if (thid == 0) query_value = new_query_value;")
    a(f"    new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);")
    a(f"    counter++;")
    a(f"    penalty_diag = penalty_left;")
    a(f"    penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here{K-1}, 1, 32);")
    a(f"    if (thid == 0) penalty_left = INFINITY;")
    a(f"")

    # MAIN LOOP — FIX B: k < k_stop instead of k < lane+WARP_SIZE-1
    a(f"    // Main loop — runs until tgt_t computes row L_query-1 (FIX B)")
    a(f"    for (index_t k = 3; k < k_stop; k++) {{")
    a(f"        const index_t i = k - l;")
    a(f"        if (counter % 32 == 0) new_query_value = cQuery[i + 2*thid - 1];")
    inner_row(pre_loop=False)
    a(f"        query_value = __shfl_up_sync(0xFFFFFFFF, query_value, 1, 32);")
    a(f"        if (thid == 0) query_value = new_query_value;")
    a(f"        new_query_value = __shfl_down_sync(0xFFFFFFFF, new_query_value, 1, 32);")
    a(f"        counter++;")
    a(f"        penalty_diag = penalty_left;")
    a(f"        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here{K-1}, 1, 32);")
    a(f"        if (thid == 0) penalty_left = INFINITY;")
    a(f"    }}")
    a(f"")
    # NO EPILOGUE — absorbed into main loop via k_stop

    # RESULT EXTRACTION
    a(f"    // Extract dp[L_query-1][L_subject-1] from (tgt_t, tgt_r)")
    a(f"    {{")
    a(f"        value_t result = (value_t)0;")
    a(f"        if ((int)thid == tgt_t) {{")
    for r in range(K):
        a(f"            if (tgt_r == {r}) result = penalty_here{r};")
    a(f"        }}")
    a(f"        result = __shfl_sync(0xFFFFFFFF, result, tgt_t);")
    a(f"        if (thid == 0) Dist[blid] = result;")
    a(f"    }}")
    a(f"}}")
    a(f"#undef SV0")
    for r in range(1, K+1):
        a(f"#undef SV{r}")
    a(f"")
    return '\n'.join(L)

def gen_dispatch(k_min, k_max):
    L = []; a = L.append
    a(f"template<typename index_t, typename value_t>")
    a(f"inline void dispatch_generic(")
    a(f"    const value_t* Subject, value_t* Dist,")
    a(f"    index_t N, index_t L_query, index_t L_subject,")
    a(f"    index_t padded_stride, int K_val, cudaStream_t stream)")
    a(f"{{")
    a(f"    const int wpb = CUDTW_GENERIC_WARPS_PER_BLOCK;")
    a(f"    const dim3 block((unsigned)(32 * wpb));")
    a(f"    const dim3 grid((unsigned)((N + wpb - 1) / wpb));")
    a(f"    switch (K_val) {{")
    for K in range(k_min, k_max+1):
        a(f"        case {K}: shfl_FullDTW_K{K}<index_t,value_t><<<grid,block,0,stream>>>"
          f"(Subject,Dist,N,L_query,L_subject,padded_stride); break;")
    a(f"        default: break;")
    a(f"    }}")
    a(f"}}")
    return '\n'.join(L)

K_MIN, K_MAX = 2, 120
hdr = """\
// cuDTW-Ada | Phase 2 | SHFL_FULLDTW_GENERIC.cuh (v3 FINAL)
// AUTO-GENERATED — DO NOT EDIT. Run: python3 scripts/gen_generic_kernel.py
// FIX A: pre_loop skip save at r=1 (keep penalty_temp1=INFINITY)
// FIX B: k_stop = L_query+tgt_t+2; epilogue absorbed into main loop
#ifndef SHFL_FULLDTW_GENERIC_CUH
#define SHFL_FULLDTW_GENERIC_CUH

// ── Occupancy tuning knobs (item #6/#7) ──────────────────────────────────────
// WARPS_PER_BLOCK: independent sequences per block (each warp = one sequence).
// MIN_BLOCKS:      __launch_bounds__ second arg — min resident blocks/SM the
//                  compiler must keep feasible (caps registers accordingly).
//
// MEASURED on RTX 4090 (sm_89), tiled K=64, Lq=395 Ls=4437 N=65536, kernel-only:
//   register-resident, WPB=8, no cap   : 204 reg,  17% occ → 4152 GCUPS  (FAST)
//   __ldg subject,      WPB=8, no cap   : 153 reg,  17% occ →  347 GCUPS
//   __ldg, forced 33% occ (MIN_BLOCKS=2): 128 reg,  33% occ →  336 GCUPS
//   __ldg, forced 50% occ (MIN_BLOCKS=3):  80 reg,  50% occ →  306 GCUPS
// Conclusion: this is a per-warp recurrence-latency-bound kernel, NOT occupancy
// bound. Keeping the subject row + penalties in registers is THE win; __ldg and
// forced register caps both regress ~12×. WPB=1 and WPB=8 are a measured WASH
// (both ~8 warps/SM, ~4150 GCUPS, within run-to-run noise); --maxrregcount below
// the __launch_bounds__-implied cap is non-binding (regs stay ~204). Default to
// WPB=1 for simpler launch geometry. Override via -D only to re-experiment.
#ifndef CUDTW_GENERIC_WARPS_PER_BLOCK
#define CUDTW_GENERIC_WARPS_PER_BLOCK 1
#endif
#ifndef CUDTW_GENERIC_MIN_BLOCKS
#define CUDTW_GENERIC_MIN_BLOCKS 1
#endif
// Subject access mode (item #7): 0 = cache subject row in registers,
// 1 = read via __ldg at point of use (frees ~K registers, raises occupancy).
#ifndef CUDTW_GENERIC_LDG_SUBJECT
#define CUDTW_GENERIC_LDG_SUBJECT 0
#endif
"""

parts = [hdr]
for K in range(K_MIN, K_MAX+1): parts.append(gen_kernel(K))
parts.append(gen_dispatch(K_MIN, K_MAX))
parts.append("#endif\n")
sys.stdout.write('\n'.join(parts))
print(f"Generated K={K_MIN}..{K_MAX}", file=sys.stderr)
