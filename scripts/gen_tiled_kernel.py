#!/usr/bin/env python3
"""
cuDTW-Ada | Phase 3 | gen_tiled_kernel.py
Generates SHFL_FULLDTW_TILED.cuh — Tiled Warp-Shuffle kernels.

Run:
  python3 scripts/gen_tiled_kernel.py > src/include/kernels/SHFL_FULLDTW_TILED.cuh
"""
import sys

def sv(r): return "penalty_temp0" if r % 2 == 0 else "penalty_temp1"
def dv(r):
    if r == 0: return "penalty_diag"
    return "penalty_temp0" if r % 2 == 1 else "penalty_temp1"

def kernel(K, rw=False):
    """Generate one tiled kernel.
    rw=True  → intermediate tile: reads left_bnd + writes right_bnd (K=64 only)
    rw=False → last tile: reads left_bnd, outputs Dist (K=4..120)
    """
    L = []; a = L.append
    nm = f"shfl_FullDTW_Tiled_K{K}" + ("_RW_impl" if rw else "")
    a(f"// K={K:3d}  {'[intermediate: left+right bnd]' if rw else '[last tile: left bnd only]'}")
    a(f"template<typename index_t,typename value_t> __global__")
    a(f"__launch_bounds__(32*CUDTW_TILED_WARPS_PER_BLOCK, CUDTW_TILED_MIN_BLOCKS)")
    a(f"void {nm}(")
    a(f"    const value_t* __restrict__ Subject, value_t* __restrict__ Dist,")
    a(f"    index_t N, index_t L_query, index_t L_subject, index_t padded_stride,")
    a(f"    const value_t* __restrict__ left_bnd, value_t* __restrict__ right_bnd)")
    a(f"{{")
    a(f"    // Multi-warp blocks (item #6): one warp per sequence; lane is intra-warp.")
    a(f"    const index_t lane = threadIdx.x & 31u;")
    a(f"    const index_t warp_in_block = threadIdx.x >> 5;")
    a(f"    const index_t blid = (index_t)blockIdx.x * (blockDim.x >> 5) + warp_in_block;")
    a(f"    if (blid >= N) return;")
    a(f"    const index_t thid=lane, l=lane;")
    a(f"    const index_t base = blid * padded_stride;")
    a(f"    const int Lq = (int)L_query;")
    a(f"")
    for r in range(K): a(f"    value_t penalty_here{r} = INFINITY;")
    a(f"    value_t penalty_left=INFINITY, penalty_diag=0, penalty_temp0, penalty_temp1;")
    a(f"    if (thid == 0) {{")
    a(f"        penalty_diag = INFINITY;")
    for r in range(K): a(f"        penalty_here{r} = INFINITY;")
    a(f"    }}")
    a(f"")
    # ── Subject access: registers vs __ldg (item #7, register pressure) ──────
    # CUDTW_TILED_LDG_SUBJECT=0 → cache all K+1 subject values in registers
    #   (fast per-access, but ~K registers held for the whole kernel → low occ).
    # CUDTW_TILED_LDG_SUBJECT=1 → read each subject value via __ldg at point of
    #   use (frees ~K registers → higher occupancy; the addresses are constant
    #   and warp-shared so they stay hot in L1/L2). svN_ADDR helpers keep the
    #   index arithmetic identical to the register version.
    a(f"#if CUDTW_TILED_LDG_SUBJECT")
    a(f"    #define SV0  ((l==0) ? (value_t)0 : __ldg(&Subject[base + {K}*l - 1]))")
    for r in range(1, K+1):
        a(f"    #define SV{r}  (__ldg(&Subject[base + {K}*l + {r-1}]))")
    a(f"#else")
    a(f"    const value_t sv0 = (l==0) ? (value_t)0 : Subject[base + {K}*l - 1];")
    for r in range(1, K+1):
        a(f"    const value_t sv{r} = Subject[base + {K}*l + {r-1}];")
    a(f"    #define SV0  sv0")
    for r in range(1, K+1):
        a(f"    #define SV{r}  sv{r}")
    a(f"#endif")
    a(f"")
    a(f"    const int tgt_t = (int)(L_subject / {K});")
    a(f"    const int tgt_r = (int)(L_subject % {K});")
    a(f"    const index_t k_stop = L_query + (index_t)tgt_t + 2;")
    a(f"")
    a(f"    index_t cnt = 1;")
    a(f"    value_t qv = INFINITY, nqv = cQuery[thid];")
    a(f"    if (thid == 0) qv = nqv;")
    # FIX (tiled boundary #3): free-start ONLY on the first tile (no left_bnd).
    # On later tiles, column 0's predecessor comes from the seam, so the
    # open-start init must NOT be applied or it wins the min() and erases it.
    if K >= 2: a(f"    if (thid == 0 && !left_bnd) penalty_here1 = (value_t)0;")
    a(f"    nqv = __shfl_down_sync(0xFFFFFFFF, nqv, 1, 32);")
    a(f"")
    a(f"    // ── Tile left-boundary (seam) injection ──────────────────────────")
    a(f"    // FIX (tiled boundary #1,#2): the seam left_bnd[i] = dp[i][last_col")
    a(f"    // of previous tile] must become the LEFT neighbour of this tile's")
    a(f"    // real column 0. Real col 0 is penalty_here1 on lane 0; its left")
    a(f"    // neighbour is penalty_here0 on lane 0 (the phantom sv0=0 column).")
    a(f"    // So we OVERWRITE penalty_here0 on lane 0 with seam[row] right after")
    a(f"    // it is recomputed each anti-diagonal. Schedule (proven in sim):")
    a(f"    //   pre-loop      -> row 0")
    a(f"    //   main loop k   -> row (k-2)")
    a(f"    // The diagonal for col 0 at row i is then seam[i-1], which arrives")
    a(f"    // automatically via penalty_temp0 (= previous step's penalty_here0).")
    a(f"")

    def inner(pre, row_expr):
        a(f"        penalty_temp0 = penalty_here0;")
        a(f"        penalty_here0 = (qv-SV0)*(qv-SV0) + min(penalty_left, min(penalty_here0, penalty_diag));")
        # Inject seam into lane-0 phantom column immediately after its recompute.
        a(f"        if (thid == 0 && left_bnd) {{")
        a(f"            const int _row = (int)({row_expr});")
        a(f"            penalty_here0 = (_row >= 0 && _row < Lq)")
        a(f"                            ? left_bnd[blid * Lq + _row] : INFINITY;")
        a(f"        }}")
        if pre: a(f"        penalty_temp1 = INFINITY;")
        for r in range(1, K):
            s = sv(r); d = dv(r); last = (r == K-1); skip = (pre and r == 1)
            if not last and not skip: a(f"        {s} = penalty_here{r};")
            a(f"        penalty_here{r} = (qv-SV{r})*(qv-SV{r}) + min(penalty_here{r-1}, min(penalty_here{r}, {d}));")

    # PRE-LOOP (lane-0 col 0 computes query row 0)
    a(f"    // Pre-loop: first anti-diagonal")
    inner(True, "0")
    a(f"    qv  = __shfl_up_sync(0xFFFFFFFF, qv, 1, 32);")
    a(f"    if (thid == 0) qv = nqv;")
    a(f"    nqv = __shfl_down_sync(0xFFFFFFFF, nqv, 1, 32);")
    a(f"    cnt++;")
    a(f"    penalty_diag = penalty_left;")
    a(f"    penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here{K-1}, 1, 32);")
    a(f"    if (thid == 0) penalty_left = INFINITY;")
    a(f"")

    # MAIN LOOP (lane-0 col 0 at step k computes query row k-2)
    a(f"    // Main loop")
    a(f"    for (index_t k = 3; k < k_stop; k++) {{")
    a(f"        const index_t i = k - l;")
    a(f"        if (cnt % 32 == 0) nqv = cQuery[i + 2*thid - 1];")
    inner(False, "(int)k - 2")
    if rw:
        a(f"        // Write right boundary from thread 31 (last real col of tile)")
        a(f"        if (thid == 31 && right_bnd) {{")
        a(f"            int rr = (int)k - 2 - 31;")
        a(f"            if (rr >= 0 && rr < Lq) right_bnd[blid * Lq + rr] = penalty_here{K-1};")
        a(f"        }}")
    a(f"        qv  = __shfl_up_sync(0xFFFFFFFF, qv, 1, 32);")
    a(f"        if (thid == 0) qv = nqv;")
    a(f"        nqv = __shfl_down_sync(0xFFFFFFFF, nqv, 1, 32);")
    a(f"        cnt++;")
    a(f"        penalty_diag = penalty_left;")
    a(f"        penalty_left = __shfl_up_sync(0xFFFFFFFF, penalty_here{K-1}, 1, 32);")
    a(f"        if (thid == 0) penalty_left = INFINITY;")
    a(f"    }}")
    a(f"")
    # RESULT EXTRACTION
    a(f"    {{ value_t result = (value_t)0;")
    a(f"      if ((int)thid == tgt_t) {{")
    for r in range(K):
        a(f"          if (tgt_r == {r}) result = penalty_here{r};")
    a(f"      }}")
    a(f"      result = __shfl_sync(0xFFFFFFFF, result, tgt_t);")
    if not rw:
        a(f"      if (thid == 0) Dist[blid] = result;")
    a(f"    }}")
    a(f"}}")
    # Undefine the per-kernel SV macros so they don't leak into the next kernel.
    a(f"#undef SV0")
    for r in range(1, K+1):
        a(f"#undef SV{r}")
    a(f"")
    return '\n'.join(L)

def dispatch():
    L = []; a = L.append
    a(f"template<typename index_t, typename value_t>")
    a(f"inline void dispatch_tiled_last(")
    a(f"    const value_t* S, value_t* D, index_t N, index_t Lq, index_t Ls,")
    a(f"    index_t ps, const value_t* lb, int K, cudaStream_t st)")
    a(f"{{")
    a(f"    const int wpb = CUDTW_TILED_WARPS_PER_BLOCK;")
    a(f"    const dim3 b((unsigned)(32*wpb)), g((unsigned)((N + wpb - 1)/wpb));")
    a(f"    switch (K) {{")
    for K in range(4, 121):
        a(f"        case {K}: shfl_FullDTW_Tiled_K{K}<index_t,value_t><<<g,b,0,st>>>")
        a(f"                     (S,D,N,Lq,Ls,ps,lb,nullptr); break;")
    a(f"        default: break;")
    a(f"    }}")
    a(f"}}")
    return '\n'.join(L)

def dispatch_inter():
    """Dispatcher for intermediate tiles (read left bnd, write right bnd)."""
    L = []; a = L.append
    a(f"template<typename index_t, typename value_t>")
    a(f"inline void dispatch_tiled_inter(")
    a(f"    const value_t* S, value_t* D, index_t N, index_t Lq, index_t Ls,")
    a(f"    index_t ps, const value_t* lb, value_t* rb, int K, cudaStream_t st)")
    a(f"{{")
    a(f"    const int wpb = CUDTW_TILED_WARPS_PER_BLOCK;")
    a(f"    const dim3 b((unsigned)(32*wpb)), g((unsigned)((N + wpb - 1)/wpb));")
    a(f"    switch (K) {{")
    for K in range(4, 121):
        a(f"        case {K}: shfl_FullDTW_Tiled_K{K}_RW_impl<index_t,value_t><<<g,b,0,st>>>")
        a(f"                     (S,D,N,Lq,Ls,ps,lb,rb); break;")
    a(f"        default: break;")
    a(f"    }}")
    a(f"}}")
    return '\n'.join(L)

if __name__ == "__main__":
    K_MIN, K_MAX = 4, 120
    print("// cuDTW-Ada | Phase 3 | SHFL_FULLDTW_TILED.cuh — AUTO-GENERATED")
    print("// Tiled Warp-Shuffle DTW (replaces Anti-Diagonal Phase 3)")
    print("// Performance: ~4150 GCUPS measured (RTX 4090, K=64, kernel-only) — see RESULTS.md")
    print("#ifndef SHFL_FULLDTW_TILED_CUH")
    print("#define SHFL_FULLDTW_TILED_CUH")
    print()
    print("// Warps per block for tiled kernels (item #6/#7, occupancy).")
    print("// MEASURED on RTX 4090, K=64, Lq=395 Ls=4437 N=65536 (kernel-only):")
    print("//   WPB=1, registers          : 206 reg,  8 warps/SM -> 4146 GCUPS")
    print("//   WPB=8, registers          : 204 reg,  8 warps/SM -> 4154 GCUPS")
    print("//   WPB=8, --maxrreg=80 (n/b)  : 204 reg,  8 warps/SM -> 4183 GCUPS")
    print("//   WPB=8, --maxrreg=64 (n/b)  : 204 reg,  8 warps/SM -> 4126 GCUPS")
    print("//   WPB=8, __ldg subject       : 153 reg,            ->  347 GCUPS")
    print("//   WPB=8, __ldg + forced 50%  :  80 reg, 24 warps/SM ->  306 GCUPS")
    print("// This kernel is per-warp recurrence-latency-bound, NOT occupancy-")
    print("// bound. Keeping subject+penalties in registers is THE win; __ldg")
    print("// regresses ~12x. WPB and --maxrregcount are a wash/non-binding here")
    print("// (the register count is fixed by __launch_bounds__, not the flag).")
    print("// Defaults: WPB=1, MIN_BLOCKS=1, LDG=0 (simplest of the equal-fastest).")
    print("#ifndef CUDTW_TILED_WARPS_PER_BLOCK")
    print("#define CUDTW_TILED_WARPS_PER_BLOCK 1")
    print("#endif")
    print("#ifndef CUDTW_TILED_MIN_BLOCKS")
    print("#define CUDTW_TILED_MIN_BLOCKS 1")
    print("#endif")
    print("// Subject access mode (item #7): 0 = cache subject row in registers")
    print("// (low occupancy), 1 = read via __ldg at point of use (frees ~K regs).")
    print("#ifndef CUDTW_TILED_LDG_SUBJECT")
    print("#define CUDTW_TILED_LDG_SUBJECT 0")
    print("#endif")
    print()
    # Intermediate tiles (read left bnd + write right bnd) for ALL K, so the
    # host can rebalance tile lengths and keep every tile's K ≤ 64 (no register
    # blow-up) while still writing a seam. K64_RW is kept as an alias below.
    for K in range(K_MIN, K_MAX + 1):
        print(kernel(K, True))    # intermediate tile, all K
    for K in range(K_MIN, K_MAX + 1):
        print(kernel(K, False))   # last tile, all K
    print(dispatch())
    print(dispatch_inter())
    print("#endif")
    print(f"// Generated K={K_MIN}..{K_MAX} (RW + last)", file=sys.stderr)
