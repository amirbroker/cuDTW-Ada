"""
Offline (no-GPU) simulator of the TILED warp-shuffle kernels.
Mirrors gen_tiled_kernel.py + cudtw_tiled.hpp (fixed boundary injection).
"""
import numpy as np
from sim import INF, f32, W, shfl_up, shfl_down

TILE_K = 64
TILE_SIZE = TILE_K * 32      # 2048
TILE_LEN = TILE_SIZE - 1     # 2047
MIN_TILE_LEN = 96            # minimum last-tile length for K in [4,120]

def compute_K(L): return (L + 32) // 32

VALID_INTER = {32*K-1 for K in range(4,65)}

def _largest_inter_width(R):
    K = (R - MIN_TILE_LEN + 1)//32
    K = min(64, max(4, K))
    return 32*K - 1

def _schedule(Ls):
    """List of (start,len). Intermediate tiles are exactly 32*K-1; last tile any>=96."""
    n = (Ls + TILE_LEN - 1)//TILE_LEN
    if n < 2: return [(0, Ls)]
    last = Ls - (n-1)*TILE_LEN
    if last >= MIN_TILE_LEN:
        lens = [TILE_LEN]*(n-1) + [last]
    else:
        R = Ls - (n-2)*TILE_LEN
        a = _largest_inter_width(R); b = R - a
        lens = [TILE_LEN]*(n-2) + [a, b]
    starts = []; acc = 0
    for L in lens: starts.append((acc, L)); acc += L
    return starts

def pad_tile(s_full, tile_start, tile_len, padded_stride):
    out = np.empty(padded_stride, dtype=np.float32)
    for c in range(padded_stride):
        out[c] = f32(s_full[tile_start + c]) if c < tile_len \
                 else f32(s_full[tile_start + tile_len - 1])
    return out

def _cq_fn(cQuery):
    def cq(i): return f32(cQuery[i]) if 0 <= i < len(cQuery) else f32(0.0)
    return cq

def sim_tiled_kernel(K, cQuery, L_query, tile_len, padded_stride,
                     subject_padded, seam, write_right):
    """
    Simulate one tiled kernel call (mirrors gen_tiled_kernel.py FIXED version).
    seam: None (first tile) or np.array[L_query] = dp column at tile seam.
    Returns (result_or_None, right_bnd_or_None).
    """
    s = subject_padded; Lq = L_query; cq = _cq_fn(cQuery)
    sv = np.zeros((K + 1, W), dtype=np.float32)
    for l in range(W):
        sv[0, l] = f32(0.0) if l == 0 else f32(s[K * l - 1])
        for r in range(1, K + 1): sv[r, l] = f32(s[K * l + (r - 1)])

    here = np.full((K, W), INF, dtype=np.float32)
    pl = np.full(W, INF, dtype=np.float32)
    pd = np.full(W, np.float32(0.0)); pd[0] = INF

    tgt_t = tile_len // K; tgt_r = tile_len % K
    k_stop = Lq + tgt_t + 2
    right = np.full(Lq, INF, dtype=np.float32) if write_right else None
    cnt = 1
    qv = np.full(W, INF, dtype=np.float32)
    nqv = np.array([cq(t) for t in range(W)], dtype=np.float32)
    qv[0] = nqv[0]
    # FIX #3: free-start only for first tile (no seam)
    if K >= 2 and seam is None: here[1, 0] = f32(0.0)
    nqv = shfl_down(nqv, 1)

    def sa(row):
        if seam is None: return INF
        return f32(seam[row]) if 0 <= row < Lq else INF

    def inner(pre, row):
        # FIX #1+#2: inject seam into here[0]@lane0 right after its recompute,
        # on schedule: pre-loop → row 0; main loop step k → row k-2.
        temp0 = here[0].copy()
        here[0] = (qv - sv[0])**2 + np.minimum(pl, np.minimum(here[0], pd))
        here[0, 0] = sa(row)           # overwrite phantom col with seam value
        temp1 = np.full(W, INF, dtype=np.float32) if pre else None
        for r in range(1, K):
            is_last = (r == K - 1); skip = (pre and r == 1)
            if not is_last and not skip:
                if r % 2 == 0: temp0 = here[r].copy()
                else: temp1 = here[r].copy()
            d = temp0 if r % 2 == 1 else temp1
            here[r] = (qv - sv[r])**2 + np.minimum(here[r-1], np.minimum(here[r], d))

    # PRE-LOOP (row 0)
    inner(True, 0)
    qv = shfl_up(qv, 1); qv[0] = nqv[0]; nqv = shfl_down(nqv, 1)
    cnt += 1; pd = pl.copy(); pl = shfl_up(here[K - 1], 1); pl[0] = INF

    k = 3
    while k < k_stop:
        if cnt % 32 == 0:
            tmp = nqv.copy()
            for thid in range(W):
                tmp[thid] = cq((k - thid) + 2 * thid - 1)
            nqv = tmp
        inner(False, k - 2)
        if write_right:
            rr = k - 2 - 31
            if 0 <= rr < Lq: right[rr] = here[K - 1, 31]
        qv = shfl_up(qv, 1); qv[0] = nqv[0]; nqv = shfl_down(nqv, 1)
        cnt += 1; pd = pl.copy(); pl = shfl_up(here[K - 1], 1); pl[0] = INF
        k += 1

    result = None if write_right else float(here[tgt_r, tgt_t])
    return result, right


def sim_dist_tiled(cQuery, L_query, s_full):
    """Full tiled pipeline for one sequence, mirrors cudtw_tiled.hpp dist_tiled."""
    Lq = L_query; Ls = len(s_full)
    sched = _schedule(Ls)
    bnd_a = np.zeros(Lq, np.float32); bnd_b = np.zeros(Lq, np.float32)
    result = None
    for t, (ts, tl) in enumerate(sched):
        first = (t == 0); last = (t == len(sched)-1)
        K = compute_K(tl); ps = K*32
        seam = None if first else bnd_a
        sp = pad_tile(s_full, ts, tl, ps)
        if not last:
            _, r = sim_tiled_kernel(K, cQuery, Lq, tl, ps, sp, seam, True)
            bnd_b = r
        else:
            result, _ = sim_tiled_kernel(K, cQuery, Lq, tl, ps, sp, seam, False)
        bnd_a, bnd_b = bnd_b, bnd_a
    return result
