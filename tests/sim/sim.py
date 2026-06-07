#!/usr/bin/env python3
"""
Bit-faithful simulator of the cuDTW-Ada warp-shuffle kernels.

We cannot run nvcc/GPU in this environment, so instead we *exactly* emulate the
generated CUDA at the warp level: 32 lanes, __shfl_up/down/sync, constant memory
query reads, and the same min/cost arithmetic the generator emits. If the
simulator matches a plain DTW reference, the generated algorithm is correct.

Everything is done in float32 (numpy.float32) to mirror GPU `float` math closely
(not bit-exact across HW, but close enough to catch boundary/off-by-one bugs).
"""
import numpy as np

INF = np.float32(np.inf)
f32 = np.float32

# ---------------------------------------------------------------------------
# Reference DTW that matches the kernel's boundary conditions.
#
# Kernel facts (from the generator):
#   * cost(i,j) = (q[i]-s[j])^2
#   * recurrence min over (up, left, diag)
#   * FREE START on the subject axis: for thid==0 (subject column 0), penalty_here1
#     is initialised to 0 before the pre-loop. penalty_here1 corresponds to the
#     cell (query row 0, subject col 0) the first time it is computed... we will
#     determine the exact semantics empirically from the simulator and bake the
#     matching boundary into this reference.
#
# Standard FullDTW init (constrained both ends):
#   dp[0][0] = c(0,0); first row = cumulative; first col = cumulative.
#
# The kernel sets penalty_here1=0 for lane0 (free start). We provide BOTH
# references and report which the kernel matches.
# ---------------------------------------------------------------------------

def dtw_constrained(q, s):
    """Standard FullDTW, constrained start and end."""
    q = np.asarray(q, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    Lq, Ls = len(q), len(s)
    dp = np.empty((Lq, Ls), dtype=np.float64)
    d = q[0]-s[0]; dp[0,0] = d*d
    for j in range(1, Ls):
        d = q[0]-s[j]; dp[0,j] = d*d + dp[0,j-1]
    for i in range(1, Lq):
        d = q[i]-s[0]; dp[i,0] = d*d + dp[i-1,0]
    for i in range(1, Lq):
        for j in range(1, Ls):
            d = q[i]-s[j]
            dp[i,j] = d*d + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[Lq-1, Ls-1]

def dtw_freestart(q, s):
    """FullDTW with free start on subject axis: the path may begin at any
    subject column j with cost c(0,j) (no accumulation across row 0)."""
    q = np.asarray(q, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    Lq, Ls = len(q), len(s)
    dp = np.empty((Lq, Ls), dtype=np.float64)
    for j in range(Ls):
        d = q[0]-s[j]; dp[0,j] = d*d          # free start: no row accumulation
    for i in range(1, Lq):
        d = q[i]-s[0]; dp[i,0] = d*d + dp[i-1,0]
    for i in range(1, Lq):
        for j in range(1, Ls):
            d = q[i]-s[j]
            dp[i,j] = d*d + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[Lq-1, Ls-1]


# ---------------------------------------------------------------------------
# Warp shuffle primitives over a 32-lane vector (numpy array length 32).
# ---------------------------------------------------------------------------
W = 32

def shfl_up(vec, delta=1):
    """__shfl_up_sync: lane l gets value from lane l-delta; lanes < delta keep own."""
    out = vec.copy()
    out[delta:] = vec[:W-delta]
    return out  # lanes [0:delta] unchanged (CUDA semantics)

def shfl_down(vec, delta=1):
    out = vec.copy()
    out[:W-delta] = vec[delta:]
    return out  # top lanes unchanged

def shfl_idx(vec, src):
    """__shfl_sync broadcast lane `src` value to all lanes."""
    out = np.empty_like(vec)
    out[:] = vec[src]
    return out


# ---------------------------------------------------------------------------
# Generic kernel simulator. Faithfully reproduces gen_generic_kernel.py.
# Subject is the PADDED subject for one sequence, length K*32.
# cQuery is the full query (length L_query). Returns Dist (scalar).
# ---------------------------------------------------------------------------
def sim_generic(K, cQuery, L_query, L_subject, subject_padded):
    s = subject_padded  # length K*32
    # subject_value[r] per lane l: r in 0..K
    # sv0 = (l==0)?0 : s[K*l - 1]; sv_r = s[K*l + (r-1)] for r=1..K
    sv = np.zeros((K+1, W), dtype=np.float32)
    for l in range(W):
        sv[0, l] = f32(0.0) if l == 0 else f32(s[K*l - 1])
        for r in range(1, K+1):
            sv[r, l] = f32(s[K*l + (r-1)])

    here = np.full((K, W), INF, dtype=np.float32)
    penalty_left = np.full(W, INF, dtype=np.float32)
    penalty_diag = np.full(W, np.float32(0.0), dtype=np.float32)
    # thid==0 sets penalty_diag=INF and here[*]=INF (already INF)
    penalty_diag[0] = INF

    tgt_t = L_subject // K
    tgt_r = L_subject % K
    k_stop = L_query + tgt_t + 2

    counter = 1
    query_value = np.full(W, INF, dtype=np.float32)
    def cq(idx):
        return f32(cQuery[idx]) if 0 <= idx < len(cQuery) else f32(0.0)
    new_query_value = np.array([cq(t) for t in range(W)], dtype=np.float32)
    query_value[0] = new_query_value[0]
    if K >= 2:
        here[1, 0] = f32(0.0)  # free start
    new_query_value = shfl_down(new_query_value, 1)

    # Helper names mirroring the generator:
    #   save_var(r) = temp0 if r%2==0 else temp1   (where here[r] is stored)
    #   diag_var(r) = diag if r==0 else (temp0 if r%2==1 else temp1)
    def inner(pre):
        nonlocal_temp0 = {}
        nonlocal_temp1 = {}
        # Emit exactly as generator does, statement by statement.
        # r=0:
        temp0 = here[0].copy()                                   # penalty_temp0 = penalty_here0
        here[0] = (query_value - sv[0])*(query_value - sv[0]) + \
                  np.minimum(penalty_left, np.minimum(here[0], penalty_diag))
        temp1 = None
        if pre:
            temp1 = np.full(W, INF, dtype=np.float32)            # penalty_temp1 = INFINITY
        for r in range(1, K):
            is_last = (r == K-1)
            skip = (pre and r == 1)                              # FIX A
            # save: sv_r = penalty_here{r}  (skipped when is_last or skip)
            if not is_last and not skip:
                if r % 2 == 0:
                    temp0 = here[r].copy()
                else:
                    temp1 = here[r].copy()
            # diag var
            if r % 2 == 1:
                d = temp0
            else:
                d = temp1
            here[r] = (query_value - sv[r])*(query_value - sv[r]) + \
                      np.minimum(here[r-1], np.minimum(here[r], d))

    # PRE-LOOP
    inner(pre=True)
    query_value = shfl_up(query_value, 1)
    query_value[0] = new_query_value[0]
    new_query_value = shfl_down(new_query_value, 1)
    counter += 1
    penalty_diag = penalty_left.copy()
    penalty_left = shfl_up(here[K-1], 1)
    penalty_left[0] = INF

    k = 3
    while k < k_stop:
        i = k  # i = k - l per-lane; but cQuery reload uses i + 2*thid -1 = k - l + 2l -1 = k + l -1
        if counter % 32 == 0:
            # new_query_value[thid] = cQuery[(k - thid) + 2*thid - 1] = cQuery[k + thid - 1]
            nqv = new_query_value.copy()
            for thid in range(W):
                idx = (k - thid) + 2*thid - 1  # = k + thid - 1
                nqv[thid] = cq(idx)
            new_query_value = nqv
        inner(pre=False)
        query_value = shfl_up(query_value, 1)
        query_value[0] = new_query_value[0]
        new_query_value = shfl_down(new_query_value, 1)
        counter += 1
        penalty_diag = penalty_left.copy()
        penalty_left = shfl_up(here[K-1], 1)
        penalty_left[0] = INF
        k += 1

    # result extraction
    result = here[tgt_r, tgt_t]
    return float(result)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # quick sanity: small case
    Lq = 10
    Ls = 7
    K = (Ls + 32)//32  # =1? no -> ensure K>=4 path. Use bigger Ls
    print("loaded sim module ok")
