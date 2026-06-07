#!/usr/bin/env python3
"""
Independent cross-validation of the cuDTW-Ada kernel logic.

Compares the bit-faithful kernel SIMULATOR (tests/sim, which emulates exactly
what gen_*.py emits) against the INDEPENDENT dtaidistance library — not the
project's own CPU reference. This rules out "shared bug between kernel and its
own reference".

Convention: dtaidistance.distance(q,s)**2 == project's constrained FullDTW
(squared cost, no sqrt, both ends constrained). Verified separately.

Sweeps a wide range of lengths: every generic K boundary, every tile boundary,
random lengths, and the real user lengths — across multiple seeds and query
lengths.
"""
import sys, numpy as np
sys.path.insert(0, 'tests/sim')
import sim                      # generic-path simulator + helpers
import sim_tiled as ST          # tiled-path simulator (matches gen_tiled_kernel)
from dtaidistance import dtw

TILE_LEN = 2047
def compute_K(L): return (L + 32) // 32

def pad_generic(s, K):
    stride = K * 32
    out = np.empty(stride, dtype=np.float32)
    out[:len(s)] = s
    out[len(s):] = s[-1]
    return out

def project_sim(q, s):
    """Run the appropriate path's simulator, mirroring dist_any."""
    Lq, Ls = len(q), len(s)
    K = compute_K(Ls)
    if K <= 120:
        return sim.sim_generic(K, q, Lq, Ls, pad_generic(s, K))
    return ST.sim_dist_tiled(q, Lq, s)

def independent_ref(q, s):
    """dtaidistance, squared to match the project's no-sqrt convention."""
    d = dtw.distance(q.astype(np.float64), s.astype(np.float64))
    return d * d

def run_sweep():
    rng_master = np.random.default_rng(20240601)
    # Length set: generic K-boundaries, tile boundaries, randoms, real data.
    generic_lengths = [96, 97, 127, 128, 159, 200, 255, 256, 500, 1000,
                       2047, 2048, 3000, 3807, 3839]
    tile_lengths    = [3840, 3841, 4093, 4094, 4095, 4096, 4097, 4159,
                       6140, 6141, 6142, 6143, 8188, 8189, 8190, 10235, 12282]
    real_lengths    = [4437, 5788]
    random_lengths  = sorted(int(x) for x in rng_master.integers(96, 9000, 25))
    all_lengths = sorted(set(generic_lengths + tile_lengths + real_lengths + random_lengths))

    query_lengths = [395, 100, 250, 64]
    tol = 1e-4

    worst = 0.0; n = 0; fails = []
    print(f"{'Lq':>5}{'Ls':>6}{'path':>9} | {'sim':>15}{'dtai^2':>15}{'rel':>11}")
    print("-" * 64)
    for Lq in query_lengths:
        for Ls in all_lengths:
            seed = (Lq * 7919 + Ls) & 0xffffffff
            rng = np.random.default_rng(seed)
            q = rng.uniform(-3, 3, Lq).astype(np.float32)
            s = rng.uniform(-3, 3, Ls).astype(np.float32)
            sim_v = project_sim(q, s)
            ref_v = independent_ref(q, s)
            rel = abs(sim_v - ref_v) / ref_v if ref_v > 1e-6 else abs(sim_v - ref_v)
            worst = max(worst, rel); n += 1
            path = "generic" if compute_K(Ls) <= 120 else "tiled"
            if rel >= tol:
                fails.append((Lq, Ls, sim_v, ref_v, rel))
            # Print only a representative subset to keep output readable
            if Lq == 395 and (Ls in generic_lengths or Ls in tile_lengths or Ls in real_lengths):
                print(f"{Lq:>5}{Ls:>6}{path:>9} | {sim_v:>15.4f}{ref_v:>15.4f}{rel:>11.2e}")

    print("-" * 64)
    print(f"Total comparisons: {n} (queries {query_lengths} × {len(all_lengths)} lengths)")
    print(f"Worst relative error: {worst:.3e}")
    if fails:
        print(f"\n*** {len(fails)} FAILURES (rel >= {tol}) ***")
        for Lq, Ls, sv, rv, rel in fails[:20]:
            print(f"  Lq={Lq} Ls={Ls}: sim={sv:.4f} ref={rv:.4f} rel={rel:.2e}")
    else:
        print(f"\nALL {n} COMPARISONS PASS (rel < {tol}) vs independent dtaidistance.")
    return len(fails) == 0

if __name__ == "__main__":
    ok = run_sweep()
    sys.exit(0 if ok else 1)
