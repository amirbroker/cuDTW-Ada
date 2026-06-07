#!/usr/bin/env python3
"""
GPU cross-validation: compares the REAL compiled kernel output (via libcudtw.so)
against the INDEPENDENT dtaidistance library, sweeping a WIDE range of QUERY
(target) lengths on both the generic and tiled subject paths.

Prereq:  make libcudtw.so
         pip install dtaidistance numpy
Run:     python3 tests/gpu_cross_validate.py

Convention: dtaidistance.distance(q,s)**2 == this project's distance
(constrained FullDTW, squared cost, no final sqrt).
"""
import sys, numpy as np
sys.path.insert(0, 'python')
import cudtw
from dtaidistance import dtw

MAX_QUERY = 8192   # must match CUDTW_MAX_QUERY in src/include/cudtw_constants.hpp

def ref_sq(q, s):
    d = dtw.distance(q.astype(np.float64), s.astype(np.float64))
    return d * d

def main():
    cudtw.init(0)
    rng = np.random.default_rng(2024)
    tol = 1e-4

    # Sweep many QUERY lengths. For each, test against a generic-path subject
    # and a tiled-path subject (and real data lengths). Includes:
    #  - very short queries (16, 32, 64)
    #  - mid queries (128, 395, 512, 1024)
    #  - long queries (2048, 4096) including Lq > Ls (query longer than subject)
    #  - near the cap (8000, 8190)
    query_lengths = [16, 32, 48, 64, 100, 128, 256, 395, 512, 768,
                     1024, 1536, 2048, 3072, 4096, 6144, 8000, 8190]

    # Subjects to test each query against: one generic, two tiled (real data).
    subject_lengths = [500, 4437, 5788]

    # Fewer sequences for big problems so dtaidistance stays fast.
    def n_for(Lq, Ls):
        work = Lq * Ls
        if work > 30_000_000: return 8
        if work > 8_000_000:  return 16
        return 48

    print(f"{'Lq':>6}{'Ls':>6}{'N':>5} | {'max_rel':>11}  verdict")
    print("-" * 42)
    worst = 0.0; fails = 0; total = 0
    for Lq in query_lengths:
        if Lq > MAX_QUERY:
            print(f"{Lq:>6}  (skipped: exceeds MAX_QUERY={MAX_QUERY})")
            continue
        for Ls in subject_lengths:
            N = n_for(Lq, Ls)
            q = rng.uniform(-3, 3, Lq).astype(np.float32)
            subjects = rng.uniform(-3, 3, (N, Ls)).astype(np.float32)

            gpu = cudtw.compute(q, subjects)      # real kernel output, shape (N,)

            case_worst = 0.0
            for i in range(N):
                r = ref_sq(q, subjects[i])
                rel = abs(gpu[i] - r) / r if r > 1e-6 else abs(gpu[i] - r)
                case_worst = max(case_worst, rel)
            worst = max(worst, case_worst); total += 1
            ok = case_worst < tol
            fails += 0 if ok else 1
            flag = "Lq>Ls" if Lq > Ls else ""
            print(f"{Lq:>6}{Ls:>6}{N:>5} | {case_worst:>11.2e}  "
                  f"{'PASS' if ok else '**FAIL**'} {flag}")

    print("-" * 42)
    print(f"worst rel = {worst:.3e}  over {total} (Lq,Ls) cases")
    print("RESULT:", "ALL PASS vs dtaidistance" if fails == 0 else f"{fails} FAILED")
    return 0 if fails == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
