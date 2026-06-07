# cuDTW-Ada — Fix & Optimization Summary

## Phase 1 — Correctness (all hardware-verified, `./validate` = 25/25 PASS)

| # | Issue | Fix |
|---|-------|-----|
| 5 | **Tiled boundary produced 74–96% wrong results on real data** | Three coupled bugs: (a) seam injected into a phantom column that never reached real col 0; (b) diagonal misaligned; (c) free-start leaking into non-first tiles. Fixed in `gen_tiled_kernel.py`. |
| 5b | Last tile with K<4 (tiny remainder) → silent `gpu=0.0`; rebalanced tile with K=65 → ~1e-3 error | Tile schedule now keeps **intermediate tiles exactly `32·K−1`** (the only width where the seam-write column is correct) and the last tile any length ≥96. RW kernels generated for all K. |
| 5c | Whole subject with `L_subject < 96` (K<4) → no kernel exists → silent wrong/zero output | Added `MIN_SUBJECT_LEN=96` guard in `dist_any` (both overloads) and the library `dtw_compute`: now returns a clear error instead of garbage. Documented in README Limits. |
| 1 | `cudaMalloc`/`cudaFree` in hot loop (implicit device sync, kills stream overlap) | Per-stream `DtwScratch`, allocated once; scratch-based `dist_any`/`dist_tiled`. |
| 3 | L2 persistence on per-batch, read-once buffer | Removed (query is in constant memory; persistence gave no reuse). |
| 4 | `MAX_Q`=8192 vs `MAX_QUERY`=16384 mismatch | Unified `CUDTW_MAX_QUERY` (8192) + `+64` storage margin (kernel over-reads to `L_query+62`). |
| 2 | index_t/int | Address-path multiplies were already 64-bit; confirmed + array sizing fixed. |
| 9 | Missing `tests/validate.cu` broke `make` | Created CPU-reference validator (generic + tiled + boundary lengths + real data). |
| 10 | `load_binary` const-correctness UB | `const value_t*` → `value_t*`. |
| 11 | Silent skip of malformed files | Clear error messages. |

## Phase 2 — Performance (measured on RTX 4090, kernel-only)

Tiled K=64, Lq=395, Ls=4437, N=65536:

| Config | regs/thread | warps/SM | GCUPS |
|--------|-------------|----------|-------|
| WPB=1, register-resident (shipped) | 206 | 8 | 4146 |
| WPB=8, register-resident | 204 | 8 | 4154 |
| WPB=8, `--maxrregcount=80` (non-binding) | 204 | 8 | 4183 |
| WPB=8, `--maxrregcount=64` (non-binding) | 204 | 8 | 4126 |
| WPB=8, `__ldg` subject | 153 | — | 347 |
| WPB=8, `__ldg` + forced 50% occ | 80 | 24 | 306 |

**Findings:**
1. The kernel is **per-warp recurrence-latency-bound, not occupancy-bound.**
   It sits at ~4150 GCUPS and does not move via WPB, `MIN_BLOCKS`, or
   `--maxrregcount`. Those first four rows are all within run-to-run noise (±1.4%).
2. WPB=1 vs WPB=8 is a **wash** — both reach 8 warps/SM (the register file caps
   resident threads either way: ~204 reg × 256 threads ≈ one block per SM).
3. `--maxrregcount` below the `__launch_bounds__`-implied cap is **non-binding**:
   the kernel keeps ~204 registers regardless, because `__launch_bounds__` already
   sets the register target.
4. `__ldg` subject loading (item #7's hypothesis) is a **~12× regression** — it
   puts a memory load on the critical path of the serial recurrence.

**Shipped defaults:** `WARPS_PER_BLOCK=1, MIN_BLOCKS=1, LDG_SUBJECT=0` — the
simplest of the equal-fastest configurations. All knobs remain as `-D` overrides;
the measured table is recorded in the generator preambles so the defaults aren't
"optimized" back into a regression.

## Phase 3 — Engineering / cleanup
- #9, #10, #11: done (see Phase 1 table).
- #12 timing unified: both drivers (`main_v4`, `main_v5`) time the identical
  window — H2D + kernel + D2H, excluding disk I/O and device allocation —
  documented inline at each timer. The `bench` harness times a *different*
  (kernel-only) window, now clearly labelled so it's never compared against the
  drivers. All throughput labels standardized to **GCUPS** (was a mix of
  GCPUS/GCUPS); stale hard-coded throughput annotations replaced with the
  measured ~4150 GCUPS + pointer to this file.

## Deferred (with rationale)
- #6 multi-warp blocks: implemented and tunable, measured a wash (8 warps/SM
  either way; not occupancy-bound). Default WPB=1.
- #7 `__ldg` / smaller TILE_K: implemented as a knob, measured a ~12×
  regression. Default off (register-resident).
- #8 remove host double-copy in `main_v5`: NOT done. It doesn't affect kernel
  GCUPS (transfers overlap compute across 4 streams), and correctness was the
  priority. The pinned-staging copy is intrinsic to async H2D; eliminating the
  prior `read_bin`→`memcpy` step would need `mmap`+`cudaHostRegister`, a
  host-side I/O change with no measured kernel-throughput benefit.

## How to reproduce
```
make validate && ./validate          # correctness, expect 25/25
make bench && ./bench 395 4437 65536  # throughput + occupancy at shipped defaults
# experiment: make bench TILELDG=1 TILEMINB=3 && ./bench ...  (will be slower)
```
