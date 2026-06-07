# cuDTW-Ada

GPU-accelerated **Dynamic Time Warping (DTW)** distance computation, optimized for
NVIDIA **Ada Lovelace (sm_89)** GPUs such as the RTX 4090.

cuDTW-Ada computes the FullDTW distance between one **query** sequence and every
**subject** sequence in a dataset, using a warp-shuffle implementation in the
style of Schmidt et al. / hpc-dtw. Each warp processes one sequence entirely in
registers, with no shared-memory barriers on the inner recurrence.

- **Two compute paths**, chosen automatically by subject length:
  - `generic` for `L_subject ≤ 3839` (a single warp per sequence)
  - `tiled` for longer subjects (split into 2047-element tiles, stitched at the
    boundaries)
- **Three interfaces**: a sequential CLI, a 4-stream streaming CLI, and a shared
  library with a Python wrapper.
- **Code-generated kernels**: CUDA kernels for `K = 4..120` (where
  `K = ceil((L_subject+1)/32)`) are emitted by Python scripts, so the inner loop
  is fully unrolled for each size.

> Measured throughput: **~4150 GCUPS** (RTX 4090, K=64, kernel-only). The kernel
> is recurrence-latency-bound, not occupancy-bound — see [`RESULTS.md`](RESULTS.md)
> for the full performance analysis.

---

## Requirements

| Component | Version / Notes |
|-----------|-----------------|
| GPU | NVIDIA Ada Lovelace (sm_89), e.g. RTX 4090. Other architectures need an `ARCH` change in the `Makefile`. |
| CUDA Toolkit | 13.x (tested with 13.1). Provides `nvcc`. |
| Host compiler | C++20 (GCC 11+ / Clang 14+). |
| Python (optional) | 3.8+ with NumPy, only if you use the Python interface. |
| OS | Linux (tested on Ubuntu). |

Check your setup:

```bash
nvcc --version          # should report CUDA 13.x
nvidia-smi              # should list your GPU
```

If your GPU is **not** sm_89, edit `ARCH = -arch=sm_89` in the `Makefile` to match
(e.g. `-arch=sm_80` for A100, `-arch=sm_90` for H100).

---

## Data format

All sequences are stored as raw little-endian **float32** binary files. The
filename encodes the sequence length:

```
4437.bin   → each sequence is 4437 floats; the file holds N sequences back-to-back
             (file size must be a whole multiple of 4437 × 4 bytes)
```

Directory layout expected by the CLI tools and Python folder API:

```
target/      one file, e.g. 395.bin   → the query (a single sequence)
sequences/   one or more L.bin files  → the subjects to compare against the query
result/      output: result/L.bin     → N float32 DTW distances, same order as input
```

> **Note:** cuDTW-Ada assumes the data is **already normalized**. It does **not**
> apply z-normalization or any other preprocessing.

---

## Limits & supported lengths

| Quantity | Minimum | Maximum | Notes |
|----------|---------|---------|-------|
| **Query length** (`target`) | 1 | **8192** | Stored in GPU constant memory. Raise the max at build time with `-DCUDTW_MAX_QUERY=<n>` (hard ceiling 16384 = 64 KB constant memory). Exceeding the limit returns a clear error. |
| **Subject length** (`sequences`) | **32** | bounded only by VRAM | Kernels exist for `K = ceil((L_subject+1)/32)` in `[2, 120]`; `K ≥ 2` requires `L_subject ≥ 32`. Long subjects use the tiled path and have no length cap beyond available memory. |

**Why the 32-element minimum?** A subject shorter than 32 maps to `K = 1`, for
which no kernel is generated (a single warp would be mostly idle). The library
and CLI reject such inputs with an explicit error rather than producing wrong
results. If you need shorter subjects, extend `scripts/gen_generic_kernel.py` to
emit a `K = 1` kernel.

**Path selection** (automatic, by subject length):

| `L_subject` | Path |
|-------------|------|
| `32 … 3839` | `generic` (single warp per sequence) |
| `≥ 3840`    | `tiled` (2047-element tiles, stitched at boundaries) |

**Memory sizing.** Peak device memory is roughly
`N × L_subject × 4 bytes` for the subjects, plus padding/scratch buffers
(generic: `N × K×32 × 4`; tiled: one tile buffer `N × 2048 × 4` plus two boundary
buffers `N × L_query × 4`). For very large datasets the streaming CLI
(`dtw_stream`) automatically batches to fit in VRAM.

---

## Build

Clone and build the pieces you need:

```bash
git clone https://github.com/<your-username>/cuDTW-Ada.git
cd cuDTW-Ada
```

| Target | Produces | Use it for |
|--------|----------|------------|
| `make dtw_stream` | `./dtw_stream` | **Recommended CLI** — 4-stream streaming pipeline |
| `make dtw_main`   | `./dtw_main`   | Sequential CLI (simpler, single stream) |
| `make libcudtw.so`| `libcudtw.so`  | Shared library for the **Python** interface |
| `make validate`   | `./validate`   | Correctness check against a CPU reference |
| `make bench`      | `./bench`      | Throughput + occupancy benchmark |

Each kernel build can take several minutes because of the unrolled,
code-generated kernels. Example:

```bash
make dtw_stream      # build the streaming CLI
make libcudtw.so     # build the Python library (~10-15 min)
```

---

## Usage

### 1. Command line (recommended)

```bash
# ./dtw_stream <target_dir> <sequences_dir> <result_dir>
./dtw_stream target/ sequences/ result/
```

This reads the query from `target/`, computes DTW distances for every subject in
`sequences/`, and writes one result file per input file into `result/`. Per-file
timing and throughput are printed as it runs. `./dtw_main` takes the same
arguments.

### 2. Python

Build the library first (`make libcudtw.so`), then:

```python
import sys; sys.path.insert(0, 'python')
import cudtw
import numpy as np

cudtw.init(0)                              # select GPU 0

# Process an entire folder (matches the CLI workflow)
query = cudtw.load_bin('target/395.bin')
cudtw.set_query(query)
stats = cudtw.process_folder('sequences/', 'result/', verbose=True)
print(f"{stats['sequences']} sequences in {stats['wall_seconds']:.1f}s")
```

A ready-to-run example is included:

```bash
python3 runFile.py
```

#### In-memory arrays

If your data is already in NumPy rather than on disk:

```python
import cudtw, numpy as np

query    = np.random.randn(395).astype(np.float32)        # shape (L_query,)
subjects = np.random.randn(1000, 4437).astype(np.float32) # shape (N, L_subject)

cudtw.init(0)
distances = cudtw.compute(query, subjects)                # shape (N,)
```

Helpers: `cudtw.load_bin(path)` / `cudtw.save_bin(path, arr)` read and write the
float32 binary format, and `cudtw.compute_file(seq_path, L_subject, out_path)`
processes one file.

---

## Validate correctness

The tiled boundary logic is the trickiest part of the implementation, so a
CPU-reference validator is included. It compares GPU output against a plain CPU
FullDTW across the generic and tiled paths, including the exact tile-boundary
lengths and the real data lengths:

```bash
make validate
./validate
```

Expected: every case reports `PASS` and the final line reads
`RESULT: 25 passed, 0 failed`.

There is also an offline (no-GPU) algorithm simulator under `tests/sim/` that
emulates the warp-shuffle kernels in NumPy, useful for verifying boundary logic
without compiling CUDA.

### Independent cross-validation (recommended)

`validate` checks against this project's own CPU reference. To rule out a shared
bug, two scripts also cross-check against the **independent** `dtaidistance`
library (a separately-authored DTW implementation):

```bash
pip install dtaidistance numpy

# Offline: kernel simulator vs dtaidistance, wide length sweep (no GPU needed)
python3 tests/cross_validate.py

# On GPU: the real compiled kernel vs dtaidistance (needs make libcudtw.so)
python3 tests/gpu_cross_validate.py
```

Convention: `dtaidistance.distance(q,s)**2` equals this project's distance
(both are constrained FullDTW with squared cost; the project omits the final
square root). The offline sweep covers every generic K-boundary, every tile
boundary, the real data lengths, and multiple query lengths/seeds — all agree
to ~1e-6 relative error.

---

## Benchmark

```bash
make bench
./bench 395 4437 65536        # Lq Ls N
```

`bench` reports kernel-only GCUPS and theoretical occupancy. Occupancy tuning
knobs can be set at build time (see the comments in the generator scripts and
`RESULTS.md`); the shipped defaults are the fastest measured configuration:

```bash
# Example: experiment with multi-warp blocks (does not help here — see RESULTS.md)
make bench TILEWPB=8 && ./bench 395 4437 65536
```

> The `bench` timer measures **kernel only** (no transfers); the CLI tools'
> GCUPS includes H2D/D2H copies. Compare bench-to-bench and CLI-to-CLI, not
> across the two.

---

## Project layout

```
cuDTW-Ada/
├── src/
│   ├── main_v4.cu              # sequential CLI (dtw_main)
│   ├── main_v5.cu              # 4-stream streaming CLI (dtw_stream)
│   ├── cudtw_api.cu            # shared-library C API (libcudtw.so)
│   └── include/
│       ├── cudtw_dispatcher_v2.hpp   # generic-path dispatch + padding
│       ├── cudtw_tiled.hpp           # tiled-path dispatch + tile schedule
│       ├── cudtw_dispatcher_v3.hpp   # unified entry point (dist_any)
│       ├── cudtw_scratch.hpp         # pre-allocated per-stream buffers
│       ├── cudtw_constants.hpp       # shared limits (max query length)
│       ├── binary_IO.hpp, hpc_helpers.hpp
│       └── kernels/                  # AUTO-GENERATED kernels (do not edit)
├── scripts/
│   ├── gen_generic_kernel.py   # generates SHFL_FULLDTW_GENERIC.cuh
│   └── gen_tiled_kernel.py     # generates SHFL_FULLDTW_TILED.cuh
├── python/cudtw.py             # Python wrapper around libcudtw.so
├── tests/
│   ├── validate.cu             # CPU-reference correctness test
│   ├── bench.cu                # throughput / occupancy benchmark
│   └── sim/                    # offline (no-GPU) kernel simulator
├── target/  sequences/  result/   # data folders
├── runFile.py                  # example Python script
├── RESULTS.md                  # fixes + performance analysis
└── Makefile
```

### Regenerating kernels

The files in `src/include/kernels/` are **auto-generated** and marked
`DO NOT EDIT`. To change kernel logic, edit the generator scripts and regenerate:

```bash
make gen_generic     # rewrites SHFL_FULLDTW_GENERIC.cuh
make gen_tiled       # rewrites SHFL_FULLDTW_TILED.cuh
```

---

## Troubleshooting

**`libcudtw.so not found`** — build it first with `make libcudtw.so`. The Python
wrapper looks for the file in the project root.

**`unsupported gpu architecture 'compute_89'`** — your CUDA Toolkit is too old
for Ada, or your GPU is a different architecture. Update CUDA, or change `ARCH`
in the `Makefile` to match your GPU.

**`file holds X floats, which is not a whole multiple of L_subject`** — the input
file size doesn't match the length encoded in its filename. Check that
`<name>.bin` really contains a whole number of `name`-length float32 sequences.

**`L_query out of range`** — the query exceeds the compile-time maximum
(`CUDTW_MAX_QUERY`, default 8192). Increase it by building with
`-DCUDTW_MAX_QUERY=<n>` (≤ 16384) or shorten the query.

**`L_subject=<n> is below the minimum supported length of 32`** — subjects must
be at least 32 elements (see [Limits](#limits--supported-lengths)). Pad shorter
subjects to ≥ 32, or extend the kernel generator for `K = 1`.

---

## License

Add your chosen license here (e.g. MIT). Create a `LICENSE` file in the repo root.

## Acknowledgements

The warp-shuffle FullDTW approach follows the design of Schmidt et al. (hpc-dtw).
