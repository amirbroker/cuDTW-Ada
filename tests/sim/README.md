# Offline (no-GPU) algorithm validator

These Python files bit-faithfully emulate the generated warp-shuffle CUDA
kernels (32 lanes, __shfl_up/down/sync, the same min/cost arithmetic) so the
DTW algorithm — especially the tiled tile-boundary stitching — can be validated
without a GPU or nvcc.

- `sim.py`        : generic-path kernel simulator + CPU DTW references
- `sim_tiled.py`  : tiled-path kernel + host stitching simulator (matches the
                    FIXED gen_tiled_kernel.py)

The on-GPU equivalent is `tests/validate.cu` (`make validate`), which checks the
real kernels against a CPU reference on the same boundary lengths.
