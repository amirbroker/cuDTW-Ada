"""
cuDTW-Ada | Phase 8 | python/cudtw.py

Python API for GPU-accelerated DTW via ctypes.

Install:
    cd ~/cuDTW-Ada && make libcudtw.so

Usage:
    import sys; sys.path.insert(0, 'python')
    import cudtw

    # Initialize
    cudtw.init(device=0)
    print(cudtw.info())

    # Simple usage
    query    = np.array([...], dtype=np.float32)   # L floats
    subjects = np.array([...], dtype=np.float32)   # N × L floats
    results  = cudtw.compute(query, subjects)       # N floats

    # Process entire folder
    cudtw.set_query(query)
    cudtw.process_folder('sequences/', 'result/')
"""

import ctypes
import os
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import time

# ── Load shared library ────────────────────────────────────────────────────────
def _find_lib():
    search = [
        Path(__file__).parent.parent / 'libcudtw.so',
        Path.cwd() / 'libcudtw.so',
    ]
    for p in search:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "libcudtw.so not found. Run 'make libcudtw.so' in ~/cuDTW-Ada first.")

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_lib())
        # dtw_init
        _lib.dtw_init.restype  = ctypes.c_int
        _lib.dtw_init.argtypes = [ctypes.c_int]
        # dtw_last_error
        _lib.dtw_last_error.restype  = ctypes.c_char_p
        _lib.dtw_last_error.argtypes = []
        # dtw_info
        _lib.dtw_info.restype  = None
        _lib.dtw_info.argtypes = [ctypes.c_char_p, ctypes.c_int]
        # dtw_set_query
        _lib.dtw_set_query.restype  = ctypes.c_int
        _lib.dtw_set_query.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        # dtw_compute
        _lib.dtw_compute.restype  = ctypes.c_int
        _lib.dtw_compute.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)]
        # dtw_compute_file
        _lib.dtw_compute_file.restype  = ctypes.c_longlong
        _lib.dtw_compute_file.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
        # dtw_process_folder
        _PROGRESS_CB = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int, ctypes.c_float)
        _lib.dtw_process_folder.restype  = ctypes.c_int
        _lib.dtw_process_folder.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, _PROGRESS_CB]
        _lib._progress_cb_type = _PROGRESS_CB
    return _lib

# ── Public API ────────────────────────────────────────────────────────────────

def init(device: int = 0) -> str:
    """Initialize CUDA device. Returns info string."""
    lib = _get_lib()
    rc = lib.dtw_init(device)
    msg = lib.dtw_last_error().decode()
    if rc != 0:
        raise RuntimeError(f"dtw_init failed: {msg}")
    return msg

def info() -> str:
    """Return GPU info string."""
    lib = _get_lib()
    buf = ctypes.create_string_buffer(256)
    lib.dtw_info(buf, 256)
    return buf.value.decode()

def set_query(query: np.ndarray) -> None:
    """Upload query sequence to GPU constant memory."""
    lib = _get_lib()
    q = np.ascontiguousarray(query, dtype=np.float32)
    rc = lib.dtw_set_query(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(q))
    if rc != 0:
        raise RuntimeError(lib.dtw_last_error().decode())

def compute(
    query:    np.ndarray,
    subjects: np.ndarray
) -> np.ndarray:
    """
    Compute DTW distances between query and all subjects.

    Args:
        query:    1D float32 array of length L_query
        subjects: 2D float32 array of shape (N, L_subject)
                  OR 1D array of length N*L_subject (will be inferred)
    Returns:
        1D float32 array of shape (N,) with DTW distances
    """
    lib = _get_lib()

    q = np.ascontiguousarray(query, dtype=np.float32)
    s = np.ascontiguousarray(subjects, dtype=np.float32)

    if s.ndim == 1:
        raise ValueError("subjects must be 2D array of shape (N, L_subject)")
    N, L_sub = s.shape

    # Upload query
    rc = lib.dtw_set_query(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(q))
    if rc != 0:
        raise RuntimeError(lib.dtw_last_error().decode())

    results = np.empty(N, dtype=np.float32)

    rc = lib.dtw_compute(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, L_sub,
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if rc != 0:
        raise RuntimeError(lib.dtw_last_error().decode())

    return results

def compute_file(
    sequences_path: str,
    L_subject:      int,
    result_path:    str
) -> int:
    """
    Process one binary sequences file.
    Returns number of sequences processed.
    """
    lib = _get_lib()
    n = lib.dtw_compute_file(
        sequences_path.encode(),
        L_subject,
        result_path.encode())
    if n < 0:
        raise RuntimeError(lib.dtw_last_error().decode())
    return int(n)

def process_folder(
    sequences_dir: str,
    result_dir:    str,
    verbose:       bool = True,
    callback:      Optional[Callable] = None
) -> dict:
    """
    Process entire sequences/ folder → result/ folder.

    Args:
        sequences_dir: path to folder with L.bin files
        result_dir:    path for output
        verbose:       print progress per file
        callback:      optional fn(filename, N, gcpus) called per file

    Returns:
        dict with keys: files, sequences, wall_seconds, avg_gcpus
    """
    lib = _get_lib()

    stats = {'files': 0, 'sequences': 0, 'wall_seconds': 0.0, 'avg_gcpus': 0.0}
    total_cells = 0.0
    t_wall = time.perf_counter()

    PROGRESS_CB = lib._progress_cb_type

    def _cb(fname_b, N, gcpus):
        fname = fname_b.decode()
        stats['files']     += 1
        stats['sequences'] += N
        if verbose:
            print(f"  {fname:20s} N={N:8d} | {gcpus:7.1f} GCPUS")
        if callback:
            callback(fname, N, gcpus)

    rc = lib.dtw_process_folder(
        sequences_dir.encode(),
        result_dir.encode(),
        PROGRESS_CB(_cb))

    stats['wall_seconds'] = time.perf_counter() - t_wall
    if rc < 0:
        raise RuntimeError(lib.dtw_last_error().decode())
    stats['files'] = rc

    if verbose and stats['wall_seconds'] > 0:
        print(f"\n  Files: {stats['files']}  "
              f"Wall: {stats['wall_seconds']:.2f}s  "
              f"Seq/s: {stats['sequences']/stats['wall_seconds']:.0f}")
    return stats


# ── Convenience: load binary file ────────────────────────────────────────────
def load_bin(path: str) -> np.ndarray:
    """Load a raw float32 binary file."""
    return np.fromfile(path, dtype=np.float32)

def save_bin(path: str, arr: np.ndarray) -> None:
    """Save array as raw float32 binary file."""
    arr.astype(np.float32).tofile(path)


# ── Example / quick test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("cuDTW-Ada Python API — quick test")
    print("Initializing GPU...")
    print(init(0))
    print("GPU:", info())
    print()

    # Synthetic test
    import numpy as np
    rng = np.random.default_rng(42)
    Lq, Ls, N = 395, 820, 1000

    query    = rng.uniform(-5, 5, Lq).astype(np.float32)
    subjects = rng.uniform(-5, 5, (N, Ls)).astype(np.float32)

    print(f"Computing DTW: query={Lq}, subjects={N}×{Ls}...")
    t0 = time.perf_counter()
    results = compute(query, subjects)
    t1 = time.perf_counter()

    gcpus = Lq * Ls * N / (t1-t0) / 1e9
    print(f"  Results shape: {results.shape}")
    print(f"  Min dist: {results.min():.2f}  Max: {results.max():.2f}")
    print(f"  Time: {(t1-t0)*1000:.1f} ms  |  {gcpus:.0f} GCPUS")
    print()
    print("✓ API working correctly")
