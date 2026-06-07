# cuDTW-Ada — Makefile (Phase 8 Final)
NVCC      = nvcc
ARCH      = -arch=sm_89
STD       = -std=c++20
OPT       = -O3 --maxrregcount=128 -lineinfo
OPT_P7    = -O3 --maxrregcount=192 -lineinfo
XFLAGS    = -Xcompiler='-fopenmp -march=native -O3 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0'
XFLAGS_SO = -Xcompiler='-fopenmp -march=native -O3 -fPIC -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0'

NVCCFLAGS    = $(ARCH) $(STD) $(OPT)    $(XFLAGS)
NVCCFLAGS_P7 = $(ARCH) $(STD) $(OPT_P7) $(XFLAGS)
NVCCFLAGS_SO = $(ARCH) $(STD) $(OPT_P7) $(XFLAGS_SO)

INCLUDES = -I src -I src/include

.PHONY: all clean bench validate validate_v2 validate_v3 dtw_main dtw_stream \
        validate_opt validate_fp16 libcudtw.so

all: validate validate_v2 validate_v3 dtw_main dtw_stream validate_opt libcudtw.so

# ── Phase 1: CPU-reference correctness validation (generic + tiled) ──────────
validate: tests/validate.cu \
          src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
          src/include/kernels/SHFL_FULLDTW_TILED.cuh \
          src/include/cudtw_constants.hpp \
          src/include/cudtw_scratch.hpp \
          src/include/cudtw_dispatcher_v2.hpp \
          src/include/cudtw_tiled.hpp \
          src/include/cudtw_dispatcher_v3.hpp
	@echo "[BUILD] validate (correctness, ~5-10 min)..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/validate.cu -o $@

# ── Phase 2: occupancy / throughput benchmark ────────────────────────────────
# Tunable occupancy knobs (override on the make command line), e.g.:
#   make bench GENWPB=8 TILEWPB=8 GENMINB=1 TILEMINB=2 REGS=128
GENWPB   ?= 8
TILEWPB  ?= 8
GENMINB  ?= 1
TILEMINB ?= 1
REGS     ?= 128
GENLDG   ?= 0
TILELDG  ?= 0
BENCHFLAGS = $(ARCH) $(STD) -O3 --maxrregcount=$(REGS) -lineinfo $(XFLAGS) \
             -DCUDTW_GENERIC_WARPS_PER_BLOCK=$(GENWPB) \
             -DCUDTW_TILED_WARPS_PER_BLOCK=$(TILEWPB) \
             -DCUDTW_GENERIC_MIN_BLOCKS=$(GENMINB) \
             -DCUDTW_TILED_MIN_BLOCKS=$(TILEMINB) \
             -DCUDTW_GENERIC_LDG_SUBJECT=$(GENLDG) \
             -DCUDTW_TILED_LDG_SUBJECT=$(TILELDG)
bench: tests/bench.cu \
       src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
       src/include/kernels/SHFL_FULLDTW_TILED.cuh \
       src/include/cudtw_constants.hpp \
       src/include/cudtw_scratch.hpp \
       src/include/cudtw_dispatcher_v2.hpp \
       src/include/cudtw_tiled.hpp \
       src/include/cudtw_dispatcher_v3.hpp
	@echo "[BUILD] bench (GENWPB=$(GENWPB) TILEWPB=$(TILEWPB) GENMINB=$(GENMINB) TILEMINB=$(TILEMINB) REGS=$(REGS) GENLDG=$(GENLDG) TILELDG=$(TILELDG))..."
	$(NVCC) $(BENCHFLAGS) $(INCLUDES) tests/bench.cu -o $@

# ── Phase 2 (~5-10 min) ───────────────────────────────────────────────────────
validate_v2: tests/validate_v2.cu \
             src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
             src/include/cudtw_dispatcher_v2.hpp
	@echo "[BUILD] validate_v2 (~5-10 min)..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@

# ── Phase 3 (~5-10 min) ───────────────────────────────────────────────────────
validate_v3: tests/validate_v3.cu \
             src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
             src/include/kernels/SHFL_FULLDTW_TILED.cuh \
             src/include/cudtw_dispatcher_v2.hpp \
             src/include/cudtw_tiled.hpp \
             src/include/cudtw_dispatcher_v3.hpp
	@echo "[BUILD] validate_v3 (~5-10 min)..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) tests/validate_v3.cu -o validate_v3

# ── Phase 4: sequential pipeline ─────────────────────────────────────────────
dtw_main: src/main_v4.cu \
          src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
          src/include/kernels/SHFL_FULLDTW_TILED.cuh \
          src/include/cudtw_dispatcher_v2.hpp \
          src/include/cudtw_tiled.hpp \
          src/include/cudtw_dispatcher_v3.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) src/main_v4.cu -o dtw_main
	@echo "[OK] dtw_main  →  ./dtw_main target/ sequences/ result/"

# ── Phase 5: streaming pipeline ───────────────────────────────────────────────
dtw_stream: src/main_v5.cu \
            src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
            src/include/kernels/SHFL_FULLDTW_TILED.cuh \
            src/include/cudtw_dispatcher_v2.hpp \
            src/include/cudtw_tiled.hpp \
            src/include/cudtw_dispatcher_v3.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) src/main_v5.cu -o dtw_stream
	@echo "[OK] dtw_stream  →  ./dtw_stream target/ sequences/ result/"

# ── Phase 7: Ada tuning benchmark ─────────────────────────────────────────────
validate_opt: tests/validate_opt.cu \
              src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
              src/include/kernels/SHFL_FULLDTW_OPT.cuh \
              src/include/cudtw_dispatcher_v2.hpp
	@echo "[BUILD] validate_opt (Phase 7, ~5-10 min)..."
	$(NVCC) $(NVCCFLAGS_P7) $(INCLUDES) tests/validate_opt.cu -o validate_opt

# ── Phase 8: Python shared library (~10-15 min) ───────────────────────────────
# Uses OPT kernels (Phase 7) + Tiled kernels (Phase 3)
libcudtw.so: src/cudtw_api.cu \
             src/include/kernels/SHFL_FULLDTW_GENERIC.cuh \
             src/include/kernels/SHFL_FULLDTW_OPT.cuh \
             src/include/kernels/SHFL_FULLDTW_TILED.cuh \
             src/include/cudtw_dispatcher_v2.hpp \
             src/include/cudtw_tiled.hpp \
             src/include/cudtw_dispatcher_v3.hpp
	@echo "[BUILD] libcudtw.so (Phase 8, ~10-15 min)..."
	$(NVCC) $(NVCCFLAGS_SO) $(INCLUDES) \
	    --shared -Xlinker=-soname,libcudtw.so \
	    src/cudtw_api.cu -o libcudtw.so
	@echo "[OK] libcudtw.so"
	@echo ""
	@echo "Python usage:"
	@echo "  export LD_LIBRARY_PATH=\$$(pwd):\$$LD_LIBRARY_PATH"
	@echo "  python3 python/cudtw.py"

# ── Regenerators ──────────────────────────────────────────────────────────────
gen_generic:
	python3 scripts/gen_generic_kernel.py > src/include/kernels/SHFL_FULLDTW_GENERIC.cuh
gen_tiled:
	python3 scripts/gen_tiled_kernel.py   > src/include/kernels/SHFL_FULLDTW_TILED.cuh

clean:
	rm -f validate bench validate_v2 validate_v3 dtw_main dtw_stream \
	      validate_opt validate_fp16 libcudtw.so
