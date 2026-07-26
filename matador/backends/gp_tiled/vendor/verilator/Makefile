# Verilator build for tm_accel_gp (vanilla_gp_tiled backend).
# Copied verbatim by GPTiledBackend.generate() into RTL/sim/verilator/ --
# no per-generate()-config templating needed (Verilator bakes the already-
# substituted RTL parameters into the generated C++ model itself).
#
# Portable: SCRIPT_DIR is computed from the Makefile's own location at
# make-time, so this directory can be copied or moved without editing paths.
# obj_dir is always deleted before Verilator runs, to avoid a generated
# Vtm_accel_gp.mk baking in a stale absolute path from a previous build
# environment (e.g. a build done on a different machine).
unexport VERILATOR_ROOT

VERILATOR  ?= verilator
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_DIR    := $(SCRIPT_DIR)../../src
OBJ_DIR    := $(SCRIPT_DIR)obj_dir
BIN        := $(OBJ_DIR)/Vtm_accel_gp

# On macOS: if the default verilator is <5.0, try Homebrew's copy.
ifeq ($(shell uname -s),Darwin)
  _VER := $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
  ifeq ($(filter 5 6 7 8 9,$(_VER)),)
    _HB := /opt/homebrew/bin/verilator
    _HB_VER := $(shell env -u VERILATOR_ROOT $(_HB) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
    ifneq ($(filter 5 6 7 8 9,$(_HB_VER)),)
      VERILATOR := $(_HB)
    endif
  endif
endif

# Fail fast with a readable message if Verilator is still too old.
_VER_FINAL := $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
ifeq ($(filter 5 6 7 8 9,$(_VER_FINAL)),)
  $(error Verilator >=5.0 required (found: $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | head -1 || echo not found)). Set VERILATOR=/path/to/verilator5)
endif

VERILATOR := env -u VERILATOR_ROOT $(VERILATOR)

VSRCS := \
    $(SRC_DIR)/axis_fifo.v     \
    $(SRC_DIR)/clause_eval.v   \
    $(SRC_DIR)/tile_mem.v      \
    $(SRC_DIR)/score_acc_rt.v  \
    $(SRC_DIR)/argmax_rt.v     \
    $(SRC_DIR)/tm_accel_gp.v

VFLAGS := \
    --cc --exe --build --trace-fst \
    --language 1800-2005            \
    --top-module tm_accel_gp        \
    --Mdir $(OBJ_DIR)               \
    --CFLAGS "-std=c++17"           \
    -Wall --Wno-TIMESCALEMOD --Wno-WIDTH --Wno-BLKSEQ --Wno-UNUSEDSIGNAL

.PHONY: all run clean

all: $(BIN)

$(BIN): $(VSRCS) $(SCRIPT_DIR)tb_top.cpp
	rm -rf $(OBJ_DIR)
	$(VERILATOR) $(VFLAGS) $(VSRCS) $(SCRIPT_DIR)tb_top.cpp

# run always rebuilds cleanly to prevent stale-binary issues across
# environments (e.g. macOS build running on Linux container). Replays the
# model-specific vectors by default (tb_top's own --stim/--exp defaults);
# pass ARGS="--stim ... --exp ..." to replay something else, e.g. a
# gen_vectors.py sequence output.
run: clean all
	@cd $(SCRIPT_DIR) && $(BIN) $(ARGS)

clean:
	rm -rf $(OBJ_DIR)
