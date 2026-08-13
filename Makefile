# =============================================================================
# coal_tm — top-level Makefile
#
# Works both on the HOST (wraps each target in `docker run`) and INSIDE
# the dev container after `make shell` (runs the same command directly --
# there's no `docker` binary in there, and no need for one).
#
# Usage:
#   make build              Build the dev Docker image           [HOST only]
#   make shell               Interactive shell (dev@coal-tm:/workspace) [HOST only]
#   make shell WORK_DIR=...  Shell with a host directory mounted as /work
#   make tmu-build           Compile the vendored tmu C extension
#   make test                Run pytest
#   make sim RTL_DIR=<path>  Run a generated testbench (iverilog by default)
#   make waves DIR=<path>    Open GTKWave on the HOST for a result directory
#   make clean               Remove generated artefacts
# =============================================================================

IMAGE   := coal_tm:dev
UID     := $(shell id -u)
GID     := $(shell id -g)

WORK_DIR    ?=
WORK_MOUNT   = $(if $(WORK_DIR),-v "$(WORK_DIR):/work",)

DOCKER_RUN := docker run --rm -it \
                --hostname coal-tm \
                -v "$(PWD):/workspace" \
                $(WORK_MOUNT) \
                -u $(UID):$(GID) \
                $(IMAGE)

DOCKER_EXEC := docker run --rm \
                 --hostname coal-tm \
                 -v "$(PWD):/workspace" \
                 $(WORK_MOUNT) \
                 -u $(UID):$(GID) \
                 $(IMAGE)

# Every Docker container has /.dockerenv (created by the engine itself) --
# use it to tell whether `make` is already running inside the dev
# container, so tmu-build/test/sim skip the docker wrapper and just run
# the underlying command directly there, instead of failing on a missing
# `docker` binary.
IN_CONTAINER := $(shell test -f /.dockerenv && echo 1)
ifeq ($(IN_CONTAINER),1)
RUN :=
else
RUN := $(DOCKER_EXEC)
endif

.DEFAULT_GOAL := help
.PHONY: help build shell tmu-build test sim waves clean

help:
	@printf '\n  \033[1mcoal_tm — available targets\033[0m\n\n'
	@printf '  %-30s %s\n' 'make build'                   'Build the dev Docker image (HOST only)'
	@printf '  %-30s %s\n' 'make shell'                   'Interactive shell (dev@coal-tm:/workspace) (HOST only)'
	@printf '  %-30s %s\n' 'make shell WORK_DIR=<path>'   'Shell with <path> mounted as /work'
	@printf '  %-30s %s\n' 'make tmu-build'               'Compile the vendored tmu C extension'
	@printf '  %-30s %s\n' 'make test'                    'Run pytest'
	@printf '  %-30s %s\n' 'make sim RTL_DIR=<path>'      'Run a generated testbench (iverilog)'
	@printf '  %-30s %s\n' 'make sim RTL_DIR=<path> VERILATOR=1' 'Same, under Verilator'
	@printf '  %-30s %s\n' 'make waves DIR=<path>'        'Open GTKWave on the HOST'
	@printf '  %-30s %s\n' 'make clean'                   'Remove obj_dir, *.vcd, *.vvp, __pycache__'
	@printf '\n  tmu-build/test/sim work both on the host and inside the\n'
	@printf '  container (after make shell) -- no need to re-wrap in Docker.\n\n'

build:
ifeq ($(IN_CONTAINER),1)
	@echo "error: 'make build' runs docker build -- run it on the HOST, not inside the container." >&2
	@exit 1
endif
	docker build \
	  --build-arg UID=$(UID) \
	  --build-arg GID=$(GID) \
	  -t $(IMAGE) \
	  -f docker/Dockerfile \
	  .

shell:
ifeq ($(IN_CONTAINER),1)
	@echo "error: already inside the container -- just run commands directly, no need for 'make shell' again." >&2
	@exit 1
endif
	$(DOCKER_RUN) /bin/bash

# Compile the tmu C extension. Run once after `make build` (or once inside
# a freshly built container) — the committed .so is pinned to a specific
# Python/platform and won't load elsewhere. Output goes to
# utils/tmu/tmulib.cpython-*.so (gitignored).
tmu-build:
	$(RUN) python3 utils/tmu/lib/tmulib_extension_build.py

test:
	$(RUN) python3 -m pytest -q

sim:
	@if [ -z "$(RTL_DIR)" ]; then echo "usage: make sim RTL_DIR=<path> [VERILATOR=1]"; exit 1; fi
	$(RUN) python3 -m coal_tm.cli sim --rtl-dir "$(RTL_DIR)" $(if $(VERILATOR),--verilator,)

waves:
	@if [ -z "$(DIR)" ]; then echo "usage: make waves DIR=<path>"; exit 1; fi
	gtkwave $(DIR)/*.vcd &

clean:
	find . -name "obj_dir" -type d -prune -exec rm -rf {} +
	find . -name "*.vcd" -delete
	find . -name "*.vvp" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
