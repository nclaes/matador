# =============================================================================
# Matador — top-level Makefile
#
# All build/test/lint/sim targets run inside the dev container against the
# bind-mounted workspace. GTKWave and the initial docker build run on the host.
#
# Usage:
#   make build              Build the dev Docker image
#   make shell              Drop into an interactive shell in the container
#   make test               Run pytest
#   make lint               Python lint (ruff) + RTL lint (verilator)
#   make sim  TARGET=<name> Run a named simulation
#   make waves DIR=<path>   Open GTKWave on the HOST for a result directory
#   make clean              Remove generated artefacts
#   make ci-local           Full CI suite (test + lint) in one shot
# =============================================================================

IMAGE   := matador:dev
UID     := $(shell id -u)
GID     := $(shell id -g)

# Optional: mount a host directory as /work inside the container.
# Set before any target, e.g.:  make shell WORK_DIR=/path/to/mydata
WORK_DIR    ?=
WORK_MOUNT   = $(if $(WORK_DIR),-v "$(WORK_DIR):/work",)

# Interactive container (tty + stdin) — used for `make shell`
DOCKER_RUN := docker run --rm -it \
                --hostname matador \
                --network=host \
                -v "$(PWD):/workspace" \
                $(WORK_MOUNT) \
                -u $(UID):$(GID) \
                $(IMAGE)

# Non-interactive one-shot container — used for test/lint/ci-local
DOCKER_EXEC := docker run --rm \
                 --hostname matador \
                 -v "$(PWD):/workspace" \
                 $(WORK_MOUNT) \
                 -u $(UID):$(GID) \
                 $(IMAGE)

# -----------------------------------------------------------------------------

.DEFAULT_GOAL := help
.PHONY: help build shell tmu-build test lint sim waves clean ci-local

help:
	@printf '\n  \033[1mMatador — available targets\033[0m\n\n'
	@printf '  %-30s %s\n' 'make build'                   'Build the dev Docker image'
	@printf '  %-30s %s\n' 'make shell'                   'Interactive shell (dev@matador:/workspace)'
	@printf '  %-30s %s\n' 'make shell WORK_DIR=<path>'   'Shell with <path> mounted as /work'
	@printf '  %-30s %s\n' 'make tmu-build'               'Compile the tmu C extension (run once after build)'
	@printf '  %-30s %s\n' 'make test'                    'Run pytest in the container'
	@printf '  %-30s %s\n' 'make lint'                    'ruff check/format + verilator --lint-only'
	@printf '  %-30s %s\n' 'make sim TARGET=<name>'       'Run a named simulation'
	@printf '  %-30s %s\n' 'make waves DIR=<path>'        'Open GTKWave on the HOST'
	@printf '  %-30s %s\n' 'make clean'                   'Remove obj_dir, *.vcd, *.fst, __pycache__'
	@printf '  %-30s %s\n' 'make ci-local'                'Full CI suite in the container'
	@printf '\n'

build:
	docker build \
	  --build-arg UID=$(UID) \
	  --build-arg GID=$(GID) \
	  -t $(IMAGE) \
	  -f docker/Dockerfile \
	  .

shell:
	$(DOCKER_RUN) /bin/bash

# Compile the tmu C extension inside the container.
# Must be run once after `make build` — output goes to tmu/tmulib.cpython-*.so
# The .so is gitignored (platform/Python-version specific).
tmu-build:
	rm /workspace/tmu/tmulib.cpython-312-aarch64-linux-gnu.so
	bash -c "python3 tmu/lib/tmulib_extension_build.py"

test:
	$(DOCKER_EXEC) pytest

lint:
	$(DOCKER_EXEC) sh -c '\
	  ruff check . && \
	  ruff format --check . && \
	  find rtl/accelerators/vanilla -name "*.sv" 2>/dev/null | \
	    xargs -r verilator --lint-only --Wall -sv'

sim:
ifndef TARGET
	$(error TARGET is not set — usage: make sim TARGET=<name>)
endif
	$(DOCKER_EXEC) sh -c 'echo "[sim] placeholder: $(TARGET)"'

# GTKWave runs on the HOST — waveform files are in the bind-mounted workspace
waves:
ifndef DIR
	$(error DIR is not set — usage: make waves DIR=<path/to/results>)
endif
	gtkwave --rcvar 'fontname_signals Monospace 10' $(DIR)

clean:
	find . -type d -name 'obj_dir'     -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true
	find . \( -name '*.vcd' -o -name '*.fst' \) -delete 2>/dev/null; true

ci-local:
	$(DOCKER_EXEC) sh -c 'pytest && ruff check . && ruff format --check .'
