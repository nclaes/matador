# Setup

## Requirements

- Docker (daemon running)
- `make`
- macOS or Linux host

## Build the container

```bash
git clone git@github.com:nclaes/matador.git
cd matador

make build                          # build the dev container (one-off, ~5 min)
make tmu-build                      # compile the tmu C extension (one-off)
```

The container is `ubuntu:24.04` with Python 3.12, Verilator 5, iverilog, GTKWave, and all Python dependencies pre-installed.

## Enter the container

```bash
make shell WORK_DIR=/path/to/your/data
```

`/path/to/your/data` is mounted as `/work` inside the container.  All Matador output (TMIR models, RTL, reports) is written there.

## Verify the setup

```bash
# Inside the container
matador version          # prints version string
matador                  # splash screen + workspace status
matador list-backends    # lists registered RTL backends
```

## GTKWave (waveform viewer)

GTKWave runs on the **host**, not inside the container:

```bash
# From the host (after simulation has produced a .vcd or .fst)
make waves DIR=/path/to/your/data/<backend>/RTL/sim
```

Or use the generated `waves.sh` inside the container:

```bash
bash /work/<backend>/RTL/sim/waves.sh tb_hw_system
```

## Make targets

```
make build                  Build the dev container
make tmu-build              Compile tmu C extension inside the container
make shell                  Interactive shell (dev@matador:/workspace)
make shell WORK_DIR=<path>  Shell with <path> mounted as /work
make test                   Run pytest
make lint                   ruff + Verilator lint
make clean                  Remove build artefacts
```
