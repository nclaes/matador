<img src="/images/banner.png" width=900/>

# Matador — Automated FPGA Accelerator Design for Tsetlin Machines

Matador trains Coalesced Tsetlin Machines (CoTM) and translates them directly into FPGA accelerator systems. Learnt clause propositions are converted into hard-coded inference circuits, which are then streamed data from the host processor over AXI. The tool's configurable design and debug options allow rapid prototyping to hit target performance and resource requirements.

<img src="/images/promotional.png" width=900/>
<img src="/images/flow.png" width=900/>

---

## Getting Started

```bash
make build        # build the dev container (one-off)
make shell        # drop into an interactive shell inside the container
pytest            # run the test suite (inside the shell)
```

GTKWave runs on the **host**, not in the container:

```bash
make waves DIR=path/to/results
```

See `make help` for the full list of targets.

---

## Background Reading

| | Paper |
|---|---|
| 1 | [Tsetlin Machine](https://arxiv.org/abs/1804.01508) |
| 2 | [Coalesced Tsetlin Machine](https://arxiv.org/abs/2108.07594) |
| 3 | [Hardware Tsetlin Machines](https://royalsocietypublishing.org/doi/epdf/10.1098/rsta.2019.0593) |
| 4 | [Matador paper](https://arxiv.org/abs/2403.10538) |

Training uses the [TMU library](https://github.com/cair/tmu) (pinned at `914e099`). Matador currently supports the Vanilla TM and Coalesced TM models.

---

## Build

### Requirements

- Linux host (Ubuntu 20.04 recommended)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Xilinx Vivado 2022.2 or later installed on the host
- X11 display server (required for the GUI)

### Quick Start

```bash
git clone git@github.com:nclaes/matador.git
cd matador
./Build_Matador
```

`Build_Matador` builds the Docker image and launches the container. The repo directory is mounted at `/app` inside the container; Vivado is passed through from the host.

### Configuration

The script accepts two environment variable overrides:

| Variable | Default | Purpose |
|---|---|---|
| `VIVADO_DIR` | `/tools/Xilinx` | Root of the Xilinx install on the host |
| `XAUTH` | `$XAUTHORITY` | X11 authority file for GUI forwarding |

```bash
VIVADO_DIR=/opt/Xilinx ./Build_Matador
```

### Docker flags reference

| Flag | Purpose |
|---|---|
| `--network=host` | Share host network stack |
| `-e DISPLAY` | Forward host display variable |
| `-v VIVADO_DIR` | Mount Vivado tools into the container |
| `-v $(pwd):/app` | Mount the Matador repo |
| `-v XAUTH` | Share X11 auth file for GUI |

> **Note:** Matador is a research tool. Build instructions may need adapting for non-standard system configurations.

---

## Usage

Once inside the container, launch the GUI:

```bash
./Matador
```

The GUI walks through the full flow:

1. **Setup** — set the output directory and Vivado path
2. **Train Model** — configure and train a TM using the TMU backend
3. **Generate RTL** — convert the trained model into synthesisable SystemVerilog
4. **Synth + Impl** — run Vivado synthesis and implementation to produce a bitstream
5. **Deploy** — deploy the bitstream to the target board (PYNQ-Z1)

---

## Contact

Developed by Tousif Rahman and Gang Mao, Microsystems Group, Newcastle University.

Issues and questions: tousifsrahman@gmail.com

---

## Video Walkthroughs

### Overview (2023-07-16)
[![Matador Overview](https://img.youtube.com/vi/-1EmmJNy2wA/0.jpg)](https://www.youtube.com/watch?v=-1EmmJNy2wA)

### Example Walkthrough (2023-07-16)
[![Example Walkthrough](https://img.youtube.com/vi/mXH5gZGkKiQ/0.jpg)](https://www.youtube.com/watch?v=mXH5gZGkKiQ)

