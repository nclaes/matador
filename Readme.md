<img src="/images/banner.png" width=900/>

# Matador — Automated FPGA Accelerator Design for Tsetlin Machines

Matador trains Tsetlin Machines and Coalesced Tsetlin Machines and generates RTL. 
This RTL can be deployed on FPGAs (specifically Xilinx SoCs) or simulated using verilator and GTKWave. 

<img src="/images/promotional.png" width=900/>
<img src="/images/flow.png" width=900/>

---

## Getting Started

First make sure you have Docker installed and the docker daemon is running. 
You need to make a directory outside of this repo for all your outputs. 
This directory will be mounted in the Docker container. 

```bash
make help         # see all available targets
make build        # build the dev container (one-off)
make shell        # open a shell → dev@matador:/workspace
```
To mount a local data or results directory as `/work` inside the container:

```bash
make shell WORK_DIR=/path/to/your/data
```

GTKWave runs on the **host**, not in the container:

```bash
make waves DIR=path/to/results
```

See `make help` for the full list of targets.

### Build and Verify Setup

```bash 
make build 
make shell WORK_DIR=/path/to/your/data
# Now you should be inside the container dev@matador:/workspace$
make tmu-build
which matador 
matador version 
matador
```
From here you can use matador commands to train your model and generate the RTL. It is recommended you begin with `matador faena`

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

## Contact

Developed by Tousif Rahman and Gang Mao, Microsystems Group, Newcastle University.

Issues and questions: tousifsrahman@gmail.com

---

## Video Walkthroughs

### Overview (2023-07-16)
[![Matador Overview](https://img.youtube.com/vi/-1EmmJNy2wA/0.jpg)](https://www.youtube.com/watch?v=-1EmmJNy2wA)

### Example Walkthrough (2023-07-16)
[![Example Walkthrough](https://img.youtube.com/vi/mXH5gZGkKiQ/0.jpg)](https://www.youtube.com/watch?v=mXH5gZGkKiQ)

