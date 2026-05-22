### <img src="/images/banner.png" width=900/>
### ```[Matador FPGA]```: Automated Data Bandwith Driven Logic Based Inference

Matador is a tool for training and translating Coalesced Tsetlin Machines (CoTM) into FPGA accelerator systems. It converts learnt clause propositions into custom hard coded fast inference circuits which are then streamed inference data from the system's processor. The tool's configurable design and debug options allow for rapid prototyping of CoTM applications to meet target performance and resource requirements. The Matador approach offers competitive performance in terms of latency, logic utilization and power consumption when compared against its nearest Binary Neural Network counterparts. The core concepts of Matador are presented in the two figures below:   
### <img src="/images/promotional.png" width=900/>
### <img src="/images/flow.png" width=900/>

## Learn about Matador

The core papers for learning about Matador are listed here: 

1) Tsetlin Machine original paper : https://arxiv.org/abs/1804.01508
2) Coalesced Tsetlin Machine paper: https://arxiv.org/abs/2108.07594
3) Hardware Tsetlin Machines paper: https://royalsocietypublishing.org/doi/epdf/10.1098/rsta.2019.0593
4) Tsetlin Machine CAIR github    : https://github.com/cair/tmu

Matador's training engine comes from the TMU repo: https://github.com/cair/tmu [```914e099```]. 
While the repo incorporates all the Tsetlin Machine models, Matador currently supports Vanilla (Tsetlin Machine) and Coalesced Tsetlin Machine. 

## Build Matador

### Requirements

- Linux host (Ubuntu 20.04 recommended)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Xilinx Vivado 2022.2 or later, installed on the host
- X11 display server (required for the GUI)

### Quick Start

Clone the repo and run the build script from the repo root:

```bash
git clone git@github.com:nclaes/matador.git
cd matador
./Build_Matador
```

This builds the Docker image and launches the container. The repo directory is automatically mounted at `/app` inside the container.

### Custom Vivado Path

By default the script expects Vivado to be under `/tools/Xilinx`. If your installation is elsewhere, set `VIVADO_DIR` before running:

```bash
VIVADO_DIR=/opt/Xilinx ./Build_Matador
```

### Custom X11 Auth

If your X authority file is not at `$XAUTHORITY`, set it explicitly:

```bash
XAUTH=/run/user/1000/gdm/Xauthority ./Build_Matador
```

### Running Matador inside the container

Once inside the container, launch the GUI with:

```bash
./Matador
```

For full build details see `Build_Instructions.md`.

## Contact

Matador was developed by Tousif Rahman and Gang Mao in the Microsystems Group at Newcastle University. 

If there are issues with the setup or the usage please contact: 
Tousif Rahman: tousifsrahman@gmail.com

## Matador Video Library 

These videos present the Matador's flow from previous releases. The flow remains unchanged so it may be useful for prospective users. 

### Matador Overview (16-07-2023)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-1EmmJNy2wA/0.jpg)](https://www.youtube.com/watch?v=-1EmmJNy2wA)
### Example Walkthrough (development branch 16-07-2023)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mXH5gZGkKiQ/0.jpg)](https://www.youtube.com/watch?v=mXH5gZGkKiQ)

