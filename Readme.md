# <img src="/images/banner.png" width=900/>
### Matador: Automated Data Bandwith Driven Logic Based Inference

Matador is a tool for training and translating Coalesced Tsetlin Machines (CoTM) into FPGA accelerator systems. It converts learnt clause propositions into custom hard coded fast inference circuits which are then streamed inference data from the system's processor. The tool's configurable design and debug options allow for rapid prototyping of CoTM applications to meet target performance and resource requirements. The Matador approach offers competitive performance in terms of latency, logic utilization and power consumption when compared against its nearest Binary Neural Network counterparts.  



#### Requirements 

Matador is currently not containerized (as of 01/08/2023), it is recommended to match the development settings as closely as possible to operate Matador without issues. 
To configure the tool the following requirements are required: 

1) Ubuntu 20.04.5
2) Vivado 2022.2 

#### Setup 

```
$ python3 -m venv MATADOR
$ source MATADOR/bin/activate

$ pip install -r requirements.txt

$ cd utils
$ make
$ cd ..

$ bash ./Matador

```

Useful links for understanding the concepts behind the Tsetlin Machine: 
Tsetlin Machine original paper : https://arxiv.org/abs/1804.01508
Coalesced Tsetlin Machine paper: https://arxiv.org/abs/2108.07594
Hardware Tsetlin Machines paper: https://royalsocietypublishing.org/doi/epdf/10.1098/rsta.2019.0593
Tsetlin Machine CAIR github    : https://github.com/cair/tmu


## Overview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/-1EmmJNy2wA/0.jpg)](https://www.youtube.com/watch?v=-1EmmJNy2wA)

## Example Walkthrough (master branch)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/mXH5gZGkKiQ/0.jpg)](https://www.youtube.com/watch?v=mXH5gZGkKiQ)

## Contact

Matador was developed by Tousif Rahman and Gang Mao in the Microsystems Group at Newcastle University. 

If there are issues with the setup or the usage please contact: 
Tousif Rahman: tousifsrahman@gmail.com
