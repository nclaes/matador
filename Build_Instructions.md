### <img src="/images/banner.png" width=1020/>

### Build Instructions 
Note Matador was design and built on a native ```Ubuntu 20.04.5 LTS x86_64```
You may replicate the following with:

First install Docker: https://docs.docker.com/engine/install/ubuntu/.

The above link is for a ubuntu base system - if you are using a different distro then install accordingly.
This version of Matador is ran from a GUI. Therefore it requires the X Window System (also known as X11, or simply X). This is a client/server windowing system for bitmap displays. It is implemented on most UNIX-like operating systems and has been ported to many other systems. The X server is the program or dedicated terminal that displays the windows and handles input devices such as keyboards, mice, and touchscreens. The clients are applications.

We are going to use X11 as a shared component between the host system and the docker container. We are going to use the Socket files, which is a UNIX technology that helps the daemon or the services running in the host Linux system to communicate with each other.

#### Update the Build Matador Script 
An example script looks like this: 

```bash 
#!/bin/bash

docker build -t matador_docker:v1 .
docker run -it --security-opt label=type:container_runtime_t --network=host -e DISPLAY=$DISPLAY -v /tools/Xilinx/Vivado/2022.2/:/tools/Xilinx/Vivado/2022.2/ -v /home/tousif/Desktop/Matador_Docker/:/home/tousif/Desktop/Matador_Docker/ -v "/run/user/1000/gdm/Xauthority:/root/.Xauthority:rw" matador_docker:v1 

```
Notice there are three directories mounted (```-v``` flag). The first is the directory on the host where Vivado is installed. This version of Matador requires use of Xilinx tools. Then there is the directory where the Matador repo is pulled. Finally there is the location of the hosts XServer. 

| build flag      | Description |
| -------------------------------|----------------------------------------------------------------------------------- |
|```--network```             		| host network driver |
|```-e```             				| environment |
|```-DISPLAY```             		| share the hosts DISPLAY variable|
|```-v```             				| mount local directories |


#### Build the Docker Environment

First navigate to the directory where the Matador repo is located. 

```bash
./Buid_Matador
```
If the build is successful then you will get a docker container that contains the Matador tools. 

### <img src="/images/build_success.png" width=300/>

```[Disclaimer] ``` Matador is a research tool - the build instructions here are not robust to all system specifications, the tool is open source and offers flexibility to users to manipulate it how they choose. 


#### Build the tmu dependencies 

This version of Matador uses tmu codebase. This is already built for the container. However if there are build issues please report them or propose new instructions for building this.  
