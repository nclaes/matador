# DOCKER FILE for Matador - you need to change the WORKDIR
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive 

WORKDIR /home/tousif/Desktop/Matador_Docker

RUN echo "Matador v1.3.0 2024"
RUN echo "Building Matador Environiment"


# RUN apt-get install libtinfo5
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN apt-get update -y 
RUN apt-get install xauth -y 
RUN apt-get install python3-pip -y
RUN apt-get install python3-tk -y  
RUN apt-get install libglib2.0-dev -y
RUN apt-get install evince -y
# install python packages for Matador GUI
RUN pip install Pillow 
RUN pip install numpy 
# install python packages for TMU
RUN pip install cffi
RUN pip install tomli
RUN pip install tqdm
RUN pip install requests
RUN pip install tkPDFViewer
RUN pip install scipy 
# VITIS and Vivado (from 2023.2 onwards these packages are needed)
RUN apt install libtinfo-dev
# RUN ln -s /lib/x86_64-linux-gnu/libtinfo.so.6 /lib/x86_64-linux-gnu/libtinfo.so.5
RUN apt-get install libtinfo5
RUN apt install libncursesw5


CMD python3 /tmu/setup.py
CMD bash utils/opener.sh