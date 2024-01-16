
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive 

WORKDIR /home/tousif/Desktop/Matador_Docker

RUN echo "Matador v1.0.0 2023"
RUN echo "Building Matador Environiment"

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

CMD python3 /tmu/setup.py
CMD bash utils/opener.sh