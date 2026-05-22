FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN echo "Matador v1.3.0 2024"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    locales \
    xauth \
    python3-pip \
    python3-tk \
    libglib2.0-dev \
    evince \
    libtinfo-dev \
    libtinfo5 \
    libncursesw5 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "Pillow>=9.0.0,<11.0.0" \
    "numpy>=1.23.0,<2.0.0" \
    "cffi>=1.15.0" \
    "tomli>=2.0.0" \
    "tqdm>=4.60.0" \
    "requests>=2.28.0" \
    "tkPDFViewer>=0.3" \
    "scipy>=1.9.0,<1.12.0"

CMD ["bash", "utils/opener.sh"]