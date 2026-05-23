# =============================================================================
# Matador development container — ubuntu:24.04
#
# Verilator: Ubuntu 24.04 (Noble) ships verilator 5.020 via the 'universe'
# repository (verified: packages.ubuntu.com/noble/verilator). This satisfies
# the >=5.x requirement so we install from apt rather than building from
# source. If a future distro downgrade requires source build, use tag v5.020
# from https://github.com/verilator/verilator.
# =============================================================================

FROM ubuntu:24.04

ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    verilator \
    iverilog \
    yosys \
    gtkwave \
    make \
    openssh-client \
    less \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with host-matchable UID/GID to avoid bind-mount permission
# issues. Override at build time: docker build --build-arg UID=$(id -u) ...
#
# GID may already exist in the base image — common on macOS where the host
# GID is a low system value (e.g. GID 20 = staff). Check with getent before
# calling groupadd to avoid "GID already exists" (exit code 4).
RUN ( getent group "${GID}" > /dev/null 2>&1 || groupadd -g "${GID}" dev ) \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash dev

# Isolated venv so pip packages don't pollute the system Python.
# Use numeric UID/GID — the group may not be named 'dev' if the GID already
# existed in the base image and groupadd was skipped (see above).
RUN python3 -m venv /opt/venv \
 && chown -R "${UID}:${GID}" /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

# Mount point for the user's work directory.
# Populated at runtime: make shell WORK_DIR=/path/to/mydata
RUN mkdir /work

# ---------------------------------------------------------------------------
# Python package installation — cache-friendly layer ordering.
#
# 1. Copy only pyproject.toml first. This layer is invalidated (and pip
#    re-runs) only when dependencies change, not on every source edit.
# 2. Stub matador/__init__.py so setuptools can resolve the editable install
#    at build time. The bind mount at runtime shadows this with real source.
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
RUN mkdir -p matador && touch matador/__init__.py
# setuptools is required by cffi on Python 3.12+ (no longer bundled in stdlib)
RUN pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir -e ".[dev]"

USER dev

CMD ["/bin/bash"]
