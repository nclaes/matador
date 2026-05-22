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
RUN groupadd -g "${GID}" dev \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash dev

# Isolated venv so pip packages don't pollute the system Python.
RUN python3 -m venv /opt/venv \
 && chown -R dev:dev /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

# ---------------------------------------------------------------------------
# Python package installation — cache-friendly layer ordering.
# pyproject.toml is copied first so pip only re-runs when dependencies
# change, not on every source edit.
#
# Uncomment both lines once Block 1.3 creates pyproject.toml:
#
#   COPY pyproject.toml ./
#   RUN pip install --no-cache-dir -e ".[dev]"
#
# Will succeed after Block 1.3 adds pyproject.toml
# ---------------------------------------------------------------------------
RUN echo "[docker] pyproject.toml not yet present — pip install pending Block 1.3"

USER dev

CMD ["/bin/bash"]
