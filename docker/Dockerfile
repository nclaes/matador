# =============================================================================
# coal_tm dev container — ubuntu:24.04
#
# Minimal, Coalesced-TM-only: iverilog + Verilator for simulation, no
# Vivado/synthesis layer (out of scope for this branch). Trimmed from
# production_ready's docker/Dockerfile.
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
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libpython3-dev \
    verilator \
    gtkwave \
    make \
    less \
    vim-tiny \
    autoconf \
    gperf \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Icarus Verilog: built from source (tag v13_0), not apt. Ubuntu 24.04's apt
# package is v12.0 (2021) — verified (during this backend's own testbench
# development) to X-propagate a value that v13.0 computes correctly for the
# same RTL, so pin to the version this project was actually validated
# against rather than whatever a distro happens to ship.
RUN git clone --depth 1 --branch v13_0 https://github.com/steveicarus/iverilog.git /tmp/iverilog-src \
 && cd /tmp/iverilog-src \
 && sh autoconf.sh \
 && ./configure \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /tmp/iverilog-src

# Non-root user with host-matchable UID/GID to avoid bind-mount permission
# issues. Override at build time: docker build --build-arg UID=$(id -u) ...
RUN ( getent group "${GID}" > /dev/null 2>&1 || groupadd -g "${GID}" dev ) \
 && useradd -m -u "${UID}" -g "${GID}" -s /bin/bash dev

RUN python3 -m venv /opt/venv \
 && chown -R "${UID}:${GID}" /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/workspace"

WORKDIR /workspace
RUN mkdir /work

# Cache-friendly layer ordering: only pyproject.toml changes invalidate this.
COPY pyproject.toml ./
RUN mkdir -p coal_tm && touch coal_tm/__init__.py
RUN pip install --no-cache-dir setuptools cffi
RUN pip install --no-cache-dir -e ".[dev]"

USER dev

CMD ["/bin/bash"]
