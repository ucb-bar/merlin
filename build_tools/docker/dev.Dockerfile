# build_tools/docker/dev.Dockerfile
#
# Dev-container image for Merlin. Mirrors a fresh Linux dev box: conda +
# uv-managed Python deps + pre-commit + system build tools. The repo itself
# is bind-mounted at runtime (see docker-compose.yml / .devcontainer/), not
# COPY'd, so edits in the host repo are live inside the container.
#
# Build:   docker compose build merlin-dev
# Smoke:   docker compose run --rm merlin-dev ./merlin --help

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages needed before conda is bootstrapped.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        git \
        sudo \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install miniforge (conda + mamba), pinned channel.
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p "$CONDA_DIR" \
    && rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the merlin-dev env from the same env file the host uses.
# This is layered separately from the repo bind mount so it is cached
# across edits to the rest of the tree.
COPY env_linux.yml /tmp/env_linux.yml
RUN conda env create -f /tmp/env_linux.yml \
    && conda clean -afy \
    && rm /tmp/env_linux.yml

# Make `conda activate merlin-dev` the default shell behaviour for
# interactive sessions.
RUN conda init bash \
    && echo "conda activate merlin-dev" >> /root/.bashrc

WORKDIR /workspace
SHELL ["/bin/bash", "-lc"]
CMD ["bash"]
