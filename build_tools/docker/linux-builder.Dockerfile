FROM mambaorg/micromamba:2.0.5

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    tar \
    xz-utils \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER env_linux.yml /tmp/env_linux.yml
RUN micromamba create -y -n merlin-dev -f /tmp/env_linux.yml && \
    micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

CMD ["/bin/bash"]
