#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="merlin-linux-builder:local"

mkdir -p "${REPO_ROOT}/dist"
mkdir -p "${REPO_ROOT}/.docker_home"

docker build \
  -f "${REPO_ROOT}/build_tools/docker/linux-builder.Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

cleanup() {
  rm -rf "${REPO_ROOT}/.docker_home"
}
trap cleanup EXIT

docker run --rm \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${REPO_ROOT}:/workspace" \
  -v "${REPO_ROOT}/.docker_home:/tmp/merlin-home" \
  -w /workspace \
  -e HOME=/tmp/merlin-home \
  "${IMAGE_NAME}" \
  /bin/bash -lc '
    micromamba run -n merlin-dev bash build_tools/docker/in_container_release.sh
  '
