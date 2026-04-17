#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="merlin-linux-builder:local"
TAG="${1:-}"

if [[ -n "${TAG}" ]]; then
  CURRENT_TAG="$(git -C "${REPO_ROOT}" describe --tags --exact-match HEAD 2>/dev/null || true)"
  if [[ "${CURRENT_TAG}" != "${TAG}" ]]; then
    echo "Error: HEAD is not at tag '${TAG}' (current: '${CURRENT_TAG:-untagged}')"
    echo "Please run:  git checkout ${TAG}"
    exit 1
  fi
  echo "Building release for tag: ${TAG}"
else
  echo "Building release from current HEAD (no tag specified)"
fi

mkdir -p "${REPO_ROOT}/dist"
mkdir -p "${REPO_ROOT}/.docker_home"

echo ""
echo "=== Building Docker image ==="
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

echo ""
echo "=== Starting release build inside container ==="
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

echo ""
echo "=== Release artifacts ==="
ls -lh "${REPO_ROOT}/dist/"*.tar.gz 2>/dev/null || echo "No artifacts found in dist/"

if [[ -n "${TAG}" ]]; then
  echo ""
  echo "To upload these to the GitHub release draft:"
  echo "  gh release upload ${TAG} dist/*.tar.gz"
fi
