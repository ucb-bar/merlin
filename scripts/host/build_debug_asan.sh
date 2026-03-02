#!/bin/bash
set -e

# --- Configuration ---
# Get the workspace root (2 levels up from this script)
export WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export IREE_SRC=${IREE_SRC:-"${WORKSPACE_DIR}/third_party/iree_bar"}

PURPOSE="spacemit"
# Fetch IREE version from version.json
IREE_VERSION=$(python3 -c "import json; print(json.load(open('${IREE_SRC}/runtime/version.json'))['package-version'])" 2>/dev/null || echo "unknown")

# Host Paths (using vanilla folder structure)
export BUILD_HOST_DIR=${WORKSPACE_DIR}/build/vanilla/host/debug/iree-${PURPOSE}-${IREE_VERSION}
export INSTALL_HOST_DIR=${BUILD_HOST_DIR}/install

REBUILD=0
if [[ "$*" == *"--rebuild"* ]]; then
    REBUILD=1
    echo ">> REBUILD MODE ENABLED: Will clean whole build directory before building"
fi

# Check for the fast flag
BUILD_LLVM_ONLY=0
if [[ "$*" == *"--fast-llvm"* ]]; then
    BUILD_LLVM_ONLY=1
    echo ">> FAST MODE ENABLED: Will only build llvm-tblgen and llc"
fi

echo "========================================================"
echo " Building IREE Host (Debug + ASAN)"
echo " IREE Version: ${IREE_VERSION}"
echo " Source:       ${IREE_SRC}"
echo " Build Dir:    ${BUILD_HOST_DIR}"
echo " Install Dir:  ${INSTALL_HOST_DIR}"
echo "========================================================"

if [ "${REBUILD}" -eq "1" ]; then
    rm -rf "${BUILD_HOST_DIR}"
fi

# Ensure build dir exists
mkdir -p "${BUILD_HOST_DIR}"

# Run CMake Configuration
# We allow this to run every time (it's fast if nothing changed)
cmake \
    -G Ninja \
    -B "${BUILD_HOST_DIR}" \
    -S "${IREE_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_HOST_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-Wno-error=cpp -Wno-error=maybe-uninitialized -fno-omit-frame-pointer -fdebug-types-section -gz=none" \
    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer -fdebug-types-section -gz=none" \
    -DIREE_ENABLE_LLD=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DIREE_ENABLE_RUNTIME_TRACING=OFF \
    -DIREE_ENABLE_COMPILER_TRACING=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_ENABLE_ASAN=OFF

# Build Step
if [ "${BUILD_LLVM_ONLY}" -eq "1" ]; then
    echo "========================================================"
    echo " FAST BUILD: Compiling only LLVM TableGen and LLC..."
    echo "========================================================"
    # We go into the build dir and run ninja explicitly
    ninja -C "${BUILD_HOST_DIR}" llvm-tblgen llc FileCheck intrinsics_gen
else
    echo "========================================================"
    echo " FULL BUILD: Compiling and Installing everything..."
    echo "========================================================"
    cmake --build "${BUILD_HOST_DIR}" --target install

    echo "Building extra LLVM tools..."
    cmake --build "${BUILD_HOST_DIR}" --target llvm-mca llvm-objdump
fi

echo "========================================================"
echo " Build Complete"
echo "========================================================"