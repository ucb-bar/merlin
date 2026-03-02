# Steps to cross compile for banana pi

Best guide is to follow the steps in the page here: [LINK](https://bianbu.spacemit.com/en/development/kernel_compile/#install-cross-compiler-toolchain).

## 1. Toolchain & Environment

**Critical:** Use the vendor toolchain (v1.1.2) to ensure GLIBC compatibility with the board's OS (Bianbu/Armbian). Do not use the generic IREE bootstrap.

### 1.1 Setup SpacemiT toolchain

From there get the linux toolchain and put inside of a folder at root level called `riscv-tools-spacemit`.

```bash
export WORKSPACE_DIR=${PWD}
mkdir -p ${WORKSPACE_DIR}/riscv-tools-spacemit

# Download v1.1.2 (matches board GLIBC)
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -P ${WORKSPACE_DIR}

# Extract
tar -xvf ${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -C ${WORKSPACE_DIR}/riscv-tools-spacemit

# Set Environment Variable for CMake
export RISCV_TOOLCHAIN_ROOT=${WORKSPACE_DIR}/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
```

### 1.2 Build Dependency: Zstandard (zstd)

IREE requires `zstd` for tracing. We must cross-compile it and install it directly into the toolchain's sysroot so the compiler finds it automatically.

```bash
# 1. Clone Zstd
cd ${WORKSPACE_DIR}
git clone https://github.com/facebook/zstd.git
cd zstd
rm -rf build-riscv # Clean any previous attempts

# 2. Configure & Install
# We install to the toolchain's /usr directory to simplify linking
cmake -G Ninja -B build-riscv \
    -S build/cmake \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="${RISCV_TOOLCHAIN_ROOT}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${RISCV_TOOLCHAIN_ROOT}/bin/clang++" \
    -DCMAKE_C_FLAGS="--sysroot=${RISCV_TOOLCHAIN_ROOT}/sysroot -march=rv64gc -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="--sysroot=${RISCV_TOOLCHAIN_ROOT}/sysroot -march=rv64gc -mabi=lp64d" \
    -DCMAKE_INSTALL_PREFIX="${RISCV_TOOLCHAIN_ROOT}/sysroot/usr" \
    -DZSTD_BUILD_PROGRAMS=OFF \
    -DZSTD_BUILD_SHARED=OFF \
    -DZSTD_BUILD_STATIC=ON

cmake --build build-riscv --target install
```

## 2. IREE Runtime Build

Strategy: We compile the Runtime Tools (`iree-run-module`, `iree-tracy-capture`) WITHOUT Vector extensions (`+v`).

- **Reason:** The C++ runtime code (parsing args, networking) causes "Bus Error" (unaligned access) crashes if compiled with vectors on this board.
- **Fix:** Use `-march=rv64gc` plus specific cache/bitmanip extensions, but omit `v`.

### 2.1 Apply Tracy Source Patch

The Tracy Capture server hits an assertion failure on 64-bit RISC-V regarding pointer packing. You must disable it manually.

- **File:** `${IREE_SRC}/third_party/tracy/server/TracyWorker.cpp`
- **Line:** ~3931 (inside `PackPointer` function)
- **Action:** Comment out the assertion.

```C++
// Comment this line out:
// assert( ( ( ptr & 0x3000000000000000 ) << 2 ) == ( ptr & 0xC000000000000000 ) );
```

### 2.2 Configuration & Build

```bash
cd ${WORKSPACE_DIR}
rm -rf ${BUILD_RISCV_DIR}-spacemit

# Define Flags
RISCV_BASE_FLAGS=(
    -G Ninja
    -B "${BUILD_RISCV_DIR}-spacemit"
    -S "${IREE_SRC}"
    -DCMAKE_TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake"
    -DIREE_HOST_BIN_DIR="${INSTALL_HOST_DIR}/bin"
    -DRISCV_CPU=linux-riscv_64
    -DIREE_BUILD_COMPILER=OFF
    -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}"
    -DIREE_HAL_DRIVER_DEFAULTS=OFF
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON
    -DIREE_BUILD_TESTS=OFF
    -DIREE_BUILD_SAMPLES=OFF
)

# Configure
cmake "${RISCV_BASE_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS="-march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d" \
    -DIREE_ENABLE_RUNTIME_TRACING=ON \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_CPUINFO=ON \
    -DIREE_BUILD_TRACY=ON \
    -DTRACY_NO_POINTER_COMPRESSION=ON

# Build
cmake --build "${BUILD_RISCV_DIR}-spacemit"
```

#### Performance build:

```bash
cmake "${RISCV_BASE_FLAGS[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_ENABLE_RUNTIME_TRACING=OFF \
    -DIREE_ENABLE_CPUINFO=OFF

cmake --build "${BUILD_RISCV_DIR}"
```

## 3. Model Compilation

Compile the model WITH vector extensions (`+v`, `+zvl256b`). IREE handles memory alignment correctly, so vectors are safe here.

```bash
${BUILD_HOST_DIR}/tools/iree-compile \
    model_quantized_ort.mlir \
    -o model_quantized_ort.vmfb \
    \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
    --iree-llvmcpu-target-abi=lp64d \
    \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause" \
    --iree-llvmcpu-enable-ukernels="all" \
    --iree-opt-level=O3 \
    \
    --iree-hal-executable-debug-level=3 \
    --iree-llvmcpu-debug-symbols=true \
    --iree-llvmcpu-link-embedded=false \
    --iree-vm-bytecode-module-strip-source-map=false
```

## 4. Profiling Script (On Board)

Save this as `run_trace.sh` on the Banana Pi.

```bash
#!/bin/bash
set -e

# --- CONFIGURATION ---
BUILD_DIR="/home/build-iree-riscv-spacemit"
RUNTIME_TOOL="${BUILD_DIR}/tools/iree-run-module"
CAPTURE_TOOL="${BUILD_DIR}/tracy/iree-tracy-capture"
MODEL_PATH="/home/agus/models/model_quantized_ort.vmfb"
OUTPUT_TRACE="trace_$(date +%Y%m%d_%H%M%S).tracy"

echo "1. Starting Tracy Capture..."
"$CAPTURE_TOOL" -f -o "$OUTPUT_TRACE" &
CAPTURE_PID=$!
sleep 1

echo "2. Running Model..."
# TRACY_NO_EXIT=1 forces wait for connection
TRACY_NO_EXIT=1 "$RUNTIME_TOOL" \
  --device=local-task \
  --module="$MODEL_PATH" \
  --function=main_graph \
  --input="1x3x224x224xf32=0"

echo "3. Stopping Capture..."
kill -SIGINT $CAPTURE_PID
wait $CAPTURE_PID 2>/dev/null || true

echo "Trace saved: $OUTPUT_TRACE"
```