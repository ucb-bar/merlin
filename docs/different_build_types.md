# IREE biulding configs

## Export variables

```bash
export WORKSPACE_DIR=${PWD}

# --- SELECT SOURCE ---
# Option A: Forked IREE
export IREE_SRC=${WORKSPACE_DIR}/third_party/iree_bar
# Option B: Standard IREE (Plugin)
# export IREE_SRC=${WORKSPACE_DIR}/third_party/iree
# ---------------------

# Host Paths
export BUILD_HOST_DIR=${WORKSPACE_DIR}/build-host
export INSTALL_HOST_DIR=${BUILD_HOST_DIR}/install

# RISC-V Paths
export RISCV_TOOLCHAIN_ROOT=${WORKSPACE_DIR}/riscv-tools-iree
export BUILD_RISCV_DIR=${WORKSPACE_DIR}/build-riscv
```

## Host Build Options

### A. Standard Release

Best for: Fast iteration, standard logs, deployment testing.

- Speed: High
- Tracing: OFF
- Checks: Minimal (assertions ON)

```bash
cmake \
    -G Ninja \
    -B "${BUILD_HOST_DIR}" \
    -S "${IREE_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_HOST_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_LLD=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DCMAKE_CXX_FLAGS="-Wno-error=cpp" \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_BUILD_TESTS=ON \
    -DIREE_BUILD_SAMPLES=ON

cmake --build "${BUILD_HOST_DIR}" --target install
```

### B. Release + Profiling (Tracy)

Best for: Profiling compilation time and host runtime performance.

- Speed: High (Optimized)
- Tracing: ON (Tracy)
- Checks: Standard

```bash
cmake \
    -G Ninja \
    -B "${BUILD_HOST_DIR}-deb-tracy" \
    -S "${IREE_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_HOST_DIR}-deb-tracy" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-Wno-error=cpp -fno-omit-frame-pointer" \
    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
    -DIREE_ENABLE_LLD=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DIREE_ENABLE_RUNTIME_TRACING=ON \
    -DIREE_ENABLE_COMPILER_TRACING=ON \
    -DIREE_TRACING_MODE=4 \
    -DIREE_LINK_COMPILER_SHARED_LIBRARY=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build "${BUILD_HOST_DIR}-deb-tracy" --target install
```

### C. Debug + Sanitizers (ASan/UBSan)

Best for: Catching memory leaks, out-of-bounds access, and undefined behavior.

- Speed: Slow (~2x slowdown)
- Tracing: OFF
- Checks Max (Address Sanitizer + Undefined Behaviour Sanitizer)

```bash
cmake \
    -G Ninja \
    -B "${BUILD_HOST_DIR}-debug-profile" \
    -S "${IREE_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_HOST_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-Wno-error=cpp -fno-omit-frame-pointer" \
    -DCMAKE_C_FLAGS="-fno-omit-frame-pointer" \
    -DIREE_ENABLE_LLD=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DIREE_ENABLE_RUNTIME_TRACING=ON \
    -DIREE_ENABLE_COMPILER_TRACING=ON \
    -DIREE_LINK_COMPILER_SHARED_LIBRARY=OFF \
    -DIREE_TRACING_MODE=4 \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_ENABLE_ASAN=ON \
    -DIREE_ENABLE_UBSAN=ON

cmake --build "${BUILD_HOST_DIR}-debug-profile" --target install
```

## Cross- Compilations RISC-V

### One-time Bootstrap

```bash
cd ${IREE_SRC} && ./build_tools/riscv/riscv_bootstrap.sh
```

### A. Performance

Best for: Final benchmarking on Chipyard/FireSim.

```bash
unset CFLAGS CXXFLAGS

cmake \
  -G Ninja \
  -B "${BUILD_RISCV_DIR}" \
  -S "${IREE_SRC}" \
  -DCMAKE_TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="${INSTALL_HOST_DIR}/bin" \
  -DRISCV_CPU=linux-riscv_64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DIREE_ENABLE_RUNTIME_TRACING=OFF \
  -DIREE_ENABLE_CPUINFO=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF

cmake --build "${BUILD_RISCV_DIR}"
```

### B. Profiling and Debug

Best for: Deep debugging and generating traces on RISC-V.

```bash
unset CFLAGS CXXFLAGS

cmake \
  -G Ninja \
  -B "${BUILD_RISCV_DIR}" \
  -S "${IREE_SRC}" \
  -DCMAKE_TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="${INSTALL_HOST_DIR}/bin" \
  -DRISCV_CPU=linux-riscv_64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
  -DIREE_ENABLE_CPUINFO=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF

cmake --build "${BUILD_RISCV_DIR}"
```

## Executing models with different configs

### A. Standard Compilation

```bash
${INSTALL_HOST_DIR}/bin/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
    ${IREE_SRC}/samples/models/simple_abs.mlir \
    -o ${WORKSPACE_DIR}/chipyard-workload/overlay/simple_abs.vmfb
```

### B. Trace Ready compilation

Required if using Tracy (Host Option B or RISC-V Option B).

```bash
${INSTALL_HOST_DIR}/bin/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 \
    --iree-hal-executable-debug-level=3 \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-debug-symbols=true \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
    ${IREE_SRC}/samples/RoboIR/onnx_models/static_trace.mlir \
    -o ${WORKSPACE_DIR}/chipyard-workload/overlay/static_trace.vmfb
```

### C. Using Sanitizers

Required if using Sanitizers (Host Option C). Adds instrumentation to the JIT/compiled model code itself.

```bash
${INSTALL_HOST_DIR}/bin/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=x86_64 \
    --iree-llvmcpu-sanitize=address \
    --iree-llvmcpu-link-embedded=false \
    ${IREE_SRC}/samples/models/simple_abs.mlir \
    -o simple_abs_asan.vmfb
```

## Executing those models with their respective build

### A. Standard

```bash
${INSTALL_HOST_DIR}/bin/iree-run-module \
    --device=local-task \
    --module=simple_abs.vmfb \
    --function=abs \
    --input="f32=-10"
```

### B. Profiled Run (Tracy)


```bash
TRACY_NO_EXIT=1 sudo ${INSTALL_HOST_DIR}/bin/iree-run-module \
    --device=local-task \
    --module=static_trace.vmfb \
    --function=main \
    --input="1xf32=10"
```

### C. Sanitizer Run (ASan)

```bash
ASAN_OPTIONS=detect_leaks=0 \
ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer) \
${INSTALL_HOST_DIR}/bin/iree-run-module \
    --device=local-task \
    --module=simple_abs_asan.vmfb \
    --function=abs \
    --input="f32=-10"
```

## Using Tracy Offline / No SSH connection

Use if you built RISC-V Option B. This runs the capture tool on the device and saves the file locally.

### 1. Copy tools for overlay or target

```bash
cp ${BUILD_RISCV_DIR}/tools/iree-run-module ${WORKSPACE_DIR}/chipyard-workload/overlay/
cp ${BUILD_RISCV_DIR}/tracy-capture ${WORKSPACE_DIR}/chipyard-workload/overlay/
```

### 2. Run Script

```bash
#!/bin/bash
echo "--- Starting IREE Offline Capture ---"

# 1. Start IREE in background.
# TRACY_NO_EXIT=1 makes it wait for a connection before running.
TRACY_NO_EXIT=1 /iree-run-module \
  --device=local-task \
  --module=/static_trace.vmfb \
  --function=main \
  --input="1xf32=10" &

# 2. Start Tracy Capture (Client connects to localhost:8086)
# It will record until iree-run-module finishes.
# -f overwrites existing file. -o specifies output.
/tracy-capture -o /trace.tracy -f -a 127.0.0.1

echo "--- Capture Finished ---"
poweroff
```

### 3. Extract Trace

Add /trace.tracy to the outputs list in your iree-workload.json so FireMarshal copies it back to the host.

/scratch2/agustin/merlin/build-iree-host-deb-tracy/tools/iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv6 --iree-hal-executable-debug-level=3 --iree-llvmcpu-link-embedded=false --iree-llvmcpu-debug-symbols=true --iree-llvmcpu-target-abi=lp64d --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl256b,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause" --iree-llvmcpu-enable-ukernels="all" --iree-vm-bytecode-module-strip-source-map=false --iree-opt-level=O3 model_quantized_ort.mlir -o model_quantized_ort.vmfb
