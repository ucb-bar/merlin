# Instructions to reproduce building this experiment

## Configuring build

```bash
cmake \
    -G Ninja \
    -B build-riscv-cross \
    -S third_party/iree \
    -DCMAKE_TOOLCHAIN_FILE=/scratch2/agustin/merlin/third_party/iree/build_tools/cmake/riscv.toolchain.cmake \
    -DIREE_HOST_BIN_DIR=build-iree-host/install/bin \
    -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
    -DCMAKE_INSTALL_PREFIX=build-riscv-cross/install \
    -DRISCV_TOOLCHAIN_ROOT=riscv-tools-iree/toolchain/clang/linux/RISCV
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_SAMPLES=ON \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_HAL_DRIVER_LOCAL_TASK=ON
```

We build this particular sample using:
```bash
cmake --build build-riscv-cross --target compile_custom_model
```