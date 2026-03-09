# IREE setup

### Step 1: Set Up a Self-Contained Conda Environment

We will use Conda to install and manage the entire C++ toolchain and all Python dependencies. This avoids conflicts with system-wide packages.

1. *Create and activate a new Conda environment:*

```bash
# Create an environment named 'iree-dev' with Python
conda create -n iree-dev python=3.11

# Activate the environment
conda activate iree-dev
```

2. *Install the C++ toolchain and build tools:*
This command installs a complete LLVM/Clang toolchain, including the `lld` linker, as well as `cmake` and `ninja`.

```bash
conda install -c conda-forge cmake ninja clang clangxx gxx_linux-64 lld
```

3. *Install required Python dependencies:*

```bash
conda install -c conda-forge numpy
```

### Step 2: Set Environment Variables

From the root of the repository, export the following variables. These paths are essential for both the host build and the cross-compilation steps.

```bash
# Set the root directory for the entire project one level above than your iree repo
export WORKSPACE_DIR=${PWD}
export IREE_SRC=${WORKSPACE_DIR}/third_party/iree

# Directories for the x86_64 host build
export BUILD_HOST_DIR=${WORKSPACE_DIR}/build-iree-host
export INSTALL_HOST_DIR=${BUILD_HOST_DIR}/install

# Directories for cross-compiled risc-v
export RISCV_TOOLCHAIN_ROOT=${WORKSPACE_DIR}/riscv-tools-iree
export BUILD_RISCV_DIR=${WORKSPACE_DIR}/build-iree-riscv

# Verify paths
echo "Host Install Dir: ${INSTALL_HOST_DIR}"
echo "IREE source: ${IREE_SRC}"
echo "Riscv tool prefix path: ${RISCV_TOOLCHAIN_ROOT}"
echo "RISCV Install Dir: ${BUILD_RISCV_DIR}"
```

### Step 3: Configure and Build the Host Tools

This command configures the build for the IREE host tools (compiler, etc.). It uses the toolchain we installed with Conda and configures it to link with `lld`. A flag is added to `CMAKE_CXX_FLAGS` to prevent deprecation warnings from being treated as errors.

*Note:* Before running this command after a failed build, it's best to clear the old configuration: `rm -rf ${BUILD_HOST_DIR}`

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
    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_BUILD_TESTS=ON \
    -DIREE_BUILD_SAMPLES=ON

# Build and install the host tools
cmake --build "${BUILD_HOST_DIR}" --target install
```

### Step 4: Cross-compile for RISC-V

You will need to download the riscv prebuilt tools:

```bash
cd ${IREE_SRC}
./build_tools/riscv/riscv_bootstrap.sh
cd ..
```

When prompted with where to install all the files you should input:

```bash
${RISCV_TOOLCHAIN_ROOT}
```

For the CMake configuration:

```bash
# Unset any host-specific compiler flags from the environment to prevent contamination
unset CFLAGS CXXFLAGS
unset LIBRARY_PATH LD_LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

# Configure the RISC-V build
cmake \
  -G Ninja \
  -B "${BUILD_RISCV_DIR}" \
  -S "${IREE_SRC}" \
  -DCMAKE_TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="${INSTALL_HOST_DIR}/bin" \
  -DRISCV_CPU=linux-riscv_64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV" \
  -DIREE_ENABLE_CPUINFO=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF

# Build the cross-compiled runtime and tools
cmake --build "${BUILD_RISCV_DIR}"
```

#### If we want to buld with Tracy we need the following:

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
    -DCMAKE_C_COMPILER="${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV/bin/clang" \
    -DCMAKE_CXX_COMPILER="${RISCV_TOOLCHAIN_ROOT}/bin/clang++" \
    -DCMAKE_C_FLAGS="--sysroot=${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV/sysroot -march=rv64gc -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="--sysroot=${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV/sysroot -march=rv64gc -mabi=lp64d" \
    -DCMAKE_INSTALL_PREFIX="${RISCV_TOOLCHAIN_ROOT}/toolchain/clang/linux/RISCV/sysroot/usr" \
    -DZSTD_BUILD_PROGRAMS=OFF \
    -DZSTD_BUILD_SHARED=OFF \
    -DZSTD_BUILD_STATIC=ON

cmake --build build-riscv --target install
```

### Step 5: Python and compilation of our model

Intalling the necessary dependencies.

Pytorch:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

IREE:

```bash
python -m pip install \
    iree-turbine \
    iree-base-compiler \
    iree-base-runtime
```

For running a simple matmul example I can recommend just running the `pytorch_aot_simple.ipynb` notebook

### Step 6: Activate Chipyard and prepare IREE files

Source the env:

```bash
source env.sh
```

1. Create the overlay directory:

```bash
mkdir -p ${WORKSPACE_DIR}/chipyard-workload/overlay
```

2. Compile a Test Model for RISC-V:
Use the host compiler to generate a `.vmfb` file, specifying the correct target architecture and features to enable hardware floating-point support.

```bash
${INSTALL_HOST_DIR}/bin/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
    ${IREE_SRC}/samples/models/simple_abs.mlir \
    -o ${WORKSPACE_DIR}/chipyard-workload/overlay/simple_abs.vmfb
```

Or for tracing and debugging:

```bash
${INSTALL_HOST_DIR}/bin/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64 \
    --iree-hal-executable-debug-level=3 \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-debug-symbols=true \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d" \
    ${IREE_SRC}/samples/RoboIR/onnx_models/static_trace.mlir \
    -o ${IREE_SRC}/samples/RoboIR/vmfb/firesim/static_trace.vmfb
```

3. Copy the IREE Runtime into the Overlay:

```bash
cp ${BUILD_RISCV_DIR}/tools/iree-run-module ${WORKSPACE_DIR}/chipyard-workload/overlay/
```

4. Create an Automation Script (`run_iree.sh`):
This script will execute automatically upon booting the simulated machine. Create it at `${WORKSPACE_DIR}/chipyard-workload/overlay/run_iree.sh`.
**IMPORTANT:** Use absolute paths for all files inside the script.

```bash
#!/bin/bash
echo "--- Running IREE Model ---"
# Redirect both stdout and stderr to the output file
/iree-run-module \
  --device=local-task \
  --module=/simple_abs.vmfb \
  --function=abs \
  --input="f32=-10" > /output.txt 2>&1
echo "--- IREE Model Finished ---"
poweroff
```

5. Make the script executable:

```bash
chmod +x ${WORKSPACE_DIR}/chipyard-workload/overlay/run_iree.sh
```

### Step 7: Create the  shal Workload Recipe
Create a file at `${WORKSPACE_DIR}/chipyard-workload/iree-workload.json`. This JSON file is the recipe for building the final disk image.

```json
{
    "name": "br-iree",
    "base": "br-base.json",
    "overlay": "/path/to/your/workspace/chipyard-workload/overlay",
    "run": "run_iree.sh",
    "outputs": [
        "/output.txt"
    ]
}
```

**CRITICAL:** You must replace `/path/to/your/workspace/` with the hardcoded, absolute path to your `WORKSPACE_DIR`. FireMarshal may not correctly expand environment variables in this context.

### Step 8: Simulation and Verification

This phase executes the generated workload on two different Chipyard simulators.

#### 8.1: Build Linux Image using FireMarshal
1. Activate the Chipyard environment:

```bash
# Deactivate other conda environments if necessary
# conda deactivate
cd /path/to/your/chipyard
source ./env.sh
```

2. Build the Image:
The `-d` (`--no-disk`) flag is required to generate the special `*-nodisk` binary needed for the Spike simulator.

```bash
# From the chipyard/software/firemarshal directory
marshal -d build ${WORKSPACE_DIR}/chipyard-workload/iree-workload.json
```

#### 8.2: Functional Verification (Spike)
This is the fastest method to verify that the software toolchain and Linux image are correct.

1. Launch the Spike simulation:
The `-s` (`--spike`) flag tells FireMarshal to use Spike.

```bash
# From the chipyard/software/firemarshal directory
marshal -d launch -s ${WORKSPACE_DIR}/chipyard-workload/iree-workload.json
```

2. Inspect the output:
After the simulation powers off, find the captured `output.txt` in the timestamped results directory.

```bash
cat ./runOutput/iree-workload-launch-*/br-iree/output.txt
```

#### 8.3: Cycle-Accurate Simulator (this step is not completely finished nor tested)
This is the slower but more realistic test, running your workload on a C++ model of the actual `GemminiRocketConfig` hardware.
1. Copy the Workload JSON to the default location:

```bash
cp ${WORKSPACE_DIR}/chipyard-workload/iree-workload.json /path/to/your/chipyard/software/firemarshal/workloads/
```

2. Clean, Build, and Run the Verilator Simulation:
These steps must be run from the `sims/verilator` directory.

```bash
cd /path/to/your/chipyard/sims/verilator

# Clean previous build artifacts to prevent errors
make clean

# Build the simulator and run the workload in a single command.
# The NAME variable MUST match the "name" field from inside your JSON file.
make CONFIG=GemminiRocketConfig NAME=br-iree run-workload
```

After the simulation completes, the UART log containing the `output` from your script will be available in the output directory within the `sims/verilator` folder.


## Extra random stuff

- Sometimes my conda `env` will be overimpossed on top of another one, meaning I had to run `conda deactivate` until there was no `env` active. Then I was able to sucessfully call `source env.sh`

**CRITICAL:** You must replace `/path/to/your/workspace/` with the hardcoded, absolute path to your `WORKSPACE_DIR`. FireMarshal may not correctly expand environment variables in this context.

After the simulation completes, the UART log containing the `output` from your script will be available in the output directory within the `sims/verilator` folder.

## Extra random stuff

- Sometimes my conda `env` will be overimpossed on top of another one, meaning I had to run `conda deactivate` until there was no `env` active. Then I was able to sucessfully call `source env.sh`
