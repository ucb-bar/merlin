# Guide: A/B Benchmarking IREE Ukernels for Saturn OPU on FireSim

This guide details the complete workflow to perform an A/B comparison benchmark between:

- **Baseline:** The default, generic `linalg.generic` implementation of a matrix multiplication, as compiled by IREE.

- **Optimized:** The new `linalg.mmt4d` implementation that is lowered to your custom-patched Saturn OPU microkernel.

In order to make possible the integration of the OPU instructions we modified a few files in the IREE code generation.

Particularly:

- `third_party/iree_bar/compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp`
- `third_party/iree_bar/runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_tiles.inl`
- `third_party/iree_bar/runtime/src/iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_v.c`

I recommend you to have a look at those files if you want to understand how we integrated the Outer Product as a replacement to the regular matrix multiplication ukernel of mmt4d.

## Part 1: Build the IREE Toolchain

Follow [getting_started.md](../getting_started.md) to bring up the conda env,
sync submodules, and build host tools (`./merlin build --profile vanilla`).
Then add the RISC-V cross-toolchain for FireSim:

```bash
./build_tools/firesim/setup_toolchain.sh
./merlin build --profile firesim
```

This produces both the host `iree-compile` (under `build/host-vanilla-release/`)
and the RISC-V `iree-benchmark-module` (under `build/firesim-merlin-release/`).

## Part 2: Generate and Compile the Model (A/B Test)

This is the most critical stage. We will compile the same model twice: once with our ukernels enabled (Optimized) and once with them disabled (Baseline).

### Step 2.1: Generate ONNX quantized model

From your `samples/custom_dispatch_ukernels_saturn` directory, run the export script.

```bash
cd samples/custom_dispatch_ukernels_saturn

# Use for a simple MLP model we just include a batch size of 16 to trigger the instruction
python export_models_onnx.py --model fc
```

### Step 2.2: Convert ONNX to MLIR

Convert the new, batched ONNX model to an MLIR file.

```bash
# This uses the compiler you built in Part 1
${BUILD_HOST_DIR}/bin/iree-import-onnx \
  compilation_phases_fc/model_quantized_ort.onnx \
  --opset-version 20 \
  -o model_quantized_ort.mlir
```

### Step 2.3: Compile A/B Benchmark Artifacts

Now we compile `model_quantized_ort.mlir` twice to generate the self-contained benchmark `.vmfb` files.

We will use the `riscv64` target triple and the `+zvl128b` feature, which is the `VLEN` we are targeting.

1. Compile the Optimized (`_s`) Kernels

```bash
# Compile with ukernels Enabled
${BUILD_HOST_DIR}/tools/iree-compile \
  model_quantized_ort.mlir \
  -o /dev/null \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-dispatch-creation-data-tiling \
  --iree-llvmcpu-enable-ukernels="all" \
  --iree-flow-export-benchmark-funcs \
  --iree-opt-level=O3 \
  --iree-hal-dump-executable-files-to=/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc/riscv/executables_opu

# --- This creates the self-contained benchmark .mlir files ---
# We now compile those .mlir files into the final .vmfb binaries

${BUILD_HOST_DIR}/tools/iree-compile \
  riscv/executables_opu/module_main_graph\$async_dispatch_1_embedded_elf_riscv_64_benchmark.mlir \
  -o ukernel_1_s.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-enable-ukernels="all" \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-opt-level=O3

${BUILD_HOST_DIR}/tools/iree-compile \
  riscv/executables_opu/module_main_graph\$async_dispatch_2_embedded_elf_riscv_64_benchmark.mlir \
  -o ukernel_2_s.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-enable-ukernels="all" \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-opt-level=O3
```

2. Compile the Baseline (normal) Kernels

This command disables ukernels, forcing the compiler to use the generic `CPUDoubleTilingExpert` pipeline.

```bash
# Compile with ukernels Disabled
${BUILD_HOST_DIR}/tools/iree-compile \
  model_quantized_ort.mlir \
  -o /dev/null \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-dispatch-creation-data-tiling \
  --iree-llvmcpu-enable-ukernels="none" \
  --iree-flow-export-benchmark-funcs \
  --iree-opt-level=O3 \
  --iree-hal-dump-executable-files-to=/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc/riscv/executables_baseline

# --- Compile the baseline .mlir benchmark files ---

${BUILD_HOST_DIR}/tools/iree-compile \
  riscv/executables_baseline/module_main_graph\$async_dispatch_1_embedded_elf_riscv_64_benchmark.mlir \
  -o ukernel_1.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-enable-ukernels="none" \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-opt-level=O3

${BUILD_HOST_DIR}/tools/iree-compile \
  riscv/executables_baseline/module_main_graph\$async_dispatch_2_embedded_elf_riscv_64_benchmark.mlir \
  -o ukernel_2.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
  --iree-llvmcpu-enable-ukernels="none" \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-opt-level=O3
```

You now have your four target files: `ukernel_1.vmfb`, `ukernel_1_s.vmfb`, `ukernel_2.`vmfb, and `ukernel_2_s.vmfb`.

## Part 3: Prepare the FireSim Workload

1. Copy binaries from `${BUILD_RISCV_DIR}\tools\`into your overlay folder. Specifically copy `iree-benchmark-executable`, `iree-benchmark-module` and `iree-run-module`.
2. Copy the generated vmfb files for each uKernel or model you want to test into that same folder.
3. Cross-compile or use your favorite way to measure cycles. Mine is:

```C
#include <stdio.h>

int main() {
    unsigned long cycles;
    // This assembly instruction reads the 'mcycle' CSR
    asm volatile ("rdcycle %0" : "=r"(cycles));
    printf("%lu\n", cycles);
    return 0;
}
```

4. Create a `run_iree.sh` to run the script:

```bash
#!/bin/bash

cd /
echo "--- Running IREE Microbenchmark Tests ---"

# --- Test Definitions ---
FUNC_1='main_graph$async_dispatch_1_embedded_elf_riscv_64_main_graph$async_dispatch_1_matmul_like_16x128x1024_i8xi8xi32'
FUNC_2='main_graph$async_dispatch_2_embedded_elf_riscv_64_main_graph$async_dispatch_2_matmul_like_16x10x128_i8xi8xi32'

# --- Array of modules to test ---
MODULES_TO_TEST=(
    "ukernel_1.vmfb"
    "ukernel_1_s.vmfb"
    "ukernel_2.vmfb"
    "ukernel_2_s.vmfb"
)

# --- Array of corresponding functions ---
FUNCTIONS_TO_CALL=(
    "$FUNC_1"
    "$FUNC_1"
    "$FUNC_2"
    "$FUNC_2"
)

# --- Run all 4 tests ---
for i in {0..3}; do
    MODULE_FILE=${MODULES_TO_TEST[$i]}
    FUNCTION_NAME=${FUNCTIONS_TO_CALL[$i]}
    TEST_NUM=$((i + 1))

    echo "--- Test $TEST_NUM: Benchmarking $MODULE_FILE ---"

    echo "--- Capturing Start Cycle ---"
    ./get_cycle > /start_cycle_$TEST_NUM.txt

    ./iree-benchmark-module \
      --device=local-sync \
      --benchmark_report_aggregates_only=true \
      --benchmark_display_aggregates_only=true \
      --benchmark_time_unit=ns \
      --benchmark_min_warmup_time=1 \
      --benchmark_repetitions=10 \
      --module=$MODULE_FILE > /output_$TEST_NUM.txt


    echo "--- Capturing End Cycle ---"
    ./get_cycle > /end_cycle_$TEST_NUM.txt
done

echo "--- All Benchmarks Finished ---"
echo

# --- Calculate and print all results ---
for i in {0..3}; do
    TEST_NUM=$((i + 1))
    MODULE_FILE=${MODULES_TO_TEST[$i]}

    START_CYCLE=$(cat /start_cycle_$TEST_NUM.txt)
    END_CYCLE=$(cat /end_cycle_$TEST_NUM.txt)
    TOTAL_CYCLES=$((END_CYCLE - START_CYCLE))

    echo "========================================="
    echo "Results for: $MODULE_FILE"
    echo "========================================="
    echo "TOTAL SIMULATION CYCLES (from ./get_cycle): $TOTAL_CYCLES"
    echo "--- iree-benchmark-module Output (use 'Time' for exec cycles) ---"
    cat /output_$TEST_NUM.txt
    echo
done

poweroff
```
