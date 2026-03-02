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

You should follow the first 5 steps in the `iree_setup.md` documentation.

Key steps are:

1. Set up the Conda environment.
2. Set the `WORKSPACE_DIR`, `IREE_SRC`, `BUILD_HOST_DIR`, etc.
3. Build the host tools (like `iree-compile`).
4. Build the RISC-V tools (like `iree-benchmark-module`).

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

## Step 4: Benchmark the executable instead of the module (WiP)

After copying the `iree-benchmark-executable` and our `.vmfb` files, we must now extract the `.so` files out of them.

```bash
# Do this on your host machine before building the workload
# IMPORTANT Correct the names of the .so files so that it doesnt get overwritten
cd /path/to/your/workload/overlay/
unzip ukernel_1.vmfb    # Extracts ukernel_1.so (placeholder name)
unzip ukernel_1_s.vmfb  # Extracts ukernel_1_s.so (placeholder name)
unzip ukernel_2.vmfb    # ...
unzip ukernel_2_s.vmfb  # ...

# IMPORTANT: Ensure all files are readable
chmod u+r *.so *.vmfb get_cycle iree-benchmark-executable
```

We now create a new `run_iree.sh` that can execute the `iree-benchmark-executable`:

```bash
#!/bin/bash

cd /
echo "--- Running IREE Microbenchmark Tests (Kernel Computation Only) ---"

# --- Tool Definitions ---
BENCH_TOOL="./iree-benchmark-executable"
CYCLE_TOOL="./get_cycle"

# --- Kernel .so Files (Extracted from VMFBs) ---
MODULES_TO_TEST=(
    "ukernel_1.so"    # Kernel 1: Generic 16x128x1024
    "ukernel_2.so"    # Kernel 2: Generic 16x10x128
    "ukernel_1_s.so"  # Kernel 3: Ukernel 16x128x1024
    "ukernel_2_s.so"  # Kernel 4: Ukernel 16x10x128
)

# --- Parameters for EACH Kernel (Derived from MLIR) ---

# Kernel 1: Generic 16x128x1024 (CPUDoubleTilingExpert)
PARAMS_1="--workgroup_count=4,4,1 --binding=18432xi8 --binding=132864xi8 --binding=18432xi8"

# Kernel 2: Generic 16x10x128 (CPUDoubleTilingExpert)
PARAMS_2="--workgroup_count=2,8,1 --binding=18432xi8 --binding=132864xi8 --binding=18432xi8"

# Kernel 3: Microkernel 16x128x1024 (Mmt4dTilingExpert)
PARAMS_3="--workgroup_count=4,1,1 --binding=20480xi8 --binding=133696xi8 --binding=20480xi8"

# Kernel 4: Microkernel 16x10x128 (Mmt4dTilingExpert)
PARAMS_4="--workgroup_count=1,1,1 --binding=20480xi8 --binding=133696xi8 --binding=20480xi8"

PARAMS_TO_USE=(
    "$PARAMS_1"
    "$PARAMS_2"
    "$PARAMS_3"
    "$PARAMS_4"
)

# --- Benchmark Settings ---
# Run 1000 dispatches per measurement (amortization)
BATCH_SIZE=1000
# Run the whole benchmark 10 times (statistical stability)
REPETITIONS=10
TOTAL_DISPATCHES=$((BATCH_SIZE * REPETITIONS))

# --- Run all 4 tests ---
    for i in {0..3}; do
        SO_FILE=${MODULES_TO_TEST[$i]}
        PARAMS=${PARAMS_TO_USE[$i]}
        TEST_NUM=$((i + 1))

        echo "--- Test $TEST_NUM: Benchmarking $SO_FILE ---"
        
        echo "--- Capturing Start Cycle ---"
        $CYCLE_TOOL > /start_cycle_$TEST_NUM.txt

        # Run the benchmark. This will run (BATCH_SIZE * REPETITIONS) total dispatches.
        $BENCH_TOOL \
          --device=local-sync \
          --executable_file=/$SO_FILE \
          --executable_format=embedded-elf-riscv_64 \
          --entry_point=0 \
          $PARAMS \
          --batch_size=$BATCH_SIZE \
          --benchmark_repetitions=$REPETITIONS \
          --benchmark_out=/output_$TEST_NUM.json \
          --benchmark_out_format=json

        echo "--- Capturing End Cycle ---"
        $CYCLE_TOOL > /end_cycle_$TEST_NUM.txt
    done

    echo "--- All Benchmarks Finished ---"
    echo

    # --- Calculate and print all results ---
    echo "--- Benchmark Results (Cycles per Dispatch) ---"
    echo "Total Dispatches per Test: $TOTAL_DISPATCHES (Batch=$BATCH_SIZE, Reps=$REPETITIONS)"
    echo

    for i in {0..3}; do
        TEST_NUM=$((i + 1))
        MODULE_FILE=${MODULES_TO_TEST[$i]}
        
        START_CYCLE=$(cat /start_cycle_$TEST_NUM.txt)
        END_CYCLE=$(cat /end_cycle_$TEST_NUM.txt)
        TOTAL_CYCLES=$((END_CYCLE - START_CYCLE))
        
        # This is your final number:
        AVG_CYCLES=$((TOTAL_CYCLES / TOTAL_DISPATCHES))

        echo "========================================="
        echo "Results for: $MODULE_FILE"
        echo "========================================="
        echo "TOTAL SIMULATION CYCLES (from ./get_cycle): $TOTAL_CYCLES"
        echo "AVERAGE CYCLES PER DISPATCH: $AVG_CYCLES"
        echo "--- (Sanity Check: Mean Time from JSON) ---"
        grep "real_time_mean" /output_$TEST_NUM.json || echo "JSON output not found."
        echo
    done

poweroff
```

## Tips

- `unzip: short read`: This error means your .vmfb file was corrupted or truncated when you copied it into the FireSim workload. Re-build the workload image.
- `Illegal instruction` (SIGILL):
If running on spike: This is expected. spike is a generic RISC-V emulator and does not implement your custom VOPACC instruction.
If running on RTL (FireSim): This means your VOPACC implementation in the processor RTL has a bug, or the opcode bits in your IREE mmt4d C-file (.insn r ...) do not match the decoder in your hardware.
- `Segmentation fault` (SIGSEGV): This almost always means your --binding=... or --workgroup_count=... parameters are wrong. You are telling the kernel to access memory that wasn't allocated. Re-check the MLIR files to derive the correct parameters.
- `FAILED_PRECONDITION` (Version Mismatch): The iree-compile you used to build the .vmfb is from a different commit than the iree-benchmark-executable you are using to run it. Rebuild both from the same source.



### Debugging

I recommend to have a look at the whole compilation process by running:

```bash
iree-compile model_quantized_ort.mlir -o model_quantized_ort_riscv.vmfb \
--iree-hal-target-backends=llvm-cpu \
--iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
--iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl512b,+zvfh,+zvbb" \
--iree-llvmcpu-target-abi=lp64d \
--dump-compilation-phases-to=riscv \
--iree-dispatch-creation-data-tiling \
--iree-llvmcpu-enable-ukernels="all" \
--iree-opt-level=O3 \
-mlir-disable-threading \
-mlir-print-ir-after-all 2>log.mlir
```

# DUMP

```bash
(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   model_quantized_ort.mlir   -o /dev/null   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d   --iree-dispatch-creation-data-tiling   --iree-llvmcpu-enable-ukernels="none" --iree-opt-level=O3 --iree-hal-dump-executable-files-to=/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc/riscv/executables --iree-hal-executable-debug-level=3     --iree-llvmcpu-debug-symbols=true     --iree-llvmcpu-link-embedded=false     --iree-vm-bytecode-module-strip-source-map=false

(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   model_quantized_ort.mlir   -o /dev/null   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d   --iree-dispatch-creation-data-tiling   --iree-llvmcpu-enable-ukernels="none" --iree-opt-level=O3 --iree-hal-dump-executable-files-to=/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc/riscv/executables --iree-hal-executable-debug-level=3 \
    --iree-llvmcpu-debug-symbols=true \
    --iree-llvmcpu-link-embedded=false \
    --iree-vm-bytecode-module-strip-source-map=false

(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   riscv/executables_opu/module_main_graph\$async_dispatch_1_system_elf_riscv_64_benchmark.mlir   -o ukernel_1_s.vmfb   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-enable-ukernels="all"   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d--iree-opt-level=O3 --iree-hal-executable-debug-level=3     --iree-llvmcpu-debug-symbols=true     --iree-llvmcpu-link-embedded=false     --iree-vm-bytecode-module-strip-source-map=false
(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   riscv/executables_opu/module_main_graph\$async_dispatch_2_system_elf_riscv_64_benchmark.mlir   -o ukernel_2_s.vmfb   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-enable-ukernels="all"   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d --iree-opt-level=O3 --iree-hal-executable-debug-level=3     --iree-llvmcpu-debug-symbols=true     --iree-llvmcpu-link-embedded=false     --iree-vm-bytecode-module-strip-source-map=false

(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   riscv/executables/module_main_graph\$async_dispatch_2_system_elf_riscv_64_benchmark.mlir   -o ukernel_2.vmfb   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-enable-ukernels="none"   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d --iree-opt-level=O3 --iree-hal-executable-debug-level=3     --iree-llvmcpu-debug-symbols=true     --iree-llvmcpu-link-embedded=false     --iree-vm-bytecode-module-strip-source-map=false
(iree-dev) agustin@garden:/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/compilation_phases_fc$ ${BUILD_HOST_DIR}-deb-tracy/tools/iree-compile   riscv/executables/module_main_graph\$async_dispatch_1_system_elf_riscv_64_benchmark.mlir   -o ukernel_1.vmfb   --iree-hal-target-backends=llvm-cpu   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu   --iree-llvmcpu-enable-ukernels="none"   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl128b,+zvfh,+zvbb"   --iree-llvmcpu-target-abi=lp64d --iree-opt-level=O3 --iree-hal-executable-debug-level=3     --iree-llvmcpu-debug-symbols=true     --iree-llvmcpu-link-embedded=false     --iree-vm-bytecode-module-strip-source-map=false

```

## Simple placeholder on last version compiled for Firesim
```bash
# 2. Compile
${BUILD_HOST_DIR}/tools/iree-compile \
  model_quantized_ort.mlir \
  -o model_quantized_ort.vmfb \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64-pc-linux-elf \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-opt-level=O3 \
  \
  --iree-llvmcpu-enable-ukernels="all" \
  --iree-opt-data-tiling \
  --iree-dispatch-creation-data-tiling \
  \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c,+v,+zvl128b,+zvfh,+zvbb" \
  --iree-llvmcpu-target-vector-width-in-bytes=  16 \
  --riscv-v-fixed-length-vector-lmul-max=2 \
  \
  --iree-hal-dump-executable-files-to="$DUMP_DIR" \
  --iree-llvmcpu-debug-symbols=false
```