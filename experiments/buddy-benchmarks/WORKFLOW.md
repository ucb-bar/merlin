# Buddy-MLIR Gemmini Workflow: From MLIR to Execution on Spike

This document describes the complete pipeline for compiling Gemmini dialect MLIR
to bare-metal RISC-V and running it on the Spike ISA simulator with the Gemmini
accelerator extension.

## Pipeline Overview

```
                        ┌──────────────────────┐
                        │  Gemmini MLIR Source  │
                        │   (gemmini.tile_*)    │
                        └──────────┬───────────┘
                                   │
                          buddy-opt --lower-gemmini
                            + standard MLIR passes
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  LLVM Dialect MLIR    │
                        │   (.llvm.mlir)        │
                        └──────────┬───────────┘
                                   │
                          buddy-translate --buddy-to-llvmir
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │     LLVM IR (.ll)     │
                        └──────────┬───────────┘
                                   │
                          buddy-llc -mattr=+buddyext
                            -mtriple=riscv64-unknown-elf
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  RISC-V Object (.o)   │
                        │  (RoCC custom insns)  │
                        └──────────┬───────────┘
                                   │
                          riscv64-unknown-elf-gcc
                            link with C harness
                            + baremetal runtime
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Bare-metal ELF       │
                        └──────────┬───────────┘
                                   │
                          spike --extension=gemmini
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Gemmini Simulator    │
                        │  Output + Cycles      │
                        └──────────────────────┘
```

## Prerequisites

### 1. RISC-V GNU Toolchain

A bare-metal cross-compiler targeting `riscv64-unknown-elf`:

```bash
# Provides: riscv64-unknown-elf-gcc, as, ld, objdump, etc.
export RISCV=/path/to/riscv-toolchain
```

Build from source: https://github.com/riscv-collab/riscv-gnu-toolchain
```bash
./configure --prefix=$RISCV --with-arch=rv64gc --with-abi=lp64d
make
```

### 2. Spike ISA Simulator (with Gemmini extension)

Spike must be built with Gemmini support from the Chipyard repository:

```bash
# Clone chipyard (includes Gemmini as a generator)
git clone https://github.com/ucb-bar/chipyard.git
cd chipyard && ./scripts/init-submodules-no-riscv-tools.sh

# Build Spike with Gemmini extension
cd sims/spike
make

# Or use a pre-built spike if available:
export SPIKE=$RISCV/bin/spike
```

### 3. Gemmini ROCC Tests (headers + baremetal runtime)

The C harnesses depend on headers and the bare-metal runtime from
[gemmini-rocc-tests](https://github.com/ucb-bar/gemmini-rocc-tests):

```bash
export GEMMINI_ROOT=/path/to/chipyard/generators/gemmini/software/gemmini-rocc-tests
```

Key files used:
- `include/gemmini.h` — Gemmini C API (`tiled_matmul_auto`, `tiled_conv_auto`, RoCC instruction macros)
- `include/gemmini_params.h` — Hardware parameters (DIM=16, scratchpad/accumulator sizes)
- `include/gemmini_testutils.h` — Test utilities (`read_cycles`, checksum helpers)
- `include/gemmini_nn.h` — NN layer helpers (for ResNet50 reference)
- `riscv-tests/benchmarks/common/` — Bare-metal startup code (`_start`, printf shims, syscall stubs)
- `riscv-tests/benchmarks/common/test.ld` — Linker script for bare-metal execution

### 4. Buddy-MLIR (with Gemmini dialect)

Build Buddy-MLIR from source with Gemmini dialect support:

```bash
# Step 1: Build LLVM/MLIR
git clone https://github.com/buddy-compiler/buddy-mlir.git
cd buddy-mlir && git submodule update --init
mkdir llvm/build && cd llvm/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DCMAKE_BUILD_TYPE=Release
ninja

# Step 2: Build buddy-mlir
cd ../../
mkdir build && cd build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release
ninja buddy-opt buddy-translate buddy-llc

export BUDDY=$PWD/bin
```

**Important:** Conv benchmarks require the fix from
[buddy-compiler/buddy-mlir#689](https://github.com/buddy-compiler/buddy-mlir/pull/689)
which corrects the `im2col` encoding in the Gemmini conv lowering.

## Step-by-Step: Compiling a Gemmini MLIR Kernel

Using `conv-buddy.mlir` as an example:

### Step 1: Write the Gemmini dialect MLIR

```mlir
// conv-buddy.mlir
module {
  func.func @conv(%input: memref<2x17x17x18xi8>,
                  %weights: memref<162x19xi8>,
                  %bias: memref<19xi32>,
                  %output: memref<162x19xi8>) attributes { llvm.emit_c_interface } {
    %c9 = arith.constant 9 : i64
    %c3 = arith.constant 3 : i64
    gemmini.tile_conv %input %weights %bias %output %c9 %c9 %c3
        {stride = 2, inputDilation = 1, kernelDilation = 1, padding = 1,
         act = 0} :
        memref<2x17x17x18xi8> memref<162x19xi8> memref<19xi32> memref<162x19xi8>
        i64 i64 i64
    return
  }
}
```

The `llvm.emit_c_interface` attribute generates a `_mlir_ciface_conv` wrapper
callable from C with memref descriptor structs.

Key `gemmini.tile_conv` operands:
- `%input` — 4D input tensor (batch × height × width × channels)
- `%weights` — 2D flattened weight matrix (patch_size × out_channels)
- `%bias` — 1D bias vector
- `%output` — 2D output matrix (n_patches × out_channels)
- `%c9 %c9` — output row/col dimensions (before pooling)
- `%c3` — kernel dimension

Key attributes: `stride`, `padding`, `act` (0=none, 1=ReLU, 3=iGELU, 4=softmax),
`poolSize`, `poolStride`, `poolPadding`, `dataflow` (0=OS, 1=WS), `bertScale`.

### Step 2: Lower to LLVM dialect

```bash
buddy-opt conv-buddy.mlir \
    -lower-gemmini \
    -convert-scf-to-cf \
    -convert-arith-to-llvm \
    -convert-func-to-llvm \
    -llvm-legalize-for-export \
    -o conv-buddy.llvm.mlir
```

Pass breakdown:
| Pass | What it does |
|------|-------------|
| `-lower-gemmini` | `gemmini.tile_conv` → Gemmini intrinsics (`gemmini.intr.loop_conv_ws`, `gemmini.intr.config_ex`, `gemmini.intr.flush`, etc.) with pre-computed tile sizes and constant offsets |
| `-convert-scf-to-cf` | SCF control flow → branch-based control flow |
| `-convert-arith-to-llvm` | Arithmetic ops → LLVM dialect |
| `-convert-func-to-llvm` | Function signatures → LLVM calling convention |
| `-llvm-legalize-for-export` | Final cleanup for LLVM IR emission |

### Step 3: Translate to LLVM IR

```bash
buddy-translate conv-buddy.llvm.mlir --buddy-to-llvmir -o conv-buddy.ll
```

This produces standard LLVM IR with inline assembly for Gemmini's RoCC custom
instructions (encoded as `.insn r` directives for the RISC-V assembler).

### Step 4: Compile to RISC-V object

```bash
buddy-llc conv-buddy.ll \
    -O3 \
    -filetype=obj \
    -mtriple=riscv64-unknown-elf \
    -mattr=+buddyext,+d,+f,+c \
    -float-abi=hard \
    -code-model=medium \
    -o conv-buddy.o
```

Key flags:
| Flag | Why |
|------|-----|
| `-mattr=+buddyext` | Enables custom Gemmini RoCC instruction support |
| `-mattr=+d,+f,+c` | Double/float/compressed RISC-V extensions |
| `-code-model=medium` | Required for large models (avoids `R_RISCV_HI20` relocation overflow) |
| `-float-abi=hard` | Hardware floating-point ABI |

### Step 5: Write a C harness

The C harness provides `main()`, initializes inputs, and calls the MLIR-generated
function through the C interface:

```c
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

// Memref descriptor matching MLIR's C interface
typedef struct {
  elem_t *basePtr;
  elem_t *data;
  int64_t offset;
  int64_t sizes[4];
  int64_t strides[4];
} MemRef4D_i8;

// The MLIR-generated function (from llvm.emit_c_interface)
extern void _mlir_ciface_conv(MemRef4D_i8 *input, ...);

int main(void) {
    // Initialize inputs, call function, measure cycles with rdcycle
    gemmini_flush(0);
    uint64_t start = read_cycles();
    _mlir_ciface_conv(&input_ref, &weights_ref, &bias_ref, &output_ref);
    gemmini_fence();
    uint64_t end = read_cycles();
    printf("Cycles: %llu\n", (unsigned long long)(end - start));
    // Compute and print output checksum for validation
}
```

### Step 6: Link into bare-metal ELF

```bash
riscv64-unknown-elf-gcc \
    -DPREALLOCATE=1 -DMULTITHREAD=1 -DBAREMETAL=1 \
    -mcmodel=medany -std=gnu99 -O2 -ffast-math \
    -fno-common -fno-builtin-printf \
    -fno-tree-loop-distribute-patterns \
    -march=rv64gc -Wa,-march=rv64gc \
    -nostdlib -nostartfiles -static \
    -T $GEMMINI_ROOT/riscv-tests/benchmarks/common/test.ld \
    -I$GEMMINI_ROOT/riscv-tests -I$GEMMINI_ROOT/riscv-tests/env \
    -I$GEMMINI_ROOT -I$GEMMINI_ROOT/include \
    -I$GEMMINI_ROOT/riscv-tests/benchmarks/common \
    conv-buddy.c conv-buddy.o \
    $GEMMINI_ROOT/riscv-tests/benchmarks/common/*.c \
    $GEMMINI_ROOT/riscv-tests/benchmarks/common/*.S \
    -lm -lgcc \
    -o conv-baremetal
```

The bare-metal runtime from `benchmarks/common/` provides:
- `_start` entry point and C runtime initialization
- `printf` via HTIF (Host-Target Interface) syscalls
- Memory management stubs

### Step 7: Run on Spike

```bash
spike --extension=gemmini conv-baremetal
```

Spike simulates the RISC-V core with the Gemmini systolic array extension
(default config: 16×16 PEs, weight-stationary dataflow). The `--extension=gemmini`
flag loads the Gemmini functional model that intercepts RoCC custom instructions.

Example output:
```
Buddy conv cycles: 149
Buddy conv output checksum: 950
Gemmini extension configured with:
    dim = 16
```

## Using the Makefiles

Instead of running each step manually, use the provided Makefiles:

```bash
# Kernel benchmarks (conv, mlp, etc.)
cd experiments/buddy-benchmarks/kernels
make conv-baremetal          # Build one benchmark
make all                     # Build all benchmarks
make run-conv                # Build + run on Spike
make run-all                 # Run everything

# ResNet50 layer validation
cd experiments/buddy-benchmarks/resnet50
make all                     # Build Gemmini C ref + Buddy + bad case
make validate                # Run all three and compare checksums

# Or run the full suite:
cd experiments/buddy-benchmarks
./scripts/run_benchmark.sh
```

## Why Buddy-MLIR Shows Fewer Cycles

The `rdcycle` instruction counts **CPU instructions executed**, not Gemmini
hardware cycles. Buddy-MLIR's lowering pre-computes tile sizes, loop bounds, and
memory offsets at compile time, emitting a flat sequence of Gemmini intrinsic calls
with constant arguments. In contrast, the Gemmini C reference (`tiled_matmul_auto`)
performs runtime tile-size search, per-tile address arithmetic, and loop iteration —
all of which execute on the CPU and inflate the `rdcycle` count.

The underlying Gemmini hardware work (systolic array compute, DMA transfers) is
the same in both cases. The speedup reflects reduced **host-side orchestration
overhead**, not faster accelerator throughput. This advantage would still manifest
on real hardware, since the CPU is freed up sooner for other work.

## Gemmini MLIR Operations Reference

| Operation | Description | Key Attributes |
|-----------|-------------|----------------|
| `gemmini.tile_matmul` | Tiled matrix multiply | `dataflow` (0=OS, 1=WS), `act` (0/1/3/4) |
| `gemmini.tile_conv` | Tiled convolution (im2col) | `stride`, `padding`, `poolSize`, `poolStride`, `act` |
| `gemmini.intr.flush` | Flush Gemmini command queue | — |
| `gemmini.intr.config_ex` | Configure execution mode | dataflow, activation, scale |
| `gemmini.intr.loop_ws` | Weight-stationary matmul loop | tile dimensions, addresses |
| `gemmini.intr.loop_conv_ws` | Weight-stationary conv loop | conv parameters, addresses |

Activation functions: 0=none, 1=ReLU, 3=iGELU, 4=softmax

Dataflows: 0=output-stationary (accumulates in place), 1=weight-stationary (keeps weights in scratchpad)
