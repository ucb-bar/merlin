# Gemmini Spike functional regression tests

This suite drives the full Merlin Gemmini codegen pipeline to a Spike-
runnable RISC-V ELF and compares the output against a numpy-computed
reference.

## What is exercised

Per fixture, `./merlin spike <fixture>.mlir --kernel <sym> --shape MxNxK
--kind matmul`:

1. `iree-opt --iree-plugin=gemmini --iree-gemmini-enable
   --iree-gemmini-lower-back-to-iree=false --pass-pipeline='...'` runs the
   full Gemmini compile pipeline — `gemmini-convert-to-gemmini`,
   `gemmini-lower-to-isa`, `gemmini-lower-tile-to-isa`,
   `merlin-gemmini-legalize-for-llvm-export`, then standard
   `convert-{scf,arith,func,memref}-to-llvm`.
2. `mlir-translate --mlir-to-llvmir` emits LLVM IR with `llvm.intr.riscv.*`
   calls (Gemmini RoCC intrinsics) wired in by
   `compiler/src/merlin/Target/LLVMIR/Dialect/Gemmini/`.
3. `clang-23 -target riscv64-unknown-elf` compiles the IR to `kernel.o`.
4. `riscv64-unknown-elf-gcc` links `kernel.o` with the rendered
   `build_tools/spike/wrapper/main_matmul.c.in` against newlib + pk.
5. `spike --extension=gemmini` (with `LD_LIBRARY_PATH=$RISCV/lib`) runs
   the ELF; stdout is compared against the `.expected` file.

## Generating reference outputs

`*.expected` files are computed by hand with numpy using the same input
pattern as the C wrapper:

```python
import numpy as np
M = K = N = 8           # match the .mlir fixture's shape
A = np.fromfunction(lambda i, j: ((i + j) & 0x7F).astype(np.int8),
                    (M, K), dtype=np.int8)
B = np.eye(K, N, dtype=np.int8)            # identity for ground-truth
C = (A.astype(np.int32) @ B.astype(np.int32))
print("\n".join(" ".join(str(int(x)) for x in row) for row in C))
```

The matching `main_*.c.in` wrapper writes the same input pattern, runs the
kernel, then prints `MERLIN_SPIKE_OUT_BEGIN ... MERLIN_SPIKE_OUT_END` —
the harness extracts that block and diffs it against the `.expected`
file.

## Skipping cleanly

`conftest.py` skips the suite when any of these are missing:
- `$RISCV/bin/{spike, riscv64-unknown-elf-gcc}`
- `$RISCV/riscv64-unknown-elf/bin/pk`
- `build/host-merlin-debug/install/bin/{iree-opt, mlir-translate}`

## Currently in scope

- DIM=16, int8 / int32 OS dataflow (matches the prebuilt
  `libgemmini.so` shipped with chipyard's `riscv-tools` env).

## Out of scope (future work)

- mxGemmini FP4/FP6/FP8 — needs a rebuilt `libgemmini.so`.
- conv2d / requantize fixtures — the corresponding wrappers are not yet
  authored. The `--kind conv2d` / `--kind requantize` argparse choices are
  reserved for those.
- Bufferization of linalg-domain `gemmini.matmul` (tensor-result) onto
  `gemmini.tile_matmul` (memref-only). Today the lit harness exercises the
  full pipeline by feeding `memref`-typed `linalg.matmul` directly. The
  64x64x64 pytest is `xfail` until this gap closes.
