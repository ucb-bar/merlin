# Saturn OPU Matmul Profiling

This folder contains compile-only profiling helpers for OPU-oriented matmul
flows using `tools/compile.py` with `--dump-artifacts`.

## Scripts

- `compile_matmul_opu_i8_ukernel_all.sh`
  - Compiles an i8 matmul workload for `models/saturn_opu.yaml` (`--hw OPU`).
  - Uses ukernel/data-tiling enabled mode.
  - Checks the first hot loop in the emitted `.s` file for:
    - required OPU opcodes, and
    - absence of stack spill accesses in the loop body.

## Usage

```bash
./benchmarks/SaturnOPU/compile_matmul_opu_i8_ukernel_all.sh
```

Or pass a custom input MLIR:

```bash
./benchmarks/SaturnOPU/compile_matmul_opu_i8_ukernel_all.sh /abs/path/to/matmul.mlir
```
