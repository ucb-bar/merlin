# Gemmini hardware bring-up reference (what ships with the RTL)
You are bringing up a compiler backend for this accelerator. You have:
- `rtl/`           — the Gemmini Chisel RTL (the hardware itself; the ground truth).
- `README.md`      — the Gemmini architecture/ISA overview that ships with the RTL.
- `isa_include/`   — the ISA C headers (`gemmini.h`, `gemmini_params.h`): opcodes, DIM, dtypes.
- `example_kernel/matmul_ws.c` — ONE worked example kernel (single-tile weight-stationary matmul),
  the canonical "hello world" showing how to drive the ISA (mvin/preload/compute/mvout). This is the
  only example you get — generalize from it + the RTL/ISA to all the benchmark ops.
