# Atlas NPU hardware bring-up reference (what ships with the RTL)

You are bringing up a compiler backend for this accelerator. You have:

- `rtl/`           — a curated, representative subset of the Atlas Chisel RTL (the hardware itself; the
  ground truth). The MXU datapath (`mxu/`), the shape/param facts (`common/`), and the self-hosted
  scalar core's ISA/decoder (`scalar/`) are here. `rtl/README_FULL_TREE.md` points at the full external
  tree for anything not curated in.
- `README.md`      — the Atlas architecture/ISA overview distilled from the Atlas NPU design spec.
- `isa_include/`   — the ISA reference the agent reads:
  - `atlas_isa_green_card.md` — the instruction "green card": opcode/format tables, register set, and
    the frozen architectural design parameters (DIM, memory map, MXU count, dtypes).
  - `isa_definition.py` — the authoritative, never-hardcoded ISA definition from the Atlas performance
    model (the decorator-based opcode/effect table the model executes against).
- `example_kernel/` — worked example kernels, the canonical "hello world" showing how to drive the ISA:
  - `mxu0_single_output_tile_bf16_k96.S` — a single-output-tile matmul `C = A@B + bias` (K=96, 3 tiles)
    demonstrating the full DMA -> VLOAD -> VTRPOSE.XLU -> VMATPUSH.W/ACC -> VMATMUL.ACC -> VMATPOP ->
    VSTORE -> DMA.STORE pipeline with weight double-buffering.
  - `sweep_mm_32x32x32.S` — a minimal single-tile 32x32x32 matmul.

  Generalize from these + the RTL/ISA to all the benchmark ops.

Atlas is a **self-hosted** accelerator: it fetches its own 32-bit instructions from IMEM behind its own
PC (it is NOT a RoCC co-processor driven by a host CPU). The compiler deliverable is therefore an
**assembled `kernel.S`** (endpoint kind `external_backend`), assembled by the Atlas assembler and run on
the mlc arc cosimulation model (`libatlas_model.so` + `atlas_hw.mlir`) / Verilator — there is no host-side
`.insn` intrinsic stream to emit.
