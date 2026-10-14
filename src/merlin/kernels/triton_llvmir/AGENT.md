# AGENT.md — src/merlin/kernels/triton_llvmir

## Purpose

Bodies for the external kernels a model2MLIR capture calls (`prov.kernel` `func.call`s). They are not
Python; `llvmlower/kernel_table.py` names each file, its entry symbol and its calling convention, and
`llvmlower/kernel_link.py` merges the chosen body into the model's LLVM IR at lowering time.

## Files

- `mlp2.ll` — `triton_exp::fused_mlp2` (model2MLIR `workloads/mlp_basic_triton`) as triton-cpu LLVM IR
  (`form="llvm_ir"`). Exported on an x86 host, so it carries an x86 triple/datalayout and is already
  optimized and unrolled for that host; the code itself is target-neutral (generic `llvm.*` intrinsics
  only), which is what `portable=True` in the table records.
- `mlp2_llvm.mlir` — the same kernel as MLIR in the `llvm` dialect, before LLVM optimization
  (`form="llvm_mlir"`). Translated with `mlir-translate --mlir-to-llvmir` at link time; carries no triple.

Both export `fused_mlp2_kernel(x, w1, b1, w2, b2, out, i32 M, i32 pid_x, pid_y, pid_z, grid_x, grid_y,
grid_z)`: triton-cpu's convention (kernel arguments, then the six launch integers). K=16, H1=H2=32 and
BLOCK_M=16 are compiled in. Weights are in nn.Linear's `[out, in]` layout.

## Provenance

Exported with triton-cpu from `ttir2linalg/examples/triton_cpu_mlp_export/mlp_kernel.py` (the `loc`
paths inside `mlp2_llvm.mlir`), the CPU twin of `_mlp2_kernel` in model2MLIR's
`workloads/mlp_basic_triton/kernels.py`. Keep the files byte-identical to the export; regenerate from
the Triton source rather than editing them by hand.

## Invariants

- A file registered `portable=True` must contain no arch intrinsic namespace (`llvm.x86.*`,
  `llvm.nvvm.*`, …), no inline asm and no address spaces — checked by
  `merlin/tests/kernels/test_kernel_table.py`. An NVPTX/GPU export is not portable and cannot be
  linked into a CPU image.
- Adding a body = adding the file here + a `KernelBody` in `kernel_table.py` whose `abi` is verified
  by driving it on the host (same test file).
