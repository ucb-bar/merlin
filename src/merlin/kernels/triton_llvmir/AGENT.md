# AGENT.md — src/merlin/kernels/triton_llvmir

## Purpose

Bodies for the external kernels a model2MLIR capture calls (`prov.kernel` `func.call`s). They are not
Python; `llvmlower/kernel_table.py` names each file, its entry `llvm.func` and how the `@triton_op`
wrapper launches it, and `llvmlower/kernel_link.py` merges the body into the model's MLIR, beside a
generated wrapper, before the upstream pipeline lowers it.

## Files

- `mlp2_wrapper_llvm.mlir` — `triton_exp::fused_mlp2` (model2MLIR `workloads/mlp_basic_triton`): the
  `_mlp2_kernel` its `@triton_op` wrapper launches, exported by triton-cpu as MLIR in the `llvm` dialect,
  before LLVM optimization. It exports `_mlp2_kernel(x, w1, b1, w2, b2, out, i32 M, i32 pid_x, pid_y,
  pid_z, grid_x, grid_y, grid_z)`: the `@triton.jit` arguments, then triton-cpu's six launch integers.
  K=16, H1=H2=32 and BLOCK_M=16 are compiled in. Weights are in nn.Linear's `[out, in]` layout. The
  pointer arguments carry Triton's `tt.divisibility = 16`, which the link does not check; this body's
  loads and stores need only 4-byte alignment (see `docs/design/external_kernels.md`).

## Provenance

Exported with triton-cpu from model2MLIR's `workloads/mlp_basic_triton/kernels.py` (the `loc` paths
inside the file). Keep the file byte-identical to the export and regenerate it from the Triton source
rather than editing it by hand. The link prints the merged module without locations, so the
exporter's paths stay in this file and never reach a lowered model.

## Invariants

- Every body is MLIR in the `llvm` dialect. There is no LLVM IR (`.ll`) form: a body is merged into the
  model module and code-generated for the model's target with it.
- A body must be target-neutral: no arch intrinsic namespace or dialect (`llvm.x86.*`, `x86vector`,
  `nvvm`, …), no inline asm and no address spaces — checked by
  `merlin/tests/kernels/test_kernel_table.py`. An NVPTX/GPU export cannot be linked into a CPU image.
- Adding a body = adding the file here + a `KernelBody` in `kernel_table.py` whose `args`/`grid`
  transcribe the `@triton_op` wrapper's launch, verified by lowering a call of it on the host against
  numpy (same test file). A wrong argument list fails MLIR verification at link time.
