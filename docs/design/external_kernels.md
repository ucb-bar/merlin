---
title: "Design: external kernels — linking model2MLIR prov.kernel calls into the model"
kind: design
status: current
owner: runtime
last_verified: 2026-10-13
related: [triton_frontend, model2mlir, rvv_e2e, llvm_toolchain]
code_refs:
  - src/merlin/llvmlower/kernel_table.py
  - src/merlin/llvmlower/kernel_link.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/common/kernel_calls.py
  - src/merlin/runtime/backends/spike_model.py
  - merlin/runtime/c/merlin_model.c
---

# External kernels

A model can call a kernel it does not define. model2MLIR captures a registered custom op — today a
Triton `@triton_op` such as `torch.ops.triton_exp.fused_mlp2` — as a **black-box call** instead of
decomposing it, and merlin supplies the kernel's code at link time. Triton is only the authoring
language; nothing runs on a GPU, and the body that gets linked is CPU code (rv64gcv on spike).

## The contract model2MLIR emits

```mlir
func.func @forward(...) {
  %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
         prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], prov.kernel_args = {}, ...}
       : (tensor<4x16xf32>, ...) -> tensor<4x32xf32>
  ...
}
func.func private @triton_exp__fused_mlp2(...) -> tensor<4x32xf32> attributes {prov.kernel = ...}
```

- `prov.kernel` is the identity. The symbol gets a `_1`, `_2`, … suffix per new shape, so it is never matched.
- Tensor operands come in the op schema's order, named by `prov.kernel_operands`. Non-tensor arguments are in `prov.kernel_args`.
- The declaration has no body, and `@forward` is the only function with one.

## The flow

```
model.mlir ──find_kernel_uses──► KernelUse(symbol, qualname, operand names, types)
     │                                   │ (unknown qualname → KernelLinkError here)
     ▼                                   ▼
preprocess (xDSL | textual) ──prepare_kernel_declarations──► emit_c_interface + read access on args
     ▼
upstream pipeline (bufferize … llvm dialect) → model.ll        calls _mlir_ciface_<sym>(res*, arg*...)
     ▼
emit_shim ─► kernel_shim.c ─clang -S -emit-llvm─► kernel_shim.ll
body (kernel_table) ─[mlir-translate]─[drop triple if portable]─► body.ll
llvm-link kernel_shim.ll model.ll body.ll ─► model.linked.ll ─clang rv64gcv─► model.o ─► spike ELF
```

`lower_model` does all of this whenever the module has a kernel declaration. Kernel-free modules skip
it after one `str` check: they pay no parse and lower byte-identically. Every existing consumer
(`codegen.compile_ll`, `spike_model.build`) compiles `res.ll_path` (the linked module) without
change. `spike_model.build(..., kernel_form=)` selects the body form and reports `kernels`.

## The kernel table (`llvmlower/kernel_table.py`)

Keyed by `prov.kernel`, mirroring m2m's `KernelSpec`. `test_table_keys_match_m2m_registry` keeps the
two tables in step. Each entry lists one or more `KernelBody`s:

| field | meaning |
|---|---|
| `form` | `llvm_ir` (textual `.ll`) or `llvm_mlir` (MLIR `llvm` dialect, `mlir-translate`d first) |
| `ir` | path under `src/merlin/kernels/` (shipped as package data) |
| `entry` | the symbol the body defines |
| `abi` | its parameters in order: `operand` (data pointer by name), `result`, `dim` (runtime extent), `program_id` / `grid` (launch coordinates), `null` |
| `block` | `(operand, axis, size)`: one call per block of rows (the Triton program grid) |
| `specialization` | extents the body was compiled for, so a use at another shape is refused |
| `portable` | the IR is target-neutral (no arch intrinsics, inline asm or address spaces), so the exporter's triple/datalayout are dropped and it links for any target |

`triton_exp::fused_mlp2` has two bodies exported by triton-cpu: `kernels/triton_llvmir/mlp2.ll` and
`mlp2_llvm.mlir`. Both take `(x, w1, b1, w2, b2, out, i32 M, pid_x, pid_y, pid_z, grid_x, grid_y,
grid_z)` with K=16, H1=H2=32 and BLOCK_M=16 compiled in. The tests drive each body through its
own `abi` on the host and compare against numpy (M = 4, 16, 20).

## Why each piece is shaped this way

- **One declaration ABI, whichever preprocess path ran.** Without the normalization, the xDSL path
  marked the declaration `emit_c_interface`, so the call took one pointer per descriptor. The textual
  path, which `spike_model` uses, did not, so the call took 31 expanded scalars. The shim would have
  had to match whichever path happened to run.
- **`bufferization.access = "read"` on every argument.** Without it, one-shot-bufferize copied all
  four weights before the call (4 of the module's 5 `llvm.memcpy`). It is applied as a text patch
  because xDSL drops `arg_attrs` when printing a body-less declaration, the same repair
  `passes_opu.patch_declaration_arg_attrs` makes.
- **The shim allocates the result.** m2m's declaration returns a fresh tensor, and
  `buffer-results-to-out-params` leaves body-less callees alone. So the callee owns the result
  allocation exactly as an MLIR function would, and the model's deallocation passes free it.
- **The shim checks the descriptors are dense row-major and traps otherwise.** A raw-pointer body
  can index nothing else. This check found a real bug, described next.
- **`llvm-link` before clang, not a separate object.** The result is one module per image, which
  lets LLVM inline the shim into `forward`. The shim goes first on the link line, so the merged
  module takes the target's triple.

## A latent runtime bug the shim exposed

`merlin_descriptor_t` reserves `MERLIN_MAX_RANK` (8) slots for sizes and for strides. MLIR, however,
reads a rank-r descriptor packed: `{alloc, aligned, offset, sizes[r], strides[r]}`.
`fill_descriptor` wrote `d->strides[i]`, so for every rank below 8 MLIR read the strides out of the
unused size slots, which were zeros.

Generated code with static shapes never reads runtime strides, which is why nothing noticed. The
shim's layout check did: on spike, `@forward`'s input arrived as sizes `[4, 16]`, strides `[0, 0]`,
and the shim trapped (`mcause=2` at its `unimp`). Strides are now written at `sizes[r..2r-1]`, covered
by `merlin/tests/runtime/test_descriptor_layout.py`. `memrefCopy` reads the same positions, so it was
exposed too.

The `kernels/ceiling_drivers/*.c` microbenchmark drivers still use the field-wise pattern.

## The staged (dialect) path

Until a backend dialect claims an external kernel, the staged passes treat the call as opaque:

- the outliner keeps the call in the driver and carries the declaration into its output;
- the dispatch program records a `kind="external"` node, never paired with an outlined dispatch;
- `support_ops` never absorbs the call. It used to: the call's only consumer is fc3's matmul, so the
  call looked like a droppable init, and materialization would have deleted two layers silently;
- `lower_to_interface` refuses the module by name;
- `compile_core.choose_route` sends the module to the LLVM route;
- the host dispatch runtimes raise `ExternalKernelNotExecutable`.

## Measured (spike rv64gcv, VLEN 128)

`mlp_basic` (plain linalg) and `mlp_basic_triton` (fc1+relu+fc2+relu as the linked kernel), compared
with the PyTorch eager golden. The report lives under `out/artifacts/measurements/spike_rv64gcv/mlp_basic_triton/`.

| variant | max abs err vs torch | spike `cycles` | `.text` | kernel body size |
|---|---|---|---|---|
| mlp_basic (no kernel) | 4.5e-8 | 18,824 | 9.1 KB | — |
| mlp_basic_triton, `llvm_ir` body | 3.0e-8 | 44,356 | 148 KB | 141 KB |
| mlp_basic_triton, `llvm_mlir` body | 3.0e-8 | 15,556 | 45 KB | 37.6 KB |

On spike, the `cycles` counter counts retired instructions, not timing. The two body forms produce
bit-identical outputs.

The `llvm_ir` export was already optimized and unrolled for the x86 host it came from, with
`<32 x float>` operations throughout. Legalized onto 128-bit RVV, it becomes 14.7k vector
instructions. The `llvm_mlir` form is optimized by clang for rv64gcv itself and comes out 3.7× smaller
and 2.9× faster. For a CPU target, prefer the pre-optimization `llvm_mlir` export.

## Limits and next steps

- One result per kernel, static shapes only, and no `prov.kernel_args` passing. Each is refused by
  name today.
- Output-buffer (`mutates_args`) kernels. m2m rejects them at import; the planned destination-passing
  design is in model2MLIR's notes.
- `runtime.kernel_spike` cannot run these models: its harness links no allocator, and kernel-free
  captures fail on `malloc` there too.
- merlin's own Triton frontend ([triton_frontend](triton_frontend.md)) could produce a `linalg` body
  from the same `@triton.jit` source and inline it before lowering, instead of linking LLVM IR. That
  would be a third body form on the same table.
