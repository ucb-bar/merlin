---
title: "Design: external kernels — linking model2MLIR prov.kernel calls into the model"
kind: design
status: current
owner: runtime
last_verified: 2026-10-15
related: [triton_frontend, model2mlir, rvv_e2e]
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
decomposing it, and merlin supplies the kernel's code when it lowers the model. Triton is only the
authoring language; nothing runs on a GPU. The body is the `@triton.jit` kernel exported by
triton-cpu as MLIR in the `llvm` dialect, and it is compiled for the model's target (rv64gcv on
spike) together with the model.

## The contract model2MLIR emits

```mlir
func.func @forward(...) {
  %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
         prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], ...}
       : (tensor<4x16xf32>, ...) -> tensor<4x32xf32>
  ...
}
func.func private @triton_exp__fused_mlp2(...) -> tensor<4x32xf32> attributes {prov.kernel = ...}
```

- `prov.kernel` is the identity. The symbol gets a `_1`, `_2`, … suffix per new shape, so it is never matched.
- Tensor operands come in the op schema's order, named by `prov.kernel_operands`. Non-tensor arguments are in
  `prov.kernel_args`, which is absent when the kernel has none.
- That is the whole contract: m2m records nothing about where a kernel's code lives or what language it is in.
  Those facts belong to merlin's kernel table.
- The declaration has no body, and `@forward` is the only function with one.

## The flow

```
model.mlir ──find_kernel_uses──► KernelUse(symbol, qualname, operand names, types)
     │                                   │ (unknown qualname → KernelLinkError here)
     ▼                                   ▼
preprocess (xDSL | textual) ──link_kernels──► declaration replaced by its wrapper (kernel_wrappers.mlir)
                                              + the body's llvm.func moved into the module
     ▼
upstream pipeline (bufferize … llvm dialect) ─► model.ll ─clang rv64gcv─► model.o ─► spike ELF
```

`lower_model` links whenever the module has a kernel declaration; kernel-free modules skip it after
one `str` check, so they pay no parse and lower byte-identically. The link is one structural rewrite
in the compiler's MLIR bindings (the same `torch_mlir` Python the pipeline runs in). What reaches
the pipeline is an ordinary closed module, so there is no generated C, no second `mlir-translate`
and no `llvm-link`. Every consumer (`codegen.compile_ll`, `build_host_shared`, `spike_model.build`)
compiles `model.ll` unchanged. `spike_model.build` reports the linked kernels as `kernels`.

## The wrapper

A `@triton_op` is a Python wrapper around a `@triton.jit` kernel. It allocates the result, computes
the program grid, and launches the kernel with each tensor's data pointer and its runtime scalars,
and triton-cpu appends six launch integers. `emit_wrappers` writes the same steps as an MLIR function
with the declaration's symbol and type, built only from upstream ops whose bufferization is defined:

```mlir
func.func private @triton_exp__fused_mlp2(%a0: tensor<4x16xf32>, ...) -> (tensor<4x32xf32>)
    attributes {llvm.linkage = #llvm.linkage<internal>, prov.kernel = "triton_exp::fused_mlp2"} {
  %b0 = bufferization.to_buffer %a0 read_only : tensor<4x16xf32> to memref<4x16xf32>   // per operand
  %r0 = memref.alloc() {alignment = 64 : i64} : memref<4x32xf32>                      // per result
  %p0 = llvm.inttoptr (memref.extract_aligned_pointer_as_index %b0) ...                // per pointer arg
  %p6 = arith.constant 4 : i32                                                         // M = x.shape[0]
  scf.for %pid2 ... scf.for %pid1 ... scf.for %pid0 = %c0 to %g0 ...                   // cdiv(M, 16) programs
    llvm.call @_mlp2_kernel(%p0, ..., %p5, %p6, %pid0, %pid1, %pid2, %g0, %g1, %g2)
  %t0 = bufferization.to_tensor %r0 restrict writable : memref<4x32xf32> to tensor<4x32xf32>
  return %t0 : tensor<4x32xf32>
}
```

The call sites stay exactly as model2MLIR wrote them. Shapes are static, so every extent and grid
size is a constant and the trip-1 loops fold away.

## The kernel table (`llvmlower/kernel_table.py`)

Keyed by `prov.kernel`, the same qualname as m2m's `KernelSpec`. `test_table_keys_match_m2m_registry` keeps
the two tables in step. Each entry has one `KernelBody`:

| field | meaning |
|---|---|
| `mlir` | path under `src/merlin/kernels/` (shipped as package data) of the triton-cpu export in the `llvm` dialect |
| `entry` | the `llvm.func` it defines |
| `args` | its arguments before the launch tail, in order: `operand` (data pointer by name), `result` (data pointer of a result the wrapper allocates), `dim` (an extent, as `i32`) |
| `grid` | per program axis, `(operand, axis, block)`: `ceil(operand.shape[axis] / block)` programs |
| `specialization` | extents the body was compiled for, so a use at another shape is refused |

`args` and `grid` transcribe the `@triton_op` wrapper's launch. For `triton_exp::fused_mlp2` the
wrapper runs `_mlp2_kernel[(cdiv(M, 16),)](x, w1, b1, w2, b2, out, M, K=16, H1=32, H2=32, BLOCK_M=16)`,
and the body (`kernels/triton_llvmir/mlp2_wrapper_llvm.mlir`) is that `_mlp2_kernel`, exported with its
Python name and its `tt.divisibility` attributes. The constexprs are compiled in. The tests lower a
call of the kernel on the host and compare it with numpy at M = 4, 16 and 20 (one program, one full
block, and two programs with a partial tail).

## Why each piece is shaped this way

- **The link happens in MLIR, before the pipeline.** A body in the `llvm` dialect can be merged into
  the model module as-is, so the pipeline's own passes do what a hand-written adapter had to:
  - *No weight copies.* One-shot-bufferize sees the wrapper's body, so `to_buffer read_only` tells it
    the operands are only read. A body-less declaration needed a `bufferization.access` text patch
    for the same effect, and that patch had to handle two printed spellings.
  - *The caller owns the result.* `buffer-results-to-out-params{hoist-static-allocs}` hoists the
    wrapper's `memref.alloc` into `@forward`, so the kernel writes straight into the caller's buffer and
    the model's deallocation frees it.
  - *Only dense buffers reach the kernel.* The wrapper's arguments are identity-layout memrefs
    (`function-boundary-type-conversion=identity-layout-map`). An operand with any other layout is
    copied dense by bufferization at the call, never handed to a raw-pointer kernel.
    `test_strided_operand_reaches_kernel_dense` passes every other row of `x` and checks the result.
- **The merge is structural.** It runs in the compiler's MLIR bindings and replaces the declaration
  by symbol, so it does not matter which preprocess path printed the declaration, or in which form.
- **The kernel's signature is the ABI check.** The wrapper calls the body's `llvm.func`, and the merged
  module is verified. A table entry whose `args` disagree with the export fails at link time, naming
  the kernel and the mismatched operand, before any pass runs.
- **The wrapper has internal linkage.** `func.func private` does not reach LLVM (convert-func-to-llvm
  drops visibility). After clang inlines the wrapper into `forward`, an external copy would stay in
  the object as dead code.
- **No locations are carried.** The merged module is printed without locations, so a body's debug
  info (the exporting checkout's absolute source paths) stays in the tracked file and never reaches
  a lowered model or an image.
- **`tt.divisibility = 16` is not checked.** It is Triton's record that the export was specialized for
  16-byte-aligned pointers. Every pointer the wrapper passes is the aligned base of an identity-layout
  buffer. Intermediates come from a 64-byte `memref.alloc`, and spike's allocator aligns to 64. A weight
  sits at the blob base plus its safetensors offset, though, so it is only as aligned as those offsets
  are. The `mlp2` body's loads and stores assume only 4-byte alignment, so this body does not depend
  on it; a body that did would need the wrapper to check, or the blob to be padded.

## A latent runtime bug the first adapter exposed

The first version of this design reached the kernel through a generated C adapter that checked each
memref descriptor was dense before calling the body. That check found a real runtime bug.

`merlin_descriptor_t` reserves `MERLIN_MAX_RANK` (8) slots for sizes and for strides. MLIR, however,
reads a rank-r descriptor packed: `{alloc, aligned, offset, sizes[r], strides[r]}`.
`fill_descriptor` wrote `d->strides[i]`, so for every rank below 8 MLIR read the strides out of the
unused size slots, which were zeros.

Generated code with static shapes never reads runtime strides, which is why nothing noticed. The
adapter's layout check did: on spike, `@forward`'s input arrived as sizes `[4, 16]`, strides `[0, 0]`,
and the adapter trapped (`mcause=2` at its `unimp`). Strides are now written at `sizes[r..2r-1]`, covered
by `merlin/tests/runtime/test_descriptor_layout.py`. `memrefCopy` reads the same positions, so it was
exposed too.

The `kernels/ceiling_drivers/*.c` microbenchmark drivers still use the field-wise pattern. (`main` fixed `merlin_model.c` independently, with the same packing.)

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
with the PyTorch eager golden. On spike, the `cycles` counter counts retired instructions, not time.

| variant | max abs err vs torch | spike `cycles` | `.text` | kernel code |
|---|---|---|---|---|
| mlp_basic (no kernel) | 4.5e-8 | 16,156 | 10.3 KB | — |
| mlp_basic_triton, MLIR-linked `_mlp2_kernel` (this design) | 3.0e-8 | **15,121** | 44.9 KB | 37.6 KB |
| mlp_basic_triton, C adapter + `llvm-link`, `llvm_mlir` body (before 2026-10-15) | 3.0e-8 | 15,558 | 46.2 KB | 37.6 KB |
| mlp_basic_triton, C adapter + `llvm-link`, `llvm_ir` body (before 2026-10-15) | 3.0e-8 | 44,358 | 150 KB | 141 KB |

The two `llvm_mlir` rows run the same kernel. The export differs only in the function name, debug
locations and `tt.divisibility`, so the compiled kernel is the same 9,983 instructions in the same
order, it retires 12,297 instructions either way, and the outputs are bit-identical. Every one of the
437 instructions saved is on the launch path, attributed from a spike instruction log:

| retired in | C adapter | MLIR wrapper |
|---|---|---|
| the C adapter (descriptor density checks, result descriptor, grid) | 113 | — (inlined into `forward`, one call) |
| `malloc` / `free` | 212 / 7 | 67 / 2 |
| `forward` + `_mlir_ciface_forward` | 2,458 + 61 | 2,323 + 22 |

Most of the saving is in deallocation. The old declaration had no body and RETURNED its result buffer,
so `ownership-based-buffer-deallocation` could not prove that buffer distinct from `forward`'s own
allocation. It fell back to a runtime alias check: a `dealloc_helper` call plus five scratch heap
arrays (`memref<2xindex>`, `memref<2xi1>`, …). Now the result is an out-param `forward` allocates
itself, so ownership is static and the dealloc is a plain `free`. `forward` also stops building six
memref descriptors for a C-interface call.

The `llvm_ir` row is history. That export was already optimized and unrolled for the x86 host it came
from, with `<32 x float>` operations throughout, which legalizes into 14.7k vector instructions on
128-bit RVV. It is why bodies are now accepted only in the `llvm` dialect, before LLVM optimization.

The `mlp_basic` row is from the bundles in `merlin/tests/data/mlp_bundles/`, regenerated from current
model2MLIR, whose relu lowering is `cmpf`+`select` rather than `maximumf`. The 2026-10-14 report's
bundle gave 16,180.

## Limits and next steps

Each gap below is refused by name today. None of them produces a silently wrong answer.

- **Kernels that write into an argument** (`mutates_args`, e.g. a Triton kernel that stores into an
  `out` tensor the caller passes in). `torch.export` wraps such an op in
  `auto_functionalized_v2(op, **kwargs, _<arg>_base_index, _all_bases=[...])`, and model2MLIR
  currently rejects a registered one at import.
  - Planned model2MLIR side:
    1. rebuild the schema argument list, taking the mutated argument from `_all_bases[...]` (v2) or
       `kwargs` (v1);
    2. refuse view arguments (`_<arg>_size`/`_stride`);
    3. emit the call in destination-passing style: the buffer's initial value is an operand, and the
       results are the op's own returns followed by one new value per mutated argument, tagged
       `prov.kernel_mutated` and `prov.kernel_tied` (the operand each result aliases);
    4. remap `getitem(af, k)` to those results.
    5. Before building it, probe the slot layout for ops that both return tensors and mutate.
  - merlin side: a `KernelArg` kind for a tied operand. The wrapper takes that operand's buffer with a
    writable `to_buffer` (no `read_only`) and returns it through `to_tensor`, instead of allocating, so
    one-shot-bufferize decides between writing in place and copying first.
- **Multiple results** per kernel. The wrapper allocates and returns any number of them, and
  `_check_use` requires the body to write each one exactly once, but no registered kernel has more
  than one result yet.
- **`prov.kernel_args`** (non-tensor arguments such as `alpha` or block sizes). This needs a
  `KernelArg` kind that reads a call-site attribute; the wrapper is per symbol, and m2m keys symbols
  by shape only.
- **Dynamic shapes**, and more than one **specialization per kernel**. A triton-cpu export bakes K,
  H1, H2 and BLOCK_M, so the same qualname at another hidden size needs another body; the table holds
  one per kernel.
- **Body export is not reproducible in this repo.** The export came from a triton-cpu run outside it.
  Its debug paths no longer reach an image, but regenerating it still needs that setup.
- **LLVM IR (`.ll`) bodies are not accepted.** A body has to be in the `llvm` dialect to be merged
  before lowering. Bringing in a `.ll` would need `mlir-translate --import-llvm` first, and an
  already-optimized host export was 2.9× slower on RVV anyway (the `llvm_ir` row above).
- `runtime.kernel_spike` cannot run these models (no allocator; kernel-free captures fail on
  `malloc` there too). The host dispatch runtimes refuse external kernels.
- merlin's own Triton frontend ([triton_frontend](triton_frontend.md)) could produce a `linalg` body
  from the same `@triton.jit` source and inline it before lowering, instead of linking the triton-cpu
  export. `_mlp2_kernel` needs five frontend features first: transposed loads, `expand_dims` on loads,
  `arith.maxnumf`, `tt.dot` at grid > 1, and a runtime `M`.
