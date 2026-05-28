# Kernel Embedding — Full Mechanism Walkthrough

A start-to-finish demonstration of how Merlin lets you replace any
linalg-level operation in a model with hand-written or LLM-generated C
code, with real MLIR snippets at every phase. This is the "show, don't
tell" companion to:

- [`embed_custom_kernel_via_manifest.md`](embed_custom_kernel_via_manifest.md) — schema details
- [`extend_kernel_coverage_to_any_model.md`](extend_kernel_coverage_to_any_model.md) — discovery + coverage
- [`inspect_kernel_embedding_phases.md`](inspect_kernel_embedding_phases.md) — IR phase walkthrough

## TL;DR

```
                      manifest.json
                ┌─────────────────────┐
                │  catalog of kernels │
                │  + match patterns   │
                └──────────┬──────────┘
                           │
     ./merlin compile model.mlir --kernels-dir <dir>
                           │
                           ▼
   ┌──────────────────────────────────────────────────────┐
   │ 1. precompile.py: clang on each kernel.c → .o        │
   │ 2. spec_gen.py: emit transform_spec.mlir from manifest│
   │ 3. iree-compile:                                      │
   │    preprocessing → matches linalg ops → flow.dispatch │
   │    flow → stream → hal → iree-lld links the .o       │
   └──────────────────────────────────────────────────────┘
                           │
                           ▼
                       model.vmfb
```

Three things you author: kernel `.c` source, `match.mlir` (or `named_op`
declaration), `manifest.json` entry. The compiler does everything else.

---

## Part 1 — Anatomy of a kernel

A kernel is **three files** plus a manifest entry:

```
benchmarks/SaturnOPU/kernels/
├── manifest.json                   ← (4) the registry
├── abi/add_f32_workgroup.c         ← (1) the C source
└── match/add_f32.match.mlir        ← (2) the linalg-DAG match body
                                       (3) is the manifest entry below
```

### (1) The C source — IREE custom-dispatch ABI

```c
// benchmarks/SaturnOPU/kernels/abi/add_f32_workgroup.c

#include <riscv_vector.h>
#include <stddef.h>

__attribute__((visibility("default")))
void add_f32_workgroup(const float *restrict binding0, size_t binding0_offset,
                       const float *restrict binding1, size_t binding1_offset,
                       float *restrict binding2, size_t binding2_offset,
                       size_t dim, size_t tid) {
  if (tid >= dim) return;
  binding2[binding2_offset + tid] =
      binding0[binding0_offset + tid] + binding1[binding1_offset + tid];
}
```

The signature follows IREE's CPU custom-dispatch convention (mirrors
`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/functions.c`):
each binding lowers to `(ptr, offset)`, then push-constants if any, then
`tid` as the flat output index. One workgroup per output element.

### (2) The match.mlir — what op chain to capture

```mlir
// benchmarks/SaturnOPU/kernels/match/add_f32.match.mlir

^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
  %add = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) {
    ^bb_inner(%a: f32, %b: f32, %_out: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
```

This is the body of `transform.iree.match.cast_compatible_dag_from_root`.
Dynamic shape (`tensor<?xf32>`) so the same matcher handles every concrete
shape variant; `cast_and_call` inserts `tensor.cast` to bridge types.

### (4) The manifest entry — what the compiler reads

```json
{
  "schema_version": 1,
  "kernels": [{
    "name": "saturnopu_add_f32",
    "source": "abi/add_f32_workgroup.c",
    "source_lang": "c",
    "entry_symbol": "add_f32_workgroup",
    "signature": {
      "operands": [
        {"role": "in",  "tensor": "tensor<?xf32>"},
        {"role": "in",  "tensor": "tensor<?xf32>"},
        {"role": "out", "tensor": "tensor<?xf32>"}
      ]
    },
    "match": {
      "kind": "linalg_dag",
      "spec_path": "match/add_f32.match.mlir"
    },
    "targets": ["llvm-cpu-spacemit-x60"]
  }]
}
```

That's everything the user writes for one kernel. The next sections show
what the compiler produces from these inputs.

---

## Part 2 — End-to-end on a 1D add

The synthetic input model:

```mlir
// tests/granularity/fixtures/embed_pipeline/add_input.mlir

!ty = tensor<8xf32>
module {
  func.func @main(%lhs: !ty, %rhs: !ty) -> !ty {
    %empty = tensor.empty() : !ty
    %sum = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%lhs, %rhs : !ty, !ty)
        outs(%empty : !ty) {
      ^bb_inner(%a: f32, %b: f32, %_out: f32):
        %s = arith.addf %a, %b : f32
        linalg.yield %s : f32
    } -> !ty
    return %sum : !ty
  }
}
```

Compile with the kernel manifest plugged in:

```bash
./merlin compile tests/granularity/fixtures/embed_pipeline/add_input.mlir \
  --target spacemit_x60 --hw RVV \
  --kernels-dir benchmarks/SaturnOPU/kernels \
  --kernels-strict-coverage \
  --output-dir build/add_demo/
```

The build directory after compile:

```
build/add_demo/
├── add_input.vmfb                                          ← final bytecode
├── kernels_cache/
│   ├── saturnopu_add_f32.<sha>.riscv64-none-elf.o          ← precompiled .o
│   └── transform_spec.mlir                                 ← auto-generated spec
└── phases/
    ├── add_input.1.input.mlir
    ├── add_input.3.preprocessing.mlir   ← rewrite landed here
    ├── add_input.6.flow.mlir            ← outlined dispatches
    ├── add_input.10.executable-targets.mlir
    └── add_input.12.vm.mlir
```

### Step 1 — precompile

`kernels/core/precompile.py` invokes clang for the spacemit-x60 target:

```bash
clang --target=riscv64-none-elf \
      -march=rv64gcv_zfh_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause \
      -mabi=lp64d -ffreestanding -fvisibility=hidden -fno-plt \
      -fno-rtti -fno-exceptions -c -O3 \
      benchmarks/SaturnOPU/kernels/abi/add_f32_workgroup.c \
      -o build/add_demo/kernels_cache/saturnopu_add_f32.<sha>.riscv64-none-elf.o
```

The `.o` is now ready to link. The sha-keyed cache is content-addressed:
unchanged sources don't recompile.

### Step 2 — auto-generate the transform spec

`kernels/core/spec_gen.py` reads the manifest and emits a single MLIR
file containing every kernel's match-and-replace machinery:

```mlir
// build/add_demo/kernels_cache/transform_spec.mlir  (excerpt)

#kb_target_llvm_cpu_spacemit_x60 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">

#pipeline_layout_saturnopu_add_f32 = #hal.pipeline.layout<constants = 0, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>]>

module attributes {transform.with_named_sequence} {

  // (a) The executable carrying our linked .o.
  hal.executable.source private @kb_saturnopu_add_f32 attributes {
    objects = #hal.executable.objects<{
      #kb_target_llvm_cpu_spacemit_x60 = [
        #hal.executable.object<{path = "saturnopu_add_f32.<sha>.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @add_f32 ordinal(0)
        layout(#pipeline_layout_saturnopu_add_f32)
        count(%device: !hal.device, %workload: index)
        -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %workload, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func private @add_f32_workgroup(memref<?xf32>, memref<?xf32>, memref<?xf32>, index)
          attributes {hal.import.static}
      func.func @add_f32() {
        %c0 = arith.constant 0 : index
        %dim = hal.interface.workgroup.count[0] : index
        %tid = hal.interface.workgroup.id[0] : index
        %binding0 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32)
            binding(0) alignment(64) offset(%c0) : memref<?xf32>{%dim}
        %binding1 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32)
            binding(1) alignment(64) offset(%c0) : memref<?xf32>{%dim}
        %binding2 = hal.interface.binding.subspan layout(#pipeline_layout_saturnopu_add_f32)
            binding(2) alignment(64) offset(%c0) : memref<?xf32>{%dim}
        func.call @add_f32_workgroup(%binding0, %binding1, %binding2, %dim, %tid)
            : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()
        return
      }
    }
  }

  // (b) The dispatch wrapper — what the rewritten linalg op becomes.
  util.func private @call_saturnopu_add_f32(%in0: tensor<?xf32>, %in1: tensor<?xf32>)
      -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %in0, %c0 : tensor<?xf32>
    %0 = flow.dispatch @kb_saturnopu_add_f32::@add_f32[%dim](%in0, %in1)
        : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
    util.return %0 : tensor<?xf32>
  }

  // (c) The matcher — splices match/add_f32.match.mlir verbatim.
  transform.named_sequence @match_saturnopu_add_f32(
      %root: !transform.any_op {transform.readonly})
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
        %c0 = arith.constant 0 : index
        %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
        %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
        %add = linalg.generic { ... } { arith.addf, linalg.yield } -> tensor<?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // (d) The rewriter — substitutes call_saturnopu_add_f32 for the matched op.
  transform.named_sequence @cast_and_call_saturnopu_add_f32(
      %ins: !transform.any_value {transform.readonly},
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @kb_saturnopu_add_f32 into %module
        if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_saturnopu_add_f32 into %module
        if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op)
        -> !transform.any_op
    transform.yield
  }

  // (e) The driver — applies every (matcher, rewriter) pair across the module.
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["util.func"]} in %module
        : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
    ^bb1(%f: !transform.any_op):
      transform.foreach_match in %f
        @match_saturnopu_add_f32 -> @cast_and_call_saturnopu_add_f32
        : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
```

**Five blocks, generated entirely from the manifest. Each plays a
distinct role; remove any one and the rewrite breaks:**

- **(a) `hal.executable.source`** — declares the precompiled object to
  iree-compile's HAL layer so the rewritten dispatch can be linked
  against it without recompiling the kernel. The binding-subspan shim
  is the IREE workgroup ABI; the runtime invokes it once per workgroup
  and it forwards into the kernel's C entry symbol.
- **(b) `util.func @call_*`** — the dispatch wrapper that the transform
  dialect can substitute *in place* of the matched linalg op. Wrapping
  the `flow.dispatch` in a `util.func` lets the rewriter perform a
  tensor-typed call swap; substituting `flow.dispatch` directly would
  fail the type-check at the call site.
- **(c) `transform.named_sequence @match_*`** — the read-only matcher.
  It is your match.mlir spliced verbatim, wrapped in the canonical
  `match.cast_compatible_dag_from_root` scaffold so the matcher tolerates
  shape and dtype variation around the anchor op.
- **(d) `transform.named_sequence @cast_and_call_*`** — the rewriter
  half. Imports the executable + dispatch wrapper into the user module,
  then performs `cast_and_call` so the substituted call gets the right
  tensor shape and dtype at the boundary.
- **(e) `@__transform_main`** — the entry point. Walks every
  `util.func` in the module and pairs each matcher with its rewriter
  via `foreach_match`; finishes with a `transform.apply_dce` so the
  inlined matchers don't leak into post-preprocessing IR.

### Step 3 — iree-compile applies it

iree-compile reads the spec via `--iree-preprocessing-transform-spec-filename`
and `--iree-hal-executable-object-search-path`. The user MLIR runs through
all 12 phases. We highlight three:

#### Phase 3 (after preprocessing) — the rewrite has landed

```mlir
// build/add_demo/phases/add_input.3.preprocessing.mlir  (excerpt)

util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, ...)
    -> !hal.buffer_view ... {
  %c8 = arith.constant 8 : index
  %0 = hal.tensor.import wait(%arg2) => %arg0 "input0" : !hal.buffer_view -> tensor<8xf32>
  %1 = hal.tensor.import wait(%arg2) => %arg1 "input1" : !hal.buffer_view -> tensor<8xf32>
  %2 = flow.tensor.reshape %0 : tensor<8xf32> -> tensor<?xf32>{%c8}
  %3 = flow.tensor.reshape %1 : tensor<8xf32> -> tensor<?xf32>{%c8}
  %4 = util.call @call_saturnopu_add_f32(%2, %3)
       : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  ...
}
```

The original `linalg.generic` is **gone** — replaced by `util.call
@call_saturnopu_add_f32`, which (per the spec) issues a `flow.dispatch`
into our kernel's executable.

#### Phase 6 (flow) — outlined into a dispatch with the .o linked

```mlir
// build/add_demo/phases/add_input.6.flow.mlir  (excerpt)

module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.func private @call_saturnopu_add_f32(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
      -> tensor<?xf32> {
    %c8 = arith.constant 8 : index
    %0 = flow.dispatch @kb_saturnopu_add_f32::@add_f32[%c8](%arg0, %arg1)
         : (tensor<?xf32>{%c8}, tensor<?xf32>{%c8}) -> tensor<?xf32>{%c8}
    util.return %0 : tensor<?xf32>
  }
  hal.executable.source private @kb_saturnopu_add_f32 attributes {
    objects = #hal.executable.objects<{
      #executable_target_embedded_elf_riscv_64 = [
        #hal.executable.object<{path = "saturnopu_add_f32.<sha>.riscv64-none-elf.o"}>
      ]
    }>
  } {
    hal.executable.export public @add_f32 ordinal(0) layout(#pipeline_layout)
        count(...) { ... }
    builtin.module {
      func.func private @add_f32_workgroup(...) attributes {hal.import.static}
      func.func @add_f32() { ... binding.subspan + call ... }
    }
  }
}
```

The flow phase has materialized the dispatch and carried the linked `.o`
forward.

#### Verifying the result

```bash
$ ls -la build/add_demo/add_input.vmfb
-rw-rw-r-- 1 ... 10204 ... build/add_demo/add_input.vmfb

$ build/host-vanilla-release/tools/iree-dump-module \
    build/add_demo/add_input.vmfb | head -3
//===----------------------------------------------------------------------===//
// @module : version 0
//===----------------------------------------------------------------------===//

$ ./merlin compile ... --kernels-strict-coverage ...
  ✅ Successfully compiled: build/add_demo/add_input.vmfb
  ✅ kernels-strict-coverage: 0 unmatched dispatches (100% kernel coverage)
```

`--kernels-strict-coverage` walked phase 5 (dispatch-creation) and verified
**every** dispatch in the model is now a kernel call, with no IREE-codegen
fallback.

---

## Part 3 — Push constants for shape arguments (matmul)

The 1D add doesn't need shape arguments — the wrapper recovers `dim` from
`hal.interface.workgroup.count[0]`. For matmul we need M, K, N as runtime
values. Manifest extension:

```json
{
  "name": "saturnopu_linear_f32",
  "source": "abi/linear_f32_workgroup.c",
  "source_lang": "c",
  "entry_symbol": "linear_f32_workgroup",
  "signature": {
    "operands": [
      {"role": "in",  "tensor": "tensor<?x?xf32>"},
      {"role": "in",  "tensor": "tensor<?x?xf32>"},
      {"role": "out", "tensor": "tensor<?x?xf32>"}
    ],
    "constants": [
      {"name": "M", "type": "i32", "from": {"input": 0, "dim": 0}},
      {"name": "K", "type": "i32", "from": {"input": 0, "dim": 1},
        "aliases": [{"input": 1, "dim": 1}]},
      {"name": "N", "type": "i32", "from": {"input": 1, "dim": 0}}
    ],
    "output_dims": ["M", "N"]
  },
  "match": {
    "kind": "linalg_dag",
    "spec_path": "match/linear_f32.match.mlir"
  },
  "targets": ["llvm-cpu-spacemit-x60"]
}
```

`signature.constants` — runtime dim values transported as i32 push
constants. Each one declares its `from` (input index + dim index) and
optional `aliases` for dims that share the same logical value (matmul's
K = lhs.dim(1) = rhs.dim(1) for the transposed-B form).

`signature.output_dims` — explicit map output-dim → constant name. Needed
when the output's dynamic dims aren't trivially `input[0].dim(k)`.

After rewrite (phase 6), the dispatch carries the constants as the
**leading** operands:

```mlir
// build/<demo>/phases/<model>.6.flow.mlir

util.func private @call_saturnopu_linear_f32(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c512 = arith.constant 512 : index
  %c16_i32 = arith.constant 16 : i32      ← N
  %c64_i32 = arith.constant 64 : i32      ← K
  %c8_i32  = arith.constant 8  : i32      ← M
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %0 = flow.dispatch @kb_saturnopu_linear_f32::@linear_f32[%c512]
       (%c8_i32, %c64_i32, %c16_i32, %arg0, %arg1)
       : (i32, i32, i32, tensor<?x?xf32>{%c8, %c64}, tensor<?x?xf32>{%c16, %c64})
       -> tensor<?x?xf32>{%c8, %c64}
  util.return %0 : tensor<?x?xf32>
}
```

Inside the executable wrapper, the constants are decoded back to `index`
values via `hal.interface.constant.load`:

```mlir
func.func @linear_f32() {
  %c0 = arith.constant 0 : index
  %M_i32 = hal.interface.constant.load layout(...) ordinal(0) : i32
  %M = arith.index_cast %M_i32 : i32 to index
  %K_i32 = hal.interface.constant.load layout(...) ordinal(1) : i32
  %K = arith.index_cast %K_i32 : i32 to index
  %N_i32 = hal.interface.constant.load layout(...) ordinal(2) : i32
  %N = arith.index_cast %N_i32 : i32 to index
  %tid = hal.interface.workgroup.id[0] : index
  %binding0 = hal.interface.binding.subspan ... : memref<?x?xf32>{%M, %K}
  %binding1 = hal.interface.binding.subspan ... : memref<?x?xf32>{%N, %K}
  %binding2 = hal.interface.binding.subspan ... : memref<?x?xf32>{%M, %N}
  func.call @linear_f32_workgroup(%binding0, %binding1, %binding2, %M, %K, %N, %tid) : ...
  return
}
```

The C kernel sees `(ptr0, off0, ptr1, off1, ptr2, off2, M, K, N, tid)` —
the IREE bareptr lowering expands each memref to `(ptr, offset)`.

---

## Part 4 — Named-op matching (matmul, conv, pool)

For ops whose body is implied by their op name (named linalg ops),
`match.kind: "named_op"` skips the hand-written `match.mlir` entirely:

```json
{
  "name": "saturnopu_conv_2d_nchw_fchw_f32",
  ...
  "match": {
    "kind": "named_op",
    "op_name": "linalg.conv_2d_nchw_fchw",
    "outs_from_input": 2,
    "op_attrs": "{dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}"
  }
}
```

The spec_gen synthesizes the canonical match scaffold:

```mlir
transform.named_sequence @match_saturnopu_conv_2d_nchw_fchw_f32(
    %root: !transform.any_op {transform.readonly})
    -> (!transform.any_value, !transform.any_value) {
  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
  ^bb0(%in0: tensor<?x?x?x?xf32>, %in1: tensor<?x?x?x?xf32>, %in2: tensor<?x?x?x?xf32>):
    %op = linalg.conv_2d_nchw_fchw
        {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
        ins(%in0, %in1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
        outs(%in2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  } : ...
  transform.yield %ins, %outs : !transform.any_value, !transform.any_value
}
```

`outs_from_input: 2` says input #2 is bound to `outs(...)` (the dronet
case where a broadcasted bias is fed into the conv accumulator). `op_attrs`
splices the required structural attributes verbatim.

---

## Part 5 — Multi-kernel demo on dronet

```bash
./merlin compile models/dronet/dronet.mlir \
  --target spacemit_x60 --hw RVV \
  --kernels-dir benchmarks/SaturnOPU/kernels \
  --dump-phases --output-dir build/dronet_demo/ \
  --iree-compile-arg='--iree-opt-data-tiling=false'
```

Coverage report from `phase_dumps/dronet_partial/COVERAGE.txt`:

```
--- Kernel call sites in flow phase (one line per actual rewrite) ---
      7 util.call @call_saturnopu_conv_2d_nchw_fchw_f32
      2 util.call @call_saturnopu_matmul_f32
      1 util.call @call_saturnopu_pooling_nchw_max_f32

--- Unmatched linalg ops (still in dispatch-creation) ---
     21 linalg.generic
      3 linalg.fill
```

**3 kernels firing simultaneously, 10 rewrites in total.** Same `manifest.json`,
multiple kernel registrations cooperating on one model.

---

## Part 6 — Discovery: bootstrap manifests automatically

For a model you've never touched:

```bash
python -m tools.kernels.discover models/dronet/dronet.mlir \
    --target spacemit_x60 --hw RVV --output kernels/ \
    --iree-compile-arg='--iree-opt-data-tiling=false'
```

Output (excerpt):

```
Wrote 7 complete kernel entries to kernels/manifest.json
  📝 4 stubs need authoring — see kernels/STUBS.md
  ⏭  2 skipped (no tensor inputs — e.g. linalg.fill)

Discoveries — ranked by impact (occurrences × output elements):
    1x  impact=     401,408  cum=  2.3%  linalg.conv_2d_nchw_fchw  -> tensor<1x32x112x112xf32>
    2x  impact=     193,600  cum=  7.0%  linalg.generic#mulf#parallel_parallel_parallel_parallel
    1x  impact=      96,800  cum=  9.3%  linalg.fill
    1x  impact=      96,800  cum= 11.6%  linalg.pooling_nchw_max
    1x  impact=      96,800  cum= 14.0%  linalg.generic#subf#parallel_parallel_parallel_parallel
    1x  impact=      96,800  cum= 16.3%  linalg.generic#addf#parallel_parallel_parallel_parallel
    ...
```

For each unique `(op_name, body_class, signature_shape)`, discovery emits:

- `abi/discovered_<name>_workgroup.c` — **complete C body** for recognized
  patterns (rsqrt, sqrt, exp, log, addf, subf, mulf, divf, maxf, minf,
  negf, absf, identity, relu) using compiler builtins so it works under
  `-ffreestanding`.
- `match/discovered_<name>.match.mlir` — auto-generated linalg-DAG match
  scaffold with dynamic-shape inputs, the recognized body verbatim, and
  the right indexing maps.
- A manifest entry with `signature.constants` and `output_dims` filled in.

Stubs (named ops needing op-specific bodies, broadcast inputs, etc.) get
parked in `stubs/` and listed in `STUBS.md` so the live manifest stays
loadable.

### `--minimum-cover`: smallest implementing set

```bash
python -m tools.kernels.discover models/dronet/dronet.mlir \
    --target spacemit_x60 --hw RVV --output kernels/ \
    --minimum-cover --iree-compile-arg='--iree-opt-data-tiling=false'
```

```
   #   cov%   cum_disp  shapes  signature
   1  35.7%     10/43        4  linalg.conv_2d_nchw_fchw
   2  53.7%     16/43        3  linalg.generic#mulf#parallel_parallel_parallel_parallel
   3  65.7%     23/43        5  linalg.generic#addf#parallel_parallel_parallel_parallel
   4  77.6%     29/43        4  linalg.generic#relu#parallel_parallel_parallel_parallel
   5  86.6%     32/43        3  linalg.generic#subf#parallel_parallel_parallel_parallel
   6  93.1%     34/43        2  linalg.fill
   7  99.6%     35/43        1  linalg.pooling_nchw_max
   8 100.0%     36/43        1  linalg.generic#relu#parallel_parallel
  ──→ 8 kernels = 100% coverage of dronet's compute
```

Greedy set-cover by cumulative compute. Each row is one author-unit
kernel covering all observed shape variants. **8 kernels for full
dronet coverage.**

### `--auto-fuse`: detect IREE-fused dispatches

```bash
python -m tools.kernels.discover models/dronet/dronet.mlir \
    --target spacemit_x60 --hw RVV --output kernels/ \
    --auto-fuse --iree-compile-arg='--iree-opt-data-tiling=false'
```

Output (excerpt):

```
Fused dispatches detected at flow phase (19 unique signatures):
  5-op fused  1x  elementwise  subf → mulf → addf → cmpf → select   ← BN+ReLU
  5-op fused  1x  elementwise  addf → negf → exp → addf → divf      ← sigmoid 1/(1+exp(-x))
  2-op fused  9x  conv         mulf → addf                          ← conv with bias-add
  2-op fused  3x  matmul_like  mulf → addf
  2-op fused  1x  elementwise  cmpf → select                        ← standalone ReLU
  1-op fused  1x  conv         maximumf                             ← max-pool
  ...
```

This is what IREE's natural fusion produced. Implementing one C kernel
per fused signature replaces N preprocessing-level kernel calls with
one — fewer dispatches, less DRAM bandwidth.

---

## Part 7 — Selecting which kernels run

### `select` — explicit opt-in per compile

The catalog can be larger than what you want enabled today. Add a
top-level `select` array:

```json
{
  "schema_version": 1,
  "select": ["saturnopu_matmul_f32", "saturnopu_pooling_nchw_max_f32"],
  "kernels": [ ... ]
}
```

Compile output:

```
🧬 Loading kernel manifest: benchmarks/SaturnOPU/kernels/manifest.json
🧬 select: 2 of 6 kernels enabled (4 kept in catalog but inert)
🧬 Precompiling 2 kernel(s) -> build/.../kernels_cache
```

The other 4 kernels stay registered in the manifest source but don't get
precompiled or wired into the spec. Reproducible compiles, easy A/B.

### `--kernels-strict-coverage` — fail-loud verification

```bash
./merlin compile <model> --kernels-dir <dir> --kernels-strict-coverage ...
```

After compile, walks phase 5 and audits remaining dispatches:

```
✅ kernels-strict-coverage: 0 unmatched dispatches (100% kernel coverage)
```

or, on partial coverage:

```
❌ --kernels-strict-coverage: dispatches survived past kernel rewrite
   (these went through IREE codegen, not your kernels):
        15x  linalg.generic
         3x  linalg.fill
   Inspect build/.../phases/<model>.5.dispatch-creation.mlir and add
   matching manifest entries (or run `python -m tools.kernels.discover`
   to auto-generate stubs).
```

---

## Part 8 — Quick-reference card

```
╔════════════════════════════════════════════════════════════════════════╗
║ MANIFEST FIELDS — what each one controls                               ║
╠════════════════════════════════════════════════════════════════════════╣
║ name              kernel id (used as flow.dispatch @kb_<name>)         ║
║ source            relative path to the .c (or .glsl/.cl/.spv)          ║
║ source_lang       c | glsl | cl | spirv                                ║
║ entry_symbol      C function name in the .o                            ║
║ signature.operands per-binding tensor types (with `?` for dynamic)     ║
║ signature.constants dim values transported as i32 push constants       ║
║   .name           short identifier (e.g. "M")                          ║
║   .from           {"input": N, "dim": K}  source SSA value             ║
║   .aliases        other (input, dim) pairs that equal this constant    ║
║ signature.output_dims ordered list of constant names = output dims     ║
║ match.kind        "linalg_dag" | "named_op" | "named_sequence"         ║
║ match.spec_path   path to body MLIR (linalg_dag / named_sequence)      ║
║ match.op_name     linalg op name (named_op)                            ║
║ match.outs_from_input N — input N becomes the matched op's outs        ║
║ match.op_attrs    verbatim attribute string for ops with required attrs║
║ targets           list of HAL keys (e.g. "llvm-cpu-spacemit-x60")      ║
║                                                                        ║
║ TOP-LEVEL: select  optional opt-in list of kernel names                ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ COMMANDS                                                                ║
╠════════════════════════════════════════════════════════════════════════╣
║ Discover ops in any model                                              ║
║   python -m tools.kernels.discover <model.mlir> \                      ║
║       --target T --hw HW --output kernels/                             ║
║                                                                        ║
║ Find smallest implementing set                                         ║
║   ... --minimum-cover                                                  ║
║                                                                        ║
║ Detect IREE-fused dispatches                                           ║
║   ... --auto-fuse                                                      ║
║                                                                        ║
║ Compile with kernels                                                   ║
║   ./merlin compile <model.mlir> --target T --hw HW \                   ║
║       --kernels-dir kernels/ \                                         ║
║       [--kernels-strict-coverage]                                       ║
║                                                                        ║
║ Build + run a kernel standalone on Spike                               ║
║   python -m tools.kernels.spike_runner \                               ║
║       --kernel kernel.c --driver driver.c --out /tmp/test.elf          ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ FILES TO READ                                                          ║
╠════════════════════════════════════════════════════════════════════════╣
║ Schema           kernels/core/manifest.py                             ║
║ Precompile       kernels/core/precompile.py                           ║
║ Spec emission    kernels/core/spec_gen.py                             ║
║ Discovery        kernels/core/discover.py                             ║
║ Compile wiring   tools/compile/cli.py                              ║
║ Snapshot tree    benchmarks/SaturnOPU/kernels/phase_dumps/             ║
║   add_f32/        — minimal 1D                                          ║
║   linear_f32/     — 2D matmul w/ push constants                         ║
║   dronet_partial/ — full model with 3 kernels firing                    ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Live verification

If you want to reproduce these exact snippets:

```bash
# Refresh the snapshot tree (regenerates everything under
# benchmarks/SaturnOPU/kernels/phase_dumps/):
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh

# Run all kernel tests (Spike + embed-pipeline integration):
conda run -n merlin-dev uv run pytest tests/granularity \
    -m "chipyard or integration" -v
# 5 PASSED in ~2s
```

The mechanism is reproducible end-to-end. Every snippet in this doc is a
real artifact under either `benchmarks/SaturnOPU/kernels/` or
`build/saturnopu_phase_dumps/`.
