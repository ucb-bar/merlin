# Inspect Kernel-Embedding MLIR Phases

This guide walks through the IREE compilation pipeline and shows what
custom-dispatch kernels look like at each stage — the analog of the IREE
canonical reference at
[`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/`](../../third_party/iree_bar/samples/custom_dispatch/cpu/embedded/),
which keeps `example_transform.mlir`, `example_stream.mlir`, and
`example_hal.mlir` as snapshot artifacts of the same flow at successive IR
levels.

The matching snapshots for the SaturnOPU kernel manifest live in:

```
benchmarks/SaturnOPU/kernels/phase_dumps/
├── add_f32/        — 1D elementwise add (no push constants)
└── linear_f32/     — 2D matmul (M, K, N as push constants)
```

Each subdirectory has six files capturing the embedding flow start-to-finish:

| File | What it is | Mirrors IREE sample |
|---|---|---|
| `0_input.mlir` | The user's MLIR — the linalg op the kernel is meant to replace. | `example_transform.mlir` (high-level user input) |
| `1_transform_spec.mlir` | The auto-generated transform-dialect spec from `kernels/core/spec_gen.py`. | `example_transform_spec.mlir` |
| `2_after_preprocessing.mlir` | Phase 3 — after `--iree-preprocessing-transform-spec-filename` has rewritten the linalg op into a `flow.dispatch`. | (between transform and stream) |
| `3_flow.mlir` | Phase 6 — flow-dialect form with the dispatch wrapper as a `util.func` and the executable as a `hal.executable.source`. | (intermediate) |
| `4_stream.mlir` | Phase 7 — stream-dialect form. | `example_stream.mlir` |
| `5_hal.mlir` | Phase 11 — hal-dialect form with the linked `.o` materialized into a `hal.executable.variant`. | `example_hal.mlir` |

To regenerate the snapshots after changing kernel source, manifest, or
match.mlir:

```bash
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh         # both
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh add     # only add
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh linear  # only linear
```

The script is a thin wrapper around
`./merlin compile --kernels-dir benchmarks/SaturnOPU/kernels --dump-phases`
followed by selective copies into the snapshot tree.

## Reading a snapshot pair

Take `linear_f32/` (the matmul case — more interesting because of push
constants).

### 0_input.mlir

The synthetic input is a single `linalg.generic` matmul-with-transposed-B
expressed via explicit indexing maps:

```mlir
func.func @main(%lhs: tensor<8x64xf32>, %rhs: tensor<16x64xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x16xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : ...) -> tensor<8x16xf32>
  %mm = linalg.generic
      {indexing_maps = [(d0,d1,d2)->(d0,d2), (d0,d1,d2)->(d1,d2), (d0,d1,d2)->(d0,d1)],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : ...) outs(%filled : ...) { ... } -> tensor<8x16xf32>
  return %mm
}
```

This is **before** the kernel-embed pipeline runs. There is no mention of
the kernel name or any `hal.executable` ops — just a normal linalg op.

### 1_transform_spec.mlir

The auto-generated spec wires three things together:

1. `hal.executable.source private @kb_saturnopu_linear_f32` carries the
   precompiled `.o` reference and the inner `builtin.module` shim that
   declares the C symbol with `hal.import.static` and emits
   `hal.interface.constant.load` for each push constant (M, K, N) plus a
   `hal.interface.binding.subspan` per binding.
2. `util.func @call_saturnopu_linear_f32` — the dispatch wrapper that
   `cast_and_call` will substitute in for the matched op. It does
   `tensor.dim` to materialize M/K/N, casts to i32, and dispatches into
   the executable.
3. `transform.named_sequence @match_saturnopu_linear_f32` + `@cast_and_call_*`
   — the matcher and rewriter that find the linalg.generic and replace it.

### 2_after_preprocessing.mlir

By phase 3 the matched linalg.generic is gone. In its place:

```mlir
%dim = tensor.dim ...
%M_i32 = arith.index_cast ...
%K_i32 = ...
%N_i32 = ...
%workload = arith.muli %M, %N : index
%out = flow.dispatch @kb_saturnopu_linear_f32::@linear_f32[%workload]
       (%M_i32, %K_i32, %N_i32, %lhs_cast, %rhs_cast)
       : (i32, i32, i32, tensor<?x?xf32>{...}, tensor<?x?xf32>{...})
       -> tensor<?x?xf32>{...}
```

Push constants (M, K, N) are the **first three** dispatch args (before the
tensor bindings), matching IREE's pipeline-layout convention.

### 3_flow.mlir

The dispatch is now outlined into a `flow.executable` with our
`hal.executable.source` carrying the linked `.o`:

```mlir
hal.executable.source private @kb_saturnopu_linear_f32 attributes {
    objects = #hal.executable.objects<{
      #executable_target_embedded_elf_riscv_64 = [
        #hal.executable.object<{path = "saturnopu_linear_f32.<sha>.riscv64-none-elf.o"}>
      ]
    }>
} {
  hal.executable.export public @linear_f32 ordinal(0) layout(...)
      count(...) -> (index, index, index) { ... }
  builtin.module {
    func.func private @linear_f32_workgroup(...) attributes {hal.import.static}
    func.func @linear_f32() {
      %M_i32 = hal.interface.constant.load layout(...) ordinal(0) : i32
      %M = arith.index_cast %M_i32 : i32 to index
      ...
      func.call @linear_f32_workgroup(%binding0, %binding1, %binding2,
                                      %M, %K, %N, %tid) : (...) -> ()
      return
    }
  }
}
```

This is the same shape the IREE canonical sample's
[example_stream.mlir](../../third_party/iree_bar/samples/custom_dispatch/cpu/embedded/example_stream.mlir)
hand-authors, just synthesized automatically from the manifest.

### 4_stream.mlir

After the flow → stream conversion the executable becomes a
`stream.executable` with a `stream.async.dispatch` call site. The push
constants now appear as `stream.cmd.dispatch` operands.

### 5_hal.mlir

The HAL phase materializes the `.o` into a concrete `hal.executable.binary`
that gets embedded in the vmfb. The executable structure mirrors the IREE
sample's
[example_hal.mlir](../../third_party/iree_bar/samples/custom_dispatch/cpu/embedded/example_hal.mlir).

## Cross-checks

Quick greps you can run on the snapshot tree:

```bash
# Confirm the linalg op was rewritten into a flow.dispatch:
grep "flow.dispatch @kb_" benchmarks/SaturnOPU/kernels/phase_dumps/linear_f32/2_after_preprocessing.mlir

# Confirm the precompiled .o is referenced:
grep "hal.executable.object" benchmarks/SaturnOPU/kernels/phase_dumps/linear_f32/3_flow.mlir

# Confirm push constants are loaded inside the wrapper:
grep "hal.interface.constant.load" benchmarks/SaturnOPU/kernels/phase_dumps/linear_f32/3_flow.mlir
```

For the simpler `add_f32` case the same files are present without the
push-constant loads — useful as the minimal baseline before adding shape
plumbing.

## Where to look in the source

- `kernels/core/manifest.py` — schema (kernels, signatures, push constants).
- `kernels/core/precompile.py` — clang invocation per target.
- `kernels/core/spec_gen.py` — emits the transform spec.
- `tools/compile/cli.py` — wires `--kernels-dir` into iree-compile.
- `benchmarks/SaturnOPU/kernels/` — the canonical SaturnOPU manifest with
  worked add_f32 and linear_f32 examples.
- `tests/granularity/test_kernel_embed_pipeline.py` — pytest checks that
  exercise the same flow on every change.
