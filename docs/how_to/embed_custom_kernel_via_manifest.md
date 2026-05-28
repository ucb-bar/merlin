# Embed A Custom Kernel Via Manifest

`./merlin compile` ships an end-to-end pipeline for plugging a hand-written
(or LLM-generated) kernel into a compiled model without touching the compiler:

1. Write the kernel source (`.c` / `.glsl` / `.cl` / `.spv`).
2. Write a linalg-DAG match pattern that the compiler will use to find every
   instance of the op in your model.
3. Wire them together in a `manifest.json`.
4. Compile with `--kernels-dir <dir>` (or set `custom_kernels.manifest` in
   the target YAML).

The compiler precompiles each kernel to its target object, auto-generates a
transform-dialect spec that rewrites matched linalg ops into `flow.dispatch`
calls into the precompiled object, and threads the right
`--iree-preprocessing-transform-spec-filename=` and
`--iree-hal-executable-object-search-path=` into `iree-compile`.

The implementation lives in `kernels/core/{manifest.py,precompile.py,spec_gen.py}`
and is invoked from `tools/compile/cli.py`.

## 1. Recipe

A kernel directory is three files (plus the optional `drivers/` folder for
standalone Spike testing):

```
my_kernels/
├── manifest.json          # ← step 3 below: wires together (1) + (2)
├── src/
│   └── my_kernel.c        # ← step 1 below: the kernel source
└── match/
    └── my_kernel.match.mlir   # ← step 2 below: linalg-DAG match pattern
```

The three subsections below correspond to steps 1, 2, 3 from the intro
(write source → write match → wire manifest). Step 4 (compile) is
covered in [§ 2](#2-compile).

### Step 1 — `src/my_kernel.c`

The C entry symbol must match `entry_symbol` in the manifest and follow the
IREE CPU custom-dispatch ABI for embedded ELFs (one workgroup per element by
default — see the spec_gen comment about the default `count(%workload) ->
(%workload, 1, 1)` region):

```c
#include <stddef.h>

void my_kernel_workgroup(const float* restrict binding0, size_t binding0_offset,
                         const float* restrict binding1, size_t binding1_offset,
                         float* restrict binding2, size_t binding2_offset,
                         size_t dim, size_t tid) {
  if (tid >= dim) return;
  binding2[binding2_offset + tid] =
      binding0[binding0_offset + tid] + binding1[binding1_offset + tid];
}
```

The reference is
`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/functions.c`.
Each binding lowers to `(ptr, offset)` (with `llvm.bareptr=true`); after all
bindings come `(dim, tid)`. Returns `void`; output is written through the last
binding pointer.

### Step 2 — `match/my_kernel.match.mlir`

Just the body of `transform.iree.match.cast_compatible_dag_from_root` —
spec_gen wraps it in the canonical match scaffold. Use dynamic shapes
(`tensor<?xf32>`) so the matcher generalizes; `cast_and_call` inserts
`tensor.cast` ops on the boundary as needed.

```mlir
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

The reference is
`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir`,
specifically the `@match_mul_abs_negate` named sequence.

### Step 3 — `manifest.json`

```json
{
  "schema_version": 1,
  "kernels": [
    {
      "name": "my_kernel",
      "source": "src/my_kernel.c",
      "source_lang": "c",
      "entry_symbol": "my_kernel_workgroup",
      "signature": {
        "operands": [
          {"role": "in",  "tensor": "tensor<?xf32>"},
          {"role": "in",  "tensor": "tensor<?xf32>"},
          {"role": "out", "tensor": "tensor<?xf32>"}
        ]
      },
      "match": {
        "kind": "linalg_dag",
        "spec_path": "match/my_kernel.match.mlir"
      },
      "targets": ["llvm-cpu-spacemit-x60"]
    }
  ]
}
```

Schema: `kernels/core/manifest.py:9-37`. `targets` keys are matched against
`kernels/core/precompile.py:_CPU_TARGET_FLAGS` (currently
`llvm-cpu-x86_64`, `llvm-cpu-aarch64`, `llvm-cpu-riscv64`,
`llvm-cpu-riscv64-rvv`, `llvm-cpu-spacemit-x60`) and against
`kernels/core/spec_gen.py:_HAL_TARGET_ATTR`.

## 2. Compile

```bash
./merlin compile path/to/input.mlir \
  --target spacemit_x60 --hw RVV \
  --kernels-dir my_kernels/ \
  --dump-phases --dump-artifacts \
  --output-dir build/my_kernel_test/
```

After the run the auto-generated artifacts land under
`build/my_kernel_test/kernels_cache/`:

| File | What it is |
|---|---|
| `<kernel>.<sha>.<arch>.o` | Precompiled object for the target (clang for C, clspv/glslang for GPU). |
| `transform_spec.mlir` | Auto-generated transform-dialect spec wiring matched linalg ops to `flow.dispatch` calls into the kernel. Inspectable with `iree-opt`. |

## 3. Confirming the rewrite landed

Inspect the dumped phase MLIR to confirm the linalg op was rewritten:

```bash
# Phase 6 — after dispatch outlining; should contain the flow.dispatch call.
grep "flow.dispatch @kb_<kernel-name>" build/my_kernel_test/phases/*.6.flow.mlir

# Phase 10 — after HAL materialization; should contain the linked .o path.
grep "hal.executable.object" build/my_kernel_test/phases/*.10.executable-targets.mlir
```

If the rewrite did not land, the most common causes are:

- The match.mlir body's tensor types don't match the payload (the matcher
  uses `cast_compatible_dag_from_root`, but the *op chain* must match
  exactly — same iterator_types, same indexing_maps, same body ops).
- The manifest's `targets` list doesn't include the HAL target the input is
  being compiled for (the spec only emits objects for declared targets).

## 4. Inspecting the auto-generated spec

`build/<out>/kernels_cache/transform_spec.mlir` is plain text and follows a
fixed structure (see `kernels/core/spec_gen.py:emit`):

1. `#kb_target_<key> = #hal.executable.target<...>` aliases.
2. `hal.executable.source private @kb_<name>` carrying `hal.executable.objects`.
3. `util.func private @call_<name>(...)` — the dispatch wrapper.
4. `transform.named_sequence @match_<name>` — wraps your match.mlir body.
5. `transform.named_sequence @cast_and_call_<name>` — invokes `cast_and_call`.
6. `transform.named_sequence @__transform_main` — the foreach driver.

This file is the canonical artifact to inspect when debugging match failures
or wanting to hand-edit the rewrite.

## 5. The auto-generated executable shim

For C-source kernels (`source_lang: "c"`), the generated `hal.executable.source`
includes a `builtin.module` body with the binding-subspan + call shim that
final IREE codegen needs (mirrors the IREE canonical sample at
`third_party/iree_bar/samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir`):

```mlir
hal.executable.source private @kb_<name> attributes { objects = #hal.executable.objects<{...}> } {
  hal.executable.export public @<export_name> ordinal(0)
      layout(#pipeline_layout_<name>)
      count(%device, %workload) -> (index, index, index) { ... }
  builtin.module {
    func.func private @<entry_symbol>(<bindings>, index, index)
        attributes {hal.import.static}
    func.func @<export_name>() {
      %dim = hal.interface.workgroup.count[0] : index
      %tid = hal.interface.workgroup.id[0] : index
      // hal.interface.binding.subspan ... per binding
      func.call @<entry_symbol>(<bindings>, %dim, %tid) : (...) -> ()
      return
    }
  }
}
```

`<export_name>` is derived from `entry_symbol` by stripping a trailing
`_workgroup` suffix when present (so a manifest with
`entry_symbol: add_8xf32_workgroup` produces export `@add_8xf32`), otherwise
appending `_dispatch`. The `flow.dispatch` call wrapper (`@call_<name>`)
references the export name, never the C symbol.

Limits of the current emit:

- **One workgroup per element** (workgroup count derived from workload), good
  for the 1D-dynamic case but suboptimal for large tensors where you'd want a
  bigger workgroup. Override by hand-editing the generated `transform_spec.mlir`
  and re-running with `--iree-compile-arg='--iree-preprocessing-transform-spec-filename=<edited>'`
  + `--no-kernel-embedding`.
- **No push constants.** Kernels that need scalar args (e.g. quantization
  scales) currently can't get them through the auto-spec — extend
  `_pipeline_layout_for` to declare `constants = N`, update `_call_wrapper` to
  pass them as the leading `flow.dispatch` operands, and update
  `_inner_module_for_c` to emit `hal.interface.constant.load` calls.
- **Multi-D memrefs.** The wrapper assumes 1D-dynamic memref binding shape.
  Multi-D will need explicit dim plumbing (push constants or workload tuple).

The fixture exercising this pipeline end-to-end lives at
`tests/granularity/fixtures/embed_pipeline/`. After `./merlin compile … --kernels-dir`
on that fixture, the resulting vmfb is a valid IREE module
(verifiable with `iree-dump-module`).

## 6. Where to look in the source

- `tools/compile/cli.py` — wires `--kernels-dir` / `--kernel-manifest` /
  YAML `custom_kernels` into the IREE flags.
- `kernels/core/manifest.py` — JSON schema + loader.
- `kernels/core/precompile.py` — clang/clspv/glslang invocation, sha-keyed
  artifact cache, per-target arch flags.
- `kernels/core/spec_gen.py` — emits `transform_spec.mlir` from the manifest.
- `third_party/iree_bar/samples/custom_dispatch/cpu/embedded/` — canonical
  hand-written reference for both the C kernel ABI and the transform spec
  shape.
