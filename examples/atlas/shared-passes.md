# Atlas shared-pass preflight

The `feat/atlas-target` branches provide a working diagnostic path for an initial
bounded compiler-authoring experiment. Merlin is based on main
`d56133b168bbee3d73cb8b4304d3622d4191fc6a`; it does not import the large
`atlas-model-compilation` compiler stack. No authoring or optimization agent is
launched by the commands below.

## What is reusable

- `merlin.frontends.linalg_float_graph` reads complete static source graphs into
  the existing `KernelRequest`. It preserves exact SSA edges, all ordered outputs,
  numerical policies and unresolved domain premises. Ordered contractions share
  their recognizer with the original single-matmul path. BF16 ReLU, explicit
  FP32-to-BF16 chop add/subtract, BF16 product rounding, and declared FP8 casts reuse
  the source recognizers from `atlas-model-compilation`.
- Existing checked dense planning and counted loops retain reduction order and
  accumulator state. FP16/f16 storage now has its correct two-byte width.
- `schedule.time_annotation` and `schedule.overlap` are consumed in actual Atlas
  compilations. The shared pair checker uses instruction-bound constraints
  exported by the existing OOT timing provider. This checks issue spacing and
  fixed-latency overlap; OOT still checks physical reservations, DMA and CFG.
- `merlin.runtime.packed_float` supplies explicit grouped storage for FP4 E2M1 and
  FP6 E2M3/E3M2 codes, including byte/lane ordering, partial groups, padding and
  separate E8M0 scale bytes. It does not select quantization scales or claim an
  MX-Gemmini instruction lowering. Atlas hardware remains E4M3/BF16 in this path.

Fusion remains a target decision over typed graph edges and retained outputs.
The initial adapter keeps producer tiles in Atlas virtual SSA and delegates
allocation and scheduling to the existing OOT passes. It does not assume that
two operations with matching shapes can be numerically fused.

## Isolated compiler library

[shared-library.json](shared-library.json) explicitly selects 40 source/namespace
members and 15 public leaf modules. Pure parsing and instruction contract types
are separated from capture, backend discovery and reference evaluation. The
export uses the existing `CompilerLibraryContract` hash, namespace and import
closure checks. It includes no Atlas compiler, reference output, model or grader.
Namespace-only initializers expose no parent package APIs; the normal installed
Merlin namespaces keep their existing compatibility imports.

From the Merlin worktree, with its Python environment:

```sh
python build_tools/scripts/export_compiler_library.py \
  examples/atlas/shared-library.json \
  --destination out/build/atlas-library
```

Use a fresh destination. The result contains `compiler-library.json` and the
standalone `merlin/` source tree. The trusted Phase 1 setup selects that contract
and mounts the matching tree as its `compiler` view. A candidate cannot admit
itself by copying the JSON or adding imports to its own manifest. Run the
existing runtime/tool/corpus admission checks when constructing that experiment.

## Reproduce the execution preflight

In the companion Atlas worktree, rebuild `atlas-opt` and `atlas-emit` from this
branch: the old binaries lack `--issue-timing-json`. Configure CMake with the
selected LLVM/MLIR installation, then build both targets. Select:

| Input | Exact revision |
| --- | --- |
| `npu_model` | `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` |
| RTL identity | `0079c0541111197741a231c002e3843fa6f545b2` |
| Atlas MLIR base | `c9f50c3310a68d9eb8fa318e68492896a5a45737` |

The newer model PACK layout is incompatible with this compiler baseline.
Use the Merlin Python environment for the driver, and the separate model Python
environment for execution. Set `MERLIN_LIBRARY`, `ATLAS_BIN`, `MODEL_ROOT`, and
`MODEL_PYTHON` to the exported library, rebuilt tools, pinned model and its Python.

```sh
PYTHONPATH="$MERLIN_LIBRARY:$PWD/python:$PWD" python tools/check_shared_graph.py \
  --output-dir out/build/shared-graph-preflight \
  --atlas-bin-dir "$ATLAS_BIN" --model-root "$MODEL_ROOT" \
  --model-python "$MODEL_PYTHON"
```

This compiles standalone ReLU, matmul→ReLU, and matmul→ReLU→PACK→matmul at
32×32 and 64×64, both source-order and optimized, then executes two witnesses
per program. The 24 executions compare every output byte, preserve inputs and
guards, independently decode/re-encode final instructions, and require halt
without model runtime errors. PACK witnesses separately establish the declared
intermediate domain. They are never provided to the compiler as expected results.

`tools/check_shared_dense.py` runs another 78 executions covering continued K,
multiple output tiles, numeric rounding/range witnesses and direct Atlas
comparators. It requires the full Merlin source tree for its test-fixture builder.
Each compilation records source, checked schedules, virtual/machine IR, final
words, tool hashes, issue constraints and consumed schedule annotations.

## Measured scope and remaining launch requirements

Observed model results on November 10, 2026:

| Shared graph | Source-order words / cycles | Optimized words / cycles |
| --- | --- | --- |
| 32×32 ReLU, retaining input | 50 / 4,716 | 49 / 4,564 |
| 32×32 matmul + ReLU | 39 / 3,337 | 44 / 3,249 |
| 32×32 two-layer chain, retaining hidden output | 111 / 6,792 | 120 / 5,090 |
| 64×64 two-layer chain, retaining hidden output | 456 / 27,131 | 491 / 20,186 |

The 32×32 matmul/ReLU/chain results match the direct Atlas comparator instruction
and cycle counts. These are shared lowering plus the hand-written Atlas physical
compiler, not measurements of a generated backend. A generated backend must be
authored independently and evaluated before making that comparison.

Initial authoring scope is full 32×32 tiles, at most 64 dense tile products, ordered
E4M3/BF16 matmul and the qualified ReLU/PACK forms above. Tails, arbitrary BF16
truncation, reductions/softmax and hardware FP4/FP6 lowering remain outside this
scope and refuse. Source domain declarations remain premises, especially the
seven-code PACK domain; the diagnostic adapter does not establish them for
arbitrary runtime graphs. The old whole-model native binding is a separate API
and still requires the `atlas-model-compilation` stack.

The public `atlas-functional` experiment remains a diagnostic template. Its
reviewed corpus, independent hardware/software/runtime intake, tool sandbox and
timing evidence must be selected through the existing Phase 0/1 gates before a
verified generation run. This preflight supplies working shared compiler APIs
and execution evidence; it does not manufacture those authorities or qualify RTL.
