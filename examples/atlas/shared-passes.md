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
- `merlin.frontends.linalg_tensor_graph` reads ordinary f16/bf16/f32/f64 graphs:
  matmul, transpose, literal splats, one-axis maximum/sum reductions, trailing
  singleton expansion, pointwise add/subtract/multiply/divide/exp and ReLU.
  It checks scalar bodies, actual SSA operands, static shapes and broadcast maps.
  It preserves source precision and unspecified reduction order; target precision
  changes are separate, explicit OOT choices. Tests include non-tile dimensions,
  rectangular matrices, both reduction axes and rank-three broadcasts/reductions.
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

[shared-library.json](shared-library.json) explicitly selects 42 source/namespace
members and 16 public leaf modules. Pure parsing and instruction contract types
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
comparators. Its test-fixture builder uses the isolated library's pure parser.
Each compilation records source, checked schedules, virtual/machine IR, final
words, tool hashes, issue constraints and consumed schedule annotations.

## Attention and MLP with explicit precision selection

The companion `atlas_mlir_native.shared_tensor_graph.compile_tensor_graph` API
uses the shared source reader and existing checked dense scheduling. It requires
`precision="atlas-f32-to-bf16-e4m3-diagnostic-v1"`. Atlas owns the E4M3 inputs,
BF16 accumulation, VPU rounding, exponential/reciprocal approximations, PACK,
32-lane reduction geometry and physical instructions. The manifest records each
source operation's replacement and unresolved numeric/quality obligations.
This is a diagnostic precision change, not an equivalence proof for f32.

```sh
PYTHONPATH="$MERLIN_LIBRARY:$PWD/python:$PWD" python tools/check_shared_attention.py \
  --output-dir out/build/shared-attention-preflight \
  --atlas-bin-dir "$ATLAS_BIN" --model-root "$MODEL_ROOT" \
  --model-python "$MODEL_PYTHON"
```

The driver starts from the captured linalg attention+MLP graph: Q/K/V projection,
QK transpose product, scaled stable softmax, attention/value product, and two
linear layers separated by ReLU. It also structurally varies projection input
width from 32 to 64 and retains all 12 matrix/pointwise intermediate outputs in
separate diagnostic programs. Two independent input sets in both scheduling
modes make 16 model executions. An independent high-level numerical reference
checks exact target-arithmetic output bytes; it does not interpret the compiler's
emitted IR. A separate f32 calculation reports max absolute error and RMSE without
claiming an application quality threshold. Test inputs are BF16-representable;
boundary tests separately cover FP8 rounding ties, flush and saturation.

The reductions cover complete 32-element rows. Wider softmax rows, partial tiles,
arbitrary intermediate domains and BF16 physical transpose remain unsupported.
The reader itself has no hardware tile restriction. The old exact-policy graph
path and its seven-code PACK premise remain separate from this diagnostic path.

Repeat the attention command with `--dense-witnesses` and a fresh output directory
to exercise dense signed weights, cancellation and accumulated rounding. This
adds another 16 executions with the same source graphs and independent reference.
It changes test inputs only, never compilation or numerical policy selection.

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
exact-policy scope and refuse. The explicit diagnostic path above additionally
covers attention/softmax. Source domain declarations remain premises, especially the
seven-code PACK domain; the diagnostic adapter does not establish them for
arbitrary runtime graphs. The old whole-model native binding is a separate API
and still requires the `atlas-model-compilation` stack.

The public `atlas-functional` experiment remains a diagnostic template. Its
reviewed corpus, independent hardware/software/runtime intake, tool sandbox and
timing evidence must be selected through the existing Phase 0/1 gates before a
verified generation run. This preflight supplies working shared compiler APIs
and execution evidence; it does not manufacture those authorities or qualify RTL.

## Full-model readiness review (November 16, 2026)

**The shared library is a working bounded compiler component, not yet a complete
model-lowering path or a launch-ready Atlas generation experiment.** The limits
below are integration work, not reasons to add target branches to shared passes.

Two source-validation defects were reproduced and fixed. Static
`tensor.expand_shape` now checks its explicit output shape against the result
type. Ordered floating matmul now checks the fill region's scalar argument types
against the initializer and destination. The selected xDSL version accepted both
malformed forms; relying on `module.verify()` alone was insufficient. Regression
tests cover mismatched expansion dimensions and both fill arguments through the
single-matmul and graph readers.

The review also checked actual compiler inputs without reading their runtime
inputs or expected outputs. The available TinyLlama batch-two and vision captures
parse and verify but are refused by this reader: they contain mixed BF16/integer
boundaries and target-prepared `atlas.qlinear`, `atlas.quantize_act`,
`atlas.qk_matmul` and `atlas.pv_matmul` operations. They additionally need batched
geometry, casts, general reshape, normalization and, for the decoder capture,
slice/concat operations. These are capture-stage examples, not evidence of a
complete decoder or policy application. The older generic quantized coverage-MLP
artifact fails xDSL verification on `arith.trunci`'s `overflowFlags` property;
that is an intake/version mismatch before any shared lowering occurs.

| Blocking area | Current evidence | Smallest reusable next step |
| --- | --- | --- |
| Whole-model API compatibility | `atlas_mlir_native.binding` fails to import `StagedCompileInputs`; it also depends on precision modules and graph APIs absent from this main-derived branch. | Agree on one staged-input/graph contract, then port the required generic staging, constant binding and precision components with their consumer tests. Do not shim the import while omitting its invariants. |
| Graph and operator coverage | The new reader covers a closed floating subset; it does not cover quantized captures, batched matmul, arbitrary views, casts, rsqrt/normalization, SiLU/GELU, masking or KV-cache updates. | Reuse existing frontend recognizers and host lowering. Preserve explicit index maps, types, source bodies and retained outputs in a shared graph. Put prepared Atlas-operation intake in the OOT adapter. |
| Model-scale execution | This adapter expands at most 64 dense tile products, uses full 32-element tiles and reduces only complete 32-element rows. A 256×256×256 matmul already needs 512 tile products. Input staging is limited to 256 physical slots. | Keep shared counted-loop planning and verification; teach the OOT emitter to consume loops or bounded partitions. Connect shared buffer ownership, weight staging, dispatch and host fallback with checked target layouts and transfer lifetimes. Raising the expansion limit alone is insufficient. |
| Numerical validity | Source f32 is explicitly approximated by E4M3/BF16, PACK and VPU approximations; runtime intermediate domains and application quality remain unproved. | Select and validate a model precision policy, including scales, exceptional values, reduction order and quality thresholds. Keep exact-source and approximate-policy APIs separate. |
| Generation setup | `atlas-functional` is still diagnostic. Its selected oracle timing file is absent, the descriptor's host commit is `UNKNOWN`, and the old bundle does not select the new compiler-library contract. | Use the existing intake/admission workflow to freeze compatible tools, reviewed corpus, host runtime, timing evidence and the exported library before launching. A library export is not an experiment admission receipt. |

The smallest useful next milestone is one statically shaped transformer block
through the agreed staged-input interface, with real weight binding and explicit
host/device coverage for every operation. Test source order and optimized order,
all intermediate tensors, at least one sequence length above 32, a tail, and a
contraction above 64 tile products. Only then scale to full prefill, decode and
application loops with their state and complete output ABI. These cases should
derive from independent synthetic graphs; held-out model outputs must not guide
lowering or optimization choices.

The fresh sparse and dense attention witnesses both match the declared target
arithmetic byte-for-byte, in source order and optimized order, including all 12
retained stages. Dense-weight f32 comparisons are materially less accurate than
the sparse witnesses: maximum absolute error ranges from **0.2765 to 0.6897**,
and RMSE from **0.0683 to 0.1919** (f32 output maxima 1.78–7.69). These observations
do not establish an acceptable model-quality threshold.

Scheduling checks retain their limited meanings. The shared dense verifier
proves actual projected loop coverage, order and residency under the adapter's
synchronous contract. The Atlas timing export is trusted provider evidence bound
to emitted words; checking it does not independently derive hardware latencies.
DMA, reservations and control flow are still checked by the OOT compiler.
Reported cycles come from the pinned `npu_model` execution, not from shared
planning costs or an RTL measurement. None of these runs generates a new OOT
backend, compares one against Atlas MLIR, or establishes full-model execution.

FP4/FP6 remain reusable codecs and explicit packed storage. There is no
MX-Gemmini instruction lowering, model precision selection, or target admission
for those formats in this branch. Fusion likewise retains producer/consumer SSA
edges and target allocation, but has no general whole-model fusion planner yet.

## Complete layer integration on the generalized branch

The separate Atlas branch `feat/generalized-tinyllama-layer` now connects this
branch to complete-layer host/device lowering. Its walkthrough is
[docs/generalized-tinyllama-layer.md](https://github.com/ucb-bar/atlas-mlir/blob/feat/generalized-tinyllama-layer/docs/generalized-tinyllama-layer.md).
The older bounded adapter above retains its original limits; this is a new,
explicit diagnostic model entry point, not a silent change to that adapter.

Merlin now parses and checks named `linalg.batch_matmul` with the same closed
scalar recurrence, positive-zero initializer, dtype and affine-map checks used
for ordinary matmul. `llvmlower.floating_matmul_outline` outlines checked f32
contractions to synchronous ranked-memref callbacks under an explicit caller
precision policy. It preserves the rest of the original graph and all returned
tensors, using the existing host lowering, ownership and MLIR C runtime.
Device conversion, layouts, addresses and instructions remain in Atlas.

The layer fixture contains RMSNorm, Q/K/V projections, RoPE, grouped-query
attention, a causal mask, softmax, the output projection, both residuals, and a
gated SiLU MLP. All nine contractions use Atlas. Other operations use compiled
host f32 code. Full geometry is 2048 hidden, 5632 intermediate, 32 query heads,
four KV heads and sequence length 32, with synthetic weights. A reduced fixture
and a 120/270-width, seven-token fixture exercise independent M/N/K tails.
This does not establish complete-model, decode, historical KV-cache, all-device
execution or pretrained quality.

The OOT consumes the actual shared dense schedule with a compact counted K loop
and bounded DMA. Exact BF16 readout/restore carries the reduction state; no host
partial-sum reassociation is used. Programs handle complete padded K up to 8192,
while the test transport dispatches batches and M/N output tiles. This reuses
Atlas's existing bounded-address and code-size support instead of increasing
unrolled program limits. The full-width K=5632 kernel is 98 words in source order
and 96 words optimized.

The selected policy is explicitly approximate: finite f32 inputs are rounded to
E4M3FN, saturated, subnormals flushed to positive zero, and accumulated in K order
with BF16 rounding. Tests separately require bit-exact device outputs on actual
operands and f32 host-partition agreement with Torch at common device boundaries.
A free-running independent Torch reference is retained as a separate observation.
It exposes a full-width RMSNorm difference of about 2.4e-7 at one element straddling
an FP8 midpoint; after the MLP the final output differs by about 54. This is not
end-to-end numerical equivalence, and increasing comparison tolerance is not the
correctness gate. Host partition tolerance is rtol/atol 1e-5; the recorded final
host-replay output differences are zero. Model quality and a stronger whole-graph
precision contract remain unresolved.

The compiler-library selection adds the batch parser dependency (42 members).
The new whole-host outlining/build helper is used from the installed Merlin
checkout; it has not been added to the closed OOT-authoring library selection or
independently admitted as a generation input. No OOT generation loop was run.
