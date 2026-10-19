---
title: Prepacking immutable weight layouts
kind: guide
status: current
owner: ir
last_verified: 2026-10-18
related: [model_lowering, shared_phase1_scheduling]
code_refs:
  - src/merlin/compile/baremetal_model.py
  - src/merlin/llvmlower/weight_prepack.py
  - src/merlin/llvmlower/weight_prepack_runtime.py
  - src/merlin/capture/rewrite.py
  - src/merlin/xdsl_dialects/lowering/weight_layout.py
---

# Prepacking immutable weight layouts

Weight prepacking moves an invariant layout transformation from inference to
compilation. For example, a stored tensor may pass through a rank-four transpose,
collapse to a vector, expand to a matrix and transpose again before computation.
Merlin can store the final matrix directly and erase the checked chain. This
preserves contiguous consumer storage rather than replacing it with arbitrary
strided accesses.

## Select the ordinary compiler route

Use the saved-model API with the same capture, selected target, numerical policy
and execution/readback settings as the baseline:

```python
from merlin.compile.baremetal_model import compile_saved_model

baseline = compile_saved_model(**options, output=baseline_directory)
optimized = compile_saved_model(
    **options, output=optimized_directory, prepack_weights=True,
)
```

`options` contains the existing required `capture`, `target`, `run` and
`arena_mb` arguments. Use `readback="full"` and the capture's unchanged reference
and tolerance when comparing execution. Selecting this option does not change
numerical admission thresholds. The default is `False`, with no preparation
parsing, copying or receipt. The registered compiler feature is
`prepack_weight_layout`; both the ordinary model lowerer and generated runtime
consume the same prepared IR, manifest and tensor payloads.

An explicitly selected optimization refuses if nothing can be prepacked or its
premises cannot be established. It never reports success while silently using
the baseline. The original capture remains available for a separate baseline
build. Unselected and unsupported chains within an otherwise admitted bundle
remain in the graph.

## Required semantics and ownership

Geometry comes from verified typed tensor IR, not model names or captured IDs.
Supported steps are static tensor transposes and contiguous collapse/expand
reshapes. Each intermediate must have the required single input use in the same
entry block; unknown operations, dynamic geometry, executable overrides and
ambiguous function calls stop or refuse the transformation. A live transpose
initialization value is preserved.

The ordinary generated-caller route establishes its ownership premise from the
original source, manifest and optional version-1 session contract. It requires
a complete pure tensor/scalar call closure, including checked serial control
flow. Buffer/address values, external calls and unknown effects refuse this
selection. Concrete stored parameters and buffers may qualify; a manifest's
`buffer` label alone supplies no permission. Updated state/stream slots,
mutable/stubbed weights, duplicate stored-name readers and overlapping physical
payloads cannot be selected.

The qualified caller here is Merlin's existing generated inference runtime:
stored arguments use its weight slots, while session commits update input slots.
This premise does not cover arbitrary external clients of the changed entry ABI.
The lower-level `prepacked_bundle(..., immutable_arguments=None)` retains its
legacy parameter-role premise; direct users must establish immutable invocation
ownership independently. An explicit immutable tuple is likewise a caller
premise, not a capability inferred from a flag.

No hardware fact is needed to permute raw stored element bytes. Element widths
come from the checked IR and safetensors records. This mechanism does not infer
accelerator packing formats, DMA alignment, address windows, completion rules
or target instruction support from those widths. Such physical transformations
require separate target capabilities and realized-program checks.

## Current-source and realized-output checks

Merlin content-pins the original flat bundle files, selected entry, immutable
argument tuple and relevant implementation sources. Cache keys use streamed
content hashes, so same-size, same-timestamp replacements invalidate them.
Original absence of a session contract is also checked. Existing stale
`model.prepared*` artifacts are omitted from the prepared bundle.

The prepared bundle is published from a private staging directory by atomic
rename. The cache is checked independently of its own reported output hashes:

1. Actual typed IR must equal the current source's checked layout rewrite.
2. Actual manifest and safetensors geometry, offsets and complete byte extent
   must agree with that rewrite.
3. Inverting the actual prepared layouts must recover every original selected
   element byte; untouched tensors must retain their original bytes.
4. Carried sidecar membership and bytes must match the original bundle.
5. The complete rewrite history and reported effects must match reconstruction.

This is a byte-preserving transformation, including BF16, NaN payloads and signed
zero; it performs no arithmetic or precision conversion. `bundle_preparation.json`
binds the preparation consumed by lowering and runtime generation. The ordinary
bare-metal `CompilationRecipe` reopens the source, implementation, prepared files
and ownership premises before completion and on completed-recipe verification.
A changed or unverifiable cache refuses; it is not deleted or self-certified.
These checks assume exclusive ownership during a build rather than protection
against concurrent writers.

## Costs, validation and limits

Prepacking adds compilation-time hashing and relayout work and stores another
weight representation in the cache. Payload processing uses bounded I/O for
unchanged tensors; a selected tensor's relayout may materialize that tensor in
host memory. Account for compilation wall time, peak memory and cache bytes
separately from inference costs.

The focused tests cover ranks one through five, different element widths,
raw floating bit patterns, aliases, live initialization, source/cache mutations
and unsupported ownership/effects. Reproduce the ordinary paired model tests
with the configured native toolchain and functional simulator:

```sh
python -m pytest merlin/tests/infra/test_weight_prepack_model.py
```

They compare complete recurrent model outputs and state against independent
integer/coordinate references and inspect actual lowered materializations,
runtime weight bytes, ABI descriptors, LLVM and linked images. These are neutral
host-model checks; they do not qualify an accelerator or a production-model
latency result. Logical bytes removed from transpose outputs are movement
accounting, not measured bus traffic. Functional Spike instruction counts are
not accelerator cycles. Use an appropriate paired timing source before claiming
a model speedup.
