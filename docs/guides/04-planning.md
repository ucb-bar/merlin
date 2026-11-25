---
title: 4. Choose regions, fusion and placement
kind: guide
status: current
owner: compiler
last_verified: 2026-11-16
related: [03-preparation, 05-transformations]
code_refs: [src/merlin/compiler/system.py, src/merlin/compiler/backend/interface.py, src/merlin/compiler/lower/c/host_target.py, src/merlin/compiler/quantization/policy_generation.py, src/merlin/compiler/planning, src/merlin/compiler/partition/regions.py, src/merlin/compiler/ir/boundary.py, src/merlin/compiler/backend/source_proposals.py, src/merlin/compiler/lower/interface.py, src/merlin/compiler/lower/interface_readout.py, src/merlin/compiler/lower/quant/readout_match.py, src/merlin/compiler/ir/readout.py]
---

# 4. Choose regions, fusion and placement

[Previous](03-preparation.md) | [Guide index](../README.md) | [Next](05-transformations.md)


Once the imported arithmetic is known, Merlin can compare implementations for
parts of the program. The backend offers implementations. The planner chooses a
compatible combination for the complete program.

A backend can expose `describe_compiler()` using the records in
`compiler/backend/interface.py`. `SourceDomain` describes accepted IR and its
conditions. `ImplementationFamily` names related source patterns and their
fusion preconditions. `TargetPipeline` connects those patterns to actual target
IR, ordered passes, ABI and artifact kinds. The description is an implementation
claim; it does not change the required domain or establish correctness.

Merlin's host CPU is described the same way, by `compiler/lower/c/host_target.py`:
its families are the checked host C emitters, with their numerical contracts and
a source pin of the emitters and host math supplier. Policy generation takes the
descriptions of every selected target, such as a device and the host, and still
requires exactly one matching family for each source operation.

Both discovery routes consume this interface. They check that its source patterns
agree with `describe()` and its pipeline declarations agree with offered lowering
recipes. Reports distinguish described providers from
`legacy_no_compiler_description`. Legacy status preserves existing callers; it
does not qualify a newly generated backend as a complete compiler.

Fusion has several owners. The selected specification defines arithmetic and
hardware behavior. OOT proposals describe implementations of canonical Interface
IR. Merlin owns upstream source matching, conversion and complete coverage in
`compiler/lower/interface.py`. Shared analysis checks live boundaries; the
original caller supplies independent implementation admission, and the planner
selects compatible alternatives. Textual preconditions in a description explain
those obligations but do not execute or prove them.

## What a backend proposal means

A proposal is a structured implementation offer for actual source operations.
It identifies the operations covered, live inputs and outputs, numerical and
layout requirements, execution resource, memory needs, guards, ABI and lowering
recipe. Cost estimates include their assumptions; missing costs remain unknown.

For an Interface backend, `Component.discover()` uses the shared rules in
`planning/interface_candidates.py` to discover source groups and convert them
to actual Interface IR. It then calls OOT `propose_interface()` for target
implementation recipes. Merlin binds those recipes to the original operations
and ports and retains independent admission. The backend never receives the
upstream model graph through this hook.

Older source-proposal providers retain an explicit compatibility route. Their
`propose()` and `propose_page()` hooks are separate from Interface proposals;
an old source ticket cannot silently become a different Interface recipe.

## Follow selection into partitioning

```text
imported program + selected topology + backend components
    -> discover bounded implementation alternatives
    -> check numerical, boundary, memory and resource requirements
    -> choose a compatible region cover
    -> materialize region boundaries
    -> lower selected regions and shared control
```

`SystemCompiler` performs discovery and selection. `PlannedCompiler` consumes an
already selected plan. Both are usable as explicitly selected backend objects
through `merlin.api.compile`; they are not interchangeable with a target name.

The current execution selector requires contiguous source regions in one memory
domain. Within its search budget it tries explicit provider preferences, larger
admitted regions, then stable identities, and backtracks around incompatible
covers or storage limits. Storage accounting retains source lifetimes through
region boundaries, maximum serial scratch and reserved host contexts. It adds
no memory-domain transfers or concurrent execution. Local tiling, instruction
selection and allocation remain inside the selected implementation and must fit
the resource budget promised to the plan.

## Fusion has more than one owner

| Kind | Example | Owner |
| --- | --- | --- |
| Semantic simplification | Remove a redundant representation change | Shared transforms |
| Region fusion | Group matmul with an eligible epilogue | Discovery, legality checks and planner |
| Target instruction fusion | Emit an accelerator's fused operation | Selected OOT backend |
| Execution coordination | Reuse storage across two engines | Dispatch and SoC integration |

For matmul, bias and activation, useful alternatives include a complete fused
accelerator region, accelerator matmul followed by a vector epilogue, a vector
implementation of the whole chain, and scalar computation.

An accelerator-to-vector route requires the accelerator to expose the needed
intermediate values, compatible memory access, and explicit completion before
the vector code reads them. Two coordinated dispatches are not a single fused
kernel. A per-channel epilogue cannot be replaced by per-tensor scaling merely
because a device readout supports the latter.

The shared Interface route now covers a dense contraction with its complete
local zero initializer, either returning raw i32 or using an admitted signed-i8
readout. The readout retains ordered binary32 scales, nearest-even rounding,
saturation and optional ReLU. Exact integer saturation is also represented.
Static final-axis channel scales and immutable f32 channel bias now use explicit
SSA parameters. Their original multiply/add order is preserved; a target
implementation must establish equivalence for every channel. Runtime qparams,
other axes and broader source chains remain explicit compatibility scope.

To review a fusion, open these three places:

| Question | Open |
| --- | --- |
| What source chain is matched, and why is it legal? | [`match_readout`](../../src/merlin/compiler/lower/interface_readout.py) |
| What does the selected backend implement? | The OOT `capabilities.py` declaration |
| How does that Interface group become target operations? | The OOT `conversion/from_interface.py` |

`planning/interface_candidates.py` discovers a raw group and bounded groups along
its unique-consumer chain. `to_interface()` checks complete source coverage and
calls `match_readout()`. A second live accumulator consumer stops this linear
fusion; it cannot be erased to make the group fit. The resulting
`interface.commit` carries its arithmetic in a typed `readout` record, whose
schema lives in `compiler/ir/readout.py`.

For static channel parameters, `readout.v2` lists ordered `scale` and `bias`
steps, each referencing an operand in `interface.commit.readout_params`.
Those operands are real `arith.constant` values. The original f32 bias remains
f32 in this IR; a later integer preload is a target implementation choice.
The shared structural matcher has one owner in `lower/quant/readout_match.py`,
used by both normal conversion and retained numerical proof adapters.

Only then does `propose_interface()` ask the backend to check its implementation.
Target store scales and any synthesized integer preload must preserve the
ordered source arithmetic for every original channel; a capability declaration
alone cannot establish that equivalence. Refused
matches and target proposals retain reasons in the discovery report. The raw
alternative stays available. Selection uses the deterministic cover policy above;
it makes no measured speedup claim.

## Partitioning preserves the program

Boundary analysis finds every live value and effect before outlining. Selected
regions must cover the claimed source operations without dropping outputs or
changing control flow. CPU operations can remain inline. Partitioning materializes
the selected plan; it does not select a new numerical variant.

Runtime guarded variants are a separate mechanism. Guards must be checked before
externally visible effects. A partially executed accelerator region cannot be
retried on scalar code as though nothing happened.

## Open the implementation

| Responsibility | Source |
| --- | --- |
| Whole-program discovery | [`system.py`](../../src/merlin/compiler/system.py) |
| Shared source groups and Interface proposals | [`planning/interface_candidates.py`](../../src/merlin/compiler/planning/interface_candidates.py) |
| Legacy source fusion admission | [`planning/fusion_admission.py`](../../src/merlin/compiler/planning/fusion_admission.py) |
| Proposal checks | [`backend/source_proposals.py`](../../src/merlin/compiler/backend/source_proposals.py) |
| Global region/resource choice | [`planning/cover_selection.py`](../../src/merlin/compiler/planning/cover_selection.py) |
| Live values and effects | [`ir/boundary.py`](../../src/merlin/compiler/ir/boundary.py) |
| Region materialization | [`partition/regions.py`](../../src/merlin/compiler/partition/regions.py) |
