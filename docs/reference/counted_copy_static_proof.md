---
title: Conditional counted-copy static proof
kind: reference
status: current
owner: core
last_verified: 2026-10-08
related: [lowering_pipeline]
code_refs: [src/merlin/targetgen/contract/pointer_storage.py, src/merlin/llvmlower/counted_copy_check.py, src/merlin/llvmlower/layout_observation.py, src/merlin_experiments/phase1/component_copy_proof.py]
---

# Conditional counted-copy static proof

The original source-only roster remains mandatory. A successful link alone
discharges no static obligation. The optional fixed checker below proves a
limited original-tensor to emitted-LLVM theorem under explicitly selected
original software pointer-object preconditions. No tensor values, golden
outputs, shaped allocations or iteration expansion enter this route.

## Original storage selection

`PointerStoragePolicy` requires explicit row-major contiguous storage, input
then output pointer order, disjoint slots, extent derivation from static original
tensor types, byte order and power-of-two tensor alignment. There is no missing
choice default. `OriginalPointerStorageContract` derives every ordered slot's
logical extent, element stride and byte extent from those declarations.
It declares that pointers denote the complete objects when executed; it proves
neither their physical existence nor allocation, lifetime or runtime equivalence.
The shared direct pointer renderer can enforce the selected contract against
its actual call ordering, byte order and alignment.

The optional experiment issuer requires the live original hardware, minimal
software and complete source roster, a protected closed software ABI selection,
and explicit native LLVM observation tools. Fresh authoring freezes this live
selection and its exact `contract/pointer_storage.json` public projection before
the author runs. The public projection contains policy, with no private shapes
or source-member answers. Saved JSON or reconstructed objects cannot issue live
selection or proof authority. Absent selection keeps the original UNKNOWNs.

## Actual product and layout join

The proof reopens the ordinary source-lowering record, emitted LLVM MLIR, stock
MLIR translation, actual selected object command, exact object and linked ELF.
The actual object compiler is re-invoked on the same original LLVM input with
the same driver options to observe its selected DataLayout. A fixed native
program compiles against the selected public LLVM API and queries pointer/index
width, integer allocation stride, ABI alignment and byte order. Missing default
pointer fields never cause Merlin to guess widths. Real tool/header/library
bytes and the complete actual invocation roster are reopened.

These observations do not authenticate historical tool provenance, all
transitive system dependencies or a physical CPU/runtime. A changed product,
missing record or stack-repaired/replaced object without a preservation theorem
leaves the static proof unavailable. Tool processes share a finite total budget;
failed/interrupted attempts remain evidence and issue no observation.

## Supported theorem

| Facet | Required actual mechanism |
| --- | --- |
| Semantic coverage | One complete original integer tensor identity or typed `linalg.copy`, with the actual scalar body yielding its input; one emitted scalar load forwarded unchanged to one output store |
| Index bounds | Closed post-tested CFG, original ordinal zero, step one, exact original element count, signed induction/byte offsets fitting the actual selected index width, correct input/output GEP bases and element stride |
| Complete output coverage | Every original ordinal from zero through count minus one visited and stored exactly once, exact complete original output roster and disjoint input/output ABI |

The checker accounts for every supported operation and branch. All load/store
alignment assertions must follow from the explicit original object alignment
and the actual element stride. Padded LLVM integer allocation cannot pass a
packed dense byte contract. Counter, pointer, bound or alignment contradictions
refute the theorem and give no qualification credit.
Unsupported aggregate literal tokens are refused before the upstream parser
can expand a dense splat into shaped data; the input IR is never rewritten.

Extra definitions, opaque calls/instructions, unsupported control flow,
metadata or poison/provenance flags, non-default address spaces, floating or
packed elements, unsupported original operations, multiple outputs and general
graphs remain UNKNOWN. The checker does not specialize by workload or model.

## Mandatory limits

The source-only evaluation consumes only the actual live fixed proof. It
rederives the theorem when reopened and counts proved, unknown and refuted
facets in the original full denominator. Resource legality, input numeric
domain, dependency legality, physical ownership/lifetime, synchronization,
machine-code equivalence and timing remain separate unproved obligations.
This route therefore cannot independently qualify a full compiler.

Controls execute real stock translation/object/link and native LLVM queries,
including wrong bounds, pointer bases, alignment, padded allocation, stale
artifacts, deleted records, forged handles and actual subprocess timeout. The
evaluator-private IR fixtures test the checker and ordinary transport; they are
not compiler seeds or a supplied implementation for Phase 1 or Phase 2.
