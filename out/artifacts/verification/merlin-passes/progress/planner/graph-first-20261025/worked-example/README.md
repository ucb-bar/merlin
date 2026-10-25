# Worked actual graph change: CSE/DCE followed by invariant motion

This is retained qualified evidence, copied for paper review on 2026-10-25.
No proof/test/native execution was rerun and no new qualification is issued.
Human specification/mapping review remains open. Start with EVIDENCE-SUMMARY.json.

## Actual change

The pinned Merlin producer revision is
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. Both transformations genuinely change
the IR. The original is `original/source.mlir`; actual intermediate/final buffers
are `original/actual/{middle,after}.mlir`. The following explanation abbreviates
the original names without replacing the complete buffers used by the checker:

```mlir
// Original loop, bounds 0..7 step 2, with an i8 accumulator:
%first   = arith.addi %a, %b : i8
%dup     = arith.addi %a, %b : i8
%dead    = arith.subi %a, %b : i8
%product = arith.muli %dup, %b : i8
"opaque.observe"(%first, %i) : (i8, index) -> ()
%next    = arith.subi %acc, %product : i8
scf.yield %next : i8
```

CSE redirects `%dup` uses to `%first`; DCE removes `%dead`. Invariant motion
moves `%first` and `%product` before the loop. The observer and accumulator update
remain inside the loop in the same order, with the same operands and return.
The proof does not reassociate arithmetic or infer that a partial call succeeds.

## Exact Lean connection and retained checker output

`original/actual/projection.json` records actual typed operation descriptors,
source identities, registry, interfaces and each stage. The generated query is
`original/checked/kernel/GraphStateScalarInstance.lean`. Both complete programs
and both independently projected middle terms are present in that query.

The query proves `F0.first_actual` and `F0.second_actual` by instantiating
`Merlin.GraphState.Scalar.check_sound`; `F0.middle_join` checks their middle
program equality. `F0.independently_composed` uses `preserves_trans` with checked
endpoint joins. `F0.actual` supplies a separate direct source→final proof.
Typing and reached-slot initialization are additional roots, not source success.

The original kernel report records returncode 0 and `conditional_graph_checked`.
Its raw transitive axiom audit reports only propext, Classical.choice and
Quot.sound. A retained fresh-process replay reports the same conditional status
and `retained_object_reproduced: true`. See the original/replay report and raw
kernel/audit/statement stdout/stderr files. Compiled objects are omitted from this
portable review copy; their original identities remain recorded, not recreated.

## What the relation says and what it assumes

For a common admitted model and correctly sized inputs, source and target have
the same complete returned values and persistent world on success, or the same
modeled error and terminal world on failure. Private SSA temporaries are excluded.
This is a theorem about modeled programs under explicit local contracts.

Pure eligible nodes mean deterministic total world-free functions. Retained
opaque operations share their exact descriptor/ordered inputs, partial persistent
world interpretation and successful-result arity. SCF control is a common finite
iteration-sequence interpretation; this old packet does not derive concrete
index/range semantics. Even the index constants appear as retained opaque
descriptors in this query, so they must not be described as concrete index laws.
The active scalar-pipeline worker is proving the stronger control bridge.

The pinned xDSL parser/validity checks and complete Python IR projection remain
trusted correspondences. Hashes identify their selected bytes; they do not prove
their semantics. No primitive fidelity, poison freedom, source success, native
MLIR/backend/hardware execution, numerical-law discharge, human review or full
model claim follows. All those receipt flags remain false. This independent leaf
validator is not evidence of automatic admission by the ordinary compiler broker.

## Review and reproduction

The paired Graham handoff requests an independent operation-by-operation mapping,
assumption review, valid mutations, original-bound replay and host-gate audit.
`qualification/REPRODUCE.md` records the exact CLI. Use a NEW output directory
and the original selected artifacts/tool identities. A laptop needs its own
identified environment; do not rewrite old absolute receipts and refresh hashes.

The 95-test qualification and 20 original/fresh pairs in delivery.json cover the
whole scalar family, not just this example and not a new run today. Counts overlap
other packets. The unexplained frontend SDK binding discrepancy and separate
configuration-elision crash remain unresolved and are not qualified by this copy.
