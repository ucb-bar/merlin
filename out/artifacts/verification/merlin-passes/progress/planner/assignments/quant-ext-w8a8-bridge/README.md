# Quant-ext next assignment: bridge the adjacent W8A8 boundary

Existing quant-ext worker owns this task **after** sealing the current schema-4 qualification,
all 60 original replays, source-only delivery and review packet. Preserve that completed
workspace/build byte for byte. Continue in a new isolated source copy/checkout and private
build; do not invalidate the qualified schema-4 evidence by adding to its live source closure.
Staging, commits, central integration and pushes remain held.

## Why this is the next pass-verification task

The retained W8A8 stage `0006_prepare-quantize-int8/after.mlir` and quant-ext stage
`0007_prepare-lower-quant-ext/before.mlir` are the **same 2,998,333 bytes**, SHA-256
`05ba41b431049ffc396f7bc146a45ee7bba1801ced4e682307e6cd3658b19fa7`.
The planner checked the actual files as well as both receipts. This gives a concrete adjacent
boundary for the scoped paper artifact; a byte join alone is not a semantic-composition proof.

W8A8's new `PolicyW8A8Correspondence.check_sound` is parametric in every context and requires
no numerical law: final W output equals its draft reference under `W8A8SourceSemantics.opSem`.
Quant-ext's `QuantExtLoweringIEEE.check_sound` relates the next target to its source under
`QuantExtLoweringMixedSource.sem width`, with the reciprocal law discharged for its explicit
model. The two source adapters have different guards and failure messages. Their names and
shared scalar implementation do not prove that they denote the same program.

## Deliverable

Prove or precisely refute the adapter correspondence needed on the supported width-64
retained boundary, then expose a checked composition result for the supported W8A8 ->
lower_quant_ext pair. This remains lower_quant_ext/source-interface verification work.

1. State the desired relation and domain first. Separate equality of successful values,
   conditional refinement, support, operational typing and initialized execution. Do not
   assert global outcome equality where guards/failure messages or wrong runtime types differ.
2. Prove supported dequantizer/quantizer adapter correspondence in owned additive modules,
   reading W8A8 definitions only. Derive widths, tensor shapes, metadata, bindings and source
   coverage from actual IR. Preserve per-tensor float subtraction and exact reciprocal order.
   Actual runtime typing/initialized parameters must be explicit where needed; SSA annotations
   alone do not prove runtime typing. Prove the required typing preservation or use an existing
   proved result with its exact premises, rather than assuming every intermediate is typed.
3. Lift through complete modules, nested regions, helper calls and common external contracts.
   Any external typing/equality premise must stay explicit. Account for unsupported operations
   and source-stuck/UB/poison cases honestly. If a stronger unrestricted bridge fails, retain
   a kernel counterexample and state the legitimate supported domain separately.
4. Instantiate at the explicit quant-ext reciprocal model. W8A8's policy-correspondence theorem
   permits that context without proving W8A8's seven-plus-three numerical laws. Do not duplicate
   the W8A8 worker's concrete-context/attention task or make whole-model source success a new
   prerequisite for every local composition theorem.
5. Check the entire actual joined boundary: W8A8 target -> complete quant-ext target, including
   all 399 source-selected sites. No filtered model fixture or bag-of-sites check. Keep the
   source-derived W8A8 policy relation explicit; this is not original-float equivalence.
6. Qualify with positive kernels/calls/regions, valid changed-output and wrong-join mutations,
   invalid/mixed runtime widths, unsupported metadata and preserved F12. Run fresh imported
   audits and receipts/replay on a new frozen private identity. Preserve older identities.

The first adapter proofs can use committed W8A8 source-semantics files already present in the
base. Any needed unpublished W8A8 correspondence modules must be an exact read-only snapshot
with declared hashes; do not edit, stage or take ownership of them. Do not apply held patches
to shared main or declare the central combined build integrated. The planner owns that step.

No new Agreement placeholder, desired-law axiom, source implementation substitution, frozen
semantic/spec/lock edit or production fix is authorized. Policy/source-contract approval,
native fidelity, selected final capture and initialized whole-model execution remain separate.

Keep this task path and the current private checkout/next command in the active resume record.
After compaction continue the current unfinished step; do not return an old branch answer or
handoff. Work through proof, checker, actual boundary and qualification before ending the task.
