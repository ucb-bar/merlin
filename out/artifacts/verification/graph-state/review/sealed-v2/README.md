# Graph-state shared-pass review packet

Implementation is in `/home/jeremy/Projects/merlin` on `feat/pass-verification`.
The producer is the read-only detached source at
`4499eebb959c0f92cefd92cc6d80646ba66474ed` in `merlin-graph-proof-intake`.
This packet contains **conditional graph-model evidence**, with pending effect
admission and integration. It is not approved full verification or a full-model
claim. No staging, commits, push, production changes or shared checker edits occur.

## Existing paper anchor and the gap closed

The existing `Merlin.StateEffects.Action` has a total deterministic state step,
write frame, and read-local writes. `commuteActions`, `exchangeAdjacent` and
`preserveExchanges` already prove independent exchanges preserve all state.
The pinned `lean_state_projection`/`lean_state_effects` derive actual effects and
individual adjacent-swap obligations. The new `GraphState.check_sound` closes
the composition link in Lean: it starts with the COMPLETE dense original roster,
checks every actual neighbor and footprint, and reaches the EXACT target order.
It then invokes the existing `preserveExchanges`. Arbitrary sequential context
and any final state observer preserve equality. An event log can be state; it
must be in the complete admitted footprint.

`GraphStateConfiguration.check_sound` covers an actual supported specialization
of `configuration_loops.hoist_loop_configuration`: one top-level flat nonempty
loop, one constant pure setup prefix, and every retained surrounding call.
Setup means **exactly** overwrite its decoded persistent fields. Idempotence is
proved from that meaning, not inferred from a flag. Remaining body actions
preserve those fields by their actual checked frames. The theorem supports
iteration-dependent body actions and does not expand the loop count.
Zero-trip motion is rejected. Unsupported nested loops, guards, multiple setup
prefixes, asynchronous annotations, pointer setup and nonconstant setup are
outside this checker mode. No loop-storage or fan-out owner's work is changed.

## Exact relations and premises

Schedule acceptance proves equality of **every modeled state cell** after the
complete selected action order, in every sequential surrounding context, for
every original `Action` interpretation satisfying its recorded footprints.
The primary-owned wrapper reopens and retains full actual before/after modules,
rederives the pinned complete typed SSA projection, and checks one composed
kernel theorem. Changed operands/results, signatures, untouched functions,
complete order and all actual effect cells participate in the correspondence.
The original host semantic identity must be selected independently for an
admitted claim; a changed declaration cannot inherit the original identity.

Configuration acceptance proves equality of the complete checked Kernel
program under the same `Action` domain and a stronger setup binding:
`run setup iteration = put fields (configured literal)` for every iteration.
Actual constant values and field order are emitted into the theorem. The
Python projection checks the complete invocation interface/attributes, every
retained call/operand/annotation, loop extent/variable, and actual moved prefix.
Lean independently checks the exact full command correspondence, dense static
roster, positive count, setup footprint and all body write conflicts.

These are explicit outstanding semantic premises:

- Actual instructions implement complete deterministic total, read-local,
  write-framed, sequentially complete actions. Traps, partial completion and
  asynchronous effects are outside the domain.
- Declared resource identity/disjointness is correct. Caller memrefs conservatively
  share a cell; distinct resource names alone do not prove independence. The
  admitted configuration field mapping must exclude hidden aliases.
- Observer/timing/device effects are complete. Pure setup has no pulse, event,
  acknowledgement or other effect beyond its exact overwrite. Declaring a
  semantic hash or adapter qualification string does not establish this fact.
- The pinned Python parsers, full IR correspondence and effect projection, the
  restricted primary Kernel projection, Python runtime and external xDSL code
  are trusted. Their measured identities are retained. Lean proves the emitted
  checker's soundness, not a parser correctness theorem.
- Standard memory accesses/indices/allocations require their original selected
  premises where exercised. Arithmetic stays in the same order and is admitted
  only as total/pure with complete effects. No floating-point law is discharged.

## Inhabited domain and proof audit

`GraphStateModels` constructs reusable total framed writes from arbitrary total
functions over the declared inputs, and proves any typed instruction roster
models its footprints. Two separate audited witness modules compare their
complete footprint rosters to the actual generated instance by kernel `decide`.
The schedule witness uses initialized `BitVec 8` cells and inputs 2,3; both
orders produce result 7 and preserve all state. The configuration witness uses
initialized integer cells (initial value -191), actual setup 7, extent 3 and
stride 4; both programs produce observation accumulator -116 and final mode 7.
Thus these premises have explicit nonzero inhabitants. These are successful
total **model** executions. They do not prove actual IR source definedness,
absence of operational poison, runtime delivery or hardware conformance.

All successful proof closures use only `propext` and `Quot.sound`, a subset of
the permitted standard three. There is no new axiom, sorry/admit, native_decide,
bv_decide, native reflection or proof implementation substitution. The raw
transitive audit checks project theorems, private helpers, axioms and forbidden
unsafe/partial/extern/implemented_by declarations in the actual imported
environment. Per-instance counts are 48 (schedule),104 (configuration),118
(schedule witness),120 (configuration witness). These overlapping counts must
not be added together. Every successful audit has zero violations. Exact
elaborated claim types and imported-object hashes are retained beside raw logs.

The selected pure Lean4.35 executable has SHA256
`bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`.
No matching full Lean-MLIR manifest is established; that route remains
**unavailable**. No tool or dependency was downloaded.

## Fixed evidence and reproducibility

`../qualification-v2/PLAN.json` was fixed before measurement. Its complete batch
contains seven actual changed originals, seven fresh-process replays and 18
retained valid mutation refusals. Four originals use actual production
`planning.apply_order` (i8, i16, i32 with extra SSA context, two exchanges).
Three use actual production configuration hoisting (extents1,3,17; strides1,4,16).
All outputs are actual producer outputs; refusals are independently changed
valid IR/Kernel artifacts, not production promises or malformed-certificate passes.
Observer ordering, unadmitted named-resource independence, caller aliasing,
clobbers, unknown calls/regions, altered SSA/untouched functions and changed
configuration signatures/operands/domains/surrounding calls are retained.

The 52 behavior tests pass. They additionally reject malformed exchange hints,
missing footprints/admissions, stale original semantic/source/code identities,
incorrect selected Lean bytes, zero-trip motion, variant setup and asynchronous
setup. Kernel mutations fail actual `by decide`; inventories are not proofs.
Ruff passes. Source-layout/access/library gates:125 passed,1 skipped,1 failed
in the unchanged real-bwrap masking test. Its exact diagnostic is
`PermissionError` reading the masked test fixture, while the test expects a
successful empty read. This is not a green access gate. `check_structure.py`
passes every check except the unrelated `zephyr_model.py` module-size ceiling
(3557>3537); that foreign file remains untouched.

Fresh replay (from the primary checkout, with its existing interpreter):

```sh
PYTHONPATH=src MERLIN_REPO_ROOT=/home/jeremy/Projects/merlin .venv/bin/python \
  -m merlin.verify.lean_graph_state_replay \
  out/artifacts/verification/graph-state/qualification-v2/originals/schedule8/checked/report.json \
  out/artifacts/verification/graph-state/fresh-review-schedule8 \
  --reference /home/jeremy/Projects/merlin-graph-proof-intake
```

Use a NEW destination. Replay validates original source/code/tool/object
identities, reopens actual inputs, rederives the full projection, reruns the
kernel and raw audit, and compares the query and exact claim types. A saved
report is not proof authority. `source_contract_approved`,
`public_acceptance_issued`, `native_fidelity_proved` and
`numerical_gates_discharged` remain false.

Early development evidence and failures stay retained: missing PATH on the
first audit; invalid scf.if fixture without yield; noncanonical Kernel expression
syntax; an invalid namespace alias in the first configuration witness; initial
formatting gate failures. Failed Lean elaboration printed inferred sorryAx and
produced no accepted object; that failed witness is not promoted. v1 precedes
format-only source cleanup; v2 is the current exact source roster. No old failure
was rewritten into a successful receipt.

## Paper row and remaining composition

| Actual boundary | Root/relation and actual binding | Evidence | Remaining paper limits |
| --- | --- | --- | --- |
| Pinned4499 shared scheduling and constant-prefix configuration hoisting; complete before/after identities in each v2 report | Existing StateEffects.Action/preserveExchanges; new GraphState.check_sound and GraphState.Configuration.check_sound; complete all-state equality under the recorded total sequential model, exact overwritten configuration values and complete trusted IR projection | Seven actual changed boundaries, seven fresh replays,18 valid refusals, initialized nonzero witnesses; std3-only raw audits; unreviewed original effect contracts | Conditional model result. Effect/alias/observer admission and Python operational bridge need review. Shared live receipt broker integration and matching full Lean-MLIR selection are pending. No retained-model, native/runtime/hardware, source-definedness/poison, numerical or full compiler claim. |

Within one projection, composed exchanges and arbitrary context are proved.
Across independently projected adjacent boundaries, SSA cell numbering can
change: a checked state/action reindexing bridge or a fresh direct complete
original-to-final projection is required. Equality of unrelated report hashes
does not supply it. Configuration Kernel state and xDSL SSA state also need an
explicit operational bridge for cross-representation composition. Existing
`verify_pipeline_change`/`accepted_pipeline_receipt` tool/source identity and
live-custody interfaces are the intended integration anchor; they are read-only
here. This leaf deliberately issues no such acceptance.

Review should assess the exact effect and overwrite semantics, the actual
Python-to-model mapping, field/alias/observer completeness, supported domain and
claim wording. Numerical counterexamples, F12, scalar-square refutations and all
old quant-ext/native qualification failures remain preserved, separately pinned
evidence. None is repaired, erased or assumed equivalent to the new `Merlin`
state domain. Frozen semantics and SPEC_LOCK are unchanged.
