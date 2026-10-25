# Actual Merlin ordered-loop unrolling — frozen qualification v1

This additive batch extends the existing Merlin shared-pass proof roots to actual
`ordered_loop_unrolling` from detached producer
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. New source is in the main checkout,
`feat/pass-verification`; no production, shared proof, old frozen source, registry,
CLI, numerical gate or another worker's code was edited. See `SEAL.json` and the
subsequent `EVIDENCE_SEAL.json` for observed qualification and retained evidence.
A file named as a seal is not itself acceptance authority; the checker reopens
actual inputs and selected dependencies and runs Lean.

## Claim and existing roots

The old `Merlin.GraphLoopsSegment.Execution` is reused byte-for-byte for SCF.
`GraphLoopsSegmentState.CarriedState` supplies the existing full carry plus
`StateEffects.State` state carrier. No separate counted-Kernel semantics is
substituted. `Foundation.lean` ordered-segment/tail-index ideas and the existing
pipeline-checker's `ordered_loop_unrolling`/`extended_obligations` anchors are
extended in additive leaves; Foundation's Lean-MLIR route remains unavailable
without an explicitly selected matching manifest. This profile uses the already
installed absolute Lean 4.35 executable, never elan auto-selection or a download.

`GraphLoopsUnroll.Repeat` represents every original body occurrence and its
intermediate complete state. `repeat_iff_execution` relates an independently
checked arithmetic count to the existing complete SCF relation in BOTH directions.
`grouped_repeat_iff`, `bulk_iff_repeat` and `emitted_iff_source` prove the actual
inline and bulk/tail forms preserve precisely the same original-body IV trace and
final state, for arbitrary partial, possibly effectful bodies. No body arithmetic
reassociation, totality, source success, floating-point law, or hardware fact is
introduced. A failing body is not silently made total. Empty inlining preserves
the initial state and actual result aliases.

`Bounds` derives source/inline IV and increment safety, every group/lane ordinal,
lane offset/addition, bulk bound and bulk final increment in a signed32/64 domain.
The actual guard is `upper + step*factor < 2^(bits-1)`. The final bulk bound can
exceed the original upper bound; the proof preserves this legitimate case.
Kernel arithmetic checks use count, division partition and interval inequalities,
not a trip-count-sized proof expansion.

## Actual boundary and explicit trust

The pinned production APIs run in isolated subprocesses. They generate the AFTER
module; no expected after-program is manufactured by the planner or checker.
Untrusted producer witnesses are retained and never supply proof authority.
Both modules are freshly parsed, verified and exported as complete operation/SSA
and block trees. The new decoder checks every unchanged operation/function,
attribute/property, observer, result use and context, as well as all generated
index constants/adds, each cloned original body in order, carry input/output
threading, captures, all source/bulk/tail yields, optional exact tail and complete
replacement bindings. Forward SSA maps correctly admit many-to-one replacement
aliases; clone-local maps cannot overwrite context bindings. No inverse mapping
is guessed for forwarded or empty-loop results. Extra or missing operations and
work-budget truncation refuse.

Lean checks the resulting control/clone/carry/context model and proves full
trace equivalence and contextual substitution under ONE explicitly shared body
interpretation. Python parsing, export, SSA normalization and code/source
identity observations are trusted. **There is no formal raw xDSL/MLIR-to-this-SCF
operational semantics bridge or opaque-operation alpha-renaming theorem here.**
That remains a paper-visible adjacent obligation, not a discharged gate.

Supported changed boundary: exactly one outer static nonnegative SCF loop per
complete module, positive step, factor >1, signed32/64 context, standard yields,
region-free original body and no unknown loop control attributes/properties.
Arbitrary carried arity, captures and region-free effectful operations are
admitted under the explicit partial body interpretation. Nested bodies, multiple
changed loops, dynamic/negative bounds and unknown control semantics refuse this
checker mode. This is qualified translation validation for this domain, not a
universal proof of all source-level inputs to the producer.

## Actual two-boundary composition

The `bulk_exact` program is actually unrolled (factor3, original bounds3..27,
step2), then its single factor-wide bulk loop is actually segmented (tile3).
The intermediate bytes are identical at both checker boundaries. Each is checked
independently. `Composition.Compatible` additionally binds the exact canonical
bulk-body key, carry type/initial ports and transformed lower/upper/stride.
`groupInterpretation` explicitly installs that key as the factor-grouped original
partial body over `(original state, original-body occurrence log)`.
`checked_composed` then uses the existing segmentation theorem to preserve this
full log and state. Independent opaque interpretations are never silently equated.

This actual no-tail unroll→segment composition uses the SAME SCF semantics.
Composition for inline disappearance, bulk-plus-tail (two loops), segmentation
into dynamic inner bounds, arbitrary adjacent pipeline transforms, counted Kernel,
renderer or hardware is not established. Python's raw-body-to-group interpretation
remains an explicit operational bridge premise even for the supported join.

## Fixed input and fault population

`actual-cases-v3/CASE_PLAN.json` fixes 12 actual input programs independently of
producer witnesses: inline empty/single/multiple, exact bulk, nonzero tail,
nonzero lower/nonunit stride, 0/1/2/3 carries, duplicate captured-yield aliases,
body-only forwarding, initialized memory effects, suffix observers and unrelated
functions, signed32/64, and trillion-scale bounds with a small fixed factor.
The large/small pair has the same nonzero tail shape; its structural-node count
is identical and query growth stays below 1KiB.

`actual-controls-v1` separately retains factor1, disabled and unknown-metadata
unchanged results; none is counted as a changed-loop receipt. Actual producer
index overflow and growth-budget failures are retained separately.

The new 63-test population includes 19 compiler-verified, type-valid IR mutations:
wrong bulk stride/bound, lane offset/IV, dropped/duplicate/reordered effects,
broken carry chain/capture/initialization, bulk/tail yields, tail initial/step/lower,
missing tail, suffix observer, unrelated function and unknown generated control.
An additional empty-result alias observer mutation and compiler-invalid yield
case are separate. Kernel rejection tests corrupt count/partition/index width/
clone index. Selected tool/proof/manifest/revision, compiled object, audit and
intervening input/tree/query/model writers are checked. The audit corruption is
an injected test report, never an accepted theorem axiom.

Eleven finite actual xDSL witnesses compare every original arithmetic/effect
before/after event, full carry outputs and memory observations for three signed
input/capture choices, repeated twice. All values and memory are initialized;
store addresses also match independently enumerated source indices. These are
finite interpreter evidence, not native MLIR or hardware execution or a general
source-definedness theorem. The trillion-scale case is checked structurally and
by the parameterized Lean theorem, not executed dynamically.

## Qualification protocol and preserved failures

`SOURCE_FREEZE.json` pins all11 new source files, old24 source bytes/seals,
complete selected proof object/source closure, actual producer/parser source
identities, runtime observer, actual inputs/original tests, fixed collection
roster and loaded Lean/host dependencies. Objects were copied byte-identically;
no old capsule was rebuilt. `AUDIT.json` covers ALL27 public new theorem roots,
transitively allowing only `propext`, `Classical.choice`, `Quot.sound`.
Qualification and ONE fresh-process replay use the same frozen profile and retain
passed pytest artifact trees (`tmp_path_retention_policy=all`). Loaded Python
modules and every selected source/object/tool/input are rechecked after the runs.
The final evidence seal also indexes retained queries/checks/runtime witnesses.
Eight original pinned4499 unrolling tests pass separately.

Development failures are preserved, never relabeled as qualified:

- `build-v4` caught a Lean structure-field syntax error; `v5` compiled with an
  unused-simp warning; `v6` removed the warning. Later explicit inline bounds are
  built in `v11`; final model/composition in `v12`/`v13`.
- `build-v7` needed the explicit factor inequality in scope; `v8` completed it.
- `development-composition-check-v1` refused an ambiguous model namespace. Its
  compiler recovery `sorryAx` diagnostics are FAILED evidence, not authored or
  accepted assumptions. Fully qualifying both existing model namespaces fixes
  query assembly; `development-composition-check-v2` succeeds.
- `development-tests-v1`: 44 pass, one test assertion fails because the initial
  small pair had no tail but the large pair did. The small independent source
  bound was corrected from103 to110. No semantics/guard/proof was weakened.
  `development-tests-v2`:56 pass; expanded `development-tests-v3`:63 pass.
- Initial `bulk_past_upper` source was actually inline. Cases v2/v3 correct its
  independent bounds/factor to exercise an actual bulk bound past original upper.
  Earlier generated data remain intact.
- Preceding capacity-v1 retention failure remains disqualified as documented in
  its old packet. Capacity-v2 and segmentation-v1 capsules remain unchanged.

Ruff/format, no-regex, no-target-name and artifact layout checks pass. Structure
passes its other checks but retains the unrelated
`src/merlin/runtime/backends/zephyr_model.py` size failure (3557 >3537).
No inherited debt was edited. Old W8A8 numerical failures/counterexamples,
all-masked NaN, native fidelity/review/capture obligations and the unobserved old
numerical full-audit attempt remain unchanged and are not used as graph premises.

## Source delivery and paper row

`source-only.patch` contains ONLY the11 new owned files. Required prior source
closure is the original11-file patch724510de... plus the13-file capacity/segment
patch79e51748... in the old `large-batch-delivery-v1` manifest, and its exact
unchanged shared StateEffects/ResourceGeometry intake. No wholesale4499 merge,
old source rewrites, build products or generated evidence enters this patch.
Use `DELIVERY.json` for exact patch/dependency identities and empty-tree application
byte comparison. Main is the final integration destination. Nothing was staged,
committed, pushed or published by this worker. Review/integration remains central.

| Existing anchor | Actual Merlin transformation | Proved domain/relation and gap closed | Remaining trust/composition gap |
|---|---|---|---|
| Foundation ordered segments/tail-index; shared checker `ordered_loop_unrolling`; existing GraphLoopsSegment/StateEffects SCF roots | pinned4499 `transforms.unrolling.unroll_ordered_loops`, inline and factor-wide bulk plus optional exact tail | Parameterized complete original-body ordered trace and full carry/state equivalence in BOTH directions; actual complete SSA/body/capture/yield/context bindings, aliases and signed control bounds; explicit actual no-tail unroll→segment logged-body join | Trusted Python/parser/projection, raw-SCF and operation alpha-renaming/group interpretation bridge, trusted index-width/runtime context; no full pipeline/Lean-MLIR/count-ed Kernel/renderer/hardware/numerical gate discharge |
