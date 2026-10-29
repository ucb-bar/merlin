# Actual Merlin buffer reuse — conditional live-memory verification

This additive batch targets pinned Merlin shared pass `buffer_reuse`, implemented
by `planning.plan_storage/reuse_storage` at
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. It does not verify LLVM-owned passes.
Implementation is in PRIMARY `feat/pass-verification`. Only13 NEW owned sources
were added; the old35, common definitions, production, registries, frozen
MerlinVerify semantics and other workers' sources remain unchanged. No commit,
push, publication, download, new agent or numerical optimization work occurred.
Observed qualification is recorded in `SEAL.json`; retained evidence and delivery
checks are recorded separately in `EVIDENCE_SEAL.json` and `DELIVERY.json`.

## Exact theorem, domain and allocation premise

`GraphLoopsReuse.Memory Value Public` separates private heap from complete
observable state. `Public` may contain caller memory, scalar registers, returned
values and the complete retained effect trace. Logical objects/epochs map to
physical slots; `Related` compares EVERY in-bounds cell of EVERY live initialized
epoch and requires identical public state. It does not equate independently named
logical/physical heap cells, unused poison, freed storage, or private pointers.

The protocol checks source live/initialized sets, physical reservations, unique
slots for simultaneous live epochs, initialization before access, retained versus
erased allocations/frees, and empty live/reserved sets at the end. Actual source
alloc/init/free instruction positions and pairwise same-slot temporal separation
are also checked. Source allocation/free clears that logical object's modeled
cells; erased candidate boundaries stutter. Initialization establishes all valid
cells, so no later epoch can read stale contents in an admitted program.
Load, store, dimension and retained operations are modeled explicitly.

`step_forward` and `step_backward` preserve the live relation for every instruction.
`execution_forward/backward`, `checked_observations` and
`Model.checked_complete_observations` prove equality of ALL possible complete
public outcomes, including arbitrary admitted partial/effectful retained
operations. `Model.checked_context` permits arbitrary identical caller contexts.
No source-success or primitive-totality hypothesis is used. The model does not
silently turn an undefined/partial primitive into a successful operation; only
complete possible outcomes are claimed, not categorized failure or divergence
trace equivalence.

**Allocation is explicitly TOTAL, PRIVATE and UNOBSERVABLE in this theorem.**
Native allocator failure/exhaustion, pointer identity, allocator events, timing,
async behavior and ABI conformance are not proved. Retained public actions cannot
introspect private allocator history or private pointers in the admitted model.
The checker requires the explicit `total_private_unobservable` argument and the
independently held selected contract digest. Alternative allocator models refuse.
The contract declares a premise; its JSON and hash do not prove physical truth or
independent review. Different user-selected digests represent different policies.
Saved reports never replace fresh checking of actual modules and dependencies.

This premise matters in an ACTUAL produced case: `resource_gap` has three objects,
first/last `memref<2xi32>` and intervening `memref<1x2xi32>`, each8 bytes. Source
frees the first before the middle allocation; reuse retains its slot. Instrumented
live storage peaks at **8 bytes before,16 after**. A capacity8 allocator can reject
the candidate while the source succeeds. Lean preserves this resource countermodel,
an allocation-call-budget countermodel, pointer observations, stale37 and
simultaneous-live nonzero17/29 collisions. This is not an unconditional finite
allocator correctness claim or a production fix.

Supported raw binding: one complete direct allocation-owner block, statically
positive nonempty-rank contiguous memrefs in default memory space, i8/i16/i32/i64,
identical type/shape/layout/alignment within a reused slot, power-of-two alignment,
explicit unique direct frees, synchronous direct load/store/dim uses, and complete
rectangular initialization before reads. Later private accesses use static valid
coordinates and valid dimension axes in this mode. Index context is signed32/64,
with element/relative byte-offset and initializer-increment bounds checked.
Unknown aliases/views/escapes, other private operations, nested noninitializer
private accesses, unsupported types/layouts/spaces, dynamic shapes or multiple
allocation scopes refuse. Identity/declined inputs are separate, not positive
reuse receipts. This validates the stated domain, not every producer input.

## Initializer proof and raw-operation trust

Parameterized `Geometry.decode_valid/flatten_decode/every_cell_covered` derive a
valid multidimensional coordinate for every linear offset. `Rectangular` states
the exact nested0..extent step1 write domain. `rectangular_iff_coordinates` and
`rectangular_initializes` prove every valid offset receives its initialization
value; they do not enumerate tensor-volume-sized lists. `extent_geometry` relates
the model's cell count to the row-major product. Relative byte offsets and every
initializer loop IV/final increment are bounded. Physical base addresses, banks
and capacity are not invented from those relative bounds.

Actual initializer recognition checks ALL loop bounds/steps/IVs, empty standard
yields, index order and final store, its scalar value/captures, every effect-free
prefix, absence of any other first-operation buffer use and all subsequent uses.
Both actual modules are independently parsed, verified and exported; every clone,
SSA binding, retained operation, caller observer/function, type, metadata and
replacement use is compared. Forward maps allow actual many-epoch-to-slot aliases,
with separate scope bindings. Full postconditions reject missing/wrong final
frees and extra operations. No StoragePlan, witness or replay of the producer is
proof authority. Python export facts and effect traits remain trusted.

**No raw SCF/frozen MerlinVerify evaluator bridge is asserted.** Complete
rectangular initializer semantics and canonical retained primitive meanings are
explicitly admitted. Their raw operational conformance, including alpha binding
and purity/partiality, remains a paper-visible adjacent obligation. The current
frozen evaluator has no general SCF route and was not changed. All native,
full-pipeline and numerical gates remain false.

## Existing proof anchors and composition

This extends the existing shared checker's lifetime arithmetic at
`Foundation.disjointLifetime` to live values and complete public outcomes.
The proof directly reuses unchanged `ResourceGeometry.separatedLiveRanges` for
same-slot temporal exclusion. `StateEffects.State` is a compatible function-valued
state carrier (the private heap is a family of optional cell states), but its
deterministic total fixed-cell action laws are not silently substituted for this
new partial allocation-changing semantics. Foundation's full Lean-MLIR route
remains unavailable without an explicitly selected matching manifest. Existing
pinned pure Lean4.35 is used; no download was attempted.

A REAL neighboring pass was captured: the exact reused output of `two_epochs`
is fed to actual4499 `segment_loops`, changing two initializer loops. Its exact
middle hash and actual outputs are retained in `actual-neighbor-segment-v1` and
`neighbor-composition-refusal-v1/GAP.json`. The old segmentation decoder correctly
refuses nested body regions. Composition is NOT proved: it needs a partial
rectangular-initializer SCF execution/epoch-memory bridge and a nested/multi-loop
boundary theorem. Allocation-changing states are not GraphStateChain's fixed
physical-cell renaming. The standalone memory result is not replaced by a fake
join or synthetic before/after pair.

## Cases, witnesses, adversaries and observed checks

Eight independently generated complete source programs were processed by the
actual pinned APIs:2/3epochs, ranks1/2/3, four byte widths, alignments1/8/16/64,
signed32/64, scalar calls and caller memory observations, an untouched function,
simultaneously live OTHER slot, a1e9x1024 compact structural case, and the actual
finite-capacity countermodel. The large shape is not dynamically executed.

Seven actual small cases run with nonzero signed inputs and caller state. Private
allocations begin with nonzero poison73; every private load checks initialization
and liveness, every final free is checked. All retained memory/arithmetic/call
before/after events, returned scalars and caller memory match an independent
reference calculation. Allocation/free observations are recorded separately and
NOT equated. An omitted second initializer exposes a concrete stale-data output
difference. These are finite instrumented xDSL witnesses, not native MLIR/hardware
or a general formal SCF interpreter bridge. A separate Lean model is inhabited:
two epochs produce nonzero observations17 then29 despite initial physical poison91.

The new fixed63-test population includes18 type-valid IR mutations/unsupported
variants: incomplete/wrong-index/wrong-value/missing initializer, early read,
wrong same-typed buffer, overlapping slot alias, added view, missing/early/double/
wrong free, changed alignment/shape/layout, reordered retained calls, changed
caller observation and untouched function. Harmless unsupported variants are
refusals, not claimed compiler bugs. Malformed IR is separate. Kernel facts are
corrupted independently (context, initializer controls/indices, live slot,
signed byte bound). Stale/coherently rehashed selected proof/tool/import/manifest,
contract, compiled object, audit report and intervening input/tree/model/query
writers are exercised. Reports/witnesses are not accepted as authority.
Seven actual identity/declined source inputs are separately retained. Two original
pinned storage tests pass (38deselected); these are planning/lifetime checks,
not an unavailable native/full-Lean-MLIR route.

## Incident, failures, freeze and review boundary

The shared `Std/Time/Format/Basic.olean` incident was inspected ONCE. That module
is NOT selected by this worker's actual import closure and has no prior binding
here. The single stable-stat read matched the coordinator's fbe7... observation;
it was not re-read. ALL actually selected imports/sidecars and host libraries
matched the original fixed bindings. `shared-incident-inspection-v1/ASSESSMENT.json`
records NOT_SELECTED and verified-unaffected context, with no foreign hash refresh.
Own later helper-name cleanup has explicit successful source builds and introduces
no new imports. The foreign cause remains UNKNOWN. No affected profile was promoted.

Failures remain intact: source parser/type/proof-development failures, first query
recursion limits at1024/8192, and audit-formatting attempts. Large serialized
string equalities are now proved symbolically with reflexivity; the finite
protocol is computed separately, retaining the SAME checker. The old strict
single-line auditor refused a wrapped40-root report; an attempted pp.width option
was invalid, and local format.width did not prevent wrapping. The new owned reader
normalizes continuation whitespace only, then invokes the unchanged strict exact
root/standard3 auditor. Unknown/missing/interrupted axioms still refuse. The first
resource-gap input used a cast unsupported by the supplied xDSL interpreter;
v3 uses different ranks at the same width. No numerical interpreter patch or
unexplained HOST retry was used. Earlier data are preserved and not qualification.

`SOURCE_FREEZE.json` fixes13 sources, old35 source/seal identities, the9-object
proof/source closure, independent contract, actual producer/parser inventory,
original foreign import/host bindings, exact positive/refusal cases and roster.
Qualification and ONE fresh process replay use this same profile. Passed pytest
artifacts are retained explicitly. All40 public theorem roots receive transitive
standard3-only audits; runtime query roots are audited separately. Source, tool,
objects, inputs, loaded code and retained artifacts are postchecked before sealing.

Ruff/format, no-regex, no-target-name and artifact-layout checks pass. The repository
structure gate retains only the unrelated `zephyr_model.py` size debt3557>3537.
Old numerical capsules, failures, all-masked NaN/counterexamples, native/review/
capture obligations and the unobserved old full-audit status remain untouched.
The source-only patch and exact prior/common/import/tool/parser closure are in
`DELIVERY.json`; all48 owned sources are byte-compared after empty-tree patch
application. Review/integration remains central; no publication was performed.

| Existing pass/root | Full proved relation and inhabitation | Actual changed boundary/domain | Trust, composition and coverage limits |
|---|---|---|---|
| Merlin `buffer_reuse`; Foundation.disjointLifetime; ResourceGeometry temporal range root | All live initialized logical-epoch cells related to slots; equal complete possible caller/scalar/memory/effect outcomes in both directions; no stale later-epoch reads; valid final reservations/frees; inhabited nonzero17/29 model | Full actual4499 SSA/owner/use/initializer/capture/free/alias/context matching, static positive contiguous integer domain;8actual cases,7finite witness families, valid mutations/controls, compact huge shape | TOTAL PRIVATE UNOBSERVABLE allocation premise; actual8→16-byte countermodel outside it; admitted partial primitive/rectangular semantics and trusted Python projection; raw SCF/native allocator/ABI/hardware unproved; actual neighboring join refused, no fixed-cell-renaming bridge; review pending |
