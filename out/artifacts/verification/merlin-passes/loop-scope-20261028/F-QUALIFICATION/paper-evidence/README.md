# Scoped unroll → ordered 1D segmentation: paper evidence

Current status: Qv6/Fv3 full successor qualification-v2 PASSED the original eight
actual positives, fifteen valid mutant triples, twelve unsupported cases, actual
consumer/CLI and stale-binding checks. A separate fresh Python-process replay
also passed with zero incoming live completions. The exact current worked example
is ../PAPER-EXAMPLE-v2/README.md (44roots,927theorems,std3). Old failures remain.

## Proposed conditional claim

For each admitted complete three-buffer program exported by the pinned Merlin
producer, Lean checks the generated entry roster, SSA/type/control/carry/body and
continuation obligations. `Merlin.GraphStatePipelineInstance.complete` proves
`Merlin.GraphState.Pipeline.PreservesEntries (M primitive external)
originalEntries finalEntries`: every entry identity, input list and initial world
has the same returned values or error and terminal world before unrolling and
after ordered segmentation. Missing entries and invalid input arity are covered
by `evalEntry`. The common model is quantified; this is not an assumption that
the original computation succeeds or the two complete runs agree.

Domain: one changed top-level static bulk loop, nonnegative lower bound, positive
constant step, source iteration count larger than and divisible by factor 2–4,
index64, tile 2–16 and one to four carries. Final short segments, nonzero lower
bounds and nonunit steps are supported. Source body is region-free; complete
unchanged functions and declarations remain bound. General nested/dynamic or
multidimensional/reordered/parallel tiling, tails, inline/empty unrolling,
floating point, async completion and allocation/alias correctness are excluded.

## Existing roots and interpretation

Existing `GraphStatePipeline{Control,Index,Literals,Scope,Loops,Segments,Program}`,
`GraphLoopsUnroll{,Bounds,Grouping}`, `GraphLoopsSegment{,Bounds,Model,State}` and
`GraphStateScalar{Contract,Normalize,Sound,Typing,Initialized}` provide the model
and composition proofs. No frozen definition or lock is changed.

`primitive : Prim → List Int → Int` is a common deterministic total pure
interpretation for supported word operations. It is not a proved LLVM poison or
integer-overflow semantics. Control constants/add/min use the existing concrete
index interpretation; checked bounds establish the signed index64 domain.
`external : External World Error registry` is a common retained-operation
interpretation. Its sole operation contract requires declared result arity on
success. It may fail, mutate the world and observe ordered operands. Both success
and failure worlds are in the result. Calls are retained descriptors; internal
callee dispatch and a native backend implementation are not proved here.

The outcome includes returns, effects represented by the world, first error and
terminal world. Pure private SSA slots need not coincide. Host parsing,
projection, producer attribution, operating system/process transport and the
faithfulness of these declared primitive/external meanings to an application
remain explicit trust boundaries. Human/independent review is pending.

## Actual IR → Lean map

| Actual compiler field | Generated/checking responsibility |
| --- | --- |
| Complete module/function/declaration names, signatures, metadata and order | Structured admitted parser; common descriptor registry; producer reexecution and complete three-buffer comparison; entry roster/type checks |
| SSA operands/results/block arguments, type and every use | Scalar capture numbers binders and uses; typed `Program`/`Expr`, `checkTypes`, common signature registry; no supplied eligibility booleans |
| Loop lower/upper/stride and index constants/add/min | Concrete control descriptor table, `controlAccepted`, `literalsAccepted`, exact control joins and checked unroll/segment bounds |
| Initial carries, lane IVs, ordered yields and returned results | Actual `S0`…`S4` programs, `body_group`, `innerAccepted`, `outer_form`, ordered suffix/context correspondence |
| Retained calls/observers/opaque operands and results | Common operation identities/signatures, sequence-sensitive model effects and checked suffix/inner relations |
| Source → middle → final | Exact original/unrolled/segmented byte refs and common projection; intermediate identity stages are compatibility steps, not claimed CSE/LICM passes |
| Consumer delivery | Actual fresh `check`, exact root/types/std3 audit, same-live completion and revalidation of sources, buffers, query, report and all private proof objects |

The IR parser/projection is Python trust, not a Lean proof of an MLIR parser.
Producer reexecution establishes attribution, not transformation soundness.
The kernel proof establishes the stated relation on the generated actual-program
representation.

## Retained initial worked example

Pinned producer: `merlin-graph-proof-intake` revision
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. Initial before/unrolled/segmented
buffers are in `../witness-v1`'s bound Query source and
`../../Q-INTEGRATION/first-example/stage{0,1,2}.mlir`. Original loop lower3,
upper27, step2; factor2, tile3. It executes IVs
`[3,5,7,9,11,13,15,17,19,21,23,25]`.

F built its own query and witness objects, independently of W's object tree.
`../witness-v1/FINAL_RESULT.json` retains the exact sixteen witness roots, command,
std3 audit and outcomes. Nonzero initialized inputs `[3,5]`, initial world7:
source and final succeed with result `[17]`,world20. A distinct admitted external
interpretation fails at IV9 with external error17,world15; subsequent operations
and final observer do not execute. These are kernel-checked model witnesses,
not execution tests of a native LLVM implementation.

`../development-v1/RESULT.json` binds an additional actual factor3/two-carry
nonaligned-upper case checked through the real gate and consumer. The completed
successor batch is ../qualification-v2/RESULT.json; current source cuts and
separate-process replay are bound in ../FINAL_STATUS.json. Cite the exact
PAPER-EXAMPLE-v2 output when referring to the newly qualified integration.

## Refusals and preserved failures

`../protocol-distinct-v1/RESULT.json` contains first-time actual component-query
refusals for swapped partial effects (`innerAccepted`), deleted final observer
(`suffixAccepted34`) and incorrect inner cutoff (`outer_form`). All are actual
MLIR-verifier-valid mutated complete programs. Component queries deliberately
omit optimizer attribution and never grant consumer authority; live gate
refusals are recorded separately.

`../qualification-v1` retains fixed2M-heartbeat failure on factor4/three carries.
`../protocol-v1/negatives/middle_wrong_call/lean` retains native rc-11, empty raw
output, cause UNKNOWN. These are unavailable checks, not proof refusals. An older
ordinary wrong-call refusal cannot replace the later crash. The successor proof
template is a real source change at the original limit; any later result must
be bound to its new cut. No retry-to-promote or historical-source qualification.

All Linux absolute-path receipts are historical evidence on this host. A laptop
needs an actual new build, checker run and fresh replay; copying reports supplies
no same-live authority. No compiler-wide enforcement, native fidelity, broad
loop family or policy approval claim is made.

Current-source model witnesses: ../witness-v2/RESULT.json binds16actual initialized/success/failure roots to the current Qv6 query. Native compile/audit/types all returned0; final wrapper output-schema handling was corrected OFFLINE, preserving the original error and without rerunning any native command. Human review/native fidelity remain separate.
