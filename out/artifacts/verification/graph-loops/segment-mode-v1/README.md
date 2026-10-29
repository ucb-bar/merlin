# Ordered SCF segmentation — scoped qualification and source delivery

This is the completed Phase1 graph batch in `/home/jeremy/Projects/merlin`,
`feat/pass-verification`. It verifies a supported, actual Merlin transformation relative
to the recorded decoder and body/context interpretation. It is not a claim of universal
SCF, complete shared-pipeline, numerical, native-renderer or hardware correctness.

## Paper row

| Item | Evidence and exact limit |
| --- | --- |
| Actual Merlin pass | `ordered_loop_segmentation`, pinned `4499eebb959c0f92cefd92cc6d80646ba66474ed`, `src/merlin/transforms/loops.py:segment_loops`. Actual producer source remains read-only in `merlin-graph-proof-intake`. New implementation is additive primary-owned code. |
| Existing anchors | Existing `lean_checker.verify_pipeline_change` LoopWitness obligations, Foundation.orderedSegments/rangeSplit/tileIndex/tileTail and the actual `_LoopRecognizer` boundary discipline. Foundation's full LeanMLIR route is unselected; these new pure modules specialize the ordered-range reasoning without silently importing another semantics. `Execution.state_actions` explicitly links the SCF execution to the unchanged existing StateEffects.execute. |
| Gap closed | The old generated arithmetic/fold obligations did not themselves prove the actual nested execution's complete body/carry/control relation. The new checker binds full actual source/result SSA trees and the new parameterized theorem proves both execution directions, exact ordered IV trace, all carried components and modeled memory/state, including tails and empty ranges. |
| Domain | Exactly one actual segmented source loop per complete module; static nonnegative index bounds; positive step; tile>1; explicitly selected signed32/64 width; one body block with region-free cloned operations. Arbitrary carried arity, captures and admitted partial/effectful body/context interpretation. Source loop attributes/properties, unknown generated control, successors and nested source-body regions conservatively refuse. Unchanged surrounding regions/metadata are preserved through their complete canonical context interpretation. |
| Actual changed boundary | Every source and target operation, result/type, operand, region/block argument, capture, attribute/property/location is exported. The primary decoder builds an injective source/target SSA correspondence, complete body syntax and whole-module hole contexts. It checks actual add/min, outer/inner bounds/steps, incoming carries, source/cloned yields, inner results and outer yield; all suffix observers and the separate untouched function remain bound. Witness/hint files are not consulted. |
| Formal relation | `Execution` and `Segmented` are distinct SCF relational semantics, not the counted Kernel semantics. `join_aligned/split_aligned`, `source_refines/segmented_refines` and `complete_trace_iff` preserve the complete trace/final state for any admitted body relation. `Model.check_sound/checked_complete_trace/checked_context` attach the checked actual model; `checked_carry_state` instantiates every carry slot plus the existing modeled memory state. A partial body is not totalized and source success is not inferred. |
| Signed safety | The actual static parameters establish source constants and generated outer add/increment, tail min, inner IV/increment and every source/target body index within the selected signed interval. This does not infer body memory safety or remove any numerical definedness gate. Index width is an explicitly trusted compilation-context input, checked against each positive actual capture's configuration. |
| Large case | Actual upper1000000000003, lower3, step7, tile17, two carries. It has the same35total structural nodes as the small corresponding case and a query-size difference below1024bytes. Proofs use induction over arbitrary executions, not generated iteration unrolling or a sampled prefix. The large case is kernel-checked symbolically; it is not run by the finite interpreter suite. |
| Actual independent cases | Seven producer captures: simple signed32, nonzero lower/nonunit stride/tail64, empty32, three-carry64, stateful two-carry64, stateful zero-carry32, large64. The actual producer also refuses signed32 control overflow. Exact before/after, full trees, producer commands/source/parser identities and executed capture implementations remain recorded under segment-actual-cases-v1. |
| Refusals | Type-valid actual IR mutations cover wrong outer/inner stride, missing/repeated tail, wrong inner lower, swapped inner initial carries, inner/outer yields, changed captures, suffix observers, untouched function and unknown control attributes. Compiler-invalid IR is separately classified. Stale source/parser/model/tool identities, unsafe signed controls, complete-node-budget exhaustion and intervening original/tree/model/query writers refuse. SSA renaming preserves acceptance; forged witness hints provide no authority. |
| Fixed qualification |88tests passed50.15s, zero skips; one fresh-process replay of the same fixed88-case roster passed50.10s. This includes the preserved48 loop/storage/capacity cases and40 segmentation checks. Both runs retain56actual queries,57CHECK records and6full runtime trace records. The original4499 focused loop/reduction/witness checks separately passed19tests,21deselected. |
| Audit and identities |19public segmentation theorem roots, zero unapproved transitive axioms; actual queries audit16roots. Only propext/Classical.choice/Quot.sound; no native_decide/bv_decide/reflection/sorry/admit/new axioms/unsafe substitutions/downloads. The six required proof objects were copied byte-identically into a new private profile, never rebuilt over older objects. Imported Lean object/side-file and dynamic-library identities, actual loaded Python code, source/parser/tool/query/model identities and postchecks are retained. Fresh replay loaded the same Python code bytes. |
| Finite execution evidence | Six small actual cases, three initial-value/capture seeds including signed32 wrapping boundaries, two successive calls each. Independent finite reference calculations and xDSL execution compare every body event/entry/carry, complete memory contents and untouched-function results. Full source/target traces are stored in RUNTIME.json. This is finite xDSL evidence, not native LLVM/MLIR or hardware execution or a formal xDSL-to-Lean bridge. |
| Adjacent composition and trust | The common StateEffects state/action interface is explicit. Actual neighboring pipeline stage binding/registry integration, raw-MLIR/parser/canonical-SSA correctness, SCF-to-model operational interpretation, body/context interpretation review, actual-model/native/target coverage and the old MerlinVerify numerical semantics bridge remain outside this claim. No pipeline receipt is issued. All prior numerical gates/counterexamples/all-masked NaNs and unobserved jobs remain unchanged. |

## Review and reproduction

`source-only.patch` adds only the nine owned GraphLoopsSegment*/lean_graph_loops_segment*
source/test files. `SOURCE_FREEZE.json` pins source, actual case corpus, selected object/tool
closure, all relevant code identities and the fixed CASE_ROSTER.txt. `selection.json` and
`proof-closure.json` select the explicit existing Lean4.35 without Lake/downloads.
`AUDIT.json` and raw audit output name every public claim root. Each qualification/replay
RUN stores exact command and loaded-runtime observation. `SEAL.json` records unchanged
postchecks; `EVIDENCE_SEAL.json` adds an index of all retained query/projection/check/runtime
artifacts and verifies empty-tree patch application against the frozen source bytes.

The trusted producer/parser selection comes from11actual producer source files and60loaded
xDSL parser files, bound to immutable Git blobs/file identities. Fresh actual files are
parsed and hashed from the same buffers and reopened after proof. Canonical body/context
syntax is a model input with an explicit arbitrary interpretation; this is not an independently
proved raw-MLIR decoder. The permitted numerical/arithmetic body interpretation remains
conditional, and source success is never supplied as an axiom. No floating reassociation or
LLVM-owned optimization is verified or assumed by the new graph proof.

Ruff and original focused tests pass. Recorded repository no-target-name, artifact-layout and
package-doc checks pass. The structure gate retains the inherited zephyr_model.py3557>3537
failure. At the recorded gate, no-regex also reports another owner's
lean_graph_regions_source_build.py:20 re.sub; this worker introduced no regex and did not edit
that foreign file. Gate logs remain available under repository-gates-segment-v1.

The original11-file724510de loop delivery and the frozen four-source capacity mode remain
byte-identical. Capacity-mode-v2's full evidence is retained; the earlier v1 pytest-retention
failure stays explicitly disqualified in capacity-mode-v1-evidence-gap.json. Segmentation's
qualification used explicit retention_policy=all from its first frozen run. New sources and
profile are now read-only. No staging, commits, push, broad4499 merge or production/shared/
frozen-source edit was performed. Independent review/publication and full-pipeline claims
remain pending; the exact scoped source-only batch is ready for the coordinator.
