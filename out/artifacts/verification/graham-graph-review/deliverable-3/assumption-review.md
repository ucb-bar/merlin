# Deliverable 3 — semantic assumptions and theorem strength

**Reviewer:** Graham independent review (agent). **Date:** 2026-10-26.
**Sources reviewed:** verification checkout `58dbc7c00acdb6770d440b773cc2954328c78472` (`feat/pass-verification`, tracked tree clean); producer `4499eebb959c0f92cefd92cc6d80646ba66474ed` in `out/build/upstreams/merlin-graph-proof-intake` (tracked tree clean); handoff base `c399e441e`, reviewed in its working-tree form, which equals `HEAD` and adds the sealed-row sections. Lean `4.35.0` (commit `eb6cafb7290e`), binary sha256 `bf8d54e4…`, `libleanshared.so` `efef27df…`. Every Lean, query, Python-projection, document and packet file this review read is listed with its full hash in [source-identities.json](source-identities.json).

**Legend.** *Verified by me*: I re-ran or re-derived it in this deliverable. *Trusted/unchecked*: I read it but did not and could not prove it. *Evidence absent locally*: the retained record names it, but its bytes are not in this checkout (see D1-G03).

## Conclusion

1. **Axioms.** Clean. I compiled the transitive Lean closure of every reviewed row from source into a fresh directory: 76 modules, 7 retained generated queries, 9 audit files and 2 reviewer witnesses, 94 steps, all returncode 0. On 171 final and supporting roots, `#print axioms` reports only `propext`, `Classical.choice` and `Quot.sound`. An independent environment audit ([ReviewerAudit.lean](axiom-audit/ReviewerAudit.lean)) finds no project axiom, `opaque`, `implemented_by`, `extern` or unsafe/partial definition. The only exceptions are compiler `_unsafe_rec` twins of safe definitions, which the kernel never uses. Theorem counts equal the retained audits: 498 scalar, 891 four-pass i32, 913 symbolic provider, 1009 integer chain, 1056 integer provider. A comment- and string-stripped token scan of all 76 sources, the 7 queries and both reviewer witnesses finds no `sorry`, `admit`, `axiom`, `native_decide`, `bv_decide`, `ofReduceBool`, `implemented_by`, `extern`, `csimp`, `unsafe`, declaration-level `opaque`/`partial`, `macro_rules`/`elab`/`run_cmd`/`initialize` or `skipKernelTC`. The only non-`Merlin` imports are `Init`, `Init.Data.List.Lemmas` and `Lean.Elab.Tactic.Omega`, plus `MerlinVerify.Core.AST`/`Eq` for declared-call; the `Std.Tactic.BVDecide`/`LeanMLIR` imports of `Foundation.lean` lie outside every reviewed closure. *Verified by me.*
2. **No vacuous hypothesis found.** Every satisfiability-sensitive root has a concrete instance that satisfies its premises. For declared-call v1, the retained packet did **not** show this: its `actual_nonzero_witness` evaluates only the logical stages. I constructed a joint instance for the actual i8 root: a model machine with nonzero outputs 33/242 ([DeclaredJointWitness.lean](witness/DeclaredJointWitness.lean)). *Verified by me.*
3. **The allocation premise is necessary.** On a program shaped like the retained `resource_gap` case, the sealed reuse checker accepts. The sealed theorem then equates complete outcomes [7,9,11] from stale memory. Yet the candidate peaks at 16 bytes against the source's 8, and a capacity-8 allocator admits the source but exhausts the candidate ([AllocationGap.lean](witness/AllocationGap.lean)). Total/private/unobservable allocation is therefore a real premise, not a modelling convenience (D1-G08). *Verified by me.*
4. **No claimed composition silently equates two different relations.** The three qualified chains join inside Lean, by quantifier specialisation (∀ models → fixed index model), explicit domain narrowing (all inputs → canonical words) or one shared interpretation. The edges that *would* mix relations are not claimed: shape→memory, possible-outcome ↔ equal-`Outcome`, and observer-only ↔ world/error. The live compiler gate does equate an unrelated local receipt with program-level proof; that is D1-G09, restated here as a relation mismatch.
5. **Theorem strength differs sharply by row** (matrix below). Only the scalar, four-pass and integer roots preserve failures with terminal world. Shape, reuse and memory-SCF equate *possible successful* complete outcomes only. Declared-call equates observer values under total actions, with no world or error. No row proves source success in general, divergence behaviour or poison freedom.
6. **The draft paper needs five wording corrections** (D3-G01). Four retained-text overstatements: unsupported-boundary refusal stated as a property, the scalar row's missing abstract-control premise, the four-pass witness attributed to the wrong query, and "complete" for "possible complete" outcomes. One draft omission: the declared-call native-Action frame premise.
7. **Limits.** The memory-SCF instance query and its inhabited whole-case witness, the six shape witnesses, the pipeline audit buffers and 11,085 of 11,086 scalar sealed-v1 files are absent locally. Their instance-level axiom audits and nonzero runs are not rechecked here; the generic theorems they instantiate are (D3-G06).

## 1. Rows, final statements and queries

Statement types were taken from my own `#check` output, compiled from query bytes whose hashes equal the retained records. Raw output is in [axiom-audit/build/logs](axiom-audit/build/logs) (`print-*.stdout`).

| Row | Generated query (sha256) | Final root(s) | What the root states |
|---|---|---|---|
| Scalar CSE/DCE→LICM (worked example) | `planner/graph-first-20261025/worked-example/original/checked/kernel/GraphStateScalarInstance.lean` (`544b1dfd…`) | `F0.actual`:13, `F0.independently_composed`:41 | `Preserves`: ∀ Value/World/Error, ∀ `Model`, ∀ inputs of correct length, `evalProgram before = evalProgram after` (`GraphStateScalarContract.lean:158`) |
| Four-pass CSE/DCE→LICM→unroll→segment, i32 | `graph-state-pipeline/qualification-v1/suite/test_actual_complete_pipeline_0/checked/kernel/GraphStatePipelineInstance.lean` (`b2d558f4…`) | `complete`:239 | ∀ `primitive : Prim → List Int → Int`, ∀ `External`, `PreservesEntries (M primitive external) originalEntries finalEntries`, where `M` uses the concrete index model (`GraphStatePipelineProgram.lean:26`) |
| Exact integer + four-pass suffix | `graph-state-integer/qualification-v2/suite/test_actual_six_buffer_chain_30/checked/lean/GraphStateIntegerInstance.lean` (`98ab1e5c…`) | `complete`:355, `prefix_actual`:350, `exactMiddle`:354 | ∀ `External` whose successful results are canonical on canonical inputs, `PreservesEntriesCanonical (Suffix.M primitive external)`, with `primitive` the word primitive (`GraphStateIntegerProgram.lean:16`) |
| Shape standalone-v1 | `graph-shape-normalization/standalone-mode-v1/qualification/pytest/test_actual_whole_module_rank_1/proof/Obligation.lean` (`2789b79d…`; the paper example's query per `CHECKER_OUTPUT.json`) | `complete`:15, `memory`:21, `contextual`:27 | ∀ `Interpretation`, ∀ name/inputs/worlds, `complete meaning module false … ↔ complete meaning module true …`: the **same** blueprint under two mode readings (`GraphLoopsShapeModel.lean:37`) |
| Reuse (inside memory-SCF) | instance query absent locally | `GraphLoopsReuse.checked_observations` (`GraphLoopsReuseSemantics.lean:180`) | ∀ `Primitives`, from equal public state, ∃-outcome sets of source (`candidate=false`) and reuse (`candidate=true`) are equal |
| Memory-SCF reuse→segment | instance query `GraphMemorySCFObligation` **absent locally**; template `verification:src/merlin/verify/lean_graph_loops_memory_scf_model.py:136-179` | `Model.checked_chain`:73, `Model.checked_complete`:85, `Model.checked_context`:94 (`GraphLoopsMemorySCFModel.lean`) | ∀ `Interpretation`, three-way ↔ between `outcomes` of the **same** instruction list under mode flags `(false,false)`, `(true,false)`, `(true,true)` |
| Declared-call v1 (i8 [3,47]) | `graph-regions-declared/qualification-v1/fresh-replay/0/Check.lean` (`a4abc041…`) | `actual_complete_observers`:128 | Under per-instruction, packing, native-Action and publication premises: `observations.map` after replacing the region by `native.step` equals the original (`GraphRegionsDeclaredBridge.lean:48`) |
| Fan-out fusion native route (v4) | no instance query reviewed (v4 qualification reported-only locally, D1) | `GraphRegions.checkRealization_native` (`GraphRegionsFusion.lean:106`) | Same observer equality, given the local `implements` premise |

## 2. Relation-by-row matrix

✓ = proved for the row's domain; *prem* = holds only via a stated premise; ✗ = not established; n/m = not modelled.

| Row | Returned values | Complete observer state | Relation kind | Failure preservation (same error + terminal world) | Divergence | Static typing | Initialized slots / reads | Source success | Poison freedom |
|---|---|---|---|---|---|---|---|---|---|
| Scalar | ✓ | ✓ persistent `World`; private SSA excluded | equality of deterministic outcomes, ∀ models | ✓ (`Fault.external` error and `arity` fault, each with terminal world) | n/m: loops are finite `List Int` | ✓ `check_typed` | ✓ `check_operands_initialized` (reads resolve; no value typing) | ✗ | ✗ abstract carrier; flagged ops refused by trusted capture |
| Four-pass | ✓ | ✓ `World`, every defined entry, missing entries, arity errors | equality under one concrete index model, ∀ primitive/external | ✓ incl. first failing external call (`F3.body_group` cases on error) | n/m | ✓ `originalEntriesChecked`/`finalEntriesChecked` | witness instance only (`SymbolicWitness.source_initialized`) | witness only | ✗ |
| Integer v2 chain | ✓ | ✓ | equality, **canonical-word entries only** | ✓ | n/m | ✓ | ✓ runtime representation of reached values derived (`typed_reached`) | under a total local model (`typed_success`) and provider witness | ✗ Int carrier has no poison; native separate |
| Shape | ✓ | ✓ whole `World`/`MemoryWorld` heap and public | set equality of **possible successful** outcomes | ✗ failure = no outcome; categories not compared | ✗ (finite call-depth derivations) | ✓ `Valid` + typed `Runtime` | descriptor conformance built into `Runtime.view` (premise) | ✗ | ✗ |
| Reuse | ✓ (public) | public + live initialized cells only; dead cells and pointers excluded | possible successful outcomes | ✗ | ✗ | geometry ✓ | ✓ protocol: no read before complete initialization | ✗ | stale contents never read (protocol) |
| Memory-SCF | ✓ (public) | as reuse; segmented edge keeps exact heap | possible successful outcomes, one interpretation, 3 stages | ✗ | ✗ | ✓ `Valid` | ✓ protocol + `Run.covers` | ✗ | as reuse |
| Declared v1 | observer values | ✗ no world; observers only | equality of total deterministic logical execution | ✗ failures excluded by premise (no-fault contract) | n/m | ✓ `checkScalar`/`checkKernel` | packing *prem* (`Refines`) | derived from no-fault contracts (`sequence_refines`) | ✗ |
| Fusion v4 | observer values | ✗ | as declared v1 | ✗ | n/m | ✓ | *prem* | ✗ | ✗ |

## 3. Composition edges

| Edge | Left relation | Right relation | Join | Verdict |
|---|---|---|---|---|
| E1 scalar CSE/DCE → LICM | `Preserves` | `Preserves` | `preserves_trans`, `middle_join` by `decide` (`GraphStateScalar.lean:45`) | Same relation ✓ |
| E2 four-pass: S0→S2 → S2→S3 (unroll) → S3→S4 (segment) | `Scalar.check_composes`, ∀ models (abstract `iterations`) | `evalProgram` equality under fixed `M` (concrete `indexIterations`) | ∀-model result specialised to `M`; equalities chained in `F3.actual` | **Differs in quantification**, joined explicitly; sound because the abstract result covers `indexIterations` ✓ |
| E3 integer prefix → four-pass suffix | `PreservesEntriesCanonical` (`ExactModel`, `LocalTyping`, canonical entries) | `PreservesEntries` (all inputs, ∀ primitive) | `compose_suffix` (`GraphStateIntegerProgram.lean:47`), `exactMiddle` by `decide` over normalized entries; suffix instantiated at the word primitive | **Differs in domain**, narrowed explicitly; the composed claim is canonical-only ✓ |
| E4 reuse → segment (memory-SCF) | `outcomes … true false` | `outcomes … true true` | `checked_reuse_then_segment` in one `Interpretation`; middle = same instruction list; `reuseMiddle = segmentationMiddle` is a **string** check (`GraphLoopsMemorySCFModel.lean:48`) | Same relation ✓; actual middle-buffer identity is host-trusted |
| E5 W unroll/segment relations → four-pass | relational complete traces (`GraphLoopsSegment.Execution`, `Repeat`) | evaluator `loopEval` | `repeat_fold`, `execution_fold`, `scalar_execution` (`GraphStatePipelineControl.lean:86,97`) | Bridged by proof ✓ |
| E6 shape → reuse → segment | possible outcomes over typed `Runtime`/`World`, call depth, allocation partial and observable | possible outcomes over epoch `Memory`/`Public`, allocation total | none: `COMPOSITION_REFUSAL.json` | **Mismatch** (state space and allocation premise); correctly refused; adapter unproved |
| E7 shape or memory-SCF ↔ scalar/four-pass/integer | possible successful outcomes | equal `Outcome` incl. error + terminal world | none | **Mismatch**: no failure transport, and no heap in the scalar rows; must not be composed (D1-G04) |
| E8 declared/fusion ↔ any other row | observer values, total actions | world/error or possible outcomes | none | **Mismatch** if composed; not claimed |
| E9 live gate `accepted_pipeline_receipt` ↔ sealed roots | local/identity lemma (`producer:src/merlin/verify/lean_checker.py:441`) | whole-boundary relations above | implicit, by receipt acceptance | **Mismatch in an actual acceptance claim**; this is D1-G09 |

## 4. Premise classification

Each premise has exactly one class:
- **proved**: kernel-checked in the rebuilt closure.
- **reviewed contract**: a precisely stated Lean hypothesis or model choice that I read, found satisfiable by a concrete instance and necessary or appropriate for the claimed use. Its truth for real MLIR, native code or hardware is **not** established by this review.
- **trusted implementation correspondence**: a host/Python/parser step binding actual IR to the model.
- **experimentally tested**: only tests or executions support it.
- **open**: false in a relevant setting, unsupported, or dependent on missing evidence.

Hash = first 16 hex of the file's sha256; full hashes are in [source-identities.json](source-identities.json). Paths are under `verification:src/merlin/verify/lean/Merlin/` unless marked.

**Scalar (worked example).**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| S1 | Pure ops are total, deterministic, world-free functions | `Model.primitive`, `GraphStateScalarContract.lean:99` | `c0802b19dd4be875` | reviewed contract | True for unflagged `addi/subi/muli/ext/trunc`; universally quantified, so no numeric law is used |
| S2 | Flagged (`nsw`/`nuw`) ops are refused, so no poison-producing op enters `Prim` | `verification:src/merlin/verify/lean_graph_state_scalar_capture.py:108` | `d03ed1697cb8316f` | trusted implementation correspondence | Python check only |
| S3 | Retained ops are deterministic in (args, world), share descriptor and order, and may fail with a terminal world | `Model.effect`, same line | `c0802b19dd4be875` | reviewed contract | Any nondeterminism must be carried in `World` |
| S4 | Local successful-result arity | `Model.arity`, same line | `c0802b19dd4be875` | reviewed contract | Matches the MLIR verifier's result counts |
| S5 | Loop control = a common finite IV list from the three bound values; IV via `model.index` | `Model.iterations`, `evalProgram` `:126` | `c0802b19dd4be875` | trusted implementation correspondence | Index constants are opaque calls in this query; no concrete index law; divergence and step ≤ 0 not modelled |
| S6 | Correct input arity | `Preserves` hypothesis `:158` | `c0802b19dd4be875` | reviewed contract | The four-pass and integer rows handle arity faults explicitly |
| S7 | Typing, result shape, reached reads resolve | `check_typed` `GraphStateScalar.lean:17`, `check_initialized` `:58`, `check_operands_initialized` `:67` | `d5e379f393138108` | proved | — |
| S8 | Before/middle/after projection, untouched functions, same middle buffer | worked-example `projection.json`; `middle_join` proves only that the two projected middles are equal | — | trusted implementation correspondence | D2 scope |

**Four-pass.**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| P1 | Word primitives total/deterministic/world-free | `complete` binder `primitive : Prim → List Int → Int` (query :239) | query `b2d558f4…` | reviewed contract | As S1 |
| P2 | External calls: deterministic `run`, successful-result arity | `External`, `GraphStatePipelineIndex.lean:64` | `f78c2e0960a8c130` | reviewed contract | Inhabited by `SymbolicWitness.provider` (pipeline_7 query) |
| P3 | Concrete control semantics for static nonnegative step > 0 | `indexModel` `:79`, `indexIterations` `GraphStatePipelineControl.lean:42` | `f78c2e0960a8c130`, `81de56f72fb5bb24` | reviewed contract | Agrees with `scf.for` on nonnegative bounds; the fallback `[]` is outside the domain and not a semantics claim |
| P4 | Generated index ops do not overflow; `minui` agrees with `min` | `literalsFit`, `F3.unrollBoundsChecked`/`segmentBoundsChecked`, `minui_natural` `:114`, `segment_control_meaning` `:136` | `f78c2e0960a8c130` | proved | Int model, nonnegative domain |
| P5 | Selected index width (`profileBits = 32`) matches the target | query :16 | — | trusted implementation correspondence | Host selection |
| P6 | Raw unroll/segment bodies match the bulk body | `Scope.check_sound`, `F3.body_group`, `F3.outer_same` | `6cd50e08d9a51d66` | proved | — |
| P7 | Entire entry roster, including missing entries and arity errors | `PreservesEntries`, `entries_cons` `GraphStatePipelineProgram.lean:26,38` | `2c807d9a09ec6875` | proved | — |
| P8 | Five-buffer projection | host | — | trusted implementation correspondence | — |

**Integer v2.**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| I1 | Entry values are canonical words | `EntryDomain`/`PreservesEntriesCanonical`, `GraphStateIntegerProgram.lean:11,16` | `43a8e1aaf9929a94` | reviewed contract | Countermodel `noncanonical_entry_refutes_neutral_deletion` (`GraphStateIntegerCountermodels.lean:21`) |
| I2 | `ExactModel`: word primitive | `GraphStateIntegerExpressions.lean:142` | `a4df60f1fda83c33` | proved for `complete` (`⟨rfl,rfl⟩`) | Real-op agreement with `word` for unflagged signless ops is a reviewed contract under I4 |
| I3 | External results canonical on canonical inputs | `complete` hypothesis `typed` (query :356) → `index_model_typing` `:56` | `43a8e1aaf9929a94` | reviewed contract | Countermodel `missing_local_result_contract_refutes_deletion` `:57`; inhabited by `IntegerProvider.laws` |
| I4 | Deleted ops have no hidden effect; signed literals map through `BitVec.ofInt` | projection whitelist | — | trusted implementation correspondence | Countermodel `hidden_effect_countermodel` `:67` |
| I5 | Runtime representation of reached values | `typed_reached` `GraphStateIntegerTrace.lean:62` | `927e945ea615214f` | proved | — |
| I6 | Host error classification | `integer-v2-host-status-limitation.json` | — | open | Static stale-identity path; affects refusal labels, not positive roots |

**Shape.**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| H1 | After-program = the mode-`true` reading of the single projected blueprint; `sourceBinding`/`targetBinding` checked only as nonempty | `Model.Valid`, `GraphLoopsShapeModel.lean:24` | `38952ca358eebd28` | trusted implementation correspondence | Python full-tree matcher (D1-G02) |
| H2 | Runtime descriptors conform to static types | `Runtime.view … (valid : Conforms …)`, `GraphLoopsShapeSSA.lean:16` | `daa129e43fc43806` | trusted implementation correspondence | Countermodels `inconsistent_dimension` `GraphLoopsShape.lean:87`, `inconsistent_metadata` `GraphLoopsShapeCountermodels.lean:41` |
| H3 | Identical cast has no effect or trap and keeps the whole descriptor | `Cast` `GraphLoopsShape.lean:39`, `identical_cast` `:54` | `7d4314ce5f13378e` | reviewed contract | Countermodels `logging_deletion` `:27`, `trapping_deletion` `:34`, `same_type_different_allocation` `:13` |
| H4 | Folded dimension fits the signed index width | `static_dimension` `GraphLoopsShape.lean:62` | `7d4314ce5f13378e` | proved | — |
| H5 | Retained actions/regions share one partial interpretation, receiving explicit args and world only | `Interpretation` `GraphLoopsShapeCode.lean:21` | `57784b3dae571bcc` | reviewed contract | — |
| H6 | Internal calls by finite depth | `invoke`/`complete` `GraphLoopsShapeModule.lean:78,102` | `17ccc635a1718ad7` | proved within the model | Real non-terminating recursion has no outcome |

**Reuse and memory-SCF.**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| M1 | Allocation total, private, unobservable | `Step.allocate`/`release`, `GraphLoopsReuseSemantics.lean:39`; header `GraphLoopsReuse.lean:1-5` | `8b5f9d5a71a7aab8` | **open** | False for finite allocators: reviewer witness `finite_capacity_distinguishes` (`witness/AllocationGap.lean:130`); retained `resource_exhaustion` `GraphLoopsReuseCountermodels.lean:14`, `allocation_call_budget` `:20` |
| M2 | No escape/alias of private buffers into `Public`; pointer identity unobservable | `Related` `GraphLoopsReuse.lean:29` | `51b46ae3c0e1bb64` | trusted implementation correspondence | Countermodel `pointer_identity` `:17` |
| M3 | No read before complete initialization; live slots unique | `Guard`/`Admitted`/`checkProgram`, `GraphLoopsReuseProtocol.lean:59,71,83` | `538c4066cb1dca7a` | proved | Countermodels `stale_cell` `:24`, `overlapping_live_values` `:27` |
| M4 | Initializer covers every cell | `Run.covers`/`materializes`, `initialization_induced` `GraphLoopsMemorySCFComposition.lean:42` | `26b341f8f6e90187` | proved | — |
| M5 | Pure prefix ops: one shared partial relation; captures depend only on `Public` | `PureMeaning` `GraphLoopsMemorySCF.lean:40`, `Interpretation` `GraphLoopsMemorySCFComposition.lean:24` | `b7c665843732da43` | reviewed contract | — |
| M6 | Final IR = `segmented = true` reading via W `Segmented` with the checked tile | `HeapRun` `GraphLoopsMemorySCFOperational.lean:45`, `Boundary.Valid` `GraphLoopsMemorySCFModel.lean:28` | `ee7ad6757ea92124` | trusted implementation correspondence | Lean checks `Segment.Model` numeric/structural facts only |
| M7 | Control and byte-offset bounds | `checked_outer_control`/`inner_control` `GraphLoopsMemorySCFBounds.lean:9,20` | `39517f78dd2987b0` | proved | — |
| M8 | Same middle buffer for both edges | `reuseMiddle = segmentationMiddle` (strings) | `35c8428218b8ae77` | trusted implementation correspondence | — |

**Declared-call v1 and fusion.**

| ID | Premise | Enters at | Hash | Class | Basis |
|---|---|---|---|---|---|
| D1 | Common total primitive interpretation | `primitive` binder, `checked_calls_preserve_observers` `GraphRegionsDeclaredBridge.lean:48` | `238812250de03f62` | reviewed contract | — |
| D2 | Per-instruction capture/compute/synchronous commit/no-fault/disjoint frame | `InstructionContract` `GraphRegionsDeclaredMemory.lean:70` | `655797e30a22b49a` | reviewed contract | Satisfiable (`witness/DeclaredJointWitness.lean:98`); necessity by `nofault_premise_required` `GraphRegionsDeclaredCountermodels.lean:7`, `completion_premise_required` `:18`. Native fidelity not self-approved |
| D3 | Packing initializes every live-in window for every logical state | `initialized : ∀ entry, Refines …` (`Refines` `:60`) | `655797e30a22b49a` | reviewed contract | Satisfiable (`witness:107`) |
| D4 | `publication` (machine live-out windows = `native.step`) | Bridge `:48` | `238812250de03f62` | proved for `native := logicalFusion` | `publication_derivable` (`witness:162`) derives it from D2+D3 for **any** machine; it does not restate region or program equality |
| D5 | Native `Action` frame/locality on logical state | `StateEffects.Action` `StateEffects.lean:18` | `f5bfccb9d8141a79` | **open** | Not connected to machine memory outside the tracked windows: prefix values read only by the suffix have no window, so a machine clobbering them is not excluded |
| D6 | Window disjointness, synchronous/no-wait/no-token fields | `callValid` `:37`, `checkSequence_sound` | `655797e30a22b49a` | proved | — |
| D7 | Exporter/literal serialization, provider intake, locator translation | host | — | trusted implementation correspondence | Packet README names these |
| D8 | Fusion v4 `implements` (native agrees with fused body on live-outs) | `checkRealization_native` `GraphRegionsFusion.lean:106` | `7076119973823d6f` | open | Region-level correctness assumed whole. It does not restate whole-program equality, but it is the full local claim; v1 replaces it with D2–D5 |
| D9 | Stronger wrapper (`actual_strengthened_complete_observers`) | bound-qualification-v1 | — | open | UNQUALIFIED, pre-oracle host gate, cause UNKNOWN; not reviewed here |

**Cross-cutting.** xDSL 0.68 parsing and validity: trusted implementation correspondence. Hash equality binds bytes, not meaning: trusted. Refusal of unsupported boundaries over the entire boundary: experimentally tested (mutation suites, evidence largely absent locally). Finite xDSL executions (memory-SCF 14, shape `[3,2,17]`): experimentally tested, evidence absent locally. Kernel/axiom cleanliness: proved (§5). Ordinary-compiler gating: open (D1-G05, D1-G09).

## 5. Axiom audit

| Row | Retained audit | My rebuild | Agreement |
|---|---|---|---|
| Scalar | `worked-example/original/checked/kernel/audit.stdout`: 498 theorems, std3, no violations; fresh replay the same | 498, std3, no findings | equal |
| Four-pass i32 / symbolic provider | `sealed-v1/AUDIT.json`: 891 / 913, std3 (raw buffers absent locally) | 891 / 913 | equal |
| Integer chain / provider | `audit.stdout`: 1009 / 1056 | 1009 / 1056 | equal |
| Shape | `CHECKER_OUTPUT.json`: 35 roots | 22 rechecked, 0 differences | the 13 others are helper lemmas, not rechecked by name |
| Memory-SCF | `AUDIT.json`: 61 roots | 38 generic roots rechecked, 0 differences | 9 `GraphMemorySCFObligation`/`GraphReuseObligation` instance roots and 3 geometry roots not rechecked; instance query absent |
| Declared i8 | `fresh-replay/0/AUDIT.json`: 5 roots | 5 of 5, 0 differences | equal |

Commands: [build-and-audit.sh](axiom-audit/build-and-audit.sh), [build-inhabited.sh](axiom-audit/build-inhabited.sh), [run-reviewer-audit.sh](axiom-audit/run-reviewer-audit.sh), [summarize.py](axiom-audit/summarize.py). Results: [axiom-summary.json](axiom-audit/axiom-summary.json) and raw `print-*`, `witness-*`, `reviewer-audit-*`, `query-*` and module stdout/stderr in [build/logs](axiom-audit/build/logs). Closure and forbidden-token scans: `axiom-audit/closure-*.json`, including `closure-inhabited-queries.json` and `closure-witness.json` ([scanner](scratch/lean_closure_scan.py)). The compiled objects were hashed into [object-identities.json](axiom-audit/build/object-identities.json) and then deleted; the scripts regenerate them in about 60 s. This is a fresh local compile from source. It is **not** an original-bound replay and does not reuse historical objects.

## 6. Domain inhabitation and nonzero witnesses

| Row | Satisfiability-sensitive premises | Witness | Status |
|---|---|---|---|
| Scalar | none beyond arity (the relation is equality over all models) | no nonzero run retained for this query | not vacuous by form |
| Four-pass | `External.arity` | `SymbolicWitness` in pipeline_7 query: [3,5]→[17]/world 14; IV 4 fault: error 17/world 16 | verified by me. Different query from the paper's i32 example; the provider returns constant 17, so outputs do not depend on the rewritten arithmetic |
| Integer | `typed` external contract | `IntegerProvider.laws`, `actual_source_success`/`final_success` [17]/32, `actual_*_failure` error 17/world 8 | verified by me; constant-17 provider, as above |
| Shape | typed runtime, partial actions | six actual witnesses (41 roots each) | **evidence absent locally**; only the generic `Eval.source_*` lemmas rechecked |
| Reuse | `Primitives`, protocol | `Witness.source_inhabited`/`target_inhabited` 17/29 from stale 91; reviewer `AllocationGap.source_complete`/`candidate_complete` [7,9,11] | verified by me |
| Memory-SCF | `Interpretation`, `same` | generic `Witness.uniform_heap`/`uniform_run_exists` rechecked; actual whole-case `INHABITED` [17,34,29,58] | generic verified by me; instance **evidence absent locally** |
| Declared v1 | D2, D3, D4, D5 jointly | retained `actual_nonzero_witness` covers logical stages only; reviewer `jointly_satisfiable`, `instantiated`, `machine_publishes_nonzero` (33 and 242 in every element) | **verified by me (new)** |

## 7. Necessity against countermodels

| Premise | Countermodel | Result |
|---|---|---|
| M1 total/private/unobservable allocation | Reviewer `AllocationGap`: `accepted` (checker admits the 3-epoch plan), `model_equates`, `peaks` (8 vs 16), `source_fits_capacity_eight`, `candidate_exhausts_capacity_eight`, `finite_capacity_distinguishes` | Necessary. A finite allocator separates programs the sealed theorem equates. The retained `resource_gap` MLIR is absent locally, so the witness models its documented shape (`graph-buffer-reuse/reuse-mode-v1/README.md:52-55`) |
| M1, allocation-call count | retained `allocation_call_budget` | Removing an allocation can remove a possible failure. In the possible-outcome relation that would give the target a success the source lacks |
| M2, M3 | `pointer_identity`, `stale_cell`, `overlapping_live_values` | rechecked, std3 |
| I1, I3, I4, ExactModel | the four integer countermodels | rechecked, std3 |
| H2, H3 | shape countermodels | rechecked, std3 |
| D2, D5 | `nofault_premise_required`, `completion_premise_required`, `alias_/async_changes_consumer`, `private_observer_required`, `suppressed_transfer_fault_required` | rechecked. The last four are arithmetic inequalities illustrating the effect, not machine models |
| S5 / P3 control | none needed | Removing the common-iteration premise is meaningless; the risk is the IR-to-model mapping (D2 scope) |

No premise looked droppable. The strongest premise is D2's ∀-memory no-fault clause: it requires the instruction never to fault on *any* memory whose reads match. That is stronger than needed for the observed run, but it is what the induction uses; weakening it would need a reachable-state invariant.

## 8. Loop control: scalar premise vs operational bridge

The scalar route (S5) treats loop control as an arbitrary function `iterations : Value → Value → Value → List Int`, common to source and target. The index constants in the worked query are opaque calls. It proves nothing about concrete IV values. The four-pass route uses one concrete model: `indexIterations` plus `IndexOp.eval` for `add`/`minui`/constants. It proves the raw unrolled and segmented bodies against it and bridges W's relational executions with `execution_fold`/`scalar_execution`. Edge E2 legitimately reuses the abstract CSE/DCE/LICM result at the concrete model. The documents keep these apart (worked-example README; `PIPELINE_WORKED_EXAMPLE.md`). The draft paper's scalar evidence row omits S5 (D3-G01).

## 9. Fusion and declared-call local contracts

Neither `implements` (v4) nor `publication` (v1) restates final whole-program equality. `implements` is region-level: `AgreeOn liveOuts (native.step e) (fused.execute e)`. It assumes the full correctness of the fused region (D8, open). In v1, `derived_local_contract` derives that region law from the finer premises. I showed `publication` is itself derivable from D2+D3 for `native := logicalFusion`, for an arbitrary machine. The remaining non-derived physical obligation is D5: logical-state frame and locality of the replacement are assumed through the `Action` structure, not obtained from the machine.

## 10. Draft paper (`PAPER_VERIFICATION_DRAFT.md`) against the roots

| Draft text | Finding |
|---|---|
| "Unsupported boundaries are refused." | Overstated as a property. It is trusted-Python behaviour, experimentally tested by mutations; Lean refuses only what it receives. |
| "The scalar and composed scalar/loop results preserve returned values and terminal state on both success and failure." | Correct for modelled finite executions; integer only on canonical words; divergence not modelled. |
| "The view and storage results equate possible complete outcomes … do not establish failure or divergence equivalence." | Correct. |
| Worked-example paragraph (binds five stages, roster, observer, first failing external call) | Correct for the i32 query (`F3.body_group` error cases; `complete` over entries). |
| Scalar evidence row: "Typed SSA and local total-pure/common opaque contracts" | Omits the common abstract iteration premise and opaque index constants (S5). The cited `graph-state-scalar/review/sealed-v1` is 1/11,086 files locally; the reviewable copy is `planner/graph-first-20261025/worked-example/`. |
| Reuse/segmentation row: "Complete public outcomes" | Say "possible complete public outcomes (successful executions only)". |
| Declared-call row | Add D5 and that machine effects outside the tracked windows are unconstrained. |
| `PIPELINE_WORKED_EXAMPLE.md`: "The model witness separately proves [3,5]→[17]/world14 …" | The witness is in the symbolic-provider chain (pipeline_7), not the i32 query. Its constant-17 provider shows inhabitation, not arithmetic sensitivity. |

## 11. What I could not check

- Memory-SCF instance query, `CHECK.json` and `INHABITED.json`; the reuse `GraphReuseObligation` query (D1-G03; 20/1061 local).
- Shape witness queries and the 13 helper roots not rechecked by name.
- Pipeline raw `audit.stdout`/statements buffers (summary `AUDIT.json` only).
- Scalar sealed-v1 bulk evidence (1/11,086 local).
- The retained `resource_gap` MLIR.
- Declared stronger wrapper (UNQUALIFIED; held profiles not rerun).
- Original-bound replay; package 5.
- The truth of every reviewed contract and trusted correspondence for real MLIR, LLVM or hardware. This review approves no numerical policy and no native fidelity.

## 12. Open questions

| ID | Priority | Question / finding | Closure criterion | D1 |
|---|---|---|---|---|
| D3-G01 | P0 for paper text | Draft and worked-example wording (§10): refusal stated as a property, scalar row omits S5, "complete" for "possible complete", declared row omits D5, witness attributed to the wrong query | Coordinator revises each row flagged in §10, or records why not | D1-G02 |
| D3-G02 | P0 for finite-resource, peak-memory or allocator-failure claims | M1 is necessary (reviewer Lean witness) | Keep M1 explicit in every reuse/memory-SCF claim, or prove a capacity/fault-refinement theorem with physical admission | D1-G08 |
| D3-G03 | P1 | Possible-outcome rows (shape, reuse, memory-SCF) preserve neither failures nor divergence. Any edge to an equal-`Outcome` row needs transport | Prove a common relation or transport lemma per intended edge; otherwise state the rows separately | D1-G04 |
| D3-G04 | P0 for shape+memory claims | E6 mismatch; adapter unproved | Adapter theorem over one interpretation with exact middle and qualified replay; retain the refusal | D1-G04 |
| D3-G05 | P1 for any native/physical fusion claim | D5: native Action frame/locality is not derived from the machine; values outside tracked windows are unprotected | Extend `Refines`/`SequenceValid` to every logical slot live across the region (or prove machine-level non-interference), then derive the Action frame | D1-G02 |
| D3-G06 | P1 for independent confirmation | Instance-level roots and witnesses absent locally (§11) | Owner bundles verified against seals, or a newly identified regeneration (package 5) that rechecks `GraphMemorySCFObligation.*`, shape witnesses and the pipeline audit buffers | D1-G03 |
| D3-G07 | P2 | Provider witnesses use constant-17 externals | Add one witness whose observation depends on rewritten arithmetic (e.g. a provider echoing its argument) | — |
| D3-G08 | P1 | Trusted correspondences S2, S5, S8, P5, P8, I4, H1, H2, M2, M6, M8, D7 | Package 2 mapping audit classifies each as reviewed or refuted; the shape/memory-SCF one-blueprint matcher first | D1-G02 |
| D3-G09 | P1 | D9 stronger declared wrapper: premises and joint satisfiability unreviewed; qualification failed with UNKNOWN cause | Separately attributed reproduction, then repeat §6/§9 for `actual_strengthened_complete_observers` | D1-G03 |
| D3-G10 | P0 for "the compiler Lean-checks" claims | E9: the live gate equates a local or identity receipt with whole-program proof | As D1-G09 | D1-G05, D1-G09 |
| D3-G11 | P2 | Scalar/pipeline divergence and `step ≤ 0` are outside the model (S5, P3); host refusal of runtime or nonpositive steps is trusted | Name the exclusion in the paper, or add a step > 0 check in Lean on the projected control | D1-G06 |
