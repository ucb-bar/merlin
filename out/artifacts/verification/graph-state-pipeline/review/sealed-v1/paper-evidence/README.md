# Paper evidence — qualified scalar/SCF graph composition, review pending

Proposed scoped sentence: **“For the enumerated supported scalar/SCF boundaries, our standalone proof-producing validator checks complete typed SSA and generated regions, calls and surrounding observations, and acceptance establishes equal returned vectors and complete success/error terminal states under the recorded operation/effect contracts.”** The eight-case engineering qualification is complete; independent projection/assumptions review and paper approval remain pending. This is per-boundary translation validation, not correctness of every production-pass input or all graph mechanisms.

## Actual five-stage worked example and retained checker output

Read `PIPELINE_WORKED_EXAMPLE.md` for the compact annotated before/middle/final example. `EVIDENCE.json:new_pipeline` gives repository-relative complete actual buffers and digests, pinned source revision, exact generated query/root statements, raw checker/audit output, original receipt and exact-path/fresh replay. Actual APIs are CSE/DCE→LICM→ordered unroll→segment at pinned4499 (`4499eebb959c0f92cefd92cc6d80646ba66474ed`). Every stage changes actual IR. Unchanged functions, declarations and the post-loop observer are included.

The frozen qualification result is **64 passing tests, eight all-four-changed originals and one positive fresh-process batch of eight replays**, with exact source/profile/tool/parser/object/sidecar/host postchecks. The i32 example audit covers 891 project theorems; selected variants reach 913, use only propext/Classical.choice/Quot.sound and report zero violations. The original acceptance path is the independent standalone leaf, not an ordinary compilation broker gate. Negative adversarial replay probes are separate from the single positive replay batch.

## Exact model, domain, roots and limits

`Merlin.GraphStatePipelineInstance.complete` composes existing scalar normalization/checker roots, actual SSA/capture/carry/body proofs, the unchanged raw early-failure evaluator representation and W grouping/segmentation preservation roots. Per-entry `F#.actual`, `body_group`, `innerAccepted`, `outer_same` and complete continuation proofs are retained. No source-success, assumed whole-program equality, opaque bulk-body correspondence or intermediate runtime-typing premise replaces these proofs. Actual bodies/controls/surrounding SSA are checked, not only hashes or diagnostic trace counts. No separate direct-validation algorithm is claimed.

The model uses Int values, explicit mathematical index controls, a common total deterministic state-free word-primitive function, and arbitrary common partial external-operation meanings with local successful result-arity contracts. The concrete finite nonnegative range is proved to match W's SCF execution; index constants/add/min, generated IVs/strides/lane bounds and lexical carry/result/yield mappings are explicit. All index literals fit the configured signed32/64 profile. W safety roots establish the generated add/min/increment bounds. Native/backend index representation fidelity is still separate.

The supported chain submode is one changed static nonnegative loop, positive constant stride, no-tail bulk, factors2..4, tiles2..16, region-free original bodies, two inputs for changed functions and one to four carries (one/two/three actually qualified). Generated nested segmentation regions are covered. Source index arithmetic beyond checked generated control bridges, original inline/empty/tail modes, internal calls, unknown CFG/regions and unchanged SCF entries needing another bridge remain unadmitted. Retained-model coverage is not qualified. The trillion-trip case has a 23 KB structural query and no trip-sized proof trace.

Boolean component checkers have Lean soundness roots (`Scope.check_sound`, `checkControl_sound`, `checkLiterals_sound`, existing typing/normalization). The whole leaf produces a closed quantified theorem that the kernel checks. Python parsing/projection, metadata/declared-call signature checks, process/path/hash/archive authenticity and receipt transport remain trusted. There is no theorem that an arbitrary Python status string is sound. Data primitive/native meaning, runtime word/opaque-value representation, external implementation/delivery, numerical definedness, no poison, alias/completion/allocator/world fidelity and independent contract review remain explicit separate obligations. These passes preserve the full ordered effect/call stream; no additional alias-independence law is silently inferred. Debug locations, SSA spelling, costs, timing and compiler diagnostics are not modeled runtime observations.

The separately named symbolic provider model proves actual source/final success [17]/world14 and the IV4 fault error17/world16 with nonzero [3,5] inputs and ordinary wrapping word functions. Source/final reached-operand initialization is separately proved. This is an inhabited modeled domain, with source success/failure derived rather than assumed; it is not actual runtime delivery or universal source definedness. The earlier literal witness's 60s timeout remains immutable/unqualified and was not retried, limit-raised or promoted by the new route.

The field map and preserved smaller qualified example below supply additional stable evidence without rerunning old qualification.

## Existing qualified worked example (reused unchanged; no rerun)

Full exact source/middle/final:

- `out/artifacts/verification/graph-state-scalar/qualification-v1/originals/chain/source.mlir`, SHA256 `6640d030463e20516948e512f0bd26736067c88ef11a7a332f669a8db4b86d4f`.
- `out/artifacts/verification/graph-state-scalar/qualification-v1/originals/chain/actual/middle.mlir`, SHA256 `e9e448db30daf78e97ebecc3c660ffd27589a7d404f2d4069f3c7779cbb0a12e`.
- `out/artifacts/verification/graph-state-scalar/qualification-v1/originals/chain/actual/after.mlir`, SHA256 `acc07b3151ca0a6a7df07015edd8fc73dbc5ce02e10775e0239d4db3aa906137`.

Readable excerpt of the actual edits (the unabridged buffers above remain authoritative):

```mlir
// Original loop body, i8 data, index IV 0,2,4,6:
%first = arith.addi %a, %b : i8
%dup = arith.addi %a, %b : i8
%dead = arith.subi %a, %b : i8
%product = arith.muli %dup, %b : i8
"opaque.observe"(%first, %i) : (i8,index) -> ()
%next = arith.subi %acc, %product : i8
scf.yield %next : i8

// CSE/DCE middle: %dup aliases %first; %dead is absent.
// LICM final: %first and %product are before scf.for.
%first = arith.addi %a, %b : i8
%product = arith.muli %first, %b : i8
%result = scf.for ... iter_args(%acc = %a) -> i8 {
  "opaque.observe"(%first, %i) : (i8,index) -> ()
  %next = arith.subi %acc, %product : i8
  scf.yield %next : i8
}
func.return %result : i8
```

The changed definitions map to equal logical pure-expression trees in `Scalar.normalize`; the live observer remains an ordered `Code.opaque` node with both actual operands. Its partial error/terminal world remains observable. The `Code.loop` keeps bounds, stride, initial carry, lexical body, yield slots and continuation. Hoisted pure expressions shift under the exact binder maps, rather than requiring private stack equality. The checked middle is the same program in the first result and second source. Source, middle and final public results are modeled as complete ordered vectors.

Existing roots are `Merlin.GraphState.Scalar.normalize_sound`, `check_sound`, `check_typed`, `check_composes`, `check_initialized` and `check_operands_initialized`. The actual query roots include `Merlin.GraphStateScalarInstance.F0.actual` and the exact source/middle/final joins/composed and direct routes listed in `EVIDENCE.json`. Original checker report: `.../originals/chain/checked/report.json`, status `conditional_graph_checked`. Exact proof query, raw kernel buffers, transitive audit and statements are retained under `.../originals/chain/checked/kernel/`. The original receipt is `.../originals/chain/receipt.json`; the ONE retained fresh-process replay report is `.../replays/chain/report.json`, status `conditional_graph_checked`, with retained object reproduction true. `EVIDENCE.json` binds all these exact files by repository-relative path, hash and size. The sealed batch has 95 tests, 20 originals and one batch of 20 fresh replays; this example is one of the three changed two-stage chains.

Valid negative: actual compiler-valid observer deletion, kind `drop-observer`, under `out/build/graph-verification/state-scalar/qualification-v1/tests/test_actual_compiler_valid_mut9/`. `mutation.json` records compiler verification and the changed actual buffer. The retained Lean kernel report has ordinary rejection return code 1 (not a native signal); `checked/kernel/kernel.stdout` records failure of the decided acceptance obligation. Removing the observer changes the normalized effect skeleton, even if returned arithmetic still looks the same. No new negative run was performed to write this packet.

## Complete IR-field map and explicit trust

| Actual compiler field | Checked Lean representation / obligation |
|---|---|
| Module/function roster, symbol, visibility, full function type, attributes/properties | Complete Python parser projection checks equality; function input/output type vectors and raw Programs enter Lean. Metadata parsing/projection is trusted Python, not a Lean parser theorem. |
| Scalar operation class, flags, attributes/properties and widths | Exact closed unflagged pure whitelist becomes `Prim`; everything retained outside it receives a complete common descriptor/signature identity or is refused. Pure whitelist interpretation is an explicit total deterministic state-free function. |
| Every ordered SSA operand, result slot, block argument and capture | Separate lexical de Bruijn indices in each complete `Program`; `checkTypes`/normalization validate availability, types and checked binder maps. Source/target private cells are not equated. |
| Scalar arithmetic inputs/constants and returned uses | `Expr` trees/complete ordered return vector, with explicit operand order. No reassociation/numerical law is inferred. |
| Opaque operation/call descriptor, complete arguments/results | `Registry`/`Signature`, ordered `Program.opaque`/`Code.opaque`; same possibly failing persistent-world meaning across stages. Successful result arity is a LOCAL contract; success/purity is not assumed. |
| SCF lower/upper/stride, iter_args, IV, captures, yield/results, continuation | `Program.loop`/`Code.loop`, exact type/count and scope checks; old model uses one common finite iteration callback. NEW work derives concrete nonnegative controls and links to W roots. |
| Entire source/middle/final buffers and checker tool/query/object/audit | Exact receipts and original/fresh replay bindings. Hashing/host delivery and Python transport are trusted, not a Lean theorem of archive authenticity or runtime execution. |

Relation: for every modeled input vector of the checked arity and every persistent world, equal successful returned vector/world OR the same modeled error and terminal world. Bodies/opaque calls may fail. Typing/reached operand initialization do not imply successful execution or no poison. There is no allocator, alias, completion or device model here: effects are included in one complete common abstract world and their physical fidelity is an external premise. No allocator privacy, alias independence or completion law is silently discharged by scalar normalization.

The model is inhabited (e.g. wrapping integer pure functions and an arity-correct external meaning), but the old arbitrary primitive/control interpretation is not numerical-source definedness or native model fidelity. The preserved scalar packet contains nonzero initialized and error/world model witnesses separately from typing. All numerical counterexamples, unknown native failures, configuration-elision zero qualification and the unresolved SDK binding incident remain separate.


## Negative cases, preserved evidence and review state

EVIDENCE.json records 20 compiler-valid actual boundary mutations, including observer deletion, plus the frozen suite's malformed type/call/region cases, different-middle substitution and nine coherent query/object/header/projection/audit/import/statement/profile/original forgeries. Rejections distinguish trusted host domain/projection checks from ordinary failed Lean obligations; timeouts/signals/setup failures do not count as refusals. All before-attack candidates and exact changed artifacts are retained.

The fixed SDK discrepancy remains unexplained; the affected target is absent from selected imports and no changed foreign identity was adopted. Elision remains zero qualified originals/replays. All55 prior sources/capsules, first failures, SPEC_LOCK and numerical counterexamples/gates remain preserved. Native power→multiplication equality remains refuted; this graph work creates no multiplication-policy approval or numerical semantic bridge.

Graham independently reviews projection/assumptions; no review approval is fabricated or duplicated. Broker/fail-closed ordinary compilation, LLVM/OOT/ABI/runtime/hardware fidelity, full-model/all-graph coverage and publication approval remain separate. The user's panel-pipelining/gSIM figure is execution evidence for another boundary and receives no Lean status or performance remeasurement from these scalar/SCF roots.
