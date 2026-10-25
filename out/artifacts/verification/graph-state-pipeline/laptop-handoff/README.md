# PAUSED BY USER — graph scalar-loop pipeline laptop handoff

**Do not resume without a NEW user instruction.** The user explicitly stopped all agents to prepare laptop handoffs. Implementation stopped immediately. Only source/report packaging followed. No owned live job remains; no foreign job was touched. No proof/build/test/qualification/replay ran after the stop. No staging, commit, push, publication, toolchain write or prior capsule edit occurred.

Historical checkout: `/home/jeremy/Projects/merlin`; branch `feat/pass-verification`; HEAD `229f42c627854c49476a101cb393d0b15044fab8`. The three new implementation files are **untracked** and are not in that HEAD. HEAD is the shared checkout position, not an owned commit. All paths below are repository-relative unless explicitly marked historical external.

Latest assignment: `out/artifacts/verification/merlin-passes/progress/planner/assignments/graph-scalar-loop-pipeline/README.md`. It was read once in full before work. An exact copy is under this handoff's `reports/` at the same relative path. Durable resume record: `out/artifacts/verification/graph-state-pipeline/active/README.md`, now marked **PAUSED BY USER**. Earlier numerical, graph-worktree, branch/status and continuous-work instructions are historical and do not authorize resumption.

## Files for the coordinator/laptop

- `SOURCE_FILES.json`: machine-readable exact list of **3 current owned sources, all 55 prior required owned sources, and 74 transitive project source dependencies**. Every source has hash/size, provenance and a copied snapshot path; zero source binding mismatches were observed while packaging. External Python source records are historical dependency information, not a vendored environment or portability proof.
- `sources/project/`: current and prior primary source snapshots at repository-relative paths; also the read-only primary source dependencies. `sources/pinned4499/`: read-only actual producer source dependency snapshot in a separate namespace. Do not overwrite primary production with it.
- `unfinished-owned.patch`: exact source-only additions for the three new unfinished files. `preserved-patches/`: byte-for-byte copies of all four prior owned patches. The original sealed packets remain unchanged.
- `reports/`: current assignment, shared scope/intake/paper alignment, incident and dependency reports, all five current raw stage buffers and capture, all four development compiler attempts with raw output, prior small delivery/review reports, and the exact elision first-failure query/raw stdout/stderr. `STATUS.json`: machine-readable paused status. `FILES.json`: checksums of the handoff contents (excluding itself).
- `assemble.py` documents source/report packaging only. It is not a portability tool, proof, qualification or recommended laptop execution entry point; it contains historical absolute checkout locations.

**Old absolute-path Linux receipts/objects are historical evidence, not laptop qualification.** No binaries, toolchain files, environments or core dump were copied into this source handoff. Do not rewrite paths/hashes in old receipts, copy successful Linux objects as fresh laptop evidence, silently download tools, or present a rehash as qualification. The coordinator will assemble the cross-stream download bundle; this directory is this worker's contribution.

## What was completed before the stop

`src/merlin/verify/lean/Merlin/GraphStatePipelineFailure.lean` compiled with the selected Lean kernel, return code 0, on exact unchanged sealed dependencies. Source SHA256: `c96aa35941bc5cfb38156823fcc6a46cd13653a42f7eb77d6ab3357130c8f215`.

Its definitions/theorems are `finish`, `advance`, `advance_failed`, `fold_failed`, `fold_early_failure`, `loop_fold`, `fold_append`, and `failure_prefix`, in `Merlin.GraphState.Pipeline`. They prove the unchanged sealed `Scalar.loopEval` equals a final arity check of an absorbing finite fold. The result covers arbitrary partial body outcomes, first external/internal error, complete terminal world, zero-trip loops and invalid carry arities. After failure the fold neither invokes the body nor changes error/world. It does **not** assume source success. This is the failure-sensitive representation needed to instantiate W's successful relational trace roots with outcome states.

This is a **development compilation**, not a sealed batch or completed transitive axiom audit. The module introduces no axiom/sorry/admit/native reflection; std3-only transitive qualification has **not** yet run for this new task.

`src/merlin/verify/lean_graph_state_pipeline_kernel.py` is development-only profile scaffolding. It checks preexisting sealed W/scalar/core/sidecar/host identities, byte-copies unchanged foreign objects into one private library without rebuilding them, retains each different source attempt and blocks on native signals. It is **not a finished checker/receipt/profile**. `MODULES` already mentions the planned `GraphStatePipelineControl`, but that file does not exist. No pipeline qualification driver exists. Linux paths and preexisting records need a separately justified laptop context, not string substitution.

`src/merlin/verify/lean_graph_state_pipeline_capture.py` ran in an isolated process importing actual pinned4499 production APIs read-only. It retained five complete exact buffers, re-opened them, and constructed one common complete descriptor/type registry across all stages using the sealed scalar projection. It additionally records actual index constant/add/minui identities. Python parsing/projection/transport remain trusted. Its diagnostics are not Lean acceptance.

Actual retained case: `out/artifacts/verification/graph-state-pipeline/development/actual-all-four-i32/`:

1. source with a dead add and redundant/invariant scalar computation;
2. actual CSE/DCE output;
3. actual LICM output;
4. actual ordered-unroll factor 2 output;
5. actual segment tile 3 output, index width 32.

**Each of the four stages changes IR.** The source loop is `0 .. 24 step 1`, with one i32 carry, captured bias, retained partial `(i32,index)->i32` call, an observer after the loop and an untouched function. The exact user input is retained beside all canonical stage buffers. All five actual modules passed the pinned parser's verification during capture. This is actual producer coverage only: **no Lean pipeline proof, operational model witness, mutation suite, strict receipt, qualification or fresh replay exists for this case**. The first two stages' sealed theorem anchors have not yet been instantiated on this new five-stage capture.

Current task totals: one compiled module; zero tests; zero qualified originals; zero replays; zero native MLIR/device checks. No pipeline acceptance was issued.

## Exact unfinished goal and next work (only after a NEW resume)

Intended final theorem: acceptance of a complete five-stage certificate must establish equality of the complete original/final returned value vector plus persistent world, including the same first error and terminal world, under **one explicit scalar/SCF operational interpretation**. Controls/index arithmetic must be derived, and all frame/scope/carry/body bridges must be proved from checked syntax. An independently reported list of JSON successes or an assumed raw-body correspondence is insufficient.

Missing now:

- No concrete SCF range/control module, index constant/add/minui semantics, signed32/64 safety bridge or `range_count` theorem has been written. The sealed scalar `Model.iterations` callback is still abstract; the new task must replace its use here with a concrete interpretation on the supported static nonnegative domain. Added index operations must not remain arbitrary effectful opaque calls with a harmlessness assumption.
- No scoped SSA clone/ghost-local/capture/carry bridge has been written. Do not equate independently numbered private cells. Establish lexical environment relations, actual ordered operands, continuation/yield/result mapping and shared types/descriptors for all five programs.
- No bridge from real cloned bulk body to W `GroupBody`, or from actual segment add/min/inner loop to W `Nested`, has been proved. W's `checked_composed` explicitly installs the grouped interpretation; that is the gap this task must close, not assume.
- No complete whole-program composition instantiates sealed scalar CSE/DCE+LICM, new operational bridges and W unroll/segment roots. No direct source→final proof exists either. Preserve partial effects/errors/world through every context and observer without a source-success premise.
- Begin with no-tail bulk (`count > factor`, remainder 0). Inline/empty/grouped-tail modes have not been supported by this new checker; refuse them unless independently proved. Huge-trip structural checking must depend on factor/structure, not materialize a trip-sized trace.
- Still needed: independent actual all-four-changed cases (different widths/operand arrangements, multicarries/captures, effectful/failing observers, untouched function), inhabited nonzero and failure/world witnesses, valid/coherently rehashed mutations, exact stage joins, explicit registry/control/index-width refusals, query/object/import/audit binding refusals, new frozen roster/profile, std3 transitive audit, fixed qualification plus exactly one fresh replay batch, postcheck, seal, source patch/dependency closure and paper row. No current evidence can stand in for those checks.

Useful read-only theorem anchors: `Scalar.normalize_sound`, `Scalar.check_sound`, `Scalar.check_composes`; W `GraphLoopsUnroll.CountCorrect`, `Repeat`, `repeat_iff_execution`, `GroupBody`, `bulk_iff_repeat`, bounds/lane control; `GraphLoopsUnroll.Composition.checked_composed`; `GraphLoopsSegment.Execution`, `Model.Nested`, `Model.checked_complete_trace` and signed outer/inner bounds. All sealed definitions remain read-only. The proposed next proof is a concrete nonnegative finite SCF range (recursive with measure `upper-lower`) equal to the progression given by checked W `CountCorrect`, then the absorbing fold/`Repeat` correspondence. This is a plan, not an existing proof.

Suggested first commands after a NEW instruction, from the restored repository (read-only inspection first):

```sh
git branch --show-current
git rev-parse HEAD
python3 -m json.tool out/artifacts/verification/graph-state-pipeline/laptop-handoff/STATUS.json
python3 -m json.tool out/artifacts/verification/graph-state-pipeline/laptop-handoff/SOURCE_FILES.json
```

Then read the copied current assignment and inspect the exact source/dependency inventory. Integrate only the exact owned patch with coordinator authorization; do not merge pinned production. Before any laptop Lean command, explicitly select an available Lean 4.35 context, verify the required source/import/object/sidecar/host provenance, and record whether that route is actually available. The current `invoke` helper assumes historical Linux selections and must not be blindly run on a laptop. Do not start a new numerical/elision job or retry old failed profiles.

For reference, the **historical Linux** actual capture command was:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/jeremy/Projects/merlin-graph-proof-intake/src \
 .venv/bin/python src/merlin/verify/lean_graph_state_pipeline_capture.py \
 --reference /home/jeremy/Projects/merlin-graph-proof-intake \
 --source out/artifacts/verification/graph-state-pipeline/development/inputs/all-four-i32.mlir \
 --destination out/artifacts/verification/graph-state-pipeline/development/actual-all-four-i32
```

It uses an existing destination and must not be rerun over retained evidence. A future authorized fresh run requires a new destination and separately bound context.

## Dependencies, toolchain incident and trust

Pinned actual producer: external historical `/home/jeremy/Projects/merlin-graph-proof-intake`, detached `4499eebb959c0f92cefd92cc6d80646ba66474ed`, read-only. Loaded project source bytes are listed/snapshotted separately. No xDSL↔Kernel semantic bridge is assumed.

Historical Lean executable: `/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean`, SHA256 `bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`, 9024 bytes. Selected `libleanshared.so` SHA256 `efef27df56f2197139afe12f1a2b63913a05f1aba81ce88824c44100431ed8c4`. Commands used `-j1` from the first pipeline invocation. W source/object/sidecar/host records come from its sealed source freeze, SHA256 `9d129bd7e01e80911ef3335c3d19d826a37481a2e31bca6f2469d7b2afb13cca`. Scalar objects came from the unique byte-identical source-matching sealed profile. Existing objects were copied byte-identically, not rebuilt. No full Lean-MLIR matching manifest/broker route is selected; it remains unavailable.

Historical Python environment: `.venv/bin/python` → `/usr/bin/python3.14`; xDSL 0.73.0, pytest 9.1.1, ruff 0.16.10, numpy 2.4.6. This metadata is descriptive and not a portable environment lock. The capture records exact loaded source files and executable; host execution and Python projection/transport remain explicit trust.

Shared SDK/toolchain-binding incident: `Std/Time/Format/Basic.olean` was previously recorded as SHA `7a21e63b182f090864b1efa3953c40bcae932d096e3d5b4c08399a35db5ed3b0`, later `fbe7f0f09cff1d44590c31036271d882ce58c208fcc478604e0c6e506d92a479`; cause **UNKNOWN**. Exact coordinator observation is copied under `reports/out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261023/shared-toolchain-binding-incident-v1/OBSERVATION.json`.

The scalar worker's one check found it absent from 36 actual audit inventories/707 proof objects, with all recorded selected object/tool pins exact. The new pipeline check recorded **3,288 exact unchanged selected bindings**, including W's preexisting sidecars/host libraries and 11 W project objects; the incident target was absent from that selected closure. See copied `active/dependency-check-once.json`. The target was not rehashed/adopted to promote evidence. These findings establish only the selected recorded unaffected context; they do not explain the incident, establish all-toolchain innocence or qualify a laptop. A changed/unconfirmed selected binding blocks that native context; no toolchain repair/download/failed-context retry/rehash-to-promote.

Std3-only policy: `propext`, `Classical.choice`, `Quot.sound`. No sorry/admit/new axioms/native_decide/bv_decide/native reflection/extern substitution. Totality/purity/effect and native arithmetic fidelity remain explicit where abstracted. The new compiled failure bridge has not received the final transitive audit yet.

## Preserved prior results and failures

All **55 prior sources/capsules remain read-only**; exact sources and prior source patches are included, not revised.

| Packet | Sources | Historical status |
|---|---:|---|
| `out/artifacts/verification/graph-state/review/sealed-v2` | 12 | 1,126-file sealed conditional graph packet |
| `out/artifacts/verification/graph-state-chain/review/sealed-packed-v1` | 15 | 12,149-file conditional chain packet; separate direct/composed claims |
| `out/artifacts/verification/graph-state-elision/review/sealed-diagnostic-v1` | 10 | 8,145-file sealed **diagnostic only; ZERO qualified originals/replays** |
| `out/artifacts/verification/graph-state-scalar/review/sealed-v1` | 18 | 11,086-file sealed conditional scalar packet; 95 tests, 20 original/20 fresh replay runs, 19 changed/1 identity, three two-changed-stage chains |

Scalar original audits were std3-only with zero violations. Its total-pure-primitive/partial-effect/common-control/Python semantics premises remain explicit; full model/source success/poison freedom/native fidelity/review approval remain false. Typing/initialized reached operands are separate from execution success and absence of poison. Native MLIR/device coverage was not proved. See copied prior README/delivery/paper rows for exact scopes.

Elision first native failure: Lean return **-11**, empty stdout/stderr, no proof object, cause **UNKNOWN**, during the unknown-barrier original. Exact query SHA `5be0634316b41e4401bd2e18be4c87c764736991cd92a02a916833382617e02b`, 5515 bytes. The first failure report/query/empty raw buffers are copied; do not replace them with equal successful earlier queries. Full qualification/replay was not retried and is not promotable. Large original core remains at `out/build/graph-verification/state-elision/lean-2236518.core`; raw backtrace remains at `out/build/graph-verification/state-elision/native-failure-backtrace.stdout` (not binaries in this handoff). Root cause is not established.

Scalar preserved monolithic execution proof timed out at 60 seconds, then owned cleanup -15; no object, unqualified. Original location: `out/build/graph-verification/state-scalar/model-tests-1/` (the two-carry LICM model test). No retry/timeout increase replaced it; separately proved stepwise execution was later qualified. Observer-deletion and empty-loop partial-speculation countermodels remain in scalar sealed evidence. A packed chain prototype's unobserved/seventh-reproduction status remains unqualified; do not promote it by the later different route's success.

Current pipeline development had three ordinary compile refusals before the corrected source compiled: first missing imported object due private library layout, then proof elaboration failures. All exact source versions/raw outputs are copied under `reports/out/build/graph-verification/state-pipeline/development/attempts/`. These are not native crashes, mutations or qualification. Latest successful attempt is directory ending `c96aa35941bc5cfb38156823fcc6a46cd13653a42f7eb77d6ab3357130c8f215`.

Ancient numerical checkpoints/counterexamples are preserved separately in the main archive: native power→multiplication equality is refuted (f32/f64), no multiplication-policy approval was fabricated; quant-ext native failures and F12/i64 gates remain open. This graph task discharges none of them. The coordinator should retain their original capsules in the cross-stream bundle rather than infer a semantic bridge from graph progress.

## Ownership/publication holds

Only new `GraphStatePipeline*.lean`, `lean_graph_state_pipeline*.py` and future matching tests are owned for this task. Existing 55 sources/capsules, W loop roots, shared `StateEffects`/`ResourceGeometry`/`Foundation`, shared checkers/CLI/registries, production, frozen semantics/SPEC_LOCK, numerical/W8A8/boolean work and other workers' files remain read-only. The task has not manufactured approval, source success, native fidelity, full compiler/model verification or numerical-gate discharge. Local policy/intent/effect/runtime assumptions and paper alignment still need review. No broad integration/publication is authorized. No new agents. **Remain paused until a NEW user instruction.**
