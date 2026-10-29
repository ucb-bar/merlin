# PAUSED BY USER — laptop handoff

The latest genuine user instruction is STOP. Do not resume from older continuous-work
or coordinator prompts. This packet documents unfinished work; a NEW user instruction
is required before implementation, builds or qualification resume.

## Checkout and scope

- Original path: `/home/jeremy/Projects/merlin`.
- Branch: `feat/pass-verification`.
- HEAD: `229f42c627854c49476a101cb393d0b15044fab8`.
- Latest assignment: `out/artifacts/verification/merlin-passes/progress/planner/assignments/graph-declared-call-correspondence/README.md`, copied to `assignment/README.md`. It was read once in full.
- Owned unfinished work: ONLY NEW `GraphRegionsDeclared*.lean`, `lean_graph_regions_declared*.py`, matching new tests and private graph-regions-declared outputs.
- Six current source additions are unstaged/uncommitted. No current test file exists.
- No owned live jobs were observed at stop. Earlier owned development jobs completed; no interruption/termination was necessary. No foreign job was touched.

All paths below are relative to the repository, unless explicitly described as an
original Linux path. Receipt paths and executable hashes from this Linux machine
are historical evidence, NOT laptop qualification. This handoff does not fix
portability or authorize changing their identities.

## Files delivered

`SOURCES-AND-DEPENDENCIES.json` is the machine-readable inventory. It lists:

- All **6 current owned sources**, exact hashes and copied `source-snapshot/` files.
- All **31 prior required owned sources**, unchanged, with `required-prior-source-snapshot/` copies.
- **55 transitive project Lean sources**, including owned sources and read-only shared/frozen dependencies, copied under `project-source-dependencies/`.
- **20 statically resolved local Python source dependencies**, copied under `project-python-dependencies/`. Dynamic reference/provider imports have a separate observed source list; do not mistake the static Python scan for a complete runtime library proof.
- **2,080 external Lean source identities** from the historical full source-bridge closure; no external binaries or whole packages are copied.
- **86 distinct observed/required reference-provider source paths**, pinned to `4499eebb959c0f92cefd92cc6d80646ba66474ed`, under repository alias `reference-4499...`.
- Copied small evidence and exact paths/hashes for larger prior reports/seals.

`current-work.patch` is the exact source-only addition patch for these six unfinished
files: **48,736 bytes**, SHA-256
`6fa2b406d995ebb26411db0e111064baf78855b2f899e592ec28b108e4108c19`.
Its construction checked that the six files were untracked additions. No patch was
applied, staged, committed or published during handoff preparation.

Prior sealed patches are copied byte-for-byte to `prior-sealed-patches/`:
`graph-region-v4.patch` (the old ten-source packet) and
`source-bridge-partial.patch` (the later 21-source packet). Their originals and seals
are untouched. The source/dependency snapshots are data copies, not permission to
edit shared, frozen, production or another worker's files.

## Current implementation — what actually compiles

The current six additions are:

| Source | Implemented status |
| --- | --- |
| `src/merlin/verify/lean/Merlin/GraphRegionsDeclaredScalar.lean` | Compiled ordered canonical SSA scalar DAG. Constants, add/sub/mul, truncation, signed/unsigned extension, widths/literals/full canonical properties. Arbitrary common primitive interpretation; no semantic hash equality. |
| `src/merlin/verify/lean/Merlin/GraphRegionsDeclaredMatch.lean` | Compiled complete standalone scalar-function AST matching: module/function/header/argument/result widths, all operation properties/SSA/return order. `checkScalar_sound` supplies common ordered `ScalarExecution` for both matched buffers. Names/visibility are explicit header parameters. |
| `src/merlin/verify/lean/Merlin/GraphRegionsDeclaredEvaluation.lean` | Compiled `Program.locality` and `Program.typed`: finite static operand bounds control environment reads; type preservation follows from an explicit per-primitive `PrimitiveInterpretation.preserves` contract. No new axiom. |
| `src/merlin/verify/lean_graph_regions_declared_context.py` | Development fixed-context selection/import discovery/private build and provider-host intake. No affected source-bridge imports. Helpers are not yet qualified; old development intake hashes may be stale after helper edits. |
| `src/merlin/verify/lean_graph_regions_declared_capture.py` | Executed once successfully as source-only intake of actual pinned source/provider declarations and original virtual-address Kernel. No machine execution, numerical oracle or physical compilation. |
| `src/merlin/verify/lean_graph_regions_declared_ast.py` | Complete relevant Core AST literal serializer, named trusted host component. Ruff passed; not yet exercised in an actual exported-buffer Lean checker instance. |

Important limits:

- These are initial compiled framework lemmas, **not a completed declared-call verification**.
- `ScalarExecution` is the newly stated common-primitive interpretation of the
  admitted canonical scalar DAG. It is NOT the frozen `MerlinVerify.denote` evaluator
  and NOT a native machine implementation. Do not silently identify them.
- The primitive typing law is an explicit parameter, not a proved native contract.
- No actual retained provider source/declaration pair has yet been instantiated
  in Lean. No current transitive axiom audit, mutation suite, formal nonzero witness,
  complete call/window checker, per-instruction machine contract proof, local fusion
  agreement or complete observer theorem has been finished for this new task.
- No fixed current-task qualification roster or fresh-process qualification batch
  has run. No current-task qualification/approval/native/full-model flag is true.

## Independent checking context and SDK/toolchain incident

The previous 21-source bridge is permanently preserved as **PARTIAL/UNQUALIFIED**.
Its completed run had **53 passing tests, 34 proof processes, 542 std3 prints**;
that historical execution result DOES NOT override the failed live binding.

The unresolved selected object is Linux
`~/.elan/toolchains/leanprover--lean4---v4.35.0/lib/lean/Std/Time/Format/Basic.olean`:

- Recorded expected SHA: `7a21e63b182f090864b1efa3953c40bcae932d096e3d5b4c08399a35db5ed3b0`.
- Independently observed SHA: `fbe7f0f09cff1d44590c31036271d882ce58c208fcc478604e0c6e506d92a479`.
- Same 257,512-byte size. Cause **UNKNOWN**. File metadata predates the run, so no
  post-run write or root-cause attribution is established.
- No toolchain write, hash reselection, repair, import trimming or retry-to-promote
  may promote that failed profile. A later matching read is not a repair.

Authoritative prior status:
`out/artifacts/verification/graph-regions-source/STATUS-CURRENT.json`, corrected
`FINAL_STATUS-v2.json`, `DEPENDENCY_DRIFT-v1.json`,
`DEPENDENCY_DRIFT-independent-v1.json`, `SEAL-v2-partial.json`.
The coordinator observation is
`out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261023/shared-toolchain-binding-incident-v1/OBSERVATION.json`.
Small reports/observations are copied under `evidence/`; larger files have exact
paths/hashes in the inventory. Any link to that older source operational bridge
remains explicitly UNQUALIFIED.

A separate pure context was examined for this NEW task. The unchanged earlier
qualified graph-v4 manifest binds **653 Init modules and 3,300 source/object/sidecar/
shared-runtime references**; all matched at intake and Basic.olean was absent.
The initial scalar actual import graph had **659 entries / 6 project modules**,
including unchanged Core AST/Eq syntax and GraphRegions/GraphRegionsFusion/StateEffects;
it did not import the affected semantic profile or Mathlib. This finding permits
independent development in that domain; it does not repair the earlier profile.

The current helper discovers imports through pinned Lean `--src-deps`, rejects
standard sources outside the earlier fixed Init manifest, rejects unknown/affected
project imports, and builds only private objects. Core AST/Eq were checked against
unchanged primary HEAD; AST additionally has the existing SPEC_LOCK pin. The initial
matcher build failed then was repaired through an owned development compile; the
last evaluation compile succeeded. **No complete fresh current-file identity/audit
packet has been rebuilt after those edits.** The development context records are
not final qualification receipts.

Provider host source-only intake recorded **526 loaded modules and 51 mapped host
files**. The fixed-host development snapshot checks those actual files, the earlier
provider interpreter and earlier compiler-oracle identities. More work is needed
before a qualification claims the complete actual parser/provider/runtime closure,
including any late imports. Some context `new_host` references predate subsequent
helper edits and must not be described as current qualified bindings.

Pinned versions, as historical dependencies:

- Lean **4.35.0** Linux executable SHA
  `bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`, `-j1`.
- Prior broader Core/Sem route used Mathlib
  `56a005855ea17e0b0958f73ec801ae140b0a1679`; it is NOT imported by the new pure layer.
- Primary Python **3.14.7**, xDSL **0.73.0**, pytest **9.1.1**.
- Existing compiler oracle: model2MLIR virtual-environment Python **3.12.14**,
  torch-mlir **20261001**. Preserve lexical virtual-environment invocation, not just
  the resolved base executable. No tools were downloaded.

## Actual provider intake and next exact proof gap

Pinned reference original Linux checkout: `/home/jeremy/Projects/merlin-graph-proof-intake`,
read-only detached revision `4499eebb959c0f92cefd92cc6d80646ba66474ed`.

Use the existing `merlin/tests/infra/test_multi_output_source.py` and
`test_multi_output_provider.py`. They have a suitable narrow actual route:
producer `P = input * weight0 + priorState`; first result `P + weight1`; second
result `P * weight1`. Source and selected/adapter scalar functions are independent
retained original buffers. Both outputs feed outside observers; the full original
source also returns state/position/input/weight/prior values. Do not claim those
unchanged computations have been operationally proved by this new layer.

Existing provider cases are `Case((3,47), 1, False)` and `Case((2,23), 2, True)`, where
width is bytes. The successful current intake covers ONLY the first case. The
source input-index rows are `((0,1,2),(-1,3),(-1,3))`. Source input order is input,
weight0, priorState, weight1. The original provider Kernel has three calls
`produce`, `add`, `multiply`; its producer scratch address is virtual 1024, so it is
**not an actual physically validated output Kernel**. The shared case uses private0/
private1 names for the same backing; comparing names alone is unsound.

Current source-only data is under
`out/artifacts/verification/graph-regions-declared/source-intake-v1/` and copied in
`evidence/`: complete source/retained MLIR, source/adapter/selected scalar buffers,
selected facts/software, original provider Kernel/bindings/objects, three baseline
per-stage Kernels/bindings, effects and structural value locators. Capturer invokes
no C/native execution or numeric reference; its empty trace is unused intake data.

Exact unfinished theorem: acceptance of complete actual scalar declarations and
actual call/role/window bindings, together with FINER per-instruction realization
contracts and explicit physical premises, must DERIVE
`GraphRegionsFusion.checked_native_contract` / `checkRealization_native`'s local
whole-region output agreement and surrounding observers. It must not assume that
same desired whole-region equality or define an implementation to equal the source.

After a NEW resume, the next bounded work is:

1. Instantiate `checkScalar` against separately authoritatively exported actual
   source/adapter/selected buffers, using the new serializer and static model hints.
   Preserve source function name `source` versus declaration `forward` explicitly.
2. Check the original generic scalar region and actual outer constant definitions,
   not only the production reader's extracted function. Recompute argument/SSA
   mapping, all properties, unread initializer and captures. Capture clones are
   inserted **at first use**, potentially interleaved with local operations, not
   necessarily as a prefix. Prove that closure relation under the common primitive
   interpretation; bind captures separately from body-local constants.
3. Implement a complete actual Kernel/role/window/shape/width/byte-order checker.
   Resolve private backing aliases; keep logical SSA IDs separate from physical
   addresses via an explicit buffer/value refinement map. Every source observer
   and retained surrounding call must be accounted for. Qualification IDs/flags
   are provenance, not proof facts.
4. Define an opaque actual machine step and per-instruction contracts: captured
   reads, value computation, commit/completion, no fault, complete effects/frame,
   runtime nonaliasing, private unobservability and suppressed-transfer policy.
   Derive the three-call producer/P/two-output relation by induction, then the old
   graph local contract and prefix/suffix observer theorem. Do not hide these
   premises inside a whole-region equality assumption or an ideal implementation.
5. Retain countermodels for alias, async/incomplete commit, private observers,
   missing no-fault and identity-only claims. No current-task countermodel exists yet.
6. Add actual second provider case and honest permitted capture/cast cases. The
   old public `test_multioutput_fusion.py` recipe contains min/max and must remain
   unsupported until explicitly covered. Do not broaden by silently discarding it.
7. Two sequential region realizations require an actual common state/ABI and exact
   middle. Otherwise retain a concrete refusal; two opaque IDs are insufficient.
8. Only in an explicitly verified unaffected LOCAL context, freeze a new fixed
   roster, run meaningful compiler-valid mutations/actual Lean refusals/witnesses,
   std3 audits, one original/fresh process batch, post-check and seal/paper row.
   A Linux snapshot does not qualify a laptop installation.

## Commands for a future explicitly authorized resume — NOT executed at stop

Source-only review/application from repository root:

```sh
git apply --check out/artifacts/verification/graph-regions-declared/laptop-handoff/current-work.patch
# The source-snapshot directory contains the exact same six files for inspection.
```

Historical Linux development entry points (require their recorded original files
and an independently verified unaffected local environment; not portable receipts):

```python
from pathlib import Path
from merlin.verify.lean_graph_regions_declared_context import select, build
selected = select(Path("out/artifacts/verification/graph-regions-declared/<new-context>"))
build(Path("out/build/graph-regions-declared/<new-build>"), selected,
      ["Merlin.GraphRegionsDeclaredEvaluation"])

from merlin.verify.lean_graph_regions_declared_capture import capture
capture(Path("/home/jeremy/Projects/merlin-graph-proof-intake"),
        Path("out/artifacts/verification/graph-regions-declared/<new-source-intake>"),
        shape=(2, 23), width=2, shared=True)
```

On a laptop, start with source-only inspection and explicitly select/document the
local toolchain/parser/provider dependencies under a NEW authorization. Do not
rewrite old absolute-path receipts or their failed expected hashes. Missing tools
or unmatched bindings are unavailable, not passing. No silent downloads.

## Failure evidence and holds

Current development logs are copied under `evidence/out/build/graph-regions-declared/`:

- `matcher-development-v1/Merlin.GraphRegionsDeclaredMatch.json`: initial pair-map
  simplification and constructor-pattern compile errors; repaired in `matcher-repair-v1.json`.
- `evaluation-v1.json`: unavailable Mathlib-style `List.Forall₂` and wrong map lemma
  syntax in the pure context. Replaced by a small local `TypedArguments` relation.
- `evaluation-v2.json`: induction argument and implicit type mismatch; repaired
  through explicit type binders in `evaluation-v3.json` (return code 0).

Prior source-bridge v1 remains UNQUALIFIED (45 pass / 4 fail, unavailable xDSL
constructor and lost virtual environment); corrected v2 completed 53 tests but has
the unresolved SDK binding. Prior pure graph profiles
`out/artifacts/verification/graph-regions/qualification-v{1,2,3}` remain unqualified:
v1/v2 SIGSEGV causes UNKNOWN; v3 metadata/context comparison failure; corrected v4
55-test packet preserved separately. No first failure was deleted.

A previous `check_lean_guard.py --help` unexpectedly ran its default shared Lake
build and crashed on empty audit output. That incident is recorded; it is not an
attribution for Basic.olean. Do not repeat that command. Full shared guard remains
unqualified; its read-only `--lock-only` result omitted pinned statement checking.
The foreign zephyr_model.py module-size failure was not edited. All old 2,719 packet
references were checked unchanged before the earlier partial handoff.

All old 31 sources, frozen definitions/specifications/SPEC_LOCK, production,
Foundation/StateEffects/ResourceGeometry, shared Main/CLI/registries and W/Q work
are read-only. Only standard3 axioms are permitted; no reflection/native_decide/
bv_decide/sorry/admit/new axiom/unsafe/extern/substituted implementations. No agents,
staging, commits, push, PR, broad integration or floating numerical work. Unapproved
contracts remain provisional. Native fidelity, review, full-model and shared-pipeline
claims remain false. The genuine STOP overrides all older resume plans.
