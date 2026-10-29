**PAUSED BY NEW USER STOP — 2026-10-24 06:12 UTC (Oct23 US/Pacific).**
The user returned and explicitly requested ALL agents stop, document progress and
prepare laptop handoffs. Normal stop messages were accepted by all three current
turns. Only safe checkpoint/documentation/packaging remains authorized now.
Do NOT resume proofs, builds, qualification or task assignment after compaction.
Older continuous-work instructions below are superseded until a NEW user resume.
Stop record: planner/eight-day-paper/USER-STOP-LAPTOP-20261024T061200Z.json.

# CURRENT: graph transformation verification first

User direction2026-10-23US/Pacific, after discussion with Agustin: prioritize graph
optimizations over numerical proofs. This supersedes the previous numerical assignments
and their queued continuations. Preserve their checkpoints; none becomes fully verified.
Eight-day scoped paper target remains2026-10-30; graph results now lead that artifact.
Agustin's priority guidance is not independent approval of a numerical/source specification.

Read the three immutable [handoff snapshots](intake/) and [source/tool intake](INTAKE.json).
Matching branch is `origin/feat/megaphase`, pinned4499eebb959c0f92cefd92cc6d80646ba66474ed.
Its764c09b82 change adds counted emission with bounded drained-iteration checking. This
currently checks EVERY dynamic iteration under a caller budget; it is not an affine
recurrence proof, universal SCF semantics or final-object correctness. External-output
coverage and renderer semantics are explicitly unproved in that checker.

## Final integration branch: feat/pass-verification

LATEST user clarification: workers MAY use their own branches; everything must end up on
`feat/pass-verification`. The primary delivery/integration checkout remains
`/home/jeremy/Projects/merlin`. Use an already prepared isolated graph worktree if useful,
or continue disjoint owned work in primary if already started. Avoid unnecessary relocation.
The earlier no-new-branch restriction is superseded by this explicit clarification.
Every worker must record its actual path/branch and provide an exact owned patch, dependency
closure and qualification record for integration into primary. Different source ancestry
must not cause a wholesale experimental-branch merge. Staging/commit/push holds persist;
reviewed new code can be collected in primary without publishing anything.

`merlin-graph-proof-intake` is a detached READ-ONLY4499 production-source reference.
The previously prepared optional worktrees are loops:merlin-graph-loops/feat/graph-loop-verification,
regions:merlin-graph-regions/pv/graph-region-verification, state:merlin-graph-effects/feat/graph-state-verification.
All began clean at4499. Coordinator copied ONLY the four unchanged shared proof/instruction
files in [shared proof intake](shared-proof-source-intake.json) into previously absent main
paths. No production/shared checker overwrite or old frozen semantics changed. Workers own
disjoint graph leaves; no shared interface edits without an agreed owner.

## Existing workers and exclusive owned tasks

| Existing worker | Graph task | Isolated checkout and branch | Assignment |
| --- | --- | --- | --- |
| W8A8 | Explicit memory↔SCF initializer bridge and actual reuse→segment composition | `/home/jeremy/Projects/merlin`, `feat/pass-verification`; NEW GraphLoopsMemorySCF leaves | [Current composition batch](../assignments/graph-memory-scf-composition/README.md); [sealed buffer-reuse batch](../assignments/graph-buffer-reuse/README.md); [sealed unrolling batch](../assignments/graph-ordered-unrolling/README.md) |
| Frontend/TorchAO | Actual scalar declaration/call correspondence and local fusion law from per-instruction contracts | `/home/jeremy/Projects/merlin`, `feat/pass-verification`; NEW GraphRegionsDeclared leaves | [Current declared-call task](../assignments/graph-declared-call-correspondence/README.md); [source bridge partial/unqualified](../assignments/graph-region-operational-bridge/README.md); [first region task](../assignments/graph-region-preservation/README.md) |
| Former Qinner/quant-ext | Actual CSE/DCE→LICM→unroll→segment in one operational semantics | `/home/jeremy/Projects/merlin`, `feat/pass-verification`; NEW GraphStatePipeline leaves | [Current scalar-loop pipeline task](../assignments/graph-scalar-loop-pipeline/README.md); [sealed scalar task](../assignments/graph-scalar-optimizations/README.md); [elision diagnostic-only](../assignments/graph-configuration-elision/README.md) |

The reference checkout is pinned4499. Coordinator intake is a separate
detached `/home/jeremy/Projects/merlin-graph-proof-intake`. Old worktrees/main remain intact.
No new agents are spawned. Send other workers only actual interface/dependency information.

## Shared proof and integration rules

State the semantic domain and soundness theorem first. Abstract arithmetic as an arbitrary
deterministic total pure function ONLY with explicit premises; effectful/partial behavior
must be modeled or refused. No float reassociation, numerical-law bypass or numerical
acceptance-gate removal. Existing numerical/runtime/hardware gates remain independent.

Bind actual source/result IR and selected semantic identity/tools to the projection. Check
complete changed boundaries, all live outputs, operand/result slots, effects and actual
order. Recompute facts instead of trusting candidate annotations or hints. Unsupported,
unavailable tools and unknown effects do not produce empty successful receipts.

Each worker owns additive GraphLoops/GraphRegions/GraphState Lean and Python leaf files,
separate tests and task artifacts. Existing Foundation/StateEffects/ResourceGeometry,
lean_checker/lean_contract, shared projection/registries, production transforms and all
frozen numerical semantics/spec/locks are read-only until an exact owner/interface request
is coordinated. Do not port or silently identify the old MerlinVerify semantics with the
new branch's Merlin proof framework. Reuse existing definitions read-only where applicable.

New claim roots permit ONLY propext/Classical.choice/Quot.sound. No sorry/admit/new axioms,
native_decide/bv_decide/native reflection or implementation substitutions. The new branch
documents an optionally admitted reflection mode; this task DOES NOT grant that permission.
Existing pinned Lean4.35 executable is identified for pure modules. Lean-MLIR integration
requires an explicit matching selected manifest; do not download or install dependencies
silently. If unavailable, continue kernel-proved structural modules and Python projection
work, reporting actual end-to-end Lean/toolchain checks as unavailable.

Use real source/transform examples, independent shapes/tails and valid changed-IR mutations.
Report Lean theorems, trusted Python correspondence, effect assumptions, tests, tool/runtime
trust, artifact binding, policy review and actual-model coverage separately. No LLVM-owned
pass proof, full compiler/model claim, target qualification or timing claim follows.

Finish/document a live old job safely before changing profiles; never rerun it to manufacture
a passing checkpoint. All counterexamples, failed builds and source holds remain preserved.
No staging, commits, broad integration, push/PR, main reset/switch/stash/prune or running-campaign
changes. Old frontend LOCAL-commit permission applies only to its old explicitly owned work;
new graph tracks remain uncommitted for coordination. Continue current graph tasks after
compaction until NEW user steering. This file supersedes old numerical continuation prompts.

## Paper alignment is required, not an optional appendix

Each task extends an EXISTING Merlin shared-pass proof/checker boundary. Read
[paper alignment](PAPER_ALIGNMENT.md) and name that existing root/domain before adding
abstractions. A toy graph theorem or Python well-formedness check is not a pass-correctness
claim. Deliver actual changed IR, checked source/model correspondence and the precise
composition improvement. The older MerlinVerify numerical/component evidence is retained
alongside these new-branch graph proof roots; their semantics are not silently identified.
Unproved bridges and deferred numerical/definedness gates remain explicit in the paper.
