# ACTIVE: fan-out region and fusion graph correspondence

Owner: existing frontend/TorchAO worker. User changed priority to GRAPH proofs after
discussion with Agustin. Safely document live qualification and preserve all old numerical
source/evidence, including unqualified outcomes. No new numerical law/primitive proof work.
Read [current shared scope/rules](../../graph-first-20261023/README.md) and its three handoffs.
Work `/home/jeremy/Projects/merlin`, branchfeat/pass-verification,
PRIMARY HEAD229f42c627854c49476a101cb393d0b15044fab8; read-only target source4499eebb959c0f92cefd92cc6d80646ba66474ed. Old frontend trees remain intact. No new implementation branches.

1. Inspect plan_connected_region, fusion_discovery.discover/_boundary, fusion_workflow.
   realize_candidate and verify_multioutput_fusion. Start with a real supported integer
   producer feeding TWO consumers while its intermediate is ALSO externally observed.
   Capture actual source/planner/realization results, not an independently invented graph.
2. State/prove graph-boundary preservation with arbitrary typed deterministic total pure
   node functions, complete operand/result-position correspondence and all external
   observers/live outputs. Explicitly model or refuse partial/effectful nodes; do not
   assume CSE/deletion is safe for arbitrary possibly failing operators. Preserve source
   order where purity/effects do not justify changes. No numerical operator proof needed.
3. Independently project actual SSA/source ordinals and slot identities into the checker;
   equal tensor shapes cannot replace SSA identity. Bind full region, fan-out, outside
   consumers, initializers and lifetimes. Recompute boundary facts instead of trusting
   the planner's proposed inputs/outputs. First prove plan boundary correspondence, then
   connect one admitted realized fusion under its explicit composition law.
4. Reject compiler-valid mutations dropping the external intermediate observer, losing
   a second consumer/output, swapping same-typed inputs/results or changing an untouched
   observer. Include source mutation/stale source/result/semantic/tool identities and a
   nontrivial positive at an independent shape. Unknown realization/composition laws
   stay unsupported; synchronous connected launches are not device/on-chip fusion.
5. Deliver Lean theorem+transitive axiom audit, actual before/after/projection, separate
   trusted Python checks and graph/realization assumptions, tests and task report. A
   well-formed graph alone is not semantic preservation or final-object correctness.

Own additive src/merlin/verify/lean/Merlin/GraphRegions*.lean, lean_graph_regions*.py and
matching new tests/artifacts. Existing connected/fusion production files and shared
verification registries/contracts/semantics stay read-only. Request an exact owned interface
change if needed. No new axioms/native reflection, staging/commit/push/integration/agents.
Continue this ACTIVE graph assignment after compaction; old numerical prompts are historical.

LATEST USER REQUIREMENTS: sole implementation branch feat/pass-verification; no new branch.
Read ../../graph-first-20261023/PAPER_ALIGNMENT.md. Your theorem/checker must extend its
named existing Merlin shared-pass roots, bind actual changed artifacts and expose remaining
composition/trust/definedness gates. Capture actual target-source artifacts using the4499
READ-ONLY intake checkout if that production API is absent from main; all NEW generator,
projection/checker/proof/test code stays in the primary checkout. Do not copy a dirty whole
branch or overwrite existing production/shared/frozen files to make an import work. Request
an exact dependency intake if required, and continue independent proof work meanwhile.

LATEST BRANCH CLARIFICATION supersedes earlier no-branch wording: worker branches ARE allowed,
but every owned implementation/dependency must be delivered to feat/pass-verification.
Choose the prepared graph worktree if useful, or keep already-started disjoint primary work;
do not churn locations. Record actual checkout/branch in ACTIVE. Deliver an exact owned
source patch and dependency inventory; do not merge the entire4499 experimental history
into primary. Publication/staging/commit holds and paper-alignment requirements remain.
