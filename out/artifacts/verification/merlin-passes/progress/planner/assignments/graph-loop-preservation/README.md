# ACTIVE: counted-loop and storage graph preservation

Owner: existing W8A8 worker. User changed priority to GRAPH proofs after discussion with
Agustin. Preserve numerical checkpoint and live-job result; no further numerical proofs.
Read [current shared scope/rules](../../graph-first-20261023/README.md) and its three handoffs.
Work `/home/jeremy/Projects/merlin`, branchfeat/pass-verification,
PRIMARY HEAD229f42c627854c49476a101cb393d0b15044fab8; read-only target source4499eebb959c0f92cefd92cc6d80646ba66474ed. Existing W8A8/frozen source remains untouched; new disjoint GraphLoops leaves go in this primary tree.

1. Inspect sched.ir Kernel/Loop/Call, periodic_effects.check_drained_period, dense_source,
   temporary_storage, examples/compiler/drained_period.py and test_drained_period.py.
   Distinguish compact counted IR, the complete ordered dynamic trace and emitted C.
   The existing Python checker enumerates ALL bounded iterations and changes no order;
   do not advertise that checker alone as a graph optimization or general induction proof.
2. State/prove the small actual counted-loop→complete-call-trace preservation relation,
   including actual index substitution, order, range/stride/width bounds and every iteration.
   Use abstract total state transitions with explicit memory/effect assumptions. Connect
   one real produced loop/scheduling pair to the model. Reuse orderedSegments/rangeSplit/
   tileTail/ResourceGeometry where the explicitly selected toolchain permits. Do not prove
   LLVM loop passes, arithmetic kernels or numerical recurrence.
3. Derive a checked loop/storage projection from actual Kernel and selected adapter data:
   full private initialization, no use-before-ready/overwrite, actual completion/drain and
   exact live ranges. Keep target adapter truth an explicit premise. Per-iteration drain
   does not imply external output nonaliasing; preserve/reject coverage failures explicitly.
4. Add a genuine tail/late-iteration address or overwrite mutation that passes a good
   prefix but is refused by the complete checker; include missing wait, final drain,
   index-width overflow and stale Kernel/effect/semantic identities. Preserve actual
   counted emission and compare full observable traces; renderer/machine semantics stay
   separate if not proved. No finite prefix may stand for the whole iteration domain.
5. Deliver kernel theorem+axiom audit, actual before/after/projection artifacts, meaningful
   positive/refusal tests and recoverable task report. Distinguish a bounded certificate
   from a parametric theorem and Python trust from Lean soundness.

Own additive src/merlin/verify/lean/Merlin/GraphLoops*.lean, lean_graph_loops*.py and
matching new tests/artifacts. Existing shared files are read-only; request exact interfaces
before edits. No production fix, axioms/reflection, staging/commit/push/integration/new agents.
If proof-tool integration is unavailable, keep independent structural work moving and report
that gate as unavailable. Continue this ACTIVE graph assignment after compaction.

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
