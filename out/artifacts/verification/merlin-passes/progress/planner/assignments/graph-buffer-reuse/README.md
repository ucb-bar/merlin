# ACTIVE large batch: actual allocation reuse preserves live memory observations

Owner: existing W8A8/GraphLoops worker. Ordered unrolling and its actual same-SCF
composition are sealed. Preserve all35 owned source bytes, source/runtime/evidence
seals and numerical checkpoints. Continue this ENTIRE graph batch autonomously
until a NEW genuine user stop; do not resume old numerical tasks after compaction.

Work in PRIMARY `/home/jeremy/Projects/merlin`, `feat/pass-verification`. Worker
branches are permitted, with final code/dependencies in primary. Own ONLY NEW
`GraphLoopsReuse*.lean`, `lean_graph_loops_reuse*.py`, matching new tests and
private graph-buffer-reuse build/artifact roots. All old35 sources and shared
StateEffects/ResourceGeometry/Foundation, production, registries/Main/CLI,
frozen specs/locks and other workers' files are read-only. No staging, commits,
push/PR, broad merge/cleanup, downloads, new agents or numerical-law work.

## Actual existing pass and proof gap

Read pinned4499 `src/merlin/transforms/planning.py`: `plan_storage`,
`_full_write_initializer`, `reuse_storage`, and its focused tests, witnesses and
actual structural validator. `buffer_reuse` is an existing Merlin shared pass.
The common Lean checker currently proves interval disjointness using
`Foundation.disjointLifetime`; ResourceGeometry proves physical range relations.
Extend this actual boundary from lifetime arithmetic to complete live-memory/
observation preservation under the allocation-changing rewrite. Your old Kernel
storage/completion proof is useful background, not an implicit xDSL/Kernel bridge.

The producer reuses IDENTICAL memref types/alignment, requires explicit disjoint
lifetimes and complete rectangular initialization before reads, rejects escapes/
unknown aliases, and removes earlier deallocation/later allocation while mapping
later uses to the first buffer. None of these facts alone proves value preservation.

## Full deliverable

1. State the semantic theorem first. Model fresh private allocation, load/store/
   dimension/free and complete caller/public observations. Relate distinct
   logical objects/epochs to one physical slot only while their lifetimes are
   active. Preserve all live values, returned scalars, caller memory and retained
   effect order, ignoring only proved-unobservable dead/private heap cells. Do
   not assert all independently named heap/SSA cells are equal. Prove that no
   later epoch observes stale prior contents and that final frees/uses are valid.
2. Treat allocation failure, allocator observability, pointer identity and
   resource exhaustion honestly. Exact success equivalence may require an
   explicit total fresh-allocation model; otherwise state the supported
   refinement direction/domain. Do not silently remove failure/poison or assume
   source success. Retained partial/effectful operations must retain their
   possible outcomes under the proved live-state relation. Hardware allocation,
   async behavior, unknown memory operations and native ABI are separate gates.
3. Independently derive complete before/after matching from BOTH actual modules:
   allocation objects/types/shapes/layout/alignment, every use and owner scope,
   direct deallocation, access ancestry, full initializer loops/indices/stores,
   alias/escape absence, actual logical-to-slot map, all retained operations and
   outside observers/functions. Do not trust StoragePlan/BufferReuseWitness or
   just reproduce the producer and compare its text. Python parsing/projection
   may remain trusted, explicitly recorded. Unsupported cases must refuse.
4. Prove the complete initializer establishes the needed memory relation, with
   every coordinate covered before reads. Start with a precise static positive
   contiguous byte-addressable integer memref domain, synchronous direct uses
   and explicit lifetimes. Use parameterized geometry/loops where possible; do
   not enumerate huge tensor shapes as the proof. Preserve source partiality or
   state the admitted initializer semantics; no hidden float/arithmetic laws.
   Reuse old definitions read-only only with an explicit semantics relation.
   The frozen MerlinVerify evaluator has no general SCF execution route: do not
   fabricate such a bridge or change frozen semantics to get an acceptance.
5. Capture actual changed producer examples via pinned4499 APIs in isolated
   subprocesses, checking source/parser identities. Include >=2 reused epochs,
   three objects/chained reuse, independent shapes/ranks/byte widths/alignment,
   intervening retained calls/caller observations, complete initialized nonzero
   model witnesses and a large-shape compact structural case. Identity/declined
   cases are separate from changed positives. Independently execute small
   logical-memory examples if useful, labeled tests rather than native proof.
6. Use VALID-IR adversaries for overlapping lifetimes/use-after-free, incomplete
   or wrong-index initialization, read before reinitialization, wrong value or
   wrong same-typed buffer, alias/escape/view, changed alignment/type/layout,
   missing/wrong final free, retained-call reordering, altered caller observation
   or untouched function. Distinguish malformed IR and refused unsupported
   semantics. Retain stale/coherently rehashed source/model/tool/proof/import/
   audit substitutions and writer guards. Keep necessary-premise countermodels.
7. If a REAL neighboring transformation is supported in the SAME explicitly
   related memory/state semantics, add one exact-middle composed case and prove
   transitivity. Do not equate allocation-changing states with GraphStateChain's
   fixed physical-cell renaming. Unsupported composition must remain a concrete
   refusal/gap. Standalone actual buffer-reuse preservation is the main result.
8. Freeze one NEW source/profile and positive/refusal roster. Run meaningful
   tests, exact-root/transitive standard3 audits, originals and ONE fresh-process
   replay batch; retain successful queries/projections/CHECKs explicitly with
   retention_policy=all, post-check identities, then seal. Preserve first
   failures; stop a newly crashing context instead of retrying to promote it.
   Deliver source-only patch, exact prior/common/import/tool/parser closure,
   reproduction recipe and a paper row: existing pass/root, full relation,
   domain/inhabitation, actual changed boundary, trusted projection/allocator/
   primitive premises, composition or gap, tests/audits/replays/review and actual
   retained-model/native/hardware coverage separately.

Only propext/Classical.choice/Quot.sound; no sorry/admit/new axioms/native_decide/
bv_decide/reflection/unsafe/extern/substituted implementations. Use existing
pinned pure Lean4.35 for new independent work; full LeanMLIR remains unavailable,
with no downloads. No numerical gate or counterexample is removed. Broad
verification/policy/native/full-model flags remain false until actually justified.
Do not contact other workers absent a concrete shared dependency. Read this ACTIVE
assignment after compaction; report one full sealed batch or a true blocker
concisely and keep independent progress going. User requests large tasks to
minimize conversation and token use.
