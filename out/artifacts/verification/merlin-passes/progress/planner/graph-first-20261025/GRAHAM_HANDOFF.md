# Graham: independent graph-correctness review and paper evidence

The [2026-10-27 pass-status audit](../../../../graham-graph-review/PASS_STATUS_20261027.md)
summarizes working subsets, fresh failures, wider pass gaps and compiler-gate
readiness against graph baseline **c62f81a8a**. The separate dtype-binding fix
is published upstream at **68f364608**, with 48 focused regressions passing.
Upstream and audit branches were pulled separately and remain separate. This
status refresh adds no new Lean execution or qualification and resumes no
implementation workers.

## Latest independent execution — 2026-10-26 Pacific

The user authorized three workers to reconcile sources, select a fresh Lean SDK
and reproduce the fixed graph candidates. All workers finished. Read
[the current review index](../../../../graham-graph-review/README.md) and
[the a26 reproduction report](../../../../graham-graph-review/execution-a26-20261026-v1/README.md).
The tested source baseline is **c62f81a8a11347ff53b60101fa75280e69d002a6** on
`feat/pass-verification`; the review branch retains its separate history.

Ten changed admission sources were reviewed and 32/32 independent parser/helper
controls passed. Fresh official Lean 4.35.0 passed eight setup diagnostics.
Three schedule queries (two changed, one identity) and the empty-inline query
compiled and passed 76 requested root-audit occurrences using only the standard
three axioms. Four nonempty inline queries and all five inline witnesses failed.
Tail stops at its arity proof, and both W profiles stop at the shared renaming
proof; 30 dependent case queries remain unexecuted.

A separate unchanged-statement tail lemma repair compiles, but no frozen
candidate was modified or requalified. Deliverables 1–3 have linked supplements
in the review index. Next work is proof successors, dependent query completion,
fresh semantic qualification, human review and actual compiler integration.
Old runtime holds and historical source/evidence identities remain in force.
The assignment and older dated status below are historical context.

This is a substantial parallel workstream, not a request to rerun one example.
Work through all six packages below; expect several sustained sessions, with
progress recorded between them. The goal is to resolve the paper's two open
verification items: an exact supported correctness claim, and a worked actual
IR example with checker output and an independently reviewed mapping to Lean.

Numerical proofs are deferred. Do not broaden a graph claim to numerical,
LLVM/backend, encoding, hardware or timing correctness. Work independently of
the implementation authors; do not approve an assumption just because their
checker accepts. A finding that limits the claim is a useful result.

## Setup, inputs and ownership

Primary repository and delivery branch: `feat/pass-verification`. Pull the
published branch first; record the exact commit and dirty source identities.
The actual graph producer/reference is separately pinned at
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. Do not merge that experimental branch
wholesale into the verification branch. The three implementation agents own:

- `GraphLoopsShape*` / `lean_graph_loops_shape*`: whole-boundary shape/view
  normalization and composition with sealed memory/SCF roots. The preceding
  GraphLoopsMemorySCF batch is sealed and now ready for independent review.
- `GraphRegionsDeclared*` / `lean_graph_regions_declared*`: declaration/call
  correspondence and local fusion agreement.
- `GraphStateSchedule*` / `lean_graph_state_schedule*`: complete sequential
  dependency scheduling and actual composition with the now-sealed integer/
  scalar/loop chain, preserving failures and terminal worlds. Prior pipeline
  and integer sources/evidence are immutable review inputs.

Do not modify those sources or their sealed tests/evidence. Own a separate
`out/artifacts/verification/graham-graph-review/` tree, independent reviewer
scripts and newly justified adversarial cases. Propose fixes and interface
changes to the coordinator with a reproduction; do not repair the production
pass or frozen semantics as part of the review. Keep final changes on the
primary branch. Publication is coordinated; do not force-push or open a PR.

Read repository AGENT.md/CLAUDE.md, relevant directory instructions,
`docs/design/merlin_lean_verification.md`, this directory's README/PAPER_TARGET,
and the graph-first-20261023 PAPER_ALIGNMENT. Existing specifications, locks,
qualified source capsules and counterexamples remain read-only.

The accompanying `worked-example/` contains exact copies of a qualified
CSE/DCE→LICM chain, its projection, query, original checker/audit output and
fresh-process replay. Read `worked-example/README.md` and its manifest first.
The original receipt paths remain historical Linux paths; do not rewrite them
and refresh hashes to make a laptop replay appear original. Full source/dependency
handoffs are also in the previously supplied laptop bundle. Compiled toolchains,
large build trees and native core dumps are deliberately not portable inputs.

## 1. Establish the graph-pass denominator and claim matrix

Independently enumerate actual registered transformations at the pinned source:
`transforms/certificates.py:TRANSFORM_SCOPES`, `transforms/pipeline.py`,
`transforms/pipeline_policy.py` and `compile/scheduling/schedule_pipeline.py`.
The common pipeline has eight transform kinds; analysis/emission stages are
not additional proved graph rewrites. Inventory broader scheduling/fusion/
transfer/storage mechanisms separately so they do not vanish from an all-graph
claim. Distinguish helpers, pass selectors, actual transformations and target
implementations. Compare with `docs/reference/shared_compiler_passes.md`.

For every kind and implemented variant, map actual source → validator entry →
projection → Lean theorem → relation/domain → assumptions → changed positive
artifact → refusals → qualification/replay → review state. Verify the cited
theorem covers the whole changed boundary, not just integer arithmetic, lifetime
disjointness or a local loop identity. Mark unsupported, unqualified and missing
composition separately; do not invent a percentage or count tests as proofs.

Deliver `coverage.csv`, `coverage-review.md` and a prioritized gap list. Check the
normalization and dependency-scheduling routes carefully: the old host broker's
local certificate results are not automatically the newer whole-program proofs.

## 2. Independently audit complete IR-to-Lean correspondence

Start with the supplied nonempty CSE/DCE→LICM chain, then the retained zero-trip
and nested chain variants. Walk the unabridged before/middle/after IR against
`actual/projection.json` and `checked/kernel/GraphStateScalarInstance.lean`.
Review the actual projection and serializer source, not just its JSON labels.

Account for all operations, integer widths/properties, ordered operands/results,
entry signatures, declarations/attributes, captures, scoped SSA, IVs, carries,
yields, return values, persistent effects, observers and untouched functions.
Check that all three stages share a justified descriptor/type/control registry
and the middle is the same actual buffer. Identify every trusted host/parser
step and every corresponding Lean check. Hash equality binds bytes; it does not
prove projection fidelity or semantic injectivity.

Create an independent correspondence table or small read-only structural audit
script. It must not import the implementation's own projection as its oracle.
Examine what unsupported metadata/regions/local calls do: refusal must cover
the entire boundary rather than silently dropping unmodeled operations.
After the active agents deliver their new packets, extend this audit to the
actual multi-initializer reuse→segment case and declaration/physical-call case.

Deliver `ir-map.md`, independent audit source/output and exact omissions/bugs.

## 3. Review the semantic assumptions and theorem strength

Read each final theorem statement and actual generated query with its full
transitive axiom audit. Only propext/Classical.choice/Quot.sound are allowed;
no sorry/admit/new axioms/native_decide/bv_decide/native reflection or substituted
implementation. A generic theorem with hypotheses is not an axiom violation,
but its hypotheses must be stated and justified for the claimed use.

Audit total deterministic world-free pure operations, partial opaque results
and terminal worlds, result arity, loop/index/control meaning, scope, aliases,
read-before-write, completion, private observations and allocation. Distinguish
returned-value equality, complete observer equality, possible successful
outcomes, failure preservation, typing, initialized slots, success and poison
freedom. Different workers currently prove different relations; do not equate
them implicitly when composing. Check domain inhabitation and nonzero witnesses.

Test necessity against retained countermodels. In particular, source allocation
can peak at 8 bytes while the reuse candidate reserves 16; a finite allocator
distinguishes them. Total/private/unobservable allocation is a real premise.
The scalar loop control premise and the new operational control bridge must
not be confused. A local fusion contract must not restate final equality.

Deliver `assumption-review.md` with each premise classified as proved, reviewed
contract, trusted implementation correspondence, experimentally tested, or open.
Record reviewer/date/source identities and unresolved questions; independent
review does not self-approve numerical policies or native fidelity.

## 4. Challenge acceptance with independent valid mutations

Inspect existing negative coverage first, then add missing compiler-valid
mutations that challenge the theorem/IR association: same-typed wrong operand,
changed observer/effect, wrong carry/capture, skipped initializer, altered loop
control, wrong middle, descriptor/width/registry drift, missing output and
coherently rehashed source/query/audit associations. Check malformed-input
rejection separately from valid-but-wrong transformation refusal.

Produce at least one independently reasoned semantic counterexample to a
mutation, with complete public outcome including world/error when applicable.
Do not call a native crash, timeout, setup error or unavailable tool a successful
refusal. Do not duplicate existing cases merely to increase the count. Retain
the first raw failure and report the actual rejected fact.

Deliver `adversarial-review.md`, complete original/mutated IR, independent
reasoning or witness, parser-validity evidence and raw checker outcomes.

## 5. Reproduce qualification and audit the host acceptance boundary

Use an explicitly selected pinned environment and a NEW output directory. The
supplied reproduction recipe gives the scalar receipt CLI and original buffers.
Independently check actual sources, kernel/tool imports/sidecars/libraries and
the exact audited roots before and after replay. Run one reviewed original-bound
replay for the worked example and meaningful independent refusal cases; retain
successful artifacts. Distinguish a fresh process on the same machine from a
clean environment on another machine. On a laptop, identify a fresh toolchain
and source build rather than treating old Linux objects as portable proof.

Audit whether the actual selected compiler host requires this validator's live
completion before proceeding. New independent graph leaf validators are not
automatically wired into `verify_pipeline_change`/`accepted_pipeline_receipt`.
Record exact gate/route and any missing broker integration. Saved receipt JSON,
identity flags and passing tests must not grant live proof authority.

Preserve the unexplained Std.Time.Format.Basic object discrepancy and separate
configuration-elision SIGSEGV. The affected source bridge/elision contexts remain
unqualified. Do not repair/reselect/retry/rehash them to get green. Use only a
verified unaffected selection; otherwise finish source review and report the
replay as unavailable. No silent dependency/toolchain download or modification.

Deliver `reproduction.md`, commands, versions, raw stdout/stderr/status, file
inventory, exact reviewed roots, acceptance-gate audit and remaining trust.

## 6. Produce the paper-ready review packet

Return a concise `REVIEW.md` with an explicit verdict per claimed row, all open
obligations and the strongest sentence the evidence supports. Supply a compact
annotated actual before/after excerpt with the exact retained checker result;
link unabridged buffers/query/audit/replay, and keep all conditional flags visible.
Also provide proposed replacement text for both verification FIXMEs and an
appendix table covering semantics, trusted mapping and assumptions.

The user's two-buffer panel-pipelining/gSIM figure has separate execution and
performance measurements. Do not attach a Lean claim to that exact change unless
its full serial→pipeline boundary and completion/effects are actually proved.
Explain the connection and limits rather than merging execution checks with a
compiler-IR proof. Do not repeat measurements or claim unverified speedups.

Conclude with a prioritized release checklist: issues that block the scoped
graph theorem, those that limit an all-graph claim, independent specification
approvals, host integration and final artifact/model coverage. Send substantive
findings to the coordinator; do not wait for all six packages to expose a soundness
bug. After compaction, reread this current task and continue the next package.

## Immediate start and completion

Begin packages 1–3 against the supplied already-qualified example. These do not
depend on the active agents finishing. Then challenge/reproduce it while reviewing
new agent packets as they arrive. This sequence provides substantial parallel
work even when a tool or implementation bridge is blocked.

Complete when the six deliverables are reviewable and every proposed paper claim
has an evidence link and explicit remaining assumptions. Do not mark the complete
shared pipeline, numerics or full model verified merely to close the paper FIXMEs.

## New qualified row ready for review

The memory-SCF batch is now sealed under
`out/artifacts/verification/graph-memory-scf/memory-scf-mode-v1/README.md`, with
its own `paper-evidence/{README,WORKED_EXAMPLE}.md` and retained checker output.
It closes the exact TWO-initializer reuse→segment join for its admitted domain:
71 tests plus one71-test replay,61/63 standard-three roots,3 provider tests.
The coordinator separately matched1061 evidence files and all72 owned/prior
source identities, then independently accepted the actual positive triple and
refused a compiler-valid tail mutation. This is engineering evidence, not your
independent review. Prioritize this new row in packages2–3 after the scalar
worked example. Its possible COMPLETE public-outcome relation excludes tagged
failure/divergence equivalence and retains trusted parser/projection and total
private/unobservable allocation premises. Compiler gating remains unintegrated.

## Frontend review inputs and a new unresolved gate

The declared-call worker's qualification-v1 has46 original cases and one fresh
process replay batch over two actual typed boundaries. Its exact19 sources,
sealed reports and current paper-evidence packet are available for review. A
NEW five-source stronger wrapper checks ABI window order/full-after-effects,
but its frozen run is UNQUALIFIED: the first i8 host-reference check failed
before any oracle invocation; i16 passed and dependent i8 cases were not retried.
The stale file identity/cause is UNKNOWN, and one bounded current comparison
found no mismatches. Keep these qualifications separate. The coordinator's
frontend-declared-observation-v1.json records a source/evidence-only inspection.

Packages3 and5 should examine this gate and the exact host trust boundary.
A later matching hash is not an explanation or an approval. Do not rerun/rehash
the failed profile to promote it. An independently identified environment can
provide separately attributed reproduction; preserve the original failure. F is
continuing source-only captured-literal/closure proof work.

## Qualified shape row and pure-development context now available

The shape worker's standalone-mode-v1 is sealed:22 new sources,78 originals and
one78-test replay,35 main/41 witness standard-three roots. Its paper-evidence
includes complete actual before/after IR, checker output and a wrong-buffer
counterexample. The coordinator matched1001 evidence files and94 current/prior
source identities. Audit this new registered shape_view_normalization row in
packages1–3. The typed/full-heap→old epoch/public adapter is still UNPROVED;
COMPOSITION_REFUSAL.json preserves that actual join. New adapter source is outside
this qualification. Descriptor/primitive truth, mapping review and compiler gating
remain separate.

For frontend proof development only, the coordinator also retained a separate
original-bound minimal Init diagnostic (zero axioms), startup/late-library traces
and SDK/runtime pre/post checks. frontend-pure-development-context/README.md
states its exact trust and scope. The held MLIR-full oracle and stronger failed
wrapper are not reopened by it. This narrower context enables checking candidate
mathematical sources; it is not actual capture qualification or an OS-loader proof.

## Four-pass scalar/loop chain is ready for independent review

`graph-state-pipeline/review/sealed-v1` now supplies a complete actual five-buffer
CSE/DCE→LICM→unroll→segment example and error/world relation,64 tests and8 fresh
positive replays. The coordinator matched4001 retained files and20 source pairs.
Review its final exact theorem types, no-tail/static domain and concrete control
model rather than inferring full arbitrary-SCF coverage. Exact-integer whole-
program normalization is Q's new nonoverlapping task. W continues the unproved
shape-to-epoch adapter; F now checks captured-literal proof candidates in the
separately documented original-bound pure development context. The old F oracle/
failed-wrapper holds remain. Update packages1–3 and the paper claim matrix using
these sealed rows; do not sum overlapping tests/theorems or grant full coverage.

## Candidate manuscript text and latest development-context limit

Review [PAPER_VERIFICATION_DRAFT.md](PAPER_VERIFICATION_DRAFT.md) as proposed text
for both FIXMEs, with the sealed four-pass example as its evidence. Check each
observation relation and admitted variant against the exact roots; the draft is
not approved merely because it is published.

Frontend's later whole-SDK pure-development-v9 stopped at a libclang postcheck
digest mismatch after its third source-header command returned0. Its cause is
UNKNOWN; the suspect file was not reread and v9 stays PARTIAL/UNQUALIFIED. The
separate frontend-original-module-runtime-context selects all five original
files for every653 Init module plus11 original runtime refs (3276 pins), using
retained metadata only. All1959 compiled Init refs also match the pre-existing
W context;18 auxiliary/alternative-runtime files are outside that dependency
rule. Offline membership tests precede NEW guarded development. Review this
selection, actual imports/runtime reporting and explicit host trust separately
from the mathematics. It does not promote any old failed profile or qualify a
captured graph; do not rerun held profiles as a review shortcut.

## Exact-integer v2 is ready for review; scheduling is the current task

`graph-state-integer/review/sealed-v2` now supplies20 new sources,106 dependency
snapshots,75 tests,17 original receipts(14 changed/3 identity) and one fresh replay
batch17. The coordinator matched all2250 retained files and20 source pairs.
Its `complete` theorem uses canonical word entries and local successful opaque
result typing; runtime typing of reached intermediates is derived. Check those
contracts, the noncanonical/provider/hidden-effect countermodels, actual six-buffer
joins and the trusted full IR mapping in packages1–3. Native, poison and broker
fidelity remain separate. V1 is an immutable unqualified host-guard diagnostic.

The coordinator also recorded a concrete STATIC v2 host-status limitation:
capture's broad ValueError handling can label stale source/runtime identities as
unsupported. No real SDK fault was created and no false Lean acceptance was
demonstrated. Review integer-v2-host-status-limitation.json; do not infer universal
error-classification correctness from the75 selected tests. The new scheduler
adapter owns stricter exception separation; all95 old owned sources stay frozen.

W's new actual shape→reuse→segment semantic adapter now checks201 std3 roots in
development; its checker/witness/adversarial/frozen qualification remains active.
It does not amend the earlier standalone seal or its historical composition
refusal. F's latest exact optional-parser-symbol review is separately retained at
frontend-finite-symbol-review-v1; new source development continues, without old
profile promotion or actual-capture qualification.
