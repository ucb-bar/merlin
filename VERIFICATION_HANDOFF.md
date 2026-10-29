# Current compiler verification checkpoint — 2026-10-29

The latest user policy keeps ordinary compiler transformations for original
sources outside an explicit Lean profile and logs those boundaries as unverified
at invocation completion. Admitted source boundaries still block on unsuccessful
proofs, unsupported targets, missing tools or custody failures. This supersedes
the proposed skip/no-op policy. See the [bounded profile catalogue](docs/guides/lean_bounded_pass_profiles.md).
End coverage is per boundary, finalized after worker join; it grants no proof of
an unverified suffix or downstream LLVM/native behavior. All ten selectors are
integrated within their declared profiles; five are selected by `all`. The
[stack guide](docs/guides/lean_compiler_stack.md) and
[offline stack](out/artifacts/handoff/pass-verif-integration-20261028/merlin-lean-stack.html)
show each location, interface, example and theorem relation. Completed gates are
integrated independently of future model extensions.

The current [handoff](out/artifacts/handoff/pass-verif-integration-20261028/README.md)
separates proof, replay, protocol and finite native evidence at their exact cuts.
Generic UNROLL/TILE have four retained fresh proofs and independent audit;
configuration has a retained three-case/four-query refresh. The compiler matrix's
11 passes and stale host-test injection failure are recorded; the repaired
actual host negative and three source-exclusion cases subsequently pass with
independent artifact review. The latest source-policy cohort passes 37 host
controls, followed by five actual source-admission cases with independent audit.
The final default-all memory refresh passes both untiled/tiled cases: four fresh
accepted workers, 24 finite native observations and six untouched-entry checks.
All 81 guarded source hashes remain unchanged; each case records two verified
boundaries, zero unverified boundaries and an unchanged CSE with zero proof credit.

Broader generic bodies/carries/nesting, computed initialization, rank-three
storage and changed memory CSE are explicit future coverage extensions. They do
not become assumptions of accepted proofs. Q/DQ remains proposed-model and MACC
remains draft pending independent numerical review. The profile catalogue lists
primitive/effect, projection and logical-allocation premises for review.

The [merge qualification](out/artifacts/audits/ledger-review-20261029/merge-README.md)
combines the ledger/fusion corrections with Graham's published memory-gate
changes through `df8799c42`. Its evidence binds the combined sources; the older
records below retain their original source cuts.

The [ledger review](out/artifacts/audits/ledger-review-20261029/README.md) corrects
the numerical and projection assumptions without changing frozen statements or
approvals. An additive MACC fusion theorem now works for noninjective extensions;
T7/A10 name the existing Python projection and selected-effect premises. F12 is
marked compiler-corrected while per-channel i64 formal source support stays out
of scope. Fresh Q/DQ, MACC and host-fusion replays bind the revised ledger; other
historical receipts retain their original cuts.

Host graph fusion now has an explicit compiler gate on both xDSL and textual
routes. It checks actual emitted helper bodies, caller bindings and every public
output under the existing typed total pure tensor-action model. All 20 new
qualification cases pass without skips, including fresh-process replay, five
actual target mutations, failure quarantine and native reference comparisons.
All 71 existing graph regressions pass after two stale mocks were corrected.
See the [reproducible graph handoff](out/artifacts/audits/host-graph-fusion-20261029/README.md)
for exact sources, theorem mapping, preserved failures and paper wording.
This is logical dispatch grouping; primitive correspondence, successful logical
allocation and downstream/native fidelity remain explicit boundaries.

Numerical review work based on `4d28efe5c` keeps f32 accumulation as the selected
MACC contract. Additive Q/DQ conversion-domain and initialized/mixed-format MACC
component proofs, plus the plain-reduction producer/report regression fix, are
described in the [review packet](docs/design/qdq_macc_numerical_contract_review.md).
They do not install source/policy approval or broaden compiler eligibility to
nonzero/mixed-format source contractions. Exact tested cuts, failures and replay
commands are retained under `out/artifacts/audits/overnight-pass-verification-20261029/`.
The [completed audit handoff](out/artifacts/audits/overnight-pass-verification-20261029/README.md)
records commits `ee99e64fb` and `9a92a929f`: 143 MACC cases pass with two explicit
retained-artifact skips, 71 Q/DQ cases pass, and 10 final regressions pass.
All six lowering receipts replay; all 235 pinned statements remain unchanged.

Use **`feat/pass-verif-integration`**. Retained deeper-body loop proofs are
committed in `ef46cc269`; explicit C64 wide-count hoisting is `91661f328`.
Ten families have scoped compiler selectors. Five belong to `all`; proposed
Q/DQ, draft MACC, ISA-specific configuration and host graph gates remain explicit.

Default `all` now composes the actual private-memory route through later DCE
and an authentic unchanged native CSE observation. Installed untiled and
tile-size-32 tail cases both pass: four accepted workers, 24 finite native
kernel observations and six untouched-entry checks. The memory DCE theorem
equates complete successful returned-value/world relations; it adds no
categorized-error, allocator or downstream fidelity proof. Unchanged CSE
submits zero jobs and receives zero proof credit; changed memory CSE now runs
with explicit unverified coverage under the bounded source-admission policy.
Private-buffer tile resource policy now permits sizes through 65,536 while
retaining signed-index64 safety checks. Evidence and earlier safe refusals are
under `out/build/default-all-memory-20261029/`; `installed-qualification-v3/`
is the completed historical compiler run before the current admission/controller
changes. The final retention-enabled refresh is tracked in the current handoff.

Selected inner-depth unrolling and tiling now preserve supported deeper bodies.
An actual depth-two unroll→tile gate preserves a dynamic depth-three body;
compiled boundary/wrap cases, fresh independent-process replay and two genuine
Lean callee/stride mutation refusals pass. Auxiliary wrappers may capture a
stride only as an unchanged block argument; actual source controls retain their
existing strict positive/static admission. The final roots have no inherited
control or source-success premise.

The explicit original `ConfigurationCConsumer(index_bits=64)` binds actual
controlled C emission, selected contracts and exact source/proof products.
It admits positive counts through 2^63−1 without changing V1–V4 defaults.
129 targeted cases pass, including symbolic wide boundaries, fresh MAX63 replay,
one genuine Lean count refusal and 15 small native state/failure observations.
All-index `Models`, complete-state and selected ISA-effect premises remain.

Recursive positive counted-Kernel hoisting is now qualified through its
original compiler and C64 consumer: 78 installed pytest cases, five separate
legacy reconstruction checks, fresh replay and 27 finite native state/failure
observations pass. Commits `30ec04967` and `0990c2806` add the proof and gate.
All-environment instruction/effect Models remains an explicit premise.

**The bounded stopping milestone is finishing qualification and publication.**
Generic-value UNROLL has an installed compiler worker and fresh replay: 41 roots
and 940 audited declarations per check, standard logical axioms only. Generic
TILE now also passes actual compiler delivery and fresh replay, with 47 roots
and 929 audited declarations per tree. All four final proof trees bind 91
current sources without drift. Computed initializer prefixes,
broader memory ranks and more generic loop bodies are documented future domain
extensions, not prerequisites for the bounded gates. Rank-one buffer source/target bridges are committed in `5f9fb92de`.
Their installed qualification now passes all 33 distinct cases: 12 host and 21
SDK/native cases, including both public compiler routes, fresh-process replay,
actual target mutations and initializer-link component refusals. Q/DQ stays proposed and MACC stays
draft pending independent review. No protected Core/Sem/Spec definitions or locks changed.

Q/DQ lowering and bf16/f16 MACC are now the colleague's workstream. This lane
continues the remaining MLIR/xDSL pass proofs and compiler gates, with one
independent reviewer. LLVM formal verification is outside the agreed scope.

Start at the [current evidence and remaining work](out/artifacts/handoff/pass-verif-integration-20261028/README.md),
[stack/interfaces/examples](docs/guides/lean_compiler_stack.md),
[completion scope](docs/design/lean_pass_integration_completion.md) and
[assumption ledger](merlin/lean/LEDGER.md). Remote publication and both GitHub
identities were verified through `df8799c42`, including the default-all memory
gates and strengthened claim documentation. Colleague numerical changes were fast-forwarded
through merge `c95fb1012`, preserving Jeremy Mills's authorship.
Earlier evidence retains its
recorded source cut; changing bound sources requires a fresh check.

The [unpublished commit map](out/artifacts/handoff/pass-verif-integration-20261028/unpublished-commit-map.json)
records a prepublication message-length correction with unchanged code trees,
identities and dates.

---

# Historical broader completion plan and loop generalization — 2026-10-29

Continue on **`feat/pass-verif-integration`**. Code `0fd1b0294` removes separate
compiler argument/carry caps (including zero carries) and scalar unroll-factor /
tile-size upper bounds. Existing standalone and private-buffer profiles retain
their declared domains and resource guards remain enforced. The colleague's
audit-only object freshness fix and handoff through `a829c1a66` are integrated.

**59 targeted checks passed on the merged cut, zero failures or skips**, plus
fresh-process replay of factor seven / tile seventeen with saved approval fields
omitted. The larger example has 126 iterations, including a short final tile.
Earlier 32-case and interrupted 51-pass runs retain their own cuts and overlap;
they are not additional current qualification. No full legacy suite was run.

The [completion plan](docs/design/lean_pass_integration_completion.md) defines
the agreed endpoint through Merlin output IR, work order and acceptance criteria.
**The human-review-only milestone is not yet complete.** Loop inline/tail/context
bridges, direct untiled reuse, semantic joins, numerical producer/source coverage
and real configuration consumers remain engineering work. Q/DQ source approval
and MACC independent numerical review remain pending; real SourceCapture
qualification still needs the previously deferred pinned tools.

See the [current checkpoint](out/artifacts/handoff/pass-verif-integration-20261028/README.md),
[updated compiler stack](docs/guides/lean_compiler_stack.md) and
[assumption ledger](merlin/lean/LEDGER.md). Earlier domain summaries and results
below retain their stated source cuts.

---

# MACC freshness fix and remaining integration — 2026-10-29

On `feat/pass-verif-integration`, `ae144d959` makes checker admission require
current audit-only proof objects as well as both executables. The
[handoff for the next owner](out/artifacts/handoff/macc-pass-integration-20261029/README.md)
records the reproduced defect, regression, current pass inventory and remaining
configuration/scalar integration work. On the combined Q/DQ/MACC cut, 119
MACC-related cases and 22 Q/DQ controls passed, with four fresh MACC boundaries
and four separate-process replays. Numerical/source-spec approval remains pending;
broader repository gate failures remain explicit.

---

# Current Q/DQ compiler integration — 2026-10-29

`feat/pass-verif-integration`, compiler code `0a463e725`, now registers explicit `lower_quant_ext` validation
for actual xDSL and textual Q/DQ lowerings. It remains outside `all`, which
still selects the same five default checks. Acceptance establishes typed
refinement under the proposed Q/DQ source semantics, with source-spec approval
and framework/native fidelity unestablished.

**80 distinct targeted cases passed**, including six matching gated/ungated LLVM
pairs and injected lowering failures that prevented product delivery. The
optional original-PT2 source intake has host controls; its native qualification
was explicitly deferred by the user until the pinned capture tools are available.
The [Q/DQ guide](docs/guides/lean_qdq_gate.md) explains the three claims and
interfaces. See the [current handoff](out/artifacts/handoff/pass-verif-integration-20261028/README.md)
for evidence, replay and retained attempts.

Earlier qualification below belongs to its stated source cut. Q/DQ integration
changed bound Python/ledger sources, so earlier source inventories are retained
history rather than current completion authority.

---

# Earlier loop/buffer/MACC qualification — 2026-10-29

Use **`feat/pass-verif-integration`**, current compiler code
`c553814aa`. The user resumed the original integration goal after the published
colleague checkpoint and chose to keep MACC draft status until independent
review. The code now has broader loop-entry proofs, explicit buffer identity
prefixes, canonical artifact/dispatch controls and an actual MACC draft-contract
compiler gate outside `all`. **204 targeted pytest cases**, fresh process replay
for the generalized loop and both MACC formats, and the full Lean guard passed.
The guard covers 567 source modules with 15,304 shared theorems and zero violations;
specification pins/locks are unchanged. No full legacy-suite claim is made.

The final remote refresh also merged the colleague's Q/DQ capture and
qualification tooling through `85b354f0d`. See its
[source-binding handoff](out/artifacts/handoff/quantization-lean-qualification-20261029/README.md).
Those changes do not overlap the bound loop/buffer/MACC proof sources; all
five retained proof/replay source inventories still match. Its qualification
results retain their separate source cut.

Start with [the current integration handoff](out/artifacts/handoff/pass-verif-integration-20261028/README.md),
[the compiler stack and examples](docs/guides/lean_compiler_stack.md), and
[quantization/MACC progress](out/artifacts/handoff/quantization-lean-20261028/README.md).
All paused/stop statements below describe historical checkpoints and do not
override the resumed goal. Numerical approval, native fidelity and broader
unsupported profiles remain separate from the implemented claims.

---

# Published colleague checkpoint — 2026-10-28

Start from **`feat/pass-verif-integration`**. Compiler-gate work is committed at
`1ac44938b`; merge `ba4fb6b34` imports the colleague's configuration update
`018c3795d3`, including primary `89ac040ae` and the elision cleanup-helper fix.
The user requested a documentation checkpoint and push so a colleague can
continue, without waiting for a full test suite. The brief follow-up review
started from `82430791d`; its findings and draft MACC theorem are recorded in
the integration and quantization handoffs. The headers below retain their
historical branch/source scopes.
`MERLIN_WORKER_ROOT` denotes the original operator-selected worktree parent;
set local checkout locations explicitly instead of copying another host’s paths.

Read [the integration checkpoint and exact next tasks](out/artifacts/handoff/pass-verif-integration-20261028/README.md),
[the end-to-end stack](docs/guides/lean_compiler_stack.md), and especially
[the quantization handoff](out/artifacts/handoff/quantization-lean-20261028/README.md).

Five compiler selectors are present: tensor DCE, CSE, ordered unrolling,
ordered segmentation and private buffer reuse. Configuration elision and
partial loop hoisting retain explicit instruction/effect APIs. Their modeled
success/error/terminal-state relations and the buffer possible-success relation
are distinct. No native/hardware or human effect approval is inferred.

The original compiler cut passed 35 distinct compiler tests; the replay update
passed three host binding controls. Latest merged-cut native replay and full
compiler qualification are pending. Both configuration families passed fresh
baseline checks, exact delivery and separate-process replay in an isolated
`018c3795d3` checkout on this host. Historical full campaigns keep their own cuts.

The interrupted legacy regression recorded **661 passed, 3 failed**. All three
failures reproduce in pre-integration `1f1b04a64` with byte-identical captured
IR and identical refusals. 116 tests were not completed. Do not label that run
as a passing full regression or relax checker expectations to conceal it.

**Quantization is ready for review, with numerical admission pending.** Source
Q/DQ proofs/corrections are inherited. The additive
`Check/Bf16F32AccBoundary.lean` theorem combines complete equality to the draft
reference with strict qualification of a nonempty set of local MACC sites.
It compiled with only the standard three axioms. This packages existing
candidate-policy results; it does not approve the policy or install a compiler
gate. W8A8 and bf16/f16 still need independent review/admission and the coverage
work in the quantization handoff and `merlin/lean/LEDGER.md`.
`all` does not request numerical policy checks; SPEC_LOCK is unchanged.

The review also identified pending canonical MACC admission-root and loop
artifact-inventory hardening. Neither finding was exercised as a bypass.
Loop tails/general argument lists and partially changing buffer chains remain
unsupported. Work is paused for this colleague handoff; no full suite was rerun.

---

# Latest combined integration — agents remain stopped

This branch integrates primary feat/pass-verification at89ac040ae with Graham's
DCE/CSE branch. Configuration-write elimination and scoped partial loop hoisting
are now present alongside the earlier loop/buffer checks. All three workers
remain stopped; only a NEW user instruction may resume them. Read the current
integration handoff under out/artifacts/handoff/configuration-graph-integration-20261028/README.md.
The opt-in compiler selector remains DCE/CSE-only; the new APIs do not install
compiler-wide enforcement or establish native/human-policy approval.
The detached e048 producer checkout is preserved read-only for source identity.
Older integration/qualification statements below retain their original revisions.

---

# Latest target integration — 2026-10-28

This branch integrates the published primary loop/buffer checkpoint d45415489
with Graham/Agustin DCE/CSE base8215c0015. Start with the
[combined-tree handoff and fresh evidence](out/artifacts/handoff/loop-buffer-integration-20261028/README.md).
The compiler selector remains DCE/CSE-only; scoped loop/buffer APIs have fresh
checks and separate-process replays. Existing structure failures remain.
Primary feat/pass-verification and the three configuration workers were not
switched or rebased; their historical instructions below retain their own scope.


---

# STOPPED — scoped configuration hoist completed, 2026-10-28

The latest user requested finish loop-invariant configuration hoisting, notify
them and STOP all agents. This checkpoint fulfills that scoped task; all three
workers are idle/completed with no live jobs and documented handoffs. No next
pass is queued. Do not resume from any older continuous-work instruction.
A NEW user instruction is required. Root integration is feat/pass-verification
0415d140b (source dbaad5090), with final evidence in this publication checkpoint.
Read [the exact result, limits and three handoffs](out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-HOIST/README.md).

Qualified scope: seven actual supported pairs; nine unsupported boundaries;
fifteen Lean + one host valid-mutant refusals; request/artifact controls, fresh
replay and independent124-root witness. Primary242tests, structure/lint and
fresh baseline check/exactconsumer/separate replay passed. Conditional declared
state/effect model only: no full-pass/all-graph/native/human-review claim.
Protected semantics/locks, failed profiles and counterexamples remain intact.

Earlier loop/buffer integration is also pushed to Graham branch at e0484f488.
That detached read-only producer stays pinned there. Configuration elision and
this hoist are on primary and have not been ported to that branch.

Everything below is historical context, not authorization to resume.

---

# Current — configuration elision integrated; loop-hoist graph work active

Latest continuous-work instruction remains active until a NEW user stop. Root
merged/pushed the loop/buffer checkpoint into Graham/Agustin branch
feat/pass-verification-dce-cse at e0484f488, including a new process-helper
source-binding fix and fresh combined-tree checks/replays. Primary source remains
feat/pass-verification. Its scoped Q3/F3 configuration-elision result is now
integrated at f6ab5cdc3 (source bc2ba713f); publication records follow in this
checkpoint. Read [the exact result, assumptions, evidence and next tasks](out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-CONFIGURATION/README.md).

Primary postmerge:211focused tests and structure passed; fresh actual baseline
check/exactconsumer plus separate-process replay. Full worker campaign:13supported
pairs,21Lean+1host validrefusals, controls/replay and independent actualwitness.
Q1unavailable/Q2UNQUALIFIED remain distinct; no oldcontext promotion. Human/ISA/
native/model/pipeline-composition obligations and frozen semantics/locks remain.

Q/W/F continue separate NEXT-HOIST phase1 tasks: actual new-producer Kernel/loop
bridge; partial/error-state loop-hoist proof; independent fixtures after finalseal.
Read worker NEXT-HOIST/{role}/TASK.md and RELEASE.json after compaction. No new
agents, numerical work, SDKforensics, completion-token/pipeline/hardware claim,
or optional old-buffer/context expansion. Root owns Git; primary final delivery.
Historical stopped/pending/oldnumerical headers below are superseded.

---

# Current continuation — 2026-10-28 buffer merged; configuration active

The current user request is to keep the three existing workers active until a
NEW interruption, merge reviewed graph results into feat/pass-verification, and
then advance to the next graph transformation. Numerical work remains deferred.

Scoped ordered unrolling/1D segmentation is integrated and published at
c3ad8e10f. Scoped original→normalized→reused→segmented private-buffer validation
is now reviewed and integrated at **09fea17e6**. Primary integration passed
84 focused SDK-free tests plus structure; all 191 source files and all five
actual query/part/root bundles match the reviewed successor.

The buffer relation preserves possible successful public/caller observations
under typed inputs and total/private/unobservable allocation. Its full five-case
qualification used the original 186-source fingerprint; the corrected 191-source
baseline check and separate replay are additional evidence. Do not relabel the
historical campaign. Finite-capacity/failure equivalence, native fidelity,
compiler-wide coverage, actual-model coverage and independent human approval
remain open. The separate one-call Context candidate is not integrated.

Q now owns actual IR/gate integration for redundant persistent-configuration
writes; W owns Lean checker/module soundness; F owns independent qualification.
They continue in $MERLIN_WORKER_ROOT/merlin-loop-verification, on the existing
feat/loop-transform-verification branch. Root owns Git and final primary merge.
The task scope is constant-field straight-line elimination, with explicit command
and effect agreement; no barrier/loop/pipeline/numerical/hardware extension.

The newest headers and released task records supersede historical stop, HELD,
old numerical and buffer-pending text below. Preserve every failed attempt,
frozen semantics/specification/lock, and counterexample. No all-graph claim.

Start with [the reviewed buffer result and current task release](out/artifacts/verification/merlin-passes/loop-scope-20261028/COORDINATOR-REVIEW/BUFFER-README.md).

---

# Current continuation — 2026-10-28 scoped graph verification

The latest user instruction is to keep the three existing workers active until a
NEW interruption. Historical stopped/delivered/MACC-only/runtime-investigation
headers below are preserved records, not the current assignment. Graph proofs
remain ahead of numerics; final implementation belongs on feat/pass-verification.

Scoped ordered unroll→1D segmentation is reviewed, integrated and qualified under
its explicit model. Read [result and paper example](out/artifacts/verification/merlin-passes/loop-scope-20261028/COORDINATOR-REVIEW/LOOP-README.md)
and [usage](docs/guides/lean_loop_transform_gate.md). Eight actual changed cases,
15 valid mutations and12 unsupported cases/settings passed, with real delivery,
fresh rebuild/separate-Python replay and binding controls. The merged primary
also passed a fresh check/replay of the same example and56 focused tests.

Q integrates bounded-buffer source/gate interfaces; W owns source-bound buffer
proofs/nonvacuity; F independently qualifies the buffer path. Q/W next extend only
the existing one-kernel-call wrapper context in isolated candidates. Their current
worktree is $MERLIN_WORKER_ROOT/merlin-loop-verification on
feat/loop-transform-verification; read its current QUEUE.md and role task first.
Its applied33 memory proof repairs and buffer Python cut remain separate pending
primary integration. Current source cuts/evidence are not interchangeable.

The buffer all-entry query and nonzero source-kernel witness have checked component
proofs; live qualification/replay remain distinct. Preserve possible-success
public-observation/total-private-unobservable-allocation limits and the8→16-byte
counterexample. Full graph/pipeline/model/native correctness and independent
human approval are NOT established. No old failed SDK/private context is promoted;
current graph checks use new selected project builds without toolchain diagnosis.
MACC code is integrated, but its numerical policy remains unapproved. Publication
of reviewed work is authorized; no numerical/specification approval is inferred.

---

> **CURRENT2026-10-28:** MACC gate/publisher code and candidate qualification reviewed for integration; human policy approval remains absent. See [review](out/artifacts/verification/merlin-passes/bf16-integration-20261028/COORDINATOR-REVIEW/README.md). Active agents are on scoped loops, then bounded buffer reuse, in `$MERLIN_WORKER_ROOT/merlin-loop-verification` (`feat/loop-transform-verification`).

> **LATEST2026-10-28:** User directed all three agents to work strictly on MACC and stop version/toolchain investigation. Read [MACC_ONLY](out/artifacts/verification/merlin-passes/bf16-integration-20261028/MACC_ONLY.md).

> **CURRENT2026-10-28 USER RESUME: bf16/f16 f32 accumulation only.** Three original workers are resumed on `feat/bf16-f32acc-verification` in `$MERLIN_WORKER_ROOT/merlin-bf16-verification`. Read [current task/ownership](out/artifacts/verification/merlin-passes/bf16-integration-20261028/README.md) and your role TASK.md. The main checkout and other workers remain untouched. This supersedes the historical stop below for this bounded assignment only; old runtime holds and human-review requirements remain.

# Merlin verification: START HERE

**2026-10-26: user requested STOP at 5% weekly allowance.** All five workers
were told to checkpoint, write portable handoffs and stop. Do not restart them
from historical prompts. Resume only after a new user instruction.

Primary delivery branch: **feat/pass-verification**. Pull it and record the
exact commit before working. Numerical work remains deferred; the goal is a
paper-defensible claim for Merlin-owned graph transformations in the shared
interval between model2MLIR and OOT. LLVM-owned passes are outside scope.

## Worker handoffs

Each short pointer below identifies the detailed handoff, exact owned source
manifest/patch, current checkpoint, open obligations and next task. Read your
pointer first, then its linked handoff. Preserve other workers' source cuts.

- [W: memory/shape/caller-context and SMT binding](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/W.md)
- [Q: graph gate, whole-entry schedule/unroll and integration](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Q.md)
- [F: declared calls and sole runtime investigation](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/F.md)
- [Prepare: independent source/claim/lowering review](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Prepare.md)
- [Assign: portable source DAGs, staging and reproduction](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/account-handoffs-20261026/Assign.md)

## Graham and the next coordinator

[Graham's substantial parallel assignment](out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025/GRAHAM_HANDOFF.md):
independent clean-host Lean4.35 rebuild, human theorem/assumption/IR mapping
review, actual compiler gate audit, and the two paper FIXME deliverables.

[Planner onboarding](out/artifacts/verification/merlin-passes/progress/planner/README.md)
and [current stop/runtime state](out/artifacts/verification/merlin-passes/progress/planner/paper-deadline-20261026/CURRENT_WORK.json).
Read repository instructions and docs/design/merlin_lean_verification.md.

## Do not lose these distinctions

- Historical qualified proofs retain their original sources, domains, relations,
  runtime bindings and counts. They do not qualify later source candidates.
- Newly published graph extensions are **UNCOMPILED / UNQUALIFIED**. Source
  preparation tests, generated queries and mock host delivery are not Lean
  acceptance, independent review or ordinary compiler enforcement.
- The original/private runtime library hash incident is unresolved. Do not run,
  reselect, rehash or promote those contexts. A fresh official SDK/context and
  new compilation/audits/qualification remain necessary. The SDK watchdog and
  archive Context candidates are source-only; no real SDK download was launched.
- The successful2144-file original IR transport is DATA evidence only. It does
  not clear the runtime hold or fix the separate Omega/IR failure.
- Memory proofs can concern possible public successes under total/private/
  unobservable allocation; scalar/control proofs can preserve complete errors
  and terminal worlds. Never silently equate these relations.
- Preserve finite-allocation, numerical-overflow and other counterexamples;
  frozen Core/Sem/Spec/locks/capsules remain read-only. No new axioms/sorry/admit/
  native_decide/bv_decide/substituted evaluator.
- New host/service/consumer integration is incomplete and public native gate
  activation is held. A saved receipt JSON grants no live proof authority.

Finished source, patches, review documents and selected portable source/query
packets are published. SDKs, compiled objects, giant build trees and duplicate
scratch outputs stay outside Git. Exact source/packet manifests record the
transfer boundary; missing external evidence must be reported, never invented.
