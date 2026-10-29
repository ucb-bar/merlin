# Compiler pass verification checkpoint — 2026-10-29

Additional audit/strengthening based on `4d28efe5c`: see the
[numerical review packet](../../../../docs/design/qdq_macc_numerical_contract_review.md)
for conditional Q/DQ definedness, initialized/mixed-format MACC component proofs,
the plain-reduction classification regression, remaining integration and paper
wording. No numerical approval is installed. Audit results retain their exact
source cuts in `out/artifacts/audits/overnight-pass-verification-20261029/`.
The [completed audit handoff](../../audits/overnight-pass-verification-20261029/README.md)
records 143 MACC passes/two absent-retained-artifact skips, 71 Q/DQ passes,
10 final regressions and six fresh lowering replays. The full new-module import
audit checks 4,272 theorems with no violations; all 235 existing pins match.

Work on **`feat/pass-verif-integration`**. Latest implementation commits are
`5f9fb92de` (rank-one complete-entry bridges), `ef46cc269` (retained deeper loop
bodies) and `91661f328` (original C64 consumer and wide-count hoisting). Remote
publication and both GitHub identities are verified through `85dc14726`,
including complete rank-one qualification documentation. Incoming numerical
changes were fast-forwarded through `c95fb1012`, preserving colleague authorship.
Earlier proof records retain their own source cuts.

All nine pass families have compiler selectors within their declared profiles.
Five belong to `all`; Q/DQ, draft MACC and the two ISA configuration checks require
explicit selection. **The human-review-only milestone remains open:** nested
Kernel commands and broader source/profile coverage still require engineering.
The proof boundary is Merlin-owned transformations through their output IR, with downstream
MLIR/LLVM/executable/hardware fidelity separate.

| Pass family | Integrated behavior and retained evidence | Main scope boundary |
| --- | --- | --- |
| DCE | Typed tensor refinement; separate ordinary-memory dead-index-constant gate qualified in actual default-all routes | Memory relation equates successful complete return/world outcomes; no categorized-error or downstream fidelity proof |
| CSE | Leading tensor gate; original live memory route binds authentic unchanged native CSE with zero jobs/credit | Changed memory CSE refuses; downstream lowering remains independent |
| Unrolling | Empty/inline/bulk/tail, siblings, full lexical contexts and explicit inner depths through dynamic ancestors; 20 actual cases, 18 API controls and public native checks | Actual source strides remain known positive; scalar/SCF/resource admission remains |
| Ordered tiling | Explicit inner depth, partial effects and short tails; independent public TILE and combined UNROLL→TILE one-worker/native controls pass with exact middle/final joins | Static selected index64 controls and supported scalar/SCF contexts |
| Buffer reuse | Actual three-buffer untiled or four-buffer tiled chain; direct check, fresh-process replay and ordinary compiler passed; 2,372 audited declarations, eight finite native observations | Rank-two i8/i16/i32/i64 profile, possible-success relation, explicit total/private/unobservable allocator |
| Q/DQ | 14 native/freshness cases, seven source-byte controls, eight actual lowerings and separate-process replays, final prefix/source-mutation control passed | Proposed source semantics; broader formats/per-group/mixed source graphs and definedness remain separate |
| MACC | Shared/init SSA binding, fresh live replay, approved atomic producer rejection; 49 producer/checker cases with three historical skips and two native regressions | Draft numerical-reference conformance, not original half-source equivalence or numerical approval |
| Configuration elision | Original-owner function/bare-instruction/Kernel gates; actual schedule→worker→encoded bytes→native paths; independent retained-work/frame/initial-setter refusals | Straight-line current profiles and explicit original ISA/effect interpretation |
| Configuration hoist | V1–V4 plus explicit C64 V5 through positive extent2^63−1; 129 cases, symbolic wide proofs, controlled fresh replay and native small cases | All-index Models remains a premise; nested Kernel commands, completion fields and dynamic configuration values remain separate |

The independent validator checks the pass agents' theorem scope, actual input/
output custody, source bindings, canonical roots, real consumer enforcement and
negative cases. It found and helped resolve owner forwarding, bare-module intake,
missing finite observer bindings, stale-source classification, early source
mutation detection and replay test-harness limits. It has found no wrong proof
acceptance in the reviewed cuts. Detailed audits are under
`out/build/validator-alignment-20261029/`.

These results retain **separate exact source cuts**; overlapping cohorts are not
one aggregate final-source suite. Saved reports never confer live authority.
Important retained artifacts:

- `out/build/unroll-modes-20261029/retained-nested-tail-v2/`: genuine complete and
  adjacent roots, check and separate-process replay, 902 declarations/STD3. Later
  registry/source changes make this a historical cut. Current context and chain
  cohorts are retained in the same directory. The independent current-source
  compiler/native refresh passes: exactly one job, 398 roots, 1,690 STD3
  declarations and all 172 references current at that cut. Its actual middle and
  target hashes join exactly; no inherited-control/source-success premises remain.
  Separate fresh chain replay and genuine control/body/frame refusals also pass.
- `out/build/buffer-untiled-20261029/`: direct/ordinary compiler proof trees,
  `fresh-process-replay-v3/`, `normalization-only-v4.xml` and native comparison.
  Normalization-only change with identity reuse still gets a real full-chain
  proof (2,367 declarations) without false positive reuse credit. Forty-one
  legacy host controls pass. Independent `buffer-widths-v1/SUMMARY.json` under
  `out/build/validator-alignment-20261029/` adds fresh i8/i16/i64 checks, exact
  delivery, 1,789 current references per width and 30 native observations against
  the true untransformed source, including wrap/high-bit cases. This qualifies
  existing widths rather than extending semantics. An earlier combined test hit pytest's 900-second
  limit; standalone fresh replay passed and the two-build test now has a bounded
  1,800-second allowance. Earlier source-drift refusals remain retained.
- `out/build/pass-verif-integration-20261028/qdq-prefix/`: exact pinned native
  wheel/producer identities, final eight lowering/replay pairs and retained
  corrected-prefix proof plus actual post-composition foreign-source refusal.
- `out/build/configuration-compiler-20261029/`: actual current bare/function/
  Kernel and schedule/native tests, source-mutation-before-proof control and
  36 legacy elision units. The independent V20 export runs without verifier
  imports; V6/V24 export/tool incompatibilities are separate baseline findings.
- `out/build/configuration-hoist-compiler-20261029/installed-v1/`: 70 cases,
  including both real public-owner/native paths with 15 finite scenarios each.
  V2 indexed-data public/native and wrong-IV canonical cases passed on recorded
  cuts; its conservative signed-index64 tightening passed 29 current host cases.
  Installed V3 qualification passes 53 unique cases: 46 host and seven canonical,
  with four V3 plus V1/V2 ordinary public/worker/native paths and a genuine
  reversed-setter refusal. V3 audits 364 declarations with STD3 and all 833 source
  references current; V1/V2 retain nine modules and 315 declarations. The
  independent audit is `hoist-prefix-installed-audit-v1.json`.
  V1 remains unchanged and the all-index Models premise is preserved. Default
  selected adapters remain conservative; broader rewrites use explicit untrusted
  proposals checked against independently selected original effects.

Remaining engineering: nested counted-Kernel commands, broader buffer ranks/element representations,
additional Q/DQ source forms/precision semantics and broader configuration
producer-language coverage. Selected innermost depths, top-level sibling/context composition,
actual unroll→tiling chains and multiple flat Kernel hoists are implemented.
Human review remains distinct: proposed Q/DQ semantics/poison/definedness, draft
MACC accumulation/fusion policy, exact ISA pure-write/frame/failure contracts,
complete/nonaliasing state and allocator/observer assumptions. Neither native
examples nor caller qualification labels approve those contracts.

See the [compiler stack and examples](../../../../docs/guides/lean_compiler_stack.md),
[assumption ledger](../../../../merlin/lean/LEDGER.md) and
[completion plan](../../../../docs/design/lean_pass_integration_completion.md).
Frozen semantic/specification definitions and locks remain unchanged. Existing
repository-wide structure/overfit/provenance findings outside this slice remain
visible; targeted verification does not certify full CI. Remote publication is verified through `969ee9340`; the GitHub API resolves
both identities for all 20 new commits to `grammy117`. Earlier authentication
failures are historical, not the current delivery state.

Earlier results below retain their stated source cuts.

[unpublished-commit-map.json](unpublished-commit-map.json) maps older retained
commit identifiers to the publication series after one message-length correction.
All trees, identities and dates are preserved; published history is unchanged.

The final broad source scan records seven target-name findings and 44 regex
call sites in 16 files unchanged from the incoming base. Staged target/regex/fact
checks pass. Global structure/provenance scans retain 16/two baseline findings.
These are not passing full-CI results; no guard or ratchet was weakened.

New retained source cuts:

- `out/build/unroll-modes-20261029/`: the 20-case selected-depth cohort and
  `depth-options-current-v2.xml` (18 host API cases). Independent audits
  `inner-installed-cohort-audit-v1.json`, `inner-tile-public-v1/RESULT.json` and
  `inner-combined-public-v1/RESULT.json` are under the validator directory.
  Combined public checking has 408 roots/1,648 STD3 declarations, all 175 refs
  current, exact actual original/middle/final hashes and matching LLVM/native
  cases for outer limits zero, negative and positive. Earlier cuts remain separate.
- `out/build/configuration-hoist-multi-20261029/INSTALLED_SUMMARY.json`: 79 host
  plus eleven canonical cases, 90 distinct IDs, no failures/skips. Seven public
  accepted paths compare 153 finite initial/failure states. Independent
  `hoist-multi-installed-audit-v1.json` confirms eleven modules/25 roots/392 STD3,
  841 current refs per V4/replay report and genuine closed kernel refusals.
  The initial diagnostic regression is retained separately; repaired routing
  preserves original single-loop refusals. Malformed zero-extent Kernel input
  is rejected by the existing producer and earns no coverage.

New independently reviewed source cuts:

- `out/build/unroll-modes-20261029/RETAINED-DEEPER-SUMMARY.json` and the validator's
  `retained-deeper-{compiler,replay,refusal}-audit-v1.json`: actual depth-two
  unroll→tile preserving a dynamic depth-three body, exact whole-entry joins,
  native boundary/wrap observations, fresh process replay and two genuine Lean
  callee/stride refusals. The 424-root/1,678-STD3 replay had all 204 source/product
  refs current at its recorded cut. Later bound-source edits require fresh checking.
- `out/build/configuration-hoist-wide-20261029/INSTALLED_SUMMARY.json` and the
  validator's `hoist-wide-installed-audit-v1.json`: 129 unique tests, no failures
  or skips, six serialized SDK invocations. Explicit C64 counts 1,000,001 and
  2^63−1, fresh controlled MAX63 replay, actual count mutation refusal and 15
  finite native comparisons pass. All eleven installed hashes match. Accepted
  closures have 9/11 modules, 15/25 roots and 315/392 STD3 declarations; refs are
  861/868/882 by profile. No huge native compilation/execution occurred.

Rank-one buffer integration is committed in `5f9fb92de`. All **33 distinct
installed qualification cases pass**: 12 host controls and 21 SDK/native cases,
without retries, failures, skips, timeouts or source edits. Actual untiled and
tiled gates audit 250 canonical roots and 2,545 STD3 theorems. Both public
compiler routes, direct extents 1/3/65,536 and two fresh-process replays pass.
Ten actual target mutations produce three genuine closed Lean refusals and
seven explicit host refusals; three initializer-link component controls also produce
closed fact refusals. The two public routes retain 24 native wrap observations
and six untouched-entry assertions across independent source/baseline/checked
arms. Evidence is under `out/build/buffer-rank-one-20261029/installed-qualification-v1/`
in `RESULT.json`, `EVIDENCE-SUMMARY.json`, both XMLs and retained proof trees;
independent review is `out/build/validator-alignment-20261029/rank-one-final33-qualification-audit-v1.json`.
The captured-uniform-seed profile excludes pure
computed initializer bodies, which the actual producer can transform. This is
remaining engineering, not human-review-only readiness. The current producer
reuses identical memref types/alignment; equal-byte differently shaped objects
are not a supported reuse case of that producer.

Further current-cut evidence and interface findings:

- Rank-one kernel cases at extent1 pass for both actual untiled three-buffer and
  tiled four-buffer output. Independent audit of the true tiled case verifies
  1,971 current refs, 134 successful processes, 250 roots and 2,545 STD3 theorems;
  actual loop count changes from two to four with one-dimensional views. The
  extent3 untiled case and raw-only fresh-process replay also pass. The completed
  33-case cohort and final independent audit supersede this initial checkpoint.
- `out/build/validator-alignment-20261029/default-all-memory-composition-audit-v1.json`
  records the initial integration gap: default `all` submitted the later changed
  memory DCE stage to a tensor-only checker and rejects skip-only memory CSE.
  Exactly one unused `%dim0 = arith.constant 3 : index` is deleted on both routes.
- `out/build/default-all-memory-20261029/installed-qualification-v3/` closes
  that delivery gap within the admitted memory profiles. Both untiled extent3
  and tiled32 extent257 tail cases pass in 1,207.78 seconds, with four accepted
  workers, 24 native kernel observations and six executed untouched-entry checks.
  Each BUFFER worker audits 250 roots/2,545 STD3 theorems; each following memory
  DCE worker audits eight roots/1,137 STD3 theorems. These are per-worker counts,
  not unique-theorem totals. The independently reviewed source/product joins
  bind authentic unchanged CSE with zero jobs and zero proof credit. Actual
  changed memory CSE, missing/skip-only records and forged owners still refuse.
  The installed fix uses a same-mode dead-index-constant proof over
  complete ordinary entries, exact independent target/frame/SSA joins, and an
  original live memory-CSE scope. It equates successful complete return/world
  outcomes, not categorized errors, divergence or native fidelity. The earlier
  exploratory preflight, native property-dictionary-order refusal and missing
  incoming checker artifacts remain separate retained attempts; the latter was
  repaired by rebuilding the current checker, without changing sources or flags.
  Current property-order/ownership host controls pass all 46 cases. Private
  buffer tiles now permit 2..65,536 with unchanged signed-index64 safety checks.
  Implementation commits are `d1c018532` (generic rule and actual endpoint
  obligations) and `1b158739a` (compiler enforcement/ownership); the practical
  tile policy is `d2b7c977b`. Each recorded run retains its exact source cut.
  Six later Python formatting changes have independently checked identical
  executable ASTs; the default-all test additionally fixes only its stale
  module description. The formatted successor passes all 76 memory host cases.
  Separate-process direct DCE delivery/replay passed in 230.12 seconds on the
  pre-format cut, including ignored forged approval fields and copied-token
  refusal; fresh reports remain evidence, never saved delivery authority.
- `out/build/buffer-rank-one-20261029/pure-prefix/` records fourteen actual host
  forms and an additive typed-prefix execution proposal. Three concrete foundation
  files remain uncompiled. They preserve the existing false-pure specialization;
  no computed-prefix compiler profile is admitted by this source work.

Q/DQ lowering and bf16/f16 MACC are now handled by the colleague. This lane
continues the other MLIR/xDSL pass proofs and compiler gates with one independent
reviewer; LLVM formal verification is outside the agreed scope. Existing numerical
draft/proposed status remains unchanged.

---

# Completion plan and scalar loop generalization — 2026-10-29

Use **`feat/pass-verif-integration`**, code `0fd1b0294`, based on the colleague's
`a829c1a66` checkpoint. The agreed endpoint is Merlin pass proofs and compiler
integration through output IR. The
[completion plan](../../../../docs/design/lean_pass_integration_completion.md)
defines the work packages, actual producer coverage and release criteria.
**The milestone where only human review remains is still open.**

The compiler now accepts scalar argument/carry lists without a separate fixed
arity cap, including zero carries. Scalar unroll factors and tile sizes are
integers at least two without the old factor-four / size-sixteen upper bounds.
List-proof emission is iterative; complete SSA syntax is charged before proof
generation. Syntax/input/query resource overflows are unavailable outcomes.
Historical standalone and private-buffer parameter domains stay explicit.
No production transformation, Lean semantic definition, specification, lock or
numerical approval changed. The ledger corrects its stale Q/DQ pruning wording
without changing A9's purity premise.

The incoming `ae144d959` fix requires the complete Lean proof library and both
executables to be current, including audit-only objects. It retains `--no-build`
and refuses stale/missing artifacts. Its three real-project regression cases
passed in this merged qualification.

**59 targeted checks passed, zero failures/errors/skips.** They cover arguments
and carries beyond sixteen, an effectful zero-carry loop with failure-state
equality, CLI/type/resource controls, artifact controls, one actual private-buffer
chain, all six Q/DQ compiler pairs, both bf16/f16 MACC compiler pairs, default
selection and corrupt-output rejection. Four loop compiler pairs have matching
gated/ungated LLVM and independent native wrapping results. Factor seven and
tile seventeen are checked on 126 iterations, exercising a short final tile.
The buffer proof retains its separate possible-success observation relation.

A separate process rebuilt and delivered the factor-seven / tile-seventeen
boundary from descriptive buffers/settings with saved approval fields omitted.
The target hash matched exactly; the audit used only the standard three axioms.
The fresh completion, rather than the saved JSON, authorized delivery.

Evidence is local under `out/build/pass-proof-completion-20261029-v1/`:
`merged.xml`, `merged.log`, `SUMMARY.json`, `replay-summary.json` and the original
proof directories. [qualification.json](qualification.json) records the source
manifest and unique current test IDs. The earlier 32-case arity run and the
premerge run stopped after 51 passes retain separate cuts; the latter records
its interruption and unfinished tail. These runs overlap and must not be summed.
Ignored toolchains/logs are not distributed by Git.

Formatting, staged source/layout/path/ratchet checks, documentation and protected
Lean file locks pass. Current boundary proofs have their own kernel/axiom audits;
the full Lean guard was not rerun because its Lean/specification inputs are
unchanged. No full legacy suite was run. Whole-repository checks retain 16
structure, two overfit-register and two provenance findings in untouched baseline
files; no ratchet or policy was weakened.

Next work is the shared semantic/context bridges, inline/tail/nested loop
execution, independently usable untiled reuse, complete numerical boundaries
and configuration consumer wiring. The current word/SCF projection and retained
proof-expansion budgets still constrain support. Real original-PT2 SourceCapture
qualification remains deferred until the pinned tools are available. Q/DQ source
semantics and MACC numerical policy retain their proposed/draft status, outside
`all`. The [stack guide](../../../../docs/guides/lean_compiler_stack.md) and offline
stack now show the updated domains.

Everything below retains earlier source cuts and qualification.

---

# Q/DQ compiler integration follow-up — 2026-10-29

Compiler code `0a463e725` integrates the colleague's `85b354f0d` tooling with explicit
`--validate-passes lower_quant_ext`. The existing xDSL recorder and an outer
textual recorder submit actual complete boundaries to the canonical owned
worker. The textual boundary includes its slice repairs and ABI annotation;
every edit must pass the full proof. Nested xDSL calls are covered once.

Acceptance is **strict typed refinement under the proposed Q/DQ source model**,
with source support and chain composition. It does not approve the source
specification, prove unrestricted framework/IEEE behavior or establish native
fidelity. Quantizer definedness remains a separate unestablished strengthening.
Q/DQ and draft MACC remain outside `all`; its five default checks are unchanged.

**80 distinct targeted cases passed**: 54 Q/DQ/capture-host controls, 21 default
and MACC dispatch controls, and five original-graph reconstruction host controls.
Six real compiler pairs (quantization, per-tensor dequantization and per-channel
dequantization, through xDSL/textual paths) have byte-identical gated/ungated LLVM.
Three injected-quantizer controls prevent compiler product delivery. Counts are
separate from the older 204-case source cut below and must not be summed.

A fresh process and worker replayed a saved actual quantizer boundary and
rechecked its exact input/output hashes. Only source bytes and tool selections
were reused; the earlier accepted status was not proof authority.

Optional `SourceCapture` selection reconstructs original PT2 nodes/returns,
reuses producer/Torch/graph/IR checks, and rechecks before and after the Lean
proof. It retains actual endpoint and source/capture hashes, and uses the
installed owner's wheel pin. **Native source-intake qualification is deferred
by the user** until the pinned capture tools are available; its five host
controls substitute the frontend and establish no native success claim.
Earlier captures require fresh qualification after changed checker-source hashes.

See [the Q/DQ guide](../../../../docs/guides/lean_qdq_gate.md),
[the nine-interface stack](../../../../docs/guides/lean_compiler_stack.md), and
[qualification.json](qualification.json) for this follow-up's source manifest,
logs, fresh replay and retained failed/interrupted attempt. The lock-only Lean
guard and docs checks passed. No Lean semantics, specification, software pin or
SPEC_LOCK changed; no full test suite or full Lean audit was repeated.

Everything below retains the earlier loop/buffer/MACC source cut. Its tests and
audits remain historical evidence; bound sources changed in this follow-up.

---

# Earlier loop/buffer/MACC qualification — code cut `c553814aa`

**Targeted compiler integration is qualified. MACC remains an explicitly selected draft contract.**
Continue from `feat/pass-verif-integration`. Compiler work is committed at
`1ac44938b`; merge `ba4fb6b34` includes the colleague's `018c3795d3` and primary
`89ac040ae`. Original histories are retained. This checkpoint is a development
branch, with no claim of full legacy regression or human semantic approval.

After the published colleague checkpoint at `4ef31ee8d`, the user resumed
development and explicitly retained **draft MACC status until independent
review**. Current code is committed at `c553814aa`, with separate commits for
loop artifact hardening (`b61631b4a`), actual MACC checking (`0bb5d0221`),
general loop entries (`d3cb979ed`) and buffer identity prefixes (`0a66d5cef`).
No full legacy suite is required for this targeted integration; earlier runs
and failed attempts retain their original evidence and source cuts.

The final remote refresh merged the colleague's three Q/DQ commits through
`85b354f0d`, including explicit fresh-capture source binding, pinned producer
and Torch payload checks, and reproducible qualification/replay tools. Read
the [Q/DQ qualification handoff](../quantization-lean-qualification-20261029/README.md)
for its source domains, numerical assumptions and separate results. No incoming
file overlaps the bound loop, buffer or MACC proof sources: all five retained
proof/replay inventories still match. The historical legacy counts below predate
these Q/DQ regression updates and were not rerun after the merge.

## Current compiler changes

- Scalar loop proofs accept multiple changed definitions, zero to sixteen
  entry arguments and one to sixteen word carries. Exact unchanged stages
  retain their captures and do not count as positive rewrite coverage. Existing
  narrower standalone profiles keep their original declared limits.
- Private-buffer proofs accept already-normalized inputs and one-to-one reuse
  plans. `GraphLoopsShape.Identity` checks identical complete bindings and zero
  normalization sites; the original changed-normalization checker is intact.
  Every allocation/free binding remains checked. Actual initializer tiling is
  still required by this memory route.
- `--macc-f32acc` / `Options(macc_f32acc=True)` runs the actual bf16/f16 MACC
  pass at compiler entry. `bf16_f32acc_draft_contract` is a separate explicit
  proof selector, excluded from `all`. Its accepted claim is draft-reference
  conformance, with numerical approval still abstracted.
- Loop objects are frozen before audit and checked again for content and tree
  membership at delivery. All formal compiler selectors reject custom handler
  aliases and additional worker import roots. MACC governance and review lookup
  bind the installed source owner, including the review publisher.

The MACC worker audits every consumed root transitively against the standard
three axioms and scans all imported project declarations for banned constructs.
Its report explicitly identifies that scope. Canonical project objects and
process evidence are revalidated through live delivery; this is not a private
object snapshot or a proven runtime import map. The default full auditor is
preserved. The full current guard and targeted compiler checks passed.

## Current qualification

**204 distinct targeted pytest cases passed**, counted by unique test IDs:

| Controls | Passed |
| --- | ---: |
| Loop profiles, kernel checks, native execution and artifact controls | 42 |
| MACC capture, compiler enforcement and canonical dispatch | 21 |
| Numerical admission, pending-result and preparation controls | 108 |
| Owned worker lifecycle and failure handling | 30 |
| Identity reuse directly and both identity prefixes through the compiler | 2 |
| Default DCE/CSE integration with draft MACC excluded | 1 |

Separate fresh processes replayed bf16 and f16 MACC boundaries and the loop
case with six arguments and two changed definitions. All delivered the exact
target bytes with no incoming live authority. Four current compiler pairs
produced byte-identical gated/ungated LLVM: combined unroll/tiling with five
carries, both buffer identity prefixes, and the two half formats. Native loop
execution checked independent expected wrapping results. Corrupt loop and MACC
producer outputs stopped compilation; no final LLVM product was delivered.

An additional full buffer proof checked identity normalization followed by
changing reuse/tiling. The direct identity-reuse and identity-normalization
reports still match all their bound source hashes. Buffer observations remain
possible-success observations under the explicit allocator model.

The current full `MerlinVerify` guard passed: **567 source modules covered,
550 shared modules / 15,304 shared theorems audited, zero violations**, unchanged
specification locks and pinned statements. Driver counts overlap and are not
summed. Loop/memory closures have their separate gate audits. The MACC worker's
per-check audit remains explicitly narrower than this full guard.

Test-group manifests, source hashes, local raw log locations and retained failed attempts
are in [qualification.json](qualification.json). The full legacy pytest suite
was not rerun, as requested. Its earlier three baseline failures and 116 pending
nodes remain recorded; neither those counts nor historical campaigns are
relabeled as current successes. Current structure checks retain the same 16
previously audited baseline findings. The provenance scan also refuses two
retained review reports whose bytes match the remote baseline; their paths and
hashes are recorded in `qualification.json`. No provenance record or ratchet
was changed. Documentation, artifact layout, local-path and staged source checks pass.

## Start here

- [Compiler stack: location, interface, example and claim](../../../../docs/guides/lean_compiler_stack.md)
- [Interactive offline stack](merlin-lean-stack.html): five default checks, explicit draft MACC and two configuration APIs
- [Machine-readable checkpoint and remaining test nodes](qualification.json)
- [Quantization progress and review tasks](../quantization-lean-20261028/README.md)
- [Assumption ledger](../../../../merlin/lean/LEDGER.md)
- [Verification design](../../../../docs/design/merlin_lean_verification.md)
- [Shared compiler tooling](../../../../docs/guides/verified_compiler_tooling.md)
- [Root handoff and historical checkpoints](../../../../VERIFICATION_HANDOFF.md)

Fetch and check out `origin/feat/pass-verif-integration` before continuing.
The old `feat/pass-verification-dce-cse` branch does not contain this compiler
integration. No numerical or hardware approval is implied by either branch.

## Implemented

The default compiler checks are `prune_dead_pure_tensor_ops`, `cse`,
`ordered_loop_unrolling`, `ordered_loop_segmentation` and `buffer_reuse`.
Transform selection is separate and off by default: `--loop-unroll-factor`,
`--loop-tile-size`, `--buffer-reuse-allocator`, or typed
`compiler_transforms.Options`. Selected changing boundaries require fresh
proofs before returned compiler products can be delivered.
The additional `bf16_f32acc_draft_contract` selector is explicit-only; it never
turns draft numerical-policy conformance into original-source refinement.

The loop adapter captures actual source SCF boundaries without a historical
producer checkout. Individual unrolling uses auxiliary segmentation only as
proof scaffolding; the auxiliary output is never delivered. Standalone tiling
uses an identity unroll stage. Explicit private memory takes a direct
SCF/memref-to-LLVM route with its existing allocation/free lifetimes.

Current-compiler replay reconstructs individual/composed profiles and optionally
delivers the exact target through a fresh completion. The current generalized
loop case and both MACC formats passed separate-process replay. Earlier
individual-loop replay cases retain their original source cut and evidence.

The colleague's configuration-write elimination and scoped partial loop-hoist
proofs are merged. They bind complete instruction-MLIR or Kernel endpoints,
selected ISA/effect facts, canonical proof sources, root/type/import audits,
process artifacts and live delivery/replay. Cleanup-helper binding fixes are
included for elision, buffer and standalone loop receipts. Hoist receipts also
observe imported helper origins. The hoist unit fixture now regenerates exactly
matching local facts/software and accepts `MERLIN_HOIST_REFERENCE`, avoiding
retained paths into another machine's home directory.

Configuration APIs remain explicit conditional interfaces, outside the
registered MLIR compiler selectors. Elision removes only already-redundant total pure
constant writes. Hoisting moves one constant pure setter before a positive
counted loop with an ordered preserving body. Both preserve complete modeled
error/terminal states, including failing retained calls. Zero-trip/nested/
completion-token/asynchronous and overlapping-body-write hoists are excluded.

## Evidence and limits

### Historical review at the published `4ef31ee8d` checkpoint

`MerlinVerify.Check.Bf16F32AccBoundary.check_sound` composes the existing exact
output/reference equality and strict local-site qualification, requiring at
least one collected source contraction. It includes unsuccessful executions
in the equality claim, under explicit SSA-renaming/call-locality laws. This is
conformance to the source-derived **draft numerical reference**, not equality
to the original half-accumulation program. The new module compiled with Lean
4.35.0; its theorem uses only `propext`, `Classical.choice` and `Quot.sound`.
No compiler selector, adapter, admission record or frozen semantics changed.

That read-only review found the following tasks. Canonical ownership, loop
artifact hardening, general loop entries and identity prefixes are addressed
above; the historical review itself ran no adversarial tests:

- `bf16_policy_gate.check_policy` obtains trusted admission/review roots through
  `repo_root()`, which honors `MERLIN_REPO_ROOT`. Bind governance records to the
  installed source owner and test override refusal before enabling a policy gate.
- `loop_transform_gate` records compiled-artifact hashes after its audit and
  completion checks the recorded roster without checking tree membership.
  Bring this in line with the buffer gate's pre-audit inventory and mutation
  checks. No bypass was demonstrated.
- The loop compiler profile requires one changed top-level loop, exactly two
  entry arguments and 1–4 word carries. Tails, empty and fully inlined loops
  are excluded. Selected mixed unchanged/changed loop stages are refused.
- The buffer compiler route requires normalization, reuse and tiling all to
  change. Already-normalized and reuse-only chains need proof/adapter work.

The new theorem's targeted compilation is the only new proof execution in this
follow-up. Earlier results below retain their original source cuts. The full
suite was not rerun, as requested.

The original compiler cut passed **35 distinct compiler tests**: a retained
33-test run plus a nine-case focused recheck overlapping seven cases. Thirteen
positive compiler cases accepted fresh proofs and emitted byte-identical gated
and ungated LLVM (ten loop and three memory cases). Native tests compared three
scalar variants over six vectors and one memory variant over three vectors,
including wrapping values and caller-memory observations. Two actual loop
corruptions reached Lean and were rejected; two memory corruptions were
structural `unsupported` refusals, not Lean proof counterexamples.

Those results precede the latest replay/helper-binding changes. Their exact
25-file source cut is retained locally in
`out/build/pass-verif-integration-20261028/qualified-before-configuration/`.
Do not relabel them as final qualification of the merged source.

On this host, isolated `018c3795d3` source passed a fresh baseline check, exact
live delivery and a separate-process replay for each configuration family:

| Family | Required roots | Transitive theorem audit | Axioms |
| --- | --- | --- | --- |
| Configuration elision | 17 | 247 | Standard three only |
| Scoped partial hoist | 15 | 315 | Standard three only |

Counts overlap and are not summed. These are new baseline repetitions, not a
new full campaign. The colleague's historical 13-pair elision and seven-pair
hoist campaigns retain their original cuts and approval limits.

The earlier full `MerlinVerify` guard passed: 566 tensor-checker modules,
15,300 shared theorems, zero violations, unchanged locks/pinned statements.
Loop and memory closures were audited separately by their gate jobs.
Latest focused check results and source hashes are in `qualification.json`.

The legacy 780-test regression was gracefully interrupted for this handoff:
**661 passed, three failed, 116 not completed**. All three failures reproduce
with the exact pre-integration Python source at `1f1b04a64`, identical captured
before/after IR and identical checker refusals. Two body checks do not verify;
one source-support check refuses generated `scf.for`. This establishes that the
failures predate this integration; it does not establish whether each underlying
rewrite is semantically incorrect. Preserve them and do not relax expectations.
The remaining 116 node IDs, including the interrupted case, are in the checkpoint.

Full-tree hygiene checks also have inherited findings: structure 16, artifact
layout 48, target names 7 and regex 44. Every flagged source matched the merged
base bytes when audited. Changed-file/staged checks passed before this latest
checkpoint; final checkpoint checks are recorded separately.

## Review and future work

1. Review code cut `c553814aa`, the current qualification above and the eight
   stack interfaces. Recheck/replay for another checkout or changed source;
   saved reports never restore live authority.
2. Extend unroll tails/empty/inline modes, arbitrary nesting or internal calls
   only with their corresponding proof bridges. Untiled memory initializers
   likewise require new simulation; do not manufacture a tile-one certificate.
3. If a broader legacy campaign is wanted, complete the 116 pending nodes and
   retain the three independently reproduced baseline failures. It is optional
   follow-up rather than a claimed all-green regression here.
4. Check/replay both configuration APIs on a final installed source path and
   chosen SDK. Isolated producer capture ignores `PYTHONPATH`; use Python with
   the producer's normal dependencies installed. `MERLIN_HOIST_REFERENCE` must
   select a read-only detached source checkout at
   `e0484f4886ae0780d0325cdfa998842b61bb647c`; elision uses
   `4499eebb959c0f92cefd92cc6d80646ba66474ed`. Preserve the hoist source manifest.
5. Review MACC through the [quantization handoff](../quantization-lean-20261028/README.md).
   Decide the source contract through independent review before approved
   numerical admission. Exact compiler captures, canonical binding and fresh
   replay are implemented; candidate-reference conformance retains its distinct
   draft status until independent review.

Raw test/proof/SDK trees remain local under
`out/build/pass-verif-integration-20261028/`; saved JSON is descriptive data.

Loop proofs concern equal returns/errors/terminal world under their common
scalar/SCF interpretation. Buffer proofs concern possible successful public
observations under total/private/unobservable allocation, not full error/world
or finite-capacity allocator equivalence. Python projection/transport, native
lowering and hardware fidelity remain separate. Frozen tensor semantics,
SPEC_LOCK, A9 and numerical admission are unchanged.
