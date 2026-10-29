# Overnight integration work — 2026-10-29

Active branch: **`feat/pass-verif-integration`**. The latest committed checker
slice is `ce4369279`, preserving repeated MACC SSA operands and initialization
used as contraction inputs. Further loop and Q/DQ integration is in the working
tree and is undergoing source-frozen qualification. The active objective still
covers all current pass interfaces through Merlin output IR; it is not complete.

Current evidence, with distinct source cuts retained:

- Current-source empty/full-inline/bulk-plus-tail unroll adapters passed 14
  targeted cases: operational kernel/audit/delivery, actual corruption refusal,
  and three compiler→LLVM→native comparisons. Legacy held entry points remain
  held. An initial native test harness reused a loaded library pathname; fresh
  process checks diagnosed it, unique per-case library paths fixed the harness,
  and the three affected native comparisons passed on rerun.
- A common-registry five-function query covers empty, inline, tail, bulk and an
  unchanged dynamic-loop entry. Kernel compilation/audit and the real complete
  compiler/native case passed. One earlier direct delivery test correctly
  refused concurrent source changes after audit; its isolated stable-cut rerun
  is pending. Actual corrupt inline and bulk functions were rejected.
- Nested-body unrolling and standalone tiling queries now compile. The nested
  body is retained operationally and related with the existing scope checker,
  not replaced by a trip-count claim. Canonical delivery/mutation/native
  qualification is running. This does not yet claim arbitrary inner-loop
  rewrites, multiple sibling changes or all nested inline/tail modes.
- MACC has 24 new direct checker cases and 29 existing direct regressions
  passing on the updated checker. Both initialized-input formats passed fresh
  live delivery and replay. Earlier shared-operand live checks retain their
  earlier cuts. The new helper theorems audit to the standard three axioms;
  frozen semantics/specifications and numerical approval are unchanged.
- Q/DQ's previously missing exact CPU Torch wheel/runtime is provisioned in
  isolated owned scratch, with the existing wheel hash and pinned producer
  verified. Real original-program capture/replay passed, including the expected
  precision-loss refusal. Eight captured xDSL/textual boundaries passed before
  the later MACC checker rebuild; their old replay correctly refused changed
  checker provenance. A fresh final source-qualified compiler matrix is running;
  its first real optional-SourceCapture compiler case passed.
- Q/DQ now has an explicit checked prefix option joining captured IR to the
  actual lowering input, with a composed captured-IR→output refinement. Exact
  input binding remains the default. This claim remains under the proposed
  source model; native examples do not approve universal framework semantics.
- Current-compiler configuration elision intake/worker/hook work is prepared in
  isolated scratch while numerical source bindings are frozen. It is not yet
  installed in the ordinary compiler; the historical standalone API is intact.

Do not sum overlapping runs or relabel scratch candidate proofs as compiler
acceptance. A 21-call tail stress query exceeded its 180-second resource budget;
no acceptance is claimed for that case. Operational budgets remain explicit.

The unapplied MACC producer-review packet contains real counterexamples and two
proposals: preserve the unsupported module with zero policy coverage, or reject
an unsupported policy request before mutation. Installation requires separate
review under `merlin/lean/AGENT.md`; numerical admission is an independent review.
Q/DQ per-group and broader precision forms still need semantic/proof coverage.
Untiled buffer reuse, configuration consumers and general loop/context coverage
remain engineering tasks. Detailed limits are tracked in the
[completion plan](../../../../docs/design/lean_pass_integration_completion.md).

Earlier results below retain their stated source cuts.

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
