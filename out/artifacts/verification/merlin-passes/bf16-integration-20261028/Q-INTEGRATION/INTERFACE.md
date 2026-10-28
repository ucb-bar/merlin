# MACC policy gate interface — current v2

Scope: MACC_ONLY.md, zero-initialized bf16/f16 two-input contractions with ordered
f32 accumulation and one final truncation. No custom SDK provider/inventory.

## Input and routing

Pass key `lower_bf16_matmul_f32acc`; built-in `bf16_f32acc_policy`.
Existing worker request: `directory`, `before`, `after`, `label`, `params` with
exactly `sdk_bin`, `compiler_python`, `index_bits` (integer64). Caller receipts,
review booleans, alternate policy paths or saved results are not inputs.

`pass_validation.observe` wraps the actual prepare-bf16-matmul-f32acc stage in
its existing position. Unselected transformation executes unchanged. Identical
before/after text submits nothing and supplies no positive policy coverage.
Aliases to another handler and custom worker import paths refuse when selected.
Environment configuration checks pending review before constructing CheckerTools;
explicit callers constructing CheckerTools themselves perform its normal preflight.

## Live proof checks

`check_policy(job)` requires installed review admission, ordinary CheckerTools
freshness, exact current Lean/exporter receipt bindings, fresh before/after export
and round trip, live `check_stage`, and final revalidation of all bound bytes.
`validate_report(report, source_export)` is pure consistency checking, NOT authority.
It requires all of:

- Complete-output `pass_boundary.semantic_match` / Exact.check_sound.
- Strict reconstructed local `pass_boundary.site_qualification_passed` and
  `checked` / Sites.check_sound; weaker conditional_match is insufficient.
- All actual source function/operation-position/result-ID entries, in order;
  exact expected/covered/candidate counts and per-site true qualification.
- Flags exactly none or contract, index64, exact checker/root/policy identity.

Source enumeration is deliberately conservative over half-result matmul,
batch_matmul and generic reductions. Unsupported candidates cannot silently
vanish. Nested half contractions refuse; supported surrounding nested IR remains
covered by complete-output equality. A changed zero-site pair refuses rather than
counting positive policy coverage. Lean checks source maps, operand/body order,
zero initialization, casts, shapes and local domain; host inventory is no proof.
Whole-capture qualification/definedness is separate and may remain false.

## Audit and receipt API

`audit_candidate(tools, work)` uses selected absolute lake env + absolute auditor;
raw stdout/stderr retained. It audits project dependency closure, exact theorem
statements and every required root's axioms (std3 only). `tools` is a temporary
live file-reference map for lake/lean/compiler_python/checker/audit, not a new
SDK admission format. Candidate auditing/checking itself grants no policy approval.

`bf16_f32acc.check_stage`, `check_stages`, and `replay` take optional
`theorem_auditor: Callable[[str,str],list[str]]`. Existing historical default is
preserved. Gate always supplies its freshly checked exact roots; no historical
~/.elan auditor fallback. F may inject the selected auditor for candidate checks.
Existing source/target/export/checker/SPEC_LOCK receipts and freshness remain.
No new runtime/environment/loader infrastructure is required.

## Human review / trusted-host installation

These fixed repository files are ABSENT; Q does not create an approval:
`merlin/contract/bf16_f32acc_admission.json` and
`merlin/contract/reviews/bf16_f32acc.json`.
Coordinator installation and authentic independent reviewer attribution are
trusted-host governance premises. Hashes bind reviewed bytes; they do not
cryptographically authenticate a human. Neither matching pins alone nor a caller
human_reviewed flag grants approval.

Admission exact fields: schema=`merlin.bf16.policy.admission.v1`, domain (DOMAIN
constant), review_sha256, sources (repository relative path->SHA256), statements
(required root->elaborated statement), lean and exporter (existing receipt blocks).
Review exact fields: schema=`merlin.bf16.policy.review.v1`, decision=`approved`,
domain, nonempty reviewer, independent_of_implementation=true, nonempty record
(attributable review reference), sources_sha256, statements_sha256. Last two hash
canonical JSON of the admission maps (sorted keys, compact separators).

All required Spec/semantic-file pins and current theorem-statement pins must be
installed in SPEC_LOCK and match. `source_bindings(root)` recomputes conservative
project Lean/verification-host/parser/hook source bindings, excluding build/SDK
trees. Sources, checker, exporter, review and lock are checked again after work.
No record/Spec/lock installation or human approval is performed by this handler.

## Outcomes and exact claim

GateUnsupported: unsupported width/site scope/no-site/identity/round-trip domain.
GateRejected: pending/fake/stale approval, failed complete/local checks, missing
or reordered coverage, impermissible flags or changed checked bindings.
GateUnavailable: missing/stale tools, timeout, malformed/unknown result, audit
execution/infrastructure failure. All refuse compilation; failures/skips are not
proofs or successful semantic refusals.

Only normal return of schema `merlin.bf16.policy.gate.v1`, status=`admitted`, is
accepted. It binds exact before/after refs, domain/index64, full site roster,
required roots/std3 audits, approval records and fresh receipt. Raw draft checker
reports remain `abstracted`, policy_reviewed=false; they are never accepted alone.
The separate admitted policy claim is complete actual output equality to the
approved source-derived f32 reference under SSA/call laws, plus strict local
policy refinement on typed initialized operands. No original half-accumulating
source equivalence, native bit exactness, arbitrary fusion, whole-model success,
other-pass correctness or human approval from agent review is asserted.

F owns actual tests/build/check/replay. W owns independent MACC review. Root owns
publication and eventual human-review/pin installation. Old cuts and F14/F18 remain.
