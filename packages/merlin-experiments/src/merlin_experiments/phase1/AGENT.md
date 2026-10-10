# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1

`conformance.py` owns transcript-derived treatment conformance and replay, not core
capsule requirement derivation or numerical grading. Preserve historical tool-set
applicability, persisted-result evidence, vendored-byte attribution and fail-closed
scanner behavior. It is grader-only; the native path is a CLI adapter, not an import API.

`context.py` owns explicit invocation context and descriptor/environment initialization.
Import must not discover a target, run git, source experiment.env or mutate process state.
Installed callers supply descriptor and repository/work root. Native defaults and permissive
legacy target-name fallback remain exclusively in the checkout `_common` adapter.

`options.py` owns the one functional-loop CLI parser and typed invocation values.
Defaults are computed at invocation, not import; parsing/help never initializes a target.
Keep scheduling defaults and policy validation separate from mutable runtime session state.

`controller.py` composes real admission, task staging, workspace transport and authoring under
one workspace lease and one restored environment. Native launchers select legacy defaults and
call this same owner; they must not retain a second execution composition. Initialize invocation
context before execution imports. Installed callers provide operator paths explicitly.
`__main__.py` is the installed command using the same option parser and controller.
Keep parsing/help target-inert; require explicit operator inputs instead of native defaults.
`workspace_transport.py` owns frozen friendly views, copy grants and mask probes. Resolve grants
through the shared resolver and inspect frozen membership for frozen views. Missing probe
completion, failed execution or unreadable public controls are not successful isolation evidence.
Copy mode is diagnostic, not OS isolation; its resume still verifies source/task/treatment
receipts but must not require a bwrap snapshot it never created.

`runtime_environment.py` prepares explicit host environment/account inputs without mutation.
Native compatibility library directories are supplied by the edge, never guessed here.
Apply the complete prepared baseline across admission and authoring, restoring the exact prior
environment on exit, including continuation changes; concurrent experiments require separate processes. Account probes receive that same
environment. Do not serialize credentials, mutate grading policy, or claim another Python ABI
is compatible merely because active-interpreter package paths are present.

`task_staging.py` owns actual task composition, runtime-scope prose and selected-bundle tool
resolution. Immutable invocation configuration produces inventoried function callbacks; retain
admission's observation order and reread tool declarations after trusted task staging. Authored
tasks and bundles are operator inputs, not files located relative to this module. Runtime language
is captured explicitly. Do not regenerate sealed task bytes during resume/provider launch.

`recovery.py` owns transcript interruption detection, quota/reset interpretation and
checkpoint/resume guidance shared by authoring and reporting. Preserve classification
precedence, no-tool-work guards, exact status/banner bytes and historical exit codes.
It never launches providers or decides numerical grades. Native recovery is CLI-only.

`session.py` owns fresh/resumed admission, source/treatment checks, private snapshot views,
the existing environment record and pre-authoring refusal gates. `PreparedRun` carries
these inputs into `authoring.execute` for authoring/completion. Keep the caller's existing
`workspace_session` wrapper alive across both preparation and continuation: ordinary returns
release its lease, uncertain exceptions retain it. Native provider/account initialization and
candidate-view transport/task composition remain explicit adapters, not package imports.
Resolve tool declarations after trusted task staging, at the original observation point.
Callback source owners must belong to the existing inventory; this does not freeze closure
state, import graphs or external tools. This boundary alone is not the installed full engine.
Bundle manifests use the canonical `input_bundle_manifest.yaml` filename; alternate names
are refused before mutation rather than reading one declaration and freezing another.
Fresh repository attribution uses the explicit context root (not an unrelated ambient root);
resume preserves the original environment record, including its repository identity.
An optional invocation readback policy is recorded separately from capsule bytes and
is immutable during resume and formal grading. Context arguments carry that same
selection to trusted grading children; absent selection preserves legacy calls.
The transport changes output serialization, never numerical tolerances or host placement.

`run_inputs.py` owns private seed/errata/treatment and frozen-input verification helpers.
It is a grader identity; context itself is not. Preserve original observation order,
failure behavior and archived identities. Native initialization and admission adapters
remain; these modules alone do not establish a standalone installed phase-1 engine.

`authoring.py` owns the actual round/checkpoint/resume and completion continuation.
`AuthoringRuntime` supplies the authored bundle identity, timing path and captured legacy
public-root override. Ordinary rounds and L3 repair share the same state and authority;
do not replace them with disconnected callback records. Public-root derivation stays lazy,
after submission/language refusal gates. `feedback/certification.py` owns cert eligibility,
promotion/tally, measured size-scaled budgets and official-evidence checks, not an agent loop.
Supplemental integrity markers are immutable `GradingInputs` chosen only from trusted
invocation options. Propagate them through ordinary/fast QA, treatment callbacks, shape
probes and in-process L3 repairs; never mutate core scanner defaults, weaken AST restrictions,
read policy from candidates or silently extend fresh official/selfcheck child policy.

`audit.py` owns `AnswerAudit`: immutable descriptor-derived tokens plus an explicit absolute
bundle directory. Grant files are reread at their historical observation points. Importing
these owners never selects a target or imports native controllers. All are host-private,
bound by the recursive Phase 1 source inventory. The five retained host text-audit regex
calls have reviewed local rationales; moving them is not parser-hardening qualification.

`providers/` owns trusted host agent transports and provider routing, not candidate SDKs.
The native controller injects its sandbox command; providers must not import the controller.
Provider discovery stays target-inert. The recursive implementation source inventory pins
these modules, including the standalone resource sampler.

`treatments.py` owns invocation-local callback selection and attribution for the native
fullsuite/RTL adapters. Callback owners must belong to the existing implementation source
inventory; record references in the native environment receipt, never a second source seal.
Keep public-root overrides separate from descriptor-derived hidden/task scope. RTL feedback
is advisory both in round QA and certification repair; timing observers must not replace
controller globals or accumulate across invocations.

`corpus_inputs.py` owns private staging and effective bundle declarations for one native
run's public grading, descriptor-policy and schema views. Preserve those distinct policies:
fullsuite's public override must not redefine task scope or promotion eligibility. The existing
native snapshot is the byte authority; environment records refer to it, never a second seal.
Resume reads archived effective declarations and frozen views, with no live-corpus fallback.
Do not grant staging or snapshot private views to candidates, or put their bytes in public CAS.

`source_inputs.py` owns one implementation inventory/fingerprint shared by the native
controller and outer runner. Native records live in the existing environment receipt;
resume refuses missing historical attribution rather than minting replacement evidence.
The explicitly broad interim closure includes retained native harness Python, extracted
Phase 1 sources/providers, complete cross-phase corpus and managed execution Python packages, public clients
and declared startup owners. It is not all core
or upstream dependencies, and source-attribution checks are not a frozen Python sandbox.
Explicit descriptor-target support adds its complete Python membership, provider
identity and contract to the same inventory. Loaded-plugin ownership is rechecked;
changed selection, members or bytes refuse resume. This binds the live selected
provider, not an automatically copied or relocated Phase 1 execution environment.

`brokers/` owns ISA/CCA and synchronous/asynchronous feedback tools selected by the
core registry. They run as canonical modules with explicit invocation context;
old native entrypoints are CLI-only launchers, never private-helper import aliases.
Public clients remain byte-identical standalone files.

`feedback/` owns trusted selfcheck execution, QA redaction, promotion and private
enqueue-time snapshot creation/recovery. `dispatch` is the one async simulator
allowlist/certification-token policy; sync request policy remains deliberately
distinct. Public grading roots and descriptor-policy roots are separate explicit
inputs. Context and schema roots do not freeze independently loaded target/toolchain
or grading-policy resources. This extraction does not qualify the full controller.

`oot_history.py` owns the run's harness-written compiler history at `<run_dir>/oot`
(`merlin.common.oot_repo`). Loop grading commits the operator-only snapshot it is about to grade,
once per graded round, and round records carry that commit sha; the official freeze tags `frozen`
on the exact submission it hashed. The agent never writes the repo, and it is refused inside the
agent's writable workspace. Only bwrap runs keep it (copy mode is a tool-free diagnostic). New runs
live at `out/runs/<target>/phase1/<run-id>/`; a run that began under the legacy
`capsule-bench/<arm>/` root resumes there and keeps no history. A failed commit is recorded in
`oot_commits.jsonl` / `freeze.json`, never silent.

`component_origin.py` owns experiment-only fresh OOT origin. Generate the inert
scaffold inside the actual normal authoring lifecycle; never accept a supplied
working compiler or mint origin from a receipt. Independently issued hardware
and protected minimal software intakes, reviewed generic library and exact
public runtime grants remain separate. The minimal software authority must share
the exact live hardware origin, coverage software-intake binding and complete
public software projection. Reviewed labels or historical source hashes cannot
grant legacy software specifications access to a fresh author session.
`component_generation_admission.py` requires actual budgeted Phase 0 coverage
v2 and the normal reader's complete source-cost, membership and policy replay.
Fresh input identity binds the exact budget and admission ledger commitments.
Legacy v1 remains inspection-only for this bounded-generation claim; no JSON
cost or resigned hash replaces actual replay. These logical reference bounds
do not qualify compiler execution limits, large compile-only legality, process
heap, physical runtime effects or timing.
Phase 1 shared-tool preflight cannot require the initial inert candidate to work.
`component_lineage.py` binds actual Phase 2 author transport and broker receipts
to descendants of the exact qualified fresh baseline. Qualification consumes
these issued objects before invoking candidate code.
`component_compile_admission.py` requires the exact live original source-only
roster, sharing the same hardware, software and descriptor selection, before a
fresh public author grant. Guard and withheld transfer members are both required.
`component_compile_roles.py` evaluates every original source through ordinary
scoped package lowering, translation, object/link and whole-ELF instruction
policy. No tensor/golden allocation, numerical execution or simplified compiler
route is permitted. Keep compilation and all original static obligations in
separate mandatory denominators. Reopen complete candidate/private clone,
source/ABI, contract, actual transport products and invocation records. Static
obligations remain UNKNOWN without independent proof producers; successful
linking cannot discharge them or qualify the compiler. Functional qualification
must consume this exact live evaluation and retain its unresolved obligations,
including expected static refusals, even when small numerical execution passes.
`component_pointer_storage` freezes an optional independently selected original
software pointer policy, exact live source roster and public policy projection
before authoring. `component_copy_proof` admits only the fixed core counted-copy
checker with actual original lowering, stock translation, unchanged object/link
and selected native layout joins. Reopening rederives every proved/refuted facet
from the exact artifacts. Unsupported original members and resource/numerical,
physical/lifetime/timing roles stay UNKNOWN in the full original denominator.
Saved flags, compiler metadata and arbitrary proof callbacks supply no authority.
`component_qualification_evidence.py` reopens the exact private compiler copy,
complete actual invocation membership and every recorded dependency/product,
including files outside the grade tree. Each mandatory member needs observed
invocations. Replay the original complete stage, source, output and effect joins;
unchanged result rows or grade-tree bytes cannot replace that check. Retain
failed attempts as refused evidence, never mint authority from saved records.
`component_qualification_domain.py` selects the original input owner from the
exact issued compiler origin. Legacy qualification retains concrete coverage
and receipt v1. An explicitly selected live `SourcePreparation` must replay all
original source requirements before grading; receipt v2 binds their complete
denominator and pending candidate predicates. Every mandatory member, including
mandatory development, needs fresh numerical/executable/stage evidence and
actual invocation reopening. Source completion never grants candidate static,
numerical, effect, hardware or runtime authority; source-only guards remain
unestablished for Phase 2. Missing source premises refuse before the grader.
The `source-preparation-qualification` installed suite retains actual native
clone/dependency/output controls and preparation refusals. Its positive wiring
facets use synthetic author/runtime/static authority and cannot admit an
experiment. An explicit native selection requires zero skips in its full roster.
Neither origin nor lineage proves
numeric correctness, target runtime independence or performance. The handwritten
implementation and its adapter remain protected final reference inputs only.

Shared-tool readiness selects the existing `strict_tool_policy` with
`candidate_writable=False`; do not rewrite a constructed mount argument list.
Native client readiness keeps its disposable copy for temporary mountpoints and
read-only original source members, then rechecks complete source membership.

`component_source_applicability` treats rank-zero tensors as one logical element
under the registered tensor SSA contract. This source-only observation grants no
native ABI, numerical, tail/resource, physical ownership or runtime effects.
The `original-pointwise-host` installed suite retains ordinary source factory,
upstream conversion and complete host value controls with explicit tool selection.
