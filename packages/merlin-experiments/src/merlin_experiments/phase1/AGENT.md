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
