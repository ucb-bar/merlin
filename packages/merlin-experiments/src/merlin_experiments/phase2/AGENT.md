# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase2

`campaign.py` owns frozen functional admission, exact fork checks, package sandbox
policy and completion accounting. `prompt.py` owns deterministic task contracts.
`functional_inputs.py` joins that admission to the immutable Phase 1 input/host-lane
snapshot, verifies its provenance, creates the public manifest projection, and
owns relocated grants and private mask surfaces. It is host-only code: public
projections never substitute for the original private provenance. Verified use
requires V4 ownership and an explicit fresh projection destination. Native
controllers import this owner; the package must not import those controllers.
`gsim_gate.py` owns exact-workload engine-equivalence certificate admission and
execution evidence checks, independently of the simulator-running producer.
`telemetry.py` owns Codex preflight, lossless round evidence and strict AET finalization;
`contracts.py` supplies shared byte-identity primitives and the stage refusal type.
`corpus.py` owns generated development-corpus admission, immutable copying/loading,
verification and exact measurement-cell schedules. It never imports native engines;
measured claims, certificate envelopes and model-portfolio selection remain separate.
`broker.py` owns action identities/serialization, locked admission and budget enforcement,
credential-free commands, HTTP/shim staging and append-only receipt production.
`broker_policy.py` owns explicit versioned workflow selection, action registry construction,
shared scientific services and action budget rules. `corpus_feedback.py` owns certified
GSIM feedback, redaction, stopping and mandatory final-byte receipt qualification;
`whole_model.py` owns global scientific dispatch/scope checks and its distinct receipt
qualification (a micro sweep cannot establish global evidence). Native controllers supply
real scientific dependencies and explicitly select the workflow, never by target-name inference.
These workflow policies remain scientific contracts, not interchangeable grading defaults.
`broker_evidence.py` owns shared identity checks: new receipt rows bind the selected
workflow; historical absent IDs remain explicitly unbound and are never inferred or rewritten.
The same bound workflow executes and qualifies a round. No policy imports a native
controller; the existing recursive source/private identities cover all these owners.
HTTP bodies have a five-second absolute read budget, capped by the experiment
deadline, and incomplete bodies cannot allocate an action. Headers and response
writes have an idle socket timeout; slow header trickling still prevents a claim
of bounded whole-server shutdown. Preserve receipt joining for admitted actions.
Tests must use direct execution or fake HTTP objects when listeners are prohibited.
The authoring-stage source role binds the actual executing authoring controller.
`stage_prompt.py` owns its explicit frozen prompt-input types, deterministic stage rendering
and immutable prompt-file materialization. The recursive package inventory binds that renderer
in addition to the authoring-stage role; never substitute one source for the other.
`authoring.py` owns bounded authoring rounds, sandbox/broker orchestration and candidate
sealing. Require explicit source, contract and run roots and attribution suite. New
renderer/telemetry records name this executing owner, not the native CLI wrapper;
historical native identities remain unchanged. `authoring_cli.py` owns argument parsing;
the native wrapper only translates its historical layout into explicit inputs.
`stage_inputs.py` owns frozen objective selection and prompt-input preparation. Source roots
are explicit; prompt destinations must resolve through exact frozen grants. Preserve the
existing snapshot-relative corpus layout and public-only selection until its migration is
qualified; never substitute live corpus bytes or infer a source checkout inside this owner.
`measurement_evidence.py` owns paired result identities, completion, byte-verified
measurement admission and the exact GSIM statistics projection. The paired writer,
coordinator and current report reader use this same implementation. Current admission
requires campaign v2, measurement plan v3 and result cells v2; do not silently decode
unknown schemas as current. Fresh-child adoption may establish its checkpoint hash;
replay/reporting must supply the existing parent/checkpoint hash. Parse the same bytes
that were hashed, and never substitute a later unpinned read for verified result rows.
Candidate admission uses the full `candidate_verification` owner, not this measurement module.
`measurement_support.py` owns counter-pass linking, measured/RTL identities and
resource coverage; counter-binding discovery takes an explicit target.
`gsim_workload.py` owns certificate workload derivation and byte-exact declared
output encoding. Producers and paired consumers use these same implementations;
do not duplicate them or route back through native modules. Actual engine execution
still requires declared runtime/toolchain prerequisites and separate qualification.
`paired_measurement.py` owns paired plans, execution, cache, schedule and raw-result
storage. Require an explicit contract root for execution; never discover native
resources or import a native controller. Preserve schedule-order receipts under
parallel execution and exact byte/engine/scope cache identities. Inputs still come
from the full candidate admission verifier, never from synthetic measurement receipts.
`paired_inputs.py` owns paired input admission and execution-time identity rechecks.
Functional-run roots are explicit; tuning and held-out corpora retain distinct admission
rules. `revealed_corpus.py` reads committed holdout reveals without importing qualification
producers or launching engines. Parse the manifest bytes whose digest was checked;
preserve exact membership, cohort and workload identities. `holdout_corpus.py` owns
commit/reveal generation with an explicit catalog and source-owner context. Verify
loaded origins and complete membership before reusing committed generation sources;
never attest running code using an unrelated or empty source tree. Candidate seals
must precede reveal; preserve the Phase 0 generator and exact cohort policy.
`gsim_certificate.py` owns build/capture validation, semantic oracle checks and same-ELF
certificate production; native producer paths are CLI-only. `heldout_qualification.py`
owns post-reveal lowering, qualification and completed-result admission. Supply an
explicit contract root to lowering. Runtime configuration belongs to the selected
backend's `pinned_runtime` context; independently verify both resolved engine hashes
before capture. Missing capability refuses, without target-specific environment fallback.
Preserve seal/reveal ordering, runtime cleanup, immutable baseline and exact workloads.
These owners do not establish simulator availability or hardware correctness by import.
`functional_cohort.py` owns exact live/frozen functional admission; live schemas and
frozen source roots are explicit inputs. `functional_coverage.py` owns the unchanged
certificate sampling policy. `functional_qualification.py` owns the cross-engine
qualification lifecycle. New v2 declarations bind readonly contract snapshots;
resume and completed-result admission verify them before reuse. Historical v1 roots
remain inspection-only under this implementation and must not be upgraded in place.
`qualification_policy.py` owns declaration-bound tool/harness snapshots, executable
mode and directory inventories, original mount destinations and captured private aliases.
Fresh functional qualification requires a source owner and seals this policy separately
from unchanged descriptor bytes. Resume uses only the sealed execution inputs; unfinished
roots without them require a new root. Recheck before lowering, dispatch and publication.
Completed legacy evidence remains inspectable; new completed declarations verify their
resource closure too. Preserve readonly CAS inodes and executable bits at finalization.
System mounts, host interpreter and executing Python provenance remain separate dependencies.
`statistics.py` owns the exact predeclared GSIM matrix and all-trial statistics.
`emission_diagnostics.py` owns structural command-buffer comparison, compiler-schema
validation, capability diagnostics and cost-plane interpretation. Supply schema paths,
gate phases and device registries explicitly. Validate the same schema bytes that were
hashed; do not rediscover checkout defaults or silently select a gate phase. Structural
findings are not calibrated timing rankings or measurement evidence.
`emission_analysis.py` owns whole-model emission analysis, baseline emission caching
and iteration readiness. Require an explicit contract root; never discover a checkout
or import a native controller. Missing gate policy leaves the cost plane incomplete;
missing device records leave device evidence unknown. Cache compiler emission bytes,
not host-verification authority. Host policy pins both gate and hardware declarations
alongside the complete source membership. The native functional guard still needs
frozen-corpus admission migration; this owner does not qualify the full controller.
`portfolio_resources.py` owns member analysis budgets, memory-based concurrency admission
and deterministic estimated-cost scheduling. Callers supply frozen member descriptors,
prior emission measurements and host observations; this module never launches workers.
Preserve the existing allocation/concurrency receipt schemas and scientific policy.
`static_identity.py` owns path-neutral static-cache identity and dynamic-evidence exclusion.
It verifies explicitly supplied source bytes; native callers retain snapshot admission and
provider selection. Portable bindings do not grant permission to reuse unverified evidence.
It also owns static compiler dependency closure and same-run immutable cache admission.
Require an explicit trusted source root, original sealed record hashes and ordered
portfolio identities; malformed entries are misses, not permission to use live rows.
Keep scientific reevaluation and cross-run reconstruction with their existing callers.
`static_cache.py` owns immutable bundle publication, pinned readonly decoding and
ordered member/artifact correspondence. Require explicit output and admitted checkpoint
roots and identities. Preserve no-overwrite publication and same-byte hash/parse reads.
Missing training artifacts may be recorded, but cannot be imported. Native callers
retain per-member scope/readiness checks before decoding artifacts and rebinding interfaces.
`edit_authority.py` owns frozen compiler-edit state, seed capture, exact source-pin
and guidance-subset checks, retained integrity and cumulative edit refusal records.
Use the canonical AST checker and reviewed-declaration loader; never infer approval
from candidate-provided guidance. The controller retains lifecycle/input-check timing
and separate mechanism/work-order policy. No implicit checkout or output-root discovery.
Freeze-started state and the original immutable binding/seed identity cannot be reset
through public metadata. Check receipt bytes, readonly seed membership/permissions and
independent authority identity before direct validation or guidance inspection. Failed
capture/publication requires a fresh owner and output root, never development-mode fallback.
An explicit guidance schema governs both automatic and host-declared inspection.
Freeze pins its bytes and selected path; default receipts retain their existing shape.
`mechanism_program.py` owns host mechanism catalogs, per-member work orders and their
static-analysis bindings. Attribution and work assignment grant neither edit permission
nor measurement authority. Keep graph-local operation IDs scoped to exact ordered portfolio
members. The native caller retains experiment-wide admission, verified continuation lookup
and revision publication; pass admitted records, never controller callbacks.
Work-order source preflight precedes candidate admission; decoding and publication follow it.
`mechanism_evidence.py` decodes retained checkpoint catalog/work-order receipts using
their historical policy. Original host metadata sources need not remain available;
recorded local paths, readonly receipts and exact initial analysis remain required.
Do not replace this decoder with stricter fresh-work-order admission or imply arbitrary
archive relocation. Native consumers retain their surrounding checkpoint admission.
`mechanism_rounds.py` owns round-start snapshots, attribution and final-byte receipts.
The caller admits inputs and candidate scope before capture, then snapshot scope before
publication. Compiler shared-source roots are explicit per operation. No-op/refused rounds
cannot become accepted analyses, and finalized candidate bytes cannot change afterward.
`agent_view.py` owns bounded agent-facing summaries and prompt paragraphs, preserving full
evidence pointers and exact edit-authority filtering. Availability is not admission, changed
object bytes are not speedup, and scoped measurements are not whole-model timing. Target
discovery and measured feedback production remain with callers, not the view module.
`reporting.py` owns current parent-pinned report admission and immutable JSON reads.
Report callers supply handoffs from the full candidate verifier. Reporting does not
construct substitute qualifications, launch engines, or infer missing source pins.
The native legacy Verilator reader shares these byte-read/error primitives, but its
scientific policy is separate. Native presentation remains outside this package.
Versioned telemetry source sets are exact; never fill missing historical pins from live code.
Historical preflight v1 has 20 roles; extraction v2 has 23; current explicit-price v3 has
27, including AET pricing owners. V4 has 29 roles: those 27 plus the complete Phase 2
Python membership/hash closure and its shared discovery implementation. Additions,
removals and changed bytes invalidate new execution without per-module role growth.
Decode historical closure evidence without reading live implementations; verify live
membership before execution. Historical native authoring-stage identities are never substituted.
V3/V4 use one normalized, sealed AET price snapshot for
rounds and finalization. Never mutate ambient pricing/cache state or silently reprice old evidence.
`claims/` owns measured decision procedures and historical-identity resolution.
Preserve independent measured-claim and model-portfolio policies and historical
receipt schemas. No import-time launches, implicit latest-run selection or hidden
corpus publication. This is trusted host code, withheld from candidate compilers.

`checkpoint_admission.py` owns explicit execution context, immutable preflight,
provenance and checkpoints; `checkpoint_controller.py` owns ordered campaign execution
and adoption. `checkpoint_cli.py` requires explicit resource/output/source-owner roots.
Never replace this lifecycle with native callbacks or omit functional regrading before
reveal. `paired_cli.py` owns paired result writing and forwards its explicit contract
root to execution; interruption must retain incomplete `NO_GO` evidence and propagate.
`chia_launch.py` binds exact script/module commands and its own policy bytes in v2
launch receipts; old v1 receipts are not new-execution authority. Keep the frozen
command builder and existing supervisors, not another process launcher.
`chia_envelope.py` owns assigned-resource admission, exact command/cwd/source checks,
immutable launch/completion receipts and composition of existing Chia task/native
supervisors. Its context requires interpreter, cwd, coordinator prefix, wrapper and
attribution inputs. The installed CLI selects the checkpoint module; its downstream
`--chia-wrapper` must name the installed envelope owner. Only the native wrapper
supplies historical layout. Pin owner bytes alongside wrapper, command and launch
policy. The measured catalog uses this installed envelope; stored native plans
retain their recorded commands. Synthetic routing/assignment/transport tests do
not qualify managed deployment or scientific outcomes. Pin selected Python source
membership and import environment across the worker handoff; backend runtime
configuration cannot add, remove or replace the five shared Python selection keys.
Optional backend completion selection remains at the native launch edge. Package-owned
modules cannot import legacy controllers. New native snapshots include this
package through the existing optional-source inventory; never rewrite old seals.
`revision_journal.py` owns chronological static iteration publication and retained
artifact/sandbox state for that controller. Distinguish fresh analysis, same-run
reuse and imported seeds; admission and scientific gates remain at their callers.
Original static-record hashes must survive later in-memory probe evidence updates.
The journal also owns retained artifact correspondence and ordered portfolio member
contexts: rehash emitted bytes, bind source/plan/compiler identities, and check source
containment. `revision_session.py` owns live candidate admission and action-local
revision access before using the journal. Optimization-baseline
bindings establish artifact identity only, never numerical proof or a verified global plan.
The session composes concrete input/edit/mechanism/evaluator owners and publishes
ready or blocked checkpoints. Nested synchronous actions may share a thread-local
admitted row; callers must refresh after external execution. Never reuse this state
across actions or threads. Checkpoint publication must retain the journal's original
static-record identity, not repin changed bytes or linked replacements. Later probe
extensions to the in-memory row do not rewrite the original static iteration file.
`development_feedback.py` owns tuning-feedback preparation, its loop and the paired
measurement adapter. Preparation requires an explicit contract resource root and binds
it into the installed executor; callers may still supply a synthetic executor directly.
`agent_workspace.py` owns answer-free input snapshots, verification, fresh candidate
copies and both agent/tool sandbox policies. Consume the canonical answer-surface
registry and frozen functional grants. Policy tests use synthetic probes, not live
sandbox/tool launches; available network is not a network-isolation guarantee.
Capture one explicit toolchain selection for policy construction, required probes
and broker execution. Installed authoring requires that selection; launch edges
alone may select legacy defaults. Broker commands must not rediscover environment
or working-directory defaults. Frozen selections are reverified before execution;
outer agent policies do not inherit inner toolchain grants or execution authority.
Reject exposing overlays that shadow frozen tool destinations; preserved earlier
mount declarations alone do not prove the final mounted bytes are unchanged.
`feedback_metrics.py`, `calibration.py` and `capsule_verdict.py` share workload pricing,
measured calibration and per-capsule verdicts with retained native consumers. Preserve
scientific thresholds, budgets, redaction and stopping semantics; missing observations
must not become successful or zero-cost measurements. Use synthetic executors for tests.
`transcript_audit.py` owns native and translated agent command-event auditing,
broker invocation accounting and answer/direct-execution refusal. Native authoring
and requalification import the same auditor. Frozen replay supplies its original
audit tokens explicitly; never replace them with current target-derived tokens.
The audit is evidence checking, not a shell sandbox or a replacement for mount
isolation. Its implementation remains private grader code under this package's
existing recursive access and source-membership policies.
`candidate_record.py` owns record/formal-claim validation; `candidate_verification.py`
owns byte verification, handoffs and audit-only requalification. New requalification
v4 pins and snapshots the complete Phase 2 Python membership plus its discovery
implementation, auditor, answer-surface and shared-access policies. Historical v1-v3
records retain their old roles and are inspection-only under the extracted live
implementation. Never relabel a native-controller hash as installed verification
or replace archived authoring identities with current implementation identities.
Global host-verification reuse pins the same complete package membership and the
discovery helper through its existing flat source map. Keep portable and located
policy identities sensitive to member additions, removals and byte changes.
`host_policy.py` owns V3 logical module/resource/controller identities and source capture.
Resolve active installed modules and verify loaded origins without import-all discovery.
Historical admission checks recorded bytes and complete closure membership inside an
independently admitted source snapshot, never current imports. V1/V2 retain historical
decoders; version changes are cache misses, not normalized equality. V3 inventories all
active Phase 2 and performance namespace roots, refusing duplicate logical ownership.
Preserve V2's fixed role roster and hashes; never rewrite historical meaning. Explicit live
resource selection does not grant arbitrary external inputs authority during frozen reuse.
`portfolio_checkpoint.py` owns retained global candidate and blocked-authoring
checkpoint admission. Supply the current trusted host policy and compiler shared-source
root explicitly; never discover a checkout or import a native controller. Its shared
portfolio selection and paired-feedback rules serve live actions and independently
verified evidence. Historical references remain engine-relative, semantic qualification
remains scoped to the selected region, and no checkpoint grants full-model speedup.
Archived dispatch stays with its transport owner and executes its original verifier.
`portfolio_evaluation.py` owns the optional fast analytical provider, its frozen
configuration/binding, integrity checks and serialized bounded calls. Use the core
portfolio policy for quality and Pareto decisions; do not duplicate its algorithms.
The caller retains experiment-wide admission and supplies explicit target/portfolio
identities and edit authority. Missing or failed evaluation cannot authorize approximate
transformations. Waiting for the provider lock counts against its observed wall budget.
`global_inputs.py` owns frozen global-experiment inputs and their ordered integrity
checks. Preparation admits sources without creating experiment output; materialization
retains historical evidence and the optional comparison compiler before exposing the
usable owner. Supply contract resources, compiler shared sources and controller identity
explicitly. Verification composes the concrete edit, mechanism and evaluator owners;
never replace them with native-controller callbacks. Preserve the distinction between
the qualified Phase-1 compiler and an unqualified optimization comparison seed.
`portfolio_analysis.py` owns serialized portfolio analysis, shared deadlines, concurrency
admission, same-run reuse, functional gates and scientific candidate selection. Compose
the concrete revision session; do not duplicate its input or authority state. Lock waiting
counts against the analysis budget. Duplicate current revisions do not create iterations;
revisited revisions create new chronology without inheriting probe evidence. Cross-run
admission remains separate. Baseline cache seeding belongs to `emission_analysis.py` and
confers no host-verification authority. Synthetic tests do not qualify native execution.
`static_analysis_import.py` owns one-shot cross-run admission and imported publication.
Compose portfolio analysis and its session, with explicit historical shared-source path
selection from the caller. Verify the source snapshot before selecting its shared root;
never execute archived verifier code or discover a live checkout. Consume the attempt
before admission, including misses and failures. Recompute readiness, rebind current
interfaces and reconstruct current sandbox policies without inheriting dynamic evidence.
`portfolio_probes.py` owns admitted witness compilation, native-policy reuse, scoped
qualification and measurement, preparation accounting and probe receipt publication.
Providers receive the concrete probe owner, not a native controller. Access revision
authorities through its analysis session; never duplicate state or relax retained sandbox
and dependency checks. Failed work consumes budget. Refresh immutable inputs after an
outer execution action and revalidate nested actions. Scoped evidence cannot establish
whole-model numerical equivalence or speedup. Tests use synthetic execution only.
`global_experiment.py` composes the concrete input, authority, journal, analysis,
probe and import owners. Its constructor requires explicit source/resource selections;
the native adapter alone supplies compatibility defaults. Preserve input preflight before
output creation and authority/materialization/publication order. Controller identity is an
explicit provenance input, never silently relabeled during migration. Optional guidance
selection retains the difference between legacy omission and an explicitly selected schema.
Installed assembly does not establish complete authoring, sandbox or hardware execution.
`portfolio_sandbox.py` owns compiler policy assembly, dependency overlays, exact
cached-policy rebinding and analysis-worker installation. Its concrete analysis
owner supplies source/resource/provenance inputs, and tool selection is explicit.
Never rediscover a native controller or accept callbacks to construct policy.
Cached records returned to callers are detached; reuse rechecks dependency and
overlay bytes, and frozen policies retain tool-shadowing and answer-mask checks.
The native launch edge still selects its optional completion-contract adapter.
`portfolio_authoring.py` owns global rounds and budgeted checkpoint continuation.
Compose the concrete experiment, explicit sandbox inputs and declared instruction evidence;
do not pass a native round callback or rediscover target configuration. Consume checkpoints
against freshly constructed host policy using the experiment's original controller identity.
Preserve ready-member regression checks, reserved budgets, narrow failure recovery and
independent post-authoring validation. Synthetic continuation tests do not establish successful
native transport, full Phase 0–2 execution or deployment isolation.

`portfolio_options.py` owns portfolio CLI declarations, ordered admission and frozen-worker
argument serialization. Serialize the same declared interface, including defaults and ordered
portfolio members; resolve input paths before changing source roots. Unsupported actions or
foreign namespace fields refuse instead of being silently omitted. This does not select an
executable or replace process supervision. `fast_evaluation_installation.py` owns pinned
host analytical-provider installation, exact-only fallback and installation receipts. Execute
the exact hashed observer source bytes, not cached bytecode; recheck corpus and observer pins
on use. These owners do not establish full installed portfolio launch or target execution.

`portfolio_launch.py` owns source freezing, resource admission, transport receipts and
the retained worker supervisor. Deployment supplies all source/output/lease roots,
provider selection, declared inputs and command/import environment explicitly. Native
layout discovery stays at the launch edge; no callbacks recreate that dependency in
the package. Preserve lease/refusal/freeze/receipt/execution/release ordering, and close
leases on failure. Fake-process tests do not establish managed-host cleanup or a fully
installed worker-assembly route.

`portfolio_cli.py` decodes explicit deployment ownership and joins the installed launcher
to the frozen worker. Hash the same deployment/campaign/descriptor bytes that were admitted;
require the copied snapshot to match before transport. Child execution requires the active
sealed Python identity, archived declarations and a pinned retained sandbox policy. Never
infer source roots or providers from checkout layout. Direct CLI destinations must not
overlap immutable inputs; the common output root is a container, not a writable subtree.
Admission tests do not establish full optimization or deployment isolation.

`portfolio_providers.py` assembles optional scientific providers from explicit backend
capabilities, never target-derived sibling names. Missing capabilities are unavailable;
malformed declarations and import failures refuse. An explicitly requested isolated probe
requires a complete primitive capability. Preserve module identity for adapter-source
receipts, construction order, semantic-only behavior and operand-movement admission.

`portfolio_worker.py` owns full frozen worker assembly, analysis/validation/authoring
and terminal receipts. Its context supplies source, functional-run, contract, compiler,
controller, historical-layout and sandbox inputs. Verify target descriptor identity before
output creation; use the installed experiment directly, never native callbacks or imports.
Keep native controller attribution for native calls. Completion-contract capabilities are
optional only when absent; declared malformed or failed capabilities refuse. Do not infer
complete deployment or scientific qualification from helper and admission tests.
