---
title: "Component-only Phase 2 feedback and launch boundaries"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-08
related: [beam_cca_architecture]
code_refs:
  - src/merlin/perf/component_cost.py
  - src/merlin/perf/component_screen.py
  - src/merlin/perf/analytical_resources.py
  - src/merlin_experiments/phase2/component_analytical.py
  - src/merlin_experiments/phase2/component_screening.py
  - src/merlin_experiments/phase2/supervised_feedback.py
  - src/merlin_experiments/phase2/component_workflow.py
  - src/merlin_experiments/phase2/component_cca.py
  - src/merlin_experiments/phase2/broker_policy.py
  - src/merlin_experiments/phase2/stage_inputs.py
  - src/merlin_experiments/phase2/stage_prompt.py
  - src/merlin_experiments/phase2/authoring.py
  - src/merlin_experiments/phase2/authoring_cli.py
  - packages/merlin-experiments/tests/test_component_workflow.py
  - packages/merlin-experiments/tests/test_component_cca.py
  - src/merlin_experiments/phase0/component_generation.py
  - src/merlin_experiments/phase0/generation.py
  - src/merlin_experiments/phase2/corpus.py
  - packages/merlin-experiments/tests/test_component_generation.py
---

# Component-only Phase 2

`component-only-v1` is an explicit scientific action profile for improving a
compiler library using generated development components. Historical
`corpus-feedback-v1` and `whole-model-v1` keep their existing identities, required
actions and receipt rules. A historical receipt is never relabelled as a new
component experiment.

The new registry includes candidate manifest commands, a compiler edit-surface
inventory, optional command-buffer structural analysis, optional complete CCA
feedback, optional analytical feedback and optional component RTL feedback. It excludes target-descriptor shell
probes, complete-model graph/analysis actions and all legacy global context,
source-pair and mechanism-probe providers. Passing those services to the policy
is an admission error. Registry inspection without providers advertises the
feedback tiers as unavailable.

## Independent Phase 0 inputs

The ordinary `phase0.generate_target(..., component_only=True)` API and
`python -m merlin_experiments.phase0 --component-only` reuse the existing sweep
expansion, builders, independent golden engines and written-program admission.
The normal `capsule_derivation` adapter forwards the explicit option. This mode
requires a fresh output directory, explicit descriptor/recipe/shared template,
and `evidence_mode: diagnostic`. The descriptor must contain no workload,
claim-boundary or grading selectors. Captures, conformance/application sidecars,
synthesis/solver/hidden sidecars, old evidence bundles and legacy profile-directory
discovery are refused. The public recipe supplies numerical/target choices;
all component members come from the shared independent sweeps.

Every selected shared template must itself be independent. A template containing
`requires_form_scope` is refused in this mode; its capture-derived families are
not silently removed. Reusing the legacy shared template therefore requires a
separately reviewed independent template selection. Existing direct builders
and their numerical engines are not a generated domain-coverage certificate.
In particular, the interface, generated operand bytes and golden output extents
must agree at each concrete sweep point, including rectangular contractions,
tails and the separate row extents of resident-weight consumers. Bare extents
resolved by a sweep take precedence over tile-count aliases; an omitted attention
key length preserves the query-length square default.

The selected reviewed software spec may declare non-MAC performance objectives:

```yaml
component_performance:
  schema: merlin.component_performance.v1
  status: reviewed
  hardware:
    contract_sha256: <selected EvidenceSelection.derivation_identity contract digest>
    raw_facts_sha256: <selected raw hardware facts byte digest>
  objectives:
    - family: <declared shared sweep family>
      operations: [<reviewed software operation declaration id>]
      objective:
        metric: complete_component_cycles
        unit: cycles
        direction: min
        basis: all preparation, device work, readout and output publication
```

The declaration explicitly commits to the selected HW contract/facts; its source
bytes and reviewed operation owners supply objective provenance. The generator
records the actual recipe, template and generator-owner source bytes, and binds
the resulting declaration identity to every generated member. Sweeps cannot
author their own generated identity or objective provenance. A named objective
owner must declare the generated operation/family. Every emitted program still
needs the existing concrete reviewed placement screen. Unknown work remains
refused; a real zero-MAC member needs its explicit typed objective.

`corpus.performance_objectives(live_or_frozen_corpus)` returns the checked
name-to-`PerformanceObjective` map for the existing `phase_policy.split_report`
and `anchors` APIs. Corpus selection, discovery and freezing preserve these
bindings. Objective admission does not supply measured cost, cycle certification,
utilization, an anchor or a target rate. Absent objectives retain the legacy
zero-MAC refusal. The recorded source digests are host-issued selection metadata,
not cryptographic proof of review authority or a fresh source-closure check when
a frozen corpus is later used. Fresh runtime/library isolation is separate.

Global application coverage remains diagnostic/unknown. Since this input mode
does not supply application conformance or a Phase 1 certification, its guard
obligation is explicitly `not_established`, with no functional guard or candidate
numerical acceptance inferred. Legacy generation keeps its original coverage and
guard-link gates. Existing generated goldens, graders, component certification
and final hidden/whole-model gates retain their independent roles. No fresh
experimental agent launch, hardware measurement or final convergence follows
from generating and freezing this corpus.

## Host-owned selection

The trusted host calls `broker_policy.select_workflow(COMPONENT_ONLY_V1, ...)`
with a verified `FrozenPerformanceCorpus`, candidate root, target descriptor,
receipt destination and explicit providers. The corpus is admitted through the
existing generated development-corpus path and retains its exact manifest,
generation and source identities. The component policy rechecks capsule source
bytes and descriptors and rejects explicit model-kind members and forged
in-memory descriptor changes. This is component input admission; it does not
certify Phase 0 derivation or approve an agent-visible filesystem view.

Candidate metadata cannot select a provider or grant a host capability. Resolve
any provider descriptor at the trusted launch edge, before creating the policy.
Source file pins bind the executing provider owner, not its entire transitive
dependency graph. Compiler-library/runtime import admission must separately bind
dependencies and resources. No implicit current-checkout or installed-target
fallback is authorized by this profile.

An active analytical or CCA policy also snapshots its selected callable and Python code
object. Replacing that callable with another function in the same pinned file or
changing its code object refuses execution. This guards active selection drift;
it does not prove arbitrary closure state or dynamic dependency integrity. The
normal fresh runtime admission remains a separate prerequisite.

## Fast and measured tiers

`ComponentAnalyticalProvider` binds an explicit callable's source file and an
adapter accepted by `prepare_phase2_calibration`. Each use revalidates the adapter,
its referenced controlled evidence and the exact resulting calibration document.
The calibration must name the current target descriptor digest. The callback
receives the sealed candidate, frozen generated corpus and remaining tool budget.
It returns exactly one baseline/candidate typed `CycleInterval` pair per member.
Resolved intervals require finite bounds and provenance. Missing costs remain
unresolved with nonempty reasons. Analytical rows are estimates; this schema
does not certify the callback's model accuracy or claim measured cycles.

When the selected provider supplies complete-cost reports and screening, the
broker also returns its conservative experiment ordering. `component_screening`
replays savings, latency-weighted priorities, gate observations and family/regime
diversity against the same exact public member roster. Importance uses equal
generated-family shares, then equal shares within a family; no validation-model
weights enter it. Unknowns, ties, regressions and refusals remain in the evidence.
Legacy feedback without screening keeps its existing scope. This projection
does not launch experiments, qualify a cost model or grant final acceptance.

`component-rtl-feedback` accepts the existing `DevelopmentGsimFeedback` evaluator
with its explicit host executor, exact frozen corpus, baseline compiler, target,
RTL configuration and workload-equivalence certificate. These identities and
the executor source are rechecked. The existing certificate/evaluation gate and
redacted feedback validator remain unchanged. This optional action is limited
to two invocations per broker round. A skipped member retains the existing
unmeasured status and null cycle fields. No fabricated measurements fill gaps.

Neither tier requires a GSIM measurement on every iteration. Neither grants final
accuracy, hardware parity or convergence acceptance. Spike functional evidence
cannot be presented as RTL or FPGA timing. Normal analytical and CCA actions run
through `supervised_feedback.bounded_feedback`, which forks the selected trusted
callback into a private worker and uses the existing portfolio process supervisor
to kill its complete tree at the wall deadline. The worker stays alive after
publishing its result until supervision enumerates its descendants, so nested tools
that opened new sessions are also cancelled. This host deadline does not replace
candidate sandbox admission or a separately qualified native worker service.

The public `component_analytical.build_component_analytical_provider` binds the
frozen baseline, exact corpus, descriptor, controlled calibration adapter,
feature-provider source and declared dependencies. It gives the selected target
feature provider one generated member at a time. The provider returns typed
`ComponentFeatureObservation` values for the two compiler arms, including emitted
executable, full dependencies, inputs and re-openable artifact identities.
Instruction decoding and hardware facts remain target-owned. No model graph,
protected complete-model observer or model sentinel enters this route.

`ComponentCostScope` requires explicit timer, accuracy and input-policy identities.
Both cold and warm observations cover preparation, allocation, packing, proof,
device work, readout, reconstruction, replay, dispatch, publication and cleanup.
Inactive stages require observed zero feature counts. Inclusive parents replace
contained regions when composing the total; every region still appears in the
report. Missing stages, unpriced features, unqualified feature domains and failed
functional evidence keep totals UNKNOWN. Region composition reuses the historical
analytical resource primitive with its explicit evidence-selected overlap operator;
separate ordered regions are summed. Public report replay checks containment and
total arithmetic without publishing private feature contexts or artifact paths.

`component_screen.qualify_component_screen` preserves the historical empirical
validator and adds a preregistered independent gate: 95% agreement on decided
pairs, 100 decided pairs overall, 20 per slice in at least three slices, at least
20 held predictions, at most 10% relative error and at least 95% interval coverage.
Complete workload groups are held out together. A qualification file binds the
controlled calibration and policy identities. Missing qualification leaves totals
UNKNOWN. The gate authorizes screening only.

Feature extraction uses a private immutable candidate copy. Parallel workers are
capped by assigned CPUs, available memory, an explicit per-worker memory grant and
shared engine capacity, with the existing host lease preventing overlapping
campaigns. The call retains the 600-second development simulation ceiling. Private
caches bind source, input, scope, calibration, qualification and dependency bytes;
cache hits re-open the original artifacts. Private result records retain extraction
cost, unresolved proposals, ties and regressions. Conservative saved complete cost
per evaluation second determines opportunity order; equal shares per generated
family and per member within family provide the stated independent corpus basis,
with one representative per family/regime before additional opportunities.

## Complete structural CCA feedback

The trusted host may explicitly supply a `ComponentCCAProvider` and select
`component-cca-feedback` through the same broker registry. Its experimental
baseline must be the independently issued output of the fresh Phase 1 author
session, subsequently qualified against the exact independent coverage domain.
The handwritten compiler is only the private final golden: its code, schedules,
CCA structure and measurement helpers cannot supply the Phase 1 or Phase 2
compiler or baseline. Candidate metadata cannot select the baseline or provider.
The callback must supply exactly one typed baseline/candidate
`ComponentCCAObservation` pair for every frozen generated member. Each observation
binds the actual emitted artifact bytes, compiler revision, generated member,
corpus and target descriptor. Provider source, executing callable/code selection,
baseline tree, candidate tree, corpus and descriptor are rechecked around the
call. Unknown callback dependencies or closure state still need separate trusted
runtime admission. The callback is responsible for a faithful source-bound lift;
matching hashes alone do not establish that the lift is scientifically correct.

The v2 CCA report also reflects provider-supplied users, effects, representations,
lifetimes, resource pressure, emitted choices, refusal reasons and editable
compiler owners. Absent facts remain UNKNOWN, including facts missing in both
arms. Historical v1 report replay retains its original field roster and meaning.

`component_cca.complete_report` uses the core all-facet `cca_compare.compare`
and `uncomparable_axes`. It reflects every facet field and refuses differing
operation, backend or scope membership. Populated unequal axes must be present
in the comparator's gaps. A missing side remains explicitly uncomparable; fields
missing on both sides remain UNKNOWN, including absent communication and coverage
facets. The existing register-block inference can appear as a structural hint
while that field is still UNKNOWN. Agreement on a subset never establishes full
agreement. Program scope describes the generated component program, not an
application: application coverage and calibrated or measured cycles remain UNKNOWN.

Only typed structural facet values and digest-projected provenance reach broker
stdout. Private artifact paths, raw lifting metadata, baseline programs and
source text are not returned. The callback executes on the trusted host with
private input access; this action does not grant that access to a participant.
The canonical private receipt binds the same stdout bytes and feedback envelope
as the other component actions. Its validator reconstructs the complete report
from both projected CCAs, rejecting omitted or changed facets, gaps, UNKNOWN
axes and input bindings. Archived report replay does not independently re-lift
the artifacts or re-admit the original corpus; it verifies internal evidence
and receipt consistency. Active policy admission owns exact corpus membership.

Synthetic broker and installed-package checks qualify these interfaces and
refusals. They do not qualify a hardware lifter, hidden oracle isolation, fresh
participant transport, a complete optimization campaign or measured convergence.
Normal component authoring remains refused until its separate launch prerequisites
are qualified. Structural feedback retains `NO_FINAL_ACCEPTANCE`.

## Prompt and normal launch

`stage_inputs.prepare_component_prompt_inputs` binds the selected policy to
explicit paths inside a separately approved materialized view. It calls no
complete-model selector and inherits no functional bundle or host-lane grant.
`stage_prompt.render_component_prompt` produces only the component declaration,
available actions, exact member identities and budgets. These functions prepare
input; they do not approve the view or start an agent.

Normal `authoring.run_stage` and `authoring_cli --workflow component-only-v1`
currently refuse before descriptor loading, executable lookup, telemetry,
legacy GSIM flags or model selection. There is no qualified fresh-session,
network-restricted authoring transport in this owner. The future launch edge
must verify an approved minimal view, content-bound runtime isolation and a
zero-history participant session, then create the explicit providers and broker.
A library already developed by an implementation agent is not evidence of a
fresh participant's convergence.

The closed launch declaration is data, not an admission credential. Its former
automatic target backend factory is quarantined: that route could select the
handwritten companion's compiler helpers and runtime renderer. It now refuses
before provider resolution, candidate qualification or paid client launch.
Positive assembly requires independently issued fresh Phase 1 compiler origin
and hardware-only runtime support derived from the selected RTL and minimal
software contract. Installed or reference-selected backend code, source hashes,
role flags and archived JSON receipts do not establish those authorities.

## Receipts and final validation

New component broker rows require an explicit profile identity. The qualifier
joins action/binding digests to the transcript, validates canonical private
feedback bytes, checks stdout identity and the action's evidence tier, and
requires successful candidate manifest commands. Final component feedback must
bind the sealed candidate digest when that check is requested. Its result still
states `final_acceptance: NOT_ESTABLISHED`.

Final unseen-component, numeric, complete-composition, compiler delivery and
whole-model hardware gates belong to the trusted final evaluator. They must use
the original fixed accuracy and performance contracts after the participant is
sealed, without exposing answers through inner-loop services. This owner does
not run those gates, change their thresholds, submit FireSim jobs, or certify a
complete-model speedup.

The command-buffer structural action and complete CCA action remain separate:
neither substitutes a partial agreement predicate for the all-facet report or
establishes final acceptance.
