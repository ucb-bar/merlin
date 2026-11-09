---
title: Functionality preservation during architectural refactoring
kind: design
status: draft
owner: core
last_verified: 2026-11-07
related: [architecture, compiler_verification, reproducing_paper_results]
code_refs: [pyproject.toml, packages, src/merlin, experiments/reference-data/paper-artifact]
---

# Functionality preservation during architectural refactoring

The refactor preserves the functionality supported by `mlsys27/megaphase` while
replacing implementation ownership and public interfaces together. Frozen paper
inputs are an additional preservation obligation. Archiving an old implementation
does not replace a working implementation of an active capability.

The destination has six domains: target, compiler, verification, analysis,
evaluation and development. Verification is an independently selected service,
not a test directory or a numerical benchmark stage. These are proposed ownership boundaries, not a claim that the current
repository already has this layout. Existing consumers continue to use their
current implementation until a migration provides and checks its replacement.

## Preserve workflows rather than historical module names

| Capability | Current owners | Destination |
| --- | --- | --- |
| Capture and numerical preparation | `capture`, `frontends`, targetgen capture worker and recipe quantizer | Compiler frontend and quantization |
| Hardware facts and test generation | `targetgen`, schemas, corpus and conformance machinery | Target, RTL analysis and evaluation cases |
| Placement, grouping and source interfaces | `system`, `compile`, xDSL grouping and outlining | Compiler planning, shared boundary analysis and partition |
| Scalar and RVV compilation | `llvmlower`, core `mining` package loader and schedule realization | Compiler providers, scalar lowering and shared LLVM finalization |
| Scheduling and semantic selection | `sched`, `liveness`, `semantic_compiler` | Compiler scheduling and selection |
| Shared transformations and validators | `transforms`, `verify`, shared xDSL passes | Compiler transforms and independent verification |
| Building, executing and checking programs | `runtime`, `model_check`, `validation`, `perf` | Compiler build, deployed runtime and evaluation |
| Expert ingestion and CCA | `kernels`, optional mining distribution | Analysis and active development |
| Backend construction and refinement | Experiment and mining distributions | Development, using public compilation and evaluation |
| Timing and result analysis | Timing and analysis distributions, `perf` | Analysis, evaluation observations and studies |
| Paper restoration and replay | Paper manifests and maintainer restoration scripts | Study reproduction with immutable historical identities |

The core `mining` directory contains real compiler dependencies, including package
admission, feature application, policy legality and schedule realization. The
optional mining distribution also supplies active CCA and refinement operations.
Neither packaging location establishes that functionality is unused.

CCA ingestion, lifting, comparison, applicability and achieved-change inspection
remain active. Static instruction counts, executed traffic, modeled cycles and
measured performance retain their different meanings. Candidate refinement may
consume CCA differences; CCA alone does not establish correctness or speedup.

## Record the preservation baseline

Maintain the task inventory under ignored `tmp/refactor/`. Capture the exact
baseline commit, tracked files and Git object identities, all declared console
scripts and plugin entrypoints, package dependencies, workflow owners, proposed
destinations and verification obligations. Capture from the baseline revision,
not from a changing working tree.

For each workflow, use these migration states:

- **Pending:** the destination is not yet implemented and checked.
- **Preserved:** existing or replacement behavior has supporting check evidence.
- **Migrated:** consumers use the new owner and the replacement obligations have
  been checked.

Source inspection is not execution qualification. A passing disappearance audit
only establishes that the inventory has not lost files or entrypoints without a
recorded decision. It does not prove semantic equivalence, discover every dynamic
consumer, or establish complete cross-repository dependency closure.

## Removal requires evidence

Before removing an implementation, record the retained capability and its callers.
Review direct imports, CLI entrypoints, plugin registration, generated source,
scripts, package data, external backend consumers and paper reproduction inputs.
Identify the replacement and the checks protecting the original obligation.

An implementation can be removed when its retained consumers have migrated, or
when evidence establishes that no retained capability needs it. Remove obsolete
exports, configuration, documentation and compatibility-only tests in the same
migration. Keep distinct numerical, ABI, scheduling and trust obligations through
the replacement tests.

Independent exploratory features may move to a feature branch after their
dependencies on the preserved baseline have been reviewed. Names such as DSE,
mining, Phase 2 or optional do not authorize bulk deletion. Missing fresh hardware
results do not establish dead code.

FireSim infrastructure has an explicit external ownership boundary. Preserve
workload/image preparation, generated invocation files and result collection in
Merlin. Provisioning, farm configuration, simulator launch and lifecycle management
belong to operator tooling outside Merlin. Existing coupled callers must migrate
to the prepared-workload and external-result contract; retaining an internal
launcher is not required preservation. A prepared workload is pending execution,
not successful execution or performance evidence. Agentic studies must retain
that distinction while awaiting external observations.

## Keep authority and evidence intact

Candidates cannot choose their acceptance thresholds, references or hidden test
inputs. Preserve snapshot identity, corpus identity, certificate attribution,
continuous grading, per-case execution-tier promotion and post-freeze evaluation.
Register trusted access identities before moving graders or numerical oracles.

Historical compiler and dependency bytes remain immutable. Saved-result
recomputation, frozen executable replay, source reconstruction and fresh agent
authoring are separate reproduction claims. A new source tree or toolchain cannot
inherit an archived execution claim merely because its public interface matches.

## Check each ownership migration

Verification must migrate with its actual source/target correspondence,
semantic models, supported domains, toolchain identities, installed proof assets,
live completion authority and downstream acceptance consumer. Keep local
structural verifiers beside their dialects, but do not treat editable compiler
checks as independent semantic evidence. Evaluation owns acceptance policy and
execution observations. A structural result, binary hash, numerical sample or
proof of one IR boundary cannot silently become a whole-program proof.

Reduce regression code by distinct obligations, not a quota of deleted files.
Consolidation must retain the independent checks, representative valid cases,
counterexamples, relevant boundary cases and real public consumers. Parameterizing
unchanged examples or moving tests to OOT repositories does not establish a
reduction in total maintenance work. Generated conformance should have bounded,
reproducible inputs and minimized retained failures; it does not replace all
permanent regressions. Required unavailable verification is not a passing CI result.

The agent-facing transformation library is a required current-branch workflow.
Preserve its hardware-parameterized tiling, operand retention, transfer merging,
command ordering and other declared library capabilities together with their
resource/effect/encoding interfaces. The agent must remain able to compose,
modify and add passes, not merely select a fixed list of invocations.

This obligation includes the versioned public API and implementation inventories,
source exports, discovery catalogs, author instructions and recipes, editable
compiler-library overlays, actual selection of edited code, candidate snapshot
identity and independent verification. Check those consumers before declaring
library code unused. Preserve both an unchanged-library path and a candidate-edited
path through real lowering; source presence or a directory rename is insufficient.
New pass implementations may live in the backend or a declared shared-library
variant. Generic improvements should have a promotion path to the shared library;
target-specific facts and encoding stay with the target provider.

Run the affected behavior checks and architecture/access checks. Update generated
documentation and package resource inclusion when their inputs move. Check wheels
and source distributions outside the checkout before claiming installed-package
compatibility. Report unavailable toolchains and hardware separately from passing
checks.

## Review usability as part of each migration

The following eight requirements are proposed acceptance criteria for the
refactor. They are not claims that the current CLI or public Python API already
implements them. Review the relevant criteria with each ownership change; do
not defer them to a final documentation pass. The architecture Word document
contains the complete public design, while this guide defines the review gates.

### Explain the actual compilation

Preserve `compiler/inspect.py` stage selection, dumps and stop points, and
`common/compile_trace.py` observed occurrences and completed/stopped/failed
records. These provide real tracing today; joining all decision and execution
evidence is the remaining obligation.

Proposed owners are compiler trace and inspection, with backend and runtime
adapters contributing records through their public contracts. A single inspection
view must connect source call sites, realized numerical semantics, candidate
groups, selected implementations/resources, rejected alternatives, data movement,
ordered pass invocations, emitted artifacts and independently issued evidence.
Source links identify the selected revision or edited source digest. A report
must distinguish a planned route from an observed execution route and distinguish
unknown cost from a measured disadvantage.

Acceptance requires answering why a representative operation used scalar code,
why an alternative was rejected and where its OOT lowering or shared pass lives,
using the saved build without rerunning compilation. Inspecting incomplete and
failed builds must retain the last completed stage and diagnostic. Human output
and versioned machine-readable output consume the same records. Do not maintain
a separate handwritten stage map or treat an editable receipt as verification.

### Diagnose failures before expensive work

Reuse `compile/model_preflight.py` route blockers and the operation-aware
diagnostics in `triton/diagnostics.py`. Keep bounded mismatch reruns in
`model_check/diagnostics.py` separate from read-only preflight and preserve their
sealed-reference handling.

Proposed owners are request resolution, frontend/backend admission and runtime
preflight. Reuse existing refusal and preflight implementations through structured
diagnostics carrying a stable code, stage, operation/source identity when known,
provider, relevant option, explanation and actionable next step. Distinguish
unsupported input, missing implementation, missing dependency, incompatible
version, unknown capability and unavailable verification. Preflight reports what
it checked; it cannot certify runtime behavior it has not observed.

Acceptance includes a missing tool, unsupported quantization and incompatible
hart request failing at the earliest stage with sufficient information to decide
the next action. Python callers receive typed results or exceptions rather than
CLI exits. The CLI renders those same diagnostics with a documented nonzero
failure status. A missing optional backend must not prevent importing scalar
compilation or inspecting an existing artifact.

### Discover and resolve options once

Extend the real catalogue and validation in `llvmlower/optional_passes.py` and
`compile/command.py`. The existing `applied` context restores process environment
settings on exit but does not isolate concurrent requests.

Proposed owners are the public request schema and registered provider/pass option
descriptions. Derive CLI help and Python option discovery from the same records,
including types, defaults, scope, applicability and provider versions. Record
resolved values and their origins: default, profile, optimization preset or
explicit override. Preserve intentional repeated pass instances, anchors and
ordering. Reject unknown, conflicting and inapplicable options; settings cannot
override approved hardware facts or numerical acceptance policy.

Acceptance requires equivalent CLI and Python requests to resolve identically,
an invalid option to fail before expensive work, and concurrent calls with
different settings to remain isolated. An explicit requirement for accelerator
execution must propagate into the independent placement evidence requirement;
scalar success cannot satisfy it. Normal fallback is visible with its reason.
Retain existing option semantics until their replacement is demonstrated.

### Reproduce and shorten development iterations

`targetgen/build_cache.py` and `perf/whole_model_object_cache.py` already cache
build products. Audit their identity coverage and migrate useful implementations;
do not introduce a competing cache authority. The build cache already requires
cached executables to be executed and judged rather than caching a verdict.

Proposed owners are pipeline stage execution and artifact provenance. Support
retaining selected intermediate representations, stopping at an admitted stage
and reproducing a failing pass with its actual input, provider, options and
toolchain. Reproduction may require the declared environment; recording a
command alone does not establish portable replay. Keep per-call configuration
and child environments explicit instead of mutating global process state.

Cache only stages with a complete declared dependency closure. Keys include
input/state/constants, numerical contract, target facts, selected options,
toolchain/provider identities and actual shared-library or candidate-overlay
source bytes. Do not use commit IDs alone for dirty source. Nondeterministic or
incompletely described stages remain uncached. Store outputs atomically and
validate their identity on reuse. Cached execution observations cannot become
fresh performance measurements or inherit a new evaluator's acceptance.

Acceptance demonstrates a legitimate reuse and invalidation after changing a
pass implementation, target fact and numerical setting. An interrupted build
must not publish a complete reusable artifact. Begin with selected expensive
stages; a universal cache framework is not a prerequisite for the first slice.

### Make artifact handoffs independent of checkout paths

`targetgen/artifact_bundle.py` already checks selected profiles, contained paths,
hashes and declared buffer dependencies. Its structural admission explicitly
does not provide the v0.2 execution adapter or prove physical execution validity.
Preserve native program/program-set admission in
`semantic_compiler/target_binding.py` while adding runtime consumers.

Proposed owners are compiler artifact/build contracts and runtime admission.
Use bundle-relative content references and hashes for required products,
constants and invocation files. Record external prerequisites explicitly;
source navigation paths are informational and must not be required for execution.
Distinguish executable images, device programs, linkable objects/libraries and
diagnostic intermediates. Only products with an admitted entry or submission
contract are runnable. Loading rejects missing content or incompatible ABI,
features, memory, platform or hart constraints rather than rebuilding silently.

Acceptance relocates a supported build to another directory and admits or runs
it in a compatible environment without the source checkout. Altered bytes and
incompatible runtime requests are rejected. An external FireSim handoff prepares
the workload, then matches collected results to its request and artifact hashes;
preparation and pending results are not execution evidence. Full machine or
simulator infrastructure remains outside the bundle's ownership.

### Demonstrate a small backend contribution

Reuse `compiler/backend/load.py` provider resolution and `composition.py` named
ownership, dependency and conflict checks. Their contract declarations and the
current matrix proposal adapter do not themselves establish complete region
planning or independently prove a replacement pipeline correct.

Proposed owners are the backend protocol, shared compiler-library catalog and
OOT conformance interface. Provide one maintained, small worked example showing
capability description, a proposal, selection of a shared transformation,
resource/command-effect contracts, target realization/encoding and artifact
emission. Link each step to its actual implementation and independent obligation.
Only create abstraction layers that this example or existing consumers require.

Acceptance adds an implementation through provider registration without editing
target-name branches in shared compilation, copying another compiler repository
or importing experiment engines. Exercise one independently rejected case as
well as a supported case. Demonstrate that an agent can compose, modify and add
a pass through the selected editable-library mechanism and that its actual
source identity appears in the resulting build. Candidate code must not alter
the evaluator. Preserve shared LLVM finalization and a non-LLVM output route.

### Retire compatibility code with consumer evidence

The owner of each moved capability records the old entrypoint, replacement,
retained consumers, behavioral differences, migration evidence and removal
condition. Check installed commands, dynamic registration, generated imports,
optional distributions, companion backends, example scripts and archived paper
inputs as well as direct imports. A zero-reference text search is insufficient.
Compatibility adapters should delegate to one implementation and have explicit
deprecation behavior; do not let both owners evolve independently.

Acceptance demonstrates the retained workflow through its new public consumer
before deleting the old implementation. Required external consumers with
unresolved migration remain outstanding. Preserve exact archival inputs without
keeping their implementation on the active default route. The explicit external
FireSim ownership boundary remains an intentional migration, not an exemption
from preserving workload preparation and result collection.

### Keep the maintenance burden bounded

Proposed owners are subsystem maintainers and independently controlled
verification/evaluation policy. Organize permanent regressions around distinct
numerical, IR, resource, ABI, artifact, trust and public-workflow obligations.
Consolidate duplicated setup and retain negative controls; generated conformance
records seeds and minimizes failures for permanent regression when appropriate.
Review total maintained setup/logic across Merlin and OOT repositories so moving
tests does not masquerade as reducing maintenance.

Acceptance includes fast dependency/access checks and representative public
workflows, affected integration gates for native boundaries, and installation
checks outside the checkout for wheels and source distributions. Heavy hardware
qualification is separately selected and reports unavailable evidence explicitly.
No ordinary CI job provisions FireSim. Example scripts use the same API and
option descriptions as users; their supported scope and prerequisites remain
accurate. Documentation changes with the owning interface instead of accumulating
parallel guides. Decompose large rewritten modules by responsibility without
creating trivial forwarding packages or tests for each helper.

### Apply the gates in a useful order

The first runnable public API slice must include inspectable decisions,
structured diagnostics, shared option resolution and a usable target example.
It may retain existing execution restrictions if they are explicit. The first
OOT migration must additionally demonstrate the contribution path, artifact
handoff and actual edited-library source selection. Cache optimization follows
complete stage identity, not the other way around. Compatibility removal follows
consumer migration and independent checks; test consolidation follows an
obligation map. Each gate records demonstrated, outstanding or unavailable
evidence in the internal review inventory rather than marking a proposal done.

The detailed architecture Word document describes the complete destination. This
repository guide states the durable preservation rules; temporary inventories
and implementation progress remain outside outward-facing documentation.
