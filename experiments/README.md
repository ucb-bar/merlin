# Experiment catalog

Start with `merlin experiment list`; the catalog is the single entry point for
phase definitions, typed settings, budgets, and frozen inputs. `inspect` and
`preflight` are read-only checks, not agent launches or hardware qualification.

The workflow follows the paper's terminology: **Hardware-Guided Test Generation**
(Phase 0), **Functional Compiler Generation** (Phase 1), and **Performance
Optimization** (Phase 2). Definitions select experiments; phase tooling and engines
live in `packages/merlin-experiments/src/merlin_experiments/phase{0,1,2}/`.
Shared compiler infrastructure and generated target backends are separate owners.

## Where inputs and results live

For a target's phase-by-phase starting points, use the
[example workflow maps](../examples/README.md#target-experiment-inputs).
The catalog above indexes one definition per target; Phase 2 templates stay
separate until an operator supplies actual frozen evidence and deployment inputs.
Whole-model lowering/inspection is not a fourth certification phase or a substitute
for the selected target's execution gates.

| Material | Home |
| --- | --- |
| Discoverable experiment IDs | [`catalog.yaml`](catalog.yaml), the single catalog |
| Target definitions, descriptors and authored public recipes | [`examples/<target>/experiment.yaml`, `target/descriptor.yaml` and `phase0/recipe.yaml`](../examples/README.md) |
| Shared performance-family definitions | [`templates/phase0/performance.yaml`](templates/phase0/README.md), selected explicitly by each definition |
| Retained public synthesis inputs | [`reference-data/phase0/`](reference-data/phase0/README.md); preserved references, not regeneration destinations |
| New run state and generated capsules | Configured `out/runs/`; Phase 0 writes `phase0/capsules/` within its run |
| Reviewed corpus releases and new synthesis products | Configured `out/artifacts/protocols/` and `out/artifacts/verification/<target>/` respectively |
| Reusable target implementations | Selected OOT support packages, separate from public example inputs |

Definitions name inputs explicitly; paths are relative to the definition file.
Example-owned descriptors still select retained task/harness/bundle resource roots;
definitions also reference generated/private sidecars separately.
Never copy private holdouts or goldens into examples. Retained synthesis references
can contain reviewed manual overrides; generate a new artifact and review it before
selecting a replacement. Historical runs keep their original inputs and receipts.

## Derive capsules

Phase 0 writes a run-owned corpus under `phase0/capsules/`; capsules are generated
artifacts, never committed inputs. Regenerate them from reviewed inputs when
needed. A Phase 1 definition selects the reviewed, sealed *functional* corpus;
Phase 2 selects its own performance workloads plus the frozen functional compiler.
Those selections have separate identities and must not be inferred from a target
name or from whichever run happened most recently. Production phase 0 and phase 1
currently require separate `--phase 0` and `--phase 1` invocations: the generated
corpus must be explicitly reviewed and
sealed before a descriptor may promote it into the functional grading corpus.
Combined selection fails preflight rather than silently grading an older corpus.
The standalone generator also requires an explicit `--output-root`; omitting it
does not authorize writes into the descriptor's source corpus. Grading cardinalities
and holdout rules are unchanged.

Phase 0 also runs from installed distributions. An external definition supplies
`config.descriptor`, `config.recipe` and `config.performance_template`, with optional
`config.conformance_spec`, `config.synth_profile`, `config.smt_profile` and
`config.hidden_profile` declarations. Selecting synthesis for verified execution
requires a newly generated, reviewed conformance artifact and adjacent application
inventory, and a synthesis artifact bound to their exact inputs. Historical
references without those identities remain inspectable but unverified.
Recipe mode never discovers adjacent sidecars. A declared optional sidecar may be
absent; run/resume binds both its presence and its bytes. The legacy explicit
`config.profiles_root` mode instead selects a complete profile directory, including
the shared performance template and applicable sidecars; do not combine the two modes.
`config.profile` optionally selects a profile name different from the target.
The runner provides the output path and pins input and implementation contents for run/resume.
The package does not ship private profiles, holdouts or golden answers. The direct
engine is `python -m merlin_experiments.phase0`; installed use requires explicit
`--descriptor`, `--target`, and `--output-root`, plus either `--recipe` with
`--performance-template` or a legacy `--profiles-root` argument. No checkout recipe
directory is selected implicitly. Start with the
[Gemmini Phase 0 walkthrough](../examples/gemmini/phase0/README.md) for the catalog interface.

## Capsules from captured compute groups

`merlin experiment corpus groups --help` discovers the existing captured-model
capsule workflow through the same command surface:

```sh
merlin experiment corpus groups --target TARGET --capture /absolute/capture.mlir \
  --out /configured/out/artifacts/group-capsules/example
```

The command shares its parser and implementation with
`python -m merlin.targetgen.group_capsules`. Existing `--manifest`, `--model`,
`--only`, `--timeout` and `--max-tier` options remain available. It derives and
builds capsules; explicit `--package` additionally evaluates them with the native
oracles, and `--promote` writes the covering subset into the descriptor's **ungraded**
model-layer category. These options can write outputs or use hardware; help does neither.
This workflow does **not** seal a release, approve a grading corpus, or substitute
for the explicit reviewed phase-0 handoff below.

## Reviewed phase-0 handoff

After a successful phase-0 run, prepare a fresh release under the configured
artifact root. Preparation copies the complete descriptor-selected source pool,
retains classified hand-authored members, overlays receipt-declared generated
members, stages task resources and the native broker shim, and regenerates native
bundles. It refuses unresolved provenance, removals, collisions, symlinks, and
descriptor admission-count changes; it never changes the canonical corpus.

```sh
merlin experiment run gemmini-functional --phase 0 --run-dir /absolute/phase0-run
merlin experiment corpus coverage /absolute/phase0-run --spec /absolute/selected-conformance.yaml
merlin experiment corpus prepare /absolute/phase0-run --output /configured/out/artifacts/protocols/review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/review-1
# Only after an operator has inspected the prepared inputs and private diagnostics:
merlin experiment corpus seal /configured/out/artifacts/protocols/review-1 \
  --expected-digest DIGEST_FROM_INSPECT --reviewed-by OPERATOR --review-note REVIEW_SUMMARY
```

`inspect` exposes only aggregate counts and commitments. Detailed assembly records,
diagnostics, and review notes are owner-only under `private/`. The explicit seal
records an attributed local acknowledgement, not a signature or numerical/hardware
certificate. Native cohort admission is reused unchanged; oracle readiness and
grading still happen in the native phase engine. No command approves data for you.
`corpus coverage` verifies the completed run's input/output receipt and reports
which public source-pool cells and other conformance axes its capsules present.
Its explicit requirement is separately hashed. It does not establish numerical
correctness, graded admission, or whole-model compilation.

For Phase 1, the catalog examples require a reviewed release and a newly generated
bundle. Keep the authored definition unchanged and select both inputs explicitly:

```sh
merlin experiment preflight TARGET_FUNCTIONAL_ID --phase 1 \
  --corpus-seal /configured/out/artifacts/protocols/review-1/private/seal.json \
  --bundle-manifest /configured/out/artifacts/targets/TARGET/BUNDLE/input_bundle_manifest.yaml
```

The seal selects its immutable release descriptor. The replacement bundle must
not grant the historical `merlin/contract/capsules` tree or an ancestor;
preflight refuses it. The bundle identity comes from the selected manifest and
both paths are frozen for resume. `inspect` and `run` accept the same flags.
Alternatively, copy the definition and set `config.descriptor` to the
reported released descriptor and `config.corpus_seal` to the reported seal path,
plus a new `config.bundle_manifest` and matching `config.bundle` (also update
any `inputs.target` descriptor pin):

```yaml
phases:
  1:
    adapter: capsule_bench
    config:
      descriptor: /configured/out/artifacts/protocols/review-1/payload/experiment/target_experiment.yaml
      corpus_seal: /configured/out/artifacts/protocols/review-1/private/seal.json
      # Retain the chosen arm, treatment, model, driver/provider and budgets.
```

Then use the same public interface; only the last command executes the agent engine:

```sh
merlin experiment inspect /absolute/reviewed-functional.yaml --phase 1
merlin experiment preflight /absolute/reviewed-functional.yaml --phase 1
merlin experiment run /absolute/reviewed-functional.yaml --phase 1 --run-dir /absolute/phase1-run
```

The existing `capsule_bench` adapter verifies both inputs before execution/resume.
Sealed runs require the native bwrap snapshot: public inputs are frozen grants;
hidden corpus and review metadata are frozen host-only inputs, never candidate
grants. Native admission verifies those exact frozen bytes before an agent starts
and refuses missing private snapshots rather than falling back to live inputs.
Sealing makes the payload read-only and pins it through the shared storage lifecycle.
Public copies may share the existing content store. Private copies use the same
copy utilities without public-store hardlinks, keeping privacy independent of
whether a public file happens to contain identical bytes.
When a broad snapshot root contains a private subtree, that whole copied root
forgoes shared-store deduplication; disjoint public/toolchain roots still share.
Do not edit a sealed release; prepare a new one for any revision.

## Generate experiment bundles

An authored descriptor may select `resources_root` independently of its own location:

```yaml
resources_root: examples/my_target/phase1
```

Relative roots are relative to the configured repository root, not the descriptor;
absolute roots support external operator workspaces. Missing `resources_root` retains
the historical sibling layout. Tasks, declared harnesses, bundle lookup, and optional
environment/timing inputs use this owner. The declaration is not a blanket access grant:
bundle manifests and sandbox checks still control visibility. Parent traversal and
empty/malformed declarations are refused. Frozen plans bind selected inputs and optional
file presence; a changed root or resource requires new execution inputs.

Reviewed release preparation copies selected resources into the new release and removes
the source-root pointer, so an approved release never follows later edits to that pointer.

The installed Phase 1 controller renders full-mode prompts from the descriptor and
capability manifest. Realistic mode uses an authored `task/TASK_realistic.md` when
present and otherwise renders its prompt too. Do not create empty source directories
just to enable these generated-prompt workflows. Frozen plans record task-directory
absence, and later appearance invalidates resume. Release preparation records that
absence and creates a release-local task directory for bundle staging; it does not
invent authored task text or approve the release. Legacy file-based launchers still
require their explicitly selected task files. The served prompt is recorded as `TASK.md`
in the run, alongside the generated workspace copy.

For standalone bundle generation, select an authored target descriptor explicitly:

```sh
python -m merlin.targetgen.generate_bundles --descriptor /path/to/target_experiment.yaml
```

The command prints a fresh `input_bundles/` directory inside a versioned
`capsule-bench` artifact under the configured `MERLIN_OUT_ROOT`. Its product manifest
records the descriptor path and digest and lists the generated files. Repeated calls
do not overwrite previous products or write beside the descriptor. Generation does
not approve a release, select new experiment inputs, or update a `latest` pointer.

Use the reviewed Phase 0 handoff above to prepare and select grading inputs. The
low-level `--dest DIRECTORY` override remains available for explicit staging and
legacy maintenance; it can overwrite generated files there, so do not point it at
historical evidence. Existing launchers still require explicitly selected bundles;
they do not automatically consume the newest generated product.

## Definitions and execution

Agent transports live in `merlin_experiments.phase1.providers`; baseline Phase 1
uses the installed `python -m merlin_experiments.phase1` controller with explicit
operator inputs. The resource sampler's standalone CLI is
`python -m merlin_experiments.phase1.providers.resource_sampler` (the old native script
path is retired). Installed proxy users supply `MERLIN_PROXY_CONFIG` and
`MERLIN_PROXY_EXECUTABLE`; provider packages do not infer executable paths from a
synthetic repository root. Bedrock SDK support is the optional `merlin-experiments[bedrock]`
extra. None of these imports launches a provider or selects a target.

`merlin experiment list` is the entry point. Definitions select phase 0 (derive
capsules), phase 1 (create a functional compiler), and phase 2 (optimize it).
An experiment may declare one or several phases. Select `--phase 0`, `1`, `2`,
or `all` on `inspect`, `preflight`, and `run`.

The six original functional definitions use existing target profiles and the installed
RTLchecks treatment. Both it and `baseline-functional-template` require explicit
descriptor, authored bundle identity and manifest, and oracle timing path.
Their operator root is `MERLIN_REPO_ROOT`, not an inferred
source checkout. Phase-2 entries are
explicit `kind: template` documents: they list the missing operator-owned frozen
run/certificate/campaign inputs and cannot execute. Copy a template, supply its
inputs and budgets, replace the placeholder hashes, and set `kind: experiment`.
Profile selectors may differ from canonical hardware target names; the phase-0
`profile` field preserves that distinction and `--descriptor` passes the actual
declared descriptor, including an out-of-tree path, to the generator.
The separate [Gemmini Universal example](../examples/gemmini_universal/README.md)
declares only Phase 1. It does not inherit Gemmini's Phase 0 recipe or Phase 2
qualification; its retained bundle is a preparation reference, not reviewed evidence.

```sh
merlin experiment list
merlin experiment inspect gemmini-functional --phase 1
merlin experiment preflight gemmini-functional --phase 1
merlin experiment run gemmini-functional --phase 1
merlin experiment status /absolute/path/to/run
merlin experiment lineage /absolute/path/to/run
merlin experiment resume /absolute/path/to/run
```

`lineage` projects the recorded definition and selected input digests, selected
corpus selection, functional-compiler handoff identity and engine output locations
from the hash-bound frozen plan. It is read-only and reports historical inputs
without a fingerprint as unverified. It does not revalidate current source bytes,
inspect private corpus members or turn a successful process into a scientific
verdict. Use `status` for attempts and native/AET records for actual evaluation.

Only `run` and `resume` execute experiment-definition phases. A run may launch paid agents, compile
models, or use hardware according to its declared phase. `preflight` checks the
definition, local inputs, and entrypoint availability; its output explicitly says
that live engine readiness has not been tested. Native admission and grading
gates remain in force.

The functional example bundle paths are explicit preparation inputs, not new
certificates. All six retain the old launcher's intended RTLchecks `public_v0`
information treatment. Gemmini, Atlas and Radiance have retained manifests;
MX Gemmini, Saturn OPU and Saturn OPU RVV do not supply the selected public
bundle here. Prepare and review those bundles before execution; do not substitute
a hardware-bringup bundle or another target's bundle to fill the gap.
For verified runs, select the reviewed release's descriptor, seal and matching
generated bundle together in a copied definition. Never fabricate missing manifests.
Retained manifests may also name unavailable toolchains or grants containing directory
symlinks. Installed preflight refuses those incomplete closures: regenerate the reviewed
bundle with explicit, ordinary input trees for the deployment instead of weakening the
fingerprint policy. Catalog discovery does not imply an example is execution-ready.

Each example names an operator-owned `.oracle_timing.json` beneath its retained
target resource root. These explicit records are required and are not supplied
here: provision genuine timing evidence or select an existing record. Missing
files refuse preflight; these paths neither create records nor authorize empty
placeholders. Changing a record after freezing invalidates resume.
Installed RTLchecks discovers FileCheck on PATH, not the old LLVM/Chipyard
candidate paths. Provision the desired tool and required library paths explicitly;
machine-specific credential and compatibility-library defaults are not inherited.
Missing checks remain unavailable advisory evidence, not simulator qualification.
Selected support-provider Python membership and identity are bound on admission
and resume, and support sources remain host-private. This is live-source binding,
not copied-provider or complete toolchain execution qualification.

Install the separate `packages/merlin-experiments` package alongside Merlin and
AET. Phase 0, baseline/RTLchecks Phase 1 and measured-claims Phase 2 use installed
implementations. Model-portfolio Phase 2 still requires its
declared native source checkout. Functional orchestration
requires an authored `task/` directory adjacent to its descriptor, the selected
bundle directory, operator-root `merlin/contract` and `merlin/schemas`, and all
declared allowed/host-input grants. Their bytes and membership are frozen before
execution. Descriptor-adjacent optional timing records and `experiment.env` bind
absence too; creating them later invalidates resume. This is the explicit operator
input closure, not a transitive fingerprint of frameworks or toolchains.
Stored historical native plans retain their recorded commands; new functional plans
never fall back to native launchers. Missing or changed historical pins still refuse.
Likewise, host-side mask construction and synthetic snapshot tests do not qualify
the local kernel's user namespaces or bwrap execution. The native sandbox probe,
compiler, simulator and hardware gates must pass in the deployment environment.

The default orchestration directory is beneath the configured run root:
`<target>/<experiment-id>/<timestamp>_<unique-id>`. `--run-dir` selects an explicit
directory. `resolved-plan.json` freezes the definition, resolved argv, input
paths, and content hashes; `orchestration.json` records only process attempts.
Native engine paths are recorded per attempt. AET and those engine directories
remain the source of token accounting, grades, measurements, and checkpoints.
`execution_succeeded` means the process returned zero, not that a compiler passed
or that a performance claim was proved.

Definition paths are relative to the YAML file. The versioned schema rejects
unknown top-level/phase fields and the adapters reject unknown options. There is
no arbitrary shell-command field. Treatment and budget settings live in the
phase's typed `config` alongside the adapter's required inputs. For a new target,
write its definition and register its path in `catalog.yaml`; an external
definition can also be passed directly without editing the catalog.

Phase 2 has two separate modes and adapters:

| Mode and adapter | Existing engine | Required inputs |
| --- | --- | --- |
| `measured_claims` | Installed `phase2.chia_envelope_cli` → `phase2.checkpoint_cli` | Managed supervisor endpoint, explicit source/resource/output roots and suite; frozen functional run identity/hash, descriptor, RTL facts, profile, certificates/hashes and agent budgets |
| `model_portfolio` | Installed `phase2.portfolio_cli` | Deployment JSON, campaign config, writable candidate checkout, authoring budgets, and optional explicit model objectives/checkpoints |

Measured claims use the current Merlin interpreter for both the envelope and
coordinator. The adapter selects the installed wrapper and module; definitions
cannot substitute an executable or arbitrary command. `source_root` supplies the
working directory, not a default checkout. Select the installed core, experiments
package and experiments namespace roots explicitly. Their Python source membership
is checked before launch/resume. Contract resources, holdout catalog and declared
scientific input bytes are immutable inputs; stage/measurement destinations and
the supervisor socket are not recursively fingerprinted as immutable data.
The endpoint must belong to an already provisioned managed worker deployment;
configuration preflight neither creates it nor proves its readiness. Scientific
and managed-launch admission still run inside the installed engines.

For phase 2, specify both `adapter` and `mode` with the corresponding value. The
adapter accepts explicit named options, using underscores for the engine's
hyphenated flag names. Its option contract is in
`packages/merlin-experiments/src/merlin_experiments/adapters.py`. A campaign config
may name further input bundles; native gates still validate and freeze those
closures. List additional immutable roots in the definition's `inputs` mapping
when they must also be checked by orchestration resume.

Model portfolios require a `deployment` input with schema
`merlin.portfolio-deployment.v1`. Its exact fields are:

- `target`, `source_root`, `output_root`, `lease_path`, `functional_runs_root`;
- `source_roots`, `python_roots`, `legacy_roots`, `internal_aliases`, `exclude_paths`;
- `provider_root` (an explicit external support directory, or null);
- `contract_root`, `compiler_shared_source_root`;
- `sandbox_root`, `sandbox_declaration`, `sandbox_declaration_sha256`.

External locations are canonical absolute paths. Source/import roots, alias mappings,
exclusions, contract and compiler locations are relative to `source_root`; import
roots may contain the selected package source directories. Contract resources include
their `schemas/` directory. Sandbox inputs come from a retained functional qualification
declaration and its exact SHA-256; its resource tree must remain available at its original
location. They are not copied into the Python snapshot or rediscovered from a checkout.
Select the actual installed source owners: copying unrelated sources cannot attest the
running installation. The catalog fingerprints the selected sources, support provider,
deployment, campaign descriptor and optional price table. The launcher checks that admitted
configuration bytes match the frozen copies before dispatch. This is not complete external
tool/interpreter closure or proof of native sandbox isolation.

### Reporting a measured phase-2 experiment

The current paired GSIM report consumes the completed native parent manifest, its
expected SHA-256, and one candidate record for each declared trial. Obtain these
from the recorded experiment outputs; do not select a latest child or the best
trial. The report revalidates candidate handoffs, parent-pinned child/result bytes,
and the complete statistics predeclaration before writing output:

```sh
python merlin/experiments/gemmini_perf_bench/scripts/gen_perf_report.py \
  --experiment-manifest /path/to/experiment_manifest.EXPECTED_PARENT_SHA256.json \
  --manifest-sha256 EXPECTED_PARENT_SHA256 \
  --candidate-record TRIAL_1=/path/to/first_candidate_record.json \
  --candidate-record TRIAL_2=/path/to/second_candidate_record.json \
  --candidate-record TRIAL_3=/path/to/third_candidate_record.json \
  --output out/artifacts/perf-bench/TARGET/paired-report.md
```

Replace trial identifiers with the exact declaration values and use the configured
output root when relocated. This CLI still requires the native checkout and the
inputs needed by its candidate verifier. GSIM supplies timing; Spike corroborates
correctness, and Verilator supplies prelaunch engine qualification. Historical
`--run-id` reports use their separate sealed Verilator schema; they cannot consume
current paired manifests or manufacture a legacy prediction/recovery verdict.

### Resuming an experiment

Resume validates every frozen input and entrypoint before executing. Completed
phases are not repeated. Functional plans additionally freeze the descriptor-selected
public and hidden corpus roots and their bytes. Before execution and resume, the
same descriptor API rediscovers category membership; adding a sibling category or
introducing a previously absent hidden corpus invalidates the plan too. Pins contain
aggregate hashes, not hidden capsule names or answers. This reproducibility boundary
complements the native engine's immutable bundle snapshots and cohort admission;
it does not grant grading approval or seal newly generated phase-0 output. Older
functional orchestration plans without a corpus closure require a new run; native
historical records are not rewritten. The functional compiler engine receives its native
`--resume`; measured claims repeat their frozen managed-envelope command and reuse
the same content-addressed coordinator root, without appending `--resume`. Stored
native plans retain their original command rather than being migrated during resume. Portfolio
outputs must be fresh, so `resume RUN --checkpoint SEALED_CHECKPOINT` creates a new
numbered policy segment and passes the checkpoint to the selected portfolio engine. It never
overwrites a prior segment or changes a measured claim into a model estimate.

Phase 0 retains the legacy generator's profile selection and derivation semantics,
but writes only its run-owned output corpus; it is not a dry run.
Revising a definition or frozen input requires a new run. Long-lived snapshots
and artifact cleanup use the shared Merlin storage lifecycle, not another store.

`reference-data/` is a separate read-only collection of committed historical
analysis evidence. The [DSE snapshot](reference-data/dse/README.md) is consumed by
analysis tools and regression tests; it is not a default output destination.
