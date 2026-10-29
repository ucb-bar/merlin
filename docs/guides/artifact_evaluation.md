---
title: Joint compiler artifact evaluation
kind: guide
status: draft
owner: experiments
last_verified: 2026-10-23
related: [getting_started, gsim_worker_setup, phase0_phase1_setup, experiment_dashboard, public_capability_planning, ../reference/compiler_optimization_workflow, ../design/joint_campaign_execution, ../design/compiler_coevolution_study]
code_refs:
  - experiments/reference-data/el4-artifact-release/provenance.json
  - build_tools/scripts/audit_compiler_integration.py
  - build_tools/scripts/package_worker_inputs.py
  - examples/gemmini/phase1/launch/launch_phase1.py
  - examples/gemmini/phase1/launch/prepare_author_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/prepare_author_library.py
  - examples/gemmini/phase1/launch/resource_reservation.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/terminal.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/qualification_audit.py
---

# Joint compiler artifact evaluation

**This is a release candidate, not a qualified clean-machine reproduction.**
The source release is being separated from local deployment state. Consult the
[release provenance and outstanding dependencies](../../experiments/reference-data/el4-artifact-release/README.md)
before allocating a long-running worker. Do not interpret source availability,
passing infrastructure tests or a successfully started grader as a completed
compiler experiment.

There are three different reproduction tasks:

- **Run the shared tooling:** install this checkout, execute public compiler
  examples and inspect run records. This does not require an authored backend.
- **Repeat the experimental procedure:** select obtainable hardware/runtime,
  corpus, shared-library and author inputs, then run a new empty-backend campaign.
  This produces a new compiler and may produce different outcomes.
- **Replay a reported result:** obtain the exact frozen compiler, selected
  library, inputs, toolchain and measurement records for that result. A newer
  compiler or a newly generated corpus cannot substitute for these inputs.

This guide documents the interfaces for all three. The complete downloadable
input closure and portable joint-run preset are still prerequisites for an
independent fresh-machine campaign; they are not supplied by a Git checkout.

## What an evaluator should be able to reproduce

| Evaluation | Inputs and expected evidence |
| --- | --- |
| Compiler tooling | Installable core and optional experiment packages; pass APIs, pipeline examples and focused behavior checks |
| Retained compiler inspection | Frozen backend, exact selected baseline library, sparse modifications and source-bound emitted diagnostics |
| Fresh joint authoring | Empty target backend, admitted shared tooling and public examples, frozen corpus, exact prompt and runtime/tool selection |
| Continuation | The same setup plus an explicitly reviewed retained compiler; report this as continuation, not from scratch |
| Correctness | Per-capsule numerical execution and policy/placement results; required RTL checks recorded separately |
| Calibration and optimization | Public numerical RTL witnesses, train/holdout separation, model admission and immutable epoch identity |
| Final physical validation | Frozen ELF/input/compiler identities and full output checks on the unchanged selected FireSim design |

The intended public workload roster has 250 capsules: 153 original correctness,
20 calibration candidates and 77 scalability/performance cases. The separate
complete-model program roster is not another 250-case denominator. See the
[exact roster](../design/joint_capsule_roster_v14.md). Original correctness cases
retain required Spike and gSIM obligations. Calibration selects additional
small/medium witnesses by unresolved model costs and evidence reuse. Large
scalability cases use cheaper functional execution and model feedback rather than
routine gSIM. They do not need to wait for all original cases before development.

Optimize primarily for the larger workloads. Do not count the repeated execution
of one case or results from incompatible compiler revisions as more capsules
passed by one compiler. Keep L2, L3, placement, timing-model admission and final
physical validation visible as distinct gates.

New joint preparations expose a compact capability inventory for all public
sources before authoring. Use it to organize compiler work by semantics, with
early executable canaries and continuous independent evaluation. They also
record a 40M all-author soft token target (including cached input), with an 8M
integration/repair reserve. The [study design](../design/compiler_coevolution_study.md)
defines accounting, strict timing qualification and future comparisons; none of
those comparisons is launched merely by installing this package.

For a fresh continuous Codex run, explicitly select `--author-token-admission`
(or `author.token_admission: true` in the launch configuration). Historical
advisory-only runs keep their original behavior. This records a v2 policy with
live all-thread usage checks every five seconds, a 1M planning allowance per
outstanding turn (`--author-token-turn-allowance`), and an 8M integration reserve.
Cached input and every descendant helper still count; an allowance is not
measured usage or a provider upper bound.

The supervisor enforces admission before each managed process and polls during
execution. At the projected 32M development boundary it terminates only its
owned process group, retains logs and workspace edits, and can resume the same
thread once for integration/checkpointing. That continuation cannot authorize
new exploration, is limited to five minutes, and is refused if remaining planned
headroom is insufficient. Partial interrupted work stays unqualified until
reviewed and checked normally. Missing live usage is not zero: new admission
refuses it, and running work gets a sixty-second telemetry grace. Under-budget
work with available usage is not interrupted merely because time elapsed.

`author_token_admission.json` retains policy, high-water observed usage,
outstanding owner reservations and every decision. A stale open turn is not
silently released. Native sub-agent spawn calls and requests already in flight
cannot be individually admitted or bounded by this supervisor; usage publication,
polling and termination can lag. Therefore this is **not a hard 40M token cap**.
Resuming preserves thread history; it does not force context compaction. Use
short file-specific helper briefs, read each catalog index once, fetch detailed
source only as needed, and retain large logs/IR as artifact paths. No model or
provider settings are changed by this policy.

## Worker and dependency setup

Use an x86-64 Linux worker on AWS for the documented native gSIM runtime; a
Graviton worker cannot run that x86 executable natively. Install Python 3.12+,
the pinned Python dependencies, LLVM/MLIR, the RISC-V target toolchain and the
selected simulator sources. Native hardware builds also require the documented
Chipyard/FIRRTL and gSIM dependency closure. The
[worker guide](gsim_worker_setup.md) contains the build commands; the
[LLVM guide](llvm_toolchain.md) describes compiler setup.

Reserve 20 CPU cores dynamically for the experiment. Build, functional execution,
RTL witnesses, calibration and agent helpers share that allocation rather than
occupying fixed CPU sub-pools. Keep infrastructure work outside that reservation.
Record actual RAM, storage, CPU model and build wall time during clean-machine
acceptance; this release has not established a minimum instance type, disk size
or cost estimate. API/model access is an additional requirement, not provided by
renting a worker. Do not package authentication state or a developer's `.env`.

For a source checkout, the documented development install is:

```sh
# Start in a new directory on the evaluation machine. Select a full release SHA;
# the feat/megaphase branch name alone is not an immutable experiment identity.
git clone https://github.com/ucb-bar/merlin.git merlin
cd merlin
git checkout --detach "${MERLIN_REVISION:?set the selected full release commit}"
uv venv --python 3.12
uv pip install -e '.[dev,xdsl,targetgen]'
uv pip install -e packages/merlin-experiments -e packages/merlin-timing
.venv/bin/merlin --help
.venv/bin/merlin experiment list
.venv/bin/python -m merlin_experiments.tracking.terminal --help
```

These commands establish a development installation, not a locked experiment.
Release acceptance must also build wheels and the sdist, install the core outside
the checkout without research extras, and check wheel contents with
`build_tools/scripts/check_distribution_layout.py`. Record resolved dependencies
for each independently installed environment. Do not silently use editable
packages from an existing worker to declare a clean installation successful.

Use a Linux host with systemd and cgroup v2 for the supplied exclusive CPU
reservation wrapper, plus bubblewrap and working user namespaces for the author
sandbox. The wrapper temporarily changes the host slices' allowed CPUs and
therefore needs administrator permission; a container with only CPU affinity
does not provide equivalent exclusivity. Leave CPUs outside the experiment's
twenty-core reservation for the operating system and monitoring. All experiment
roles share the reserved cores. Do not copy core numbers from another machine.

Select the author executable, model, effort and helper limit explicitly in the
deployment. Configure authentication privately on the worker; the source tree
does not provide an account, credentials or model access. Record the executable
version and prompt bytes. Keep credential files outside delivery bundles and
the authored compiler. Replacing the model changes the experimental configuration.

## Input inventory for a new machine

Obtain or regenerate each row before running a long campaign. The release
provenance records historical source identities; those hashes alone are not
download locations or build recipes.

| Input | Required identity and preparation |
| --- | --- |
| Merlin and Python packages | Full source SHA, installed dependency versions and package build/install receipt |
| LLVM/MLIR, CIRCT and guest toolchain | Exact revisions/binaries and build flags; follow the [toolchain guide](llvm_toolchain.md) |
| RTL and hardware facts | Chipyard/Rocket/accelerator revisions, elaboration configuration, extracted facts, neutral contract and their hashes |
| gSIM | Matching elaborated design, selected simulator revision, executable and build receipt; follow [worker setup](gsim_worker_setup.md) |
| Functional/timing Spike | Obtainable source revision and runtime build closure; selected timing profile and coverage status, separately from functional execution |
| Reviewed capsule release | Payload, review/seal, descriptor, input bundles and public scalability selection; inspect counts against the selected roster |
| Numerical references | Exact reference implementation/policy and inputs; retain operator-only references outside the author sandbox |
| Shared compiler library | Exported source tree, contract, selection and dependency closure; select new modules explicitly rather than exposing the whole checkout |
| Workload captures | Capture source/configuration, framework versions and quantization parameters; complete-model validation needs its own inputs and all-output plan |
| Launch and resource policy | Portable native selection/configuration, joint policy, prompt/errata, budgets, CPU allocation, author/runtime environment |
| Result replay bundle | Frozen compiler plus the original selected library, capsule/input identities, ELF and raw result records for each reported comparison |

The target backend starts empty for a fresh run. A historical `gemmini-mlir`
compiler is not an installation dependency to seed into authoring. Trusted
simulator/runtime support is a separate operator input. Reviewed public examples
require their own pinned source selection and redistribution rights; private
reference kernels and comparison scores remain outside author inputs.

## Preparing operator and author inputs

Follow [Phase 0 and Phase 1 setup](phase0_phase1_setup.md) to derive/review the
corpus, bind numerical references and qualify execution endpoints. Those
historical names identify implementation modules; the joint experiment keeps
correctness and performance development in the same authoring loop.

The launch configuration requires a reviewed corpus, numerical policy, hardware
facts, instruction policy, simulator build receipts and the selected library.
Existing `examples/gemmini/phase1/launch/` files document the configuration schema
and historical deployments. They are not a portable, self-contained joint-run
preset. Do not copy their historical hashes and call newly built inputs identical.
Construct a new deployment declaration and review its actual selected bytes.

Use `package_worker_inputs.py pack` to transport an explicitly selected worker
input closure, and `verify --sha256 ... --extract ...` to reopen it on a new
worker. Packing proves transport integrity, not redistribution rights, hardware
equivalence, corpus approval or a functional compiler. Operator bundles are not
author-visible inputs. Keep hidden references, private comparison kernels,
private baseline scores, access tokens and cloud account state out of the author
sandbox and public source release.

Before a long campaign, execute a tiny capsule through real source emission,
target consumption, linking, Spike and full numerical comparison. Then run one
representative case per selected execution class. Missing symbols and shared ABI
failures must stop affected admission early. These are execution checks, not a
reason to rerun every unrelated unit test on each author edit.

Use immutable compiler snapshots with continuous per-case admission and feedback.
Reuse evidence only with matching source/input/policy/tool identities. Record
queue wait, preparation, verification, emission, linking, simulation and completion
processing separately. Keep the author working while independent snapshots run.
Do not relabel an older best snapshot as the latest compiler.

## Prepare and launch a joint campaign

The tracked `phase1_el4_config.yaml` is an **18-CPU capsule-only deployment
template with unset source/digest placeholders**. It is not the current
twenty-core joint preset. The launcher `launch_phase1.py` accepts a completed
configuration; without `--run` it performs readiness checks only:

```sh
.venv/bin/python examples/gemmini/phase1/launch/launch_phase1.py \
  --config "${CAMPAIGN_CONFIG:?select a fully populated deployment config}" \
  --task-root "${CAMPAIGN_TASK:?set the managed task directory}" \
  --prep "${CAMPAIGN_PREP:?set a new preparation directory}" \
  --run-id "${CAMPAIGN_RUN_ID:?select a unique run ID}"
```

Adding `--run` starts the selected native controller only after its checks pass.
A readiness report is not a numerical pass. Do not mix this config workflow's
outputs with another launch selection or manually bypass refused input checks.

The current fresh joint workflow uses `prepare_author_launch.py`. It requires
an existing, independently reviewed native launch JSON (`argv`, environment and
environment policy), a reviewed corpus release, a compiler-library selection
and a joint policy. It refreshes these identities, removes seed/resume settings,
selects an empty backend and writes a separate no-author preflight selection.
It does **not** create the underlying inputs or run the author:

```sh
.venv/bin/python examples/gemmini/phase1/launch/prepare_author_launch.py \
  --template "${NATIVE_LAUNCH_TEMPLATE:?obtain the reviewed native selection}" \
  --release "${CORPUS_RELEASE:?obtain the reviewed corpus release}" \
  --library-selection "${COMPILER_LIBRARY_SELECTION:?select the exported library}" \
  --repo "$PWD" \
  --output "${JOINT_LAUNCH_DIR:?set a new directory under out/artifacts}" \
  --policy "${JOINT_POLICY:?select the joint campaign policy}" \
  --run-id "${CAMPAIGN_RUN_ID:?select a unique run ID}"
```

Review `proposed-launch.json` and `preflight-selection.json`. All executable and
input paths must resolve on the new worker. The selected policy duration must
match the native author clock. The author, preparation and compute CPU lists
must be the same twenty distinct CPUs; slot directories are private to the
worker user. The script preserves the template's model/provider selection, so
review those fields as well as the corpus, library and prompt hashes.

Complete the native no-author preflight using its selected environment and
arguments, retain its receipt, and check the small execution canaries described
above. There is currently no self-contained published template plus deployment
recipe for this entire step. Once that closure is provided and accepted, the
prepared command starts the campaign and its finite resource reservation:

```sh
bash "${JOINT_LAUNCH_DIR:?}/proposed-reserved-command.sh"
```

This command launches work and may incur model/cloud costs. Its wrapper restores
host CPU reservations on completion, failure or timeout. Retain `campaign.log`
and the CPU reservation state, and confirm release after the run. A historical
short rehearsal bound is not a full-campaign budget. Editing prompts, libraries
or launch inputs while they are selected by a running author changes custody;
prepare a separately identified successor instead.

## Monitor and collect results

Use the [unified terminal console](experiment_dashboard.md#unified-animated-terminal-console)
from a separate SSH session on the worker. It shows cores, jobs, author/helper
lineage, capsule classes, L2/L3 progress and recorded performance feedback in one
view. It does not launch tests:

```sh
.venv/bin/python -m merlin_experiments.tracking.terminal \
  --task "${CAMPAIGN_TASK:?set the managed campaign task}" \
  --window 120 --width 160 --height 55
```

Keep the owner/selection records with the managed task; `--task` is not a bare
run directory. If the selected launch did not write compatible owner records,
the console reports unavailable data. The guide explains this distinction and
the static run-directory dashboard fallback. Do not create synthetic owner
records to make an unavailable campaign look active.

For replay, preserve the run directory, exact compiler revisions, selected
library source, corpus/reference identities, tool/build receipts and per-case
raw outputs together. Preserve the launch selection and resource logs separately
from credentials. A useful release bundle includes a manifest mapping each
published result to these inputs and its measurement scope. A new author run is
not guaranteed to regenerate the same compiler or timing numbers.

Count unique capsules on a single compatible compiler snapshot. Show original
correctness, calibration and scalability classes separately, and keep numerical
Spike, required gSIM, placement/instruction admission, model qualification and
complete-model physical validation distinct. Historical best/union counts,
current snapshot counts and numerical snapshot/capsule pairs answer different
questions; the result table must say which it uses. No missing result is a pass.

## Timing feedback and physical validation

Functional Spike counts are not physical cycle measurements. An untuned or
structurally incomplete timing model can provide labelled provisional diagnostics;
it cannot be reported as a calibrated whole-SoC predictor. Prefer an admitted
tuned epoch once available, preserve its applicability domain, and report unknown
coverage explicitly. Derive geometry and queue/resource facts from the selected
RTL. Fit uncertain costs using public, numerically checked gSIM evidence and
evaluate separate holdouts. Keep private FireSim comparisons outside fitting and
automatic author feedback.

Final validation uses the unchanged selected `FireSimRocketGemminiConfig`, its
regular scalar in-order Rocket core, and the matching memory/clock configuration.
Record the build/configuration chain and executable, not just an AGFI or a config
nickname. Bitstream access, licenses, regional F2 capacity, IAM permissions and
an account-specific deployment are prerequisites that still require release
acceptance. Do not substitute a convenient bitstream and call results comparable.

On a workhorse provisioned with the supported wrappers, submit only
`fpga-run-elf ELF NAME` (or the canned `fpga-run`). One job owns the FPGA at a
time; the wrapper handles manager/F2 setup and teardown. This deployment has no
`MERLIN_EXT_FIRESIM_QUEUE`. Do not use raw FireSim lifecycle commands or start/stop
instances manually. Collect UART logs and wrapper receipts from the configured
results bucket, verify full outputs and timer scope, and confirm teardown. The
wrapper and account resources are not yet bundled as portable infrastructure.

## Release acceptance

Publication readiness requires all of these, with retained receipts:

1. Preserve source revision mappings, third-party notices and redistribution
   rights for all provided sources and inputs.
2. Review every published tree and commit for credentials, private evidence,
   third-party rights, machine dependencies and stale deployment instructions.
3. Pinned, obtainable source/build inputs for the simulator, toolchains, corpus
   and model captures; record missing assets instead of inventing provenance.
4. Wheel/sdist and outside-checkout installation acceptance; focused pass and
   harness checks; representative actual target execution on a clean worker.
5. A portable joint launch configuration with exact prompt, resource policy,
   bounds and a reproducible result collector.
6. An explicit claims table separating observed correctness, unknown gates,
   provisional timing and physical measurements. A failed research run remains
   useful evidence, but is not a successful full-campaign result.
