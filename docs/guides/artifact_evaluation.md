---
title: Joint compiler artifact evaluation
kind: guide
status: draft
owner: experiments
last_verified: 2026-10-23
related: [getting_started, gsim_worker_setup, phase0_phase1_setup, ../reference/compiler_optimization_workflow, ../design/joint_campaign_execution]
code_refs:
  - experiments/reference-data/el4-artifact-release/provenance.json
  - build_tools/scripts/audit_compiler_integration.py
  - build_tools/scripts/package_worker_inputs.py
  - examples/gemmini/phase1/launch/launch_phase1.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/qualification_audit.py
---

# Joint compiler artifact evaluation

**This is a release candidate, not a qualified clean-machine reproduction.**
The source release is being separated from local deployment state. Consult the
[release provenance and outstanding dependencies](../../experiments/reference-data/el4-artifact-release/README.md)
before allocating a long-running worker. Do not interpret source availability,
passing infrastructure tests or a successfully started grader as a completed
compiler experiment.

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
uv venv --python 3.12
uv pip install -e '.[dev,xdsl,targetgen]'
uv pip install -e packages/merlin-experiments -e packages/merlin-timing
.venv/bin/merlin --help
.venv/bin/merlin experiment list
```

These commands establish a development installation, not a locked experiment.
Release acceptance must also build wheels and the sdist, install the core outside
the checkout without research extras, and check wheel contents with
`build_tools/scripts/check_distribution_layout.py`. Record resolved dependencies
for each independently installed environment. Do not silently use editable
packages from an existing worker to declare a clean installation successful.

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

1. At most 1,200 reachable commits on the new lineage, with preserved source
   revision mappings and notices; no existing published branch rewritten.
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
