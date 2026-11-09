---
title: "Component launch qualification"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-11-09
code_refs:
  - src/merlin/experiments/phase1/component_qualification.py
  - src/merlin/experiments/phase1/component_witness.py
  - src/merlin/experiments/phase1/component_source_applicability.py
  - src/merlin/experiments/phase1/component_package_execution.py
  - src/merlin/experiments/phase2/component_launch.py
  - src/merlin/experiments/phase2/component_launch_inputs.py
  - src/merlin/experiments/phase2/component_runtime.py
  - src/merlin/experiments/phase2/authoring_cli.py
  - src/merlin/experiments/phase2/broker.py
  - src/merlin/experiments/phase2/component_runtime_authority.py
---

# Component launch qualification

The default `component-only-v1` authoring command refuses launch. Admission
requires an independently generated reviewed Phase0 component plan, full-output
functional guard and withheld-transfer grading, authentic source/import/build/
execution witnesses, frozen edit authority and actual native sandbox probes.
Source graph effects alone do not prove layout, aliases, epochs or host/device
execution. Missing capture, effects, runtime closure or execution evidence refuses
qualification. A qualified launch is not the final held-out acceptance gate.
Qualification requires the installed `aet` distribution and pins its actual
source and distribution metadata. A Python path injection is insufficient.

Source applicability uses the actual verified registered tensor MLIR. A closed
static tensor function can establish source-only absence of observable input
writes, addresses, persistent state and external synchronization. Unknown ops,
memory buffers, dynamic interfaces and missing FX/import correspondence remain
UNKNOWN. Every physical runtime effect stays UNKNOWN until independently
observed target execution discharges it; source purity supplies no numerical,
partition, alias/lifetime/epoch or repeated-invocation credit.

## Preparation

Freeze the public `ComponentView` with `materialize_component_view` and the original
corpus with `freeze_performance_corpus`. The view contains the admitted generic
compiler library and the canonical `public_component_manifest(corpus)` projection
as `contract/performance_corpus_manifest.json`. Do not copy private goldens,
coverage declarations, handwritten baselines, history or final graphs into it.

Inventory reviewed trusted executables and explicitly needed resource files:

```sh
python -m merlin.experiments.phase2.component_runtime \
  --executable /usr/bin/bwrap=/usr/bin/bwrap \
  --executable /usr/bin/python3=/usr/bin/python3 \
  --executable /usr/bin/dash=/bin/sh \
  --tree /usr/lib/python3.11=/usr/lib/python3.11 \
  --output /absolute/private/runtime.json
```

Use the selected Python's real standard library, compiler, linker, simulator and
ISA/CCA runtime files; the example paths are illustrative. Build a separate control
runtime inventory containing the selected Codex executable, its dynamic libraries,
native sandbox helper, shell, Python stdlib and required TLS CA file. Credentials
are a separate `auth_source`; they are not runtime files. `/usr/bin/bwrap` must be
pinned in the control inventory. Extra resource trees become individual file
grants; no directory mount is generated. The inventory records membership and
hashes. The actual readiness probes establish whether the closure runs.

Every admitted tool runtime grant must occur with the same source, destination
and digest in the author control runtime. Readiness commands must use an exact
absolute admitted executable destination. A successful probe in a separate
compiler transport does not prove that the fresh author can reach that tool.
Missing membership refuses before readiness or a paid model starts; this check
does not automatically add file grants or issue runtime/isolation authority.

## Closed configuration

`--component-launch-inputs` accepts exactly these fields. All host paths must be
absolute. Every digest is a current 64-character SHA256. Runtime lists are the
inventory lists above. `readiness` contains exactly one trusted bounded command
per capability `compiler`, `linker`, `simulator`, `isa`, `cca`; each must produce
the pre-reviewed exact `stdout_sha256`. Exit zero alone grants nothing.
The selected private `stage_root` must keep its derived public Unix socket path
within Linux's 107-byte filesystem path budget. Overlong paths refuse before
readiness/model launch; no frozen IPC binding is silently redirected.

```json
{
  "schema": "merlin.component_launch_inputs.v1",
  "candidate": "/absolute/fresh/compiler",
  "view": {"root": "/absolute/public/view", "manifest_sha256": "SHA256", "generation_sha256": "SHA256", "library_sha256": "SHA256"},
  "corpus": {"root": "/absolute/private/corpus", "manifest_sha256": "SHA256", "capsules_sha256": "SHA256", "coverage_root": "/absolute/private/phase0"},
  "descriptor": "/absolute/private/target_experiment.yaml",
  "source_root": "/absolute/frozen/merlin",
  "contract_root": "/absolute/frozen/contract",
  "stage_root": "/absolute/private/fresh-launch",
  "qualification_root": "/absolute/private/fresh-functional-qualification",
  "edit_authority_root": "/absolute/private/fresh-edit-authority",
  "edit_contract": "/absolute/reviewed/compiler_edit_contract.json",
  "runtime": [{"source": "/usr/bin/bwrap", "destination": "/usr/bin/bwrap", "sha256": "SHA256"}],
  "control_runtime": [{"source": "/usr/bin/bwrap", "destination": "/usr/bin/bwrap", "sha256": "SHA256"}],
  "readiness": [{"capability": "compiler", "command": ["/usr/bin/reviewed-probe"], "stdout_sha256": "SHA256"}],
  "codex_binary": "/absolute/reviewed/codex",
  "codex_destination": "/usr/bin/codex",
  "auth_source": "/absolute/private/auth.json",
  "price_table": "/absolute/frozen/codex_prices.json",
  "analytical": {
    "calibration_adapter": "/absolute/frozen/calibration_adapter.json",
    "qualification": null,
    "scope": {"timer_sha256": "SHA256", "accuracy_sha256": "SHA256", "input_policy_sha256": "SHA256"},
    "output": "/absolute/private/analytical",
    "lease_path": "/absolute/private/analytical-lease.json",
    "dependencies": [{"path": "/absolute/frozen/adapter.py", "sha256": "SHA256"}],
    "max_workers": 1, "memory_per_worker_bytes": 1073741824,
    "engine_slots": 1, "objective": "warm"
  }
}
```

The shortened lists and placeholder hashes are intentionally not an admitted
configuration. Fill all five readiness capabilities and complete runtime lists.
Factories resolve the selected target's capabilities; JSON cannot import a callback.
The selected backend's `prepare_component_execution_service` receives typed frozen
baseline, qualification, view/runtime, corpus, target, contract/source roots and
cost scope; its result is bound through `prepare_component_feature_provider`.
Missing actual cold/warm engine, histogram, full-output or scope authority refuses
the normal factory. No detached diagnostic activates it.

## Actual qualification and authoring

The qualification-only command evaluates the baseline domain, freezes edit
authority, runs all five tool readiness commands, runs Codex's native namespace
probe, denies malicious private/public sibling files and credentials, denies a
live reachable host TCP listener, and calls the ordinary broker over Unix IPC.
It does not invoke a paid model:

```sh
python -m merlin.experiments.phase2.authoring_cli \
  --workflow component-only-v1 \
  --component-launch-inputs /absolute/private/launch.json \
  --qualification-only --model SELECTED_MODEL --effort high
```

For authoring, use fresh evidence/stage destinations and the same configuration
with `--wall-budget-seconds 3600 --suite component-authoring`. The normal provider owns
one fresh model home and session, standard event/deadline/billing records and the
broker. Candidate tools are networkless. Only four public IPC files are mounted;
private broker receipts and detailed cost/CCA records remain evaluator-owned.
The final changed compiler is fully requalified against the original goldens
before a frozen candidate is marked consumable. Failed probes and refused
qualifications retain private records with the exact observed failure.

No-model native controls exercise an owned reachable TCP listener and private
canary files visible to the outer control process. The nested candidate profile
must deny direct, symlink, sibling and self/parent process-root access and TCP,
while the selected public interpreter can read admitted inputs and write its
candidate. These controls establish only their observed boundary and tool
usability. An actual fresh author session and independent target execution
qualification remain separate requirements.

`component_launch_inputs.py` assembles the closed launch data from live owners.
`component_launch.py` performs qualification and launch;
`component_runtime_authority.py` owns the independent runtime capability.
Serializing these fields or satisfying a readiness command cannot reconstruct
that capability or authorize a changed compiler.
