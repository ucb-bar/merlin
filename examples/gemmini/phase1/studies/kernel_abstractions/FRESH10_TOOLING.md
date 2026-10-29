# Fresh10 capability-removal study

The [2026-10-28 configuration registry](configurations/20261028/README.md)
preserves the exact specs, source identities, received-input inventories,
resource budgets and comparison limits for the earlier EL studies, external
fresh10 baselines, and the next three fresh10 tooling ablations.

This is a new native study derived from the deployed fresh10CCA framework
`f9d7fb5c3d6fa2e5168e982953dd0b1d9b480076`, plus enumerated common harness/profile
fixes. It starts three empty target backends with fresh homes and state. It does
not use the earlier EL3 study generator or its deferred-evaluation protocol.

| Arm / profile suffix | Scheduling semantics | Supplied transform implementations | Raw reference kernels | CCA / automatic learning |
|---|---|---|---|---|
| A: `infra` | absent | absent | original complete set | off |
| B: `scheduling` | present | absent | identical to A | off |
| C: `no-reference-kernels` | present | present | absent | off |

A and B are **reference-assisted**: the original reference set includes completed
optimized schedules matching some public shapes. Their difference isolates the
reviewed scheduling abstraction on that reference-assisted base. B to C changes
both the supplied implementation library and reference exposure; it cannot be
reported as an isolated pass-library effect. C retains general infrastructure,
scheduling and pass abstractions, with all raw reference material removed.

The complete A/B raw set is 23 members from nine origins and the **155-entry**
reference map. The 68 newer ELF entries are a subset of that map. Separately,
the common ABI/document/schema native-read gate covers **23 public data files**;
it is not the raw-reference inventory.

All arms keep the actual fresh10 core TASK and native compiler/library
integration, 152 public cases (109 core, 40 scalability, 3 connected programs),
14 protected cases, the six selected full models, instruction selection, numerical
judge, performance selection, native calibration and continuous promotion. One
common neutral external guidance replaces the old guidance's unavailable tool
instructions. Profile-specific API disclosures remain factual and separate.

Common settings are `gpt-6.1-sol`, high lead effort, medium helper effort, eight
helpers maximum, fifteen shared CPUs, 14400 author seconds plus 900 drain seconds
(15300 total), and 900 seconds of final-model allowance. The old fresh10 final
used eleven author hours plus one drain hour; it is not a matched four-hour
control. No hardware-loop/sequencer instruction role is permitted.

CPU pools are A:38–47,56–60, B:61–75 and C:76–90. Rendering specs and native
preflight do not claim them. The reviewed dispatcher preserves peer host masks,
enforces at most three jobs for this study and the user's five full-size campaign
limit, and records durable launch intent. It is selected by exact file pins,
including its capacity-classification helper; it is not imported from moving HEAD.

## Preparation commands

Run with the selected runtime Python and set `TMPDIR` to the operator task's
`.tmp`. `SOURCE_REVISION` is the full frozen revision containing only the reviewed
common/profile fixes; `EXPORT_ID` is its first twelve hex digits. Do not use HEAD
as a substitute. `COORDINATOR` is the independently reviewed frozen `study.py`.

```bash
python -B examples/gemmini/phase1/studies/kernel_abstractions/fresh10_tooling_study.py specs \
  --base-spec /operator/task/.tmp/fresh10/fresh-campaign-fresh10-cca.yaml \
  --output /operator/task/.tmp/fresh10-tooling-wave \
  --baseline-audit /operator/task/.tmp/fresh10-tooling-ablation-20261028/baseline/baseline-audit.json \
  --source-revision "$SOURCE_REVISION" --export-id "$EXPORT_ID" \
  --coordinator "$COORDINATOR"
```

The renderer resolves the actual inherited spec chain and pins it. It preserves
full-model validation, private model-spec and instruction-selection arguments,
compiler-library integration and all other native arguments. It selects the new
`fresh10_common` inputs and explicit versioned profiles. C's public references,
manifest and map are `null`; all arms have reference experiments/guidance off.
No input build, author call, hardware job or CPU reservation runs here.

For each of the three generated YAML files, use the selected export's ordinary
launch owner. Its profile and no-reference implementations must already be frozen:

```bash
python -B "$FRAMEWORK/examples/gemmini/phase1/launch/fresh_campaign.py" inputs \
  --spec "$ARM_SPEC" --export-id "$EXPORT_ID"
python -B "$FRAMEWORK/examples/gemmini/phase1/launch/fresh_campaign.py" seal \
  --spec "$ARM_SPEC" --export-id "$EXPORT_ID" --reviewed-by "$REVIEWER" --review-note "$NOTE"
python -B "$FRAMEWORK/examples/gemmini/phase1/launch/fresh_campaign.py" ordering \
  --spec "$ARM_SPEC" --export-id "$EXPORT_ID"
python -B "$FRAMEWORK/examples/gemmini/phase1/launch/fresh_campaign.py" plan \
  --spec "$ARM_SPEC" --export-id "$EXPORT_ID"
python -B "$FRAMEWORK/examples/gemmini/phase1/launch/fresh_campaign.py" preflight \
  --spec "$ARM_SPEC" --export-id "$EXPORT_ID" --execute
```

Then independently qualify actual author, descendant/helper and candidate
boundaries, selected source closure and useful semantic APIs, denied CCA and
private/previous-run access, all promised public data through the real tool
permission boundary, and C's complete reference absence. Compare actual served
TASK/tool/skill prompts; host file existence is insufficient evidence.

The launch review has schema
`merlin.study.fresh10_tooling_launch_review.v1`, exact `source_revision`,
`export_id`, `baseline_sha256`, and `common_workloads_sha256`. Every key in the
module's `REVIEW_CHECKS` must be `true`. Its `arms` list has exactly the declared
arm order, with each row containing:

```json
{
  "arm": "infra",
  "plan_sha256": "SHA256_OF_ACTUAL_PROPOSED_LAUNCH",
  "preflight_returncode": 0,
  "all_passed": true,
  "initial_files": 0,
  "public_abi_files_read": 23,
  "evidence": [{"path": "/canonical/operator/receipt.json", "sha256": "RECEIPT_SHA256"}]
}
```

These are required evidence attestations, not a template to mark successful
before tests. Native receipts must prove every assertion. The operator then runs:

```bash
python -B "$FRAMEWORK/examples/gemmini/phase1/studies/kernel_abstractions/fresh10_tooling_study.py" prepare \
  --study "$STUDY/study.json" --plans "$PLAN_A" "$PLAN_B" "$PLAN_C" --review "$REVIEW"
```

`prepare` validates arguments, profiles, exact native source marker, CPU masks,
helper ceilings, all public counts, normal policy/calibration, common model and
guidance pins, reference asymmetry and evidence hashes before writing
`state: prepared`. The dispatch envelope intentionally retains schema
`merlin.study.xs_ablation.v1`; `study_kind` and `preparation_schema` distinguish this
study. Its `panel` field is an integrity-only alias of `common_workloads`, which
describes the actual native workload. No private 34-cell transfer panel is added
or claimed by these specs.

Only an explicitly authorized launch runs:

```bash
python -B "$FRAMEWORK/examples/gemmini/phase1/studies/kernel_abstractions/fresh10_tooling_study.py" run \
  --study "$STUDY/study.json" --confirm
```

It repeats qualification and verifies all pinned coordinator code before
delegating to the reviewed fleet dispatcher. A failed or interrupted arm is not
automatically restarted.

## Operator measurements

Five operator-only immutable checkpoints are scheduled at author offsets
0, 1800, 3600, 7200 and 14400 seconds. Use a fresh runtime-index file:

```json
{
  "schema": "merlin.study.fresh10_runtime_index.v1",
  "source_revision": "FROZEN_FULL_REVISION",
  "arms": [
    {"arm": "infra", "workspace": "/actual/native/workspace-A", "run_dir": "/actual/native/run-A"},
    {"arm": "scheduling", "workspace": "/actual/native/workspace-B", "run_dir": "/actual/native/run-B"},
    {"arm": "no-reference-kernels", "workspace": "/actual/native/workspace-C", "run_dir": "/actual/native/run-C"}
  ]
}
```

Start the operator observer before authors (it waits for native author clocks).
Use the frozen framework's `src` and `packages/merlin-experiments/src` on
`PYTHONPATH`; the observer refuses an ambient snapshot implementation.

```bash
python -B "$FRAMEWORK/examples/gemmini/phase1/studies/kernel_abstractions/fresh10_tooling_study.py" observe \
  --study "$STUDY/study.json" --runtime-index "$RUNTIME_INDEX"
```

The observer uses `joint_campaign.started_at_epoch`, checks the prepared policy,
rejects clock resets, and copies coherently using the native graded-source hash.
Snapshots are read-only and outside author workspaces. A moving source or missed
historical checkpoint is recorded as unavailable; it never substitutes a later
source under an earlier timestamp. The observer does not change lifecycle,
execute checks, provide author feedback or grant grade credit.

Retain ordinary native round transcripts, lead/helper usage, selfcheck requests
and results, compiler/build/validation logs, performance history, per-case numerical
and model receipts, full-model checks, promotion and physical-validation state.
Join every reported performance point to source, emission, case/scope, numerical
pass, model/facts/domain and settings identities. Report whole-source outcomes;
do not combine earlier per-case winners. Record elapsed time/tokens to coherent
milestones, failed/refused attempts and repair/rework, emitted use of supplied
abstractions, source reuse, and common-case/held-out shape performance. Unobserved
internal API attempts remain unknown or lower bounds.

Full-model/RTL work incomplete at the deadline remains **pending**, with no pass
credit. Provisional Spike timings do not establish RTL/FPGA performance. A later
34-case panel may be a separately pinned, post-author supplemental evaluation of
the same frozen sources without feedback; it is not part of this native baseline.
