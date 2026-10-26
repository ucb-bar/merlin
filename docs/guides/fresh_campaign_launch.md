---
title: Launch a fresh open-compiler campaign from a reviewed spec
kind: guide
status: current
owner: experiments
last_verified: 2026-10-26
related: [artifact_evaluation, frozen-execution-source]
code_refs:
  - examples/gemmini/phase1/launch/fresh_campaign.py
  - examples/gemmini/phase1/launch/fresh-campaign-fresh2.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-next.yaml
  - examples/gemmini/phase1/launch/prepare_author_launch.py
  - examples/gemmini/phase1/launch/prepare_author_library.py
  - examples/gemmini/phase1/launch/prepare_source_release.py
  - examples/gemmini/phase1/launch/resource_reservation.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/full_compiler_workspace.py
---

# Launch a fresh open-compiler campaign from a reviewed spec

`examples/gemmini/phase1/launch/fresh_campaign.py` runs a fresh (unseeded)
open-compiler campaign from start to finish. It takes one YAML launch spec.
The spec `fresh-campaign-fresh2.yaml` reproduces
`gemmini-open-compiler-20261026-fresh2`. For a new run, use
`fresh-campaign-next.yaml` with `--run-id`. It extends the fresh2 spec and adds
the settings the launch builder gained after fresh2:

- `--author-helper-effort medium`;
- `gsim_measurement.long_max_jobs: 4` for the long gSIM measurement lane;
- no inherited seal note, so pass `--review-note`;
- the reservation unit derived from the run id, unless `--unit` selects one.

Only `launch --confirm` starts a campaign. Every other subcommand prepares
inputs, renders files or reads status.

## Stages

Run each stage under the Phase 1 runtime Python, with the worktree on
`PYTHONPATH` (`src:packages/merlin-experiments/src:packages/merlin-timing/src`):

```bash
L=examples/gemmini/phase1/launch; S=$L/fresh-campaign-fresh2.yaml
$PY -B $L/fresh_campaign.py export    --spec $S   # git archive HEAD + .env + host-lane package
$PY -B $L/fresh_campaign.py inputs    --spec $S   # author library (fresh2: V20 as recorded; next: V22), full compiler, base references, corpus
$PY -B $L/fresh_campaign.py seal      --spec $S   # corpus inspect, then the reviewed seal
$PY -B $L/fresh_campaign.py plan      --spec $S   # policy, pins, proposed-launch.json
$PY -B $L/fresh_campaign.py preflight --spec $S --execute   # no-author --preflight-only run
$PY -B $L/fresh_campaign.py launch    --spec $S --confirm   # reserved campaign (systemd unit)
$PY -B $L/fresh_campaign.py check     --spec $S [--watch | --until HHMM] [--console]
```

- `export` refuses a dirty tree or an export that already exists. It names the
  export after `HEAD[:12]`. Later stages take `--export-id` (default: the repo's `HEAD`).
- `inputs` starts the preparation runtime on the preparation CPUs at `nice 5`.
  That runtime is the campaign environment plus `inputs.environment`. The
  export's own sibling modules then build every input:
  - `prepare_author_library` builds the library at the spec's inventory version.
  - `full_compiler_workspace` builds the full compiler.
  - `prepare_source_release` builds the corpus on the spec's common substrate.
  - The public references are copied from the base manifest. A member under a
    `forbidden_reference_prefixes` entry (`measured-exo/`) refuses the
    preparation before anything is copied.
- `seal` writes `inspect.json`, then seals with that digest and the spec's
  reviewer and note. `--inspect-only` stops after the inspect step.
- `plan` writes `policy.json`, `author-guidance.md`, `performance-selection.json`,
  `selected-library-contract.json` and `proposed-launch.json` under `${ARTIFACTS}`.
  It edits the base policy as follows:
  - drops `compiler_origin`;
  - sets the clock from the run section;
  - sets the gSIM worker budget to the harness default for that clock;
  - merges the performance scope's `excluded` ids, sorted and unique;
  - fills `plan.policy.defaults` into keys the base policy leaves unset.

  The reserved command comes from `prepare_author_launch.reservation_argv`,
  with the spec's unit, state and log. `--author-helper-effort` overrides
  `run.author_helper_effort`; `null` passes no flag.
- `preflight` writes the no-author script, using the plan's exact command with
  `<run-id>-preflight` and `--preflight-only`. With `--execute`, it first moves
  earlier preflight state aside and then runs the script under the time limit.
  `--run` writes the unreserved full-run script instead. That script keeps the
  open-file limit and the wall-clock bound that the reservation would set.
- `check` is read-only. It reports:
  - whether the unit is active;
  - the summary, if there is one;
  - error markers in `campaign.log` (in-order fragments, case-insensitive);
  - free disk;
  - QA progress and recent self-check responses.

  `--watch` returns when the campaign needs attention.

## The spec

- `variables`: every path, as `${NAME}`. Override one with `--var NAME=VALUE`.
  `RUN_ID`, `RUN_TAG` (the run id's last word, for example `fresh2`), `UNIT`,
  `CPUS`, `PREPARATION_CPUS`, `EXPORT_ID` and `PREFLIGHT_RUN_ID` are built in. They come from `run` or the command line
  (`--run-id`, `--unit`, `--cpus`, `--export-id`). `HOME` and `USER` come from
  the invoking environment.
- `extends`: a parent spec, relative to this file. Mappings merge key by key;
  lists and scalars replace. The plan's `launch_spec` record pins this file and
  every parent by sha256.
- `plan.argv` and `plan.environment`: the campaign command and runtime, written
  out in full. They are derived from v8's recorded launch: no seed,
  `--fresh-backend`, `--author-token-unbounded`, `MERLIN_CAPSULE_ORDER=information-first`,
  and a private `MERLIN_BUNDLE_CAS` under `${RUN_STATE}`. A reviewer reads
  the exact settings rather than a rewrite rule.
- `plan.required_flags`, `forbidden_flags`, `forbidden_fragments` and
  `input_flags`: fresh-launch invariants. `plan` refuses to write when one fails.
- `run.workers`: an optional cap on the parallel settings, the simulator job
  count and the policy's gSIM `max_jobs`.

## Reproduction check

`test_fresh_campaign.py` plans the fresh2 spec against the recorded export when
it is on the host. It writes only under the test's temporary directory. The
planned argv, environment, policy, library contract, timing selection and
guidance are identical to the recorded ones. Three things differ on purpose:

- The `env -i` assignments are sorted, because `reservation_argv` sorts them.
- `full_compiler_manifest_sha256` records the manifest the argv pins. fresh2
  carried v8's stale value.
- A new `launch_spec` field pins the spec file and its sha256.
