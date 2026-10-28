---
title: Launch a fresh open-compiler campaign from a reviewed spec
kind: guide
status: current
owner: experiments
last_verified: 2026-10-27
related: [artifact_evaluation, frozen-execution-source]
code_refs:
  - examples/gemmini/phase1/launch/fresh_campaign.py
  - examples/gemmini/phase1/launch/fresh-campaign-fresh2.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-next.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7-reference.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7-paired.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7-raw.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7-cca.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7ref-paired.yaml
  - examples/gemmini/phase1/launch/fresh-campaign-fresh7ref-cca.yaml
  - examples/gemmini/phase1/launch/fresh_campaign_pair.py
  - examples/gemmini/phase1/launch/prepare_author_launch.py
  - examples/gemmini/phase1/launch/prepare_author_library.py
  - examples/gemmini/phase1/launch/prepare_source_release.py
  - examples/gemmini/phase1/launch/resource_reservation.py
  - examples/gemmini/phase1/launch/run_watchdog.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/full_compiler_workspace.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_facts.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/reference_analysis.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/reference_map.py
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

The prepared `fresh-campaign-fresh7.yaml` extends that spec with the reviewed r7
corpus, its correctness roster and its timing-selection bindings. Its campaign
clock, public program selection and provisional runtime retain their existing
settings; inspect the prepared plan before starting it.

Only `launch --confirm` starts a campaign. Every other subcommand prepares
inputs, renders files or reads status.

## Select reference guidance

Use `fresh-campaign-fresh7.yaml` for ordinary Fresh7. Its
`plan.reference_experiments: null` keeps reference feedback inactive. Use
`fresh-campaign-fresh7-reference.yaml` to enable the pinned reference-guided
treatment. Both use the same compiler library and ordinary execution loop.

The optional setting is one explicit pin:

```yaml
plan:
  reference_experiments:
    path: /absolute/operator-only/selection.json
    sha256: <selection SHA256>
```

The path accepts a single selection or the complete `packet-shards.json` generated
by reconstruction. Planning verifies every selected packet and context pin, then
passes the existing `--reference-experiments` and
`--reference-experiments-sha256` options. All tasks remain host-side; the author
receives a bounded projection before the treatment seal. Missing semantic,
numerical or scheduling-consumption evidence remains a construction task.
Reference similarity and static counts confer no performance qualification.

`--reference-guidance off|on` explicitly selects the mechanism when preparing,
checking or launching a plan. `on` requires a pinned reference selection; `off`
does not read it or add reference feedback. The selected value is recorded in
the plan, and execution refuses a different value until the plan is rebuilt.

For the matched raw-reference study, use `fresh-campaign-fresh7-raw.yaml` with
`--reference-guidance off` and `fresh-campaign-fresh7-cca.yaml` with
`--reference-guidance on`. Both inherit the same raw ZIP members, original ZIPs,
instruction listing and upstream Exo sources from one reviewed reference
manifest. They share immutable inputs, SDK, corpus, V90 runtime and budgets;
each has ten disjoint CPUs and separate writable state. Standard Fresh7 retains
its upstream-only defaults.

In the guided arm, authors use the existing selfcheck channel to inspect raw
references and reanalyze emitted candidates:

```bash
python3 agent_selfcheck.py --reference-analysis inventory
python3 agent_selfcheck.py --reference-analysis lift \
  --reference-path reference-kernels/path/to/kernel.txt --reference-format listing
python3 agent_selfcheck.py --reference-analysis candidate \
  --reference-path submission/build/CASE/kernel.dis --reference-format disassembly
```

For candidate code, first use the existing `inspect-kernel` build script with
the returned lowered IR and `--out submission/build/CASE`. The endpoint reuses
the source, ISA, CCA and scheduling analyzers; reports retain
unknown runtime operands and semantic bindings. Pagination uses
`--reference-offset` and `--reference-limit`. The ordinary selfcheck timeout and
resumable `--wait-budget` apply, including verification of frozen input bytes.
The raw arm receives the same files but this additional endpoint is disabled.
Analysis feeds experiment proposals; correctness and performance still require
the existing compilation, validation and measurement path.

### Reference map and withheld prior tasks (fresh7ref)

`fresh-campaign-fresh7ref-paired.yaml` pins an operator-built reference map
(`plan.reference_map: {path, sha256}`, schema `merlin.reference_map.v1`) for
both arms. `plan` verifies its bytes and schema, refuses a map whose
`reference_manifest_sha256` is not this launch's reference manifest, copies it
to `ARTIFACTS/REFERENCE_MAP.json` and passes the pin as `MERLIN_REFERENCE_MAP`;
`fresh_campaign_pair.py` refuses arms whose maps differ. The session stages the
same bytes read-only as `REFERENCE_MAP.json` with one identical TASK.md section
in each arm. Set the spec variable `FRESH7_REFERENCE_MAP_SHA256` to the
reviewed map's sha256 (or pass `--var FRESH7_REFERENCE_MAP_SHA256=<sha256>`);
until then `plan` refuses and names the variable.

Self-check feedback (`feedback/reference_map.py`) adds at most two lines per
case, only for performance targets of the family roll-up (a passing bundle
representative, or a large or connected passing case with headroom), and only
for entries of the same operation kind, graded `exact` (same dimensions),
`same_class` (same shape class under the tile edge from the RTL facts) or
`technique`. The lines appear in `performance_brief.reference_map` and under
the case's step in `next_steps`; regions with hardware loops say to express the
technique with ordinary commands.

`fresh-campaign-fresh7ref-cca.yaml` sets `plan.reference_experiment_tasks:
"off"`: the guided arm keeps the pinned selection, and so the
`--reference-analysis` endpoint and derived tools, but the controller receives
`--reference-experiments-tasks off`. No `REFERENCE_EXPERIMENTS.json` is staged,
no prior task reaches feedback or the idle dispatcher, and the TASK.md section
points at `REFERENCE_MAP.json` and the analysis commands instead.

After both no-author preflights, prepare the concurrent launch through
`fresh_campaign_pair.py prepare --raw-plan RAW/proposed-launch.json --cca-plan
CCA/proposed-launch.json --output NEW_PAIR_DIRECTORY --unit UNIQUE_UNIT`.
It verifies identical raw-reference/input/library/runtime pins, equal budgets
and disjoint writable resources, then generates one existing reservation wrapper
around both ordinary per-arm scripts. This preserves one owner of host CPU-mask
restoration. Each arm retains its own log and state; failure or cancellation
terminates and reaps both process groups. Only `fresh_campaign_pair.py launch
--pair PAIR_DIRECTORY/pair.json --sha256 PAIR_SHA256 --confirm` starts the pair.

To select another corpus, override `FRESH7_REFERENCE_EXPERIMENTS` and
`FRESH7_REFERENCE_EXPERIMENTS_SHA256` with `--var` in the enabled spec. After
changing mode or selection, run `plan` and `preflight --execute` again with that
spec, then use the same spec for `launch --confirm`. Execution refuses a plan
prepared under a different spec. A running campaign's sealed mode cannot change
on resume; use a separately recorded run or continuation.

## Stages

Run each stage under the Phase 1 runtime Python, with the worktree on
`PYTHONPATH` (`src:packages/merlin-experiments/src:packages/merlin-timing/src`):

```bash
L=examples/gemmini/phase1/launch; S=$L/fresh-campaign-fresh2.yaml
$PY -B $L/fresh_campaign.py export    --spec $S   # git archive HEAD + .env + host-lane package
$PY -B $L/fresh_campaign.py inputs    --spec $S   # author library (fresh2: V20 as recorded; next: V22), full compiler, base references, corpus
$PY -B $L/fresh_campaign.py seal      --spec $S   # corpus inspect, then the reviewed seal
$PY -B $L/fresh_campaign.py ordering  --spec $S   # completion-order selection from the run's facts (next)
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
- `ordering` derives the command-ordering selection from the facts and contract
  in `plan.environment` (`MERLIN_RTL_FACTS`, `MERLIN_TARGET_CONTRACT`) and writes
  `plan.ordering_selection`; it reuses one already bound to those bytes. `plan`
  refuses a spec whose selection is missing, copies it next to the plan and pins it
  as `MERLIN_COMPLETION_ORDER_SELECTION`. Two consumers read that pin: the
  completion-order check, and the session's capability selection
  (`OPTIMIZATION_SUGGESTIONS.json`), which derives queue, asynchronous-issue,
  dependency, configuration, residency and strided-row capabilities from it
  when it binds the staged facts and contract. Run it after `inputs`: a selection
  derived for another bundle leaves those capabilities unknown.
  When `ordering.command_recorder` selects an engine, its matching installed headers
  and an output directory, this step also builds the operator's compact command
  recorder from the frozen framework. `plan` rechecks its source, engine and library
  digests and pins `MERLIN_COMMAND_RECORDER`. Unsupported or incomplete recorder
  observations use the ordinary commit-log check; the recorder is withheld from
  compiler authors. Specs without this selection retain commit logging.
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

## Run health and the quiet host

A reserved run degrades silently when the host is busy: a disk scan
saturating shared I/O, hundreds of squashfs mounts serializing sandbox starts
on the mount lock, or one reader of `/proc/locks` stalling every file-lease
operation. `run_watchdog.py` reports these. It never remediates.

- `preflight` prints one `HOST HEALTH` line. It warns about high I/O pressure,
  more than 150 squashfs/loop mounts or another live reservation, and never
  refuses.
- `launch` (and `fresh_campaign_pair.py launch`) start `run_watchdog.py watch
  --unit UNIT` at nice 19 on CPUs outside every reservation. Pass
  `--no-watchdog` to skip it. The watchdog samples every 60 s and prints a
  warning when a condition holds for five samples. Conditions: reserved CPUs
  under 25% busy while the newest grade is over 20 min old, 10 or more lock or
  mount-lock waiters, sandbox starts stuck in D state, high I/O pressure and
  the mount count. It writes `watchdog-status.json` under
  `<task>/.tmp/<run-id>/operator/` and exits when the unit ends. The
  thresholds are constants at the top of the script.
- The reservation wrapper keeps an entry in `RUN_IN_PROGRESS.json` (default
  `/work/merlin/`, or `MERLIN_RUN_MARKER`) while it runs. Other sessions call
  `run_watchdog.py check` before a disk-wide scan or a new mount. It exits 3
  while any reservation is live, including one launched without a marker.

The watchdog never reads `/proc/locks`. Don't read it yourself on a host
running a lease-cache experiment.

## Reproduction check

`test_fresh_campaign.py` plans the fresh2 spec against the recorded export when
it is on the host. It writes only under the test's temporary directory. The
planned argv, environment, policy, library contract, timing selection and
guidance are identical to the recorded ones. Three things differ on purpose:

- The `env -i` assignments are sorted, because `reservation_argv` sorts them.
- `full_compiler_manifest_sha256` records the manifest the argv pins. fresh2
  carried v8's stale value.
- A new `launch_spec` field pins the spec file and its sha256.
