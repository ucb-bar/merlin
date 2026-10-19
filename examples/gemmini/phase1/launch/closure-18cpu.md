# Fresh EL4 closure launch, 18 CPUs

`phase1_el4_config.yaml` is the canonical operator configuration.
`el4-closure-18cpu-v25.yaml` is a compatibility symlink to that same file.
It records the native controller configuration used by
`el4-phase1-closure-20261018-reboot-v1`. It was reconstructed from the live
controller and verified against its argv, initial environment, affinity, selected
file hashes and rendered operator plan. The verification receipt is in the
campaign's `repro-profile/verification.json`. This is an operator-only input,
excluded from author sandbox grants.

`launch_phase1.py` reads the config directly and refuses authoring until the
source revision, config pins, complete private input seal and CPU allocation
verify. It never repairs or seals inputs as part of launching. Until input
preparation finishes and its manifest/sums pins are added, it intentionally
refuses to launch. The current campaign predates this strict entry point and
continues under its original supervisor.

The running compiler author started from an empty submission. There is no seed,
resume, previous compiler, curated kernel provider or handwritten adapter. The
frozen Merlin source is `67f3527265aa5cb9eb303773dc8e8ece39751455`. The profile
was saved after launch; it does not claim to have launched this run. Adding it
does not alter the running harness or its served inputs.

The sealed v25 release contains 161 capsules: 153 public (147 executable and six
expected refusals) and eight private. Capsules are restricted to accelerator
work. Full-model captures select ResNet50, TinyLlama's 22 layers with batch one,
32 prefill tokens and 32 decode steps, and SmolVLA's complete batch-one chain
with a 512-pixel camera, action chunk 50 and ten denoising steps. Existing input
and capture identities remain in the private model selection; they are not
recaptured on each author launch.

Both simulators use the pinned FireSim-aligned configuration. Spike remains
untuned. Native startup verifies the selected release, facts correspondence,
trace readiness and isolation. This launch profile selects inputs; it does not
replace those checks or certify compiled full-model results. Full-model
numerical certification remains pending and no older numerical signoff is
claimed for the newly prepared private dataset.

## Resource and time policy

The allocation is CPU IDs 6–23 inclusive: 13 compute CPUs (6–18), three reference
preparation CPUs (19–21), and two author CPUs (22–23). Native grading allows 12
jobs, gSIM six slots, Spike 12 slots and device builds eight jobs; these share
the compute allocation. Configured capacity does not establish full utilization.
The observed setup bottleneck is serialized per-capsule metadata/custody work.

The native author cap is 7,200 seconds. The separate campaign supervisor enforces
a four-hour total cap, a 75-minute final verification reserve, a 192-GiB
proportional-memory ceiling and 24-GiB host headroom. Those supervisor controls
are not native launch CLI flags. The new entry point reads them from
`budgets.supervisor_backstop_s` and the `SUPERVISOR_*` config variables. The
current campaign records them in `status.json` and `deadline-guard.py`.
For this campaign the author freeze is 2026-10-19
01:48:03 UTC and the absolute stop is 03:03:03 UTC. A new campaign needs newly
computed deadlines and a fresh run ID.

## Check and launch

Use a fresh operator preparation directory containing the pinned
`private-models.yaml`; `--run-id` must be new when launching. This checks only
unless `--run` is explicitly supplied:

```sh
python examples/gemmini/phase1/launch/launch_phase1.py \
  --task-root "$TASK_ROOT" --prep "$PREP" --run-id "$RUN_ID" \
  --var REPO="$REPO"
```

Add `--run` to launch the native controller after all gates pass. Its console
goes to `$PREP/phase1-native.log`; config and private input verification receipts,
controller identity and memory samples are kept in `$PREP`. The supervisor
enforces the configured deadline and memory limits. A controller exit alone
does not establish Phase 1 completion: native formal results and full-model
execution receipts must still satisfy the exit criteria. The new launcher
does not replace the current campaign's supplemental full-chain verification.

## Inspect rendered inputs

Run from the frozen Merlin checkout with its Python packages on `PYTHONPATH`.
`TASK_ROOT` names the task, `REPO` the actual editable source checkout and
`PREP` the operator preparation directory containing `private-models.yaml` and
the archived operator errata. Keep the selected immutable release, simulators,
capture/reference inputs and instruction selection available at the configured
paths. The YAML does not regenerate private artifacts or start a supervisor.

```sh
python -m merlin_experiments.phase1.launch_config render \
  --config examples/gemmini/phase1/launch/phase1_el4_config.yaml \
  --task-root "$TASK_ROOT" --prep "$PREP" --run-id "$RUN_ID" \
  --var REPO="$REPO" --out "$PREP/repro-profile" \
  --plan-template "$PREP/phase1-plan.json"

python -m merlin_experiments.phase1.launch_config verify \
  --config examples/gemmini/phase1/launch/phase1_el4_config.yaml \
  --task-root "$TASK_ROOT" --prep "$PREP" --run-id "$RUN_ID" \
  --var REPO="$REPO" --controller-pid "$CONTROLLER_PID"
```

Live monitoring for the current task is `.tmp/phase1-status` and
`tail -f .tmp/phase1-live.log`. The heartbeat updates every 30 seconds.
Reference preparation counts and materialized capsules are reported separately
from compiler passes. Full-model execution waits for an immutable final
submission and a verified private input seal.
