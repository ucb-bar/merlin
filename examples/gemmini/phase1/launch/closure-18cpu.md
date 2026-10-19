# Fresh EL4 closure launch, 18 CPUs

`phase1_el4_config.yaml` is the canonical operator configuration.
`el4-closure-18cpu-v25.yaml` is a compatibility symlink to that same file.
It selects an empty-origin author, sealed v26 accelerator inputs, the native
controller, full-model verification, and the resource limits below. These files
are operator inputs, excluded from author sandbox grants. The earlier
`el4-phase1-closure-20261018-reboot-v1` campaign retains its original config and
verification receipts in its own preparation directory; edits here do not
change its historical score or certify its incomplete model checks.

`launch_phase1.py` reads the config directly and refuses authoring until the
source revision, config pins, complete private input seal and CPU allocation
verify. It also requires a byte-bound readiness receipt. The original
`merlin.phase1.full_model_readiness.v1` receipt still requires all three retained
models to pass. The separate `merlin.phase1.infrastructure_readiness.v2` receipt
admits authoring after current infrastructure and reference checks, preserving
the old compiler's failed or partial results as historical diagnostics. A failed
retained compiler does not qualify a fresh compiler or certify current references.
The launcher never repairs or seals inputs. A missing or mismatched receipt
refuses authoring.

Preparation also invokes the existing compiler-independent private source
preflight before starting the author. Every selected model and program must have
verified source inputs under the selected facts, source contracts and host build
observation. The source-only report deliberately grants no candidate or retained
compiler qualification. Its private report and frozen sources remain under
`source-preflight/` in the preparation directory. Repeated preparation reuses
them only after reopening the exact source/input/tool identity and retained
output bytes; changed inputs select a fresh check, and modified receipts refuse.

Each new compiler author starts from an empty submission. There is no seed,
resume, previous compiler, curated kernel provider or handwritten adapter.
`variables.SOURCE_REVISION` selects the source baseline; the launcher checks
that only separately pinned operator metadata has changed since that revision.
The retained compiler and readiness reports are private diagnostics and are
not available to the fresh author.

The sealed v26 release contains 161 capsules: 153 public (147 executable and six
expected refusals) and eight private. Capsules are restricted to accelerator
work. Full-model captures select ResNet50, TinyLlama's 22 layers with batch one,
32 prefill tokens and 32 decode steps, and SmolVLA's complete batch-one chain
with a 512-pixel camera, action chunk 50 and ten denoising steps. Existing input
and capture identities remain in the private model selection; they are not
recaptured on each author launch.

Both simulators use the pinned FireSim-aligned configuration. Spike remains
untuned. Native startup verifies the selected release, facts correspondence,
trace readiness and isolation. This launch profile selects inputs; it does not
replace those checks. A model passes only after its full chain executes,
every accelerator-kernel replay compares exactly, and all declared model
outputs match the sealed quantized reference. Floating outputs use fixed
`atol=2e-6`, `rtol=1e-5`, with all values finite; integer and boolean outputs
compare exactly. Shape, dtype and output count must agree. These thresholds
are selected before execution, not adjusted after observing results.

This explicitly selected comparison does not need a redundant CPU-only Spike
twin or a fresh NumPy recomputation of the entire reference chain. Neither is
reported as passing. Comparisons against the original floating model remain
quality diagnostics. The operator checker records its selected contract version;
the immutable release contract remains valid for its existing commands.
The selected reference policies are `native` for ResNet50,
`portable_exp_trig` for TinyLlama and `portable_neural_f32` for SmolVLA.
The last policy applies only to float32 operations; float64 operations retain
the native implementation. Reference derivation and final compiler qualification
have separate receipts.

## Infrastructure admission

The v2 receipt pins current source, tools, private specification, complete input
seal and numerical-reference policy. It cross-checks each catalog's source graph
against the selected capture, preselection and execution attestation. An older
capture with different input dimensions cannot certify the selected workload.
Native preflight retains its full capture-tree and hardware-correspondence checks.

Positive target evidence comes from a private infrastructure probe, separate from
the public accelerator capsule corpus. Its ordinary full-model build is relinked
with two different inputs and executed on the selected Spike configuration. The
validator reopens original and relink compilation recipes, tools, ELF files,
input arrays and raw observed process output. It decodes every output through
Merlin's normal readback and recomputes strict comparisons for mixed int32/float32
arrays; changed inputs must produce changed outputs. Source-only tests and
asserted JSON pass flags do not establish target execution. The generic probe
may use host code and does not qualify any accelerator compiler implementation.

A separate bounded nested-process timeout records captured child identities and
the actual guardian cleanup result. Admission checks the pinned observations and
rejects any surviving observed process. Both probe reports name the exact
collector pinned as `runtime_pins.operator_infrastructure_witness`.

`full_model_verification.build_infrastructure_receipt(launch, retained, diagnostics,
target_witness, cleanup_witness)` builds and validates the receipt from artifact
paths. `diagnostics` maps every selected model to its original report and invocation
pin; stopped diagnostics retain their partial wrapper. Historical source/input
identities remain unchanged. The caller publishes the returned JSON and pins its
SHA256 in the canonical config. Final model qualification still requires all three
complete models on the exact newly authored compiler. A v2 readiness receipt is
only permission to begin that experiment; it is not Phase 1 completion.

## Resource and time policy

The allocation is CPU IDs 6–23 inclusive: 16 compute CPUs (6–21) and two author
CPUs (22–23). Reference derivation finishes before launch and reserves no CPUs
during the campaign. Native grading allows 12
jobs, gSIM six slots, Spike 12 slots and device builds eight jobs; these share
the compute allocation. Configured capacity does not establish full utilization.

The native author cap is 7,200 seconds. The separate campaign supervisor enforces
a four-hour total cap, a 75-minute final verification reserve, a 192-GiB
proportional-memory ceiling and 24-GiB host headroom. Those supervisor controls
are not native launch CLI flags. The new entry point reads them from
`budgets.supervisor_backstop_s` and the `SUPERVISOR_*` config variables. The
supervisor records the actual deadlines, process identities and memory samples
in the preparation directory. A new campaign always needs a fresh run ID.
Retained-compiler readiness is a separate bounded diagnostic before this cap.
An outer operator guard also applies the original overnight deadline to every
attempt; restarting the campaign does not reset that deadline.

The selected release requires public compilation-scale checks during authoring.
Five type-only convolution scales share one 120-second compilation deadline,
with a 1-MiB limit on LLVM function code. Constant tensor globals do not count
toward that code limit. These checks expose expansion failures without allocating
full model inputs or running additional numerical simulations. The background
checks use at most two workers in the same shared compute pool.

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
controller identity and memory samples are kept in `$PREP`. Once authoring
stops, three full-model checks start on the immutable compiler snapshot in
parallel with the native final grade. Their actual model commands and pins
are retained. A changed candidate supersedes those checks. Completion requires
the checked snapshot to match the native ledger and final published compiler.
The supervisor owns both native and full-model process trees and enforces one
deadline and memory policy across them. A controller exit alone does not
establish Phase 1 completion.

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
`tail -f .tmp/phase1-live.log`; inspect their selected preparation directory
before attributing a status to a new run. The heartbeat updates every 30 seconds.
Reference preparation counts and materialized capsules are reported separately
from compiler passes. Full-model execution waits for an immutable final
submission and a verified private input seal. During authoring,
`qa/workflow_conformance.json` updates approximately every five seconds with
public workflow evidence and source locations. It does not run another grade
or disclose private grading answers.
