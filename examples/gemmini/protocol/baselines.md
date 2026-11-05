# Use the shared contract for baseline comparisons

Start with [`experiment.yaml`](experiment.yaml). All compiler treatments and
baselines use this entry point. Each rule has one authored owner:

| Definition | Authoritative file |
| --- | --- |
| Full networks, checkpoint revisions, tensor interfaces, recurrence, selected inputs, host math and timing boundaries | [`experiment.yaml`](experiment.yaml) |
| Functional tolerances, numerical diagnostics, quality thresholds, reductions and perplexity selection | [`../numerics/models.yaml`](../numerics/models.yaml) |
| Exact integer arithmetic and quantization variant | The three hash-bound [`recipes/`](recipes/) and the selected captured scale bytes |
| Evaluation/calibration selection and preprocessing | [`../numerics/sources.yaml`](../numerics/sources.yaml), with derived input hashes in [`MANIFEST.json`](../numerics/MANIFEST.json) |
| Run-specific compiler, reference, hardware and measurement identities | A completed [`run-manifest.template.yaml`](run-manifest.template.yaml), archived with the results |

Do not copy thresholds into a baseline's own configuration. The shared comparison
code reads the selected numerical-policy file. A resolved launch configuration
and a generated JSON snapshot are records of a selection; edit the authored files
to define a new experiment. Historical sealed configurations keep their original
meaning and receipts.

## Freeze the comparison selection

Pin a Merlin revision, provision the selected input/reference bundle, and export
the shared definitions without capturing, calibrating or running a model:

```sh
python examples/gemmini/protocol/check.py \
  --export "$TMPDIR/gemmini-comparison-contract.json"
```

The export includes the protocol, numerical policy, recipes and source-file
hashes. Collaborators can read it in another language; Python implementations
can use the shared checker below. Keep the snapshot with each result. The export
contains definitions, not model weights, captured scales or reference outputs.
Those assets must be provisioned separately and identified by the same hashes
across implementations. Never publish private inputs or hidden answers here.

The selected workloads are the complete networks in the protocol. A reduced
network, a new quantization recipe, independently recalibrated scales, different
camera count or changed input tokens defines a different numerical profile.
Report such variants explicitly. They can still be compared against FP32 on
paired inputs, but cannot claim to implement the selected quantized program
without its compiled-versus-quantized-reference comparison.

## Compare outputs from any implementation

Each implementation exports its actual arrays as NPZ files, one file per program
invocation. The keys are exactly `out0`, `out1`, and so on, in the protocol's output
order. Preserve shapes and dtypes; do not cast outputs to make a check pass.
The recurrent state and padded channels are required, including outputs outside
the final logits or six semantic action channels.

Copy [`baseline-outputs.template.yaml`](baseline-outputs.template.yaml) into the
run's result directory and bind its paths and SHA256 values, including both the
protocol and its linked numerical-policy and evaluation-input manifests. Paths are relative to that
manifest. Add every program/step in the protocol; obtain their counts and
interfaces from the shared definitions rather than writing another workload
list. Record the same selected input and quantized-reference manifest identities
for every compared implementation.

```sh
python -m merlin.model_check.baseline_numerics \
  --manifest "$RUN/baseline-outputs.yaml" \
  --out "$RUN/numerics.json"
```

This offline command checks the supplied arrays with the same functional
comparison and diagnostic implementations used by Merlin. It does not execute a
model. Its report includes the selected policy/file identities, program/step/output
results, absolute errors and reference RMS. Missing arrays, wrong shape/dtype,
nonfinite values or stale file hashes cannot establish agreement.

For FP quality, add the manifest's optional `quality.fp32` file; SmolVLA's
baseline-relative rule also needs `quality.native_baseline`. These NPZ files use
the judged-output order from `models.yaml`: one `out0` for ResNet/SmolVLA;
TinyLlama `out0` stacks the emitted logits distributions in invocation order.
TinyLlama's input NPZ supplies `tokens`, so the checker can reuse the existing
teacher-forced continuation NLL/perplexity implementation. Perplexity stays a
reported measurement with the policy's target alignment; it has no added bar.

`functional_v1` checks the selected input's complete quantized program. FP quality
is reported separately. Select `quality_v1` only with the policy's complete
visible/sealed cohort and noise-draw coverage. Partial cohorts or missing FP/BF16
measurements remain unavailable. The checker uses the existing per-split quality
rules and reductions; it does not invent a baseline-specific numerical allowance.
Input hashes must match the selected evaluation manifest, so renaming a different
input to an expected case ID cannot satisfy cohort coverage.

The output checker establishes agreement between supplied files. Execution
provenance, the origin of references, accelerator placement and timing require
their own evidence. Retain compiler/build identities, real runtime dispatch,
all eligible Gemmini work, and the measurement scope alongside this report.
Matching final arrays alone does not demonstrate accelerator execution.

## Merlin-integrated implementations

If a baseline provides Merlin's device-package interface, use the normal model
checker with the actual selected models configuration and quantized-reference
mode:

```sh
python -m merlin.model_check check \
  --submission "$DEVICE_PACKAGE" --models "$SELECTED_MODELS_CONFIG" \
  --bundle "$REFERENCE_BUNDLE" --model tiny_llama \
  --split visible --input visible/v00 --mode full \
  --full-comparison quantized-reference --out "$RUN/full-check.json"
```

The selected models configuration must bind the protocol's host-math policies
and reference identities. The generic models-file defaults alone do not select
all those run-specific bindings. Preserve the resolved configuration in the run
manifest. Existing all-output audits can also produce the
[section report](../numerics/README.md#offline-section-reports) without a rerun.

## Compare latency on the same scope

Use `performance_reporting` in the protocol for every implementation. Report
the complete chain and its stages, host computation and required transfers;
keep initialization, compilation, validation dumps and simulator wall time
distinct. Reset per-case recurrent state consistently. Agree and record warm-up
and repetition counts before measurement, and retain the raw per-repetition data.
Label FireSim/gSIM measurements and Spike estimates explicitly, with hardware
and clock identities. A faster result on a different workload, input, quality
cohort or timing boundary is a separate comparison.
