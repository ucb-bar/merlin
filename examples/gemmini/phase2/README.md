# Phase 2: performance optimization

Phase 2 consumes a frozen functional compiler; it does not tune Merlin's shared
implementation for Gemmini. Target-specific generated compiler code stays in the
run's candidate payload and may later be published to the target's OOT repository.
Its workloads answer a different question from Phase 1's network-compile bar:
they measure placement, cycles and model-level cost after functional capability
has been established. A fast Phase 2 objective is not a substitute for a
successful ResNet-50 or SmolVLA whole-model compile and correctness receipt.

Start from the [Phase 1 workflow](../phase1/README.md). There are two distinct
experiment modes, each with one shared definition template:

| Mode | Template | Required handoff |
| --- | --- | --- |
| Measured claims | [`measured-claims-template.yaml`](../../../experiments/definitions/measured-claims-template.yaml) | Frozen functional run ID and submission hash, descriptor, RTL facts, performance profile, both declared GSIM certificates and their exact hashes |
| Model portfolio | [`model-portfolio-template.yaml`](../../../experiments/definitions/model-portfolio-template.yaml) | Deployment record, frozen campaign configuration, writable candidate checkout and explicit authoring budgets |

Do not interpret a model-portfolio estimate as measured hardware performance.
The measured mode's admission gates validate the functional handoff and timing
evidence; a zero exit status from Phase 1 is not a substitute for those records.

The measured-claims catalog adapter selects the installed managed Chia envelope
and checkpoint coordinator. The model-portfolio adapter selects the installed
portfolio launcher with explicit deployment inputs. The model-portfolio adapter
retains its native campaign and checkpoint semantics; the shared catalog only
freezes and dispatches its declared inputs. See the [installed explicit-resource guide](installed-measured-claims.md)
for deployment inputs and direct coordinator diagnostics; neither route alone
establishes a qualified deployment. The global
model-portfolio and corpus measured-claims handoffs are different schemas; do not
substitute one for the other.

## Author one explicit definition

Copy the selected template into your own experiment input directory, not into
`out/` as an execution script. Keep the template's `kind: template` while editing.
Set a unique `id`, set `target: gemmini`, and supply every operator-owned input
and budget. Resolve paths relative to the new definition's location, or use
explicit absolute paths. There is no placeholder substitution or template
inheritance: edit the YAML values themselves.

For measured claims, select the descriptor associated with the frozen functional
run, not the mutable example descriptor merely because its target name matches.
Take submission and certificate hashes from their recorded evidence; never
rehash edited payloads and present them as the original certification. New verified
execution requires the current frozen-input provenance, including source ownership
and private support classification. Preserve older runs for inspection.
Also select the provisioned managed supervisor endpoint, installed package roots,
contract resources, holdout catalog, functional-run root and new stage/measurement
destinations. The template keeps these separate from scientific evidence. Never
use a generated output directory as an immutable input root containing its own
future outputs. The catalog selects the installed wrapper and current interpreter.

For model portfolio, select the exact campaign inputs, candidate checkout and
`merlin.portfolio-deployment.v1` record described in the
[catalog deployment guide](../../../experiments/README.md#definitions-and-execution).
Declare additional immutable roots in `inputs` when orchestration must also pin
them. The installed engine validates its nested input closure. Set explicit
agent budgets; do not inherit a long-running study's budget accidentally.

After completing the definition, change `kind` to `experiment`. Inspect and
preflight it through the existing interface:

```sh
merlin experiment inspect /absolute/inputs/gemmini-performance.yaml --phase 2
merlin experiment preflight /absolute/inputs/gemmini-performance.yaml --phase 2
```

Templates intentionally cannot execute. Preflight checks local configuration,
not live simulator readiness. Only when inputs, managed execution resources and
costs are approved, launch explicitly:

```sh
merlin experiment run /absolute/inputs/gemmini-performance.yaml --phase 2
```

The runner allocates a fresh directory beneath the configured run root and records
resolved inputs and process attempts. Do not create launch scripts in generated
output folders. See the shared [execution and reporting guide](../../../experiments/README.md#definitions-and-execution)
for each mode's deployment requirements, evidence reporting and checkpoint/resume rules.
Compiler payloads remain immutable; certification and publication records belong
alongside them, not inside an edited certified manifest.

The legacy FireSim GEMM-window study has its queue budget, contention and failure
policy in [firesim_loop.py](firesim_loop.py). This is example-owned policy for that
board and workload, not a Merlin core default. It submits nothing on import;
the study's launcher opts into it when running a measured batch.

The [primitive probe](primitive_probe.py) is likewise an example-owned Gemmini
diagnostic for splitting a decoded RoCC short kernel into setup, compute and
readback. It is not part of Merlin's shared Phase 2 API or the measured-claims
runner; use the target-neutral provider interface for new accelerators.
