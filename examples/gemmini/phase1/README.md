# Phase 1: functional compiler

Start from [the experiment definition](../experiment.yaml). Its
[target descriptor](../target/descriptor.yaml) selects the authored prompts in
`task/`, independently of retained harness and bundle resources.

The realistic mode uses `TASK_realistic.md`; the legacy fullsuite and one-shot
launchers use `TASK_full.md` and `TASK.md`. `TASK_pilot.md` is a retained authored
reference. Generated-prompt modes use Merlin's shared renderer. Every run archives
the exact final prompt served, including its selected arm's additions.

Prepare and review a fresh corpus release through the
[reviewed handoff](../../../experiments/README.md#reviewed-phase-0-handoff)
before verified execution. Preparation copies these inputs into the release and
removes the live source pointers. New bundles grant the declared task directory;
old bundles or frozen runs are not rewritten to adopt this layout.

## What the functional finish line must prove

Passing generated operation capsules is necessary but not sufficient to claim a
network compiler. For each selected whole-network capture (including ResNet-50
and a specified full SmolVLA graph, not only one denoise step), a functional
Phase 1 result needs a complete
operator inventory, an accounted route for every region (Gemmini or a declared
host lane), a whole-model compile with no unsupported-op escape, executable
artifacts, numerical comparison against an independent framework reference,
and observed dispatch evidence that the admitted accelerator work actually ran.
Keep the capture, quantization scheme, weights/manifest, compiler submission
hash, intermediate MLIR and run receipts together; a storage dtype alone does
not establish the arithmetic or accelerator placement. The
[whole-model example](../whole-model/README.md) explains IR inspection, but its
lowering smoke is not this functional certificate.

The current example does **not** claim that finish line has been reached.
The descriptor makes `M2_microvit_gemmini`, `M3_host_island_seam_gemmini`, and
`SY_micro_model` mandatory admitted L3 representatives, while
`SY_model_resnet50`, `SY_model_smolvla`, and the SmolVLA denoise-step capstone
are resource-excluded from mandatory L3 simulation. Exclusion is a cost policy,
not proof of a compile failure or a successful compile. Inspect a real
whole-model compilation receipt before asserting either. The target's
`workload_spec.models` also keeps ResNet-50 as a held-out generalization claim;
do not use its capture to derive the tests against which it is evaluated.

Before attempting that claim, inspect an explicit capture with the
[whole-model preflight](../whole-model/README.md#check-model-readiness). It
compares declared target routes with the operand formats and accelerator groups
in the captured graph. It cannot produce a compiler certificate: a requested
`int8` deployment format does not quantize an FP32 or BF16 model by itself.

This is deliberately separate from [Phase 2](../phase2/README.md): Phase 1
establishes functional compiler capability and a frozen submission; Phase 2
holds that functional bar fixed while optimizing cycles, placement, and
whole-model cost under its own measured or model-portfolio evidence.

## Run the installed Phase 1 controller

The catalog ID `gemmini-functional` selects the single definition above. Its
`treatment: rtlchecks` selects the installed Phase 1 module with explicit bundle
identity, bundle manifest and oracle-timing path. When copying the definition for
a reviewed release, select the release's descriptor, `corpus_seal`, and generated
RTL-checks bundle manifest together. The retained manifest path is a preparation
reference, not proof that its inputs are current or reviewed. Inspect and preflight
the copied definition before running it. Provision its declared toolchains and replace
legacy directory-symlink grants with explicitly owned input trees in a newly generated,
reviewed bundle; the frozen-input check deliberately refuses incomplete closures.

The direct installed CLI below selects the same treatment. For an installed
**baseline** catalog route instead, use the shared
[`baseline-functional-template`](../../../experiments/definitions/baseline-functional-template.yaml)
with its required operator inputs; changing treatment changes the experiment.

For direct invocation, set the variables below to actual operator-selected inputs.
`CORPUS_SEAL` is the release's `private/seal.json`; `DESCRIPTOR` must belong to
that release. `RESOURCE_ROOT` resolves declared resource paths. `BUNDLE_ID` must
match `BUNDLE_MANIFEST`; use reviewed RTL-checks inputs, not an invented bundle.
`ORACLE_TIMING` must name an existing operator-owned timing record. The example
path is not supplied here: provision a genuine record or select an existing one;
do not fabricate an empty placeholder or change it after freezing a run.

```sh
MERLIN_CORPUS_SEAL="${CORPUS_SEAL:?}" python -m merlin_experiments.phase1 \
  --descriptor "${DESCRIPTOR:?}" --repo "${RESOURCE_ROOT:?}" \
  --bundle "${BUNDLE_ID:?}" --bundle-manifest "${BUNDLE_MANIFEST:?}" \
  --oracle-timing "${ORACLE_TIMING:?}" \
  --run-id "${RUN_ID:?}" --arm merlin_assisted --treatment rtlchecks \
  --driver claudecode --provider subscription --model "${MODEL:?}" --effort high \
  --schedule continuous --max-wall-s 43200 --round-timeout 43200 --grade-interval 900
```

This starts authoring and grading: approve the provider and budget before running
it. The budget matches the example definition; reduce it deliberately if needed.
Provision FileCheck on the selected execution PATH, and supply required toolchain
library paths and credentials explicitly. This installed route does not inherit
the old launcher's LLVM/Chipyard FileCheck candidates or compatibility-library defaults.
Use `python -m merlin_experiments.phase1 --help` for inspection only. The configured
output root owns the run. Resume the same invocation with `--resume` only while
its frozen inputs remain valid; changed inputs require a fresh run.

Installed ownership is not a sandbox or compiler certificate. Verified execution
still requires real bwrap isolation, nonvacuous answer masking, qualified tools and
public/frozen/hidden grading in the prescribed order. Diagnostic unsandboxed runs
cannot become Phase 2 inputs by waiving these integrity gates. Retain the exact
submission hash and qualification evidence for the [Phase 2 handoff](../phase2/README.md).

Compatibility links at the former source paths are for navigation only. Regenerate
and review bundles instead of relying on those links as sandbox grants. Generated
capsules, workspaces, compiler payloads and certification records remain artifacts,
not files in this example. The remaining harness resources still require a checkout.

## Public runtime harness

`contracts/harness_curated/gemmini-rocc-tests/` contains the supplied headers,
linker scripts and runtime scaffolding used by the compiler experiment. The
descriptor's `contracts_root` selects this example-owned tree; generated bundles
and local environment files keep their separately declared resource locations.
Vendor licenses and the three internal linker-script aliases are preserved.

The former harness directory is a compatibility link for retained experiment
variants. Gemmini Universal keeps its own parameter overrides while linking shared
headers to this owner. The pinned G3 batch descriptor is unchanged. These links do
not authorize verified execution of old frozen runs; prepare and review fresh inputs.

This does not relocate or qualify the external Chipyard ISA/RTL bring-up links.
Those still contain machine-specific paths and require a separate pinned-source
provisioning migration. No external referents were copied into this example.
