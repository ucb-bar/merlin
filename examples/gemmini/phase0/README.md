# Gemmini: deterministic Phase 0

Start with [experiment.yaml](../experiment.yaml), catalog ID `gemmini-functional`.
Phase 0 has no agent: selected input bytes and pinned tools determine extraction,
requirements and candidate capsules. Unknown semantics remain explicit blockers.
Phase 1 develops the functional compiler; Phase 2 uses a separate performance cohort.

For a fresh AWS experiment, use the [neutral worker preparation guide](../../../docs/guides/aws_phase0_phase1.md)
and its operator-only deployment and verified-definition templates. This retained
diagnostic example does not supply a reviewed execution contract or native launch
evidence. The shared neutral tooling needs explicit contract/facts data and no
handwritten support package.

## 1. Select the inputs

| Input | Responsibility |
| --- | --- |
| [Software spec](../target/software-spec.yaml) | Operation signatures, numerical semantics, placement/transfers and quantization eligibility |
| [Hardware selection](../target/hardware.yaml) | Source-production requirements and direct RTL audit questions |
| [Host capabilities](../target/host-capabilities.yaml) | Separately pinned scalar Rocket host package and per-operation semantic declarations; the approved typed dequantization rule does not qualify Rocket execution |
| [Recipe](recipe.yaml) | Derived-only policy, comparison tolerances and oracle tiers; no authored capsule list |
| [Descriptor](../target/descriptor.yaml) | Independent iteration roster, held-out validation roster and experiment resources |
| [Selected capability contract](../target/contracts/target_contract.yaml) | Prototype command order and runner intent, explicitly frozen by the experiment; not an OOT support certificate |
| Shared performance template | Phase 2 objectives and families, not additional functional capability |

The selected configuration has signed 8-bit operands, a 20-bit MAC result and
32-bit accumulator storage. An int32 mathematical golden needs a no-overflow
proof or an independently qualified width-aware model; storage width is not
compute precision.

The preferred later timing board is the content-pinned U250
`FireSimGemminiRocketConfig` in [Phase 2](../phase2/whole-model-machines.yaml).
This Phase 0 selection currently extracts the `GemminiRocketConfig` Verilator
elaboration. They are separate artifacts and their generator pins name different
revisions; the shared Gemmini configuration name alone does not prove their
facts identical. A board-level claim must compare the selected facts against
the FireSim source/elaboration or regenerate facts from a matching source
selection before treating them as one hardware revision.

[regression-seeds.yaml](regression-seeds.yaml) preserves historical authored tests
for explicit compatibility studies, excluding a former SmolVLA-derived seed
that crossed the held-out validation boundary. It is **not** a default
derivation input.
The small hand-authored g0–g2 interface samples in [reference/](reference/)
exercise the historical OOT contract; they are not a generated Phase 0 release.
Generated capsules, weights and goldens are artifacts, never committed examples.
Read [the shared specification guide](../../../docs/guides/phase0_specification.md)
for what belongs in a SW spec versus extracted evidence.

## 2. Produce and audit one coherent RTL source selection

Follow the exact source production and audit commands in
[the target input guide](../target/README.md). The source bundle binds selected
FIRRTL, prepared generation inputs, the selected `firtool`, generated HW-MLIR and
module hierarchy. Extraction consumes that bundle explicitly; it must not silently
combine unrelated cached elaborations. Save `source-selection.json`, `facts.json`
and `validation.json` under a fresh configured artifact root.

Source consistency proves provenance, not operation legality or numerical agreement.
Manual RTL audit improves the deterministic extractor; it is not an agentic runtime
step in Phase 0. Keep raw facts separate from the effective consumer views.

## 3. Capture the independent iteration workloads

Use [the four shared loaders](../../workloads/README.md): `coverage_mlp`,
`residual_cnn`, `causal_decoder` and `multimodal_policy`. The worker seeds Python,
NumPy and Torch before loading the model and requires deterministic algorithms.
Use the same explicitly selected model2MLIR interpreter/build for all four.
The integerized CNN MLIR can erase the im2col window, so convolution geometry is
read from the original PyTorch graph in that capture's receipt-bound frontend
trace. No second workload or held-out model is used to derive this obligation;
missing or inconsistent trace geometry is reported, not guessed.

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$CAPTURE_ROOT/coverage_mlp"
```

Repeat with the other three loader names and distinct output directories.
`--materialize-bundle` writes `model.mlir`, external `weights.safetensors`,
inputs/goldens and `capture_receipt.json` from the **same conversion and model
instance**; it does not recapture an unrelated model. Inspect `frontend-trace.json`
and `pytorch-opset.json` for source correspondence and build-specific operator scope.
The direct worker command is a development capture, not a verified source-closure
receipt. For a fresh, checkpoint-free capture whose source and runtime bytes must
be selected *before* execution, use the explicit sealed diagnostic workflow:

```sh
merlin experiment corpus capture select \
  --m2m-root "$MODEL2MLIR_ROOT" \
  --workload-root examples/workloads/coverage_mlp \
  --venv "$MODEL2MLIR_VENV" --dtype fp32 \
  --run-dir "$CAPTURE_ROOT/sealed-coverage-mlp" \
  --output "$CAPTURE_ROOT/selection-coverage-mlp"
merlin experiment corpus capture issue \
  --selection "$CAPTURE_ROOT/selection-coverage-mlp/capture-selection.json" \
  --expected-sha256 "$SELECTION_SHA256"
```

Record the exact `sha256` returned by `select` as `SELECTION_SHA256`; both
directories must be absent before selection. Inspect the owner-only
`capture-selection.json`, `sealed_m2m_pending.json` and captured sidecars in
the run directory. Repeat separately for each iteration workload. When deriving
from those exact captures, pass a matching
`--application-capture-selection "LABEL=PATH@SHA256"` for **every** roster label;
mixing selected and legacy captures is refused. This establishes an auditable
preselection and replay. Those records alone **do not grant Phase 0 admission**.
The separate sealed-M2M attestation can admit a fresh replay only after the
selected source, runtime and output bytes pass its policy-restricted sandbox
checks; Phase 0 re-verifies that attestation from disk. Its explicit policy
accepts an unsigned receipt and a copied, rather than independently pinned,
Python runtime. A host that cannot create the required network namespace
cannot issue this attestation. External checkpoints are not supported yet.
The worker anchors `--out` before Model2MLIR writes its weight reference, so a
relative command-line output path still yields an absolute source reference.
Phase 0 later copies the receipt-bound weights and rewrites that one reference
to the capsule-local sidecar; do not edit the captured MLIR by hand.
FP32 captures inventory frontend demand; they do not imply FP32 device support.
For an int8 capture, the reviewed rank-3/4 dequantization signature is only a
semantic screen. Mint the selected scalar package from
[its recipe](../target/scalar-host-recipe.yaml) before deriving; the old RVV
package requires a V extension absent from Rocket. The scalar package's ISA
fits the selected DTS, and one saved rank-4 capture passed Spike, but this does
not qualify Rocket RTL/FireSim execution or every declared host operation.
A deterministic native-host diagnostic can
be generated for a saved capture with
`python -m merlin_experiments.model_qualification --bundle "$CAPTURE_ROOT/residual_cnn" --out "$QUALIFICATION_ROOT/residual_cnn" --native-host-only --atol 0 --rtol 0`.
The generated `qualification.json` binds the input bytes, compiler route and
numerical result. It proves that saved whole program on the CPU, not independent
per-operation support, Rocket execution, or Gemmini execution; Phase 0 keeps the
Rocket-host execution obligation unresolved until those separate obligations are met.
The target-independent dequantization probes isolate the PyTorch operation at
[rank 3](../../workloads/quant_boundary/loader.py) and
[rank 4](../../workloads/quant_boundary/loader_rank4.py), with separate scales and
single-result ABIs. Each saved capture can be qualified through the same
native-host command. Exact agreement on both finite probes is review evidence
for the typed host signatures, not a substitute for sealed-source checks,
Rocket execution, or all scales and values.
If the selected Model2MLIR build lacks the same-conversion bundle/receipt APIs,
`--materialize-bundle` fails before capture. For raw frontend inventory only,
replace it with `--diagnostic-model-copy` and use a fresh output directory.
This copies the exact converted MLIR to `model.mlir` and hashes the observed
files in `diagnostic-capture.json`, but emits no `capture_receipt.json`, runtime
input/golden ABI, or Phase 0 admission. It cannot substitute for a verified
materialized derivation input.
The diagnostic record lists missing same-conversion, frontend-trace and static
integerization APIs separately. The frozen M2M selection records those API
checks too; an API reported as available only permits a sealed preflight, not
source-closure verification or corpus admission. In particular, the older
`write_bundle` that calls `m2m.convert` again cannot establish that its runtime
bundle has the argument map of the inspected conversion.

TinyLlama, SmolVLA and ResNet50 remain held-out validation workloads. Their
captures and layer frequencies do not select or tune the derivation corpus.

## 4. Derive requirements and candidate capsules

With installed `merlin-experiments`, explicit captures and fresh facts:

Select the *same* reviewed execution contract and original RTL facts for
derivation and the subsequent Phase 0 run. Set `MERLIN_TARGET_CONTRACT` and
`MERLIN_RTL_FACTS` to those exact inputs. A contract declaring
`runner.backend: chipyard_rocc`, logical harness ABI v2, operand roles and pinned
toolchain/engines uses Merlin's shared neutral tooling without executable
support. The checked-in prototype contract is reference metadata and does not
already provide this deployment selection. See the
[AWS handoff](../../../docs/guides/aws_gsim.md).
The generated requirement binds raw facts, the contract, effective readout
facets and selected tooling source bytes. Changing those inputs requires fresh
derivation rather than a resumed corpus run. The selected Gemmini readout supports
`acc_scale` but not the distinct integer-shift `requant` epilogue, so the latter
must remain an explicit rejected/host obligation rather than a fabricated
accelerator capability.

The recipe declares stable performance oracle names (`spike` for L2 and
`verilator` for L3). Capsule derivation does not probe which simulators happen
to be installed; execution still verifies the selected engine and hardware
revision. Before freezing a run that will execute those members, select
`MERLIN_EXT_CHIPYARD` for the Chipyard tree containing the concrete Gemmini L3
simulator. For separate capture/derivation, set `MERLIN_M2M_DIR` to the selected
Model2MLIR source tree and `MERLIN_M2M_PYTHON` to its pinned interpreter.
The frozen Phase 0 runner does not inherit those ambient variables. For live
PyTorch-sourced capsules, explicitly select the source tree and interpreter
with `--phase0-m2m-root` and `--phase0-m2m-python` on the run command. That
selection copies and checks the chosen source package and workload loaders,
and checks the host interpreter tree on launch and resume. It remains a
diagnostic managed-host execution, not a sandboxed capture-source attestation
or verified Phase 0 admission.
The L3 performance members cannot execute if the
simulator cannot be resolved; a missing exporter also leaves source capsules unwritten.
Static-int8 source capsules require `m2m/capture/pt2e_integerize.py` in the
selected tree. Captures that may fold Conv+BatchNorm also require the
`m2m.capture.trace` PT2E fold-provenance API
(`pt2e_conv_bn_fold_candidates`, `attach_pt2e_conv_bn_folds`). The generated
micro-model can omit that API only after its complete original and PT2E-input
ATen graphs prove that no BatchNorm call exists; an unknown or opaque graph
still fails closed. Check the selected modules and interpreter together before
freezing; a different checkout with the same project name is not interchangeable.
The selected independent integer reference must also report every integerized
contraction kind used by a generated source capsule, including `matmul` when
present. A portable TorchAO Q/DQ comparison is diagnostic and cannot replace
that exact reference; missing accounting leaves the capsule unbuilt.
Record these dependencies in the frozen run rather than relying on a later
resume to supply them.

```sh
merlin experiment corpus derive gemmini-functional \
  --application-capture "coverage_mlp=$CAPTURE_ROOT/coverage_mlp/model.mlir" \
  --application-capture "residual_cnn=$CAPTURE_ROOT/residual_cnn/model.mlir" \
  --application-capture "causal_decoder=$CAPTURE_ROOT/causal_decoder/model.mlir" \
  --application-capture "multimodal_policy=$CAPTURE_ROOT/multimodal_policy/model.mlir" \
  --rtl-facts "$RTL_ROOT/facts.json" --output "$DERIVATION_ROOT"
```

Inspect `requirements.yaml`, `application-demands.json`, `synthesis-plan.json`,
`synthesis.yaml` when a candidate plan is expressible, and `derivation.json`.
For an output-memory check, follow
`requirements.yaml` → `accumulator_output_boundary` → the generated
`SY_accumulator_output_boundary` entry in `synthesis.yaml` → its `capsule.yaml`,
`capsule.interface.mlir` and `golden.yaml` in the Phase 0 run. The bound comes
from the selected RTL facts' addressable accumulator, not a model shape: the
capsule writes one row through the first output tile beyond that capacity.
Coverage checks the materialized operand shapes, so a missing boundary remains
an uncovered obligation. A passing golden alone does not qualify the compiler;
the Phase 1 grade must also execute that capsule against the selected oracle.
The producer requires the complete declared roster and binds the selected SW spec,
recipe, workload policy and requirement bytes. A successful diagnostic derivation
is **not** a compiler certificate or reviewed corpus. Missing mappings remain
obligations; an unexpressible plan is retained as a blocked artifact.

For a concrete composition audit, inspect
`requirements.yaml` → `scope.typed_required_instances.instances`. Each record
names the originating capture, its SHA-256, exact MLIR operation IDs and the
typed SSA edges between regions. Compare those operations with the selected
software spec and each generated capsule's `capsule.interface.mlir` and
`capsule.yaml` software-screen decision.
Raw source adjacency is not device placement: a chain containing host-side
casts or maps cannot become a Phase 2 performance obligation merely because a
synthetic capsule has the same sequence of semantic families. Explicit SW
admission, a matching implementation and measured execution are separate gates.
The derived `scope.performance` section records eligible, SW-refused and
unresolved exact chains separately. Phase 1 keeps `scope.required` as the raw
source-demand census; Phase 2 selects only `scope.performance.required`.
An unresolved chain blocks that performance claim instead of being counted as
covered by a merely similar synthetic program.
The current PN writer uses a synthetic scalar map, so even matching operation
names and dtypes are not enough to establish source-body equivalence; a future
source-bound emitter must supply that correspondence before PN is eligible.

### Realize the selected precision, then derive again

The first FP32 pass inventories source demand. Inspect the generated
`evidence/software/quantization-recipes.json` and select its matching format entry.
Resolve that entry's relative `path` against the evidence directory and set
`GENERATED_RECIPE` to the resulting JSON file. The content-addressed recipe is
generated from the selected spec and hardware; do not author a replacement.

Capture each iteration workload into a new scoped directory:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype int8 --recipe "$GENERATED_RECIPE" \
  --seed 0 --materialize-bundle \
  --out "$SCOPED_CAPTURE_ROOT/coverage_mlp"
```

Repeat for the other three loaders. The recipe scopes eligible contractions;
normalization, embeddings and unsupported operations are not blanket-quantized.
Inspect each `meta.json` for `quantization_stats`, calibration count/source,
`recipe_agreement` and `integerization_receipt`. The generated recipe selects
the SW spec's `integer_reference` engine: `golden_agreement` must exactly match
the byte-bound independent integer output in `integer-reference.json`.
`portable_agreement` against TorchAO's Q/DQ graph is diagnostic, and
`recipe_agreement` measures error against the original FP32 model. These are
separate observations. A single synthetic calibration example is a smoke
input, not workload-accuracy validation or proof of accelerator execution.

Derive a fresh corpus plan from these exact realized bundles:

```sh
merlin experiment corpus derive gemmini-functional \
  --application-capture "coverage_mlp=$SCOPED_CAPTURE_ROOT/coverage_mlp/model.mlir" \
  --application-capture "residual_cnn=$SCOPED_CAPTURE_ROOT/residual_cnn/model.mlir" \
  --application-capture "causal_decoder=$SCOPED_CAPTURE_ROOT/causal_decoder/model.mlir" \
  --application-capture "multimodal_policy=$SCOPED_CAPTURE_ROOT/multimodal_policy/model.mlir" \
  --rtl-facts "$RTL_ROOT/facts.json" --output "$REALIZED_DERIVATION_ROOT"
```

Preserve the bootstrap plan and FP32 bundles. For the next step, select
`REALIZED_DERIVATION_ROOT`, not the initial FP32 derivation. New recipe, framework,
source or calibration bytes require newly captured bundles and a fresh plan.
Derivation checks each quantized capture's recorded recipe digest against the
recipes derived from the *currently selected* provider and SW spec; captures
from an older provider cannot be reused just because their MLIR parses.

`evidence/evidence-manifest.json` links to the exact input bytes and consumer views:

- `hardware/circt/facts.json`: byte-identical extraction.
- `hardware/effective-views/`: actual profile, readout, quantization and execution inputs.
- `software/source-snapshots/`: selected RTL/source/parser inputs with hashes.
- `software/frontend/` and `software/framework/`: saved lineage and operator catalogs.
- `coverage/operation-accounting.json`: per-model and combined accelerator/host/unresolved splits, signatures and precisions.
- `software/quantization-contract.json`: format alternatives, eligibility and unknowns.
- `software/quantization-recipes.json`: generated prospective recipes and their exact byte identities.
- `coverage/README.md`: automatically rendered navigation of those same reports.

## 5. Generate a fresh corpus and keep cohorts distinct

Select the realized requirement/profile together for inspect, preflight and run:
the selected materialized bundles already bind their Model2MLIR capture outputs.
The frozen runner does not recapture from an ambient sibling checkout or
inherit a live capture interpreter. Source capsules require the explicit
diagnostic runtime selection below; if it or a required API is missing they
remain omissions, not verified Phase 0 coverage.

```sh
merlin experiment run gemmini-functional --phase 0 \
  --phase0-rtl-facts "$RTL_ROOT/facts.json" \
  --phase0-conformance-spec "$REALIZED_DERIVATION_ROOT/requirements.yaml" \
  --phase0-synth-profile "$REALIZED_DERIVATION_ROOT/synthesis.yaml" \
  --phase0-m2m-root "$MODEL2MLIR_ROOT" \
  --phase0-m2m-python "$CAPTURE_PYTHON" \
  --phase0-evidence-mode diagnostic --run-dir "$RUN_ROOT"
```

Inspect `<run>/phase0/coverage/generation.json` for written, omitted and failed
members, then `capsules/MANIFEST.yaml` for distinct functional, performance and
diagnostic selections. A candidate list is not proof that its writers/oracles can
execute every member. Provision the selected independent numerical model and
OOT toolchain before execution; missing mandatory private evidence blocks admission.

The separate `phase1-capsule-coverage.json` and `phase2-capsule-coverage.json`
reports inventory only each selected cohort's exact bytes. Interface-command
observations and source-model MLIR witnesses remain distinct; a performance
cohort cannot borrow functional source coverage or claim whole-model validation.
For Phase 2, inspect `form_perf_coverage.missing` and
`form_perf_coverage.source_windows_without_form`. The latter catches an
independent iteration convolution whose integerized MLIR appears as matmul but
whose source padding/window and movement regime have no performance capsule.
The form sweep reuses each exact derived functional window member as a bounded
paired candidate/vendor performance member; inspect `form_perf_coverage.source_windows`
to see the emitted match. This covers the window mechanism, not a model-scale
timing claim or a fused epilogue that the member does not declare.
Either gap makes the Phase 2 cohort incomplete. A covered form still has an
unmeasured candidate/vendor ratio until the selected timing engine executes it;
predicted array issue cycles do not price host im2col, epilogues or DMA stalls.
Open `capsules/model/SY_micro_model/capsule.pytorch.py` to inspect the derived
A→H→A layer order. Its header lists accelerator capabilities that the emitted
standalone statements cannot exercise; for Gemmini, a float GELU, transpose or
reduction must not be read as proof of an int8 fused epilogue, transfer or pool.
`frontend-source.mlir`, `frontend-trace.json`, `capsule.interface.mlir`, and
`capsule.weights.safetensors` beside it show the captured source, trace, selected
interface and separate weights. The trace and host golden are capture evidence,
not a receipt that the target compiler executed the mixed-lane model.

### Review and release

A successful controller receipt proves that the selected generator finished; it
does not establish capsule coverage or target execution. Check
`coverage/generation.json` for zero writer failures **and** zero omissions, then
inspect `phase1-capsule-coverage.json` and `phase2-capsule-coverage.json`
separately. The derived micro-model composition test reads the run's frozen
iteration captures, not an ambient `out/artifacts/recaptures` directory.
The memory-mapping axis reads the exact CIRCT facts bytes recorded in
`capsules/_phase0/coverage-inputs.json`; a missing or changed facts snapshot
leaves that axis unmeasured. This is pre-compiler corpus coverage, not evidence
that a compiled program used the on-chip store correctly.
Host-only and host-lane coverage likewise classify each capsule against the
selected capability contract, without consulting an ambient provider. Mixed
host/device composition stays unmeasured until an emitted boundary is bound to
that same selected source; a declared legal boundary is not an execution proof.

The release records two separate verdicts: `phase0_readiness` for the deterministic
corpus handoff, and `whole_workload_phase1` for compiler qualification. Phase 0
readiness requires reviewed software and host semantics, verified capture-source
closure, complete graph/source correspondence, witnessed functional capsules,
selected hardware evidence, and an operator-owned hidden cohort. Artifact-backed
support lowering, emitted host/device routes and transfers, numerical target
checks, and executed composition remain explicit Phase 1 obligations; they are
not silently counted as passed at release time. None of these claims follows
from a diagnostic run or a changed status field. Review the generated model inventory and `grading.resource_bound` for
the *selected* cohort; if policy changes, freeze a new run rather than editing
an old receipt. With a separately selected private hidden category, prepare a
candidate release using:

```sh
merlin experiment corpus prepare "$NEW_RUN_ROOT" --generated-only \
  --private-baseline "$PRIVATE_HIDDEN_ROOT" --output "$NEW_RELEASE_ROOT"
```

Preparation reports unmatched public models and a missing or withheld hidden
cohort; it does not choose exclusions for the operator. Acknowledging a seal
does not repair incomplete coverage or missing execution receipts.

Review coverage, placement and independent numerical checks before preparing
[the reviewed Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff).
Changing a status field cannot qualify old artifacts. New inputs require newly
frozen runs; preserve old outputs unchanged. See [the artifact map](../artifacts/README.md)
and [whole-model walkthrough](../whole-model/README.md) for member MLIR, external
tensors and intermediate lowering snapshots.

### Check a generated kernel on the oracle ladder

With `merlin-experiments` and its AET dependency installed, select the same
reviewed neutral contract and exact facts used by derivation. The
compiler interpreter and LLVM tools are independent of the PyTorch capture
interpreter. An installed Merlin checkout does not imply they are installed or
selected; check `python -c 'import aet'` and resolve these paths before grading.

```sh
MERLIN_TARGET_PATH= \
MERLIN_TARGET_CONTRACT="${PREPARED_CONTRACT:?reviewed neutral execution contract required}" \
MERLIN_RTL_FACTS="$RTL_ROOT/facts.json" \
MERLIN_EXT_CHIPYARD="$SIMULATOR_CHIPYARD_ROOT" \
MERLIN_COMPILER_PYTHON="$COMPILER_PYTHON" \
MERLIN_CLANG="$LLVM_BIN/clang-23" \
MERLIN_MLIR_TRANSLATE="$LLVM_BIN/mlir-translate" \
MERLIN_OBJDUMP="$LLVM_BIN/llvm-objdump" \
python -m merlin.targetgen.capsule_runner \
  --package "$COMPILER_PACKAGE" \
  --capsule "$RUN_ROOT/phase0/capsules/layers/SY_int_mm_m8_k32_n32_1a2863611f" \
  --runs-root "$CAPSULE_RUN_ROOT" --target gemmini --timeout 600
```

The member name above is an example from one realized capture, not a stable
authored input; choose a member present in your run. Its `capsule_result.json`
must show measured passes for every mandatory tier. L2 is Spike's functional
model, while L3 is elaborated RTL; L0/L1 or a generated ELF alone are not an RTL
verdict. Record the selected simulator/build provenance separately: a passing
kernel result with `UNKNOWN` hardware pins is diagnostic, not a pinned release
claim. `testbench_timeout` at L3 means that tier is **unmeasured**, even when
the same capsule passed L0–L2 with zero numerical mismatches. Use a measured
small source-derived capsule to exercise RTL, and keep large model-derived
shapes as separate functional-model checks when their RTL cost exceeds the
budget; neither result substitutes for the other's coverage obligation. This
check does not qualify a complete model or the Phase 0 corpus.
