# Atlas: deterministic Phase 0

Start with [experiment.yaml](../experiment.yaml), catalog ID `atlas-functional`.
Phase 0 has no agent: selected input bytes and pinned tools determine extraction,
requirements and candidate capsules. Unknown semantics remain explicit blockers.
Phase 1 develops the functional compiler; Phase 2 uses a separate performance cohort.

## 1. Select the inputs

| Input | Responsibility |
| --- | --- |
| [Software spec](../target/software-spec.yaml) | Operation signatures, numerical semantics, placement/transfers and quantization eligibility |
| [Hardware selection](../target/hardware.yaml) | Source-production requirements and direct RTL audit questions |
| [Capability contract](../target/contracts/target_contract.yaml) | Prototype ISA/runner intent that RTL facts cannot supply; selected and frozen by the experiment, not a support certificate |
| [Host capabilities](../target/host-capabilities.yaml) | Separately pinned host compiler and reviewed operation/precision support |
| [Recipe](recipe.yaml) | Derived-only policy, comparison tolerances and oracle tiers; no authored capsule list |
| [Descriptor](../target/descriptor.yaml) | Independent iteration roster, held-out validation roster and experiment resources |
| Shared performance template | Phase 2 objectives and families, not additional functional capability |

The selected matrix datapath uses E4M3 inputs and BF16 accumulation with
RTL-specific rounding, clamping and subnormal behavior. The frontend currently
reports FP8-to-FP32 projection as a blocker, not native accelerator support.

[regression-seeds.yaml](regression-seeds.yaml) preserves historical authored tests
for explicit compatibility studies. It is **not** a default derivation input.
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

For a machine-dialect generation campaign, audit the **selected decoder-mode
population** separately from the model-derived capsule population. Supply the
exact selected RTL pattern and decoder files, the corresponding model ISA file,
and a reviewed mode ledger from the explicitly selected OOT support package:

```sh
merlin-targetgen audit-isa \
  --patterns "$RTL_SOURCE/Instructions.scala" \
  --decoder "$RTL_SOURCE/IDecode.scala" \
  --model-isa "$MODEL_SOURCE/isa_definition.py" \
  --rtl-revision "$RTL_COMMIT" \
  --model-revision "$MODEL_COMMIT" --verify-revisions \
  --out "$DERIVATION_ROOT/isa-census.json"
merlin-targetgen audit-dialect-modes \
  --census "$DERIVATION_ROOT/isa-census.json" \
  --inventory "$MODE_LEDGER" \
  --require phase1-inputs \
  --out "$DERIVATION_ROOT/dialect-mode-scope.json"
```

The revision check compares both selected checkout HEADs and the three exact
file bodies with their Git commit objects. A commit string alone is insufficient.
These commands keep missing and changed modes in the denominator, including
scalar/control modes that do not appear in tensor workloads. They return a
nonzero status for source disagreements or incomplete mode scope. This is the
pre-authoring Phase 0 input check: it does not require a generated dialect,
software admission, or closed instruction qualification. Exclusions require a
reviewed `scope_exclusion` with a reason and evidence reference; the selected Atlas ledger currently declares
all 99 decoder modes required. The source
crosswalk and OOT ledger can establish exact mode accounting; their reported
test flags do not certify legality, arithmetic, timing, or integrated execution.
The handwritten ledger's current parameter-domain prose must be replaced by
reviewed finite machine-value domains with explicit units before this check
can pass. A range such as an even BF16 register pair must state its step;
offset domains must state their physical units, and each domain cites pinned
RTL source files. This remains an authored
hardware contract to qualify, not a bound inferred from a mode name.
If model and RTL encodings disagree, the mode ledger may include exact
`source_resolutions` entries with the census discrepancy `kind` and `item`,
`authority: selected_rtl`, `reviewed: true`, and an evidence reference. The
same mechanism covers an individual ledger-to-model conflict with kind
`mode_model_encoding_disagrees` and the decoder mode as `item`. The audit
rejects stale or invented resolutions. Such a declaration records the
source-selection decision; numerical and temporal qualification still needs
independent tests.
After Phase 1 authors a dialect, rerun `audit-dialect-modes` with
`--require complete-dialect --dialect-plan "$REVIEWED_DIALECT_PLAN"`.
The reviewed **typed** plan is required for that completion result. Its explicit
mode attributes must cover every required decoder variant with legal values;
the audit does not accept a matching operation name alone. Such a plan is not present in
this example yet. A handwritten dialect's inventory is useful as an
independent comparator and is not a substitute for Merlin-generated signatures
and verifiers. Keep the ledger and generated compiler outside this example
directory.

The current descriptor selects `AtlasRocketConfig`. If the campaign instead
targets `EE290SimConfig`, create a new selected source bundle and Phase 0 run
with that exact elaboration and simulator identity. Existing receipts must not
be relabeled as evidence for the new SoC configuration.
For that new campaign, use the generic
[`rtl.elaboration` receipt](../../../docs/guides/phase0_specification.md#bind-the-selected-configuration-to-its-elaborated-source)
to bind the exact `EE290SimConfig` invocation to fresh FIRRTL, then supply
`--elaboration-receipt` when producing `source-selection.json`. The selected
Chipyard Git link for `generators/atlas-npu` must be checked against the exact
Atlas revision; a matching configuration name in a command alone does not
establish the selected hardware source or execution behavior.
Then run `merlin-targetgen audit-dialect-source-scope` with that selection,
the ISA census regenerated from the pinned `generators/atlas-npu` checkout,
the reviewed OOT mode ledger, and `--expected-config EE290SimConfig`. The
ledger must name its selected target explicitly. The current handwritten
ledger does not yet carry that field, and the selected EE290 elaboration is
not available here, so this pre-authoring check cannot currently pass for
the intended configuration. Source-scope readiness would still leave numeric,
temporal, typed-dialect, emission, and execution work for Phase 1.

## 3. Capture the independent iteration workloads

Use [the four shared loaders](../../workloads/README.md): `coverage_mlp`,
`residual_cnn`, `causal_decoder` and `multimodal_policy`. The worker seeds Python,
NumPy and Torch before loading the model and requires deterministic algorithms.
Use the same explicitly selected model2MLIR interpreter/build for all four.

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
FP32 captures inventory frontend demand; they do not imply FP32 device support.

TinyLlama, SmolVLA and ResNet50 remain held-out validation workloads. Their
captures and layer frequencies do not select or tune the derivation corpus.

## 4. Derive requirements and candidate capsules

With installed `merlin-experiments`, explicit captures and fresh facts:

```sh
merlin experiment corpus derive atlas-functional \
  --application-capture "coverage_mlp=$CAPTURE_ROOT/coverage_mlp/model.mlir" \
  --application-capture "residual_cnn=$CAPTURE_ROOT/residual_cnn/model.mlir" \
  --application-capture "causal_decoder=$CAPTURE_ROOT/causal_decoder/model.mlir" \
  --application-capture "multimodal_policy=$CAPTURE_ROOT/multimodal_policy/model.mlir" \
  --rtl-facts "$RTL_ROOT/facts.json" --output "$DERIVATION_ROOT"
```

Inspect `requirements.yaml`, `application-demands.json`, `synthesis-plan.json`,
`synthesis.yaml` when a candidate plan is expressible, and `derivation.json`.
The producer requires the complete declared roster and binds the selected SW spec,
recipe, workload policy and requirement bytes. A successful diagnostic derivation
is **not** a compiler certificate or reviewed corpus. Missing mappings remain
obligations; an unexpressible plan is retained as a blocked artifact.

Inspect `evidence/software/quantization-recipes.json` alongside the quantization
contract. The current FP8 format has unresolved scale encoding and block size,
so no automatic realization recipe is emitted. Keep the FP32 source captures
for demand accounting and retain the unresolved format as an obligation; do not
substitute generic FP8 quantization or call the captured graph accelerator-ready.
When a format becomes expressible, capture its scoped recipe into new bundles
and derive again from those exact bytes before generating a realized corpus.

`evidence/evidence-manifest.json` links to the exact input bytes and consumer views:

- `hardware/circt/facts.json`: byte-identical extraction.
- `hardware/effective-views/`: actual profile, readout, quantization and execution inputs.
- `software/source-snapshots/`: selected RTL/source/parser inputs with hashes.
- `software/frontend/` and `software/framework/`: saved lineage and operator catalogs.
- `coverage/operation-accounting.json`: per-model and combined accelerator/host/unresolved splits, signatures and precisions.
- `software/quantization-contract.json`: format alternatives, eligibility and unknowns.
- `software/quantization-recipes.json`: actual emitted recipes, or explicit reasons realization is unavailable.
- `coverage/README.md`: automatically rendered navigation of those same reports.

## 5. Generate a fresh corpus and keep cohorts distinct

Select the newly derived requirement/profile together for inspect, preflight and run:
Pin the same OOT support and independent SpecIR oracle selected for derivation.
The NPU model selection is required for an established ISA taxonomy; leaving it
unset produces a diagnostic unknown and stops capsule materialization. Atlas is a
self-hosted ISA target: do not add a command-ISA `corpus_issue_order` to bypass a
missing model environment. Select and pin the model's Python environment, then
rerun from a new artifact root; the failed frozen run remains diagnostic evidence.
The frozen runner does not inherit an ambient Model2MLIR interpreter.
PyTorch-sourced capsules requiring on-demand capture are reported as omissions
until their tool/runtime has an explicit frozen selection; the four
already-materialized iteration captures remain exact Phase 0 inputs.

```sh
MERLIN_TARGET_PATH="$ATLAS_SUPPORT_ROOT" SPECIR_ROOT="$SPECIR_ROOT" \
  MERLIN_EXT_NPU_MODEL="$NPU_MODEL_ROOT" \
  merlin experiment run atlas-functional --phase 0 \
  --phase0-rtl-facts "$RTL_ROOT/facts.json" \
  --phase0-conformance-spec "$DERIVATION_ROOT/requirements.yaml" \
  --phase0-synth-profile "$DERIVATION_ROOT/synthesis.yaml" \
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

Review coverage, placement and independent numerical checks before preparing
[the reviewed Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff).
For a new Atlas release, use that handoff's `--generated-only` mode; the
descriptor's retained BF16 corpus is historical input, not proof that this
FP8 selection can execute those members. Supply any required hidden cohort as
a separate private baseline. Do not seal a run with omitted source capsules or
missing L2/L3 oracles.
Changing a status field cannot qualify old artifacts. New inputs require newly
frozen runs; preserve old outputs unchanged. See [the artifact map](../artifacts/README.md)
and [whole-model walkthrough](../whole-model/README.md) for member MLIR, external
tensors and intermediate lowering snapshots.
