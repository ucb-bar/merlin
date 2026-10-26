# EL4 shared inputs and numerical release

For every new release apply [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md)
before the historical recipe descriptions below. Freeze grouped-16 weight and
per-tensor activation scales and report the new model-quality gates separately
from compiler/reference agreement. Historical diagnostic probes are not an
admitted new release.

This guide uses the workload contract at
[`0cc289de52a10bed256f32c90cfbb54a21b3c81f`](https://github.com/ucb-bar/merlin/tree/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol).
The current scope is ResNet50, TinyLlama **32+32**, and SmolVLA's complete
12-call chain. It is a repo/reference audit, not an inventory of one machine.
The protocol and recipes settle the model choices; the remaining release work
is to make their concrete prepared bytes and execution artifacts shareable.
This is the EL4 tip fetched on 2026-10-23; the selected protocol, base recipes and
stored input definitions are unchanged from the earlier `db2f524` reference.
New prospective speed recipes are explicitly unselected/unmeasured. They do not
silently replace the three base recipes or their numerical identity.

## Inputs already committed, per workload

| Workload | What every member can reference in Git | Preparation still needed |
| --- | --- | --- |
| ResNet50 | Eight visible 224×224 RGB uint8 crops plus metadata, including selected `v00`, Imagenette member/label/source hash. | Convert the stored crop to FP32 with `(pixels/float32(255) - mean) / std`, transpose to NCHW and serialize deterministically. The audit reproduced all eight pinned derived hashes. No new evaluation-image selection is needed. |
| TinyLlama | Eight visible metadata records containing 160 tokenizer IDs each; selected `v00` is WikiText2 test article 0. | Serialize the stored IDs to the pinned input NPZ. The audit reproduced all eight derived hashes. Keep all 160 stored IDs; the selected chain consumes only the first 64 as 32 prefill + 32 fixed continuation tokens. No sampling or new prompt is needed. |
| SmolVLA | Eight raw top-camera frames, raw states and metadata containing task/episode/frame/noise seed. `v00` selects episode 0/frame 100/seed 1000. | Use the selected processor/tokenizer/statistics to produce image/masks/language/state/noise. Raw inputs and the expected final hash exist; the exact processed observation has not been established by this audit. |

Sources: [committed input directory](https://github.com/ucb-bar/merlin/tree/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/inputs),
[stored-form and derivation instructions](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/README.md#L91-L127),
[ResNet derivation code](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/model_check/input_worker.py#L34-L69),
[input hashes](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/MANIFEST.json).

The selected functional cohort is **one `v00` per model**, not all eight visible
cases and not the larger quality/calibration cohorts. Repeated benchmark passes
can reset and reuse that input. Calibration computes scales during preparation;
it is not part of each measured inference. The selected counts are ResNet 260
training images (26/class), TinyLlama 128 prefill and 1,024 decode samples, and
SmolVLA one sample per program. Historical input/calibration inventories do not
override these selected counts. Capture inputs also differ from external
evaluation inputs: for example SmolVLA capture uses frame 0/seed 0, while `v00`
uses frame 100/seed 1000. See the
[selected model contracts](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L77-L202).

| Selected prepared input | SHA256 |
| --- | --- |
| ResNet50 `visible/v00/inputs.npz` | `aef47b171c1132f1aaec18a957499f85bc5c80bfbfde197121c8255709311263` |
| TinyLlama `visible/v00/inputs.npz` | `065d978c1f302bc4c2ee8f74dc6823d72b81224baf30a51b877919cbfa450629` |
| SmolVLA `visible/v00/observation.npz` | `8c4c0dca184abf8da7af74c187531cecc172c3eb50f51aa7b6d6d0c61fef3daf` |

SmolVLA's pinned processing uses LeRobot 0.6, resize/pad to 512,
STATE/ACTION mean/std normalization and VISUAL identity. Its backbone revision
is `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; the policy revision is
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`. The input worker also needs the pinned
dataset statistics and tokenizer. The prior delivery's recovery attempt produced
`e4863c12cb7f9ae9bdf9f20ec48d6788083a81f29bb680738f420daed5405414`, which differs
from the selected hash and was **not admitted**. Its v2 `smolvla-input-recovery/`
record preserves the attempted environment and per-array hashes. That diagnostic
is not evidence that the repo lacks the raw input or that a new camera/task
selection is needed. Resolve the processed-byte mismatch before claiming the
historical input identity.

## What the recipes answer, and what must be frozen

Model parameters have two parts here: the architecture/checkpoint selection and
the prepared numerical bytes used by the quantized program. The first is pinned
in the protocol. The recipe specifies how to obtain the second, including INT8
ranges, grouping, observers, rounding, bias/epilogue arithmetic and attention
planes. Calibration supplies the actual scales; a recipe JSON alone does not
contain every quantized weight, scale or expected output.

| Workload | Numerical definition already selected | Payload still needed for a shared run |
| --- | --- | --- |
| ResNet50 | `IMAGENET1K_V2`, checkpoint state-dictionary hash `8bea63dc89b0700f2acbc5b170ed0b4637daed4ab748949eefe50ebe95149ac0`; grouped-16 INT8 recipe; native host math. The state hash is not the downloaded `.pth` container hash. | Complete prepared model/constants/scales and independent reference for all 1,000 logits on the selected image. |
| TinyLlama | Full 22-layer checkpoint, BF16 loaded FP32 before INT8 capture; static attention planes; SmoothQuant alpha 0.5, groups qkv/gate_up/o, 32 statistics segments, outlier split ratio 8 and alignment 16; `portable_exp_trig`. | Prepared prefill and decode programs, their scales/constants, and references for all 99 outputs. The delivery's public-loader calibration/SmoothQuant/outlier integration remains unfinished. |
| SmolVLA | Selected 16-layer VLM / 16-layer expert, pinned policy/backbone, static attention planes and `portable_neural_f32`; final action decode retains all 32 channels in normalized policy space. | Prepared prefix/flow/action-decode programs, scales/constants, exact processed observation and references for all 33 outputs, including caches/time/padding. |

Sources: [numerical authority and model settings](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml#L61-L202),
[recipe files](https://github.com/ucb-bar/merlin/tree/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/recipes),
[calibration application](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/targetgen/_recipe_quantizer.py#L1332-L1363).

A new repeatable release can be prepared from these sources without a previous
capture. Freeze it once and have all members consume the same package. If new
calibration changes scales, transformed widths or IR, record a new numerical
identity and its differences; do not rewrite the pinned historical selection or
reuse its qualification. Reproducing that old selection exactly requires its
frozen bytes or a verified identical rebuild. Full all-output references must
come from the selected prepared IR evaluated by Merlin `linalg_numpy` with the
selected host math; capture fake-quant output alone is not the sealed oracle.
The producer already exists:
[all-output reference workflow](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/phase1/launch/all-output-references.md)
and [producer](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/phase1/launch/produce_all_outputs_reference.py).

## Required shared release and process

1. **Publish one manifest and fetch location.** Record the immutable workload,
   source/compiler/runtime revisions, model checkpoint and environment identities,
   recipe/configuration hashes and every payload's checksum. Use relative paths
   or an explicit relocation map. A path on a previous owner's machine does not
   let the team reproduce the run. Keep heavyweight/protected data in shared
   artifact storage and a referenceable manifest in the repo.
2. **Bind exact preparation.** Include the selected prepared inputs,
   preprocessing/statistics/tokenizer pins, calibration selection, weights/scales,
   integerized IR and stage `.pt2` exports. Reuse the existing input/capture
   producers; keep preparation separate from the benchmark command.
3. **Publish the resolved numerical configuration and references.** Use ResNet
   `native`, TinyLlama `portable_exp_trig`, SmolVLA `portable_neural_f32`, and the
   existing tolerance/quality policy. Include capture-selection/attestation files,
   reference plans/catalogs, all referenced blobs and per-step NPZ arrays with
   `out0`, `out1`, etc. The total is 46 calls / 133 arrays. Names or recipe hashes
   alone do not replace these payloads.
4. **Build complete sessions.** Bind the selected inputs and references to stage
   packages/routes, final ELF and memory/instruction/placement receipts. Reuse
   this delivery's package-session bridge. Carry computed state in guest memory
   and reset between repetitions. Per-step Spike execution is not a timed
   complete FireSim chain. EL4 now has a native retained-object complete-chain
   builder with full-array or compact all-element validation. TVM needs an
   explicit adapter for its different package ABI, manifest and UART protocol;
   this remains an integration step, not missing workload policy.
5. **Bind, run and qualify FireSim.** Fill the manager contract and model assets
   in the existing selection, choose strict, lightweight or legacy direct
   transport, and use its prepare/run/collect path. Strict staging uploads to S3
   and emits an unsent SSM plan; lightweight uses the installed helper and actual
   boot-file evidence. The supplied strict publisher and lightweight mode require
   compact comparison. Full arrays use the shared offline checker; compact mode
   compares every element in the guest. Complete independent placement,
   instruction and hardware checks. Publish the complete result
   and repetition/timing policy. See the [main handoff](README.md#bind-the-actual-firesim-device-and-run-complete-chains)
   and [pinned execution workflow](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/README.md).

**Current position:** definitions, stored inputs and producer/checker code exist.
The shared numerical release still needs assembly/publication (steps 1–3).
Step 4 has native chain tooling and a separately tested small TVM bridge, but no
qualified selected full-model TVM package/transport integration. Step 5 has tested
offline tools, not a recorded three-model FireSim completion. The existing
[V26 status](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/phase2/el4_20261018/selected-phase1-v26.json)
records capsule evidence separately and full-model validation as not run.

## Preparation hazards verified in the audited revision

- `inputs-prepare` writes the public files and `MANIFEST.json` beside `--sources`.
  Work from a copy and compare against the immutable original before admission;
  do not regenerate over the pinned selection. [Documented write behavior](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/numerics/README.md#L113-L127).
- SmolVLA preparation takes its dataset root from `source.local_root` rather than
  the supplied `--data` path. Bind/fix that path explicitly. The generic setup
  helper installs LeRobot 0.5.1; it is not the selected 0.6 environment.
  [Input code](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/model_check/inputs.py#L357-L365),
  [setup helper](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/build_tools/scripts/setup_model2mlir.sh#L23-L40).
- The generic models configuration defaults to native host math; a plain default
  invocation does not select both portable policies from the protocol. Publish
  the resolved configuration rather than requiring private hand edits.
  [Model resolver](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/src/merlin/model_check/models.py#L35-L42).

## Audit validation

At fetched EL4 `0cc289de5`, the protocol checker passes with three models, 46 calls
and 133 arrays. Fresh derivation reproduces all eight ResNet plus eight TinyLlama
NPZ hashes, and all three source/calibration manifest bindings match. This uses
only the committed visible bytes; no download of the calibration dataset is
needed to recover those benchmark inputs. SmolVLA processed-input equality remains
unestablished. The small ResNet-recipe graph again passes TVM host comparison at
`atol=2e-6, rtol=1e-5`; this is a compiler integration control, not full ResNet.

The re-audit ran 283 EL4 input/numerics/native-chain/FireSim transport tests and
218 TVM backend/CLI tests successfully. The comparison group had 256 passes,
three failures and five skips. Two failures and three skips require pinned
AVX512 source-arithmetic environments unavailable on this AVX2 host. The other
failure is a SmolVLA loader import conflict: installed LeRobot 0.5.1 pulls diffusers
0.35.2, which references removed TorchAO 0.16.0 APIs and hits an undefined
`logger`. The two capture tests skipped without an explicit interpreter then
passed in a separate two-test run. The initial transport collection failed for
missing `jsonschema` in the Torch environment; its 283-test rerun used Merlin's
dependency-complete environment. These limitations are retained, not hidden by
disabling numerical guards. No cloud execution was performed.

Run with the appropriate per-model dependencies and `src` plus
`packages/merlin-experiments/src` on `PYTHONPATH`:

```bash
python examples/gemmini/protocol/check.py --export /absolute/run/contract.json
python -m pytest -q \
  merlin/tests/infra/test_model_check_inputs.py \
  merlin/tests/infra/test_gemmini_protocol_export.py \
  merlin/tests/infra/test_model_check_baseline_numerics.py \
  merlin/tests/infra/test_model_check_numerics_policy.py \
  merlin/tests/infra/test_chain*.py \
  packages/merlin-experiments/tests/test_post_freeze_firesim.py \
  packages/merlin-experiments/tests/test_post_freeze_elf_upload.py \
  packages/merlin-experiments/tests/test_manager_elf_provenance.py \
  packages/merlin-experiments/tests/test_chain_observation.py \
  packages/merlin-experiments/tests/test_post_freeze_runtime_driver.py
```

## Historical capture locations: optional exact-run recovery

The old source records use `P1=/work/merlin/tasks/phase1-harness-v2`. These are
recovery clues only, not the current team's required download mechanism:

| Recorded item | Path relative to `P1` |
| --- | --- |
| Selected historical numerics bundle | `private/gemmini/numerics/launch-v15` |
| TinyLlama capture | `private/gemmini/captures/functional-phase1-v1/tiny_llama/run/capture` |
| SmolVLA capture | `private/gemmini/captures/launch-v14-methodology/smolvla/run/capture` |
| Derived inputs | `private/gemmini/numerics/derived/<model>/visible/v00/` |
| Retained configuration/readbacks | `.tmp/handoff/functional-phase1-baseline-v1/` |

Sources: [launch selection](../../phase1/launch/el4-fire-aligned.yaml),
[regression selection](../../phase2/el4_20261018/model_regression.py),
[recorded delivery status](../../../../docs/guides/phase2_status_20261020.md).
For exact recovery, resolve `MANIFEST.json:selected_capture_roots` and every
referenced catalog/attestation/blob. The historical ResNet capture root and PREP
path were not resolved by those records. The v8 manifest hash
`6177d5626831cfc1a5a2335e41022436d94c2b1ec3f777b8a2c76cde34bc7a32` and V26 models-config
hash `cd028d6af0a6f5540164b3026c8a6dc2e7137e6b5a03d8783a38a46dc28fb536` belong to
different selections; neither identifies the v15 manifest. None of these location
questions blocks preparing and publishing a new, explicitly identified release.
