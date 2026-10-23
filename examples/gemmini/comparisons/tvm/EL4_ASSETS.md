# Locate the selected EL4 captures and references

**Historical reproduction only.** The current user request is a fresh preliminary
ResNet50 and TinyLlama 16+1 run using EL4's checked-in quantization recipes.
Regenerate scales and references from public checkpoints and calibration data;
the old bundle paths below are not blockers. See the README's current preparation
and implementation status. Retain these paths only for exact old-run reproduction.

The source branch `feat/el4-experiment` at
`0cc289de52a10bed256f32c90cfbb54a21b3c81f` contains the workload protocol,
quantization recipes, source-input files and manifests. It does not contain the
frozen full-model capture/reference bundle. The [target README](../../target/README.md)
places generated captures and goldens outside its folder;
[descriptor.yaml](../../target/descriptor.yaml) says the private specification
selects the exact captures and source identities.

The [shared protocol](../../protocol/experiment.yaml) requires the captured scale
bytes and integerized IR. Recreating a checkpoint with the existing TVM model
factories does not reproduce that selection: their quantization, source precision
and some output interfaces differ. A fresh calibration needs a separate profile;
it cannot silently replace the selected experiment.

## Recorded machine paths

The branch's [Phase 2 status](../../../../docs/guides/phase2_status_20261020.md)
records `P1=/work/merlin/tasks/phase1-harness-v2`. The following paths are
reconstructed from that root and the linked source selections. They are location
clues, not evidence that files are currently present or that a historical capture
is the current selection. No asset-download URL or machine hostname was found in
these records. `/work` is absent on the local handoff machine.

| Item | Recorded location relative to `P1` | Evidence / meaning |
| --- | --- | --- |
| Numerics bundle | `private/gemmini/numerics/launch-v15` | [`el4-fire-aligned.yaml`](../../phase1/launch/el4-fire-aligned.yaml), `NUMERICS_VERSION=15`, `NUMERICS_BUNDLE` |
| Bundle parent | `private/gemmini/numerics/launch-v13` | Same configuration's copy/reuse source |
| Approved lineage bundle | `private/gemmini/numerics/launch-v8` | Same configuration's approved numerical anchor; this is not a pin for v15 |
| TinyLlama capture | `private/gemmini/captures/functional-phase1-v1/tiny_llama/run/capture` | [`model_regression.py`](../../phase2/el4_20261018/model_regression.py), `stages/prefill` and `stages/decode` |
| SmolVLA capture | `private/gemmini/captures/launch-v14-methodology/smolvla/run/capture` | Same regression script, `stages/prefix_encode`, `stages/flow_denoise`, `stages/action_decode` |
| Prepared evaluation inputs | `private/gemmini/numerics/derived/<model>/visible/v00/` | Same script's `input_root`; model names are `resnet50`, `tiny_llama`, `smolvla` |
| Retained handoff/configuration | `.tmp/handoff/functional-phase1-baseline-v1/` | Same script's `HANDOFF`; includes `models.yaml`, per-stage `*-compile-v2/status.json`, and `<model>-full-readback-spike-v1/` |

The exact **ResNet50 capture directory is unresolved**. The selected bundle's
`MANIFEST.json:selected_capture_roots.resnet50` must identify it. The branch also
mentions `out/artifacts/recaptures/resnet50_sum_and_mean_20260917`, but that is an
older schedule-proxy capture, not proof of the selected 54-contraction recipe.

The launch's `--prep` argument selects another important directory. The old
configuration reads `${PREP}/private-full-models.yaml`; successor configurations
use `${PREP}/private-models.yaml` and `${PREP}/all-output-plan.json`. The precise
PREP value is not recorded in the inspected tracked selections. Do not equate it
with the retained handoff directory without checking the launch's resolved config.

Additional clues: the portable launch templates use `${TASK_ROOT}/inputs/numerics`;
the generic model-check fallback is `/work/merlin/private/gemmini/numerics/current`.
Neither overrides the selected bundle. The original SmolVLA dataset root in
[`sources.yaml`](../../numerics/sources.yaml) is
`/work/merlin/tasks/p0-gaps/private-inputs/fullmodel/inputs/aloha_sim_insertion_human`.

## Required files and what each supplies

1. **Selected bundle and configuration:** `MANIFEST.json`, `SHA256SUMS`,
   `configuration/models.yaml`, and the resolved launch configuration. Preserve
   `selected_capture_roots`, model recipes, host-math policies and their pins.
2. **Private model selection:** `private-full-models.yaml` or `private-models.yaml`,
   plus every referenced `capture_selection` and `capture_execution_attestation`
   file. These bind the exact capture roots, checkpoint/source identities and hashes.
3. **Complete captures:** each selected `run/capture` tree, including stage
   `model.mlir`, parameters/constants, captured scales, prepared quantized programs,
   source/quantization metadata and reference receipts. Preserve referenced files
   outside the tree too. Captures without their constant/scale payloads are unusable.
4. **Numerics catalogs and payloads:** `<model>/programs/<program>/catalog.json`,
   `ref/`, `dev/`, `census.json`, `<model>/blobs/<sha256>.npy`, and the selected
   `<model>/visible/v00/` reference/call files, including `int8.npz` and available
   FP32/native-baseline references. The sealed quantized oracle is the prepared
   IR evaluated by Merlin `linalg_numpy` with the selected host math.
5. **Exact model inputs:** ResNet/TinyLlama `inputs.npz`; SmolVLA `observation.npz`,
   from the derived directories above. Keep the 160-token TinyLlama file intact;
   the selected 32-prefill/32-decode chain consumes its first 64 tokens.
6. **All-output references:** `all-output-plan.json`, each referenced
   `reference.json` and `SHA256SUMS`, and every program/step NPZ with `out0`,
   `out1`, etc. These cover all 133 outputs over 46 calls, including KV caches,
   positions, recurrent state and padded action channels. Judged logits/actions
   alone are insufficient. See [all-output references](../../phase1/launch/all-output-references.md).

[`full_model_verification.py`](../../phase1/launch/full_model_verification.py)
cross-checks the private specification's `capture` against
`<selected_capture_roots[model]>/run/capture`; it also binds catalog stage paths
and `model.mlir` hashes to the execution attestation. Follow those references to
find the actual files; do not guess from a directory's version suffix. If files
move between machines, retain their original manifest bytes and a separate path
mapping rather than silently rewriting pinned historical records.

## Known hashes

The [input manifest](../../numerics/MANIFEST.json) and protocol agree on:

| Selected input | SHA256 |
| --- | --- |
| ResNet50 `visible/v00/inputs.npz` | `aef47b171c1132f1aaec18a957499f85bc5c80bfbfde197121c8255709311263` |
| TinyLlama `visible/v00/inputs.npz` | `065d978c1f302bc4c2ee8f74dc6823d72b81224baf30a51b877919cbfa450629` |
| SmolVLA `visible/v00/observation.npz` | `8c4c0dca184abf8da7af74c187531cecc172c3eb50f51aa7b6d6d0c61fef3daf` |

The approved **v8** bundle manifest hash is
`6177d5626831cfc1a5a2335e41022436d94c2b1ec3f777b8a2c76cde34bc7a32`.
The [V26 selection](../../phase2/el4_20261018/selected-phase1-v26.json) records
models-configuration hash
`cd028d6af0a6f5540164b3026c8a6dc2e7137e6b5a03d8783a38a46dc28fb536`.
These belong to different recorded selections; neither establishes the v15
manifest's hash. Resolve the actual selected lineage before admission.

## Search on the original machine

These commands only list/read files; they do not restart the stopped campaign:

```sh
P1=/work/merlin/tasks/phase1-harness-v2
rg --files --hidden --no-ignore "$P1/private" "$P1/.tmp/handoff" "$P1/out" "$P1/merlin/out" \
  -g MANIFEST.json -g private-full-models.yaml -g private-models.yaml \
  -g all-output-plan.json -g numerical-reuse-proof.json
python3 - "$P1/private/gemmini/numerics/launch-v15/MANIFEST.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
doc = json.loads(path.read_text())
print(json.dumps(doc.get("selected_capture_roots"), indent=2))
PY
```

Start with the manifest, resolved launch config, private model specification and
all-output plan; they identify the potentially large payload closure. Keep the
full captures, referenced symlink targets and numerical references in private
artifact storage outside Git. The remaining access requirement is the machine
holding these paths or an archive of those selected files and payloads.

## Available locally

The branch tracks ResNet crops, SmolVLA raw frames/state/task/seeds and TinyLlama
token IDs under [`numerics/inputs`](../../numerics/inputs). On 2026-10-22 the
ResNet and TinyLlama `visible/v00/inputs.npz` files were reconstructed from these
public bytes and matched the exact hashes above. Their producer and verification
record are retained in the v2 handoff artifact, `recovered-visible-inputs/`.
SmolVLA still needs its exact processed observation. Recovery using the pinned
public dataset statistics, model configuration and tokenizer, including
LeRobot 0.6.0/Transformers 5.4.0 and an AVX-512 emulation attempt, produced
`e4863c12cb7f9ae9bdf9f20ec48d6788083a81f29bb680738f420daed5405414`.
That differs from the selected hash and was **not admitted**. The v2 artifact's
`smolvla-input-recovery/` retains source identities, per-array hashes and failed
attempts for comparison with the original preprocessing environment. Emulation
is only a recovery diagnostic, not hardware or model qualification. Source inputs
do not supply captured quantization scales or the independent full-output oracle.
