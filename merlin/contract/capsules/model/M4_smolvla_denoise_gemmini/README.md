# M4_smolvla_denoise_gemmini

One flow-matching **denoise step of a vision-language-action policy** (smolVLA), int8, as a whole-model
capsule. `kind: model`, `label: public`, gated on `after_op_pass_fraction: 0.8`, `lanes.require:
[on_mesh, scalar_rvv_lane]`, `required_oracle_tiers: [L0, L1, L2, L3]`.

## Why it exists

`smolvla` is named in this target's own `workload_spec.applications` and was captured five times under
`out/artifacts/recaptures/smolvla_*`, and until this capsule nothing **graded** it. The corpus's other
whole-model capstones are a decoder LLM (`M0_small_llama_gemmini`), a vision+recurrent control net
(`M1_lstmnetvit_gemmini` / `M2_microvit_gemmini`) and, on the synth axis, a classifier
(`SY_model_resnet50`). A VLA policy is a different composition: a vision-language **prefix** built by
concatenating image, instruction and proprioceptive tokens, and a separate **action expert** that
cross-attends into that prefix to denoise an action chunk. Cross-attention from one sequence into
another is a region shape none of the other model capsules contain.

## What it is, and what it is not

It is **not** the 1.2 GB capture. The real denoise step is 1.2 GB of weights over 6103 linalg regions at
512x512, which no target here can schedule at the cycle-accurate tier. This capsule keeps the *shape*
and drops the *scale*: the same six inputs in the same roles and the same action-chunk width, at 566
regions / 174 KB of weights.

What the network contains is **derived from the real capture**, not chosen: the layer inventory
discharges the `prov.op` / `prov.family` census of
`out/artifacts/recaptures/smolvla_denoise_step_fp32_app/model.mlir`, family by family, with the
reasoning and the census table in `capsule.pytorch.py`'s module docstring. Extents are multiples of this
target's own tile edge (16), except the action width (32) and state width (32), which are the real
policy's interface to its robot and are carried over unchanged.

Interface, positionally identical to the real capture's `input_order`:

| arg | name | type | role |
|-----|------|------|------|
| I0 | `img` | f32 `[1, 3, 32, 32]` | observation frame |
| I1 | `img_mask` | i1 `[1]` | whether the frame is present |
| I2 | `lang_tokens` | i64 `[1, 11]` | the instruction |
| I3 | `lang_masks` | i1 `[1, 11]` | which instruction tokens are real |
| I4 | `state` | f32 `[1, 32]` | proprioceptive state |
| I5 | `noise` | f32 `[1, 16, 32]` | the noised action chunk |
| Y0 | — | f32 `[1, 16, 32]` | the denoised action chunk |

The flow-matching timestep is a graph constant, not an input — as in the real capture, whose exported
ABI also has six inputs and no timestep tensor.

## What it grades

**Compiler correctness**, not policy accuracy. The inputs are a seeded synthetic stream; the reference
is the host torch-eager output of the same loader over those exact inputs. A pass proves the compiled
program reproduces that reference. It says nothing about the policy's success rate on a robot, and
`input_provenance.accuracy_claim_supported` is `false` for that reason.

## Regenerating it

The capsule is **generated**, not typed — its entry lives in `profiles/gemmini.yaml`:

    PYTHONPATH=$SPECIR_ROOT .venv/bin/python \
        merlin/contract/capsules/generate_corpus.py --target gemmini

`capsule.pytorch.py` and this README are the hand-authored files (the generator never overwrites
either); `capsule.yaml`, `capsule.interface.mlir`, `capsule.weights.safetensors*`, `golden.yaml` and
`expected_instruction_coverage.yaml` are minted from the loader through model2MLIR. Requires
`MERLIN_M2M_DIR` / its venv (torch lives only there).

`golden.yaml`, `expected_instruction_coverage.yaml` and the `.safetensors` are **answer surfaces and are
never tracked** — see `merlin/contract/capsules/.gitignore`.

## Measured cost

| | this capsule | M2_microvit | M0_small_llama | M1_lstmnetvit | SY_model_resnet50 |
|---|---|---|---|---|---|
| linalg regions | 566 | — | — | — | — |
| interface MLIR | 248 KB | 123 KB | 200 KB | 422 KB | 659 KB |
| weights | 174 KB | 17 KB | 602 KB | 4.0 MB | 94 MB |
| written output elements | 512 | 16 | 2048 | 771 | 1000 |
| predicted L3 seconds | 171 | 4 | 762 | 266 | 352 |

Predictions from `merlin.targetgen.cert_cost.predict_seconds_from_output` (the measured
`0.20509 * out^1.0782` law); the gemmini history fit (`fit_for('gemmini')`, n=188, floor 141 s,
0.0082 s/element, r2 0.15) puts it at 146 s. Either way it is the second-cheapest model capsule in the
gemmini corpus and well inside the ~19,000-element / 300 s affordability budget.
