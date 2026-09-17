# SY_model_smolvla

The **whole smolVLA policy**, int8, as a synthesized roster capsule. `kind: model`, `label: public`,
gate `after_op_pass_fraction: 0.8`, `lanes.require: [on_mesh]`.

## Why it exists

`smolvla` is named in this target's `workload_spec.applications`, and the roster axis synthesizes one
whole-model capsule per `workload_spec.models` entry. Until now that list was `[resnet50, tiny_llama]`,
so the third named driver had no whole-model capsule and could not be a graded portfolio member or
carry a measured performance reference. Adding `smolvla` to the roster makes the synthesizer emit this
entry (`build_tools/scripts/synth_capsule_corpus.py --target gemmini --write`), and the corpus
materializer captures it through model2MLIR with `int8_dyn_act_int8_weight` — the target's own
arithmetic, not a float matmul over dequantized weights.

## What it is, and what it is NOT

- 20,745 provenance regions, 503 MB of int8 weights, 6 declared inputs, entry `@forward`, 816
  arguments.
- It is **not** the `smolvla_flow_denoise` program measured at 33,085,199,302 cycles on FireSim
  (queue job 610). That bundle is a single flow-matching denoise step with a 1,416-entry ABI; this
  capsule is the full policy with 816. They are different programs, so this capsule is deliberately
  **not** mapped to that reference in `merlin/contract/perf_reference_targets.yaml`. Map it only
  against a number measured from this capsule's own program.
- It is also distinct from `M4_smolvla_denoise_gemmini`, which is a reduced denoise slice.

## Tracked vs regenerable

`capsule.yaml`, `capsule.interface.mlir`, `capsule.pytorch.py` and this README are tracked.
`capsule.weights.safetensors`, `golden.yaml` and `expected_instruction_coverage.yaml` are graded
answers or bulk data and stay untracked by `.gitignore` — regenerate them with the corpus
materializer rather than committing them.
