# Saturn OPU: Phase 0 — hardware-guided test generation

Start with [the experiment definition](../experiment.yaml), catalog ID
`saturn-opu-functional`, runtime target `saturn_opu_mxv256d128`.

[`recipe.yaml`](recipe.yaml) is the authored public coverage recipe, not a generated
corpus or evidence of hardware correctness. The definition explicitly names its shared
`performance_template` and generated `synth_profile` / `smt_profile` inputs.
Private `hidden_profile` stays outside examples; never copy holdouts, goldens, weights,
or generated capsules here. Optional sidecars may be absent; frozen runs bind their
presence as well as their bytes.

The authored descriptor lives at
[`target/descriptor.yaml`](../target/descriptor.yaml); its `resources_root`
still selects retained native resources.
This is not yet a self-contained target package. Install Merlin plus
`merlin-experiments`, provision the selected OOT support and the descriptor's declared
inputs, and make any required capture/toolchain dependencies available before running.
An example recipe does not establish simulator, RTL, hardware, or model-download availability.

Inspect first; then use fresh paths under your configured output root. Replace
`/configured/out` below with that root:

```sh
merlin experiment inspect saturn-opu-functional --phase 0
merlin experiment preflight saturn-opu-functional --phase 0
merlin experiment run saturn-opu-functional --phase 0 --run-dir /configured/out/runs/saturn_opu_mxv256d128/phase0/example-1
merlin experiment corpus prepare /configured/out/runs/saturn_opu_mxv256d128/phase0/example-1 --output /configured/out/artifacts/protocols/saturn_opu-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/saturn_opu-review-1
```

Stop for human review of the prepared inputs and owner-only diagnostics. **Only after
review**, acknowledge the exact inspected digest:

```sh
merlin experiment corpus seal /configured/out/artifacts/protocols/saturn_opu-review-1 \
  --expected-digest DIGEST_FROM_INSPECT --reviewed-by OPERATOR --review-note REVIEW_SUMMARY
```

The seal records a review acknowledgement, not numerical or hardware certification.
It does not automatically approve or start Phase 1. Follow the
[shared reviewed handoff](../../../experiments/README.md#reviewed-phase-0-handoff)
to select the reviewed release explicitly; retain old run receipts unchanged.
