# Radiance: workflow map

Start with [experiment.yaml](experiment.yaml), registered as `radiance-functional` in the
[central catalog](../../../experiments/catalog.yaml). Its descriptor declares runtime
target `radiance`; provider identity and hardware configuration are separate inputs.

- [Target descriptor](target/descriptor.yaml): target policy and declared external resources.
- [test generation guide](test-generation/README.md) and [public recipe](test-generation/recipe.yaml): derive tests, then prepare and review the resulting corpus.
- [backend generation guide](backend-generation/README.md): supplied compiler-authoring inputs and functional experiment requirements.
- [backend optimization handoff](backend-generation/optimization/README.md): select frozen compiler evidence and the existing optimization templates.
- [Whole-model entrypoints](whole-model/README.md): capture/lowering inspection and separate deployment prerequisites.

[Target setup](target/README.md) distinguishes the experiment identity `radiance` from its OOT support identity `muon`.

Inspect the definition and discover retained runs without launching an experiment:

```sh
merlin experiment inspect radiance-functional --workflow backend-generation
merlin experiment runs --target radiance
```

Follow the [reviewed test generation handoff](../../../docs/guides/10-compiler-generation.md)
before verified backend generation execution. New capsules, compiler payloads and receipts
belong under configured artifact/run roots. Private holdouts remain host-only
declared inputs, never public example content.
Historical receipts retain their original bytes.

Backend generation develops correctness and performance together through the
named workflow in [`experiment.yaml`](experiment.yaml). Supply its reviewed
corpus release, matching input bundle, campaign policy and timing inputs as
shown in the [experiment guide](../../../docs/guides/12-experiments.md).

This map does not establish a complete backend optimization or whole-model workflow, native
toolchain availability, or simulator/hardware qualification. Provision and verify
the descriptor's external inputs separately.
