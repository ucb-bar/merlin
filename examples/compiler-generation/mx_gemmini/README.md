# MX Gemmini: workflow map

Start with [experiment.yaml](experiment.yaml), registered as `mx-gemmini-functional` in the
[central catalog](../../../experiments/catalog.yaml). Its descriptor declares runtime
target `mx_gemmini`; provider identity and hardware configuration are separate inputs.

- [Target descriptor](target/descriptor.yaml): target policy and declared external resources.
- [test generation guide](test-generation/README.md) and [public recipe](test-generation/recipe.yaml): derive tests, then prepare and review the resulting corpus.
- [backend generation guide](backend-generation/README.md): supplied compiler-authoring inputs and functional experiment requirements.
- [backend optimization handoff](backend-generation/optimization/README.md): select frozen compiler evidence and the existing optimization templates.
- [Whole-model entrypoints](whole-model/README.md): capture/lowering inspection and separate deployment prerequisites.

The descriptor selects the MX configuration and retained external resources; the public ISA/MMIO inputs are described in the backend generation guide.

Inspect the definition and discover retained runs without launching an experiment:

```sh
merlin experiment inspect mx-gemmini-functional --workflow backend-generation
merlin experiment runs --target mx_gemmini
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
