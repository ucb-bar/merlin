# Atlas: workflow map

Start with [experiment.yaml](experiment.yaml), registered as `atlas-functional` in the
[central catalog](../../experiments/catalog.yaml). Its descriptor declares runtime
target `atlas`; provider identity and hardware configuration are separate inputs.

- [Target descriptor](target/descriptor.yaml): target policy and declared external resources.
- [Phase 0 guide](phase0/README.md) and [public recipe](phase0/recipe.yaml): derive tests, then prepare and review the resulting corpus.
- [Phase 1 guide](phase1/README.md): supplied compiler-authoring inputs and functional experiment requirements.
- [Phase 2 handoff](phase2/README.md): select frozen compiler evidence and the existing optimization templates.
- [Whole-model entrypoints](whole-model/README.md): capture/lowering inspection and separate deployment prerequisites.

[Target setup](target/README.md) describes explicit OOT support and local tooling prerequisites.

Inspect the definition and discover retained runs without launching an experiment:

```sh
merlin experiment inspect atlas-functional --phase 1
merlin experiment runs --target atlas
```

Follow the [reviewed Phase 0 handoff](../../experiments/README.md#reviewed-phase-0-handoff)
before verified Phase 1 execution. New capsules, compiler payloads and receipts
belong under configured artifact/run roots. Private holdouts remain host-only
declared inputs, never public example content.
Historical receipts retain their original bytes.

Phase 2 starts from explicit functional evidence using the shared
[measured-claims](../../experiments/definitions/measured-claims-template.yaml) or
[model-portfolio](../../experiments/definitions/model-portfolio-template.yaml) template.
See the [execution guide](../../experiments/README.md#definitions-and-execution);
these templates are not ready-to-run target-specific definitions.

This map does not establish a complete Phase 2 or whole-model workflow, native
toolchain availability, or simulator/hardware qualification. Provision and verify
the descriptor's external inputs separately.
