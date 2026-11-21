# Atlas: workflow map

Start with [experiment.yaml](experiment.yaml), registered as `atlas-functional` in the
[central catalog](../../../experiments/catalog.yaml). Its descriptor declares runtime
target `atlas`; provider identity and hardware configuration are separate inputs.

- [Target descriptor](target/descriptor.yaml): target policy and declared external resources.
- [Software spec](target/software-spec.yaml) and [hardware selection](target/hardware.yaml): separate authored semantics from selected structural evidence.
- [Hand-authored Atlas MLIR reference](https://github.com/ucb-bar/atlas-mlir/tree/handwritten-implementation): the OOT machine dialect, passes, and [numbered MLP/attention handoff examples](https://github.com/ucb-bar/atlas-mlir/tree/5485aa0aaeca222e1c158507460db06a621eeba4/examples/handoff) from Atlas MLIR through LLVM MLIR, LLVM IR, assembly, and ELF.
- [test generation guide](test-generation/README.md) and [public recipe](test-generation/recipe.yaml): derive tests, then prepare and review the resulting corpus.
- [backend generation guide](backend-generation/README.md): supplied compiler-authoring inputs and functional experiment requirements.
- [backend optimization handoff](backend-generation/optimization/README.md): select frozen compiler evidence and the existing optimization templates.
- [Whole-model entrypoints](whole-model/README.md): capture/lowering inspection and separate deployment prerequisites.
- [Artifact navigation](artifacts/README.md): raw CIRCT facts, actual consumer views, coverage and capsule lineage.

[Target setup](target/README.md) describes support selection (the vendored [`support/`](support/README.md)
provider is selected when `MERLIN_TARGET_PATH` is unset) and local tooling prerequisites.
The hand-authored reference is separate from Merlin's generated compiler and ACT
comparison. Its examples are fixed 32×32 diagnostic tiles; their ELF words ran on
the selected standalone AtlasCore through ModeLIR. They have not been run as
complete programs in `npu_model` or as full MLP/attention layers. See the
[dialect and pass reference](https://github.com/ucb-bar/atlas-mlir/blob/5485aa0aaeca222e1c158507460db06a621eeba4/docs/dialect-reference.md)
for what each operation and pass currently checks.
test generation is explicitly diagnostic until source consistency, software semantics and
coverage are qualified. Generated files remain outside the example.

Inspect the definition and discover retained runs without launching an experiment:

```sh
merlin experiment inspect atlas-functional --workflow backend-generation
merlin experiment runs --target atlas
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
