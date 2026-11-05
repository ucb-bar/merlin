# Atlas: workflow map

Start with [experiment.yaml](experiment.yaml), registered as `atlas-functional` in the
[central catalog](../../experiments/catalog.yaml). Its descriptor declares runtime
target `atlas`; provider identity and hardware configuration are separate inputs.

- [Target descriptor](target/descriptor.yaml): target policy and declared external resources.
- [Software spec](target/software-spec.yaml) and [hardware selection](target/hardware.yaml): separate authored semantics from selected structural evidence.
- [Hand-authored Atlas MLIR reference](https://github.com/ucb-bar/atlas-mlir/tree/handwritten-implementation): the OOT machine dialect, passes, and [numbered MLP/attention handoff examples](https://github.com/ucb-bar/atlas-mlir/tree/5485aa0aaeca222e1c158507460db06a621eeba4/examples/handoff) from Atlas MLIR through LLVM MLIR, LLVM IR, assembly, and ELF.
- [Phase 0 guide](phase0/README.md) and [public recipe](phase0/recipe.yaml): derive tests, then prepare and review the resulting corpus.
- [Phase 1 guide](phase1/README.md): supplied compiler-authoring inputs and functional experiment requirements.
- [Phase 2 handoff](phase2/README.md): select frozen compiler evidence and the existing optimization templates.
- [Whole-model entrypoints](whole-model/README.md): capture/lowering inspection and separate deployment prerequisites.
- Model bringup ([PR #55](https://github.com/ucb-bar/merlin/pull/55)): the TinyLlama and SmolVLA [captures and hand-written host graphs](comparisons/README.md), the [currently disabled atlas-kernels selection and materialization example](phase1/README.md#kernel-library), the [atlas-kernels](https://github.com/ucb-bar/atlas-kernels/tree/feat/kernel-library) library itself, and the [relocated runtime, builders, tests, and runbooks](model_bringup/README.md).
- [Artifact navigation](artifacts/README.md): raw CIRCT facts, actual consumer views, coverage and capsule lineage.

[Target setup](target/README.md) describes support selection (the vendored [`support/`](support/README.md)
provider is selected when `MERLIN_TARGET_PATH` is unset) and local tooling prerequisites.
The hand-authored reference is separate from Merlin's generated compiler and ACT
comparison. Its examples are fixed 32×32 diagnostic tiles; their ELF words ran on
the selected standalone AtlasCore through ModeLIR. They have not been run as
complete programs in `npu_model` or as full MLP/attention layers. See the
[dialect and pass reference](https://github.com/ucb-bar/atlas-mlir/blob/5485aa0aaeca222e1c158507460db06a621eeba4/docs/dialect-reference.md)
for what each operation and pass currently checks.
Phase 0 is explicitly diagnostic until source consistency, software semantics and
coverage are qualified. Generated files remain outside the example.

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

The [November 4 entrypoint check](artifacts/workflow-inspection-2026-11-04.md) records the selected descriptor, support identity, and actual local results at Merlin `0a68cf0a`. Phase 0 inspect and configuration preflight completed with diagnostic evidence and absent synthesis; the diagnostic run then refused missing fresh derivation inputs. Phase 1 inspect completed, but preflight reported 11 errors. Both shared Phase 2 templates inspected as `OPERATOR_TARGET` placeholders. No generated or frozen functional compiler was selected, and the descriptor's RVV host compiler commit remains `UNKNOWN`.

As in the [Radiance phase guides](../radiance/README.md), target inspection, diagnostic corpus generation, reviewed functional admission, and optimization are separate stages. Atlas uses Merlin's declared target and workflow entrypoints; a new reviewed Phase 0 release and qualified Phase 1/2 campaign are **PENDING**. The comparisons workflow remains separate model-specific bringup.

Phase 2 starts from explicit functional evidence using the shared
[measured-claims](../../experiments/definitions/measured-claims-template.yaml) or
[model-portfolio](../../experiments/definitions/model-portfolio-template.yaml) template.
See the [execution guide](../../experiments/README.md#definitions-and-execution);
these templates are not ready-to-run target-specific definitions.

This map does not establish a complete Phase 2 or whole-model workflow, native
toolchain availability, or simulator/hardware qualification. Provision and verify
the descriptor's external inputs separately.
