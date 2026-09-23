# Saturn OPU RVV: workflow map

The single [definition](experiment.yaml) is registered as `saturn-opu-rvv-functional` in the
[catalog](../../experiments/catalog.yaml). Its declared runtime target is
`saturn_opu_mxv256d128_rvv`; do not replace it with a provider name or another OPU configuration.

- [Phase 0](phase0/README.md): derive the declared corpus, then prepare, inspect and review its release.
- [Phase 1](phase1/README.md): explicitly select that reviewed release for functional compiler authoring.
- [Phase 2 templates](../../experiments/README.md#definitions-and-execution): supply exact frozen compiler/campaign evidence and deployment inputs; no ready-to-run target-specific campaign is supplied.
- [Whole-model inspection](../../docs/guides/model_lowering.md): inspect captured IR without claiming accelerator correctness. [Accelerator deployment](../../docs/guides/whole_model_on_accelerator.md) has separate qualification requirements.

`merlin experiment inspect saturn-opu-rvv-functional --phase 1` inspects configuration without
starting an experiment. `merlin experiment runs --target saturn_opu_mxv256d128_rvv` discovers
retained orchestration runs; it neither resumes them nor certifies their outputs.

The [descriptor](target/descriptor.yaml) retains external resource dependencies.
OPU-specific resource migration and standalone deployment remain incomplete.
Keep generated capsules, compilers, measurements and receipts beneath the configured
output root. Never copy private holdouts or credentials into examples or rewrite
historical evidence.
