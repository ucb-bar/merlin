# Phase 1: functional compiler

The [catalog definition](../experiment.yaml) uses the installed RTLchecks
treatment with an explicit bundle identity, manifest and timing path. These
retained inputs must be regenerated and reviewed for the selected corpus before
verified execution. See [execution prerequisites](../../../experiments/README.md#definitions-and-execution)
for missing artifacts, tool provisioning and historical-resume limitations.

[The target descriptor](../target/descriptor.yaml) selects the public inputs in
`contracts/hwbringup_atlas_v0/`:

- Architecture overview and ISA green card.
- ISA definition requiring the external `npu_model` dependency.
- Curated RTL source evidence, with a note identifying the full external tree.
- Worked assembly examples.
- Preflight assembly fixtures and their target-owned assembler/runner adapter.

These are supplied experiment inputs, not newly generated compilers or certification
results. Their bytes are preserved from the former experiment directory. The curated
RTL subset is not a replacement for the complete external hardware repository.
Merlin's shared preflight protocol consumes the declared adapter; it contains no Atlas
instruction encoding. Path validation does not imply that a hardware probe passed.

Start from [the experiment definition](../experiment.yaml) and
[local tooling setup](../target/README.md). The full-mode task prompt is generated
by the shared renderer. Prepare and review a fresh release through the
[Phase 0 handoff](../../../experiments/README.md#reviewed-phase-0-handoff) before
verified execution. Generate new bundles for these paths; historical bundles and
receipts retain their original bytes. Other harness resources still use the
descriptor's retained `resources_root`.
