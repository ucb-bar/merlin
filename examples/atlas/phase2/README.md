# Atlas: Phase 2 handoff

Start with the [functional workflow](../phase1/README.md). Retain the exact
frozen submission, functional run identity, descriptor and grading evidence;
an authoring exit code or a changed compiler with a freshly computed hash is
not that handoff.

The [catalog](../../../experiments/catalog.yaml) provides two shared templates:
[model portfolio](../../../experiments/definitions/model-portfolio-template.yaml)
for bounded compiler authoring and structural analysis, and
[measured claims](../../../experiments/definitions/measured-claims-template.yaml)
for its separately qualified measurement protocol. They are different evidence
schemas. Neither is a ready-to-run Atlas performance experiment.

Copy a template into an authored input directory, keep `kind: template` until
every declared input and budget is supplied, and set `target: atlas`.
Use the descriptor and compiler identity from the frozen functional run.
The measured protocol additionally requires its exact engine certificates,
performance inputs and provisioned managed execution resources; a target name
does not establish that this target supports that protocol.

Follow the [installed deployment and execution guide](../../../experiments/README.md#definitions-and-execution)
for required package/source roots, campaign inputs and supervisor selection.
After completing the definition, inspect and preflight its path with
`merlin experiment inspect /absolute/inputs/performance.yaml --phase 2` and
`merlin experiment preflight /absolute/inputs/performance.yaml --phase 2`.
Preflight is not hardware or simulator qualification.

Generated candidates, checkpoints and measurements belong beneath the configured
output root, not this directory. Preserve original compiler bytes and retain
certification/publication records separately. Resume only the recorded run with
its original inputs; changed sources or evidence require new qualification.

For model captures and intermediate IR, see [whole-model inspection](../whole-model/README.md).
No target-specific end-to-end optimization or accelerator performance result is
established by this guide.
