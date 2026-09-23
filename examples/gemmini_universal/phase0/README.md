# Phase 0: derive a Universal capsule corpus

The [Universal descriptor](../target/descriptor.yaml) records a separately
selected cohort and the readout-width reason it differs from Gemmini. The old
source pool was shared for historical comparison; that is not permission to use
Gemmini's phase-0 recipe or grading verdict as this device's own derivation.

There is currently no reviewed Universal-specific `recipe.yaml`, synthesis/SMT
profile set, or fresh sealed corpus release in this example. Therefore the
[definition](../experiment.yaml) declares no executable Phase 0. To add one,
author and review those public inputs here, declare them explicitly in the
definition, run `merlin experiment inspect ... --phase 0`, then execute into a
fresh run under the configured output root. Prepare, inspect and explicitly seal
the release before selecting it in a *new* Phase 1 definition. The
[review procedure](../../../experiments/README.md#reviewed-phase-0-handoff)
never edits an old corpus or automatically admits generated tests.
