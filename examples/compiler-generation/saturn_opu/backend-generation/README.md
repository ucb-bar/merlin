# Saturn OPU: functional compiler prerequisites

The [definition](../experiment.yaml) selects **EL4 (RTL-informed Merlin)**,
whose retained bundle is `merlin_assisted_rtlchecks_public_v0`. That public bundle is not
supplied here: prepare and review it for this target. Do not substitute a
hardware-bringup bundle to satisfy a missing file.

Follow [test generation review](../test-generation/README.md), then the
[explicit reviewed handoff](../../../../experiments/README.md#reviewed-phase-0-handoff).
In a copied definition, select the release descriptor, corpus seal and matching
bundle manifest together. Supply a genuine existing oracle-timing record and
provision the descriptor's target support, tools and sandbox requirements.
The retained example paths are prerequisites, not generated certificates.

Use the [catalog execution guide](../../../../experiments/README.md#definitions-and-execution)
to inspect and preflight before authorizing authoring budgets.
Successful formal grading must retain the exact frozen submission hash and
public/hidden evidence. An unsandboxed diagnostic run is not a backend optimization handoff.
Changed inputs require a new qualified run; preserve old manifests and receipts.
