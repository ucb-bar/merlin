# AGENT.md — src/merlin/transforms

This package owns target-independent, structural xDSL transformations and analyses.
Hardware policies, instruction selection and device resource assignments stay in OOT
repositories. Derive facts from the current IR; unknown effects and unsupported layouts
must remain conservative. A structural witness is not a formal proof receipt.

Keep compilation space proportional to IR size and rank, rather than tensor elements or
tile count. All covered changes must go through the host verification hook, except an
explicitly unverified planning invocation. Tests belong in `merlin/tests/ir/`.
