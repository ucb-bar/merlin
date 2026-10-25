# Supplemental negative checker cases

Passed on 2026-10-23 UTC. Four explicit retained cases cover two unsafe-domain
contracts, a nonzero initializer and a valid mutated target. Every complete checker
JSON output matches its historical record; all four refuse the requested claim.
Structural correspondence can still hold in the two unsafe-domain cases: the
source numerical bound fails separately. This distinction is retained in cases.json.

The run reuses the 339 freshly rebuilt project modules from the sibling ordered
source reproduction, without repeating that build or using old worker project
objects. Inputs, commands and stdout/stderr are preserved. Actual imported objects,
original sources, all 105 model/shared-library records and pinned inputs are unchanged.
This is numerical checker reexecution on retained exports, not fresh pass/exporter
execution, new host receipt qualification, native fidelity, clean-environment or
combined qualification. It does not discharge the separate unresolved integrity
observation from the worker's NEW bound-wrapper run.

Session49676 is terminal/exit0. Do not sum these cases with historical test totals.
