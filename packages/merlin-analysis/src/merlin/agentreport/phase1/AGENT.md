# AGENT.md — packages/merlin-analysis/src/merlin/agentreport/phase1

`runs` owns shared run/evidence readers and explicit `ReportInputs`.
`by_treatment` and `by_model` aggregate those same artifacts. These are post-hoc
readers, not grading authorities. Imports never select a target or read a native
experiment directory; callers provide target, run and report roots.

Preserve historical arm/condition attribution, cohort denominators, unavailable
evidence, abandoned certification, billing modes and lower-bound qualifications.
Do not reprice historical runs or substitute missing grades with zeros. The old
native commands are CLI adapters only; installed callers use these owners.
