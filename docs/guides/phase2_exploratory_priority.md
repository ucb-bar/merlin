---
title: Exploratory next-parent guidance
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-20
related: [phase2_public_diagnostic_launch, phase2_archive_screen]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase2/exploratory_development_priority.py]
---

# Exploratory next-parent guidance

After the original normal queue completes and its parent validates source,
numerical and count custody, an operator can pass its `result.data` feedback to
`exploratory_development_priority.decide(data, required_members=members,
minimum_improvement=settings.minimum_improvement)`. This pure helper launches
nothing and supplies no qualification. The default improvement threshold is
the existing one percent. It leaves the formal native decision, objective,
incumbent and plateau paths unchanged.

Every declared selected member needs both functional PASS arms, observed
compiled executables and positive finite estimates under the same model,
comparison domain and scope. The versioned
`native_exploration.exploratory_estimate` preserves model/report digests,
calibration residual metadata and domain-deviation warnings. Warnings remain
visible even when approximate estimates guide exploration. Missing or false
`event_accounting_complete` prevents latency-based parent advice; numeric
ratios remain descriptive where available.

The helper reports candidate/baseline ratios per member. It recommends a
candidate only when none of the selected estimates regress and at least one
improves by the selected threshold. The reverse rule recommends the baseline.
Mixed tradeoffs, incomplete coverage and smaller changes abstain. This
dominance rule introduces no aggregate weights and does not remove pending
members from the frozen workload. The result stays `EXPLORATORY`, with no
measured cycles, aggregate objective, promotion authority or plateau evidence.

## Selecting numeric feedback

Use `prepare_public_native_exploration` with all three explicit protocols:
`count_protocol=UNQUALIFIED_COUNT_PROTOCOL`,
`count_feedback_protocol=OBSERVED_COUNT_FEEDBACK_PROTOCOL`, and
`exploratory_estimate_protocol=EXPLORATORY_ESTIMATE_PROTOCOL`.
This selects owner V7 and member projection V3. Existing selections retain
their original schemas and behavior.

An out-of-domain formal prediction can remain `UNKNOWN` while its sibling
`exploratory_estimate.estimated_cycles` contains the original fixed-model
counter-window estimate. The normal parent reopens the original numerical
receipt, executable, count report, process records and model before publishing
that estimate. It rejects changed positive cycle values as well as missing
provenance. Calibration residuals describe the held calibration domain; no
out-of-domain confidence interval is invented. Incomplete event accounting
stays explicit even when a numeric estimate is available.

For partial resource feedback, `merlin.perf.capsule_roofline.partial_roofline`
exposes known useful-work compute floors, caller-proven padded work, logical
request ideals and separately observed client requests. It preserves missing
host, physical-memory, port and overlap terms. These numbers can identify
padding or reload opportunities without claiming complete attainment.
