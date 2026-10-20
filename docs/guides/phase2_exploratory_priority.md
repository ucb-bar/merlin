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
