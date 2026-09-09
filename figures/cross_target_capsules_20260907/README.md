# Gemmini / Atlas / Radiance campaign comparison

Snapshot of three active Merlin-assisted capsule campaigns on 2026-09-07. The plots use the
repository house style from `merlin.plotting.merlin_plotstyle`; no palette or typography is copied
locally.

## Main readout

| target | latest pass | L3 pass | measured agent span | matched tool calls | rollout tokens |
|---|---:|---:|---:|---:|---|
| Gemmini | 93/97 (95.9%) | 89 | 8.26 h | 449 | 160.72M |
| Atlas | 49/61 (80.3%) | 49 | 12.00 h | 1,650 | 290.45M |
| Radiance | 28/29 (96.6%) | 0 | 8.00 h | 257 | 127.35M |

The latest pass rate is coverage, not target quality in isolation: the corpus sizes and mixes differ.
Atlas has the most expensive L2 adapter distribution in this snapshot, while Radiance has the
highest latest pass fraction on the smallest corpus. Gemmini covers the largest corpus and has the
most L3 passes. L3 status is intentionally reported separately from overall status: in the latest
Gemmini grade, 88 of 89 L3 passes are carried evidence and one has unspecified provenance; Atlas
freshly executed all 49 L3 passes. A capsule can pass L3 yet fail or remain incomplete overall.
The table counts matched tool calls; Figure 6 shows calls started, including 2 Gemmini, 10 Atlas, and
2 Radiance calls that had not terminated at the timing snapshot.

## Figures

- `fig1_campaign_overview`: normalized coverage, agent activity decomposition, and rollout-token
  composition (cached input, fresh input, and output).
- `fig2_capsule_taxonomy`: corpus composition and success by semantic family.
- `fig3_shape_runtime`: median L2 adapter time by inferred shape regime, plus declared problem size
  versus L2 wall time.
- `fig4_campaign_progress`: grade history from the first grade of each campaign.
- `fig5_matched_cohorts`: the closest defensible cross-target “ablation”: only coarse semantic-family ×
  shape-regime strata represented in all three corpora. It is still observational, not causal.
- `fig6_tool_token_profile`: tool mix, logged tool I/O, and client-observed response throughput.
- `fig7_tier_cost`: L2/L3 adapter-time distributions and fresh-versus-carried L3 evidence.

Each figure is exported as PDF, SVG, and PNG. `cross_target_snapshot.json`, `campaigns.csv`, and
`capsules.csv`, and `tools.csv` are the frozen source data.

## Important limits

1. Token totals use the consistent response-level rollout snapshots (`llm.tokens`) and are available
   for all three campaigns. The separate provider-event aggregate (`tokens`) is available for Atlas
   and Gemmini but missing for Radiance, so it is not mixed into the comparison. Throughput is output
   tokens divided by client-observed response turnaround—not server decode rate or TTFT.
2. L2 adapter wall time includes engine, oracle wait, host load, and adapter overhead. It is useful
   for campaign-cost planning, not a hardware-performance comparison.
3. There are zero exact capsule names shared by all three latest verdicts. Nine family × shape-regime
   strata overlap; Figure 5 compares those coarse cohorts. A causal ablation needs an intentionally
   shared capsule manifest and the same tier/engine protocol.
4. These were live campaigns. Re-running the extractor intentionally creates a newer snapshot and
   can change the values.

## Reproduce

From the repository root:

```bash
.venv/bin/python figures/cross_target_capsules_20260907/extract_cross_target_data.py
for script in figures/cross_target_capsules_20260907/gen_fig*.py; do
  PYTHONPATH=merlin/python .venv/bin/python "$script"
done
```
