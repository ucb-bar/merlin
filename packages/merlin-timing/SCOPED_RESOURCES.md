# Scoped model resource feedback

`spike-timing scoped-resources --report REPORT --output FEEDBACK` uses the same
`merlin_timing.scoped_resources.project` function called automatically by Phase 1.
Optional `--structural STRUCTURAL` supplies the existing structural compute bound.
It creates diagnostics, never qualification or a numerical verdict.

The trusted Spike observer selects the producer with model JSON:

```json
{"diagnostics":{"scoped_resources":{"enabled":true,"pairs":[{"left":6,"right":7}]}}}
```

IDs must come from that model's explicit resource roster. They are not portable
hardware roles. The diagnostic is default-off and requires the existing exact
counter-PC scope binding. Selecting it changes the model/source identity; the
operator must create a reviewed runtime selection. Old model/observer identities
must not be rewritten to claim that old evidence collected these metrics.

`merlin.scoped_resource_activity.v1` records model, scope, ELF and counter-binding
hashes; absolute opening/closing model cycles; capacity and resource name; busy
unit-cycles, active cycles, generic waiting-entry cycles and peaks; optional pair
intersection cycles. The producer sweeps demand intervals online, retains
pre-window activity that crosses the opening counter, and clips both boundaries.
It never subtracts whole-trace lifetime sums to approximate window activity.
At most 65,536 pending endpoints and 32 overlap pairs are retained. Unknown timing,
retroactive intervals, overflow or budget exhaustion makes capture incomplete.

The public projection requires complete same-window execution/counts and successful functional exit
and exact source identities. Capacity-normalized activity is busy unit-cycles /
(capacity × window cycles). Overlap is simultaneous activity, never an additive
latency component. Reservations can include conservative storage lifetimes;
they are **not physical engine utilization, PE occupancy, ROB occupancy or measured
stall cycles**. Generic modeled start queues are not accelerator reservation stations.

Useful/padded MAC ratio describes arithmetic work, not PE utilization. Structural
floor / estimate is a bound-to-model ratio, not attainable performance. Device,
CPU-data and fetch request rates remain separate from modeled DRAM/cache/coherence
traffic; no traffic across boundaries is summed or called physical bandwidth.
Host instruction counts are available, but host packing/preparation, dispatch and
synchronization cycle attribution remain UNKNOWN until a scoped attribution
producer exists. No guest instrumentation is added.

Automatic feedback keeps the full bounded resource report. Author briefs retain
at most 16 ranked resources, eight overlaps and 16 nonzero host counters with
omission counts. The terminal shows up to three model-booking percentages only
when its existing latency/source joins and the new scope/model/cycle joins agree.
Latency remains the objective; suggestions invite legal variants rather than
prescribing a target-specific schedule.

Native acceptance is retained under
`experiments/reference-data/scoped-resources-20261023`. The public probe's old,
new-disabled and new-enabled estimates all equal 354 cycles, with identical
functional output and feature counts. This is observer neutrality evidence,
not timing accuracy or calibration evidence. One wall-time sample was 17.91s
disabled and 21.82s enabled; this opt-in diagnostic adds host processing cost.

For automatic calibration ancestry, the selected performance document must carry
a `scoped_observer_diagnostics` implementation revision. Its operator review pins
the before/after runtime, models, binaries, source manifests, compiled sources,
public neutrality receipts and continued calibration denial. The verifier allows
only the feature-engine identity and explicit bounded diagnostic enablement;
provider, coefficients, hardware, memory model, functional command and scope
changes refuse. The new model identity must retain the old structural fitting
denial. A source identity change cannot turn an existing denied fit into an
allowed one. An accepted implementation revision still does not qualify timing
accuracy or authorize calibration promotion.

Select the reviewed performance document and its continued admission-denial pin
together. The calibration admission guard checks this pairing again at actual
panel/round execution; a prior model's denial environment cannot authorize a new
diagnostic model. Missing or stale review pins refuse before any native fit work.
