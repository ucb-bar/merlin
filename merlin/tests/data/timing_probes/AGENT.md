# AGENT.md — merlin/tests/data/timing_probes

Subsets of a selected design's inputs for the timing-probe tests:

- `gemmini_facts.json`: the funct table, register layouts, elaborated features, geometry and
  datapaths of the GemminiGsimFireSimAlignedConfig RTL facts (`fixture.source_sha256` names the
  full artifact it was cut from).
- `gemmini_timing_profile.json`: the annotated-Spike profile's price and resource roster only; every
  price is null, as in the source profile.

Regenerate both from the selected artifacts rather than editing values by hand.
