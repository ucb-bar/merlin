# Generated products

Products are grouped by **concern**, then by the axis that concern cares about
(target, model, workload or framework). The list of concerns is data, in
[storage.yaml](../../src/merlin/resources/contracts/storage.yaml); a directory it
does not declare is flagged by `merlin storage layout` and the artifact-layout check.

Write through `merlin.common.artifacts`, never a hand-built path:

- `new_product(topic, version=..., target=...)` creates
  `<concern>/<target>/v<ver>/<concern>_<target>_v<ver>_<TS>_<sha7>/` with a
  `manifest.yaml` and a relative `latest` link.
- `new_measurement(substrate, model, experiment, version=...)` creates
  `measurements/<substrate>/<model>/...`; the substrate names the execution
  environment (`k1_spacemit`, `firesim_<bitstream>`, `spike_<config>`, ...).
- `cache_dir(ns)` gives `cache/<ns>/`, purgeable by definition.
- `recaptures_dir()` gives the model captures (large, regenerable; the golden
  `*_ref.npz` references are the part worth keeping).

Places worth knowing:

| Look for | Concern |
| --- | --- |
| Compiler-generation results | `capsule-bench/`, `agentic-report/` |
| Measured performance | `perf-bench/`, `perf-studies/`, `measurements/` |
| Generated target packages and exported champions | `targets/<target>/` |
| Board hand-off bundles (`build_tools/deploy/make_delivery.py`) | `delivery/` |
| Frozen experiment protocols | `protocols/` |
| Paper and talk figures | `presentation/` |
| One-off investigations | `probes/` |
| Kept only because something cites it | `archive/` |

Exported champions under `targets/<target>/champions/` and sealed phase-0 releases
under `protocols/` are retention-pinned; never edit them by hand.
