# Generated products

Products are grouped by concern, then the axis that matters for that concern
(target, model, workload or framework). The complete category list and legacy-path
mappings live in [storage.yaml](../../merlin/contract/storage.yaml), not this index.

Common starting points:

| Look for | Category |
| --- | --- |
| Compiler-generation reports | `capsule-bench/`, `agentic-report/` |
| Optimization results and measured performance | `perf-bench/`, `measurements/` |
| Generated target packages | `targets/` |
| Paper/talk figures and supporting products | `presentation/` |
| Readiness and qualification evidence | `audits/` |
| One-off investigations | `probes/` |

Curated source inputs belong in `experiments/reference-data/`, not here. Buildable
trees and environments belong in `out/build/`; experiment execution belongs in
`out/runs/`. Do not treat every artifact as a disposable cache: manifests, citations
and retention pins can require keeping it. Use `merlin storage layout` to inspect
placement and the [storage guide](../../docs/guides/storage.md) before cleanup.
