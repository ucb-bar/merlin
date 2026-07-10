# CI workflows

| Workflow | Trigger | What it gates |
|---|---|---|
| [`pr-fast.yml`](pr-fast.yml) | push / PR | Fast gate: structure + artifact-layout + docs anti-drift + a fast test subset (blocking); ruff lint/format (advisory). |
| [`docs.yml`](docs.yml) | push (docs/schemas/pkg) / PR | Documentation anti-drift: stale generated docs, invalid front-matter, retired-path references. |

Heavier test buckets (`rvv/`, `gemmini/`, `runtime/`) need hardware or RTL sims (spike, verilator,
FireSim, boards) and run out-of-band, not in these hosted-runner workflows.
